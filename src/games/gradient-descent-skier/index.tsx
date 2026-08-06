"use client";

import { useEffect } from "react";
import { FastForward, RotateCcw, Sparkles, StepForward } from "lucide-react";
import { Button, Dial, Slider, type MetricSpec } from "@/components";
import { GameShell } from "@/engine/GameShell";
import { levelFromXp, useProgression } from "@/engine/progression";
import { getGameMeta } from "@/lib/catalog";
import {
  GLOBAL_MINIMUM,
  MATH_CODE,
  MATH_EQUATION,
  MATH_NOTES,
  MAX_LEARNING_RATE,
  MAX_MOMENTUM,
  MIN_LEARNING_RATE,
  STEP_BUDGET,
  WIN_SCORE,
  gradientNorm,
} from "./ml";
import { SLUG, useGradientSkierStore } from "./store";
import { VisualLane } from "./VisualLane";
import { CodeLane } from "./CodeLane";

const STAR_CRITERIA = [
  "Reach the deepest valley",
  `Score ${Math.round(WIN_SCORE * 100)}% or better`,
  "Reach it from the code lane",
];

function Controls() {
  const skier = useGradientSkierStore((s) => s.skier);
  const gradient = useGradientSkierStore((s) => s.gradient);
  const stepsRemaining = useGradientSkierStore((s) => s.stepsRemaining);
  const diverged = useGradientSkierStore((s) => s.diverged);
  const settled = useGradientSkierStore((s) => s.settled);
  const won = useGradientSkierStore((s) => s.won);

  const setLearningRate = useGradientSkierStore((s) => s.setLearningRate);
  const setMomentum = useGradientSkierStore((s) => s.setMomentum);
  const step = useGradientSkierStore((s) => s.step);
  const runToEnd = useGradientSkierStore((s) => s.runToEnd);
  const check = useGradientSkierStore((s) => s.check);
  const reset = useGradientSkierStore((s) => s.reset);

  const finished = diverged || settled || stepsRemaining <= 0;
  const slope = gradientNorm(gradient);

  return (
    <div className="flex flex-col gap-4">
      {/* THE dial. Log scale on purpose: on a linear track from 0.001 to 1.5,
          everything interesting would sit in the first 2% of travel. */}
      <div className="flex flex-col items-center gap-1">
        <Dial
          label="Learning rate"
          value={skier.learningRate}
          min={MIN_LEARNING_RATE}
          max={MAX_LEARNING_RATE}
          scale="log"
          onChange={setLearningRate}
          format={(value) => value.toPrecision(2)}
          hint="Multiplier on the slope. Arrow keys, or drag up and down."
        />
        <p className="text-center text-xs text-text-muted">
          slope {slope.toFixed(2)} × rate {skier.learningRate.toPrecision(2)} ≈{" "}
          <span className="font-mono">
            {(slope * skier.learningRate).toFixed(3)}
          </span>{" "}
          of travel next step
        </p>
      </div>

      <Slider
        label="Momentum"
        value={skier.momentum}
        min={0}
        max={MAX_MOMENTUM}
        step={0.01}
        onChange={setMomentum}
        format={(value) => value.toFixed(2)}
        hint="How much of the previous step carries over. Needed to cross a ridge."
      />

      <fieldset className="border-t border-border pt-4">
        <legend className="text-sm font-medium">Descend</legend>
        <p className="mt-1 mb-2 text-xs text-text-muted">
          {diverged
            ? "Diverged — reset to try another rate."
            : settled
              ? "Settled. Score it, or reset and try again."
              : `${stepsRemaining} of ${STEP_BUDGET} steps left.`}
        </p>
        <div className="flex flex-col gap-2">
          <Button
            variant="secondary"
            onClick={step}
            disabled={finished}
            icon={<StepForward className="size-4" />}
          >
            One step
          </Button>
          <Button
            variant="secondary"
            onClick={runToEnd}
            disabled={finished}
            icon={<FastForward className="size-4" />}
          >
            Run out the budget
          </Button>
          <Button
            variant="ghost"
            onClick={reset}
            icon={<RotateCcw className="size-4" />}
          >
            Back to the top
          </Button>
        </div>
      </fieldset>

      <div className="border-t border-border pt-4">
        <Button
          variant="primary"
          className="w-full"
          onClick={check}
          icon={<Sparkles className="size-4" />}
        >
          Score this run
        </Button>
        {won ? (
          <Button variant="secondary" className="mt-2 w-full" onClick={reset}>
            Try a different rate
          </Button>
        ) : null}
      </div>
    </div>
  );
}

export default function GradientDescentSkier() {
  const meta = getGameMeta(SLUG);

  const currentLoss = useGradientSkierStore((s) => s.currentLoss);
  const gradient = useGradientSkierStore((s) => s.gradient);
  const stepsTaken = useGradientSkierStore((s) => s.stepsTaken);
  const diverged = useGradientSkierStore((s) => s.diverged);
  const lastStepOvershot = useGradientSkierStore((s) => s.lastStepOvershot);
  const lastEvaluation = useGradientSkierStore((s) => s.lastEvaluation);
  const failure = useGradientSkierStore((s) => s.failure);
  const whyCard = useGradientSkierStore((s) => s.whyCard);
  const lane = useGradientSkierStore((s) => s.lane);
  const setLane = useGradientSkierStore((s) => s.setLane);
  const reset = useGradientSkierStore((s) => s.reset);
  const won = useGradientSkierStore((s) => s.won);

  const hydrate = useProgression((s) => s.hydrate);
  const hydrated = useProgression((s) => s.hydrated);
  const lastGain = useProgression((s) => s.lastGain);
  const games = useProgression((s) => s.games);
  const xp = useProgression((s) => s.xp);
  const level = levelFromXp(xp);

  useEffect(() => {
    if (!hydrated) void hydrate();
  }, [hydrated, hydrate]);

  /**
   * The live metric (pedagogy contract #3): loss at the skier's feet, from the
   * real surface. Goes red on an overshoot or divergence, which is the spec's
   * "spikes red on overshoot".
   */
  const metric: MetricSpec = {
    label: "Loss",
    value: currentLoss,
    format: "decimal",
    precision: 3,
    goodDirection: "down",
    state: diverged ? "bad" : lastStepOvershot ? "warn" : undefined,
    caption: diverged
      ? "diverged"
      : `deepest is ${GLOBAL_MINIMUM.loss.toFixed(3)}`,
  };

  const secondaryMetrics: MetricSpec[] = [
    {
      label: "Slope",
      value: gradientNorm(gradient),
      format: "decimal",
      precision: 3,
      goodDirection: "down",
      caption: "‖∇f‖ at your feet",
    },
    {
      label: "Steps",
      value: stepsTaken,
      format: "integer",
      goodDirection: "down",
      caption: `of ${STEP_BUDGET}`,
    },
    {
      label: "Score",
      value: lastEvaluation?.score ?? Number.NaN,
      format: "percent",
      goodDirection: "up",
      caption:
        lastEvaluation === null
          ? `need ${Math.round(WIN_SCORE * 100)}%`
          : `${Math.round(lastEvaluation.progress * 100)}% of the descent`,
      state: won ? "good" : undefined,
    },
  ];

  return (
    <GameShell
      slug={SLUG}
      title={meta?.title ?? "Gradient Descent Skier"}
      metric={metric}
      secondaryMetrics={secondaryMetrics}
      math={{
        equation: MATH_EQUATION,
        code: MATH_CODE,
        codeLanguage: "javascript",
        notes: MATH_NOTES,
      }}
      controls={<Controls />}
      visual={<VisualLane />}
      code={<CodeLane />}
      whyCard={whyCard}
      failure={failure}
      lane={lane}
      onLaneChange={setLane}
      progress={{
        level: level.level,
        xpIntoLevel: level.xpIntoLevel,
        xpForNextLevel: level.xpForNextLevel,
        stars: games[SLUG]?.stars ?? 0,
        recentGain: lastGain,
        starCriteria: STAR_CRITERIA,
      }}
      onRetry={reset}
      onNext={won ? reset : undefined}
      nextLabel="Another run"
    />
  );
}
