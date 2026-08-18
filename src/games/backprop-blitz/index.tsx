"use client";

import { useEffect, useMemo } from "react";
import { Eye, RotateCcw, SkipForward } from "lucide-react";
import { Button, type MetricSpec } from "@/components";
import { GameShell } from "@/engine/GameShell";
import {
  HIGH_SCORE_THRESHOLD,
  levelFromXp,
  useProgression,
} from "@/engine/progression";
import { getGameMeta } from "@/lib/catalog";
import {
  MATH_CODE,
  MATH_EQUATION,
  MATH_NOTES,
  PARAM_IDS,
  SCENARIOS,
  STEP_COUNT,
  TRAINING_STEPS,
  nodeById,
} from "./ml";
import { SLUG, scenario, useBlitzStore } from "./store";
import { CorrectnessMeter } from "./CorrectnessMeter";
import { VisualLane } from "./VisualLane";
import { CodeLane } from "./CodeLane";

const STAR_CRITERIA = [
  "Match every node gradient in a scenario",
  `Score ${Math.round(
    HIGH_SCORE_THRESHOLD * 100,
  )}% or better by routing a scenario with no wrong turns`,
  "Clear all three scenarios, including the one with the shut gate",
];

function Controls() {
  const evaluation = useBlitzStore((s) => s.evaluation);
  const showTruth = useBlitzStore((s) => s.showTruth);
  const phase = useBlitzStore((s) => s.phase);
  const clearedIds = useBlitzStore((s) => s.clearedIds);
  const routing = useBlitzStore((s) => s.routing);
  const toggleTruth = useBlitzStore((s) => s.toggleTruth);
  const nextScenario = useBlitzStore((s) => s.nextScenario);
  const reset = useBlitzStore((s) => s.reset);

  const answered = Object.keys(routing).length;
  const { correctness } = evaluation;

  return (
    <div className="flex flex-col gap-4">
      <CorrectnessMeter
        fraction={correctness.fraction}
        matched={correctness.matched}
        total={correctness.total}
        answered={answered}
        coincidences={correctness.coincidences.length}
        verdicts={correctness.verdicts}
        showTruth={showTruth}
      />

      <div className="border-t border-border pt-4">
        {phase === "cleared" ? (
          <Button
            variant="primary"
            className="w-full"
            onClick={nextScenario}
            icon={<SkipForward className="size-4" />}
          >
            Next scenario
          </Button>
        ) : null}

        <Button
          variant={showTruth ? "primary" : "secondary"}
          className="mt-2 w-full"
          aria-pressed={showTruth}
          onClick={toggleTruth}
          icon={<Eye className="size-4" />}
        >
          {showTruth ? "Hide the real trace" : "Show the real trace"}
        </Button>

        <Button
          variant="ghost"
          className="mt-2 w-full"
          onClick={reset}
          icon={<RotateCcw className="size-4" />}
        >
          Start this scenario again
        </Button>

        <p className="mt-2 text-xs text-text-muted">
          {clearedIds.length} of {SCENARIOS.length} scenarios cleared. Wrong
          gradients keep flowing on purpose — nothing here quietly corrects your
          arithmetic.
        </p>
      </div>
    </div>
  );
}

export default function BackpropBlitz() {
  const meta = getGameMeta(SLUG);

  const evaluation = useBlitzStore((s) => s.evaluation);
  const phase = useBlitzStore((s) => s.phase);
  const failure = useBlitzStore((s) => s.failure);
  const whyCard = useBlitzStore((s) => s.whyCard);
  const lane = useBlitzStore((s) => s.lane);
  const routing = useBlitzStore((s) => s.routing);
  const clearedIds = useBlitzStore((s) => s.clearedIds);
  const setLane = useBlitzStore((s) => s.setLane);
  const reset = useBlitzStore((s) => s.reset);
  const nextScenario = useBlitzStore((s) => s.nextScenario);
  const current = useBlitzStore(scenario);
  const games = useProgression((s) => s.games);

  const hydrate = useProgression((s) => s.hydrate);
  const hydrated = useProgression((s) => s.hydrated);
  const lastGain = useProgression((s) => s.lastGain);
  const xp = useProgression((s) => s.xp);
  const level = levelFromXp(xp);

  useEffect(() => {
    if (!hydrated) void hydrate();
  }, [hydrated, hydrate]);

  const answered = Object.keys(routing).length;
  const { correctness } = evaluation;

  const paramGrads = useMemo(
    () =>
      PARAM_IDS.map((id) => {
        const verdict = correctness.verdicts.find(
          (candidate) => candidate.id === id,
        );
        return { id, label: nodeById(id).label, verdict };
      }),
    [correctness.verdicts],
  );

  const settledParams = paramGrads.filter(
    (entry) => entry.verdict?.mine !== null && entry.verdict !== undefined,
  ).length;

  /**
   * The live metric (pedagogy contract #3): "Gradient correctness", per the
   * catalog — the fraction of node gradients matching a real autograd trace.
   *
   * Node gradients rather than rules-chosen, and that is the sharper of the two.
   * Rules-chosen would be a quiz score. This is a measurement against a reference
   * implementation, which means it also captures the thing the game most wants the
   * player to feel: one wrong rule near the loss drops this number a long way,
   * because everything behind it inherited the mistake.
   */
  const metric: MetricSpec = {
    label: "Gradient correctness",
    value: correctness.fraction,
    format: "percent",
    goodDirection: "up",
    state:
      phase === "cleared" ? "good" : failure !== null ? "bad" : undefined,
    caption:
      answered === 0
        ? `${STEP_COUNT} nodes to route`
        : `${correctness.matched} of ${correctness.total} nodes match autograd · ${answered}/${STEP_COUNT} routed`,
  };

  const secondaryMetrics: MetricSpec[] = [
    {
      label: "Rules right",
      value: answered === 0 ? Number.NaN : correctness.rulesRight / answered,
      format: "percent",
      goodDirection: "up",
      caption:
        correctness.coincidences.length > 0
          ? `${correctness.coincidences.length} wrong rule got the right answer`
          : `${correctness.rulesRight} of ${answered} choices`,
      state:
        correctness.coincidences.length > 0 ? "warn" : undefined,
    },
    {
      // The angle is the honest damage report: a wrong backward pass is wrong in
      // DIRECTION, and a single step's loss will not tell you that.
      label: "Gradient angle",
      value: evaluation.nextStep === null ? evaluation.angle : Number.NaN,
      format: "decimal",
      precision: 1,
      goodDirection: "down",
      caption:
        evaluation.nextStep !== null
          ? "finish the walk to compare"
          : evaluation.angle > 90
            ? "degrees off — this points uphill"
            : "degrees from the true gradient",
      state:
        evaluation.nextStep === null && evaluation.angle > 90 ? "bad" : undefined,
    },
    {
      label: "Loss after training",
      value: evaluation.mineRun?.diverged
        ? Number.POSITIVE_INFINITY
        : (evaluation.mineRun?.final ?? Number.NaN),
      format: "decimal",
      precision: 4,
      goodDirection: "down",
      caption:
        evaluation.mineRun === null
          ? `${TRAINING_STEPS} steps, once you finish`
          : evaluation.mineRun.diverged
            ? "diverged — the gradient pointed uphill"
            : `real gradients reach ${evaluation.truthRun?.final.toFixed(4) ?? "?"}`,
      state: evaluation.mineRun?.diverged ? "bad" : undefined,
    },
    {
      label: "Parameters settled",
      value: settledParams / PARAM_IDS.length,
      format: "percent",
      goodDirection: "up",
      caption: `${settledParams} of ${PARAM_IDS.length} weights have a gradient`,
    },
  ];

  return (
    <GameShell
      slug={SLUG}
      title={meta?.title ?? "Backprop Blitz"}
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
      onNext={phase === "cleared" ? nextScenario : undefined}
      nextLabel={
        phase === "cleared"
          ? clearedIds.length >= SCENARIOS.length
            ? "Route it again"
            : `Next: ${SCENARIOS[(SCENARIOS.findIndex((s) => s.id === current.id) + 1) % SCENARIOS.length]!.title}`
          : undefined
      }
    />
  );
}
