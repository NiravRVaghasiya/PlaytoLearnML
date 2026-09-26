"use client";

import { useEffect } from "react";
import { FastForward, Minus, Plus, Sparkles, StepForward } from "lucide-react";
import { Button, Slider, type MetricSpec } from "@/components";
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
  MAX_K,
  MIN_K,
  WIN_SCORE,
} from "./ml";
import { SLUG, useKMeansStore } from "./store";
import { VisualLane } from "./VisualLane";
import { CodeLane } from "./CodeLane";
import { ElbowMeter } from "./ElbowMeter";

/**
 * Exactly the engine's rule (`starsFor`): ★1 completed, ★2 best score at or
 * above `HIGH_SCORE_THRESHOLD`, ★3 that and a code-lane clear. The threshold is
 * imported rather than restated so the copy can't drift from the rule. A map is
 * only "settled" (won) at a score of 1/LOCAL_MINIMUM_RATIO ≈ 87% or more, so in
 * practice the first two stars arrive together.
 */
const STAR_CRITERIA = [
  "Settle a map",
  `Score ${Math.round(HIGH_SCORE_THRESHOLD * 100)}% or better`,
  "Settle a map from the code lane",
];

function Controls() {
  const centroids = useKMeansStore((s) => s.centroids);
  const selectedFlag = useKMeansStore((s) => s.selectedFlag);
  const elbowPoints = useKMeansStore((s) => s.elbowPoints);
  const assignmentStale = useKMeansStore((s) => s.assignmentStale);
  const converged = useKMeansStore((s) => s.converged);
  const iteration = useKMeansStore((s) => s.iteration);
  const won = useKMeansStore((s) => s.won);
  const round = useKMeansStore((s) => s.round);

  const addFlag = useKMeansStore((s) => s.addFlag);
  const removeFlag = useKMeansStore((s) => s.removeFlag);
  const moveFlag = useKMeansStore((s) => s.moveFlag);
  const assign = useKMeansStore((s) => s.assign);
  const update = useKMeansStore((s) => s.update);
  const step = useKMeansStore((s) => s.step);
  const settle = useKMeansStore((s) => s.settle);
  const check = useKMeansStore((s) => s.check);
  const newRound = useKMeansStore((s) => s.newRound);

  const flag =
    selectedFlag !== null ? centroids[selectedFlag] : undefined;

  return (
    <div className="flex flex-col gap-4">
      <fieldset>
        <legend className="mb-1 text-sm font-medium">
          Flags (k = {centroids.length})
        </legend>
        <p className="mb-2 text-xs text-text-muted">
          How many groups you think the villages form.
        </p>
        {/* The visible text is the whole accessible name on purpose. An
            aria-label of "Remove a flag" over visible text "Fewer" is a WCAG
            2.5.3 failure — voice-control users say what they see. */}
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => removeFlag(centroids.length - 1)}
            disabled={centroids.length <= MIN_K}
            icon={<Minus className="size-4" />}
          >
            Remove flag
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => addFlag()}
            disabled={centroids.length >= MAX_K}
            icon={<Plus className="size-4" />}
          >
            Add flag
          </Button>
        </div>
      </fieldset>

      <div>
        <h3 className="mb-1 text-sm font-medium">Elbow chart</h3>
        <p className="mb-2 text-xs text-text-muted">
          Best possible inertia at each k. Find where the curve stops dropping
          steeply — that is how many groups the data has.
        </p>
        <ElbowMeter
          curve={elbowPoints}
          currentK={centroids.length}
        />
      </div>

      {/* The guaranteed-accessible path to a 2-D position: two real sliders. */}
      {flag && selectedFlag !== null ? (
        <fieldset className="border-t border-border pt-4">
          <legend className="text-sm font-medium">
            Flag {selectedFlag + 1} position
          </legend>
          <Slider
            className="mt-2"
            label="Across"
            value={flag.x}
            min={0}
            max={1}
            step={0.01}
            onChange={(x) => moveFlag(selectedFlag, x, flag.y)}
            format={(v) => (v * 100).toFixed(0)}
          />
          <Slider
            className="mt-3"
            label="Up"
            value={flag.y}
            min={0}
            max={1}
            step={0.01}
            onChange={(y) => moveFlag(selectedFlag, flag.x, y)}
            format={(v) => (v * 100).toFixed(0)}
          />
        </fieldset>
      ) : null}

      <fieldset className="border-t border-border pt-4">
        <legend className="text-sm font-medium">The loop</legend>
        <p className="mt-1 mb-2 text-xs text-text-muted">
          {assignmentStale
            ? "Flags have moved — assign again."
            : converged
              ? `Converged after ${iteration} iterations.`
              : "Alternate these two until nothing moves."}
        </p>
        <div className="flex flex-col gap-2">
          <div className="flex gap-2">
            <Button
              className="flex-1"
              variant={assignmentStale ? "primary" : "secondary"}
              size="sm"
              onClick={assign}
            >
              1. Assign
            </Button>
            <Button
              className="flex-1"
              variant="secondary"
              size="sm"
              onClick={update}
            >
              2. Update
            </Button>
          </div>
          <Button
            variant="secondary"
            size="sm"
            onClick={step}
            icon={<StepForward className="size-4" />}
          >
            Both, once
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={settle}
            icon={<FastForward className="size-4" />}
          >
            Run to convergence
          </Button>
        </div>
      </fieldset>

      <div className="flex flex-col gap-2 border-t border-border pt-4">
        <Button
          variant="primary"
          // Wrapped: `check(source)` would otherwise receive the click event.
          onClick={() => check()}
          icon={<Sparkles className="size-4" />}
        >
          Score this map
        </Button>
        {won ? (
          <Button variant="secondary" onClick={newRound}>
            New map ({round} settled)
          </Button>
        ) : null}
      </div>
    </div>
  );
}

export default function KMeansTerritoryWars() {
  const meta = getGameMeta(SLUG);

  const inertia = useKMeansStore((s) => s.inertia);
  const k = useKMeansStore((s) => s.k);
  const iteration = useKMeansStore((s) => s.iteration);
  const converged = useKMeansStore((s) => s.converged);
  const lastEvaluation = useKMeansStore((s) => s.lastEvaluation);
  const failure = useKMeansStore((s) => s.failure);
  const whyCard = useKMeansStore((s) => s.whyCard);
  const lane = useKMeansStore((s) => s.lane);
  const setLane = useKMeansStore((s) => s.setLane);
  const reset = useKMeansStore((s) => s.reset);
  const newRound = useKMeansStore((s) => s.newRound);
  const won = useKMeansStore((s) => s.won);

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
   * The live metric (pedagogy contract #3): inertia, recomputed from the real
   * assignment every time a flag moves or a step runs.
   *
   * Note the trap this metric embodies: it falls monotonically as k rises, so
   * "smaller is better" is only true at a fixed k. The elbow chart in the rail
   * and the "Bad k" failure are what stop that from being a lie.
   */
  const metric: MetricSpec = {
    label: "Inertia",
    value: inertia,
    format: "decimal",
    precision: 2,
    goodDirection: "down",
    caption: converged ? "converged" : "sum of squared distances",
  };

  /**
   * k and the iteration count have no good direction: adding the right third
   * flag is progress, and so is stepping the loop. With a direction set, the
   * readout pulsed red on both — teaching "fewer is better" for exactly the two
   * numbers where it isn't. `state: "neutral"` keeps the ▲/▼ glyph and pins
   * the colour neutral, and MetricReadout suppresses its improvement sparkle
   * for an explicit neutral state too.
   */
  const secondaryMetrics: MetricSpec[] = [
    {
      label: "Flags (k)",
      value: k,
      format: "integer",
      state: "neutral",
      caption: "your choice",
    },
    {
      label: "Iterations",
      value: iteration,
      format: "integer",
      state: "neutral",
      caption: converged ? "settled" : "still moving",
    },
    {
      label: "Score",
      // A verdict on stale territories judged nothing, so it shows no number.
      value:
        lastEvaluation && !lastEvaluation.assignmentStale
          ? lastEvaluation.score
          : Number.NaN,
      format: "percent",
      goodDirection: "up",
      caption:
        lastEvaluation === null
          ? "score the map to see"
          : lastEvaluation.assignmentStale
            ? "assign first"
            : `need ${Math.round(WIN_SCORE * 100)}%`,
      state: won ? "good" : undefined,
    },
  ];

  return (
    <GameShell
      slug={SLUG}
      title={meta?.title ?? "K-Means Territory Wars"}
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
      onNext={won ? newRound : undefined}
      nextLabel="Next map"
    />
  );
}
