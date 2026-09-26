"use client";

import { useEffect, useMemo } from "react";
import { ChefHat, ChevronRight, RotateCcw } from "lucide-react";
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
  METRIC_LABELS,
  SCENARIOS,
  THRESHOLD_STEP,
  confusionAt,
  majorityBaseline,
  metricValue,
  metricsOf,
} from "./ml";
import { SLUG, currentScenario, useChefStore } from "./store";
import { CriticScenario } from "./CriticScenario";
import { VisualLane } from "./VisualLane";
import { CodeLane } from "./CodeLane";

/**
 * The engine's rule, stated per game: ★1 finish, ★2 a best score of at least
 * HIGH_SCORE_THRESHOLD, ★3 that plus a code-lane clear. A shift's score is 60%
 * for meeting the brief plus up to 40% for how far inside the band the headline
 * metric was pushed (see BRIEF_MET_SHARE), so the second star is earned by
 * where in the band you serve, not by finishing.
 */
const STAR_CRITERIA = [
  "Sign off all four shifts",
  `Score ${Math.round(
    HIGH_SCORE_THRESHOLD * 100,
  )}% or better across the week by pushing each shift's headline metric as far as its brief allows`,
  "Clear a shift with api.serve() from the code lane",
];

function Controls() {
  const scenario = useChefStore(currentScenario);
  const samples = useChefStore((s) => s.samples);
  const threshold = useChefStore((s) => s.threshold);
  const phase = useChefStore((s) => s.phase);
  const attempts = useChefStore((s) => s.attempts);
  const setThreshold = useChefStore((s) => s.setThreshold);
  const serve = useChefStore((s) => s.serve);
  const nextShift = useChefStore((s) => s.nextShift);
  const restart = useChefStore((s) => s.restart);
  const scenarioIndex = useChefStore((s) => s.scenarioIndex);

  // Derived, never stored: four counts kept alongside the slider could disagree
  // with it, in a game about the matrix recomputing exactly.
  const matrix = useMemo(
    () => confusionAt(samples, threshold),
    [samples, threshold],
  );
  const metrics = useMemo(() => metricsOf(matrix), [matrix]);
  const constraints = useMemo(
    () =>
      scenario.constraints.map((constraint) => {
        const achieved = metricValue(metrics, constraint.metric);
        return { ...constraint, achieved, met: achieved >= constraint.floor };
      }),
    [scenario, metrics],
  );

  const flagged = matrix.truePositives + matrix.falsePositives;

  return (
    <div className="flex flex-col gap-4">
      <CriticScenario
        scenario={scenario}
        constraints={constraints}
        attempts={attempts}
      />

      {/* THE control (spec: <ThresholdSlider>). */}
      <div className="border-t border-border pt-4">
        <Slider
          label="Decision threshold"
          value={threshold}
          min={0}
          max={1}
          step={THRESHOLD_STEP}
          onChange={setThreshold}
          format={(value) => value.toFixed(2)}
          hint="Flag every case scoring at or above this. Nothing retrains — this is purely where you cut."
        />
        <p className="mt-1 text-xs text-text-muted">
          Flagging{" "}
          <span className="font-mono">{flagged}</span> of{" "}
          <span className="font-mono">{samples.length}</span> cases ·{" "}
          {matrix.truePositives} caught, {matrix.falseNegatives} missed,{" "}
          {matrix.falsePositives} false alarm
          {matrix.falsePositives === 1 ? "" : "s"}
        </p>
      </div>

      <div className="border-t border-border pt-4">
        {phase === "cleared" ? (
          <Button
            variant="primary"
            className="w-full"
            onClick={nextShift}
            icon={<ChevronRight className="size-4" />}
          >
            Next shift
          </Button>
        ) : phase === "complete" ? (
          <Button
            variant="primary"
            className="w-full"
            onClick={restart}
            icon={<RotateCcw className="size-4" />}
          >
            Start the week again
          </Button>
        ) : (
          <Button
            variant="primary"
            className="w-full"
            // Arrow, not `onClick={serve}`: serve's argument is the lane it
            // came from, and a click would pass the event in its place.
            onClick={() => serve("visual")}
            icon={<ChefHat className="size-4" />}
          >
            Serve this cutoff
          </Button>
        )}

        <Button
          variant="ghost"
          className="mt-2 w-full"
          onClick={restart}
          // Only pointless when already at the untouched start of shift one —
          // on a fresh shift three it is still a real way back.
          disabled={scenarioIndex === 1 && phase === "tuning" && attempts === 0}
        >
          Back to shift one
        </Button>
      </div>
    </div>
  );
}

export default function ConfusionMatrixChef() {
  const meta = getGameMeta(SLUG);

  const scenario = useChefStore(currentScenario);
  const samples = useChefStore((s) => s.samples);
  const threshold = useChefStore((s) => s.threshold);
  const phase = useChefStore((s) => s.phase);
  const clearedScores = useChefStore((s) => s.clearedScores);
  const failure = useChefStore((s) => s.failure);
  const whyCard = useChefStore((s) => s.whyCard);
  const lane = useChefStore((s) => s.lane);
  const setLane = useChefStore((s) => s.setLane);
  const nextShift = useChefStore((s) => s.nextShift);
  const retryShift = useChefStore((s) => s.retryShift);
  const restart = useChefStore((s) => s.restart);
  const games = useProgression((s) => s.games);

  const hydrate = useProgression((s) => s.hydrate);
  const hydrated = useProgression((s) => s.hydrated);
  const lastGain = useProgression((s) => s.lastGain);
  const xp = useProgression((s) => s.xp);
  const level = levelFromXp(xp);

  useEffect(() => {
    if (!hydrated) void hydrate();
  }, [hydrated, hydrate]);

  const metrics = useMemo(
    () => metricsOf(confusionAt(samples, threshold)),
    [samples, threshold],
  );

  const baseline = majorityBaseline(scenario.prevalence);
  const primaryValue = metricValue(metrics, scenario.primary);
  const primaryFloor =
    scenario.constraints.find(
      (constraint) => constraint.metric === scenario.primary,
    )?.floor ?? null;

  /**
   * The live metric (pedagogy contract #3).
   *
   * Its LABEL changes per shift, which is the point rather than a convenience:
   * "the right metric depends on the cost of each error type" is hard to argue in
   * prose and self-evident when the headline number is called Precision on the
   * prank shift and Recall on the allergen shift. The catalog names F1 as this
   * game's metric, so F1 stays permanently visible below — but it is not always
   * the number that decides the round, and pretending otherwise would teach the
   * opposite of the lesson.
   */
  const metric: MetricSpec = {
    label: METRIC_LABELS[scenario.primary],
    value: primaryValue,
    format: "percent",
    precision: 1,
    goodDirection: "up",
    state:
      failure !== null
        ? "bad"
        : phase === "complete" || phase === "cleared"
          ? "good"
          : primaryFloor !== null && primaryValue >= primaryFloor
            ? "good"
            : undefined,
    caption:
      primaryFloor === null
        ? "this shift's headline number"
        : `this shift needs ${Math.round(primaryFloor * 100)}%`,
  };

  const secondaryMetrics: MetricSpec[] = [
    {
      // Kept visible on every shift precisely so the player can watch it stay
      // high while the metric that matters collapses.
      label: "Accuracy",
      value: metrics.accuracy,
      format: "percent",
      precision: 1,
      goodDirection: "up",
      caption: `${Math.round(baseline * 100)}% for ignoring the model`,
      state:
        baseline >= 0.8 && metrics.accuracy >= baseline - 0.01 && metrics.recall < 0.5
          ? "warn"
          : undefined,
    },
    {
      label: "F1",
      value: metrics.f1,
      format: "percent",
      precision: 1,
      goodDirection: "up",
      caption: `precision ${Math.round(
        metrics.precision * 100,
      )}% · recall ${Math.round(metrics.recall * 100)}%`,
    },
    {
      label: "Shifts signed off",
      value: clearedScores.length,
      format: "integer",
      goodDirection: "up",
      caption: `of ${SCENARIOS.length}`,
      state: phase === "complete" ? "good" : undefined,
    },
  ];

  return (
    <GameShell
      slug={SLUG}
      title={meta?.title ?? "Confusion Matrix Chef"}
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
      // Retry means this shift again. Wiping every shift already signed off is
      // "Back to shift one", which says so.
      onRetry={retryShift}
      onNext={
        phase === "cleared"
          ? nextShift
          : phase === "complete"
            ? restart
            : undefined
      }
      nextLabel={
        phase === "cleared"
          ? `Shift ${scenario.index + 1}`
          : phase === "complete"
            ? "Run the week again"
            : undefined
      }
    />
  );
}
