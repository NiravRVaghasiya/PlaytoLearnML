"use client";

import { useEffect } from "react";
import { Anchor, Eye, RotateCcw, SkipForward } from "lucide-react";
import { Button, Slider, type MetricSpec } from "@/components";
import { GameShell } from "@/engine/GameShell";
import {
  HIGH_SCORE_THRESHOLD,
  levelFromXp,
  useProgression,
} from "@/engine/progression";
import { getGameMeta } from "@/lib/catalog";
import {
  CLOUDS,
  MATH_CODE,
  MATH_EQUATION,
  MATH_NOTES,
  VARIANCE_TARGET,
} from "./ml";
import { SLUG, cloudSpec, useDiverStore } from "./store";
import { PCAHintButton } from "./PCAHintButton";
import { VarianceGauge } from "./VarianceGauge";
import { VisualLane } from "./VisualLane";
import { CodeLane } from "./CodeLane";

/**
 * Exactly the progression engine's rule: a surfaced cloud, a best score at or
 * above `HIGH_SCORE_THRESHOLD`, and that plus a code-lane clear. The third line
 * used to be "separate the groups as well as any plane can" — the per-dive grade
 * in ml.ts, which the engine never reads — so the star it promised never came.
 * The first line names both halves of surfacing: keeping the variance alone
 * (the needle at its PCA plane) is "mixed", and records nothing.
 */
const STAR_CRITERIA = [
  `Surface a cloud: keep ${Math.round(
    VARIANCE_TARGET * 100,
  )}% of the best possible variance, and show the groups wherever a flat shadow can`,
  `Score ${Math.round(HIGH_SCORE_THRESHOLD * 100)}% or better by landing on the optimum without the hint`,
  "Surface a cloud from the code lane",
];

function Controls() {
  const angles = useDiverStore((s) => s.angles);
  const analysis = useDiverStore((s) => s.analysis);
  const evaluation = useDiverStore((s) => s.evaluation);
  const submitted = useDiverStore((s) => s.submitted);
  const hintUsed = useDiverStore((s) => s.hintUsed);
  const showGroups = useDiverStore((s) => s.showGroups);
  const phase = useDiverStore((s) => s.phase);
  const surfacedIds = useDiverStore((s) => s.surfacedIds);

  const setAngle = useDiverStore((s) => s.setAngle);
  const submit = useDiverStore((s) => s.submit);
  const usePcaHint = useDiverStore((s) => s.usePcaHint);
  const toggleGroups = useDiverStore((s) => s.toggleGroups);
  const nextCloud = useDiverStore((s) => s.nextCloud);
  const reset = useDiverStore((s) => s.reset);

  return (
    <div className="flex flex-col gap-4">
      <section aria-labelledby="rotation-heading" className="flex flex-col gap-2">
        <h3 id="rotation-heading" className="text-sm font-semibold">
          Turn the cloud
        </h3>
        <Slider
          label="Yaw"
          value={angles.yaw}
          min={-180}
          max={180}
          step={1}
          format={(value) => `${value.toFixed(0)}°`}
          hint="Swings the discarded axis around the vertical."
          onChange={(value) => setAngle("yaw", value)}
        />
        <Slider
          label="Pitch"
          value={angles.pitch}
          min={-90}
          max={90}
          step={1}
          format={(value) => `${value.toFixed(0)}°`}
          hint="Tips it up and down. Yaw and pitch together are the only things that move the gauge."
          onChange={(value) => setAngle("pitch", value)}
        />
        <Slider
          label="Roll"
          value={angles.roll}
          min={-180}
          max={180}
          step={1}
          format={(value) => `${value.toFixed(0)}°`}
          hint="Spins the shadow in its own plane. Cannot change what the projection retains — try it."
          onChange={(value) => setAngle("roll", value)}
        />
      </section>

      <div className="border-t border-border pt-4">
        {phase === "surfaced" ? (
          <Button
            variant="primary"
            className="w-full"
            onClick={nextCloud}
            icon={<SkipForward className="size-4" />}
          >
            Next cloud
          </Button>
        ) : (
          <Button
            variant="primary"
            className="w-full"
            // Wrapped: handing `submit` straight to onClick would pass the click
            // event in as the commit's `source`.
            onClick={() => submit()}
            disabled={submitted}
            icon={<Anchor className="size-4" />}
          >
            {submitted ? "Committed" : "Commit this projection"}
          </Button>
        )}

        <div className="mt-3">
          <PCAHintButton
            analysis={analysis}
            used={hintUsed}
            disabled={false}
            onUse={usePcaHint}
          />
        </div>

        <Button
          variant="ghost"
          className="mt-2 w-full"
          aria-pressed={showGroups}
          onClick={toggleGroups}
          icon={<Eye className="size-4" />}
        >
          {showGroups ? "Hide the groups" : "Reveal the groups early"}
        </Button>

        <Button
          variant="ghost"
          className="mt-2 w-full"
          onClick={reset}
          icon={<RotateCcw className="size-4" />}
        >
          Start this cloud again
        </Button>

        <p className="mt-2 text-xs text-text-muted">
          {surfacedIds.length} of {CLOUDS.length} clouds surfaced.
          {evaluation.separable
            ? ""
            : " This one cannot be separated by any flat shadow — see the card below."}
        </p>
      </div>

      <div className="border-t border-border pt-4">
        <VarianceGauge
          retained={evaluation.retained}
          analysis={analysis}
          separation={evaluation.separation}
          separationShare={evaluation.separationShare}
          separable={evaluation.separable}
          submitted={submitted}
        />
      </div>
    </div>
  );
}

export default function DimensionDiver() {
  const meta = getGameMeta(SLUG);

  const evaluation = useDiverStore((s) => s.evaluation);
  const analysis = useDiverStore((s) => s.analysis);
  const angles = useDiverStore((s) => s.angles);
  const submitted = useDiverStore((s) => s.submitted);
  const phase = useDiverStore((s) => s.phase);
  const failure = useDiverStore((s) => s.failure);
  const whyCard = useDiverStore((s) => s.whyCard);
  const lane = useDiverStore((s) => s.lane);
  const surfacedIds = useDiverStore((s) => s.surfacedIds);
  const setLane = useDiverStore((s) => s.setLane);
  const reset = useDiverStore((s) => s.reset);
  const nextCloud = useDiverStore((s) => s.nextCloud);
  const cloud = useDiverStore(cloudSpec);
  const games = useProgression((s) => s.games);

  const hydrate = useProgression((s) => s.hydrate);
  const hydrated = useProgression((s) => s.hydrated);
  const lastGain = useProgression((s) => s.lastGain);
  const xp = useProgression((s) => s.xp);
  const level = levelFromXp(xp);

  useEffect(() => {
    if (!hydrated) void hydrate();
  }, [hydrated, hydrate]);

  /**
   * The live metric (pedagogy contract #3): "Variance retained", per the catalog.
   *
   * Reported as an absolute fraction rather than as a share of the optimum, because
   * the absolute number is the one that means something — it is literally how much
   * of the cloud survives being flattened. The ceiling is a caption and a tick mark
   * on the gauge, which keeps the headline honest without hiding what is achievable.
   */
  const metric: MetricSpec = {
    label: "Variance retained",
    value: evaluation.retained,
    format: "percent",
    precision: 1,
    goodDirection: "up",
    state:
      phase === "surfaced" ? "good" : failure !== null ? "bad" : undefined,
    caption: `ceiling ${(analysis.best * 100).toFixed(1)}% · ${(
      evaluation.share * 100
    ).toFixed(1)}% of it`,
  };

  const secondaryMetrics: MetricSpec[] = [
    {
      // Beside the headline on purpose: these two numbers are the game's argument.
      label: "Group separation",
      value: evaluation.separation.ratio,
      format: "decimal",
      precision: 2,
      goodDirection: "up",
      caption: evaluation.separable
        ? `best reachable ${analysis.reachableSeparation.ratio.toFixed(2)}`
        : "no flat shadow separates these",
      state:
        evaluation.separable && submitted && evaluation.separationShare < 0.7
          ? "bad"
          : undefined,
    },
    {
      label: "Discarded axis",
      value: 1 - evaluation.retained,
      format: "percent",
      precision: 1,
      goodDirection: "down",
      caption: `the direction pointing at you, at yaw ${angles.yaw.toFixed(
        0,
      )}° pitch ${angles.pitch.toFixed(0)}°`,
    },
    {
      label: "Clouds surfaced",
      value: surfacedIds.length / CLOUDS.length,
      format: "percent",
      goodDirection: "up",
      caption: `${surfacedIds.length} of ${CLOUDS.length} · now on "${cloud.title}"`,
    },
  ];

  return (
    <GameShell
      slug={SLUG}
      title={meta?.title ?? "Dimension Diver"}
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
      onNext={phase === "surfaced" ? nextCloud : undefined}
      nextLabel={
        phase === "surfaced"
          ? surfacedIds.length >= CLOUDS.length
            ? "Dive again"
            : `Next: ${
                CLOUDS[(CLOUDS.findIndex((c) => c.id === cloud.id) + 1) % CLOUDS.length]!
                  .title
              }`
          : undefined
      }
    />
  );
}
