"use client";

import { useEffect } from "react";
import { RotateCcw, Sparkles } from "lucide-react";
import { Button, type MetricSpec } from "@/components";
import { GameShell } from "@/engine/GameShell";
import {
  HIGH_SCORE_THRESHOLD,
  levelFromXp,
  useProgression,
} from "@/engine/progression";
import { getGameMeta } from "@/lib/catalog";
import {
  CHANCE_RATE,
  FILTER_BUDGET,
  MATH_CODE,
  MATH_EQUATION,
  MATH_NOTES,
  TARGET_ACCURACY,
  THRIFTY_FILTERS,
  filtersUsed,
  outputChannels,
  stackGeometry,
} from "./ml";
import { SLUG, currentLearned, useKitchenStore } from "./store";
import { LayerStack } from "./LayerStack";
import { LearnedKernels } from "./LearnedKernels";
import { VisualLane } from "./VisualLane";
import { CodeLane } from "./CodeLane";

/**
 * The engine's rule, in this game's terms: ★1 served, ★2 a best score of at least
 * HIGH_SCORE_THRESHOLD, ★3 that plus a code-lane serve. `servedPoints` is built so
 * the threshold falls exactly at THRIFTY_FILTERS with no duplicates, and the tests
 * hold the sentence below to that arithmetic. The third star used to promise a
 * filter-count rubric the game computed and nobody read.
 */
const STAR_CRITERIA = [
  `Read all four dishes at ${Math.round(TARGET_ACCURACY * 100)}% or better`,
  `Score ${Math.round(
    HIGH_SCORE_THRESHOLD * 100,
  )}% or better: serve with ${THRIFTY_FILTERS} filters or fewer, none duplicated`,
  "Serve a stack you built from the code lane",
];

function Controls() {
  const layers = useKitchenStore((s) => s.layers);
  const evaluation = useKitchenStore((s) => s.evaluation);
  const learning = useKitchenStore((s) => s.learning);
  // Only for the stack on screen: a result learned for another shape is not
  // "this shape" — see `currentLearned`.
  const learned = useKitchenStore(currentLearned);
  const windowKernel = useKitchenStore((s) => s.windowKernel);

  const setWeight = useKitchenStore((s) => s.setWeight);
  const applyPreset = useKitchenStore((s) => s.applyPreset);
  const addKernel = useKitchenStore((s) => s.addKernel);
  const removeKernel = useKitchenStore((s) => s.removeKernel);
  const setPool = useKitchenStore((s) => s.setPool);
  const addLayer = useKitchenStore((s) => s.addLayer);
  const removeLayer = useKitchenStore((s) => s.removeLayer);
  const setWindowKernel = useKitchenStore((s) => s.setWindowKernel);
  const letItLearn = useKitchenStore((s) => s.letItLearn);
  const reset = useKitchenStore((s) => s.reset);

  // Deliberately NOT `scoring || learning`. Refitting the head takes a second or
  // two and happens after every edit; freezing the steppers for it would make the
  // whole panel feel broken. Stale results are dropped in the store instead.
  const busy = learning;

  return (
    <div className="flex flex-col gap-4">
      <LayerStack
        layers={layers}
        health={evaluation?.health ?? []}
        disabled={busy}
        selectedKernel={windowKernel}
        onWeight={setWeight}
        onPreset={applyPreset}
        onAddKernel={addKernel}
        onRemoveKernel={removeKernel}
        onPool={setPool}
        onAddLayer={addLayer}
        onRemoveLayer={removeLayer}
        onSelectKernel={setWindowKernel}
      />

      <div className="border-t border-border pt-4">
        <Button
          variant="secondary"
          className="w-full"
          onClick={() => void letItLearn()}
          disabled={busy || filtersUsed(layers) === 0}
          icon={<Sparkles className="size-4" />}
        >
          {learning ? "Training…" : "Let it learn"}
        </Button>
        <p className="mt-1.5 text-xs text-text-muted">
          {learning
            ? `Training a stack of this shape end to end. Edits wait until it is done.`
            : learned === null
              ? `Trains a stack of your shape end to end and shows the kernels gradient descent picks instead. It is the honest check on whether your design or your architecture is the limit.`
              : `Gradient descent got ${Math.round(
                  learned.accuracy * 100,
                )}% out of this shape.`}
        </p>
        {learned !== null && !learning ? <LearnedKernels learned={learned} /> : null}

        <Button
          variant="ghost"
          className="mt-3 w-full"
          onClick={() => reset()}
          disabled={busy}
          icon={<RotateCcw className="size-4" />}
        >
          Empty the kitchen
        </Button>
      </div>
    </div>
  );
}

export default function ConvolutionKitchen() {
  const meta = getGameMeta(SLUG);

  const layers = useKitchenStore((s) => s.layers);
  const score = useKitchenStore((s) => s.score);
  const baseline = useKitchenStore((s) => s.baseline);
  const evaluation = useKitchenStore((s) => s.evaluation);
  const learned = useKitchenStore(currentLearned);
  const scoring = useKitchenStore((s) => s.scoring);
  const phase = useKitchenStore((s) => s.phase);
  const failure = useKitchenStore((s) => s.failure);
  const whyCard = useKitchenStore((s) => s.whyCard);
  const lane = useKitchenStore((s) => s.lane);
  const setLane = useKitchenStore((s) => s.setLane);
  const runScore = useKitchenStore((s) => s.score_);
  const reset = useKitchenStore((s) => s.reset);
  const games = useProgression((s) => s.games);

  const hydrate = useProgression((s) => s.hydrate);
  const hydrated = useProgression((s) => s.hydrated);
  const lastGain = useProgression((s) => s.lastGain);
  const xp = useProgression((s) => s.xp);
  const level = levelFromXp(xp);

  useEffect(() => {
    if (!hydrated) void hydrate();
  }, [hydrated, hydrate]);

  // The opening blur has to be scored, or the metric would start blank and the
  // player would have nothing to react to.
  useEffect(() => {
    if (evaluation === null && !scoring) void runScore();
  }, [evaluation, scoring, runScore]);

  const filters = filtersUsed(layers);
  const channels = outputChannels(layers);
  const geometry = stackGeometry(layers);

  /**
   * The live metric (pedagogy contract #3): "Detection score", per the catalog.
   *
   * Validation accuracy of the fixed head. It is the honest number here because
   * the head cannot grow: a point of detection score is a point that the player's
   * kernels earned, which is the entire claim the game makes about itself.
   */
  const metric: MetricSpec = {
    label: "Detection score",
    value: evaluation === null ? Number.NaN : score.accuracy,
    format: "percent",
    goodDirection: "up",
    state:
      phase === "served" ? "good" : failure !== null ? "bad" : undefined,
    caption:
      evaluation === null
        ? "scoring the opening filter…"
        : scoring
          ? "refitting the classifier…"
          : `${Math.round(TARGET_ACCURACY * 100)}% needed · ${Math.round(
              CHANCE_RATE * 100,
            )}% is guessing`,
  };

  const secondaryMetrics: MetricSpec[] = [
    {
      label: "Filters used",
      value: filters,
      format: "integer",
      goodDirection: "down",
      caption: `of ${FILTER_BUDGET} · ${channels} channel${
        channels === 1 ? "" : "s"
      } out`,
      state: filters > FILTER_BUDGET ? "bad" : undefined,
    },
    {
      // Depth made visible as a number, because "one cell sees 10 pixels" is the
      // mechanism behind the whole hierarchy idea.
      label: "Receptive field",
      value: geometry.receptiveField,
      format: "integer",
      goodDirection: "up",
      caption: `pixels per output cell · maps ${geometry.sizes.join("→") || "—"}`,
    },
    {
      label: "Raw pixels score",
      value: baseline === null ? Number.NaN : baseline,
      format: "percent",
      goodDirection: "up",
      caption:
        baseline === null
          ? "fitting the baseline…"
          : "same classifier, no convolution",
    },
    {
      label: "Learned kernels",
      value: learned === null ? Number.NaN : learned.accuracy,
      format: "percent",
      goodDirection: "up",
      caption:
        learned === null
          ? "press Let it learn"
          : learned.accuracy > score.accuracy + 0.05
            ? "this shape has room left"
            : "you matched gradient descent",
    },
  ];

  return (
    <GameShell
      slug={SLUG}
      title={meta?.title ?? "Convolution Kitchen"}
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
      onRetry={() => reset()}
      onNext={phase === "served" ? () => reset() : undefined}
      nextLabel={phase === "served" ? "Try it with fewer filters" : undefined}
    />
  );
}
