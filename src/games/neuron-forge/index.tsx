"use client";

import { useEffect, useMemo } from "react";
import { Layers, Minus, Play, Plus, RotateCcw, Square, Trash2 } from "lucide-react";
import { Button, type MetricSpec } from "@/components";
import { GameShell } from "@/engine/GameShell";
import {
  HIGH_SCORE_THRESHOLD,
  levelFromXp,
  useProgression,
} from "@/engine/progression";
import { getGameMeta } from "@/lib/catalog";
import {
  ACTIVATIONS,
  MATH_CODE,
  MATH_EQUATION,
  MATH_NOTES,
  MAX_LAYERS,
  MAX_NEURONS_PER_LAYER,
  MIN_NEURONS_PER_LAYER,
  PATTERNS,
  architectureOf,
  isScored,
  type Activation,
  type PatternId,
} from "./ml";
import {
  SLUG,
  budgetRemaining,
  canAddLayer,
  pattern as patternSelector,
  useNeuronForgeStore,
} from "./store";
import { BudgetBar } from "./BudgetBar";
import { VisualLane } from "./VisualLane";
import { CodeLane } from "./CodeLane";
import { useNeuronTrainer, type TrainerApi } from "./useTrainer";

/**
 * The engine's rule, stated for this game: ★1 any solve, ★2 a best score of at
 * least HIGH_SCORE_THRESHOLD, ★3 that plus a solve started from the code lane.
 * The score is held-out accuracy × the efficiency bonus, so the second star is
 * earned by solving lean, not by solving at all.
 */
const STAR_CRITERIA = [
  "Solve a pattern",
  `Score ${Math.round(
    HIGH_SCORE_THRESHOLD * 100,
  )}% or better — held-out accuracy × efficiency, so solve it lean`,
  "Solve one with api.train() in the code lane",
];

function PatternPicker() {
  const patternId = useNeuronForgeStore((s) => s.patternId);
  const setPattern = useNeuronForgeStore((s) => s.setPattern);
  const training = useNeuronForgeStore((s) => s.training);

  return (
    <div>
      <label
        htmlFor="pattern-select"
        className="block text-sm font-medium"
      >
        Pattern
      </label>
      <select
        id="pattern-select"
        value={patternId}
        disabled={training}
        onChange={(event) => setPattern(event.target.value as PatternId)}
        className="mt-1 min-h-11 w-full rounded-md border border-border bg-surface-2 px-2 py-1.5 text-sm disabled:opacity-60"
      >
        {PATTERNS.map((spec) => (
          <option key={spec.id} value={spec.id}>
            {spec.name} — {spec.budget} neurons, {Math.round(spec.target * 100)}%
          </option>
        ))}
      </select>
      <p className="mt-1 text-xs text-text-muted">
        Switching pattern clears the architecture — a design that fits one shape
        rarely fits the next.
      </p>
    </div>
  );
}

/**
 * The architecture editor.
 *
 * This is "player action = algorithm" (pedagogy contract #2): every control here
 * writes to the `layers` array that `buildModel` hands to TensorFlow. There is no
 * translation layer and no simulation — the stepper below IS the `units:` argument
 * of a `tf.layers.dense` call.
 */
function LayerEditor() {
  const layers = useNeuronForgeStore((s) => s.layers);
  const training = useNeuronForgeStore((s) => s.training);
  const addLayer = useNeuronForgeStore((s) => s.addLayer);
  const removeLayer = useNeuronForgeStore((s) => s.removeLayer);
  const setNeurons = useNeuronForgeStore((s) => s.setNeurons);
  const setActivation = useNeuronForgeStore((s) => s.setActivation);
  const canAdd = useNeuronForgeStore(canAddLayer);
  const remaining = useNeuronForgeStore(budgetRemaining);
  // Derived from selected state so the +/− buttons re-enable reactively. Reading
  // this via getState() during render would not re-render when the budget frees up.
  const spare = remaining;

  return (
    <fieldset className="border-t border-border pt-4">
      <legend className="text-sm font-medium">Hidden layers</legend>

      {layers.length === 0 ? (
        <p className="mt-1 mb-2 text-xs text-text-muted">
          None. Inputs go straight to the output, so the network is a single
          straight cut — logistic regression.
        </p>
      ) : (
        <ul className="mt-2 flex flex-col gap-2">
          {layers.map((layer, index) => {
            const canGrow =
              !training &&
              layer.neurons < MAX_NEURONS_PER_LAYER &&
              spare >= 1;
            return (
              <li
                key={index}
                className="rounded-md border border-border bg-surface-2 p-2"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-medium">Layer {index + 1}</span>
                  {/* 44×44 (DESIGN.md §9) around the same 14px icon. */}
                  <button
                    type="button"
                    onClick={() => removeLayer(index)}
                    disabled={training}
                    aria-label={`Remove layer ${index + 1}`}
                    className="inline-flex min-h-11 min-w-11 items-center justify-center rounded text-text-muted hover:bg-surface hover:text-wrong focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none disabled:opacity-50"
                  >
                    <Trash2 aria-hidden="true" className="size-3.5" />
                  </button>
                </div>

                {/* Wraps rather than overflowing: at 44px the steppers, the
                    count and the select can outgrow the 320px rail. */}
                <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                  <button
                    type="button"
                    onClick={() => setNeurons(index, layer.neurons - 1)}
                    disabled={training || layer.neurons <= MIN_NEURONS_PER_LAYER}
                    aria-label={`Remove a neuron from layer ${index + 1}, currently ${layer.neurons}`}
                    className="inline-flex min-h-11 min-w-11 items-center justify-center rounded border border-border hover:bg-surface focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none disabled:opacity-40"
                  >
                    <Minus aria-hidden="true" className="size-3.5" />
                  </button>

                  <span
                    className="min-w-[5.5rem] text-center font-mono text-sm"
                    aria-hidden="true"
                  >
                    {layer.neurons} neuron{layer.neurons === 1 ? "" : "s"}
                  </span>

                  <button
                    type="button"
                    onClick={() => setNeurons(index, layer.neurons + 1)}
                    disabled={!canGrow}
                    aria-label={`Add a neuron to layer ${index + 1}, currently ${layer.neurons}`}
                    className="inline-flex min-h-11 min-w-11 items-center justify-center rounded border border-border hover:bg-surface focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none disabled:opacity-40"
                  >
                    <Plus aria-hidden="true" className="size-3.5" />
                  </button>

                  <label className="sr-only" htmlFor={`activation-${index}`}>
                    Activation for layer {index + 1}
                  </label>
                  <select
                    id={`activation-${index}`}
                    value={layer.activation}
                    disabled={training}
                    onChange={(event) =>
                      setActivation(index, event.target.value as Activation)
                    }
                    className="ml-auto min-h-11 rounded border border-border bg-surface px-1.5 py-1 font-mono text-xs disabled:opacity-60"
                  >
                    {ACTIVATIONS.map((activation) => (
                      <option key={activation} value={activation}>
                        {activation}
                      </option>
                    ))}
                  </select>
                </div>
              </li>
            );
          })}
        </ul>
      )}

      <Button
        variant="secondary"
        size="sm"
        className="mt-2 w-full"
        onClick={addLayer}
        disabled={!canAdd}
        icon={<Layers className="size-4" />}
      >
        Add a layer
      </Button>
      {!canAdd && !training ? (
        <p className="mt-1 text-xs text-text-muted">
          {layers.length >= MAX_LAYERS
            ? `${MAX_LAYERS} layers is the maximum.`
            : `No budget left for another layer (${remaining} spare).`}
        </p>
      ) : null}
    </fieldset>
  );
}

function Controls({ trainer }: { trainer: TrainerApi }) {
  const layers = useNeuronForgeStore((s) => s.layers);
  const spec = useNeuronForgeStore(patternSelector);
  const training = useNeuronForgeStore((s) => s.training);
  const reset = useNeuronForgeStore((s) => s.reset);
  const used = useMemo(() => architectureOf(layers).totalNeurons, [layers]);

  return (
    <div className="flex flex-col gap-4">
      <PatternPicker />
      <BudgetBar used={used} budget={spec.budget} />
      <LayerEditor />

      <div className="border-t border-border pt-4">
        {training ? (
          <>
            <Button
              variant="secondary"
              className="w-full"
              onClick={trainer.stop}
              icon={<Square className="size-4" />}
            >
              Stop training
            </Button>
            <p className="mt-1.5 text-center font-mono text-xs text-text-muted">
              epoch {trainer.epoch} of {trainer.totalEpochs}
            </p>
          </>
        ) : (
          <Button
            variant="primary"
            className="w-full"
            onClick={() => void trainer.train()}
            icon={<Play className="size-4" />}
          >
            Train network
          </Button>
        )}

        <Button
          variant="ghost"
          className="mt-2 w-full"
          onClick={reset}
          disabled={training}
          icon={<RotateCcw className="size-4" />}
        >
          Clear the forge
        </Button>

        {trainer.error ? (
          <p className="mt-2 text-xs text-wrong">{trainer.error}</p>
        ) : null}
      </div>
    </div>
  );
}

export default function NeuronForge() {
  const meta = getGameMeta(SLUG);
  const trainer = useNeuronTrainer();

  const patternId = useNeuronForgeStore((s) => s.patternId);
  const spec = useNeuronForgeStore(patternSelector);
  const layers = useNeuronForgeStore((s) => s.layers);
  const loss = useNeuronForgeStore((s) => s.loss);
  const accuracy = useNeuronForgeStore((s) => s.accuracy);
  const epoch = useNeuronForgeStore((s) => s.epoch);
  const training = useNeuronForgeStore((s) => s.training);
  const lastEvaluation = useNeuronForgeStore((s) => s.lastEvaluation);
  const failure = useNeuronForgeStore((s) => s.failure);
  const whyCard = useNeuronForgeStore((s) => s.whyCard);
  const won = useNeuronForgeStore((s) => s.won);
  const lane = useNeuronForgeStore((s) => s.lane);
  const setLane = useNeuronForgeStore((s) => s.setLane);
  const setPattern = useNeuronForgeStore((s) => s.setPattern);
  const retry = useNeuronForgeStore((s) => s.retry);

  const totalNeurons = useMemo(
    () => architectureOf(layers).totalNeurons,
    [layers],
  );

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
   * The live metric (pedagogy contract #3): training loss, updated every epoch
   * from real `model.fit` logs, with the loss curve beside it making "converged
   * but still wrong" visible.
   *
   * Deliberate deviation: the spec states the win as "loss below threshold", but
   * the win here is held-out accuracy (the readout below). Training loss says
   * how well the network fits the points it has seen; only held-out accuracy can
   * support a claim about what the architecture can represent, and a threshold
   * on loss would also reward memorising the 2% of flipped labels.
   */
  const metric: MetricSpec = {
    label: "Loss",
    value: loss,
    format: "decimal",
    precision: 4,
    goodDirection: "down",
    state: failure ? "bad" : won ? "good" : undefined,
    caption: training
      ? `epoch ${epoch} of ${trainer.totalEpochs}`
      : Number.isFinite(loss)
        ? "binary cross-entropy on training points"
        : "train to measure",
  };

  const secondaryMetrics: MetricSpec[] = [
    {
      label: "Held-out accuracy",
      value: accuracy ?? Number.NaN,
      format: "percent",
      goodDirection: "up",
      caption: `need ${Math.round(spec.target * 100)}%`,
      // Red only for a judged miss. A stopped run's accuracy is real but was
      // never scored, so it gets no verdict colour either way.
      state: won ? "good" : failure ? "bad" : undefined,
    },
    {
      label: "Neurons",
      value: totalNeurons,
      format: "integer",
      goodDirection: "down",
      caption: `of ${spec.budget} budget`,
    },
    {
      label: "Score",
      // A stopped or unfinished run is not scored, so it shows "—", not a
      // "0%" that reads as a measured result.
      value: isScored(lastEvaluation) ? lastEvaluation.score : Number.NaN,
      format: "percent",
      goodDirection: "up",
      caption:
        lastEvaluation === null
          ? "accuracy × efficiency"
          : isScored(lastEvaluation)
            ? `${Math.round(lastEvaluation.efficiency * 100)}% efficiency bonus`
            : "not scored",
      state: won ? "good" : undefined,
    },
  ];

  // Advance through the escalating puzzles on a win.
  const currentIndex = PATTERNS.findIndex((entry) => entry.id === patternId);
  const nextPattern = PATTERNS[currentIndex + 1];

  return (
    <GameShell
      slug={SLUG}
      title={meta?.title ?? "Neuron Forge"}
      metric={metric}
      secondaryMetrics={secondaryMetrics}
      math={{
        equation: MATH_EQUATION,
        code: MATH_CODE,
        codeLanguage: "javascript",
        notes: MATH_NOTES,
      }}
      controls={<Controls trainer={trainer} />}
      visual={<VisualLane />}
      code={<CodeLane trainer={trainer} />}
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
      // Keep the architecture: the failure copy names the control to change, and
      // the network it refers to has to still be there to change it. A fit in
      // flight is stopped and waited out first, so it cannot land afterwards.
      onRetry={() => void trainer.stopAndWait().then(retry)}
      onNext={
        won && nextPattern ? () => setPattern(nextPattern.id) : undefined
      }
      nextLabel={nextPattern ? `Next: ${nextPattern.name}` : undefined}
    />
  );
}
