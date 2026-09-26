"use client";

import { create } from "zustand";
import { useProgression } from "@/engine/progression";
import type { Lane, NamedFailure } from "@/engine/types";
import type { WhyCardContent } from "@/components";
import { clamp } from "@/lib/utils";
import {
  MAX_LAYERS,
  MAX_NEURONS_PER_LAYER,
  MIN_NEURONS_PER_LAYER,
  TRAIN_EPOCHS,
  architectureOf,
  evaluate,
  generateDataset,
  patternById,
  type Activation,
  type Architecture,
  type Collapse,
  type Dataset,
  type Evaluation,
  type Layer,
  type PatternId,
  type PatternSpec,
} from "./ml";
import { whyCardFor } from "./why-cards";

export const SLUG = "neuron-forge";

/** Fixed so the puzzle a player sees matches the one the tests measure. */
export const DATASET_SEED = 7001;

/**
 * Neuron Forge state.
 *
 * The spec's data model is:
 *
 *   Layer        { neurons:int, activation:enum }
 *   Architecture { layers[], totalNeurons }
 *   GameState    { arch, loss, epoch, budget, patternId }
 *
 * `layers` is the single source of truth and `arch` is derived from it, so
 * `totalNeurons` can never drift out of sync with the layers it counts. `budget`
 * is not stored either — it belongs to the pattern, and duplicating it here would
 * be a second thing to keep correct.
 *
 * ONE store, both lanes (CLAUDE.md two-lane rule). The code lane calls
 * `applyArchitecture` with a layer list it computed; the visual lane calls
 * `addLayer`/`setNeurons`/`setActivation`. Both land in the same `layers` array
 * and both train through the same path, so the lanes cannot disagree.
 */
export interface NeuronForgeState {
  patternId: PatternId;
  layers: Layer[];
  dataset: Dataset;

  // ── training ─────────────────────────────────────────────────────────
  training: boolean;
  /** 1-based, for "epoch 40 of 160". */
  epoch: number;
  /** Training loss, the live metric. NaN before the first epoch. */
  loss: number;
  /** Loss per epoch, for `<LossCurve>`. */
  lossHistory: number[];
  /** Held-out accuracy after training, null before. */
  accuracy: number | null;
  /**
   * P(class 1) over the input grid, for `<DecisionSurface>`. Refreshed during
   * training so the boundary visibly forms, per the spec's live feedback.
   */
  surface: Float32Array | null;

  lastEvaluation: Evaluation | null;
  failure: NamedFailure | null;
  whyCard: WhyCardContent | null;
  won: boolean;
  lane: Lane;
  /**
   * Which training run is current. Bumped by `beginTraining`, and handed back by
   * the trainer when the fit resolves, so a result can only land on the run that
   * asked for it. Every edit is already locked while `training`, so no second
   * run can start under a first; the token makes "a stale fit never writes into
   * a fresh run" a property of the store rather than of every caller remembering
   * to check.
   */
  attempt: number;

  // ── actions ──────────────────────────────────────────────────────────
  setLane: (lane: Lane) => void;
  setPattern: (patternId: PatternId) => void;

  addLayer: () => void;
  removeLayer: (index: number) => void;
  setNeurons: (index: number, neurons: number) => void;
  setActivation: (index: number, activation: Activation) => void;
  /** Replace the whole architecture — the code lane's entry point. */
  applyArchitecture: (layers: Layer[]) => void;

  /** Returns the run's `attempt` token, for `finishTraining`. */
  beginTraining: () => number;
  recordEpoch: (epoch: number, loss: number) => void;
  setSurface: (surface: Float32Array) => void;
  /**
   * Training finished: score it, name any failure, bank progress.
   *
   * Returns null — and changes nothing — when `attempt` names a run that is no
   * longer current.
   */
  finishTraining: (
    result: TrainingResult,
    attempt?: number,
  ) => Evaluation | null;

  /**
   * Clear the last result but keep the architecture — the named failure's
   * Retry. The failure copy tells the player which control to change; wiping
   * the network they were told to adjust would throw that advice away.
   */
  retry: () => void;
  /** Clear the architecture, keep the puzzle. */
  reset: () => void;
}

export interface TrainingResult {
  /** Held-out accuracy, or null when the network could not be measured. */
  accuracy: number | null;
  surface: Float32Array | null;
  /** From `diagnoseCollapse`, when the output came out flat. */
  collapse?: Collapse | null;
  /** Epochs the fit ran. Omitted means all of them. */
  epochsRun?: number;
  /**
   * True when a code-lane `api.train()` started this run. Mastery's third star
   * and the code-lane XP multiplier go to the path that did the work, not to
   * whichever tab happened to be open when the rail's Train button was pressed.
   */
  fromCode?: boolean;
}

/** What an edit did, for the why-card's opening clause. */
function changeBetween(
  before: Layer[],
  after: Layer[],
): "layer-added" | "layer-removed" | "neurons" | "activation" {
  if (after.length > before.length) return "layer-added";
  if (after.length < before.length) return "layer-removed";
  return after.some((layer, index) => layer.activation !== before[index]?.activation)
    ? "activation"
    : "neurons";
}

/**
 * Selectors below return primitives or stable references only.
 *
 * Deliberately no `architecture` selector: `architectureOf` builds a new object,
 * and a selector returning a fresh reference each call breaks React's
 * `useSyncExternalStore` contract that snapshots be cached. Components select
 * `layers` — a stable array — and memoise the derived architecture themselves.
 *
 * `patternById` is safe because it returns an element of a module constant, so
 * the same puzzle always yields the identical object.
 */
export function pattern(state: NeuronForgeState): PatternSpec {
  return patternById(state.patternId);
}

/** Neurons still available under the cap. */
export function budgetRemaining(state: NeuronForgeState): number {
  return patternById(state.patternId).budget - architectureOf(state.layers).totalNeurons;
}

/**
 * Can another layer be added?
 *
 * The budget is enforced by disabling controls rather than by failing the player
 * afterwards. Going over budget is a rule of the puzzle, not a machine-learning
 * mistake, and dressing it up as a named failure would dilute the two that are.
 */
export function canAddLayer(state: NeuronForgeState): boolean {
  return (
    state.layers.length < MAX_LAYERS &&
    budgetRemaining(state) >= MIN_NEURONS_PER_LAYER &&
    !state.training
  );
}

export function canGrowLayer(state: NeuronForgeState, index: number): boolean {
  const layer = state.layers[index];
  if (!layer || state.training) return false;
  return layer.neurons < MAX_NEURONS_PER_LAYER && budgetRemaining(state) >= 1;
}

/** Blank slate for a puzzle: results cleared so no stale number is on screen. */
function untrained() {
  return {
    training: false,
    epoch: 0,
    loss: Number.NaN,
    lossHistory: [] as number[],
    accuracy: null,
    surface: null,
    lastEvaluation: null,
    failure: null,
    won: false,
  };
}

export const useNeuronForgeStore = create<NeuronForgeState>((set, get) => ({
  patternId: "linear",
  layers: [],
  dataset: generateDataset("linear", DATASET_SEED),
  ...untrained(),
  whyCard: whyCardFor({ kind: "reset" }),
  lane: "visual" as Lane,
  attempt: 0,

  setLane: (lane) => set({ lane }),

  setPattern: (patternId) => {
    if (get().training) return;
    const spec = patternById(patternId);
    set({
      patternId,
      // The architecture resets with the puzzle. Carrying a 16-neuron design
      // into a 6-neuron budget would start the player in an invalid state.
      layers: [],
      dataset: generateDataset(patternId, DATASET_SEED),
      ...untrained(),
      whyCard: whyCardFor({ kind: "pattern-changed", pattern: spec }),
    });
  },

  addLayer: () => {
    const state = get();
    if (!canAddLayer(state)) return;
    // One neuron, relu: the smallest honest starting point. Defaulting to
    // something generous would hide the capacity ladder the game is teaching.
    const layers = [
      ...state.layers,
      { neurons: MIN_NEURONS_PER_LAYER, activation: "relu" as Activation },
    ];
    set({
      layers,
      ...untrained(),
      whyCard: whyCardFor({
        kind: "architecture-changed",
        architecture: architectureOf(layers),
        pattern: patternById(state.patternId),
        change: "layer-added",
      }),
    });
  },

  removeLayer: (index) => {
    const state = get();
    if (state.training || !state.layers[index]) return;
    const layers = state.layers.filter((_, at) => at !== index);
    set({
      layers,
      ...untrained(),
      whyCard: whyCardFor({
        kind: "architecture-changed",
        architecture: architectureOf(layers),
        pattern: patternById(state.patternId),
        change: "layer-removed",
      }),
    });
  },

  setNeurons: (index, neurons) => {
    const state = get();
    const current = state.layers[index];
    if (state.training || !current) return;

    // Clamp to the cap AND to whatever the budget still allows, so the control
    // physically cannot produce an over-budget architecture.
    const others = architectureOf(state.layers).totalNeurons - current.neurons;
    const ceiling = Math.min(
      MAX_NEURONS_PER_LAYER,
      patternById(state.patternId).budget - others,
    );
    const next = clamp(
      Math.round(neurons),
      MIN_NEURONS_PER_LAYER,
      Math.max(MIN_NEURONS_PER_LAYER, ceiling),
    );
    if (next === current.neurons) return;

    const layers = state.layers.map((layer, at) =>
      at === index ? { ...layer, neurons: next } : layer,
    );
    set({
      layers,
      ...untrained(),
      whyCard: whyCardFor({
        kind: "architecture-changed",
        architecture: architectureOf(layers),
        pattern: patternById(state.patternId),
        change: "neurons",
      }),
    });
  },

  setActivation: (index, activation) => {
    const state = get();
    if (state.training || !state.layers[index]) return;
    const layers = state.layers.map((layer, at) =>
      at === index ? { ...layer, activation } : layer,
    );
    set({
      layers,
      ...untrained(),
      whyCard: whyCardFor({
        kind: "architecture-changed",
        architecture: architectureOf(layers),
        pattern: patternById(state.patternId),
        change: "activation",
      }),
    });
  },

  applyArchitecture: (incoming) => {
    const state = get();
    if (state.training) return;
    const spec = patternById(state.patternId);

    // Code from the code lane is untrusted input: clamp it into a legal
    // architecture rather than letting it build something the visual lane could
    // never express. Silently dropping the overflow would be confusing, so the
    // trailing layers are cut and the why-card reports the spend.
    const layers: Layer[] = [];
    let spent = 0;
    for (const layer of incoming.slice(0, MAX_LAYERS)) {
      const room = spec.budget - spent;
      if (room < MIN_NEURONS_PER_LAYER) break;
      const neurons = clamp(
        Math.round(Number(layer?.neurons) || MIN_NEURONS_PER_LAYER),
        MIN_NEURONS_PER_LAYER,
        Math.min(MAX_NEURONS_PER_LAYER, room),
      );
      layers.push({ neurons, activation: layer.activation });
      spent += neurons;
    }

    set({
      layers,
      ...untrained(),
      whyCard: whyCardFor({
        kind: "architecture-changed",
        architecture: architectureOf(layers),
        pattern: spec,
        // Derived from the diff: a script that swaps an activation or adds a
        // layer should not be told "Width changed".
        change: changeBetween(state.layers, layers),
      }),
    });
  },

  beginTraining: () => {
    const state = get();
    const attempt = state.attempt + 1;
    set({
      attempt,
      training: true,
      epoch: 0,
      loss: Number.NaN,
      lossHistory: [],
      accuracy: null,
      lastEvaluation: null,
      failure: null,
      won: false,
      whyCard: whyCardFor({
        kind: "training-started",
        architecture: architectureOf(state.layers),
        pattern: patternById(state.patternId),
      }),
    });
    return attempt;
  },

  recordEpoch: (epoch, loss) => {
    set((state) => ({
      epoch: epoch + 1,
      loss,
      lossHistory: [...state.lossHistory, loss].slice(-TRAIN_EPOCHS),
    }));
  },

  setSurface: (surface) => set({ surface }),

  finishTraining: (
    { accuracy, surface, collapse = null, epochsRun, fromCode = false },
    attempt,
  ) => {
    const state = get();
    if (attempt !== undefined && attempt !== state.attempt) return null;

    const spec = patternById(state.patternId);
    const architecture = architectureOf(state.layers);
    const evaluation = evaluate({
      pattern: spec,
      architecture,
      accuracy,
      loss: state.loss,
      collapse,
      epochsRun,
    });

    set({
      training: false,
      accuracy,
      ...(surface ? { surface } : {}),
      lastEvaluation: evaluation,
      failure: evaluation.failure,
      won: evaluation.outcome === "win",
      whyCard: whyCardFor({
        kind: "trained",
        evaluation,
        pattern: spec,
        architecture,
        collapse,
      }),
    });

    if (evaluation.outcome === "win") {
      useProgression.getState().recordResult({
        slug: SLUG,
        score: evaluation.score,
        lane: fromCode ? "code" : "visual",
        completed: true,
        codeLaneCleared: fromCode,
      });
    }

    return evaluation;
  },

  retry: () => {
    const state = get();
    if (state.training) return;
    set({
      ...untrained(),
      whyCard: whyCardFor({
        kind: "retry",
        architecture: architectureOf(state.layers),
        pattern: patternById(state.patternId),
      }),
    });
  },

  reset: () => {
    if (get().training) return;
    set({
      layers: [],
      ...untrained(),
      whyCard: whyCardFor({ kind: "reset" }),
    });
  },
}));

export type { Activation, Architecture, Evaluation, Layer, PatternId, PatternSpec };
