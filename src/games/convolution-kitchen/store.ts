"use client";

import { create } from "zustand";
import { useProgression } from "@/engine/progression";
import type { Lane, NamedFailure } from "@/engine/types";
import type { WhyCardContent } from "@/components";
import { clamp } from "@/lib/utils";
import {
  CLASS_COUNT,
  FILTER_BUDGET,
  IMAGE_SIZE,
  KERNEL_CELLS,
  MAX_KERNELS_PER_LAYER,
  MAX_LAYERS,
  WEIGHT_MAX,
  WEIGHT_MIN,
  describeKernel,
  evaluateKitchen,
  featureMapsFor,
  filtersUsed,
  generateDataset,
  inspectFilters,
  learnStack,
  makeKernel,
  makeLayer,
  presetByLabel,
  scoreRawPixels,
  scoreStack,
  type Dataset,
  type Evaluation,
  type Kernel,
  type Layer,
  type LearnedResult,
  type PoolType,
  type Sample,
  type ScoreResult,
} from "./ml";
import { whyCardFor } from "./why-cards";

export const SLUG = "convolution-kitchen";
export const DATASET_SEED = 4711;

export type Phase = "designing" | "served";

const emptyScore = (): ScoreResult => ({
  accuracy: 0,
  trainAccuracy: 0,
  perClass: new Array<number>(CLASS_COUNT).fill(0),
  confusion: Array.from({ length: CLASS_COUNT }, () =>
    new Array<number>(CLASS_COUNT).fill(0),
  ),
});

/**
 * Convolution Kitchen state.
 *
 * The spec's data model is:
 *
 *   Kernel    { weights[3x3], stride, padding }
 *   Layer     { kernels[], poolType }
 *   GameState { layers[], inputImage, featureMaps[][] }
 *
 * `stride` and `padding` are not on `Kernel` — see deviation 2 in ml.ts. They are
 * fixed at 1 and "valid" and explained in the math drawer rather than exposed as
 * two more controls for one idea.
 *
 * `featureMaps[][]` is NOT in state, and that is deliberate rather than an
 * omission. Feature maps are a pure function of one image and the layer stack, and
 * they are cheap — a single forward pass over a 24x24 picture. Storing them would
 * mean a second copy of the truth that has to be invalidated on every weight
 * change, and a stale feature map is the one bug in this game that would be
 * actively misleading. The visual lane derives them in `useMemo` instead.
 *
 * `inputImage` is `previewIndex` into the validation split: the images are
 * generated, seeded and immutable, so an index is the whole of the state.
 */
export interface KitchenState {
  dataset: Dataset;
  layers: Layer[];

  /** Which validation image the gallery and the sliding window are showing. */
  previewIndex: number;
  /** Top-left corner of the sliding 3x3 window, in image coordinates. */
  windowRow: number;
  windowCol: number;
  /** Which layer-1 kernel the sliding window is demonstrating. */
  windowKernel: number;

  score: ScoreResult;
  /** Same head on raw pixels. Fitted once; it never changes. */
  baseline: number | null;
  evaluation: Evaluation | null;
  learned: LearnedResult | null;

  scoring: boolean;
  learning: boolean;
  phase: Phase;
  failure: NamedFailure | null;
  whyCard: WhyCardContent | null;
  lane: Lane;

  setLane: (lane: Lane) => void;
  setPreview: (index: number) => void;
  moveWindow: (row: number, col: number) => void;
  setWindowKernel: (index: number) => void;

  setWeight: (layerIndex: number, kernelIndex: number, cell: number, value: number) => void;
  applyPreset: (layerIndex: number, kernelIndex: number, preset: string) => void;
  addKernel: (layerIndex: number, preset?: string) => void;
  removeKernel: (layerIndex: number, kernelIndex: number) => void;
  setPool: (layerIndex: number, pool: PoolType) => void;
  addLayer: () => void;
  removeLayer: (layerIndex: number) => void;

  /** Fit the fixed head on the current features and judge the kitchen. */
  score_: () => Promise<void>;
  /** Train the same-shaped stack end to end, for comparison. */
  letItLearn: () => Promise<void>;
  reset: () => void;
}

// ── selectors: primitives and stable references only ─────────────────────

export const detectionScore = (state: KitchenState): number =>
  state.score.accuracy;
export const filterCount = (state: KitchenState): number =>
  filtersUsed(state.layers);
export const budgetLeft = (state: KitchenState): number =>
  Math.max(0, FILTER_BUDGET - filtersUsed(state.layers));
export const previewSample = (state: KitchenState): Sample =>
  state.dataset.validation[state.previewIndex]!;

/**
 * The starting kitchen.
 *
 * One layer, one blur. That is the mistake almost everyone makes unprompted — a
 * 3x3 average feels like "look at the neighbourhood" — and it is provably useless
 * here, so the game opens on a filter the player will have to fix rather than on a
 * blank grid that tells them nothing.
 */
function startingLayers(): Layer[] {
  return [makeLayer([makeKernel("Blur")], "avg")];
}

const dataset = generateDataset(DATASET_SEED);

/**
 * Supersedes in-flight scoring runs.
 *
 * Fitting the head takes a second or two, and edits must not wait for it — a
 * stepper that swallows your second click because a background fit is running
 * feels broken, and the first version of this store did exactly that by guarding
 * every edit on `scoring`. Instead every run takes a ticket, and a run that
 * finishes holding a stale ticket throws its result away. The player edits freely;
 * only the last request gets to write.
 */
let scoreTicket = 0;

export const useKitchenStore = create<KitchenState>((set, get) => ({
  dataset,
  layers: startingLayers(),

  previewIndex: 0,
  windowRow: 8,
  windowCol: 8,
  windowKernel: 0,

  score: emptyScore(),
  baseline: null,
  evaluation: null,
  learned: null,

  scoring: false,
  learning: false,
  phase: "designing",
  failure: null,
  whyCard: whyCardFor({ kind: "briefing" }),
  lane: "visual" as Lane,

  setLane: (lane) => set({ lane }),

  setPreview: (index) =>
    set({
      previewIndex: clamp(
        Math.round(index),
        0,
        get().dataset.validation.length - 1,
      ),
    }),

  moveWindow: (row, col) =>
    set({
      windowRow: clamp(Math.round(row), 0, IMAGE_SIZE - 3),
      windowCol: clamp(Math.round(col), 0, IMAGE_SIZE - 3),
    }),

  setWindowKernel: (index) => {
    const kernels = get().layers[0]?.kernels ?? [];
    if (kernels.length === 0) return;
    set({ windowKernel: clamp(Math.round(index), 0, kernels.length - 1) });
  },

  setWeight: (layerIndex, kernelIndex, cell, value) => {
    const state = get();
    // Edits are NOT blocked while scoring: see scoreTicket. Only an explicit
    // end-to-end training run, which is a deliberate click with its own label,
    // holds them off.
    if (state.learning) return;
    const layer = state.layers[layerIndex];
    const kernel = layer?.kernels[kernelIndex];
    if (!layer || !kernel) return;
    if (cell < 0 || cell >= KERNEL_CELLS) return;

    const weight = clamp(Math.round(value), WEIGHT_MIN, WEIGHT_MAX);
    if (kernel.weights[cell] === weight) return;

    const weights = [...kernel.weights];
    weights[cell] = weight;
    const updated: Kernel = { ...kernel, weights };
    updated.label = describeKernel(updated);

    const layers = state.layers.map((candidate, index) =>
      index !== layerIndex
        ? candidate
        : {
            ...candidate,
            kernels: candidate.kernels.map((k, i) =>
              i === kernelIndex ? updated : k,
            ),
          },
    );

    set({
      layers,
      whyCard: whyCardFor({
        kind: "weight-changed",
        kernel: updated,
        layerIndex,
      }),
      failure: null,
      phase: "designing",
    });
    void get().score_();
  },

  applyPreset: (layerIndex, kernelIndex, preset) => {
    const state = get();
    // Edits are NOT blocked while scoring: see scoreTicket. Only an explicit
    // end-to-end training run, which is a deliberate click with its own label,
    // holds them off.
    if (state.learning) return;
    const chosen = presetByLabel(preset);
    if (!chosen) return;

    const layers = state.layers.map((candidate, index) =>
      index !== layerIndex
        ? candidate
        : {
            ...candidate,
            kernels: candidate.kernels.map((kernel, i) =>
              i === kernelIndex
                ? { ...kernel, label: chosen.label, weights: [...chosen.weights] }
                : kernel,
            ),
          },
    );

    set({
      layers,
      whyCard: whyCardFor({ kind: "preset-applied", preset: chosen, layerIndex }),
      failure: null,
      phase: "designing",
    });
    void get().score_();
  },

  addKernel: (layerIndex, preset = "Vertical edge") => {
    const state = get();
    // Edits are NOT blocked while scoring: see scoreTicket. Only an explicit
    // end-to-end training run, which is a deliberate click with its own label,
    // holds them off.
    if (state.learning) return;
    const layer = state.layers[layerIndex];
    if (!layer || layer.kernels.length >= MAX_KERNELS_PER_LAYER) return;
    if (filtersUsed(state.layers) >= FILTER_BUDGET) {
      set({ whyCard: whyCardFor({ kind: "budget-spent" }) });
      return;
    }

    const layers = state.layers.map((candidate, index) =>
      index !== layerIndex
        ? candidate
        : { ...candidate, kernels: [...candidate.kernels, makeKernel(preset)] },
    );
    set({ layers, failure: null, phase: "designing" });
    void get().score_();
  },

  removeKernel: (layerIndex, kernelIndex) => {
    const state = get();
    // Edits are NOT blocked while scoring: see scoreTicket. Only an explicit
    // end-to-end training run, which is a deliberate click with its own label,
    // holds them off.
    if (state.learning) return;
    const layer = state.layers[layerIndex];
    if (!layer || layer.kernels.length <= 1) return;

    const layers = state.layers.map((candidate, index) =>
      index !== layerIndex
        ? candidate
        : {
            ...candidate,
            kernels: candidate.kernels.filter((_, i) => i !== kernelIndex),
          },
    );
    set({
      layers,
      windowKernel: 0,
      failure: null,
      phase: "designing",
    });
    void get().score_();
  },

  setPool: (layerIndex, pool) => {
    const state = get();
    // Edits are NOT blocked while scoring: see scoreTicket. Only an explicit
    // end-to-end training run, which is a deliberate click with its own label,
    // holds them off.
    if (state.learning) return;
    const layer = state.layers[layerIndex];
    if (!layer || layer.pool === pool) return;

    const layers = state.layers.map((candidate, index) =>
      index !== layerIndex ? candidate : { ...candidate, pool },
    );
    const previous = state.score.accuracy;
    set({
      layers,
      whyCard: whyCardFor({
        kind: "pool-changed",
        pool,
        layerIndex,
        layers,
        previousAccuracy: previous,
      }),
      failure: null,
      phase: "designing",
    });
    void get().score_();
  },

  addLayer: () => {
    const state = get();
    // Edits are NOT blocked while scoring: see scoreTicket. Only an explicit
    // end-to-end training run, which is a deliberate click with its own label,
    // holds them off.
    if (state.learning) return;
    if (state.layers.length >= MAX_LAYERS) return;
    if (filtersUsed(state.layers) >= FILTER_BUDGET) {
      set({ whyCard: whyCardFor({ kind: "budget-spent" }) });
      return;
    }

    // A pass-through plus an asymmetric filter: the arrangement of the layer
    // below, and the amount of it, side by side. Both are needed — see the
    // why-card, which explains why the pass-through is not padding.
    const kernels = [makeKernel("Pass through")];
    if (filtersUsed(state.layers) + 2 <= FILTER_BUDGET) {
      kernels.push(makeKernel("Horizontal edge"));
    }

    const layers = [...state.layers, makeLayer(kernels, "avg")];
    set({
      layers,
      whyCard: whyCardFor({ kind: "layer-added", layers }),
      failure: null,
      phase: "designing",
    });
    void get().score_();
  },

  removeLayer: (layerIndex) => {
    const state = get();
    // Edits are NOT blocked while scoring: see scoreTicket. Only an explicit
    // end-to-end training run, which is a deliberate click with its own label,
    // holds them off.
    if (state.learning) return;
    if (state.layers.length <= 1) return;

    const layers = state.layers.filter((_, index) => index !== layerIndex);
    set({
      layers,
      windowKernel: 0,
      learned: null,
      failure: null,
      phase: "designing",
    });
    void get().score_();
  },

  score_: async () => {
    const state = get();
    const ticket = ++scoreTicket;
    set({ scoring: true });

    try {
      const layers = get().layers;
      const score = await scoreStack(state.dataset, layers);
      const { health, duplicates } = inspectFilters(state.dataset, layers);
      const evaluation = evaluateKitchen({
        layers,
        score,
        health,
        duplicates,
        learned: get().learned,
      });

      // Fitted once, then kept: it is a property of the dataset, not of the run.
      let baseline = get().baseline;
      if (baseline === null) {
        baseline = await scoreRawPixels(state.dataset);
      }

      // A newer edit has already asked for a different answer. Drop this one
      // rather than writing a score that belongs to a stack no longer on screen.
      if (ticket !== scoreTicket) return;

      const served = evaluation.outcome === "served";

      /**
       * Keep the edit's own explanation unless the verdict actually changed.
       *
       * Scoring lands a second or two after every stepper click, and the first
       * version of this store let it stamp the verdict card over the top each
       * time — which meant the copy explaining what a zero-sum kernel does was
       * never on screen long enough to read, and the same three sentences about
       * the same unchanged verdict reappeared on every click. So the why-card
       * only yields to the score when the score has news: a different outcome, or
       * the first one. The per-kernel detail stays put while a player is fiddling
       * inside one verdict, which is exactly when they want it.
       */
      const previous = get().evaluation?.outcome;
      const isNews = previous === undefined || previous !== evaluation.outcome;
      set({
        score,
        baseline,
        evaluation,
        scoring: false,
        phase: served ? "served" : "designing",
        failure: evaluation.failure,
        whyCard: isNews
          ? whyCardFor({ kind: "scored", evaluation, layers, baseline })
          : get().whyCard,
      });

      if (served) {
        useProgression.getState().recordResult({
          slug: SLUG,
          score: evaluation.points,
          lane: get().lane,
          completed: true,
          codeLaneCleared: get().lane === "code",
        });
      }
    } catch (error) {
      if (ticket !== scoreTicket) return;
      set({
        scoring: false,
        whyCard: whyCardFor({
          kind: "error",
          message: error instanceof Error ? error.message : String(error),
        }),
      });
    }
  },

  letItLearn: async () => {
    const state = get();
    if (state.learning || state.scoring) return;
    if (filtersUsed(state.layers) === 0) return;
    set({ learning: true });

    try {
      const layers = get().layers;
      const learned = await learnStack(state.dataset, layers);
      const evaluation =
        get().evaluation === null
          ? null
          : evaluateKitchen({
              layers,
              score: get().score,
              health: get().evaluation!.health,
              duplicates: get().evaluation!.duplicates,
              learned,
            });

      set({
        learned,
        learning: false,
        evaluation: evaluation ?? get().evaluation,
        failure: evaluation?.failure ?? get().failure,
        whyCard: whyCardFor({
          kind: "learned",
          learned,
          layers,
          mine: get().score.accuracy,
        }),
      });
    } catch (error) {
      set({
        learning: false,
        whyCard: whyCardFor({
          kind: "error",
          message: error instanceof Error ? error.message : String(error),
        }),
      });
    }
  },

  reset: () => {
    set({
      layers: startingLayers(),
      previewIndex: 0,
      windowRow: 8,
      windowCol: 8,
      windowKernel: 0,
      score: emptyScore(),
      evaluation: null,
      learned: null,
      scoring: false,
      learning: false,
      phase: "designing",
      failure: null,
      whyCard: whyCardFor({ kind: "briefing" }),
    });
    void get().score_();
  },
}));

/**
 * What the code lane can do, and nothing more.
 *
 * Every verb writes the same store the sliders and steppers write, so a snippet
 * and a stepper click are the same operation (CLAUDE.md two-lane rule).
 */
export interface KitchenCodeApi {
  setWeights: (layerIndex: number, kernelIndex: number, weights: number[]) => void;
  applyPreset: (layerIndex: number, kernelIndex: number, preset: string) => void;
  addKernel: (layerIndex: number, preset?: string) => void;
  removeKernel: (layerIndex: number, kernelIndex: number) => void;
  setPool: (layerIndex: number, pool: string) => void;
  addLayer: () => void;
  removeLayer: (layerIndex: number) => void;
  reset: () => void;

  layers: () => Array<{
    pool: PoolType;
    kernels: Array<{ label: string; weights: number[] }>;
  }>;
  score: () => ScoreResult & { filters: number; channels: number };
  baseline: () => number | null;
  outcome: () => string;

  /**
   * Score a stack WITHOUT adopting it.
   *
   * The point of the code lane here: "no single layer can pass this" is a claim
   * about a search over kernel sets, and checking it by hand one stepper at a time
   * is not a thing anybody will do. This lets a loop settle it.
   */
  trial: (
    spec: Array<{ pool?: string; kernels: Array<number[] | string> }>,
  ) => Promise<ScoreResult & { filters: number; channels: number }>;
  /** Train a stack of the given shape end to end. Does not touch the player's. */
  learn: (
    spec: Array<{ pool?: string; kernels: number }>,
  ) => Promise<LearnedResult>;
  /** Per-filter mean activation and duplicate pairs for the current stack. */
  inspect: () => ReturnType<typeof inspectFilters>;
  /** One image's feature maps, as arrays. */
  maps: (imageIndex?: number) => Array<Array<{ label: string; size: number; values: number[] }>>;
  labels: () => string[];
}

const POOLS: readonly PoolType[] = ["none", "max", "avg"];

function asPool(value: string | undefined, fallback: PoolType = "avg"): PoolType {
  if (value === undefined) return fallback;
  const match = POOLS.find((pool) => pool === value);
  if (match === undefined) {
    throw new Error(`Unknown pool "${value}". Use one of: ${POOLS.join(", ")}.`);
  }
  return match;
}

function kernelFrom(entry: number[] | string): Kernel {
  if (typeof entry === "string") {
    const preset = presetByLabel(entry);
    if (!preset) {
      throw new Error(`Unknown preset "${entry}".`);
    }
    return makeKernel(preset);
  }
  if (!Array.isArray(entry) || entry.length !== KERNEL_CELLS) {
    throw new Error(`A kernel needs exactly ${KERNEL_CELLS} weights.`);
  }
  const kernel = makeKernel("Blank");
  kernel.weights = entry.map((weight) => {
    if (!Number.isFinite(weight)) throw new Error("Weights must be numbers.");
    return clamp(Math.round(weight), WEIGHT_MIN, WEIGHT_MAX);
  });
  kernel.label = describeKernel(kernel);
  return kernel;
}

export function createCodeApi(): KitchenCodeApi {
  const store = useKitchenStore;
  const summarise = (score: ScoreResult, layers: readonly Layer[]) => ({
    ...score,
    filters: filtersUsed(layers),
    channels: layers
      .filter((layer) => layer.kernels.length > 0)
      .reduce((total, layer) => total * layer.kernels.length, 1),
  });

  return {
    setWeights: (layerIndex, kernelIndex, weights) => {
      if (!Array.isArray(weights) || weights.length !== KERNEL_CELLS) {
        throw new Error(`setWeights needs exactly ${KERNEL_CELLS} numbers.`);
      }
      weights.forEach((weight, cell) => {
        store.getState().setWeight(layerIndex, kernelIndex, cell, weight);
      });
    },
    applyPreset: (layerIndex, kernelIndex, preset) =>
      store.getState().applyPreset(layerIndex, kernelIndex, preset),
    addKernel: (layerIndex, preset) =>
      store.getState().addKernel(layerIndex, preset),
    removeKernel: (layerIndex, kernelIndex) =>
      store.getState().removeKernel(layerIndex, kernelIndex),
    setPool: (layerIndex, pool) =>
      store.getState().setPool(layerIndex, asPool(pool)),
    addLayer: () => store.getState().addLayer(),
    removeLayer: (layerIndex) => store.getState().removeLayer(layerIndex),
    reset: () => store.getState().reset(),

    layers: () =>
      store.getState().layers.map((layer) => ({
        pool: layer.pool,
        kernels: layer.kernels.map((kernel) => ({
          label: kernel.label,
          weights: [...kernel.weights],
        })),
      })),
    score: () => summarise(store.getState().score, store.getState().layers),
    baseline: () => store.getState().baseline,
    outcome: () => store.getState().evaluation?.outcome ?? "unscored",

    trial: async (spec) => {
      if (!Array.isArray(spec) || spec.length === 0) {
        throw new Error("trial needs at least one layer.");
      }
      if (spec.length > MAX_LAYERS) {
        throw new Error(`At most ${MAX_LAYERS} layers.`);
      }
      const layers = spec.map((entry) => {
        if (!Array.isArray(entry.kernels) || entry.kernels.length === 0) {
          throw new Error("Every layer needs at least one kernel.");
        }
        if (entry.kernels.length > MAX_KERNELS_PER_LAYER) {
          throw new Error(`At most ${MAX_KERNELS_PER_LAYER} kernels per layer.`);
        }
        return makeLayer(entry.kernels.map(kernelFrom), asPool(entry.pool));
      });
      const score = await scoreStack(store.getState().dataset, layers);
      return summarise(score, layers);
    },

    learn: async (spec) => {
      if (!Array.isArray(spec) || spec.length === 0) {
        throw new Error("learn needs at least one layer.");
      }
      const layers = spec.map((entry) => {
        const count = Math.round(entry.kernels);
        if (!Number.isFinite(count) || count < 1 || count > MAX_KERNELS_PER_LAYER) {
          throw new Error(
            `Kernel count must be between 1 and ${MAX_KERNELS_PER_LAYER}.`,
          );
        }
        return makeLayer(
          Array.from({ length: count }, () => makeKernel("Blank")),
          asPool(entry.pool),
        );
      });
      return learnStack(store.getState().dataset, layers);
    },

    inspect: () =>
      inspectFilters(store.getState().dataset, store.getState().layers),

    maps: (imageIndex = 0) => {
      const state = store.getState();
      const sample =
        state.dataset.validation[
          clamp(Math.round(imageIndex), 0, state.dataset.validation.length - 1)
        ]!;
      // Plain arrays, never tensors: the code lane must not be handed anything
      // it can dispose or mutate out from under the store.
      return featureMapsFor(sample, state.layers).map((perLayer) =>
        perLayer.map((map) => ({
          label: map.label,
          size: map.size,
          values: [...map.values],
        })),
      );
    },

    labels: () => {
      const state = store.getState();
      return state.dataset.validation
        .slice(0, 8)
        .map((sample) => String(sample.label));
    },
  };
}
