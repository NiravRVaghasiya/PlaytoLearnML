"use client";

import { create } from "zustand";
import { useProgression } from "@/engine/progression";
import type { Lane, NamedFailure } from "@/engine/types";
import type { WhyCardContent } from "@/components";
import { clamp } from "@/lib/utils";
import {
  CLASS_COUNT,
  FILTER_BUDGET,
  FitCancelledError,
  IMAGE_SIZE,
  KERNEL_CELLS,
  KERNEL_PRESETS,
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
  shapeKey,
  type CancelToken,
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

/**
 * How long an edit waits before the stack is rescored.
 *
 * Every stepper click used to start a full fit of the head on the spot, and
 * `api.setWeights` clicked nine steppers in one call — so one line of code could
 * put nine fits on the GPU at once, eight of them already out of date. A short
 * pause lets a burst of edits share one fit. It is short enough that a single
 * click still feels immediate, and the caption says "refitting" from the moment
 * of the click, not from the moment the fit starts.
 */
export const SCORE_DEBOUNCE_MS = 200;

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
 *
 * Every edit takes an optional `source` lane. It is the lane the edit CAME from,
 * which is not the same thing as the tab on screen — scoring lands a second or
 * two after the edit, and a player can switch tabs in between. The code-lane star
 * is awarded on where the stack was built, so the code lane's api passes "code"
 * and every visual control leaves it at the default.
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
  /**
   * The last "Let it learn" result, and the shape it was learned for.
   *
   * The result is a property of the SHAPE — kernel counts and pooling — and it
   * used to be cleared only by removing a layer, so "Let it learn" on one layer
   * followed by "Add layer 2" reported the one-layer number as "this shape". Read
   * it through `currentLearned`, which returns null whenever the stack on screen
   * is not the shape that was learned. Weight and preset edits keep it, correctly.
   */
  learned: LearnedResult | null;
  learnedShape: string | null;

  scoring: boolean;
  learning: boolean;
  phase: Phase;
  failure: NamedFailure | null;
  whyCard: WhyCardContent | null;
  lane: Lane;
  /** Which lane made the last edit — see the note on `source` above. */
  editSource: Lane;

  setLane: (lane: Lane) => void;
  setPreview: (index: number) => void;
  moveWindow: (row: number, col: number) => void;
  setWindowKernel: (index: number) => void;

  setWeight: (
    layerIndex: number,
    kernelIndex: number,
    cell: number,
    value: number,
    source?: Lane,
  ) => void;
  /** All nine weights of one kernel, as ONE edit and one rescore. */
  setWeights: (
    layerIndex: number,
    kernelIndex: number,
    weights: readonly number[],
    source?: Lane,
  ) => void;
  applyPreset: (
    layerIndex: number,
    kernelIndex: number,
    preset: string,
    source?: Lane,
  ) => void;
  addKernel: (layerIndex: number, preset?: string, source?: Lane) => void;
  removeKernel: (layerIndex: number, kernelIndex: number, source?: Lane) => void;
  setPool: (layerIndex: number, pool: PoolType, source?: Lane) => void;
  addLayer: (source?: Lane) => void;
  removeLayer: (layerIndex: number, source?: Lane) => void;

  /**
   * Fit the fixed head on the current features and judge the kitchen, now.
   *
   * Edits do not call this directly: they schedule it (`SCORE_DEBOUNCE_MS`). A
   * direct call supersedes anything scheduled or in flight.
   */
  score_: () => Promise<void>;
  /** Train the same-shaped stack end to end, for comparison. */
  letItLearn: () => Promise<void>;
  reset: (source?: Lane) => void;
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
/** The learned result, if and only if it was learned for the stack on screen. */
export const currentLearned = (state: KitchenState): LearnedResult | null =>
  state.learned !== null && state.learnedShape === shapeKey(state.layers)
    ? state.learned
    : null;

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

/** Lanes arrive as click events too (`onClick={reset}`), so read them defensively. */
const laneOf = (source: unknown): Lane => (source === "code" ? "code" : "visual");

/**
 * Keep the previous failure object when the verdict has not changed.
 *
 * Every edit used to clear the failure and every rescore put it back, so the
 * shell's alert unmounted and remounted on each stepper click and a screen reader
 * re-read the same hundred-word paragraph four times for one weight nudged from
 * -2 to +2. The previous verdict now stays up while the rescore runs (the caption
 * says "refitting"), and when the new one is word-for-word the same, the same
 * object is handed back so React does not touch the DOM at all.
 */
function sameFailure(
  previous: NamedFailure | null,
  next: NamedFailure | null,
): NamedFailure | null {
  if (
    previous !== null &&
    next !== null &&
    previous.name === next.name &&
    previous.detail === next.detail
  ) {
    return previous;
  }
  return next;
}

/**
 * Supersedes in-flight scoring runs.
 *
 * Fitting the head takes a second or two, and edits must not wait for it — a
 * stepper that swallows your second click because a background fit is running
 * feels broken, and the first version of this store did exactly that by guarding
 * every edit on `scoring`. Instead every run takes a ticket, and a run that
 * finishes holding a stale ticket throws its result away. The player edits freely;
 * only the last request gets to write.
 *
 * An edit bumps the ticket the moment it happens, not when its debounced rescore
 * starts: otherwise a fit for the PREVIOUS stack that finished inside the
 * debounce window would still hold the current ticket and write its score onto a
 * stack that no longer exists. It also cancels that fit, so it stops spending the
 * GPU on an answer nobody will read.
 */
let scoreTicket = 0;
let activeScore: CancelToken | null = null;
let scoreTimer: ReturnType<typeof setTimeout> | null = null;
/** The newest scoring run, so "Let it learn" can wait for it. Never rejects. */
let scoreInFlight: Promise<void> = Promise.resolve();
/** Callers of the code-lane setters waiting for the rescore to land. */
let settledWaiters: Array<() => void> = [];

function supersedeScoring(): void {
  scoreTicket += 1;
  if (activeScore !== null) activeScore.cancelled = true;
  activeScore = null;
  if (scoreTimer !== null) {
    clearTimeout(scoreTimer);
    scoreTimer = null;
  }
}

function releaseSettledWaiters(): void {
  const waiting = settledWaiters;
  settledWaiters = [];
  for (const resolve of waiting) resolve();
}

/**
 * Invalidates in-flight "Let it learn" runs.
 *
 * The shell's Retry used to reset the kitchen while an end-to-end training run
 * was still going, and when that run finished it wrote its learned result, its
 * why-card and a re-judged verdict onto the fresh opening kitchen. A reset now
 * bumps the generation and cancels the fit; the old run checks the generation
 * when it resolves and writes nothing.
 */
let learnGeneration = 0;
let activeLearn: CancelToken | null = null;

/**
 * The raw-pixel baseline, fitted once per page and shared.
 *
 * It is a property of the seeded dataset, not of any stack, so every caller gets
 * the same promise — the first score no longer waits on it, and a second caller
 * (React Strict Mode mounts twice in development) no longer fits it again. It is
 * still MEASURED, never a constant: CLAUDE.md does not allow a hardcoded metric.
 */
let baselinePromise: Promise<number> | null = null;

function ensureBaseline(data: Dataset): Promise<number> {
  if (baselinePromise === null) {
    baselinePromise = scoreRawPixels(data).catch((error: unknown) => {
      // Let a later score try again rather than caching the failure.
      baselinePromise = null;
      throw error;
    });
  }
  return baselinePromise;
}

/**
 * The store, for the helpers below. They live at module level beside the tickets
 * and tokens they manage, and only ever run after the store exists.
 */
const kitchen = () => useKitchenStore.getState();
const setKitchen = (partial: Partial<KitchenState>) =>
  useKitchenStore.setState(partial);

/**
 * Every edit ends here: supersede the old answer at once, say so at once, and
 * fit the new stack once the burst of edits is over.
 */
function scheduleScore(): void {
  supersedeScoring();
  scoreTimer = setTimeout(() => {
    scoreTimer = null;
    void kitchen().score_();
  }, SCORE_DEBOUNCE_MS);
}

/**
 * The body of `score_`. Resolves true when it published a result for the stack
 * still on screen. Catches everything, so `scoreInFlight` never rejects.
 */
async function runScore(): Promise<boolean> {
  supersedeScoring();
  const ticket = scoreTicket;
  const cancel: CancelToken = { cancelled: false };
  activeScore = cancel;
  const state = kitchen();
  const layers = state.layers;
  const source = state.editSource;
  setKitchen({ scoring: true });

  try {
    const score = await scoreStack(state.dataset, layers, cancel);
    // A newer edit has already asked for a different answer. Drop this one
    // rather than writing a score that belongs to a stack no longer on screen.
    if (ticket !== scoreTicket) return false;

    const { health, duplicates } = inspectFilters(state.dataset, layers);
    const current = kitchen();
    const evaluation = evaluateKitchen({
      layers,
      score,
      health,
      duplicates,
      learned: current.learnedShape === shapeKey(layers) ? current.learned : null,
    });

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
    const previous = current.evaluation?.outcome;
    const isNews = previous === undefined || previous !== evaluation.outcome;
    setKitchen({
      score,
      evaluation,
      scoring: false,
      phase: served ? "served" : "designing",
      failure: sameFailure(current.failure, evaluation.failure),
      whyCard: isNews
        ? whyCardFor({
            kind: "scored",
            evaluation,
            layers,
            baseline: current.baseline,
          })
        : current.whyCard,
    });
    activeScore = null;

    if (served) {
      useProgression.getState().recordResult({
        slug: SLUG,
        score: evaluation.points,
        lane: source,
        completed: true,
        codeLaneCleared: source === "code",
      });
    }
    // Last, so a snippet awaiting a setter sees the result AND the credit.
    releaseSettledWaiters();
    return true;
  } catch (error) {
    if (ticket !== scoreTicket || error instanceof FitCancelledError) return false;
    activeScore = null;
    setKitchen({
      scoring: false,
      whyCard: whyCardFor({
        kind: "error",
        message: error instanceof Error ? error.message : String(error),
      }),
    });
    releaseSettledWaiters();
    return false;
  }
}

/** The score is already on screen by now; the baseline follows when it is ready. */
async function fillBaseline(): Promise<void> {
  if (kitchen().baseline !== null) return;
  try {
    const baseline = await ensureBaseline(kitchen().dataset);
    if (kitchen().baseline === null) setKitchen({ baseline });
  } catch {
    // Leaves the tile on "fitting the baseline…"; the next score retries.
  }
}

/** Run any rescore that is still waiting out its debounce, and wait for the newest. */
async function flushScore(): Promise<void> {
  if (scoreTimer !== null) {
    clearTimeout(scoreTimer);
    scoreTimer = null;
    void kitchen().score_();
  }
  await scoreInFlight;
}

/**
 * Disown any "Let it learn" still running: a reset, or a new run, makes it an
 * answer to a question nobody is asking any more. Edits cannot get here — they
 * are held off while learning, in the store and by name in the api.
 */
function invalidateLearn(): void {
  learnGeneration += 1;
  if (activeLearn !== null) activeLearn.cancelled = true;
  activeLearn = null;
}

/** Swap one kernel for another and rescore: the shared tail of every weight edit. */
function replaceKernel(
  layerIndex: number,
  kernelIndex: number,
  updated: Kernel,
  whyCard: WhyCardContent,
  source: unknown,
): void {
  const layers = kitchen().layers.map((candidate, index) =>
    index !== layerIndex
      ? candidate
      : {
          ...candidate,
          kernels: candidate.kernels.map((k, i) =>
            i === kernelIndex ? updated : k,
          ),
        },
  );
  // `failure` is deliberately left alone — see `sameFailure`.
  setKitchen({
    layers,
    whyCard,
    phase: "designing",
    scoring: true,
    editSource: laneOf(source),
  });
  scheduleScore();
}

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
  learnedShape: null,

  scoring: false,
  learning: false,
  phase: "designing",
  failure: null,
  whyCard: whyCardFor({ kind: "briefing" }),
  lane: "visual" as Lane,
  editSource: "visual" as Lane,

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

  setWeight: (layerIndex, kernelIndex, cell, value, source) => {
    const state = get();
    // Edits are NOT blocked while scoring: see scoreTicket. Only an explicit
    // end-to-end training run, which is a deliberate click with its own label,
    // holds them off.
    if (state.learning) return;
    const kernel = state.layers[layerIndex]?.kernels[kernelIndex];
    if (!kernel) return;
    if (cell < 0 || cell >= KERNEL_CELLS) return;

    const weight = clamp(Math.round(value), WEIGHT_MIN, WEIGHT_MAX);
    if (kernel.weights[cell] === weight) return;

    const weights = [...kernel.weights];
    weights[cell] = weight;
    const updated: Kernel = { ...kernel, weights };
    updated.label = describeKernel(updated);

    replaceKernel(
      layerIndex,
      kernelIndex,
      updated,
      whyCardFor({ kind: "weight-changed", kernel: updated, layerIndex }),
      source,
    );
  },

  setWeights: (layerIndex, kernelIndex, values, source) => {
    const state = get();
    if (state.learning) return;
    const kernel = state.layers[layerIndex]?.kernels[kernelIndex];
    if (!kernel || values.length !== KERNEL_CELLS) return;

    const weights = values.map((value) =>
      clamp(Math.round(value), WEIGHT_MIN, WEIGHT_MAX),
    );
    if (weights.every((weight, cell) => weight === kernel.weights[cell])) return;

    const updated: Kernel = { ...kernel, weights };
    updated.label = describeKernel(updated);

    replaceKernel(
      layerIndex,
      kernelIndex,
      updated,
      whyCardFor({ kind: "weight-changed", kernel: updated, layerIndex }),
      source,
    );
  },

  applyPreset: (layerIndex, kernelIndex, preset, source) => {
    const state = get();
    // Edits are NOT blocked while scoring: see scoreTicket. Only an explicit
    // end-to-end training run, which is a deliberate click with its own label,
    // holds them off.
    if (state.learning) return;
    const chosen = presetByLabel(preset);
    if (!chosen) return;
    const kernel = state.layers[layerIndex]?.kernels[kernelIndex];
    if (!kernel) return;

    replaceKernel(
      layerIndex,
      kernelIndex,
      { ...kernel, label: chosen.label, weights: [...chosen.weights] },
      whyCardFor({ kind: "preset-applied", preset: chosen, layerIndex }),
      source,
    );
  },

  addKernel: (layerIndex, preset = "Vertical edge", source) => {
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
    set({ layers, phase: "designing", scoring: true, editSource: laneOf(source) });
    scheduleScore();
  },

  removeKernel: (layerIndex, kernelIndex, source) => {
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
      phase: "designing",
      scoring: true,
      editSource: laneOf(source),
    });
    scheduleScore();
  },

  setPool: (layerIndex, pool, source) => {
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
      phase: "designing",
      scoring: true,
      editSource: laneOf(source),
    });
    scheduleScore();
  },

  addLayer: (source) => {
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
      phase: "designing",
      scoring: true,
      editSource: laneOf(source),
    });
    scheduleScore();
  },

  removeLayer: (layerIndex, source) => {
    const state = get();
    // Edits are NOT blocked while scoring: see scoreTicket. Only an explicit
    // end-to-end training run, which is a deliberate click with its own label,
    // holds them off.
    if (state.learning) return;
    if (state.layers.length <= 1) return;

    const layers = state.layers.filter((_, index) => index !== layerIndex);
    // No need to clear `learned`: `currentLearned` compares shapes, and going
    // back to a shape that WAS learned correctly brings its result back.
    set({
      layers,
      windowKernel: 0,
      phase: "designing",
      scoring: true,
      editSource: laneOf(source),
    });
    scheduleScore();
  },

  score_: () => {
    const run = runScore();
    // "Let it learn" waits on the score, not on the baseline behind it.
    scoreInFlight = run.then(() => undefined);
    return run.then((published) => (published ? fillBaseline() : undefined));
  },

  letItLearn: async () => {
    const state = get();
    if (state.learning) return;
    if (filtersUsed(state.layers) === 0) return;

    invalidateLearn();
    const generation = learnGeneration;
    const cancel: CancelToken = { cancelled: false };
    activeLearn = cancel;
    // Edits hold off from here, so the stack cannot change under the run.
    set({ learning: true });

    try {
      // A rescore can still be waiting out its debounce from the click just
      // before this one. It used to make this button silently do nothing; now
      // the rescore runs first, so the comparison below is against the score of
      // the stack actually being learned rather than the one before it.
      await flushScore();
      if (generation !== learnGeneration) return;

      const layers = get().layers;
      const learned = await learnStack(state.dataset, layers, cancel);
      if (generation !== learnGeneration) return;

      const current = get();
      const evaluation =
        current.evaluation === null
          ? null
          : evaluateKitchen({
              layers,
              score: current.score,
              health: current.evaluation.health,
              duplicates: current.evaluation.duplicates,
              learned,
            });

      set({
        learned,
        learnedShape: shapeKey(layers),
        learning: false,
        evaluation: evaluation ?? current.evaluation,
        failure: sameFailure(
          current.failure,
          evaluation?.failure ?? current.failure,
        ),
        whyCard: whyCardFor({
          kind: "learned",
          learned,
          layers,
          mine: current.score.accuracy,
        }),
      });
    } catch (error) {
      if (generation !== learnGeneration || error instanceof FitCancelledError) {
        return;
      }
      set({
        learning: false,
        whyCard: whyCardFor({
          kind: "error",
          message: error instanceof Error ? error.message : String(error),
        }),
      });
    } finally {
      if (activeLearn === cancel) activeLearn = null;
    }
  },

  reset: (source) => {
    // Stop and disown any "Let it learn" still running, so its result cannot
    // land on the fresh kitchen (the shell's Retry is live during training).
    invalidateLearn();
    set({
      layers: startingLayers(),
      previewIndex: 0,
      windowRow: 8,
      windowCol: 8,
      windowKernel: 0,
      score: emptyScore(),
      evaluation: null,
      learned: null,
      learnedShape: null,
      scoring: true,
      learning: false,
      phase: "designing",
      failure: null,
      whyCard: whyCardFor({ kind: "briefing" }),
      editSource: laneOf(source),
    });
    void get().score_();
  },
}));

/**
 * What the code lane can do, and nothing more.
 *
 * Every verb writes the same store the sliders and steppers write, so a snippet
 * and a stepper click are the same operation (CLAUDE.md two-lane rule).
 *
 * The setters check their arguments and throw a named error for anything the
 * steppers could not have produced — an unknown preset, a layer or slot that
 * does not exist, a full layer, a spent budget. The store itself quietly ignores
 * those (a button cannot ask for slot 7), and in a snippet a quiet no-op is a
 * typo that reports success: `addKernel(0, "Sobel")` used to add a Blank filter
 * and then diagnose it as dead.
 *
 * They also return a promise that resolves when the kitchen has been rescored,
 * so `await api.addLayer(); log(api.score().accuracy)` reads the new score rather
 * than the one from before the edit. Ignoring the promise is fine too.
 */
export interface KitchenCodeApi {
  setWeights: (layerIndex: number, kernelIndex: number, weights: number[]) => Promise<void>;
  applyPreset: (layerIndex: number, kernelIndex: number, preset: string) => Promise<void>;
  addKernel: (layerIndex: number, preset?: string) => Promise<void>;
  removeKernel: (layerIndex: number, kernelIndex: number) => Promise<void>;
  setPool: (layerIndex: number, pool: string) => Promise<void>;
  addLayer: () => Promise<void>;
  removeLayer: (layerIndex: number) => Promise<void>;
  reset: () => Promise<void>;

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

function presetNamed(label: unknown) {
  const preset = typeof label === "string" ? presetByLabel(label) : undefined;
  if (!preset) {
    throw new Error(
      `Unknown preset "${String(label)}". Try one of: ${KERNEL_PRESETS.map(
        (candidate) => candidate.label,
      ).join(", ")}.`,
    );
  }
  return preset;
}

function checkedWeights(entry: unknown, what: string): number[] {
  if (!Array.isArray(entry) || entry.length !== KERNEL_CELLS) {
    throw new Error(`${what} needs exactly ${KERNEL_CELLS} weights.`);
  }
  return entry.map((weight: unknown) => {
    if (typeof weight !== "number" || !Number.isFinite(weight)) {
      throw new Error("Weights must be numbers.");
    }
    return clamp(Math.round(weight), WEIGHT_MIN, WEIGHT_MAX);
  });
}

function kernelFrom(entry: number[] | string): Kernel {
  if (typeof entry === "string") return makeKernel(presetNamed(entry));
  const kernel = makeKernel("Blank");
  kernel.weights = checkedWeights(entry, "A kernel");
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

  /** Resolves once no rescore is waiting or running. Never rejects. */
  const settled = (): Promise<void> => {
    if (!store.getState().scoring && scoreTimer === null) return Promise.resolve();
    return new Promise<void>((resolve) => {
      settledWaiters.push(resolve);
    });
  };

  const editable = () => {
    const state = store.getState();
    if (state.learning) {
      throw new Error(
        'The kitchen is busy with "Let it learn". Wait for it to finish before changing the stack.',
      );
    }
    return state;
  };

  const layerAt = (layerIndex: unknown) => {
    const { layers } = editable();
    if (
      typeof layerIndex !== "number" ||
      !Number.isInteger(layerIndex) ||
      layerIndex < 0 ||
      layerIndex >= layers.length
    ) {
      throw new Error(
        `There is no layer ${String(layerIndex)}. The kitchen has ${layers.length} layer${
          layers.length === 1 ? "" : "s"
        }, numbered from 0.`,
      );
    }
    return layers[layerIndex]!;
  };

  const slotAt = (layerIndex: unknown, kernelIndex: unknown) => {
    const layer = layerAt(layerIndex);
    if (
      typeof kernelIndex !== "number" ||
      !Number.isInteger(kernelIndex) ||
      kernelIndex < 0 ||
      kernelIndex >= layer.kernels.length
    ) {
      throw new Error(
        `Layer ${String(layerIndex)} has no filter ${String(kernelIndex)}. It has ${
          layer.kernels.length
        }, numbered from 0.`,
      );
    }
  };

  const roomForAFilter = () => {
    if (filtersUsed(store.getState().layers) >= FILTER_BUDGET) {
      throw new Error(`All ${FILTER_BUDGET} filters are spent. Remove one first.`);
    }
  };

  return {
    setWeights: (layerIndex, kernelIndex, weights) => {
      slotAt(layerIndex, kernelIndex);
      const values = checkedWeights(weights, "setWeights");
      // One kernel swap and one rescore, not nine stepper clicks and nine fits.
      store.getState().setWeights(layerIndex, kernelIndex, values, "code");
      return settled();
    },
    applyPreset: (layerIndex, kernelIndex, preset) => {
      slotAt(layerIndex, kernelIndex);
      const chosen = presetNamed(preset);
      store.getState().applyPreset(layerIndex, kernelIndex, chosen.label, "code");
      return settled();
    },
    addKernel: (layerIndex, preset) => {
      const layer = layerAt(layerIndex);
      const chosen = preset === undefined ? undefined : presetNamed(preset);
      if (layer.kernels.length >= MAX_KERNELS_PER_LAYER) {
        throw new Error(`Layer ${layerIndex} is full: ${MAX_KERNELS_PER_LAYER} filters at most.`);
      }
      roomForAFilter();
      store.getState().addKernel(layerIndex, chosen?.label, "code");
      return settled();
    },
    removeKernel: (layerIndex, kernelIndex) => {
      slotAt(layerIndex, kernelIndex);
      if (layerAt(layerIndex).kernels.length <= 1) {
        throw new Error(
          `Layer ${layerIndex} needs at least one filter. Remove the layer instead.`,
        );
      }
      store.getState().removeKernel(layerIndex, kernelIndex, "code");
      return settled();
    },
    setPool: (layerIndex, pool) => {
      layerAt(layerIndex);
      store.getState().setPool(layerIndex, asPool(pool), "code");
      return settled();
    },
    addLayer: () => {
      const { layers } = editable();
      if (layers.length >= MAX_LAYERS) {
        throw new Error(`At most ${MAX_LAYERS} layers.`);
      }
      roomForAFilter();
      store.getState().addLayer("code");
      return settled();
    },
    removeLayer: (layerIndex) => {
      layerAt(layerIndex);
      if (store.getState().layers.length <= 1) {
        throw new Error("The kitchen needs at least one layer.");
      }
      store.getState().removeLayer(layerIndex, "code");
      return settled();
    },
    reset: () => {
      store.getState().reset("code");
      return settled();
    },

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
        if (!Array.isArray(entry?.kernels) || entry.kernels.length === 0) {
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
      if (spec.length > MAX_LAYERS) {
        throw new Error(`At most ${MAX_LAYERS} layers.`);
      }
      const layers = spec.map((entry) => {
        const count = typeof entry?.kernels === "number" ? Math.round(entry.kernels) : Number.NaN;
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
      if (typeof imageIndex !== "number" || !Number.isFinite(imageIndex)) {
        throw new Error("maps takes an image index, e.g. api.maps(0).");
      }
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
