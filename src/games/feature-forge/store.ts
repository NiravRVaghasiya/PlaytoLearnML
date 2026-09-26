"use client";

import { create } from "zustand";
import { useProgression } from "@/engine/progression";
import type { Lane, NamedFailure } from "@/engine/types";
import type { WhyCardContent } from "@/components";
import {
  COLUMNS,
  TRANSFORMS,
  baselineFeatures,
  buildMatrices,
  describeFeature,
  evaluateForge,
  featureProblem,
  fitAndScore,
  generateDataset,
  isValidFeature,
  legendariesIn,
  transformById,
  usesLeakyColumn,
  type Dataset,
  type Evaluation,
  type Feature,
  type FitResult,
  type Transform,
} from "./ml";
import { whyCardFor } from "./why-cards";

export const SLUG = "feature-forge";

/** Same seed the tests measure the lifts against. */
export const DATA_SEED = 5309;

export type Phase = "forging" | "forged" | "ruined";

/**
 * Feature Forge state.
 *
 * The spec's data model is:
 *
 *   Column    { name, dtype, values[] }
 *   Feature   { id, sourceCols[], transform, importance }
 *   GameState { baselineScore, currentScore, features[] }
 *
 * `Column.values` is not stored per column — the table is row-major in `dataset`,
 * and keeping a column-major copy would be the same numbers twice. `COLUMNS`
 * carries the metadata, which never changes.
 *
 * `Feature.importance` is not stored on the feature either. A feature can widen
 * into several matrix columns (a bin becomes five), so importance belongs to matrix
 * columns, not to features. It lives in `importances` alongside `columnNames`,
 * written only by a retrain.
 *
 * ── Why this game does not use `useModel` ───────────────────────────────────
 * The engine's `useModel` drives ONE model through a long fit with epoch
 * callbacks. This game fits a short-lived model per forge and cares only about the
 * final score, so `fitAndScore` builds, fits, scores and disposes in one call —
 * including the optimizer, which `model.dispose()` alone would leave behind. Using
 * `useModel` here would mean holding a model across retrains that never share
 * weights.
 */
export interface ForgeState {
  dataset: Dataset;

  /** Features the player has forged, on top of the baseline. */
  features: Feature[];
  /** Columns currently dropped in the forge slot. */
  slot: string[];
  /** Transform selected in the picker. */
  transform: Transform;

  baselineScore: number;
  currentScore: number;
  /** |weight| per matrix column from the last retrain. */
  importances: number[];
  columnNames: string[];
  /** Train accuracy, to show when the model is memorising rather than learning. */
  trainScore: number;

  /** True while a retrain is in flight. */
  training: boolean;
  /**
   * Why the last fit failed, if it did — a lost WebGL context, say. Cleared by
   * the next fit that succeeds. The score on screen is then still the last one
   * that was real, for the feature list it was computed from.
   */
  fitError: string | null;
  /** True once the player has committed the forge for scoring. */
  submitted: boolean;
  /**
   * The lanes that have already recorded the current win with the progression
   * service. A win is recorded once per lane, not once per forge: the code-lane
   * star has to stay earnable with api.submit() after the same forge was scored
   * from the rail.
   */
  recordedLanes: Lane[];
  ready: boolean;

  phase: Phase;
  evaluation: Evaluation | null;
  failure: NamedFailure | null;
  whyCard: WhyCardContent | null;
  lane: Lane;

  setLane: (lane: Lane) => void;
  /** Establish the baseline. Must run before anything can be scored against it. */
  initialise: () => Promise<void>;
  toggleColumn: (name: string) => void;
  clearSlot: () => void;
  setTransform: (transform: Transform) => void;
  /** Commit the slot as a feature and retrain. */
  forge: () => Promise<void>;
  /**
   * Add a feature directly — the code lane's entry point. Resolves to what
   * happened, so a caller can tell a forge that retrained from one that was
   * refused, overtaken by a reset, or failed to fit.
   */
  addFeature: (transform: Transform, sourceCols: string[]) => Promise<ForgeOutcome>;
  removeFeature: (id: string) => Promise<ForgeOutcome>;
  clearForge: () => Promise<ForgeOutcome>;
  /** Retrain without changing the features. */
  retrain: () => Promise<ForgeOutcome>;
  /**
   * Submit for scoring. `source` is the lane the submission came from, and is
   * what earns (or does not earn) the code-lane star — not which tab is open.
   */
  submit: (source?: Lane) => Evaluation;
  reset: () => Promise<void>;
}

/**
 * What a request to change the forge came to.
 *
 *   applied  — the model retrained and the score describes the new features
 *   invalid  — the feature was refused before any fit (bad pairing, duplicate)
 *   busy     — another retrain was already in flight, so nothing changed
 *   stale    — the forge was reset while this fit ran, so its result was dropped
 *   failed   — the fit threw; the features and score are as they were before
 */
export type ForgeOutcome = "applied" | "invalid" | "busy" | "stale" | "failed";

/**
 * A generation counter for fits.
 *
 * Every fit captures the ticket when it starts and only writes its result if the
 * ticket is still current when it finishes. `reset` advances it, so a retrain
 * that was in flight when the player pressed Retry cannot land in the fresh run —
 * which it used to, leaving a Leakage failure on an empty forge, or a score and
 * importances computed from a matrix that was no longer on screen. The same
 * check is handed to `fitAndScore` as `shouldStop`, so the orphaned fit also
 * stops early instead of competing with the new one.
 */
let fitTicket = 0;

// ── selectors: primitives and stable references only ─────────────────────

export function lift(state: ForgeState): number {
  return state.ready ? state.currentScore - state.baselineScore : Number.NaN;
}

export function featureCount(state: ForgeState): number {
  return state.features.length;
}

export function hasLeak(state: ForgeState): boolean {
  return state.features.some(usesLeakyColumn);
}

/** Full feature list handed to the model: baseline plus whatever was forged. */
export function allFeatures(features: Feature[]): Feature[] {
  return [...baselineFeatures(), ...features];
}

function freshState() {
  return {
    dataset: generateDataset(DATA_SEED),
    features: [] as Feature[],
    slot: [] as string[],
    transform: "bin" as Transform,
    baselineScore: 0,
    currentScore: 0,
    importances: [] as number[],
    columnNames: [] as string[],
    trainScore: 0,
    training: false,
    fitError: null,
    submitted: false,
    recordedLanes: [] as Lane[],
    ready: false,
    phase: "forging" as Phase,
    evaluation: null,
    failure: null,
  };
}

export const useForgeStore = create<ForgeState>((set, get) => ({
  ...freshState(),
  whyCard: whyCardFor({ kind: "briefing" }),
  lane: "visual" as Lane,

  setLane: (lane) => set({ lane }),

  initialise: async () => {
    const state = get();
    if (state.ready || state.training) return;
    const ticket = fitTicket;
    const current = () => ticket === fitTicket;
    set({ training: true });

    const matrices = buildMatrices(state.dataset, baselineFeatures());
    if (matrices === null) {
      set({ training: false });
      return;
    }

    let result: FitResult;
    try {
      result = await fitAndScore(matrices, undefined, {
        shouldStop: () => !current(),
      });
    } catch (cause) {
      if (!current()) return;
      const message = messageOf(cause);
      set({
        training: false,
        fitError: message,
        whyCard: whyCardFor({ kind: "fit-failed", stage: "baseline", message }),
      });
      return;
    }
    if (!current()) return;

    set({
      baselineScore: result.validationAccuracy,
      currentScore: result.validationAccuracy,
      trainScore: result.trainAccuracy,
      importances: result.importances,
      columnNames: result.columnNames,
      training: false,
      fitError: null,
      ready: true,
      // The baseline is the start of a run, so nothing judged before it survives.
      features: [],
      submitted: false,
      recordedLanes: [],
      evaluation: null,
      failure: null,
      phase: "forging",
      whyCard: whyCardFor({
        kind: "baseline-ready",
        baselineScore: result.validationAccuracy,
        trainScore: result.trainAccuracy,
      }),
    });
  },

  toggleColumn: (name) => {
    const state = get();
    if (state.training) return;
    const spec = transformById(state.transform);

    const slot = state.slot.includes(name)
      ? state.slot.filter((entry) => entry !== name)
      : // Keep the newest selection when the slot is full, so clicking a third
        // column replaces rather than silently doing nothing.
        [...state.slot, name].slice(-spec.arity);

    set({ slot });
  },

  clearSlot: () => set({ slot: [] }),

  setTransform: (transform) => {
    const state = get();
    if (state.training) return;
    const spec = transformById(transform);
    set({
      transform,
      // Drop columns the new transform cannot accept, rather than leaving an
      // invalid pairing sitting in the slot.
      slot: state.slot
        .filter((name) => {
          const column = COLUMNS.find((entry) => entry.name === name);
          return column !== undefined && spec.accepts.includes(column.dtype);
        })
        .slice(-spec.arity),
      whyCard: whyCardFor({ kind: "transform-picked", transform }),
    });
  },

  forge: async () => {
    const state = get();
    const feature: Feature = {
      id: `${state.transform}:${state.slot.join("+")}`,
      transform: state.transform,
      sourceCols: [...state.slot],
    };
    if (!isValidFeature(feature)) return;
    if (state.features.some((entry) => entry.id === feature.id)) return;

    const outcome = await applyFeatures(set, get, [...state.features, feature], {
      kind: "forged-feature",
      feature,
    });
    // Only an applied forge empties the slot; a refused or overtaken one leaves
    // whatever the player has selected since alone.
    if (outcome === "applied") set({ slot: [] });
  },

  addFeature: async (transform, sourceCols) => {
    const state = get();
    const feature: Feature = {
      id: `${transform}:${sourceCols.join("+")}`,
      transform,
      sourceCols: [...sourceCols],
    };
    if (featureProblem(transform, sourceCols, state.features) !== null) {
      return "invalid";
    }

    return applyFeatures(set, get, [...state.features, feature], {
      kind: "forged-feature",
      feature,
    });
  },

  removeFeature: async (id) => {
    const state = get();
    const feature = state.features.find((entry) => entry.id === id);
    if (!feature) return "invalid";
    return applyFeatures(
      set,
      get,
      state.features.filter((entry) => entry.id !== id),
      { kind: "removed-feature", feature },
    );
  },

  clearForge: async () => applyFeatures(set, get, [], { kind: "cleared" }),

  retrain: async () => {
    const state = get();
    return applyFeatures(set, get, state.features, { kind: "retrained" });
  },

  submit: (source = "visual") => {
    const state = get();

    // Nothing to judge until the score describes the features on screen: while
    // a retrain is in flight, currentScore is still the previous matrix's.
    if (!state.ready || state.training) {
      return evaluateForge({
        features: state.features,
        baselineScore: state.baselineScore,
        currentScore: state.currentScore,
        submitted: false,
      });
    }

    // Idempotent once won, per lane: submitting the same winning forge again
    // from the same lane (running a snippet twice, say) must not record the
    // result twice. The other lane's first submission still counts — winning
    // on the rail and then calling api.submit() is exactly what the code-lane
    // star asks for.
    const alreadyWon = state.phase === "forged" && state.submitted;
    if (alreadyWon && state.evaluation && state.recordedLanes.includes(source)) {
      return state.evaluation;
    }

    const evaluation = evaluateForge({
      features: state.features,
      baselineScore: state.baselineScore,
      currentScore: state.currentScore,
      submitted: true,
    });

    const won = evaluation.outcome === "forged";
    set({
      submitted: true,
      recordedLanes: won
        ? [...(alreadyWon ? state.recordedLanes : []), source]
        : [],
      evaluation,
      failure: evaluation.failure,
      phase: won ? "forged" : evaluation.outcome === "leakage" ? "ruined" : "forging",
      whyCard: whyCardFor({ kind: "submitted", evaluation }),
    });

    if (won) {
      useProgression.getState().recordResult({
        slug: SLUG,
        score: evaluation.score,
        lane: source,
        completed: true,
        codeLaneCleared: source === "code",
      });
    }

    return evaluation;
  },

  reset: async () => {
    fitTicket += 1;
    set({ ...freshState(), whyCard: whyCardFor({ kind: "briefing" }) });
    await get().initialise();
  },
}));

function messageOf(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

/**
 * Retrain on a new feature list and fold the result back in.
 *
 * Every path that changes the features goes through here, so the score, the
 * importances and the column labels can never describe a different matrix from the
 * one on screen.
 */
async function applyFeatures(
  set: (partial: Partial<ForgeState>) => void,
  get: () => ForgeState,
  features: Feature[],
  event:
    | { kind: "forged-feature"; feature: Feature }
    | { kind: "removed-feature"; feature: Feature }
    | { kind: "cleared" }
    | { kind: "retrained" },
): Promise<ForgeOutcome> {
  const state = get();
  // Not before the baseline exists either: a lift is meaningless without it.
  if (state.training || !state.ready) return "busy";

  const ticket = fitTicket;
  const current = () => ticket === fitTicket;
  const previousFeatures = state.features;

  set({ training: true, features, submitted: false, failure: null });

  const combined = allFeatures(features);
  const matrices = buildMatrices(state.dataset, combined);
  if (matrices === null) {
    set({ training: false, features: previousFeatures });
    return "invalid";
  }

  const previousScore = state.currentScore;
  let result: FitResult;
  try {
    result = await fitAndScore(matrices, undefined, {
      shouldStop: () => !current(),
    });
  } catch (cause) {
    if (!current()) return "stale";
    // Put the feature list back, so the score, importances and labels on screen
    // still describe one matrix: the last one that actually trained. Its
    // verdict comes back with it — a leaky forge whose next fit failed is still
    // a leaky forge, and the named failure must not quietly disappear.
    const message = messageOf(cause);
    set({
      training: false,
      features: previousFeatures,
      submitted: state.submitted,
      failure: state.failure,
      fitError: message,
      whyCard: whyCardFor({ kind: "fit-failed", stage: "retrain", message }),
    });
    return "failed";
  }
  if (!current()) return "stale";

  const leak = features.some(usesLeakyColumn);

  // Leakage is named the moment it enters the forge, not only at submission. A
  // player should not get to admire an impossible score for several turns first.
  const evaluation = evaluateForge({
    features,
    baselineScore: state.baselineScore,
    currentScore: result.validationAccuracy,
    submitted: false,
  });

  set({
    currentScore: result.validationAccuracy,
    trainScore: result.trainAccuracy,
    importances: result.importances,
    columnNames: result.columnNames,
    training: false,
    fitError: null,
    // A new matrix is a new forge: no lane has recorded a win for it yet.
    recordedLanes: [],
    evaluation: leak ? evaluation : null,
    failure: leak ? evaluation.failure : null,
    phase: leak ? "ruined" : "forging",
    whyCard: whyCardFor({
      ...event,
      baselineScore: state.baselineScore,
      currentScore: result.validationAccuracy,
      trainScore: result.trainAccuracy,
      previousScore,
      features,
      legendary: legendariesIn(features),
      leaked: leak,
    }),
  });
  return "applied";
}

/** Feature ids, for the code lane to list. */
export function featureSummaries(features: Feature[]): Array<{
  id: string;
  label: string;
  transform: Transform;
  sourceCols: string[];
  leaky: boolean;
}> {
  return features.map((feature) => ({
    id: feature.id,
    label: describeFeature(feature),
    transform: feature.transform,
    sourceCols: [...feature.sourceCols],
    leaky: usesLeakyColumn(feature),
  }));
}

export { COLUMNS, TRANSFORMS };
