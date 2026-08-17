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
  fitAndScore,
  generateDataset,
  isValidFeature,
  legendariesIn,
  transformById,
  usesLeakyColumn,
  type Dataset,
  type Evaluation,
  type Feature,
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
  /** True once the player has committed the forge for scoring. */
  submitted: boolean;
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
  /** Add a feature directly — the code lane's entry point. */
  addFeature: (transform: Transform, sourceCols: string[]) => Promise<boolean>;
  removeFeature: (id: string) => Promise<void>;
  clearForge: () => Promise<void>;
  /** Retrain without changing the features. */
  retrain: () => Promise<void>;
  /** Submit for scoring. */
  submit: () => Evaluation;
  reset: () => Promise<void>;
}

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
    submitted: false,
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
    set({ training: true });

    const matrices = buildMatrices(state.dataset, baselineFeatures());
    if (matrices === null) {
      set({ training: false });
      return;
    }
    const result = await fitAndScore(matrices);

    set({
      baselineScore: result.validationAccuracy,
      currentScore: result.validationAccuracy,
      trainScore: result.trainAccuracy,
      importances: result.importances,
      columnNames: result.columnNames,
      training: false,
      ready: true,
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

    await applyFeatures(set, get, [...state.features, feature], {
      kind: "forged-feature",
      feature,
    });
    set({ slot: [] });
  },

  addFeature: async (transform, sourceCols) => {
    const state = get();
    const feature: Feature = {
      id: `${transform}:${sourceCols.join("+")}`,
      transform,
      sourceCols: [...sourceCols],
    };
    if (!isValidFeature(feature)) return false;
    if (state.features.some((entry) => entry.id === feature.id)) return false;

    await applyFeatures(set, get, [...state.features, feature], {
      kind: "forged-feature",
      feature,
    });
    return true;
  },

  removeFeature: async (id) => {
    const state = get();
    const feature = state.features.find((entry) => entry.id === id);
    if (!feature) return;
    await applyFeatures(
      set,
      get,
      state.features.filter((entry) => entry.id !== id),
      { kind: "removed-feature", feature },
    );
  },

  clearForge: async () => {
    await applyFeatures(set, get, [], { kind: "cleared" });
  },

  retrain: async () => {
    const state = get();
    await applyFeatures(set, get, state.features, { kind: "retrained" });
  },

  submit: () => {
    const state = get();
    const evaluation = evaluateForge({
      features: state.features,
      baselineScore: state.baselineScore,
      currentScore: state.currentScore,
      submitted: true,
    });

    const won = evaluation.outcome === "forged";
    set({
      submitted: true,
      evaluation,
      failure: evaluation.failure,
      phase: won ? "forged" : evaluation.outcome === "leakage" ? "ruined" : "forging",
      whyCard: whyCardFor({ kind: "submitted", evaluation }),
    });

    if (won) {
      useProgression.getState().recordResult({
        slug: SLUG,
        score: evaluation.score,
        lane: state.lane,
        completed: true,
        codeLaneCleared: state.lane === "code",
      });
    }

    return evaluation;
  },

  reset: async () => {
    set({ ...freshState(), whyCard: whyCardFor({ kind: "briefing" }) });
    await get().initialise();
  },
}));

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
): Promise<void> {
  const state = get();
  if (state.training) return;

  set({ training: true, features, submitted: false, failure: null });

  const combined = allFeatures(features);
  const matrices = buildMatrices(state.dataset, combined);
  if (matrices === null) {
    set({ training: false });
    return;
  }

  const previousScore = state.currentScore;
  const result = await fitAndScore(matrices);
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
