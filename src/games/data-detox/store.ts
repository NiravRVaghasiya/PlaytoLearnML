"use client";

import { create } from "zustand";
import { useProgression } from "@/engine/progression";
import type { Lane, NamedFailure } from "@/engine/types";
import type { WhyCardContent } from "@/components";
import {
  BATCH_SIZE,
  applyPipeline,
  columnStats,
  evaluate,
  generateDataset,
  seedForRound,
  testMatrix,
  type CleaningAction,
  type ColumnStats,
  type Dataset,
  type Evaluation,
  type PipelineResult,
  type Row,
} from "./ml";
import { whyCardFor } from "./why-cards";

export const SLUG = "data-detox";

/**
 * Data Detox state.
 *
 * The spec's data model is:
 *
 *   Row       { id, features:{}, isNull, isOutlier, playerAction }
 *   GameState { rows[], modelAccuracy, timeElapsed, binCounts }
 *
 * All present. `label` is added to Row because the downstream model is supervised
 * and the spec's schema omits it; `pipeline` caches the derived training matrix so
 * the belt doesn't recompute it on every render.
 *
 * ── Why training lives outside this store ────────────────────────────────────
 * The model is a real TensorFlow.js net, and TF.js tensors must never sit in React
 * state (the tfjs-model-lifecycle skill). So `useModel` is owned by the component
 * and calls `beginTraining()` / `reportAccuracy()` here. The store holds only
 * plain numbers, which keeps it serialisable and keeps disposal in one place.
 */

export interface DataDetoxState {
  round: number;
  seed: number;

  rows: Row[];
  dataset: Dataset;
  stats: ColumnStats;
  pipeline: PipelineResult;

  /** Index of the row at the head of the belt. */
  cursor: number;

  /** Held-out accuracy of the last completed retrain. Null before the first. */
  modelAccuracy: number | null;
  /** True while a retrain is in flight. */
  training: boolean;
  /** Decisions made since the last retrain. */
  sinceRetrain: number;
  /** Retrains completed this round. */
  retrains: number;

  /** Seconds since the round started (spec: `timeElapsed`). */
  timeElapsed: number;

  failure: NamedFailure | null;
  whyCard: WhyCardContent | null;
  lastEvaluation: Evaluation | null;
  won: boolean;
  lane: Lane;

  // ── actions ──────────────────────────────────────────────────────────
  setLane: (lane: Lane) => void;
  /** Sort the row at the head of the belt. The core mechanic. */
  sortRow: (action: CleaningAction) => void;
  /** Re-decide a specific row, e.g. from the code lane. */
  setRowAction: (rowId: string, action: CleaningAction) => void;
  /** Move the belt without deciding, so a player can come back to a row. */
  skip: () => void;
  /** Jump the belt to a specific position. */
  focusRow: (index: number) => void;

  /** True when enough decisions have accumulated to justify a retrain. */
  needsRetrain: () => boolean;
  beginTraining: () => void;
  /** Called by the component when the model finishes. */
  reportAccuracy: (accuracy: number) => void;
  /** Wall clock, ticked by the component. */
  setElapsed: (seconds: number) => void;

  check: () => Evaluation;
  reset: () => void;
  startRound: (round: number) => void;
  newRound: () => void;
  /** The held-out matrix, for the component to score the model against. */
  testSet: () => { xs: number[][]; ys: number[] };
}

function freshRound(round: number) {
  const seed = seedForRound(round);
  const dataset = generateDataset(seed);
  const stats = columnStats(dataset.train);
  const rows = dataset.train.map((row) => ({ ...row }));

  return {
    round,
    seed,
    rows,
    dataset,
    stats,
    pipeline: applyPipeline(rows, stats),
    cursor: 0,
    modelAccuracy: null,
    training: false,
    sinceRetrain: 0,
    retrains: 0,
    timeElapsed: 0,
    failure: null,
    whyCard: whyCardFor({ kind: "reset" }),
    lastEvaluation: null,
    won: false,
  };
}

/** Next undecided row at or after `from`, wrapping once. */
function nextUndecided(rows: Row[], from: number): number {
  for (let index = from; index < rows.length; index += 1) {
    if (rows[index]!.playerAction === null) return index;
  }
  for (let index = 0; index < from; index += 1) {
    if (rows[index]!.playerAction === null) return index;
  }
  // Everything decided — park on the last row.
  return Math.min(from, rows.length - 1);
}

export const useDataDetoxStore = create<DataDetoxState>((set, get) => ({
  ...freshRound(1),
  lane: "visual" as Lane,

  setLane: (lane) => set({ lane }),

  sortRow: (action) => {
    const { rows, cursor, stats, sinceRetrain } = get();
    const row = rows[cursor];
    if (!row) return;

    const nextRows = rows.map((candidate, index) =>
      index === cursor ? { ...candidate, playerAction: action } : candidate,
    );
    const pipeline = applyPipeline(nextRows, stats);

    set({
      rows: nextRows,
      pipeline,
      cursor: nextUndecided(nextRows, cursor + 1),
      sinceRetrain: sinceRetrain + 1,
      failure: null,
      won: false,
      whyCard: whyCardFor({
        kind: "sorted",
        action,
        row,
        pipeline,
      }),
    });
  },

  setRowAction: (rowId, action) => {
    const { rows, stats, sinceRetrain } = get();
    const index = rows.findIndex((row) => row.id === rowId);
    if (index < 0) return;

    const wasUndecided = rows[index]!.playerAction === null;
    const nextRows = rows.map((row, i) =>
      i === index ? { ...row, playerAction: action } : row,
    );
    const pipeline = applyPipeline(nextRows, stats);

    set({
      rows: nextRows,
      pipeline,
      sinceRetrain: sinceRetrain + (wasUndecided ? 1 : 0),
      failure: null,
      won: false,
    });
  },

  skip: () => {
    const { rows, cursor } = get();
    set({ cursor: (cursor + 1) % Math.max(1, rows.length) });
  },

  focusRow: (index) => {
    const { rows } = get();
    if (index < 0 || index >= rows.length) return;
    set({ cursor: index });
  },

  // Spec: "every N rows, a downstream model retrains".
  needsRetrain: () => {
    const { sinceRetrain, pipeline, training } = get();
    if (training || pipeline.kept === 0) return false;
    return sinceRetrain >= BATCH_SIZE || (pipeline.undecided === 0 && sinceRetrain > 0);
  },

  beginTraining: () => set({ training: true }),

  reportAccuracy: (accuracy) => {
    const { modelAccuracy, retrains, pipeline } = get();
    set({
      modelAccuracy: accuracy,
      training: false,
      sinceRetrain: 0,
      retrains: retrains + 1,
      whyCard: whyCardFor({
        kind: "retrained",
        accuracy,
        previous: modelAccuracy,
        pipeline,
      }),
    });
  },

  setElapsed: (seconds) => set({ timeElapsed: seconds }),

  check: () => {
    const { pipeline, modelAccuracy, timeElapsed, lane } = get();
    const evaluation = evaluate(pipeline, modelAccuracy ?? 0, timeElapsed);

    set({
      failure: evaluation.failure,
      won: evaluation.outcome === "win",
      lastEvaluation: evaluation,
      whyCard: whyCardFor({ kind: "checked", evaluation }),
    });

    if (evaluation.outcome === "win") {
      useProgression.getState().recordResult({
        slug: SLUG,
        score: evaluation.score,
        lane,
        completed: true,
        codeLaneCleared: lane === "code",
      });
    }

    return evaluation;
  },

  reset: () => {
    const { rows, stats } = get();
    const cleared = rows.map((row) => ({ ...row, playerAction: null }));
    set({
      rows: cleared,
      pipeline: applyPipeline(cleared, stats),
      cursor: 0,
      modelAccuracy: null,
      training: false,
      sinceRetrain: 0,
      retrains: 0,
      timeElapsed: 0,
      failure: null,
      won: false,
      lastEvaluation: null,
      whyCard: whyCardFor({ kind: "reset" }),
    });
  },

  startRound: (round) => set(freshRound(round)),
  newRound: () => set(freshRound(get().round + 1)),

  testSet: () => testMatrix(get().dataset),
}));

export { BATCH_SIZE };
export type { CleaningAction, Evaluation, Row };
