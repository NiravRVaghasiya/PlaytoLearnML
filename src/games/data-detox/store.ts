"use client";

import { useEffect } from "react";
import { create } from "zustand";
import { useProgression } from "@/engine/progression";
import type { Lane, NamedFailure } from "@/engine/types";
import type { WhyCardContent } from "@/components";
import {
  BATCH_SIZE,
  CLEANING_ACTIONS,
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
 *
 * ── Why an accuracy carries the pipeline it was trained on ───────────────────
 * A fit takes a second or two, and the pipeline keeps moving underneath it: rows
 * get sorted mid-fit, Retry wipes the decisions, a new batch swaps the dataset,
 * the code lane fires a second retrain. An accuracy that is just "the last
 * number that came back" can therefore describe a pipeline that no longer exists
 * — the meter showing 84% with nothing sorted, or Score judging a 70-row model as
 * if it had seen all 72. So every change to the decisions bumps
 * `pipelineVersion`, every training gets a ticket, and a result is accepted only
 * if its ticket is still the current one. `check()` scores an accuracy only when
 * it was trained on exactly the pipeline being scored.
 */

export interface TrainingTicket {
  /** Unique per training. A result whose id is no longer current is dropped. */
  id: number;
  /** The training matrix, snapshotted when the training was requested. */
  xs: number[][];
  ys: number[];
  /** The held-out matrix for the same round. */
  test: { xs: number[][]; ys: number[] };
}

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

  /**
   * Bumped on every change to the decisions, and on Retry and every new round.
   * Never reused, so a stale result can't match a later pipeline by accident.
   */
  pipelineVersion: number;
  /** The `pipelineVersion` that `modelAccuracy` was trained on. */
  trainedVersion: number | null;
  /** Id of the most recent training requested. Monotonic, never reused. */
  trainingId: number;
  /** What the in-flight training snapshotted: its version, and its decisions. */
  trainingVersion: number | null;
  trainingDecisions: number;
  trainingPipeline: PipelineResult | null;
  /** Every row's action as the in-flight training snapshotted it. */
  trainingActions: ReadonlyArray<CleaningAction | null> | null;
  /**
   * Every row's action as `modelAccuracy` was trained on it. The retrain card
   * compares against this, so "same pipeline" means the same decisions — not
   * merely "no NEW rows sorted", which a code-lane re-decision also is.
   */
  trainedActions: ReadonlyArray<CleaningAction | null> | null;
  /**
   * The pipeline version whose training last came back unusable (failed,
   * stopped, unmounted). The automatic retrain won't retry that exact pipeline
   * — a backend that fails every fit would otherwise retrain in a loop — but
   * the next decision, Retrain now, or api.retrain() all try again.
   */
  abandonedVersion: number | null;

  /** Seconds spent on this attempt with the game on screen (spec: `timeElapsed`). */
  timeElapsed: number;
  /**
   * Wall-clock moment `timeElapsed` counts from, ms. Retry restarts it, and
   * `resumeClock` moves it so time spent away from the game isn't charged.
   */
  startedAt: number;

  failure: NamedFailure | null;
  whyCard: WhyCardContent | null;
  lastEvaluation: Evaluation | null;
  won: boolean;
  /**
   * The pipeline version a clear was last recorded for, and from which lanes.
   * Pressing Score again on the same cleaning records nothing new — the same
   * once-per-attempt rule as the Skier's once-per-run.
   */
  recordedVersion: number | null;
  recordedFor: Lane[];
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
  /** Claim the model for a training on the pipeline as it stands right now. */
  beginTraining: () => TrainingTicket;
  /** Is this ticket still the one whose result would be accepted? */
  isCurrentTraining: (id: number) => boolean;
  /**
   * Called by the component when a fit finishes. Returns false, and changes
   * nothing, if a newer training, a Retry or a new round made the ticket stale.
   */
  reportAccuracy: (accuracy: number, trainingId: number) => boolean;
  /** The fit produced no usable model (stopped, unmounted, failed). */
  abandonTraining: (trainingId: number) => void;
  /** True when `modelAccuracy` was trained on exactly the current pipeline. */
  isTrained: () => boolean;
  /** Wall clock, ticked by the component. */
  setElapsed: (seconds: number) => void;
  /**
   * Carry on from the seconds already charged: moves `startedAt` so the next
   * tick reads `timeElapsed` again, not the wall-clock gap. `useAttemptClock`
   * calls it on mount and when the tab becomes visible, because this store
   * outlives the page — a player who follows a WhyCard to the Concept Library
   * and reads for two minutes wasn't cleaning for those two minutes.
   */
  resumeClock: () => void;

  /**
   * Score the pipeline. `source` is where the request came from — "code" only
   * for `api.check()` — and that, not the visible tab, is what earns the
   * code-lane star.
   */
  check: (source?: Lane) => Evaluation;
  reset: () => void;
  startRound: (round: number) => void;
  newRound: () => void;
  /** The held-out matrix, for the component to score the model against. */
  testSet: () => { xs: number[][]; ys: number[] };
}

/** Counters that must keep counting across rounds, so tickets are never reused. */
interface Counters {
  pipelineVersion: number;
  trainingId: number;
}

function freshRound(round: number, counters: Counters) {
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
    pipelineVersion: counters.pipelineVersion + 1,
    trainedVersion: null,
    // Bumping the id orphans any fit still running for the previous round.
    trainingId: counters.trainingId + 1,
    trainingVersion: null,
    trainingDecisions: 0,
    trainingPipeline: null,
    trainingActions: null,
    trainedActions: null,
    abandonedVersion: null,
    timeElapsed: 0,
    startedAt: Date.now(),
    failure: null,
    whyCard: whyCardFor({ kind: "reset" }),
    lastEvaluation: null,
    won: false,
    recordedVersion: null,
    recordedFor: [] as Lane[],
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
  ...freshRound(1, { pipelineVersion: 0, trainingId: 0 }),
  lane: "visual" as Lane,

  setLane: (lane) => set({ lane }),

  sortRow: (action) => {
    const { rows, cursor, stats, sinceRetrain, pipelineVersion } = get();
    const row = rows[cursor];
    if (!row) return;

    const nextRows = rows.map((candidate, index) =>
      index === cursor ? { ...candidate, playerAction: action } : candidate,
    );
    const pipeline = applyPipeline(nextRows, stats);

    set({
      rows: nextRows,
      pipeline,
      pipelineVersion: pipelineVersion + 1,
      cursor: nextUndecided(nextRows, cursor + 1),
      sinceRetrain: sinceRetrain + 1,
      failure: null,
      won: false,
      whyCard: whyCardFor({
        kind: "sorted",
        action,
        row,
        pipeline,
        stats,
      }),
    });
  },

  setRowAction: (rowId, action) => {
    const { rows, stats, sinceRetrain, pipelineVersion } = get();
    const index = rows.findIndex((row) => row.id === rowId);
    if (index < 0) return;
    // Same decision again — nothing changed, so nothing is stale. This is what
    // lets a snippet be run twice without invalidating the model it trained.
    if (rows[index]!.playerAction === action) return;

    const wasUndecided = rows[index]!.playerAction === null;
    const nextRows = rows.map((row, i) =>
      i === index ? { ...row, playerAction: action } : row,
    );
    const pipeline = applyPipeline(nextRows, stats);

    set({
      rows: nextRows,
      pipeline,
      pipelineVersion: pipelineVersion + 1,
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

  // Spec: "every N rows, a downstream model retrains". Also once the belt is
  // empty, whenever the model hasn't seen the final pipeline yet — rows sorted
  // during the last fit, or a re-decision from the code lane.
  needsRetrain: () => {
    const {
      sinceRetrain,
      pipeline,
      training,
      trainedVersion,
      pipelineVersion,
      abandonedVersion,
    } = get();
    if (training || pipeline.kept === 0) return false;
    if (abandonedVersion === pipelineVersion) return false;
    if (sinceRetrain >= BATCH_SIZE) return true;
    return (
      pipeline.undecided === 0 &&
      (sinceRetrain > 0 || trainedVersion !== pipelineVersion)
    );
  },

  beginTraining: () => {
    const state = get();
    const id = state.trainingId + 1;
    set({
      training: true,
      trainingId: id,
      trainingVersion: state.pipelineVersion,
      trainingDecisions: state.sinceRetrain,
      trainingPipeline: state.pipeline,
      trainingActions: state.rows.map((row) => row.playerAction),
    });
    return {
      id,
      xs: state.pipeline.xs,
      ys: state.pipeline.ys,
      test: testMatrix(state.dataset),
    };
  },

  isCurrentTraining: (id) => {
    const { trainingId, training } = get();
    return training && id === trainingId;
  },

  reportAccuracy: (accuracy, trainingId) => {
    const state = get();
    if (!state.isCurrentTraining(trainingId) || state.trainingVersion === null) {
      return false;
    }

    const trained = state.trainingPipeline ?? state.pipeline;
    const actions =
      state.trainingActions ?? state.rows.map((row) => row.playerAction);
    // Rows whose decision differs from the ones the previous model saw: new
    // sorts AND re-decisions. `trainingDecisions` counts only the first kind,
    // so a code-lane re-decision of every blank read as "same pipeline".
    const previous = state.trainedActions;
    const changed =
      previous === null
        ? state.trainingDecisions
        : actions.filter((action, index) => action !== previous[index]).length;
    set({
      modelAccuracy: accuracy,
      trainedVersion: state.trainingVersion,
      trainedActions: actions,
      training: false,
      trainingVersion: null,
      trainingDecisions: 0,
      trainingPipeline: null,
      trainingActions: null,
      // Subtract what this fit saw rather than zeroing: rows sorted while it ran
      // still count toward the next retrain, so the final pipeline always gets
      // trained on.
      sinceRetrain: Math.max(0, state.sinceRetrain - state.trainingDecisions),
      retrains: state.retrains + 1,
      whyCard: whyCardFor({
        kind: "retrained",
        accuracy,
        previous: state.modelAccuracy,
        pipeline: trained,
        decisions: changed,
      }),
    });
    return true;
  },

  abandonTraining: (trainingId) => {
    const state = get();
    if (!state.isCurrentTraining(trainingId)) return;
    set({
      training: false,
      abandonedVersion: state.trainingVersion,
      trainingVersion: null,
      trainingDecisions: 0,
      trainingPipeline: null,
      trainingActions: null,
    });
  },

  isTrained: () => {
    const { trainedVersion, pipelineVersion, modelAccuracy } = get();
    return modelAccuracy !== null && trainedVersion === pipelineVersion;
  },

  setElapsed: (seconds) => set({ timeElapsed: seconds }),
  resumeClock: () =>
    set({ startedAt: Date.now() - get().timeElapsed * 1000 }),

  check: (source = "visual") => {
    const state = get();
    const accuracy = state.isTrained() ? state.modelAccuracy : null;
    const evaluation = evaluate(state.pipeline, accuracy, state.timeElapsed);

    // One clear per cleaning per lane: Score pressed three times on the same
    // pipeline used to record three plays. Any change to the decisions, Retry
    // and a new batch all bump pipelineVersion, which starts a fresh record.
    const recorded =
      state.recordedVersion === state.pipelineVersion ? state.recordedFor : [];
    const record =
      evaluation.outcome === "win" &&
      evaluation.score !== null &&
      !recorded.includes(source);

    set({
      failure: evaluation.failure,
      won: evaluation.outcome === "win",
      lastEvaluation: evaluation,
      whyCard: whyCardFor({ kind: "checked", evaluation }),
      ...(record
        ? {
            recordedVersion: state.pipelineVersion,
            recordedFor: [...recorded, source],
          }
        : {}),
    });

    if (record && evaluation.score !== null) {
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

  reset: () => {
    const { rows, stats, pipelineVersion, trainingId } = get();
    const cleared = rows.map((row) => ({ ...row, playerAction: null }));
    set({
      rows: cleared,
      pipeline: applyPipeline(cleared, stats),
      cursor: 0,
      modelAccuracy: null,
      training: false,
      sinceRetrain: 0,
      retrains: 0,
      pipelineVersion: pipelineVersion + 1,
      trainedVersion: null,
      // Orphans a fit still in flight: its result can no longer be accepted.
      trainingId: trainingId + 1,
      trainingVersion: null,
      trainingDecisions: 0,
      trainingPipeline: null,
      trainingActions: null,
      trainedActions: null,
      abandonedVersion: null,
      timeElapsed: 0,
      startedAt: Date.now(),
      failure: null,
      won: false,
      lastEvaluation: null,
      recordedVersion: null,
      recordedFor: [],
      whyCard: whyCardFor({ kind: "reset" }),
    });
  },

  startRound: (round) => set(freshRound(round, get())),
  newRound: () => set(freshRound(get().round + 1, get())),

  testSet: () => testMatrix(get().dataset),
}));

/**
 * The spec's `timeElapsed`, ticked once a second while the game is on screen.
 *
 * The store outlives the page, so the clock resumes from the seconds already
 * charged on mount and whenever the tab comes back, and doesn't tick while the
 * tab is hidden. Measuring against a wall-clock start instead charged a player
 * who followed a WhyCard to the Concept Library for every minute spent reading
 * it. Retry and a new batch still restart the clock, through `startedAt`.
 */
export function useAttemptClock(): void {
  useEffect(() => {
    const store = useDataDetoxStore;
    store.getState().resumeClock();
    const timer = setInterval(() => {
      if (document.visibilityState === "hidden") return;
      const state = store.getState();
      state.setElapsed(Math.round((Date.now() - state.startedAt) / 1000));
    }, 1000);
    const onVisibility = () => {
      if (document.visibilityState === "visible") store.getState().resumeClock();
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);
}

// ── The code lane's API ─────────────────────────────────────────────────────

/** A row as the code lane sees it: plain data, no label (that would be cheating). */
export interface CodeRow {
  id: string;
  features: Row["features"];
  isNull: boolean;
  isOutlier: boolean;
}

function isCleaningAction(value: unknown): value is CleaningAction {
  return (
    typeof value === "string" &&
    (CLEANING_ACTIONS as readonly string[]).includes(value)
  );
}

const ACTION_LIST = CLEANING_ACTIONS.map((action) => `'${action}'`).join(", ");

/**
 * The `api` object the code lane's snippet receives. Built here, next to the
 * store, so its argument checking is unit-testable without rendering anything.
 *
 * Every verb validates what it is given and throws a named, readable Error
 * rather than writing garbage into the pipeline.
 */
export function createCodeApi(retrain: () => Promise<number | null>) {
  const store = useDataDetoxStore;

  const codeRow = (row: Row): CodeRow => ({
    id: row.id,
    features: { ...row.features },
    isNull: row.isNull,
    isOutlier: row.isOutlier,
  });

  return {
    /**
     * Apply a decision function to every row. The pipeline, as code.
     *
     * All-or-nothing: every decision is collected and checked before any is
     * applied, so a rule that throws or returns nonsense on row 40 leaves the
     * pipeline exactly as it was.
     */
    forEachRow: (decide: (row: CodeRow) => CleaningAction) => {
      if (typeof decide !== "function") {
        throw new TypeError(
          "forEachRow needs a function, e.g. api.forEachRow((row) => 'keep')",
        );
      }
      const rows = store.getState().rows;
      const decisions = rows.map((row) => {
        const action: unknown = decide(codeRow(row));
        if (!isCleaningAction(action)) {
          throw new TypeError(
            `Row ${row.id}: expected one of ${ACTION_LIST}, got ${JSON.stringify(action) ?? String(action)}`,
          );
        }
        return [row.id, action] as const;
      });
      for (const [id, action] of decisions) {
        store.getState().setRowAction(id, action);
      }
    },

    /** Decide one row by id. Same action the bins call. */
    setAction: (rowId: string, action: CleaningAction) => {
      if (!store.getState().rows.some((row) => row.id === rowId)) {
        throw new RangeError(
          `setAction: no row with id ${JSON.stringify(rowId) ?? String(rowId)} — ids look like 'row-0' to 'row-${store.getState().rows.length - 1}'`,
        );
      }
      if (!isCleaningAction(action)) {
        throw new TypeError(
          `setAction: expected one of ${ACTION_LIST}, got ${JSON.stringify(action) ?? String(action)}`,
        );
      }
      store.getState().setRowAction(rowId, action);
    },

    /** The rows, as plain data. */
    rows: () =>
      store.getState().rows.map((row) => ({
        ...codeRow(row),
        action: row.playerAction,
      })),

    kept: () => store.getState().pipeline.kept,
    dropped: () => store.getState().pipeline.dropped,
    blanksLeft: () => store.getState().pipeline.keptWithNaiveFill,
    extremesLeft: () => store.getState().pipeline.keptWithOutlier,
    balanceDrift: () => store.getState().pipeline.balanceDrift,
    binCounts: () => ({ ...store.getState().pipeline.binCounts }),

    /** Column medians and cap fences, as the pipeline computes them. */
    stats: () => {
      const { stats } = store.getState();
      return {
        median: { ...stats.median },
        low: { ...stats.low },
        high: { ...stats.high },
      };
    },

    /** Retrain and return held-out accuracy. Real TF.js, awaited. */
    retrain: async (): Promise<number> => {
      if (store.getState().pipeline.kept === 0) {
        throw new Error(
          "retrain: no rows are kept, so there is nothing to train on. Keep, impute or cap some rows first.",
        );
      }
      const accuracy = await retrain();
      if (accuracy === null) {
        throw new Error(
          "retrain: the fit was cancelled before it finished — a Retry, a new batch or another retrain took over the model.",
        );
      }
      return accuracy;
    },

    /** Accuracy of the model trained on the CURRENT pipeline, or null. */
    accuracy: () => {
      const state = store.getState();
      return state.isTrained() ? state.modelAccuracy : null;
    },
    check: () => store.getState().check("code"),
    reset: () => store.getState().reset(),
  };
}

export { BATCH_SIZE };
export type { CleaningAction, Evaluation, Row };
