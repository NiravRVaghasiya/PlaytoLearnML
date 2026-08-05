"use client";

import { create } from "zustand";
import { useProgression } from "@/engine/progression";
import type { Lane, NamedFailure } from "@/engine/types";
import type { WhyCardContent } from "@/components";
import {
  KNOT_COUNTS,
  MIN_PARAMS,
  ROUND_SEEDS,
  boundaryAt,
  classifyPoint,
  complexityCostOf,
  evaluate,
  fitKnots,
  generateTestPoints,
  generateTrainPoints,
  penaltyFor,
  resampleParams,
  scoreFor,
  seedForRound,
  type BoundaryType,
  type Evaluation,
  type Point,
} from "./ml";
import { whyCardFor } from "./why-cards";

export const SLUG = "sort-it-arcade";

/**
 * Sort-It Arcade state.
 *
 * The spec's data model is:
 *
 *   Point      { x, y, label, predictedSide }
 *   Boundary   { type, params[], complexityCost }
 *   GameState  { points[], accuracy, penalty, score }
 *
 * Two deliberate extensions, both forced by the pedagogy contract rather than
 * by taste:
 *
 * 1. `points[]` is split into `trainPoints` / `testPoints`. The contract demands
 *    a NAMED failure mode backed by numbers, and the only honest way to say "you
 *    overfit" is to compare performance on points the player fitted against
 *    points they never saw. Without a held-out set, the word "overfitting" would
 *    be decoration.
 * 2. `testAccuracy` stays `null` until the player checks. Showing it live would
 *    turn the test set into a second training set — the exact mistake the game
 *    is teaching against.
 *
 * Both lanes mutate this one store (CLAUDE.md two-lane rule). The visual lane
 * drags knots; the code lane calls the same actions by name.
 */

export interface Boundary {
  type: BoundaryType;
  /** Knot heights at evenly spaced x positions. The spec's `params[]`. */
  params: number[];
  /** Parameter count — the spec's "complexity = param count". */
  complexityCost: number;
}

export interface SortItState {
  seed: number;
  round: number;

  trainPoints: Point[];
  testPoints: Point[];
  boundary: Boundary;

  /** Train accuracy: the always-visible live metric (contract #3). */
  accuracy: number;
  /** Regularization pressure paid for extra parameters. */
  penalty: number;
  /** accuracy − penalty. What the round is scored on. */
  score: number;

  /** Held-out accuracy. `null` until `check()` — see the note above. */
  testAccuracy: number | null;
  /** Ids of currently misclassified training points, for the flash. */
  misclassifiedIds: string[];

  failure: NamedFailure | null;
  whyCard: WhyCardContent | null;
  lastEvaluation: Evaluation | null;
  won: boolean;
  lane: Lane;

  // ── actions ──────────────────────────────────────────────────────────
  setLane: (lane: Lane) => void;
  /** Swap model capacity: line (2 params) → curve (4) → wiggle (10). */
  setBoundaryType: (type: BoundaryType) => void;
  /** Move one knot. This is the player fitting the classifier by hand. */
  setKnot: (index: number, y: number) => void;
  /** Keyboard-sized nudge. */
  nudgeKnot: (index: number, delta: number) => void;
  /** Run the real optimizer — the same operation the dragging performs. */
  autoFit: () => void;
  /** Score the round against the held-out set and name any failure. */
  check: () => Evaluation;
  /** Same points, boundary back to a flat line. */
  reset: () => void;
  /** Load a specific round from the vetted seed list. */
  startRound: (round: number) => void;
  /** Fresh sample from the same generative process. */
  newRound: () => void;
}

function makeBoundary(type: BoundaryType, params: number[]): Boundary {
  return { type, params, complexityCost: complexityCostOf(params) };
}

/** A flat line through the middle — the honest "no model yet" starting point. */
function flatParams(type: BoundaryType): number[] {
  return Array.from({ length: KNOT_COUNTS[type] }, () => 0.5);
}

/** Recompute everything derived from the boundary. Keeps the metric truthful. */
function derive(
  trainPoints: Point[],
  params: number[],
): {
  trainPoints: Point[];
  accuracy: number;
  penalty: number;
  score: number;
  misclassifiedIds: string[];
} {
  const scored = trainPoints.map((point) => ({
    ...point,
    predictedSide: classifyPoint(params, point),
  }));

  const misclassifiedIds = scored
    .filter((point) => point.predictedSide !== point.label)
    .map((point) => point.id);

  const accuracy =
    scored.length === 0
      ? 0
      : (scored.length - misclassifiedIds.length) / scored.length;

  return {
    trainPoints: scored,
    accuracy,
    penalty: penaltyFor(params),
    score: scoreFor(accuracy, params),
    misclassifiedIds,
  };
}

function freshRound(round: number) {
  const seed = seedForRound(round);
  const type: BoundaryType = "line";
  const params = flatParams(type);
  const trainPoints = generateTrainPoints(seed);
  const testPoints = generateTestPoints(seed);

  return {
    seed,
    round,
    testPoints,
    boundary: makeBoundary(type, params),
    testAccuracy: null,
    failure: null,
    whyCard: whyCardFor({ kind: "reset" }),
    lastEvaluation: null,
    won: false,
    ...derive(trainPoints, params),
  };
}

export const useSortItStore = create<SortItState>((set, get) => ({
  ...freshRound(1),
  lane: "visual" as Lane,

  setLane: (lane) => set({ lane }),

  setBoundaryType: (type) => {
    const { boundary, trainPoints } = get();
    if (type === boundary.type) return;

    // Sample the current boundary at the new knot positions, so changing
    // capacity keeps the player's work instead of wiping it. The lesson is about
    // capacity, not about punishing experimentation.
    const params = resampleParams(boundary.params, KNOT_COUNTS[type]);
    const next = derive(trainPoints, params);

    set({
      boundary: makeBoundary(type, params),
      ...next,
      // A capacity change invalidates the previous verdict.
      testAccuracy: null,
      failure: null,
      won: false,
      whyCard: whyCardFor({
        kind: "complexity-changed",
        from: boundary.type,
        to: type,
        params: params.length,
        penalty: next.penalty,
      }),
    });
  },

  setKnot: (index, y) => {
    const { boundary, trainPoints, accuracy } = get();
    if (index < 0 || index >= boundary.params.length) return;

    const params = [...boundary.params];
    params[index] = Math.min(1, Math.max(0, y));

    const next = derive(trainPoints, params);

    set({
      boundary: makeBoundary(boundary.type, params),
      ...next,
      testAccuracy: null,
      failure: null,
      whyCard: whyCardFor({
        kind: "boundary-moved",
        accuracy: next.accuracy,
        accuracyDelta: next.accuracy - accuracy,
        misclassified: next.misclassifiedIds.length,
        total: trainPoints.length,
      }),
    });
  },

  nudgeKnot: (index, delta) => {
    const current = get().boundary.params[index];
    if (current === undefined) return;
    get().setKnot(index, current + delta);
  },

  autoFit: () => {
    const { boundary, trainPoints } = get();
    const params = fitKnots(trainPoints, boundary.params);
    const next = derive(trainPoints, params);

    set({
      boundary: makeBoundary(boundary.type, params),
      ...next,
      testAccuracy: null,
      failure: null,
      whyCard: whyCardFor({
        kind: "auto-fitted",
        accuracy: next.accuracy,
        params: params.length,
      }),
    });
  },

  check: () => {
    const { trainPoints, testPoints, boundary, lane } = get();
    const evaluation = evaluate(trainPoints, testPoints, boundary.params);

    set({
      testAccuracy: evaluation.testAccuracy,
      failure: evaluation.failure,
      won: evaluation.outcome === "win",
      lastEvaluation: evaluation,
      whyCard: whyCardFor({ kind: "checked", evaluation }),
    });

    if (evaluation.outcome === "win") {
      // Third mastery star is the code-lane challenge (spec §4).
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
    const { boundary, trainPoints } = get();
    const params = flatParams(boundary.type);

    set({
      boundary: makeBoundary(boundary.type, params),
      ...derive(trainPoints, params),
      testAccuracy: null,
      failure: null,
      won: false,
      lastEvaluation: null,
      whyCard: whyCardFor({ kind: "reset" }),
    });
  },

  startRound: (round) => {
    set(freshRound(round));
  },

  newRound: () => {
    set(freshRound(get().round + 1));
  },
}));

/** Boundary height at `x` for the current params. Used by the renderers. */
export function boundaryHeightAt(state: SortItState, x: number): number {
  return boundaryAt(state.boundary.params, x);
}

/** Evenly spaced knot x positions for the current boundary. */
export function knotPositions(state: SortItState): number[] {
  const count = state.boundary.params.length;
  if (count <= 1) return [0.5];
  return Array.from({ length: count }, (_, i) => i / (count - 1));
}

export { KNOT_COUNTS, MIN_PARAMS, ROUND_SEEDS };
export type { BoundaryType, Evaluation, Point };
