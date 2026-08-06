"use client";

import { create } from "zustand";
import { useProgression } from "@/engine/progression";
import type { Lane, NamedFailure } from "@/engine/types";
import type { WhyCardContent } from "@/components";
import { clamp } from "@/lib/utils";
import {
  CONVERGENCE_EPSILON,
  MAX_K,
  MIN_K,
  ROUNDS,
  assignPoints,
  clusterSizes,
  elbowCurve,
  elbowKFor,
  evaluate,
  generateVillages,
  inertiaOfAssignment,
  largestShift,
  runToConvergence,
  solveKMeans,
  updateCentroids,
  type Centroid,
  type ElbowPoint,
  type Evaluation,
  type Point,
} from "./ml";
import { whyCardFor } from "./why-cards";

export const SLUG = "k-means-territory-wars";

/**
 * K-Means Territory Wars state.
 *
 * The spec's data model is:
 *
 *   Point     { x, y, clusterId }
 *   Centroid  { x, y, id }
 *   GameState { points[], centroids[], k, inertia, iteration }
 *
 * All of it is here. The additions all exist to serve the pedagogy contract:
 *
 * - `assignmentStale` — true when flags have moved since the last assign. This
 *   makes the two halves of the algorithm visibly distinct, which the core
 *   intuition depends on ("centroids converge by alternating assign/update").
 * - `elbow` — the real inertia-vs-k curve, so the player can *read* the elbow
 *   rather than be told the answer. Choosing k is the skill being taught.
 * - `converged` / `lastShift` — convergence is a state you reach, not a timer.
 * - `bestInertiaAtK` — a multi-restart reference, so "you hit a local minimum"
 *   is a measured claim rather than a guess.
 *
 * Both lanes mutate this one store (CLAUDE.md two-lane rule): the visual lane
 * drags flags and clicks Assign/Update, the code lane calls `api.assign()` and
 * `api.update()`. Same actions, same numbers.
 */

export interface KMeansState {
  round: number;
  seed: number;
  /**
   * How many blobs the generator actually made. Never rendered — this is
   * unsupervised learning, and revealing it would remove the entire task. It
   * exists so tests can verify the elbow recovers it.
   */
  trueK: number;

  points: Point[];
  centroids: Centroid[];
  k: number;
  inertia: number;
  iteration: number;

  converged: boolean;
  /** How far the furthest flag moved on the last update. */
  lastShift: number | null;
  /** Flags have moved since the last assign, so the colouring is out of date. */
  assignmentStale: boolean;

  elbowPoints: ElbowPoint[];
  elbowK: number;
  /** Villages per flag, for the empty-cluster warning. */
  sizes: number[];

  selectedFlag: number | null;

  failure: NamedFailure | null;
  whyCard: WhyCardContent | null;
  lastEvaluation: Evaluation | null;
  won: boolean;
  lane: Lane;

  // ── actions ──────────────────────────────────────────────────────────
  setLane: (lane: Lane) => void;
  selectFlag: (index: number | null) => void;

  /** Drop a flag. This is the player choosing k. */
  addFlag: (x?: number, y?: number) => void;
  removeFlag: (index: number) => void;
  /** Move a flag. This is the player choosing the initialization. */
  moveFlag: (index: number, x: number, y: number) => void;
  nudgeFlag: (index: number, dx: number, dy: number) => void;

  /** ASSIGN half of the loop: villages join their nearest flag. */
  assign: () => void;
  /** UPDATE half of the loop: flags move to their villages' mean. */
  update: () => void;
  /** Both halves, once. */
  step: () => void;
  /** Iterate until no flag moves. */
  settle: () => void;

  check: () => Evaluation;
  reset: () => void;
  startRound: (round: number) => void;
  newRound: () => void;
}

/**
 * Elbow curves are the expensive part (eight k values × twelve restarts), so
 * they're computed once per map.
 *
 * Keyed by seed AND blob count, not seed alone: the same seed with a different
 * blob count is a different map, and a seed-only key would hand back a curve for
 * the wrong one.
 */
const elbowCache = new Map<string, { points: ElbowPoint[]; k: number }>();

function elbowFor(seed: number, trueK: number, points: Point[]) {
  const key = `${seed}:${trueK}`;
  const cached = elbowCache.get(key);
  if (cached) return cached;
  const curve = elbowCurve(points, MAX_K, seed);
  const computed = { points: curve, k: elbowKFor(curve) };
  elbowCache.set(key, computed);
  return computed;
}

/** Reference inertia per map and k. Also memoised — it runs 12 restarts. */
const referenceCache = new Map<string, number>();

function bestInertiaAt(
  seed: number,
  trueK: number,
  points: Point[],
  k: number,
): number {
  if (k <= 0) return Number.POSITIVE_INFINITY;
  const key = `${seed}:${trueK}:${k}`;
  const cached = referenceCache.get(key);
  if (cached !== undefined) return cached;
  const inertia = solveKMeans(points, k, { seed }).inertia;
  referenceCache.set(key, inertia);
  return inertia;
}

function nextFlagId(centroids: Centroid[]): number {
  return centroids.reduce((max, c) => Math.max(max, c.id), -1) + 1;
}

/** Two deliberately wrong starting flags — the player has to choose k. */
const STARTING_FLAGS: Centroid[] = [
  { id: 0, x: 0.3, y: 0.3 },
  { id: 1, x: 0.7, y: 0.7 },
];

function freshRound(round: number) {
  const config = ROUNDS[(Math.max(1, round) - 1) % ROUNDS.length]!;
  const { points: raw, trueK } = generateVillages(config.seed, config.trueK);

  const centroids = STARTING_FLAGS.map((flag) => ({ ...flag }));
  // Assign immediately so the metric is meaningful on arrival rather than 0.
  const points = assignPoints(raw, centroids);
  const elbow = elbowFor(config.seed, trueK, raw);

  return {
    round,
    seed: config.seed,
    trueK,
    points,
    centroids,
    k: centroids.length,
    inertia: inertiaOfAssignment(points, centroids),
    iteration: 0,
    converged: false,
    lastShift: null,
    assignmentStale: false,
    elbowPoints: elbow.points,
    elbowK: elbow.k,
    sizes: clusterSizes(points, centroids),
    selectedFlag: 0,
    failure: null,
    whyCard: whyCardFor({ kind: "reset" }),
    lastEvaluation: null,
    won: false,
  };
}

/** Recompute everything derived from (points, centroids). */
function derive(points: Point[], centroids: Centroid[]) {
  return {
    points,
    centroids,
    k: centroids.length,
    inertia: inertiaOfAssignment(points, centroids),
    sizes: clusterSizes(points, centroids),
  };
}

export const useKMeansStore = create<KMeansState>((set, get) => ({
  ...freshRound(1),
  lane: "visual" as Lane,

  setLane: (lane) => set({ lane }),
  selectFlag: (index) => set({ selectedFlag: index }),

  addFlag: (x, y) => {
    const { centroids, points } = get();
    if (centroids.length >= MAX_K) return;

    const flag: Centroid = {
      id: nextFlagId(centroids),
      // Default to the middle of the field, nudged so flags don't stack.
      x: clamp(x ?? 0.5 + (centroids.length % 3) * 0.04, 0, 1),
      y: clamp(y ?? 0.5 + (centroids.length % 2) * 0.04, 0, 1),
    };
    const next = [...centroids, flag];

    set({
      ...derive(points, next),
      selectedFlag: next.length - 1,
      assignmentStale: true,
      converged: false,
      failure: null,
      won: false,
      whyCard: whyCardFor({
        kind: "k-changed",
        k: next.length,
        elbowK: get().elbowK,
        added: true,
      }),
    });
  },

  removeFlag: (index) => {
    const { centroids, points } = get();
    if (centroids.length <= MIN_K || index < 0 || index >= centroids.length) {
      return;
    }

    const next = centroids.filter((_, i) => i !== index);
    // Village cluster ids point at array positions, so removing a flag
    // invalidates every assignment above it. Reassign rather than reindex.
    const reassigned = assignPoints(points, next);

    set({
      ...derive(reassigned, next),
      selectedFlag: next.length > 0 ? Math.min(index, next.length - 1) : null,
      assignmentStale: false,
      converged: false,
      failure: null,
      won: false,
      whyCard: whyCardFor({
        kind: "k-changed",
        k: next.length,
        elbowK: get().elbowK,
        added: false,
      }),
    });
  },

  moveFlag: (index, x, y) => {
    const { centroids, points } = get();
    if (index < 0 || index >= centroids.length) return;

    const next = centroids.map((centroid, i) =>
      i === index
        ? { ...centroid, x: clamp(x, 0, 1), y: clamp(y, 0, 1) }
        : centroid,
    );

    set({
      ...derive(points, next),
      assignmentStale: true,
      converged: false,
      failure: null,
      won: false,
      whyCard: whyCardFor({
        kind: "flag-moved",
        flag: index + 1,
        inertia: inertiaOfAssignment(points, next),
      }),
    });
  },

  nudgeFlag: (index, dx, dy) => {
    const centroid = get().centroids[index];
    if (!centroid) return;
    get().moveFlag(index, centroid.x + dx, centroid.y + dy);
  },

  assign: () => {
    const { points, centroids } = get();
    const assigned = assignPoints(points, centroids);
    const derived = derive(assigned, centroids);

    set({
      ...derived,
      assignmentStale: false,
      failure: null,
      whyCard: whyCardFor({
        kind: "assigned",
        inertia: derived.inertia,
        sizes: derived.sizes,
      }),
    });
  },

  update: () => {
    const { points, centroids, iteration } = get();
    const moved = updateCentroids(points, centroids);
    const shift = largestShift(centroids, moved);
    const derived = derive(points, moved);

    set({
      ...derived,
      iteration: iteration + 1,
      lastShift: shift,
      // Flags moved, so the colouring is now one step behind.
      assignmentStale: shift > 0,
      converged: shift < CONVERGENCE_EPSILON,
      failure: null,
      whyCard: whyCardFor({
        kind: "updated",
        inertia: derived.inertia,
        shift,
        converged: shift < CONVERGENCE_EPSILON,
      }),
    });
  },

  step: () => {
    const { points, centroids, iteration } = get();
    const assigned = assignPoints(points, centroids);
    const moved = updateCentroids(assigned, centroids);
    const shift = largestShift(centroids, moved);
    // Re-assign after moving so what's on screen matches where the flags are.
    const settled = assignPoints(assigned, moved);
    const derived = derive(settled, moved);

    set({
      ...derived,
      iteration: iteration + 1,
      lastShift: shift,
      assignmentStale: false,
      converged: shift < CONVERGENCE_EPSILON,
      failure: null,
      whyCard: whyCardFor({
        kind: "stepped",
        inertia: derived.inertia,
        shift,
        iteration: iteration + 1,
        converged: shift < CONVERGENCE_EPSILON,
      }),
    });
  },

  settle: () => {
    const { points, centroids, iteration } = get();
    const result = runToConvergence(points, centroids);
    const settled = assignPoints(points, result.centroids);
    const derived = derive(settled, result.centroids);

    set({
      ...derived,
      iteration: iteration + result.iterations,
      lastShift: 0,
      assignmentStale: false,
      converged: true,
      failure: null,
      whyCard: whyCardFor({
        kind: "settled",
        inertia: derived.inertia,
        iterations: result.iterations,
      }),
    });
  },

  check: () => {
    const { points, centroids, seed, trueK, converged, elbowK, lane } = get();

    const evaluation = evaluate({
      points,
      centroids,
      converged,
      elbowK,
      bestInertiaAtK: bestInertiaAt(seed, trueK, points, centroids.length),
      bestInertiaAtElbowK: bestInertiaAt(seed, trueK, points, elbowK),
    });

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
    const { points: current } = get();
    const centroids = STARTING_FLAGS.map((flag) => ({ ...flag }));
    const points = assignPoints(current, centroids);

    set({
      ...derive(points, centroids),
      iteration: 0,
      converged: false,
      lastShift: null,
      assignmentStale: false,
      selectedFlag: 0,
      failure: null,
      won: false,
      lastEvaluation: null,
      whyCard: whyCardFor({ kind: "reset" }),
    });
  },

  startRound: (round) => set(freshRound(round)),
  newRound: () => set(freshRound(get().round + 1)),
}));

export { MAX_K, MIN_K, ROUNDS };
export type { Centroid, ElbowPoint, Evaluation, Point };
