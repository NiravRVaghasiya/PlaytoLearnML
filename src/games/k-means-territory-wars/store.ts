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
  /**
   * The last verdict. Cleared by any action that changes what it judged —
   * the flags, the memberships, or whether the board is converged — so the
   * Score tile never shows a number for a board that no longer exists.
   */
  lastEvaluation: Evaluation | null;
  /** The CURRENT board was scored a win. Cleared with `lastEvaluation`. */
  won: boolean;
  /**
   * The last board on this map whose clear was recorded (see `boardKey`), and
   * which lane's call recorded it. This is what lets `check()` be called
   * twice — or the starter snippet run twice, re-scattering and re-converging
   * to the same clustering — without counting the same clear twice, while a
   * code-lane `api.check()` can still earn the third star on a board first
   * scored from the rail. A new map starts with neither.
   */
  clearedBoard: string | null;
  clearedFrom: Lane | null;
  lane: Lane;

  // ── actions ──────────────────────────────────────────────────────────
  setLane: (lane: Lane) => void;
  selectFlag: (index: number | null) => void;

  /** Drop a flag. This is the player choosing k. */
  addFlag: (x?: number, y?: number) => void;
  removeFlag: (index: number) => void;
  /**
   * Move a flag. This is the player choosing the initialization. A
   * non-integer index or a non-finite coordinate is ignored rather than stored:
   * a NaN flag silently drops out of every nearest-flag test.
   */
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

  /**
   * Score the board. `source` is the lane the call came from — the code lane's
   * `api.check()` passes "code" — so the third star follows the action rather
   * than whichever tab is showing. The rail's button is visual in either tab.
   */
  check: (source?: Lane) => Evaluation;
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

/**
 * A board's identity for "was this clear already counted?": its flag positions,
 * in any order. A win is only ever scored on a converged, freshly assigned board
 * with no empty flag, where every flag sits exactly at its villages' mean — so
 * two wins with the same flags are the same clustering, however they were
 * reached.
 */
function boardKey(centroids: Centroid[]): string {
  return centroids
    .map((centroid) => `${centroid.x.toFixed(6)},${centroid.y.toFixed(6)}`)
    .sort()
    .join(" ");
}

type Board = Pick<
  KMeansState,
  "points" | "centroids" | "converged" | "assignmentStale"
>;

/**
 * Whether a loop action left alone everything a verdict reads: the flags (a
 * sub-epsilon shift is "didn't move", the convention `update` uses), every
 * village's membership, and whether the board counts as converged and fresh.
 * Pressing Run to convergence on a settled map changes none of them, and must
 * not take a cleared map's Next button away.
 */
function sameBoard(before: Board, after: Board): boolean {
  return (
    before.converged === after.converged &&
    before.assignmentStale === after.assignmentStale &&
    before.centroids.length === after.centroids.length &&
    largestShift(before.centroids, after.centroids) < CONVERGENCE_EPSILON &&
    before.points.every(
      (point, index) => point.clusterId === after.points[index]?.clusterId,
    )
  );
}

/** The verdict fields to drop when the board they judged has changed. */
function verdictAfter(before: Board, after: Board) {
  return sameBoard(before, after)
    ? {}
    : { failure: null, lastEvaluation: null, won: false };
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
    clearedBoard: null,
    clearedFrom: null,
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
    if (
      (x !== undefined && !Number.isFinite(x)) ||
      (y !== undefined && !Number.isFinite(y))
    ) {
      return;
    }

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
      lastEvaluation: null,
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
    if (
      !Number.isInteger(index) ||
      centroids.length <= MIN_K ||
      index < 0 ||
      index >= centroids.length
    ) {
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
      lastEvaluation: null,
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
    if (!Number.isInteger(index) || index < 0 || index >= centroids.length) {
      return;
    }
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;

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
      lastEvaluation: null,
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
    const before = get();
    const assigned = assignPoints(before.points, before.centroids);
    const derived = derive(assigned, before.centroids);

    set({
      ...derived,
      assignmentStale: false,
      ...verdictAfter(before, {
        ...derived,
        converged: before.converged,
        assignmentStale: false,
      }),
      whyCard: whyCardFor({
        kind: "assigned",
        inertia: derived.inertia,
        sizes: derived.sizes,
      }),
    });
  },

  update: () => {
    const before = get();
    const { points, centroids, iteration, assignmentStale: wasStale } = before;
    const moved = updateCentroids(points, centroids);
    const shift = largestShift(centroids, moved);
    const derived = derive(points, moved);
    // Convergence is a fixed point of the WHOLE loop: memberships fresh for
    // these flags, and flags already at their members' means. An update on
    // stale memberships moves nothing the second time you press it — it is
    // re-averaging the same old villages — so "nothing moved" only means
    // "converged" when the assignment was current.
    const converged = !wasStale && shift < CONVERGENCE_EPSILON;
    // Flags moved (or memberships were already behind), so the colouring is
    // out of date until the next assign. A sub-epsilon shift is "didn't move",
    // by the same convention `runToConvergence` uses — otherwise a converged
    // board would be flagged stale by floating-point dust, and `converged` and
    // `assignmentStale` could both be true.
    const assignmentStale = wasStale || !(shift < CONVERGENCE_EPSILON);

    set({
      ...derived,
      iteration: iteration + 1,
      lastShift: shift,
      assignmentStale,
      converged,
      ...verdictAfter(before, { ...derived, converged, assignmentStale }),
      whyCard: whyCardFor({
        kind: "updated",
        inertia: derived.inertia,
        shift,
        converged,
        stale: wasStale,
      }),
    });
  },

  step: () => {
    const before = get();
    const { points, centroids, iteration } = before;
    const assigned = assignPoints(points, centroids);
    const moved = updateCentroids(assigned, centroids);
    const shift = largestShift(centroids, moved);
    // Re-assign after moving so what's on screen matches where the flags are.
    const settled = assignPoints(assigned, moved);
    const derived = derive(settled, moved);
    const converged = shift < CONVERGENCE_EPSILON;

    set({
      ...derived,
      iteration: iteration + 1,
      lastShift: shift,
      assignmentStale: false,
      converged,
      ...verdictAfter(before, {
        ...derived,
        converged,
        assignmentStale: false,
      }),
      whyCard: whyCardFor({
        kind: "stepped",
        inertia: derived.inertia,
        shift,
        iteration: iteration + 1,
        converged,
      }),
    });
  },

  settle: () => {
    const before = get();
    const { points, centroids, iteration } = before;
    const result = runToConvergence(points, centroids);
    const settled = assignPoints(points, result.centroids);
    const derived = derive(settled, result.centroids);

    set({
      ...derived,
      iteration: iteration + result.iterations,
      lastShift: 0,
      assignmentStale: false,
      converged: true,
      ...verdictAfter(before, {
        ...derived,
        converged: true,
        assignmentStale: false,
      }),
      whyCard: whyCardFor({
        kind: "settled",
        inertia: derived.inertia,
        iterations: result.iterations,
      }),
    });
  },

  check: (source = "visual") => {
    const {
      points,
      centroids,
      seed,
      trueK,
      converged,
      assignmentStale,
      elbowK,
      clearedBoard,
      clearedFrom,
    } = get();

    const evaluation = evaluate({
      points,
      centroids,
      // Belt and braces: every flag-changing action already clears
      // `converged`, but a stale board is never a converged one.
      converged: converged && !assignmentStale,
      assignmentStale,
      elbowK,
      bestInertiaAtK: bestInertiaAt(seed, trueK, points, centroids.length),
      bestInertiaAtElbowK: bestInertiaAt(seed, trueK, points, elbowK),
    });

    const win = evaluation.outcome === "win";
    // Record a clear once per board — unless this call is the first from the
    // code lane, which is worth recording for the third star. "Per board", not
    // "per edit": the same clustering reached again is the same clear.
    const board = boardKey(centroids);
    const record =
      win &&
      (board !== clearedBoard ||
        (source === "code" && clearedFrom !== "code"));

    set({
      failure: evaluation.failure,
      won: win,
      ...(record ? { clearedBoard: board, clearedFrom: source } : {}),
      lastEvaluation: evaluation,
      whyCard: whyCardFor({ kind: "checked", evaluation }),
    });

    if (record) {
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
