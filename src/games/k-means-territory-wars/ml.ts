import type { NamedFailure } from "@/engine/types";
import { clamp, gaussian, seededRandom } from "@/lib/utils";

/**
 * K-Means Territory Wars — the real machine learning.
 *
 * Per the spec this is pure-JS k-means, no ML library. Villages are points,
 * flags are centroids, and the game is the two-step loop that defines the
 * algorithm:
 *
 *   assign  — every village joins its nearest flag
 *   update  — every flag moves to the mean of the villages that joined it
 *
 * Those are separate functions here, and separate buttons in the UI, because the
 * core intuition names them explicitly: "centroids converge by alternating
 * assign/update steps." Fusing them into one "step" would hide the thing the
 * game exists to show.
 *
 * ── The trap this game is built around ──────────────────────────────────────
 * Inertia (within-cluster sum of squares) is the live metric, and it *always*
 * falls as you add flags — at k = n it reaches exactly zero and means nothing.
 * So "minimise inertia" is a gameable objective, in the same way that "maximise
 * training accuracy" was gameable in Sort-It Arcade.
 *
 * The guard is the elbow: `elbowCurve` runs a proper multi-restart solve at every
 * k and `elbowKFor` finds the kink with the standard kneedle construction. A
 * player who drives inertia down by spamming flags is diagnosed with a bad k,
 * and the failure copy says why with real numbers.
 */

export interface Point {
  id: string;
  /** Field coordinates, both normalised to 0–1. */
  x: number;
  y: number;
  /** Index of the flag this village has joined; null before the first assign. */
  clusterId: number | null;
  /** Which blob it was generated from. Never shown — this is unsupervised. */
  trueCluster: number;
}

export interface Centroid {
  id: number;
  x: number;
  y: number;
}

// ── Tuning ─────────────────────────────────────────────────────────────────

export const MIN_K = 1;
export const MAX_K = 8;

/** Villages per blob. */
export const POINTS_PER_BLOB = 55;

/** Gaussian spread of a blob, in field units. */
export const BLOB_SPREAD = 0.055;
/** Blob centres sit on a ring this far from the middle of the field. */
export const RING_RADIUS = 0.31;

/** Restarts used by the reference solver. */
export const REFERENCE_RESTARTS = 12;
export const MAX_ITERATIONS = 60;

/** A centroid that moves less than this is considered settled. */
export const CONVERGENCE_EPSILON = 1e-4;

/**
 * How much score each step away from the elbow costs. At 0.25, being one flag
 * out caps the score at 0.75 — below the win bar. Choosing k correctly is the
 * lesson, so it is not optional.
 */
export const K_FITNESS_DECAY = 0.25;

/**
 * Player inertia this many times worse than a multi-restart solve at the same k
 * means the flags settled in a bad local optimum.
 */
export const LOCAL_MINIMUM_RATIO = 1.15;

export const WIN_SCORE = 0.8;

// ── The generative process ─────────────────────────────────────────────────

/**
 * Scatter villages in `trueK` compact blobs on a ring.
 *
 * A ring guarantees the blobs are well separated whatever `trueK` is, which is
 * what makes the elbow crisp. If the blobs overlapped, the "right" k would be
 * genuinely ambiguous and the game would be punishing players for a judgement
 * call rather than teaching one.
 */
export function generateVillages(
  seed: number,
  trueK: number,
): { points: Point[]; trueK: number } {
  const random = seededRandom(seed);
  const points: Point[] = [];
  // Rotate the whole ring per seed so rounds don't look identical.
  const phase = random() * Math.PI * 2;

  for (let blob = 0; blob < trueK; blob += 1) {
    const angle = phase + (blob / trueK) * Math.PI * 2;
    const cx = 0.5 + RING_RADIUS * Math.cos(angle);
    const cy = 0.5 + RING_RADIUS * Math.sin(angle);

    for (let i = 0; i < POINTS_PER_BLOB; i += 1) {
      points.push({
        id: `v-${blob}-${i}`,
        x: clamp(cx + gaussian(random) * BLOB_SPREAD, 0.02, 0.98),
        y: clamp(cy + gaussian(random) * BLOB_SPREAD, 0.02, 0.98),
        clusterId: null,
        trueCluster: blob,
      });
    }
  }

  return { points, trueK };
}

/**
 * The rounds this game ships — curated, not arbitrary.
 *
 * Every entry is screened so that all four verdicts are genuinely reachable:
 * the elbow recovers the true blob count, the correct k wins, too many flags
 * lands on "Bad k" *while producing lower inertia*, and there exist starting
 * layouts that reach a local minimum and that strand a flag.
 *
 * That last requirement is the awkward one. Well-separated blobs are what make
 * the elbow crisp, but they also make k-means robust to a bad start — so the two
 * goals pull against each other, and only some seeds satisfy both. Of 60
 * candidates, 55 worked at three blobs, 42 at four, and just 14 at five.
 *
 * Seeds are distinct across rounds on purpose: the store memoises elbow curves,
 * and reusing a seed with a different blob count would serve a cached curve for
 * the wrong map.
 */
export const ROUNDS = [
  { seed: 3001, trueK: 3 },
  { seed: 3003, trueK: 4 },
  { seed: 3005, trueK: 3 },
  { seed: 3002, trueK: 5 },
  { seed: 3007, trueK: 4 },
  { seed: 3004, trueK: 5 },
] as const;

// ── The algorithm ──────────────────────────────────────────────────────────

function squaredDistance(
  a: { x: number; y: number },
  b: { x: number; y: number },
): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

/** Index of the nearest centroid, or null when there are none. */
export function nearestCentroid(
  point: { x: number; y: number },
  centroids: Centroid[],
): number | null {
  if (centroids.length === 0) return null;

  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < centroids.length; index += 1) {
    const distance = squaredDistance(point, centroids[index]!);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  }
  return bestIndex;
}

/**
 * ASSIGN STEP — every village joins its nearest flag.
 * Half of the k-means loop, and half of the player's mechanic.
 */
export function assignPoints(points: Point[], centroids: Centroid[]): Point[] {
  return points.map((point) => ({
    ...point,
    clusterId: nearestCentroid(point, centroids),
  }));
}

/**
 * UPDATE STEP — every flag moves to the mean of the villages that joined it.
 *
 * A flag with no villages is left exactly where it is rather than being quietly
 * re-seeded. Real implementations often re-seed; here the stranded flag is the
 * point — it's a named failure the player needs to see.
 */
export function updateCentroids(
  points: Point[],
  centroids: Centroid[],
): Centroid[] {
  const sums = centroids.map(() => ({ x: 0, y: 0, count: 0 }));

  for (const point of points) {
    if (point.clusterId === null) continue;
    const bucket = sums[point.clusterId];
    if (!bucket) continue;
    bucket.x += point.x;
    bucket.y += point.y;
    bucket.count += 1;
  }

  return centroids.map((centroid, index) => {
    const bucket = sums[index]!;
    if (bucket.count === 0) return centroid;
    return {
      ...centroid,
      x: bucket.x / bucket.count,
      y: bucket.y / bucket.count,
    };
  });
}

/** How far the furthest flag moved between two configurations. */
export function largestShift(before: Centroid[], after: Centroid[]): number {
  let largest = 0;
  for (let index = 0; index < before.length; index += 1) {
    const a = before[index];
    const b = after[index];
    if (!a || !b) continue;
    largest = Math.max(largest, Math.hypot(a.x - b.x, a.y - b.y));
  }
  return largest;
}

/** Villages per flag, by centroid index. */
export function clusterSizes(points: Point[], centroids: Centroid[]): number[] {
  const sizes = centroids.map(() => 0);
  for (const point of points) {
    if (point.clusterId === null) continue;
    if (sizes[point.clusterId] !== undefined) sizes[point.clusterId]! += 1;
  }
  return sizes;
}

/**
 * Inertia of the CURRENT assignment — what the readout shows.
 *
 * Uses each village's stored `clusterId`, so moving a flag without pressing
 * Assign leaves this stale. That is deliberate and visible: it's how the player
 * learns that assignment and update are distinct steps.
 */
export function inertiaOfAssignment(
  points: Point[],
  centroids: Centroid[],
): number {
  let total = 0;
  for (const point of points) {
    if (point.clusterId === null) continue;
    const centroid = centroids[point.clusterId];
    if (!centroid) continue;
    total += squaredDistance(point, centroid);
  }
  return total;
}

/**
 * Inertia implied by a set of centroids, assigning every village to its nearest.
 * This is the objective the solver minimises and the elbow curve plots.
 */
export function inertiaOfCentroids(
  points: Point[],
  centroids: Centroid[],
): number {
  if (centroids.length === 0) return Number.POSITIVE_INFINITY;
  let total = 0;
  for (const point of points) {
    const index = nearestCentroid(point, centroids)!;
    total += squaredDistance(point, centroids[index]!);
  }
  return total;
}

// ── Reference solver ───────────────────────────────────────────────────────

/** k-means++ seeding: spread the initial flags out, proportional to distance². */
export function kMeansPlusPlusInit(
  points: Point[],
  k: number,
  random: () => number,
): Centroid[] {
  if (points.length === 0 || k <= 0) return [];

  const first = points[Math.floor(random() * points.length)]!;
  const chosen: Centroid[] = [{ id: 0, x: first.x, y: first.y }];

  while (chosen.length < k) {
    const distances = points.map((point) => {
      const index = nearestCentroid(point, chosen)!;
      return squaredDistance(point, chosen[index]!);
    });
    const total = distances.reduce((sum, d) => sum + d, 0);

    if (total <= 0) {
      // Degenerate (all points identical) — fall back to an arbitrary pick.
      const fallback = points[Math.floor(random() * points.length)]!;
      chosen.push({ id: chosen.length, x: fallback.x, y: fallback.y });
      continue;
    }

    let target = random() * total;
    let picked = points.length - 1;
    for (let index = 0; index < distances.length; index += 1) {
      target -= distances[index]!;
      if (target <= 0) {
        picked = index;
        break;
      }
    }
    const point = points[picked]!;
    chosen.push({ id: chosen.length, x: point.x, y: point.y });
  }

  return chosen;
}

export interface SolveResult {
  centroids: Centroid[];
  inertia: number;
  iterations: number;
}

/** Run the assign/update loop to convergence from a given starting set. */
export function runToConvergence(
  points: Point[],
  start: Centroid[],
  maxIterations = MAX_ITERATIONS,
): SolveResult {
  let centroids = start.map((c) => ({ ...c }));
  let iterations = 0;

  for (let step = 0; step < maxIterations; step += 1) {
    const assigned = assignPoints(points, centroids);
    const moved = updateCentroids(assigned, centroids);
    iterations += 1;
    const shift = largestShift(centroids, moved);
    centroids = moved;
    if (shift < CONVERGENCE_EPSILON) break;
  }

  return {
    centroids,
    inertia: inertiaOfCentroids(points, centroids),
    iterations,
  };
}

/**
 * Best-of-N k-means++ restarts. This is the reference the player is measured
 * against — both for "did you converge well" and for the elbow curve.
 */
export function solveKMeans(
  points: Point[],
  k: number,
  { restarts = REFERENCE_RESTARTS, seed = 1 }: { restarts?: number; seed?: number } = {},
): SolveResult {
  const random = seededRandom(seed + k * 7919);
  let best: SolveResult | null = null;

  for (let attempt = 0; attempt < restarts; attempt += 1) {
    const result = runToConvergence(
      points,
      kMeansPlusPlusInit(points, k, random),
    );
    if (!best || result.inertia < best.inertia) best = result;
  }

  return best ?? { centroids: [], inertia: Number.POSITIVE_INFINITY, iterations: 0 };
}

// ── Choosing k ─────────────────────────────────────────────────────────────

export interface ElbowPoint {
  k: number;
  inertia: number;
}

/** Best achievable inertia at every k from MIN_K to maxK. */
export function elbowCurve(points: Point[], maxK = MAX_K, seed = 1): ElbowPoint[] {
  const curve: ElbowPoint[] = [];
  for (let k = MIN_K; k <= maxK; k += 1) {
    curve.push({ k, inertia: solveKMeans(points, k, { seed }).inertia });
  }
  return curve;
}

/**
 * The elbow: the k after the largest *relative* drop in inertia.
 *
 * Equivalently, the largest single drop in log-inertia — adding the flag that
 * finds a genuinely separate group cuts inertia by a large factor, while adding
 * one that merely splits an existing group shaves a few percent.
 *
 * ── Why not kneedle ─────────────────────────────────────────────────────────
 * The textbook construction (normalise the curve into a unit box, take the point
 * furthest from the diagonal) was tried first and is wrong here. On five
 * well-separated blobs it reported k=3 on all twelve test seeds: inertia
 * declines steadily through k=4 before falling off a cliff at k=5, and measuring
 * distance from a diagonal spanning k=1..8 flattens that cliff. Absolute
 * distances mislead when the curve spans an order of magnitude; ratios don't.
 *
 * This is a heuristic, not an oracle. It works here because `generateVillages`
 * produces well-separated blobs, and the shipped seeds are screened to confirm
 * it recovers the true number of them.
 */
export function elbowKFor(curve: ElbowPoint[]): number {
  if (curve.length === 0) return MIN_K;
  if (curve.length === 1) return curve[0]!.k;

  let bestK = curve[0]!.k;
  let bestRatio = -Infinity;

  for (let index = 0; index < curve.length - 1; index += 1) {
    const current = curve[index]!;
    const next = curve[index + 1]!;

    // A zero-inertia step means k has reached the point count; adding flags past
    // that is meaningless, so stop rather than dividing by zero.
    if (next.inertia <= 0) break;

    const ratio = current.inertia / next.inertia;
    if (ratio > bestRatio) {
      bestRatio = ratio;
      bestK = next.k;
    }
  }

  return bestK;
}

// ── Evaluation ─────────────────────────────────────────────────────────────

export type Outcome =
  | "win"
  | "empty-cluster"
  | "bad-k"
  | "local-minimum"
  | "not-converged"
  | "near-miss";

export interface Evaluation {
  k: number;
  inertia: number;
  bestInertiaAtK: number;
  elbowK: number;
  emptyClusters: number[];
  clusterSizes: number[];
  converged: boolean;
  /** best/player inertia, in (0, 1]. 1 means as good as the reference solve. */
  convergenceQuality: number;
  /** 1 at the elbow, decaying by K_FITNESS_DECAY per flag away from it. */
  kFitness: number;
  score: number;
  outcome: Outcome;
  failure: NamedFailure | null;
}

export interface EvaluateInput {
  points: Point[];
  centroids: Centroid[];
  converged: boolean;
  elbowK: number;
  /** Best inertia at the player's k, from the reference solver. */
  bestInertiaAtK: number;
  /**
   * Best inertia at the ELBOW k. Needed for the too-few-flags message: a player
   * who converged perfectly with two flags has inertia equal to the best at two
   * flags, so comparing those two numbers produces "stuck at 8.92 versus 8.92
   * achievable" — true, and useless. The number that teaches is what the right
   * k would have reached.
   */
  bestInertiaAtElbowK?: number;
}

/**
 * Score the board and name the failure — honestly.
 *
 * Ordering is diagnostic, not arbitrary. An empty cluster is checked first
 * because it's unambiguous and it invalidates the other numbers. A bad k is
 * checked before a local minimum because when k is wrong, comparing against the
 * best solve *at that wrong k* would praise a well-converged wrong answer.
 */
export function evaluate({
  points,
  centroids,
  converged,
  elbowK,
  bestInertiaAtK,
  bestInertiaAtElbowK,
}: EvaluateInput): Evaluation {
  const k = centroids.length;
  const inertia = inertiaOfAssignment(points, centroids);
  const sizes = clusterSizes(points, centroids);
  const emptyClusters = sizes
    .map((size, index) => (size === 0 ? index : -1))
    .filter((index) => index >= 0);

  // Guard: with nothing assigned there is nothing to judge. Without this, a
  // board that has never been assigned reports every cluster as empty and an
  // inertia of 0 — a confident verdict built on no data.
  const assignedCount = points.filter((point) => point.clusterId !== null).length;
  if (k === 0 || assignedCount === 0) {
    return {
      k,
      inertia: 0,
      bestInertiaAtK,
      elbowK,
      emptyClusters: [],
      clusterSizes: sizes,
      converged: false,
      convergenceQuality: 0,
      kFitness: 0,
      score: 0,
      outcome: "not-converged",
      failure: null,
    };
  }

  const convergenceQuality =
    inertia > 0 ? clamp(bestInertiaAtK / inertia, 0, 1) : 1;
  const kFitness = Math.max(0, 1 - K_FITNESS_DECAY * Math.abs(k - elbowK));
  const score = clamp(convergenceQuality * kFitness, 0, 1);

  const round2 = (value: number) => value.toFixed(2);

  let outcome: Outcome;
  let failure: NamedFailure | null = null;

  if (emptyClusters.length > 0) {
    outcome = "empty-cluster";
    const labels = emptyClusters.map((index) => `#${index + 1}`).join(", ");
    failure = {
      name: "Empty cluster",
      detail: `Flag ${labels} owns 0 of ${points.length} villages, so it has no mean to move to and just sits there. That's ${k} flags doing the work of ${k - emptyClusters.length}.`,
    };
  } else if (!converged) {
    outcome = "not-converged";
  } else if (k !== elbowK) {
    outcome = "bad-k";
    const tooMany = k > elbowK;
    failure = {
      name: "Bad k",
      detail: tooMany
        ? `Inertia ${round2(inertia)} looks good, but you used ${k} flags where the elbow is at ${elbowK}. Inertia ALWAYS falls as you add flags — at k=${points.length} it would be 0 and tell you nothing. You split real clusters in half.`
        : `${k} flags can't cover ${elbowK} groups. Inertia ${round2(inertia)} is the best ${k} flags can do here${
            bestInertiaAtElbowK !== undefined
              ? `, where ${elbowK} flags reach ${round2(bestInertiaAtElbowK)}`
              : ""
          }. At least one flag is straddling two separate clusters, serving both badly.`,
    };
  } else if (convergenceQuality < 1 / LOCAL_MINIMUM_RATIO) {
    outcome = "local-minimum";
    failure = {
      name: "Local minimum",
      detail: `Your flags settled at inertia ${round2(inertia)}, but the same ${k} flags reach ${round2(bestInertiaAtK)} from a better starting layout. The algorithm converged — just not to the best answer. Where you place flags decides which optimum you land in.`,
    };
  } else if (score >= WIN_SCORE) {
    outcome = "win";
  } else {
    outcome = "near-miss";
  }

  return {
    k,
    inertia,
    bestInertiaAtK,
    elbowK,
    emptyClusters,
    clusterSizes: sizes,
    converged,
    convergenceQuality,
    kFitness,
    score,
    outcome,
    failure,
  };
}

// ── Reveal the math ────────────────────────────────────────────────────────

export const MATH_EQUATION = String.raw`\textbf{assign:}\quad c_i = \arg\min_{j} \left\lVert x_i - \mu_j \right\rVert^2
\qquad
\textbf{update:}\quad \mu_j = \frac{1}{|C_j|} \sum_{i \in C_j} x_i
\\[1.2em]
\textbf{inertia:}\quad J = \sum_{i=1}^{n} \left\lVert x_i - \mu_{c_i} \right\rVert^2`;

export const MATH_CODE = `// ASSIGN — every village joins its nearest flag.
export function assignPoints(points, centroids) {
  return points.map((point) => ({
    ...point,
    clusterId: nearestCentroid(point, centroids),
  }));
}

// UPDATE — every flag moves to the mean of its villages.
// A flag with no villages stays put: that stranded flag is a
// failure mode you're meant to see, not something to paper over.
export function updateCentroids(points, centroids) {
  const sums = centroids.map(() => ({ x: 0, y: 0, count: 0 }));
  for (const point of points) {
    if (point.clusterId === null) continue;
    const bucket = sums[point.clusterId];
    bucket.x += point.x;
    bucket.y += point.y;
    bucket.count += 1;
  }
  return centroids.map((centroid, index) => {
    const bucket = sums[index];
    if (bucket.count === 0) return centroid;
    return { ...centroid, x: bucket.x / bucket.count, y: bucket.y / bucket.count };
  });
}

// INERTIA — the live metric. Squared distance from every village
// to the flag it joined.
export function inertiaOfAssignment(points, centroids) {
  let total = 0;
  for (const point of points) {
    const centroid = centroids[point.clusterId];
    total += (point.x - centroid.x) ** 2 + (point.y - centroid.y) ** 2;
  }
  return total;
}`;

export const MATH_NOTES = `x_i is a village, μ_j is flag j, and C_j is the set of villages that joined it. Assign, then update, then repeat: when no flag moves any further you have converged. Inertia only ever falls as you add flags — with one flag per village it hits exactly 0 — so a low number is not automatically a good answer. That is what the elbow chart is for: it plots the best possible inertia at every k, and the kink is where extra flags stop buying you real structure.`;
