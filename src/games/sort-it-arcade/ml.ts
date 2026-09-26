import type { NamedFailure } from "@/engine/types";
import { clamp, seededRandom } from "@/lib/utils";

/**
 * Sort-It Arcade — the real machine learning.
 *
 * Per the spec this is geometric classification, so there's no TensorFlow here:
 * a decision boundary is a function `f(x)` and a point is classified by which
 * side of it the point falls on. Every number the game shows is computed by the
 * functions below. Nothing is hardcoded, nothing is cosmetic.
 *
 * ── Why label noise is the load-bearing detail ───────────────────────────────
 * The lesson is "a simpler boundary that misses a few points often generalises
 * better than a perfect-but-complex one." That is only TRUE when the data
 * contains irreducible noise. On clean, separable data a wiggly boundary
 * generalises perfectly well, and the game would be teaching a falsehood while
 * appearing to work.
 *
 * So `generatePoints` flips `NOISE_RATE` of the labels. A high-capacity boundary
 * can then chase those flipped points and score brilliantly on the training set
 * while getting *worse* on held-out data — which is exactly what overfitting is,
 * and why the failure can be named honestly.
 *
 * The best achievable accuracy is therefore about `1 - NOISE_RATE` (≈92%), not
 * 100%. A player who reaches 95% on the training set has, by construction,
 * fitted noise.
 */

export type BoundaryType = "line" | "curve" | "wiggle";

export interface Point {
  id: string;
  /** Field coordinates, both normalised to 0–1. */
  x: number;
  y: number;
  /** Ground truth. 0 = class A (blue), 1 = class B (orange). */
  label: 0 | 1;
  /** Which side of the current boundary the point fell on. */
  predictedSide: 0 | 1;
}

// ── Tuning ─────────────────────────────────────────────────────────────────

/**
 * Model capacity per boundary type, as a parameter count.
 *
 * `curve` gets 5 knots so its x positions (0, ¼, ½, ¾, 1) land exactly on the
 * true boundary's peak and trough — the mid-complexity model is *meant* to be
 * capable of a good fit, otherwise "the middle option wins" would be luck.
 */
export const KNOT_COUNTS: Record<BoundaryType, number> = {
  line: 2,
  curve: 5,
  wiggle: 25,
};

/** Fraction of labels flipped. This is what makes overfitting real. */
export const NOISE_RATE = 0.08;

/**
 * Training points: the ones the player can see and fit.
 *
 * Sized for low variance rather than for a tidy screen. At 120 points the fitted
 * mid-capacity curve was itself flagged as overfitting on roughly a third of
 * samples, which would make the game unwinnable at the capacity it is teaching
 * you to choose.
 */
export const TRAIN_SIZE = 200;
/**
 * Held-out points. Deliberately larger than the training set: it is never fitted
 * to, and a big sample makes the generalisation estimate stable enough to accuse
 * the player of overfitting and mean it.
 */
export const TEST_SIZE = 400;

/**
 * Half-width of the band points are scattered into around the true boundary.
 *
 * Points are sampled *near* the boundary rather than uniformly over the field.
 * Uniform sampling wastes most points in trivially-separable corners, which
 * leaves so little headroom that a flat line scores 83% and the lesson has
 * nowhere to live.
 */
export const SPREAD = 0.25;

/** Amplitude of the true boundary's sine. */
export const AMPLITUDE = 0.2;

/** The simplest boundary pays no penalty. */
export const MIN_PARAMS = 2;
/**
 * Regularization pressure per extra parameter. The 5-knot curve pays 0.024; the
 * 25-knot wiggle pays 0.184.
 *
 * Sized so the wiggle CANNOT win on score even at the best training accuracy its
 * capacity can reach. At 0.006 it paid 0.138, which left it scoring 0.802
 * against a 0.80 win bar — a browser session found a route where the
 * over-parameterised boundary won by two thousandths, rewarding exactly the
 * behaviour the game is built to punish. `sort-it-arcade.test.ts` now asserts
 * that headroom directly rather than trusting the arithmetic.
 */
export const PENALTY_PER_EXTRA_PARAM = 0.008;

/**
 * Train − test gap that justifies the word "overfitting".
 *
 * A 0.12 threshold made the wiggle's verdict depend on the player's route to it.
 * At 0.10, on the shipped rounds: the flat-start wiggle's gap runs 0.105–0.182,
 * so "Fit it for me" on a fresh wiggle is always named, as are about 95% of
 * fits from random starts (gaps 0.06–0.23). Refining a fitted curve is the
 * exception — win with the curve, switch to Wiggle, Fit — because the extra
 * knots only add bends to a boundary that already follows the truth. That gap
 * runs 0.02–0.12: named on two rounds, a near-miss on the other six, where
 * held-out accuracy falls by under 3 points or even rises. Calling those
 * overfitting would be false; the 0.184 penalty still sinks them, and the
 * near-miss card says "fewer parameters". The curve, over 400 fits per round
 * from near-truth and random starts, stays at or below 0.09 on seven rounds and
 * crosses 0.10 on 3 random starts in 200 on the eighth. `sort-it-arcade.test.ts`
 * re-checks that route-robustness, because the curve being branded "memorised
 * the noise" is the one false verdict this game must never hand out routinely.
 */
export const OVERFIT_GAP = 0.1;
/**
 * How close to what the optimizer reaches from the current knots a boundary
 * must be to count as "fitted" — i.e. nothing nearby improves it much.
 */
export const CAPACITY_TOLERANCE = 0.03;
/** Score needed to clear the round. */
export const WIN_SCORE = 0.8;

// ── The generative process ─────────────────────────────────────────────────

/**
 * The true class boundary: one full sine period, up then down.
 *
 * A straight line cannot follow "up then down" at all, so 2 parameters genuinely
 * underfit. 5 parameters can trace it. 25 parameters have capacity to spare —
 * and spend it on noise.
 */
export function trueBoundary(x: number): number {
  return 0.5 + AMPLITUDE * Math.sin(2 * Math.PI * clamp(x, 0, 1));
}

/**
 * Sample a labelled point set, deterministic for a given seed so tests and
 * leaderboards agree.
 *
 * `x` is uniform; the offset from the true boundary is triangular on
 * ±`SPREAD` (the sum of two uniforms), so points cluster near the boundary where
 * the decision is actually difficult. The clean label is the sign of that offset,
 * which makes the true boundary exactly `trueBoundary(x)` — then `NOISE_RATE` of
 * the labels are flipped.
 *
 * With AMPLITUDE + SPREAD = 0.45, y stays inside (0.05, 0.95), so no clamping is
 * needed and the label always agrees with the geometry.
 */
export function generatePoints(
  seed: number,
  idPrefix: string,
  count: number,
): Point[] {
  const random = seededRandom(seed);
  const points: Point[] = [];

  for (let index = 0; index < count; index += 1) {
    const x = random();
    // Triangular on [-SPREAD, SPREAD], peaked at 0.
    const offset = (random() + random() - 1) * SPREAD;
    const y = trueBoundary(x) + offset;

    const clean: 0 | 1 = offset > 0 ? 1 : 0;
    // Irreducible noise — see the module docblock.
    const label: 0 | 1 = random() < NOISE_RATE ? ((1 - clean) as 0 | 1) : clean;

    points.push({
      id: `${idPrefix}-${index}`,
      x,
      y,
      label,
      predictedSide: label,
    });
  }

  return points;
}

/** The training sample for a round. */
export function generateTrainPoints(seed: number): Point[] {
  return generatePoints(seed, "train", TRAIN_SIZE);
}

/** The held-out sample for a round. Never shown until the player checks. */
export function generateTestPoints(seed: number): Point[] {
  // Offset the seed so the two sets are independent draws, not the same points.
  return generatePoints(seed + 9973, "test", TEST_SIZE);
}

/**
 * The rounds this game ships — curated, not arbitrary.
 *
 * Random sampling can produce a draw where the mid-capacity curve happens to
 * generalise badly, or where the over-capacity wiggle happens not to overfit. On
 * such a round the game would teach the opposite of its lesson, or be unwinnable
 * at the capacity it is nudging you toward.
 *
 * Every seed here is screened against the full lesson, not just the verdicts:
 * the line is never accused of overfitting, the curve wins, and the wiggle
 * scores *higher on training data* while scoring *lower on held-out data* and
 * losing on score. That middle condition matters — a wiggle that doesn't even
 * look better on the training set isn't tempting, and the round teaches nothing.
 *
 * About a quarter of all seeds fail this bar, almost always because the wiggle
 * doesn't overfit hard enough.
 *
 * Two more screens were added after the lesson was found to hold on only one
 * route through it:
 *
 * - **A straight line must not be able to clear the round**, checked by an
 *   exhaustive search rather than the optimizer's own line. 20260801 (round 1)
 *   and 20260805 were dropped: a hand-placed line reached 80% on the first and
 *   79.5% on the second, where "Fit it for me" found only 77.5% and 79%.
 * - **The curve must win from any reasonable start**, not only from a flat
 *   line. 20260808 was dropped: its held-out sample is intrinsically harder
 *   than its training sample, so about 40% of ordinary 5-parameter fits were
 *   branded "memorised the noise" while scoring below the true boundary itself.
 *
 * `sort-it-arcade.test.ts` re-checks every seed here against all three on each
 * run, so a change to the tuning constants can't silently break the lesson.
 */
export const ROUND_SEEDS = [
  20260826, 20260802, 20260804, 20260838, 20260823, 20260809, 20260810,
  20260811,
] as const;

/** Seed for a 1-based round number, cycling through the vetted list. */
export function seedForRound(round: number): number {
  const index = (Math.max(1, Math.floor(round)) - 1) % ROUND_SEEDS.length;
  return ROUND_SEEDS[index]!;
}

// ── The model ──────────────────────────────────────────────────────────────

/**
 * Boundary height at `x`: piecewise-linear interpolation between knots placed at
 * evenly spaced x positions. Two knots give a straight line; more knots give
 * more bends, which is precisely more capacity.
 */
export function boundaryAt(params: number[], x: number): number {
  const count = params.length;
  if (count === 0) return 0.5;
  if (count === 1) return params[0]!;

  const position = clamp(x, 0, 1);
  const span = 1 / (count - 1);
  const segment = Math.min(count - 2, Math.floor(position / span));
  const t = (position - segment * span) / span;

  const left = params[segment]!;
  const right = params[segment + 1]!;
  return left + (right - left) * t;
}

/** Which side of the boundary a point falls on. This IS the classifier. */
export function classifyPoint(
  params: number[],
  point: { x: number; y: number },
): 0 | 1 {
  return point.y >= boundaryAt(params, point.x) ? 1 : 0;
}

export function accuracyOf(points: Point[], params: number[]): number {
  if (points.length === 0) return 0;
  let correct = 0;
  for (const point of points) {
    if (classifyPoint(params, point) === point.label) correct += 1;
  }
  return correct / points.length;
}

/** The spec's "complexity = param count". */
export function complexityCostOf(params: number[]): number {
  return params.length;
}

/**
 * Regularization pressure. Charging for parameters is what makes the wiggly
 * boundary a bad deal even before the held-out set is revealed.
 */
export function penaltyFor(params: number[]): number {
  return (
    PENALTY_PER_EXTRA_PARAM * Math.max(0, params.length - MIN_PARAMS)
  );
}

/** Score = accuracy − complexity penalty (spec's scoring rule). */
export function scoreFor(trainAccuracy: number, params: number[]): number {
  return clamp(trainAccuracy - penaltyFor(params), 0, 1);
}

/**
 * Resample a boundary onto a different number of knots, preserving its shape.
 * Used when the player changes capacity so their work carries over.
 */
export function resampleParams(params: number[], count: number): number[] {
  if (count <= 1) return [boundaryAt(params, 0.5)];
  return Array.from({ length: count }, (_, index) =>
    boundaryAt(params, index / (count - 1)),
  );
}

// ── Fitting ────────────────────────────────────────────────────────────────

/**
 * Fit the boundary to `points` by coordinate descent: sweep each knot over a
 * grid of heights, keep whichever maximises training accuracy, repeat.
 *
 * This is a real (greedy) optimizer over a real objective, and it is the same
 * thing the player does by dragging — which is the pedagogy contract's second
 * point made executable. The code lane calls it directly via `api.autoFit()`.
 *
 * It maximises *training* accuracy with no regularization, exactly like an
 * unregularised learner. Given capacity it will fit the noise, which is how the
 * game earns the right to say "you overfit".
 *
 * ── Only recount what a knot can move ────────────────────────────────────────
 * A point's side depends on just the two knots either side of it (see
 * `boundaryAt`), so trying a new height for knot i can only re-classify the
 * points in segments i − 1 and i. Those are recounted; the rest keep their
 * count. The result is the same integer, divided by the same n, as a full
 * `accuracyOf` — every comparison, and so every fitted knot, is identical — but
 * a 25-knot fit touches about a twelfth of the points per candidate. It was the
 * difference between a 70 ms and a few-ms "Fit it for me" on a wiggle.
 */
export function fitKnots(
  points: Point[],
  params: number[],
  { passes = 3, steps = 40 }: { passes?: number; steps?: number } = {},
): number[] {
  const fitted = [...params];
  if (points.length === 0) return fitted;

  const movedBy = pointsEachKnotMoves(points, fitted.length);
  const correctAmong = (subset: readonly Point[]) => {
    let correct = 0;
    for (const point of subset) {
      if (classifyPoint(fitted, point) === point.label) correct += 1;
    }
    return correct;
  };

  for (let pass = 0; pass < passes; pass += 1) {
    let improved = false;

    for (let index = 0; index < fitted.length; index += 1) {
      const original = fitted[index]!;
      const affected = movedBy[index]!;
      // Correct points this knot cannot change, whatever height it takes.
      const fixed = correctAmong(points) - correctAmong(affected);
      let bestHeight = original;
      let bestAccuracy = (fixed + correctAmong(affected)) / points.length;

      for (let step = 0; step <= steps; step += 1) {
        const candidate = step / steps;
        fitted[index] = candidate;
        const candidateAccuracy =
          (fixed + correctAmong(affected)) / points.length;
        if (candidateAccuracy > bestAccuracy) {
          bestAccuracy = candidateAccuracy;
          bestHeight = candidate;
        }
      }

      fitted[index] = bestHeight;
      if (bestHeight !== original) improved = true;
    }

    // Converged — further passes would change nothing.
    if (!improved) break;
  }

  return fitted;
}

/**
 * For each knot, the points whose side it can change: those in the segment to
 * its left and the segment to its right. The segment arithmetic is copied from
 * `boundaryAt` exactly, so a point at a knot's x lands in the same segment the
 * classifier puts it in.
 */
function pointsEachKnotMoves(points: Point[], knotCount: number): Point[][] {
  // One or zero knots: the boundary is flat, and every knot moves every point.
  if (knotCount < 2) return Array.from({ length: knotCount }, () => points);

  const span = 1 / (knotCount - 1);
  const movedBy: Point[][] = Array.from({ length: knotCount }, () => []);
  for (const point of points) {
    const position = clamp(point.x, 0, 1);
    const segment = Math.min(knotCount - 2, Math.floor(position / span));
    movedBy[segment]!.push(point);
    movedBy[segment + 1]!.push(point);
  }
  return movedBy;
}

/**
 * Training accuracy the greedy fit reaches from one particular start.
 *
 * NOT the ceiling of the model class — coordinate descent stops at the first
 * local optimum it finds, so this depends on where it began. Use
 * `capacityCeiling` for any claim about what a parameter count "can do".
 */
export function capacityOf(points: Point[], params: number[]): number {
  return accuracyOf(points, fitKnots(points, params));
}

// ── The capacity ceiling ───────────────────────────────────────────────────

/**
 * Grid for the exhaustive straight-line search: both knots swept over 0–1 in
 * steps of 1/200, so every one of the 201² lines is scored.
 */
export const LINE_GRID_STEPS = 200;

/**
 * Seeded random starts added to the flat start when estimating the ceiling of a
 * curve or a wiggle. Greedy coordinate descent from a single start is route-
 * dependent — from "every handle at the bottom" it can stall 12 points below
 * what the same five parameters reach from a flat line — so the ceiling takes
 * the best of several routes rather than trusting one.
 */
export const CEILING_RESTARTS = 4;

/**
 * The exact best a straight line can do, over every line on the grid.
 *
 * Scoring all 201² lines point by point is 8M comparisons — 130 ms in V8 on a
 * click. It doesn't need to be. Fix the left knot and raise the right one: the
 * boundary only rises, so each point is predicted "above" (class 1) up to some
 * right-knot height and "below" after it — one flip. Find each point's flip,
 * and a running sum gives the count of correct points for every right-knot
 * height at once. Roughly 150k steps instead of 8M.
 *
 * The flip is located with the classifier's own arithmetic
 * (`left + (right − left)·x`, exactly `boundaryAt` for two knots), not with a
 * rearranged formula, so the counts match `accuracyOf` on every line — the
 * test suite checks this against the brute force.
 */
function bestLineAccuracy(points: Point[]): number {
  const n = points.length;
  const steps = LINE_GRID_STEPS;
  // `edges` holds +1/−1 steps; its running sum at j is the number of correct
  // points when the right knot sits at j / steps.
  const edges = new Int32Array(steps + 2);
  let best = 0;

  for (let a = 0; a <= steps; a += 1) {
    const left = a / steps;
    edges.fill(0);

    for (const point of points) {
      const x = clamp(point.x, 0, 1);
      const above = (j: number) =>
        point.y >= left + (j / steps - left) * x;

      // Last right-knot index still predicting class 1 (−1 if none). Start
      // from the algebraic estimate, then settle it with the real comparison;
      // `above` is monotone in j, and the walk is bounded by the grid.
      let last =
        x > 0
          ? Math.floor(steps * (left + (point.y - left) / x))
          : above(0)
            ? steps
            : -1;
      last = Math.max(-1, Math.min(steps, last));
      while (last < steps && above(last + 1)) last += 1;
      while (last >= 0 && !above(last)) last -= 1;

      if (point.label === 1) {
        // Correct while predicted above: j = 0..last.
        if (last >= 0) {
          edges[0]! += 1;
          edges[last + 1]! -= 1;
        }
      } else if (last < steps) {
        // Correct once predicted below: j = last+1..steps.
        edges[last + 1]! += 1;
        edges[steps + 1]! -= 1;
      }
    }

    let correct = 0;
    for (let j = 0; j <= steps; j += 1) {
      correct += edges[j]!;
      if (correct > best) best = correct;
    }
  }

  return best / n;
}

/** Best greedy fit over the flat start and `CEILING_RESTARTS` seeded ones. */
function multiStartCeiling(points: Point[], knotCount: number): number {
  let best = capacityOf(
    points,
    Array.from({ length: knotCount }, () => 0.5),
  );
  // Seeded on the knot count so the estimate is deterministic per capacity.
  const random = seededRandom(0x5eed + knotCount * 7919);
  for (let attempt = 0; attempt < CEILING_RESTARTS; attempt += 1) {
    const start = Array.from({ length: knotCount }, () => random());
    best = Math.max(best, capacityOf(points, start));
  }
  return best;
}

/**
 * Ceilings are deterministic per (sample, knot count) and cost a few
 * milliseconds each — several times that on a slow phone — so they are
 * computed once. The key is the sample's content, not
 * its array identity: the store re-derives `trainPoints` on every handle move.
 * Bounded so a test run that screens hundreds of samples can't grow it forever.
 */
const ceilingCache = new Map<string, number>();
const CEILING_CACHE_LIMIT = 64;

function sampleKey(points: Point[], knotCount: number): string {
  let key = `${knotCount}|${points.length}`;
  for (const point of points) key += `|${point.x},${point.y},${point.label}`;
  return key;
}

/**
 * The best training accuracy `knotCount` knots can reach on these points,
 * computed without reference to any player — the model class's own ceiling.
 * Memoised per (sample, knot count).
 */
export function independentCeiling(points: Point[], knotCount: number): number {
  if (points.length === 0 || knotCount < MIN_PARAMS) return 0;

  const key = sampleKey(points, knotCount);
  const cached = ceilingCache.get(key);
  if (cached !== undefined) return cached;

  const ceiling =
    knotCount === MIN_PARAMS
      ? bestLineAccuracy(points)
      : multiStartCeiling(points, knotCount);
  if (ceilingCache.size >= CEILING_CACHE_LIMIT) {
    const oldest = ceilingCache.keys().next().value;
    if (oldest !== undefined) ceilingCache.delete(oldest);
  }
  ceilingCache.set(key, ceiling);
  return ceiling;
}

/**
 * The best training accuracy `params.length` knots can reach on these points,
 * INDEPENDENT of where the player's knots happen to be.
 *
 * This is the number behind "X% is the best N parameters can do", so it has to
 * be a property of the model class, not of the player's route. The line is
 * searched exhaustively; curves and wiggles take the best of several greedy
 * starts. The player's own fit is always included as one more candidate, so the
 * ceiling can never sit below a number the player can already see.
 */
export function capacityCeiling(points: Point[], params: number[]): number {
  return Math.max(
    independentCeiling(points, params.length),
    capacityOf(points, params),
  );
}

// ── Evaluation ─────────────────────────────────────────────────────────────

export type Outcome = "win" | "overfit" | "underfit" | "near-miss";

export interface Evaluation {
  trainAccuracy: number;
  testAccuracy: number;
  penalty: number;
  score: number;
  /** trainAccuracy − testAccuracy. The overfitting signal. */
  generalizationGap: number;
  paramCount: number;
  /**
   * Best training accuracy achievable at this parameter count, whatever the
   * starting knots — see `capacityCeiling`.
   */
  bestAtThisComplexity: number;
  /** The score `bestAtThisComplexity` would earn: this capacity at its best. */
  ceilingScore: number;
  outcome: Outcome;
  failure: NamedFailure | null;
}

/**
 * Score a round and name the failure — honestly.
 *
 * The ordering matters, and so does the strictness:
 *
 * - **Overfitting** requires both a train/test gap above `OVERFIT_GAP` *and*
 *   spare capacity to have memorised with. A 2-parameter straight line cannot
 *   memorise 200 points; when it shows a gap that is sampling variance, not
 *   overfitting, and saying otherwise would teach the wrong word. Never claimed
 *   on training accuracy alone.
 * - **Underfitting** needs two separate facts. The boundary must be *fitted* —
 *   within `CAPACITY_TOLERANCE` of what the optimizer reaches from where it
 *   stands, so a flat, unfitted line is never accused. And the model class must
 *   be *too simple* — even its ceiling, found without reference to the player's
 *   route (`capacityCeiling`), sorts too few points to clear the round before
 *   any penalty is charged. Judging the second fact from the player's own start
 *   was the old bug: a curve stalled in a greedy local optimum was told "75% is
 *   the best 5 parameters can do" on a round where five parameters reach 88%.
 *   That boundary is unfitted, not underfit, so it returns `near-miss`, and the
 *   copy quotes the real ceiling.
 *
 *   "Too simple" is an accuracy test, not a score test. The wiggle's ceiling
 *   *score* is under the bar too, but only because of its penalty: 25
 *   parameters are too costly, never too simple. And it replaces a fixed
 *   accuracy cut-off (78%) that sat exactly on the best line on three rounds,
 *   so the most carefully placed line there escaped the name.
 */
export function evaluate(
  trainPoints: Point[],
  testPoints: Point[],
  params: number[],
): Evaluation {
  const trainAccuracy = accuracyOf(trainPoints, params);
  const testAccuracy = accuracyOf(testPoints, params);
  const penalty = penaltyFor(params);
  const score = scoreFor(trainAccuracy, params);
  const generalizationGap = trainAccuracy - testAccuracy;
  // Where the optimizer gets to from the player's own knots: the test for
  // "already fitted" (a local optimum nothing nearby improves on).
  const fromHere = capacityOf(trainPoints, params);
  // The class's ceiling, whatever the start — the test for "too simple".
  const bestAtThisComplexity = Math.max(
    independentCeiling(trainPoints, params.length),
    fromHere,
  );
  // What the model class would score at its very best. Below the bar means no
  // amount of adjusting at this capacity can clear the round.
  const ceilingScore = scoreFor(bestAtThisComplexity, params);
  // Even the class's best fit sorts too few points, penalty aside.
  const tooSimple = bestAtThisComplexity < WIN_SCORE;
  const fitted = trainAccuracy >= fromHere - CAPACITY_TOLERANCE;

  const asPercent = (value: number) => `${Math.round(value * 100)}%`;

  let outcome: Outcome;
  let failure: NamedFailure | null = null;

  const hasSpareCapacity = params.length > MIN_PARAMS;

  if (hasSpareCapacity && generalizationGap >= OVERFIT_GAP) {
    outcome = "overfit";
    failure = {
      name: "Overfitting",
      detail: `${asPercent(trainAccuracy)} on the ${trainPoints.length} points you fitted, ${asPercent(
        testAccuracy,
      )} on ${testPoints.length} points it never saw. ${params.length} parameters memorised the noise.`,
    };
  } else if (fitted && tooSimple) {
    outcome = "underfit";
    failure = {
      name: "Underfitting",
      detail: `${asPercent(trainAccuracy)} train, and ${asPercent(
        bestAtThisComplexity,
      )} is the best ${params.length} parameters can do on this data. The boundary is too simple to follow the real one.`,
    };
  } else if (score >= WIN_SCORE) {
    outcome = "win";
  } else {
    outcome = "near-miss";
  }

  return {
    trainAccuracy,
    testAccuracy,
    penalty,
    score,
    generalizationGap,
    paramCount: params.length,
    bestAtThisComplexity,
    ceilingScore,
    outcome,
    failure,
  };
}

/**
 * The real implementation, as a string, for the ƒ Math drawer.
 *
 * Kept beside the code it describes so the two can't drift — `ml-verifier`
 * fails a game whose math reveal disagrees with its implementation.
 */
export const MATH_CODE = `// The classifier: which side of the boundary is the point on?
export function classifyPoint(params, point) {
  return point.y >= boundaryAt(params, point.x) ? 1 : 0;
}

// The boundary: piecewise-linear between knots at evenly spaced x.
// More knots = more bends = more capacity.
export function boundaryAt(params, x) {
  const position = clamp(x, 0, 1);
  const span = 1 / (params.length - 1);
  const segment = Math.min(params.length - 2, Math.floor(position / span));
  const t = (position - segment * span) / span;
  const left = params[segment];
  const right = params[segment + 1];
  return left + (right - left) * t;
}

// Accuracy: the fraction the classifier gets right. Recomputed every time
// you move a handle — never stored, never faked.
export function accuracyOf(points, params) {
  let correct = 0;
  for (const point of points) {
    if (classifyPoint(params, point) === point.label) correct += 1;
  }
  return correct / points.length;
}

// Scoring: accuracy minus a charge per extra parameter, clamped to 0..1.
export function penaltyFor(params) {
  return ${PENALTY_PER_EXTRA_PARAM} * Math.max(0, params.length - ${MIN_PARAMS});
}
export function scoreFor(trainAccuracy, params) {
  return clamp(trainAccuracy - penaltyFor(params), 0, 1);
}`;

export const MATH_EQUATION = String.raw`\hat{y}(x, y) = \mathbb{1}\!\left[\, y \ge f(x) \,\right]
\qquad
\text{score} = \underbrace{\frac{1}{n}\sum_{i=1}^{n} \mathbb{1}\!\left[\hat{y}_i = y_i\right]}_{\text{accuracy}} \;-\; \underbrace{\lambda \max\!\left(0,\; |\theta| - 2\right)}_{\text{complexity penalty}}`;

export const MATH_NOTES = `f(x) is the boundary you drag — a piecewise-linear function through your knots. Any point above it is predicted orange, below it blue. |θ| is how many knots you're using, and λ = ${PENALTY_PER_EXTRA_PARAM} is the price of each one beyond the two a straight line needs. About ${Math.round(
  NOISE_RATE * 100,
)}% of these labels are deliberately wrong, so roughly ${Math.round(
  (1 - NOISE_RATE) * 100,
)}% is the best any boundary can honestly do. Beat that on the training set and you are fitting noise.`;
