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
 * The best achievable accuracy is therefore about `1 - NOISE_RATE` (≈88%), not
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
 * Observed wiggle gaps run 0.117–0.148 depending on where the greedy fit starts,
 * so a 0.12 threshold made the verdict depend on the player's route to it. 0.10
 * sits below the whole range while staying well above the curve's 0.037–0.070.
 */
export const OVERFIT_GAP = 0.1;
/**
 * Train accuracy below which a maxed-out model is genuinely underfitting.
 * Sits between what 2 parameters can reach (~75%) and what 5 can (~90%).
 */
export const UNDERFIT_ACCURACY = 0.78;
/** How close to its own ceiling a boundary must be to count as "fitted". */
export const CAPACITY_TOLERANCE = 0.03;
/** Score needed to clear the round. */
export const WIN_SCORE = 0.8;

// ── The generative process ─────────────────────────────────────────────────

/**
 * The true class boundary: one full sine period, up then down.
 *
 * A straight line cannot follow "up then down" at all, so 2 parameters genuinely
 * underfit. 5 parameters can trace it. 17 parameters have capacity to spare —
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
 * doesn't overfit hard enough. `sort-it-arcade.test.ts` re-checks every seed
 * here on each run, so a change to the tuning constants can't silently break
 * the lesson.
 */
export const ROUND_SEEDS = [
  20260801, 20260802, 20260804, 20260805, 20260808, 20260809, 20260810,
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
 */
export function fitKnots(
  points: Point[],
  params: number[],
  { passes = 3, steps = 40 }: { passes?: number; steps?: number } = {},
): number[] {
  const fitted = [...params];
  if (points.length === 0) return fitted;

  for (let pass = 0; pass < passes; pass += 1) {
    let improved = false;

    for (let index = 0; index < fitted.length; index += 1) {
      const original = fitted[index]!;
      let bestHeight = original;
      let bestAccuracy = accuracyOf(points, fitted);

      for (let step = 0; step <= steps; step += 1) {
        const candidate = step / steps;
        fitted[index] = candidate;
        const candidateAccuracy = accuracyOf(points, fitted);
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

/** Best training accuracy this parameter count can reach on these points. */
export function capacityOf(points: Point[], params: number[]): number {
  return accuracyOf(points, fitKnots(points, params));
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
  /** Best training accuracy achievable at this parameter count. */
  bestAtThisComplexity: number;
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
 *   memorise 120 points; when it shows a gap that is sampling variance, not
 *   overfitting, and saying otherwise would teach the wrong word. Never claimed
 *   on training accuracy alone.
 * - **Underfitting** is claimed only when the boundary is already at the ceiling
 *   of what its parameter count can achieve *and* that ceiling is poor. A
 *   badly-placed 17-knot boundary is an unfitted model, not an underfitting one,
 *   so it returns `near-miss` with a nudge to keep adjusting.
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
  const bestAtThisComplexity = capacityOf(trainPoints, params);

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
  } else if (
    trainAccuracy < UNDERFIT_ACCURACY &&
    trainAccuracy >= bestAtThisComplexity - CAPACITY_TOLERANCE
  ) {
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
