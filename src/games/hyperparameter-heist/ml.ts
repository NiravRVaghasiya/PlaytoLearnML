import type { NamedFailure } from "@/engine/types";
import { clamp, seededRandom } from "@/lib/utils";

/**
 * Hyperparameter Heist — the search maths.
 *
 * The claim this game makes is Bergstra & Bengio's: under a fixed budget, random
 * search beats grid search, and it beats it *because* most hyperparameters barely
 * matter. A grid with k levels per dial spends k^d trials but only ever sees k
 * distinct values of the one dial that counts. Random search with n trials sees n
 * distinct values of every dial. When the objective is dominated by one or two
 * dials, that is an enormous difference in resolution where it matters.
 *
 * For that claim to be taught rather than asserted, the objective has to actually
 * have low effective dimensionality — and the comparison has to be measured over
 * many runs, because any single run is luck. Both are done here: the surface below
 * puts most of its range in one dial (`dialInfluence` measures how much, rather
 * than this comment asserting a figure that could drift), and the tests simulate
 * hundreds of runs of each strategy.
 *
 * ── Deviations from the spec, flagged ───────────────────────────────────────
 * 1. The spec asks for a "real objective surface in TF.js". This is a closed-form
 *    surrogate instead, and deliberately so. Two reasons. Verifying "random beats
 *    grid" needs hundreds of simulated runs, which is only affordable if a trial
 *    costs microseconds rather than a training run. And the surface is *shown* to
 *    the player at the end; a noisy per-trial retrain would make the optimum
 *    region ill-defined exactly where the lesson needs it crisp. The shape is not
 *    invented — see `objectiveAt` for what each term models.
 * 2. The Bayesian mode is a Gaussian process with an RBF kernel and Expected
 *    Improvement, written out in plain arithmetic below. TF.js has no Cholesky or
 *    general linear solve, and a 16×16 exact decomposition is twenty lines and
 *    inspectable. Approximating a GP with a neural surrogate to keep TF.js in the
 *    loop would be worse maths for no pedagogical gain.
 */

// ── The dials ─────────────────────────────────────────────────────────────

export type DialScale = "log10" | "log2" | "linear";

export interface HyperParam {
  name: string;
  /** Short label for the surface axes. */
  short: string;
  min: number;
  max: number;
  scale: DialScale;
  /** How the value reads on the dial face. */
  format: (value: number) => string;
  description: string;
}

/**
 * Four dials, and only one of them really moves the needle.
 *
 * That is not a trick — it is the empirical finding the whole game rests on. Real
 * hyperparameter response surfaces are dominated by a handful of dials, and the
 * rest are close enough to flat that spending grid budget on them is waste.
 */
export const DIALS: readonly HyperParam[] = [
  {
    name: "learning rate",
    short: "lr",
    min: 1e-4,
    max: 1,
    scale: "log10",
    format: (value) => value.toPrecision(2),
    description:
      "Step size. Too small and nothing moves; too large and it diverges. The dial that decides the run.",
  },
  {
    name: "momentum",
    short: "mom",
    min: 0,
    max: 0.99,
    scale: "linear",
    format: (value) => value.toFixed(2),
    description:
      "How much of the previous step carries over. Interacts with the learning rate.",
  },
  {
    name: "batch size",
    short: "batch",
    min: 8,
    max: 256,
    scale: "log2",
    format: (value) => String(Math.round(value)),
    description: "Samples per gradient step. Mostly a throughput choice here.",
  },
  {
    name: "weight decay",
    short: "decay",
    min: 1e-6,
    max: 1e-1,
    scale: "log10",
    format: (value) => value.toPrecision(2),
    description: "L2 penalty on the weights. Only bites at the top of its range.",
  },
] as const;

export const DIAL_COUNT = DIALS.length;

/** Indices, so the code reads as the thing it means. */
export const LEARNING_RATE = 0;
export const MOMENTUM = 1;
export const BATCH_SIZE = 2;
export const WEIGHT_DECAY = 3;

/**
 * A point in search space, as a unit hypercube coordinate.
 *
 * Every strategy works in [0,1]^4 and converts out only to evaluate or display.
 * Searching in raw units would make the learning rate's four decades incomparable
 * to momentum's zero-to-one, and a grid would be uneven in exactly the dimension
 * that matters most.
 */
export type UnitPoint = number[];

export function toRealValue(dial: HyperParam, unit: number): number {
  const u = clamp(unit, 0, 1);
  switch (dial.scale) {
    case "log10": {
      const lo = Math.log10(dial.min);
      return 10 ** (lo + u * (Math.log10(dial.max) - lo));
    }
    case "log2": {
      const lo = Math.log2(dial.min);
      return 2 ** (lo + u * (Math.log2(dial.max) - lo));
    }
    case "linear":
      return dial.min + u * (dial.max - dial.min);
  }
}

export function toUnitValue(dial: HyperParam, value: number): number {
  switch (dial.scale) {
    case "log10": {
      const lo = Math.log10(dial.min);
      return clamp(
        (Math.log10(value) - lo) / (Math.log10(dial.max) - lo),
        0,
        1,
      );
    }
    case "log2": {
      const lo = Math.log2(dial.min);
      return clamp((Math.log2(value) - lo) / (Math.log2(dial.max) - lo), 0, 1);
    }
    case "linear":
      return clamp((value - dial.min) / (dial.max - dial.min), 0, 1);
  }
}

export function toRealPoint(unit: UnitPoint): number[] {
  return DIALS.map((dial, index) => toRealValue(dial, unit[index] ?? 0.5));
}

export function describePoint(unit: UnitPoint): string {
  return DIALS.map(
    (dial, index) => `${dial.short} ${dial.format(toRealValue(dial, unit[index] ?? 0.5))}`,
  ).join(" · ");
}

// ── The objective ─────────────────────────────────────────────────────────

export const BASE_SCORE = 0.48;
/** How much of the objective's range each dial controls. */
export const LR_WEIGHT = 0.42;
export const MOMENTUM_WEIGHT = 0.07;
export const BATCH_WEIGHT = 0.02;
export const DECAY_PENALTY = 0.04;

/**
 * Validation accuracy as a function of the four dials.
 *
 * Every term models something real:
 *
 *   - The learning rate peak is Gaussian in log10(lr), because a rate ten times
 *     too small and one ten times too large fail about equally badly. It is the
 *     dominant term at 0.42 of the range.
 *   - Its optimum SHIFTS with momentum. Effective step size is roughly
 *     lr / (1 − momentum), so more momentum wants a smaller rate. This is why the
 *     surface is not separable, and why reasoning about one dial at a time — which
 *     is what a coarse grid amounts to — goes wrong.
 *   - Momentum has its own mild bump around 0.85, worth 0.07.
 *   - Batch size is nearly flat, with a slight preference for smaller batches
 *     (more updates per epoch), worth 0.02.
 *   - Weight decay does nothing until the top of its range, where it starts
 *     costing accuracy.
 *
 * Deterministic, so the same dials always give the same answer. A player who
 * re-tries a combination should not get a different number — that would teach that
 * search is about luck rather than about coverage.
 */
export function objectiveAt(unit: UnitPoint): number {
  const [lrUnit = 0.5, momentum = 0.5, batchUnit = 0.5, decayUnit = 0.5] = unit;

  const logLr = Math.log10(toRealValue(DIALS[LEARNING_RATE]!, lrUnit));
  const momentumValue = toRealValue(DIALS[MOMENTUM]!, momentum);

  // Effective-step reasoning: the faster the momentum, the smaller the best rate.
  //
  // The constants place the peak deliberately. An earlier version put the optimum
  // at log10(lr) ≈ -1.9, which is almost exactly the midpoint of a four-decade
  // dial — so centring every dial scored 94.7% and opened the safe on the first
  // try without searching for anything. The peak now sits near -2.5, which is
  // away from the centre AND away from both levels a two-level grid samples
  // (-3.0 and -1.0), so neither doing nothing nor being thorough is enough.
  const optimalLogLr = -2.4 - 0.3 * momentumValue;
  const lrTerm =
    LR_WEIGHT * Math.exp(-0.5 * ((logLr - optimalLogLr) / 0.42) ** 2);

  const momentumTerm =
    MOMENTUM_WEIGHT * Math.exp(-0.5 * ((momentumValue - 0.85) / 0.35) ** 2);

  // Smaller batches help slightly. batchUnit 0 is the smallest batch.
  const batchTerm = BATCH_WEIGHT * (1 - batchUnit);

  // Only the top ~40% of the decay range does damage.
  const decayTerm = -DECAY_PENALTY * Math.max(0, (decayUnit - 0.6) / 0.4);

  return clamp(
    BASE_SCORE + lrTerm + momentumTerm + batchTerm + decayTerm,
    0,
    1,
  );
}

/**
 * The objective a try has to reach to open the safe.
 *
 * Not the optimum — the true peak sits near 0.99, which the tests confirm by a
 * dense sweep. The bar is set below it so that "opening the safe" means landing
 * in the optimum REGION, as the spec asks, rather than on one exact point.
 */
export const CRACK_THRESHOLD = 0.9;

export interface Trial {
  /** Unit-cube coordinates of the dials tried. */
  params: UnitPoint;
  objectiveValue: number;
  /** 1-based order in the run. */
  index: number;
  /** Which strategy produced it. */
  source: Strategy | "manual";
}

export const BUDGET = 16;

/** Warm, warmer, hot — the safecracking readout (spec: temperature). */
export type Temperature = "freezing" | "cold" | "warm" | "hot" | "cracked";

export function temperatureOf(objective: number): Temperature {
  if (objective >= CRACK_THRESHOLD) return "cracked";
  if (objective >= 0.84) return "hot";
  if (objective >= 0.72) return "warm";
  if (objective >= 0.6) return "cold";
  return "freezing";
}

// ── Strategies ────────────────────────────────────────────────────────────

export type Strategy = "grid" | "random" | "bayesian";

export const STRATEGIES: readonly Strategy[] = ["grid", "random", "bayesian"];

/**
 * Whether a value names a strategy.
 *
 * The code lane needs this before calling `runStrategy`, which would otherwise
 * treat any unknown string — `"Random"`, say — as Bayesian and label the result
 * with the typo.
 */
export function isStrategy(value: unknown): value is Strategy {
  return (STRATEGIES as readonly unknown[]).includes(value);
}

export const STRATEGY_LABELS: Record<Strategy, string> = {
  grid: "Grid",
  random: "Random",
  bayesian: "Bayesian",
};

/**
 * Grid search: the coarsest full-factorial grid that fits the budget.
 *
 * `levels` is the largest k with k^d ≤ budget, which for four dials and sixteen
 * tries is exactly two. Levels sit at the quarter and three-quarter points rather
 * than at the extremes, which is the standard choice and the kinder one — corners
 * of a search space are usually the worst places to spend a trial.
 *
 * The thing to notice: this spends all sixteen tries and still only ever sees TWO
 * learning rates.
 */
export function gridLevels(budget: number, dials = DIAL_COUNT): number {
  let levels = 1;
  while ((levels + 1) ** dials <= budget) levels += 1;
  return levels;
}

export function gridPoints(budget: number): UnitPoint[] {
  const levels = gridLevels(budget);
  const coordinates = Array.from(
    { length: levels },
    (_, index) => (index + 0.5) / levels,
  );

  let points: UnitPoint[] = [[]];
  for (let dial = 0; dial < DIAL_COUNT; dial += 1) {
    const next: UnitPoint[] = [];
    for (const point of points) {
      for (const coordinate of coordinates) next.push([...point, coordinate]);
    }
    points = next;
  }
  return points.slice(0, budget);
}

export function randomPoints(budget: number, seed: number): UnitPoint[] {
  const random = seededRandom(seed);
  return Array.from({ length: budget }, () =>
    Array.from({ length: DIAL_COUNT }, () => random()),
  );
}

/** Van der Corput radical inverse, for the Halton candidate set. */
function radicalInverse(index: number, base: number): number {
  let result = 0;
  let f = 1 / base;
  let i = index;
  while (i > 0) {
    result += f * (i % base);
    i = Math.floor(i / base);
    f /= base;
  }
  return result;
}

const HALTON_BASES = [2, 3, 5, 7];

/**
 * A deterministic, evenly spread candidate set for maximising the acquisition
 * function.
 *
 * Halton rather than uniform random: the acquisition maximiser should not itself
 * be a source of run-to-run variation, or a comparison between strategies would be
 * measuring the maximiser's luck alongside the strategy's merit.
 */
export function haltonPoints(count: number, skip = 1): UnitPoint[] {
  return Array.from({ length: count }, (_, index) =>
    HALTON_BASES.slice(0, DIAL_COUNT).map((base) =>
      radicalInverse(index + skip, base),
    ),
  );
}

// ── Gaussian process surrogate ────────────────────────────────────────────

/** RBF lengthscale on the unit cube. */
export const GP_LENGTHSCALE = 0.28;
/** Jitter on the diagonal: numerical, and a stand-in for evaluation noise. */
export const GP_NOISE = 1e-4;

function rbf(a: UnitPoint, b: UnitPoint): number {
  let squared = 0;
  for (let index = 0; index < DIAL_COUNT; index += 1) {
    const delta = (a[index] ?? 0) - (b[index] ?? 0);
    squared += delta * delta;
  }
  return Math.exp(-squared / (2 * GP_LENGTHSCALE * GP_LENGTHSCALE));
}

/** Lower-triangular Cholesky factor. Returns null if not positive definite. */
function cholesky(matrix: number[][]): number[][] | null {
  const size = matrix.length;
  const lower: number[][] = Array.from({ length: size }, () =>
    new Array<number>(size).fill(0),
  );

  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column <= row; column += 1) {
      let sum = matrix[row]![column]!;
      for (let k = 0; k < column; k += 1) {
        sum -= lower[row]![k]! * lower[column]![k]!;
      }
      if (row === column) {
        if (sum <= 0) return null;
        lower[row]![column] = Math.sqrt(sum);
      } else {
        lower[row]![column] = sum / lower[column]![column]!;
      }
    }
  }
  return lower;
}

function forwardSubstitute(lower: number[][], rhs: number[]): number[] {
  const size = lower.length;
  const out = new Array<number>(size).fill(0);
  for (let row = 0; row < size; row += 1) {
    let sum = rhs[row]!;
    for (let k = 0; k < row; k += 1) sum -= lower[row]![k]! * out[k]!;
    out[row] = sum / lower[row]![row]!;
  }
  return out;
}

function backSubstitute(lower: number[][], rhs: number[]): number[] {
  const size = lower.length;
  const out = new Array<number>(size).fill(0);
  for (let row = size - 1; row >= 0; row -= 1) {
    let sum = rhs[row]!;
    for (let k = row + 1; k < size; k += 1) sum -= lower[k]![row]! * out[k]!;
    out[row] = sum / lower[row]![row]!;
  }
  return out;
}

export interface Surrogate {
  /** Posterior mean and standard deviation at a point. */
  predict: (point: UnitPoint) => { mean: number; sd: number };
  /** Best objective observed so far. */
  best: number;
  observations: number;
}

/**
 * Fit a GP to the trials so far.
 *
 * The objectives are standardised before fitting and un-standardised on the way
 * out, which is what keeps a zero-mean prior honest: without it the GP would pull
 * predictions toward zero accuracy in unexplored regions and Expected Improvement
 * would refuse to look anywhere new.
 */
export function fitSurrogate(trials: Trial[]): Surrogate | null {
  const size = trials.length;
  if (size === 0) return null;

  const values = trials.map((trial) => trial.objectiveValue);
  const best = Math.max(...values);
  const mean = values.reduce((total, value) => total + value, 0) / size;
  const variance =
    values.reduce((total, value) => total + (value - mean) ** 2, 0) / size;
  const sd = Math.sqrt(Math.max(variance, 1e-12));

  const standardised = values.map((value) => (value - mean) / sd);
  const points = trials.map((trial) => trial.params);

  const gram: number[][] = points.map((a, row) =>
    points.map((b, column) => rbf(a, b) + (row === column ? GP_NOISE : 0)),
  );

  const lower = cholesky(gram);
  if (lower === null) {
    // Degenerate — duplicate points, most likely. Fall back to nearest observed.
    return {
      best,
      observations: size,
      predict: (point) => {
        let nearest = Number.POSITIVE_INFINITY;
        let nearestValue = mean;
        for (let index = 0; index < size; index += 1) {
          const distance = 1 - rbf(point, points[index]!);
          if (distance < nearest) {
            nearest = distance;
            nearestValue = values[index]!;
          }
        }
        return { mean: nearestValue, sd };
      },
    };
  }

  const alpha = backSubstitute(lower, forwardSubstitute(lower, standardised));

  return {
    best,
    observations: size,
    predict: (point) => {
      const covariance = points.map((observed) => rbf(point, observed));

      let predictedStandard = 0;
      for (let index = 0; index < size; index += 1) {
        predictedStandard += covariance[index]! * alpha[index]!;
      }

      const v = forwardSubstitute(lower, covariance);
      let reduction = 0;
      for (const entry of v) reduction += entry * entry;
      const posteriorVariance = Math.max(1 + GP_NOISE - reduction, 1e-12);

      return {
        mean: predictedStandard * sd + mean,
        sd: Math.sqrt(posteriorVariance) * sd,
      };
    },
  };
}

const standardNormalPdf = (z: number) =>
  Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);

/** Φ, via the Abramowitz–Stegun erf approximation. */
export function standardNormalCdf(z: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const poly =
    t *
    (0.319381530 +
      t *
        (-0.356563782 +
          t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const upper = standardNormalPdf(z) * poly;
  return z >= 0 ? 1 - upper : upper;
}

/** How much better than the incumbent this point is worth exploring for. */
export const EI_EXPLORATION = 0.01;

/**
 * Expected Improvement.
 *
 * The acquisition function the spec asks to surface. It is worth reading as two
 * terms: the first rewards a high predicted mean, the second rewards uncertainty.
 * A point the surrogate is confident is mediocre scores zero on both; a point it
 * knows nothing about scores on the second alone. That is what "learns from prior
 * cracks" actually means — not remembering the good ones, but knowing where it has
 * not looked.
 */
export function expectedImprovement(
  surrogate: Surrogate,
  point: UnitPoint,
): number {
  const { mean, sd } = surrogate.predict(point);
  return improvementFrom(mean, sd, surrogate.best);
}

/**
 * EI from a prediction already in hand.
 *
 * Split out so the acquisition scan can predict each candidate once and use the
 * same numbers for the score and for the suggestion it reports. It used to
 * predict twice per new leader, and the scan is the hot loop of every Bayesian
 * simulation the code lane runs.
 */
function improvementFrom(mean: number, sd: number, best: number): number {
  if (sd <= 1e-9) return 0;
  const gap = mean - best - EI_EXPLORATION;
  const z = gap / sd;
  return gap * standardNormalCdf(z) + sd * standardNormalPdf(z);
}

/** How many seed points Bayesian search spends before it can model anything. */
export const BAYESIAN_SEED_TRIALS = 4;

/** The visual lane's warm-up: the first Halton points, identical on every safe. */
const DEFAULT_WARMUP: readonly UnitPoint[] = haltonPoints(BAYESIAN_SEED_TRIALS, 1);

/**
 * The acquisition maximiser's candidate set, built once.
 *
 * Every hint in the visual lane scanned a freshly generated copy of the same 512
 * points. They are deterministic, so they are computed once and shared; nothing
 * downstream mutates them, and `suggestNext` hands back a copy of the winner.
 */
let sharedCandidates: UnitPoint[] | null = null;
function defaultCandidates(): UnitPoint[] {
  sharedCandidates ??= haltonPoints(512);
  return sharedCandidates;
}

export interface Suggestion {
  point: UnitPoint;
  expectedImprovement: number;
  predictedMean: number;
  predictedSd: number;
}

/**
 * Where Bayesian search wants to look next.
 *
 * With fewer than `BAYESIAN_SEED_TRIALS` observations it returns a warm-up point
 * instead: a surrogate fitted to one or two samples has no information to offer,
 * and pretending otherwise would just be an expensive random guess.
 *
 * The warm-up defaults to the first Halton points, so the visual lane's hint is
 * the same on every safe and matches what "Spend up to … on bayesian" then does.
 * `runStrategy` passes a SEEDED warm-up instead. That matters more than it looks:
 * the warm-up is the only part of Bayesian search that varies from run to run in
 * practice, so with a fixed one, sixty "different" simulated runs were four
 * distinct sequences, and the crack rate they reported was one run's result
 * dressed up as an average.
 */
export function suggestNext(
  trials: Trial[],
  candidates = defaultCandidates(),
  warmup: readonly UnitPoint[] = DEFAULT_WARMUP,
): Suggestion {
  if (trials.length < BAYESIAN_SEED_TRIALS) {
    const point = [...warmup[trials.length]!];
    return {
      point,
      expectedImprovement: Number.NaN,
      predictedMean: Number.NaN,
      predictedSd: Number.NaN,
    };
  }

  const surrogate = fitSurrogate(trials);
  if (surrogate === null) {
    return {
      point: [...candidates[0]!],
      expectedImprovement: Number.NaN,
      predictedMean: Number.NaN,
      predictedSd: Number.NaN,
    };
  }

  // Skip candidates that duplicate something already tried: re-evaluating a
  // deterministic objective learns nothing and wastes a try.
  let best: Suggestion | null = null;
  for (const candidate of candidates) {
    const tooClose = trials.some(
      (trial) => 1 - rbf(trial.params, candidate) < 1e-4,
    );
    if (tooClose) continue;

    const { mean, sd } = surrogate.predict(candidate);
    const ei = improvementFrom(mean, sd, surrogate.best);
    if (best === null || ei > best.expectedImprovement) {
      best = {
        point: candidate,
        expectedImprovement: ei,
        predictedMean: mean,
        predictedSd: sd,
      };
    }
  }

  // A copy: the candidate set can be shared between calls, and a caller that
  // records this point must not be holding a reference into it.
  return best
    ? { ...best, point: [...best.point] }
    : {
        point: [...candidates[0]!],
        expectedImprovement: 0,
        predictedMean: Number.NaN,
        predictedSd: Number.NaN,
      };
}

/** Run a strategy for a whole budget, from scratch. Used by the tests and the code lane. */
export function runStrategy(
  strategy: Strategy,
  budget: number,
  seed: number,
): Trial[] {
  const trials: Trial[] = [];

  const record = (point: UnitPoint) => {
    trials.push({
      params: point,
      objectiveValue: objectiveAt(point),
      index: trials.length + 1,
      source: strategy,
    });
  };

  if (strategy === "grid") {
    for (const point of gridPoints(budget)) record(point);
    return trials;
  }

  if (strategy === "random") {
    for (const point of randomPoints(budget, seed)) record(point);
    return trials;
  }

  // Bayesian: seeded warm-up points, then Expected Improvement, one at a time.
  //
  // The warm-up is drawn from the seed like a random run's first tries, because
  // that is where a real Bayesian optimiser's run-to-run variation comes from.
  // The candidate set stays Halton (shifted by the seed) so the maximiser itself
  // is not a second source of luck — see `haltonPoints`.
  const warmup = randomPoints(BAYESIAN_SEED_TRIALS, seed);
  const candidates = haltonPoints(512, 1 + (Math.abs(seed) % 97));
  while (trials.length < budget) {
    record(suggestNext(trials, candidates, warmup).point);
  }
  return trials;
}

/**
 * The highest-scoring try, ignoring any that did not produce a number.
 *
 * A non-finite reading can only come from bad input, and it must not become the
 * "best": `x > NaN` is always false, so a NaN incumbent could never be displaced
 * and every later genuine crack would go unnoticed.
 */
export function bestTrial(trials: Trial[]): Trial | null {
  return trials.reduce<Trial | null>(
    (best, trial) =>
      Number.isFinite(trial.objectiveValue) &&
      (best === null || trial.objectiveValue > best.objectiveValue)
        ? trial
        : best,
    null,
  );
}

// ── Diagnosing a run ──────────────────────────────────────────────────────

/** How many distinct values of a dial a set of trials actually explored. */
export function distinctValuesTried(
  trials: Trial[],
  dial: number,
  tolerance = 0.02,
): number {
  const seen: number[] = [];
  for (const trial of trials) {
    const value = trial.params[dial] ?? 0.5;
    if (!seen.some((existing) => Math.abs(existing - value) <= tolerance)) {
      seen.push(value);
    }
  }
  return seen.length;
}

/**
 * Which dial the objective is most sensitive to, measured rather than declared.
 *
 * Sweeps each dial across its range with the others held at the midpoint and takes
 * the spread. Used by the "Grid trap" diagnosis so it can name the dial the player
 * under-explored without that name being hardcoded — if the surface is retuned,
 * the diagnosis follows it.
 */
export function dialInfluence(samples = 41): number[] {
  return DIALS.map((_, dial) => {
    let lowest = Number.POSITIVE_INFINITY;
    let highest = Number.NEGATIVE_INFINITY;
    for (let step = 0; step < samples; step += 1) {
      const point = new Array<number>(DIAL_COUNT).fill(0.5);
      point[dial] = step / (samples - 1);
      const value = objectiveAt(point);
      lowest = Math.min(lowest, value);
      highest = Math.max(highest, value);
    }
    return highest - lowest;
  });
}

export function dominantDial(): number {
  const influence = dialInfluence();
  return influence.indexOf(Math.max(...influence));
}

export type Outcome = "cracked" | "grid-trap" | "budget-exhausted" | "searching";

export interface Evaluation {
  outcome: Outcome;
  best: Trial | null;
  triesUsed: number;
  budget: number;
  score: number;
  failure: NamedFailure | null;
}

const percent = (value: number) => `${Math.round(value * 100)}%`;
const points = (value: number) => `${(value * 100).toFixed(1)}%`;
/**
 * An objective DIFFERENCE, in accuracy points. Kept apart from `percent` on
 * purpose: "the learning rate moves the objective by 42 points" and "the learning
 * rate controls 42% of the range" are different claims, and only the first is
 * what `dialInfluence` measures.
 */
const swing = (value: number) => `${Math.round(value * 100)} points`;

/**
 * The score for opening the safe on a given try.
 *
 * Cracking with tries to spare is worth more than cracking on the last turn, per
 * the spec's "fewer tries = higher score": 1.0 on the first try, falling by an
 * equal step per try spent, and never below 0.6 for a crack. Exported so the star
 * criteria can state which try still earns the high-score star, rather than
 * leaving the player to invert this formula.
 */
export function crackScore(crackedOn: number, budget: number): number {
  const efficiency = 1 - (crackedOn - 1) / Math.max(1, budget);
  return clamp(0.6 + 0.4 * efficiency, 0, 1);
}

/** The latest try that still scores at least `threshold` when it opens the safe. */
export function latestCrackScoring(threshold: number, budget: number): number {
  let latest = 0;
  for (let crackedOn = 1; crackedOn <= budget; crackedOn += 1) {
    if (crackScore(crackedOn, budget) >= threshold) latest = crackedOn;
  }
  return latest;
}

/**
 * How often a from-scratch random run on this budget opens the safe.
 *
 * Measured, over many seeds, so that a WhyCard for one unlucky random run can say
 * what random search does ON AVERAGE without quoting a number nobody computed.
 * Random runs cost sixteen closed-form evaluations each, so a thousand of them is
 * a few milliseconds, and the result is cached per budget.
 */
export const CRACK_RATE_RUNS = 1000;
const randomCrackRates = new Map<number, number>();
export function randomCrackRate(budget: number, runs = CRACK_RATE_RUNS): number {
  const key = budget * 100_000 + runs;
  const cached = randomCrackRates.get(key);
  if (cached !== undefined) return cached;
  let cracked = 0;
  for (let seed = 1; seed <= runs; seed += 1) {
    const best = bestTrial(runStrategy("random", budget, seed));
    if (best !== null && best.objectiveValue >= CRACK_THRESHOLD) cracked += 1;
  }
  const rate = cracked / runs;
  randomCrackRates.set(key, rate);
  return rate;
}

/** The best a full grid ever does on this budget. Deterministic, so one run. */
export function gridBest(budget: number): number {
  return bestTrial(runStrategy("grid", budget, 0))?.objectiveValue ?? Number.NaN;
}

/**
 * How few distinct values of the dominant dial counts as having not searched it.
 *
 * A full-factorial grid at two levels tries exactly two, so this catches the grid
 * trap and any manual search that behaved like one.
 */
export const GRID_TRAP_DISTINCT = 3;

/**
 * Score the heist.
 *
 * Cracking with tries to spare is worth more than cracking on the last turn, per
 * the spec's "fewer tries = higher score". A failed run still scores something for
 * how close it got, because "you were 2 points off" and "you never left the
 * freezing zone" are not the same outcome and should not collapse to zero.
 */
export function evaluateRun({
  trials,
  budget,
  finished,
}: {
  trials: Trial[];
  budget: number;
  finished: boolean;
}): Evaluation {
  const best = bestTrial(trials);
  const triesUsed = trials.length;

  if (best === null) {
    return {
      outcome: "searching",
      best: null,
      triesUsed,
      budget,
      score: 0,
      failure: null,
    };
  }

  if (best.objectiveValue >= CRACK_THRESHOLD) {
    // Efficiency: cracking on try 5 of 16 beats cracking on try 16.
    const crackedOn =
      trials.find((trial) => trial.objectiveValue >= CRACK_THRESHOLD)?.index ??
      triesUsed;
    return {
      outcome: "cracked",
      best,
      triesUsed,
      budget,
      score: crackScore(crackedOn, budget),
      failure: null,
    };
  }

  if (!finished) {
    return {
      outcome: "searching",
      best,
      triesUsed,
      budget,
      score: 0,
      failure: null,
    };
  }

  // How far it got, as a fraction of the distance from a coin-flip to the crack.
  const progress = clamp(
    (best.objectiveValue - BASE_SCORE) / (CRACK_THRESHOLD - BASE_SCORE),
    0,
    1,
  );
  const score = clamp(progress * 0.5, 0, 1);

  const dominant = dominantDial();
  const distinct = distinctValuesTried(trials, dominant);
  const dial = DIALS[dominant]!;

  if (distinct <= GRID_TRAP_DISTINCT) {
    const levels = gridLevels(budget);
    return {
      outcome: "grid-trap",
      best,
      triesUsed,
      budget,
      score,
      failure: {
        name: "Grid trap",
        detail: `${triesUsed} tries spent, and only ${distinct} distinct ${
          dial.name
        } value${distinct === 1 ? "" : "s"} among them — best ${points(
          best.objectiveValue,
        )} against the ${percent(CRACK_THRESHOLD)} needed. ${
          dial.name.charAt(0).toUpperCase() + dial.name.slice(1)
        } alone swings this safe's objective by ${swing(
          dialInfluence()[dominant] ?? 0,
        )}, more than any other dial, and you looked at just ${distinct} setting${
          distinct === 1 ? "" : "s"
        } of it. A full grid over ${DIAL_COUNT} dials at ${levels} levels each costs every one of your ${budget} tries and still only ever sees ${levels} values of it. The same ${budget} tries placed at random would have sampled ${budget} different values — the same budget, ${Math.round(
          budget / Math.max(1, levels),
        )} times the resolution where it counts.`,
      },
    };
  }

  return {
    outcome: "budget-exhausted",
    best,
    triesUsed,
    budget,
    score,
    failure: {
      name: "Budget exhausted",
      detail: `${budget} tries gone. Best was ${points(
        best.objectiveValue,
      )} at ${describePoint(best.params)}, short of the ${percent(
        CRACK_THRESHOLD,
      )} that opens it — ${temperatureOf(best.objectiveValue)} rather than cracked. You did vary ${
        dial.name
      } across ${distinct} values, so this was not a coverage failure; the tries landed near the right region without landing in it. Bayesian mode spends its tries where the surrogate says improvement is most likely, which is what closes a gap this size.`,
    },
  };
}

// ── Reveal the math ───────────────────────────────────────────────────────

export const MATH_EQUATION = String.raw`\text{grid: } k^d \text{ trials, } k \text{ values per dial}
\qquad
\text{random: } n \text{ trials, } n \text{ values per dial}
\\[1.2em]
\mathrm{EI}(x) = \underbrace{(\mu(x) - f^{*} - \xi)\,\Phi(z)}_{\text{expect better}} \;+\; \underbrace{\sigma(x)\,\phi(z)}_{\text{expect surprise}}
\qquad
z = \frac{\mu(x) - f^{*} - \xi}{\sigma(x)}`;

export const MATH_CODE = `// Grid search over ${DIAL_COUNT} dials, at the finest resolution the budget allows.
export function gridLevels(budget, dials) {
  let levels = 1;
  while ((levels + 1) ** dials <= budget) levels += 1;
  return levels;          // 16 tries, 4 dials -> 2 levels. Two.
}

// Expected Improvement: where to look next, given everything tried so far.
export function expectedImprovement(surrogate, point) {
  const { mean, sd } = surrogate.predict(point);
  const gap = mean - surrogate.best - ${EI_EXPLORATION};
  const z = gap / sd;

  //   first term: this point probably scores better than the incumbent
  //  second term: we are unsure, and unsure is worth something
  return gap * Phi(z) + sd * phi(z);
}`;

export const MATH_NOTES = `A grid feels thorough and is not. With ${DIAL_COUNT} dials and ${BUDGET} tries the finest full grid is ${gridLevels(
  BUDGET,
)} levels per dial, so every one of your tries goes into a corner of a ${DIAL_COUNT}-dimensional box and you still learn the learning rate at only ${gridLevels(
  BUDGET,
)} points. Spend the same ${BUDGET} tries at random and you learn it at ${BUDGET} points. That is the whole of Bergstra and Bengio's argument: the grid spends its budget on dials that do not matter, and it cannot help doing so, because a full factorial has to vary everything equally.

The reason it bites here is that two of these four dials carry almost the whole surface. Sweeping each one alone moves the objective by ${dialInfluence()
  .map((influence, index) => `${DIALS[index]!.name} ${swing(influence)}`)
  .join(", ")} — so half your grid budget goes into dials that cannot move the answer by more than a few points. Note that momentum measures higher than its own weight suggests: it drags the learning rate's optimum with it, and a sweep picks up that interaction too. Real response surfaces look like this more often than not, which is why random search is a genuinely strong default rather than a lazy one.

Bayesian search does something different again: it fits a model of the objective to what it has already seen, then picks the point where Expected Improvement is highest. Read EI as two competing terms. One says "go where the model predicts a good score", the other says "go where the model has no idea". Pure exploitation would circle the first decent result it found; pure exploration would ignore everything it learned. The sum is why it beats random — not by being cleverer about any single try, but by never wasting one on a region it has already ruled out.

The catch worth knowing: the surrogate needs data before it says anything useful, which is why the first ${BAYESIAN_SEED_TRIALS} tries here are spread out rather than modelled. Under a very small budget, Bayesian search is mostly just paying for its own warm-up.`;
