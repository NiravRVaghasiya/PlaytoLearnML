import * as tf from "@tensorflow/tfjs";
import type { NamedFailure } from "@/engine/types";
import { clamp, gaussian, seededRandom } from "@/lib/utils";

/**
 * Data Detox — the real machine learning.
 *
 * The player sorts dirty rows into Impute / Drop / Cap / Keep. That IS a
 * preprocessing pipeline, and a real TensorFlow.js classifier retrains on
 * whatever survives it. The accuracy meter is that model's score on a held-out
 * set, so every cleaning decision moves a number that was genuinely earned.
 *
 * ── A note on the spec's "small decision tree in TF.js" ──────────────────────
 * TensorFlow.js has no decision-tree learner; it's a neural-network library. The
 * spec's Tech row says TensorFlow.js, so this uses a small dense classifier
 * instead of a tree. Trees are the subject of Decision Tree Architect (Phase 2),
 * where they're the lesson rather than an implementation detail.
 *
 * ── Why the dirt is worth cleaning ──────────────────────────────────────────
 * Labels are generated from CLEAN features, and only then are values corrupted.
 * So the dirt is genuinely noise obscuring a real signal: median-imputing a null
 * puts a plausible value back, and capping a spike stops one impossible input
 * from swamping the rest of its row. If labels were derived from the dirty
 * values, cleaning would destroy signal rather than restore it, and the game
 * would teach the opposite of its lesson.
 *
 * ── The four actions are orthogonal on purpose ───────────────────────────────
 *   Keep   — use the row as-is. A null becomes 0, which is not "unknown", it's a
 *            specific wrong number. This is the mistake the game is built to show.
 *   Impute — fill nulls with the column median. Does nothing about outliers.
 *   Cap    — clamp every value into the column's Tukey fences (see
 *            `columnStats`). Does nothing about nulls (you cannot clamp a
 *            missing value), so nulls still become 0.
 *   Drop   — exclude the row. Costs you data.
 *
 * Each dirt type therefore has one best answer, and rows that are BOTH null and
 * outlier have no good answer at all — which is the core intuition's "no single
 * right answer, only tradeoffs", made mechanical.
 */

export const FEATURES = ["moisture", "temperature", "nitrogen"] as const;
export type FeatureName = (typeof FEATURES)[number];

export type CleaningAction = "keep" | "impute" | "cap" | "drop";

export const CLEANING_ACTIONS: readonly CleaningAction[] = [
  "keep",
  "impute",
  "cap",
  "drop",
] as const;

export interface Row {
  id: string;
  /** Observed values — possibly null, possibly wildly out of range. */
  features: Record<FeatureName, number | null>;
  isNull: boolean;
  isOutlier: boolean;
  playerAction: CleaningAction | null;
  /** Ground truth, derived from the CLEAN values before corruption. */
  label: 0 | 1;
}

// ── Tuning ─────────────────────────────────────────────────────────────────

/**
 * Training rows. Deliberately scarce.
 *
 * At 140 rows with a linear boundary, dropping every dirty row left 85 clean ones
 * — plenty to learn from — so dropping beat imputing on every seed and the game
 * taught "just throw the mess away". Data has to be genuinely scarce before
 * preserving a partially-corrupted row is worth more than discarding it. That is
 * also the honest version of the lesson: with abundant clean data, dropping IS
 * often fine.
 */
export const TRAIN_ROWS = 72;
/** Held-out rows. Clean, because they stand in for the world you deploy to. */
export const TEST_ROWS = 400;

/**
 * ── Missingness is NOT random, and that is the whole design ──────────────────
 *
 * Blanks land far more often on healthy rows than on blighted ones. The cover
 * story is that thriving plots got surveyed less carefully; the ML name is
 * "missing not at random" (MNAR).
 *
 * This matters because the obvious first design — dirt sprinkled uniformly — does
 * not teach anything reliably. Three attempts confirmed it:
 *
 *   1. 140 rows, linear boundary: dropping every dirty row left 85 clean ones and
 *      BEAT careful imputation on all three seeds. With abundant clean data,
 *      dropping really is fine, so the game taught "throw the mess away".
 *   2. A curved boundary to make data scarcer: accuracy fell to 0.70–0.81, below
 *      the win bar, and the ranking between strategies became luck.
 *   3. 60 rows with heavy uniform dirt: 13 clean rows still scored 0.71–0.84,
 *      because a plane needs very few points.
 *
 * With missingness skewed by class, dropping blank rows removes mostly one class
 * and skews the training distribution — a bias that does NOT wash out with more
 * data or a simpler model. Imputation keeps those rows and preserves the balance.
 * That gives the game a mechanism that is robust, measurable without training,
 * and a more honest lesson than sample size alone.
 */
export const NULL_RATE_HEALTHY = 0.62;
export const NULL_RATE_BLIGHTED = 0.12;
export const OUTLIER_RATE = 0.28;

/** The physical range every sensor reads in. Anything outside it is impossible. */
export const SENSOR_MIN = 0;
export const SENSOR_MAX = 1;

/**
 * ── Outliers are sensor spikes, and spikes carry no signal ───────────────────
 *
 * A spiked reading is drawn uniformly from [OUTLIER_MIN, OUTLIER_MAX], with no
 * trace of the true value — a glitch, not an exaggeration.
 *
 * The first design got this wrong, measurably. An outlier was a stretched copy
 * of the true reading (7x + 2), and Cap clamped to the 5th/95th percentile. With
 * roughly one reading in ten corrupted, the 95th percentile landed INSIDE the
 * outliers — 3.5 to 6.5 on a 0–1 sensor — so capping left 65 of 67 capped rows
 * still out of range. And because 7x + 2 preserves the ordering, a ReLU network
 * simply read through it: keeping the outliers scored the same as capping them
 * (0.848 against 0.843 on seed 4102), so "Outlier contamination" was a named
 * failure that the accuracy meter contradicted.
 *
 * Spikes this far out do real damage to this model, and for a reason that has
 * nothing to do with squared error (the loss is cross-entropy, and the targets
 * are clean): every neuron sums weight × input, so one input of 60 swamps the
 * rest of its row, and the gradient on that weight scales with the input too.
 * Smaller spikes (2–9) were tried first and did not hurt this model at all. On
 * the shipped seeds, keeping these costs 5–12 points of held-out accuracy
 * against capping them — see ROUND_SEEDS, and the lesson test that holds every
 * round to it.
 */
export const OUTLIER_MIN = 20;
export const OUTLIER_MAX = 100;

/**
 * Tukey's fence multiplier. Cap clamps a column into [Q1 − k·IQR, Q3 + k·IQR].
 *
 * Quartiles, not the 5th/95th percentiles, because the bound has to survive the
 * contamination it exists to remove: with about one reading in ten spiked, the
 * 95th percentile sits among the spikes, while a quartile can't be moved by
 * anything short of a quarter of the column.
 */
export const FENCE_K = 1.5;
/** Irreducible label noise, so no pipeline can reach 100%. */
export const LABEL_NOISE = 0.06;

/** Rows retrain the model every this many decisions (spec: "every N rows"). */
export const BATCH_SIZE = 10;

/** Below this many surviving rows the model is starved. */
export const MIN_ROWS = 30;
/** Share of kept rows still carrying a zero-filled null that counts as dirty. */
export const UNHANDLED_NULL_SHARE = 0.15;
/** Share of kept rows still carrying an uncapped outlier that counts as dirty. */
export const UNCAPPED_OUTLIER_SHARE = 0.15;

/**
 * Held-out accuracy needed to clear a round.
 *
 * A secondary gate, not the main one. To pass the checks above you must already
 * have imputed the blanks and capped the outliers — any other policy leaves a
 * detectable share of dirt or bends the class balance — so by the time a pipeline
 * reaches this bar it is a good pipeline. The bar exists to catch the remainder.
 *
 * Set at 0.70 for real headroom. A strong pipeline scores 0.860–0.885 across the
 * shipped seeds, and `score` is accuracy MINUS a time penalty of up to 0.04, so
 * even a player who takes as long as they like to read the cards scores at least
 * 0.82 — past the bar, and past HIGH_SCORE_THRESHOLD for the second star. The
 * lesson test asserts both with the penalty at its cap.
 */
export const WIN_ACCURACY = 0.7;

/**
 * Time penalty (spec: "score = final model accuracy − time penalty").
 *
 * Deliberately small and capped: the WhyCards are the teaching, and a timer that
 * punishes reading them would work against the point of the game.
 */
export const TIME_PENALTY_PER_SECOND = 0.0004;
export const MAX_TIME_PENALTY = 0.04;

/** What a Keep does with a missing value. Not "unknown" — just wrong. */
export const NAIVE_FILL = 0;

// ── The generative process ─────────────────────────────────────────────────

/**
 * True weights. The signal the player is trying to preserve.
 *
 * A plane, not a curve. A closed curved region was tried and abandoned: with only
 * ~110 surviving rows a small net reached just 0.70–0.81 on it, which put the
 * round below the win bar and made the ranking between cleaning strategies depend
 * on luck rather than on the cleaning. The scarcity lever does the teaching here,
 * not task difficulty.
 */
const WEIGHTS: Record<FeatureName, number> = {
  moisture: 1.7,
  temperature: -1.5,
  nitrogen: 1.0,
};
/** Chosen to keep the two classes close to balanced. */
const BIAS = -0.6;

export function labelFor(
  clean: Record<FeatureName, number>,
  noiseDraw: number,
): 0 | 1 {
  const score =
    FEATURES.reduce(
      (total, feature) => total + WEIGHTS[feature] * clean[feature],
      0,
    ) + BIAS;
  const clean01: 0 | 1 = score > 0 ? 1 : 0;
  return noiseDraw < LABEL_NOISE ? ((1 - clean01) as 0 | 1) : clean01;
}

export interface Dataset {
  /** Dirty rows the player must clean. */
  train: Row[];
  /** Clean held-out rows, used only to score the model. */
  test: Array<{ features: Record<FeatureName, number>; label: 0 | 1 }>;
}

export function generateDataset(seed: number): Dataset {
  const random = seededRandom(seed);
  // Spike magnitudes come from their own stream, drawn once per row. Taking
  // them from `random` would shift every later draw, and with it every label,
  // blank and outlier position — including the round-1 class balance the
  // Concept Library quotes (39 rows, 33% healthy against 53%).
  const spikeRandom = seededRandom(seed + 3301);

  const train: Row[] = [];
  for (let index = 0; index < TRAIN_ROWS; index += 1) {
    const clean = {
      moisture: clamp(0.5 + gaussian(random) * 0.22, 0, 1),
      temperature: clamp(0.5 + gaussian(random) * 0.22, 0, 1),
      nitrogen: clamp(0.5 + gaussian(random) * 0.22, 0, 1),
    } satisfies Record<FeatureName, number>;

    const label = labelFor(clean, random());

    // Corrupt AFTER labelling, so the dirt hides signal rather than defining it.
    const observed: Record<FeatureName, number | null> = { ...clean };
    let isNull = false;
    let isOutlier = false;

    // Every row consumes exactly the same number of draws, whether or not it ends
    // up corrupted. Drawing the victim indices only inside the branches made the
    // stream position depend on earlier outcomes, which correlated the label draw
    // with the blank draw and pushed the measured blank rate on healthy rows to
    // 0.69 against a configured 0.62.
    const nullDraw = random();
    const nullVictim = FEATURES[Math.floor(random() * FEATURES.length)]!;
    const outlierDraw = random();
    const outlierVictim = FEATURES[Math.floor(random() * FEATURES.length)]!;
    const spikeDraw = spikeRandom();

    // Missing not at random: blanks favour the healthy class. Dropping them
    // therefore drops mostly one class — see the NULL_RATE_* docblock.
    const nullRate = label === 1 ? NULL_RATE_HEALTHY : NULL_RATE_BLIGHTED;
    if (nullDraw < nullRate) {
      observed[nullVictim] = null;
      isNull = true;
    }
    if (outlierDraw < OUTLIER_RATE && observed[outlierVictim] !== null) {
      // A glitch: independent of the true reading, and far outside 0–1.
      observed[outlierVictim] =
        OUTLIER_MIN + spikeDraw * (OUTLIER_MAX - OUTLIER_MIN);
      isOutlier = true;
    }

    train.push({
      id: `row-${index}`,
      features: observed,
      isNull,
      isOutlier,
      playerAction: null,
      label,
    });
  }

  const test: Dataset["test"] = [];
  const testRandom = seededRandom(seed + 7717);
  for (let index = 0; index < TEST_ROWS; index += 1) {
    const clean = {
      moisture: clamp(0.5 + gaussian(testRandom) * 0.22, 0, 1),
      temperature: clamp(0.5 + gaussian(testRandom) * 0.22, 0, 1),
      nitrogen: clamp(0.5 + gaussian(testRandom) * 0.22, 0, 1),
    } satisfies Record<FeatureName, number>;
    test.push({ features: clean, label: labelFor(clean, testRandom()) });
  }

  return { train, test };
}

// ── Column statistics ──────────────────────────────────────────────────────

export interface ColumnStats {
  median: Record<FeatureName, number>;
  low: Record<FeatureName, number>;
  high: Record<FeatureName, number>;
}

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const position = clamp(q, 0, 1) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower]! * (1 - weight) + sorted[upper]! * weight;
}

/**
 * Medians and cap bounds, computed from the DIRTY training rows with nulls
 * skipped — which is what an analyst actually has to work with.
 *
 * The cap bounds are Tukey's fences, Q1 − 1.5·IQR and Q3 + 1.5·IQR. Rank-based
 * statistics are what make that honest: the spikes can't drag a quartile, so a
 * capped spike lands just past the top of the real readings (about 1.1–1.5 on a
 * 0–1 sensor) — still high, no longer absurd — and on every shipped seed the
 * fences contain the whole 0–1 range, so capping a clean reading changes
 * nothing. The tests hold every round to both.
 */
export function columnStats(rows: Row[]): ColumnStats {
  const median = {} as Record<FeatureName, number>;
  const low = {} as Record<FeatureName, number>;
  const high = {} as Record<FeatureName, number>;

  for (const feature of FEATURES) {
    const values = rows
      .map((row) => row.features[feature])
      .filter((value): value is number => value !== null)
      .sort((a, b) => a - b);

    const q1 = quantile(values, 0.25);
    const q3 = quantile(values, 0.75);
    const spread = q3 - q1;

    median[feature] = quantile(values, 0.5);
    low[feature] = q1 - FENCE_K * spread;
    high[feature] = q3 + FENCE_K * spread;
  }

  return { median, low, high };
}

/**
 * Is this reading physically impossible for the sensor? Exact rather than
 * statistical: clean readings are clamped to 0–1 and a spike is at least
 * OUTLIER_MIN, so this flags every spike and nothing else. It is what the belt
 * highlights, and what "out of range" means everywhere in the copy.
 */
export function isOutOfRange(value: number | null): boolean {
  return value !== null && (value < SENSOR_MIN || value > SENSOR_MAX);
}

// ── The pipeline ───────────────────────────────────────────────────────────

export interface PipelineResult {
  xs: number[][];
  ys: number[];
  /** Rows that survived. */
  kept: number;
  dropped: number;
  /** Kept rows whose missing value ended up as a naive 0. */
  keptWithNaiveFill: number;
  /** Kept rows still carrying an out-of-range value. */
  keptWithOutlier: number;
  /** Per-bin counts (spec: `binCounts`). */
  binCounts: Record<CleaningAction, number>;
  /** Rows the player hasn't decided on yet. */
  undecided: number;
  /** Share of surviving rows labelled healthy. */
  keptPositiveShare: number;
  /** Share of ALL decided rows labelled healthy — the distribution to preserve. */
  sourcePositiveShare: number;
  /** How far the surviving sample drifted from the source. Signed. */
  balanceDrift: number;
}

/**
 * Turn player decisions into a training matrix.
 *
 * This is the whole "player action = algorithm" claim, executable: the enum on
 * each row selects a real transformation, and the matrix that comes out is what
 * the model actually sees.
 */
export function applyPipeline(rows: Row[], stats: ColumnStats): PipelineResult {
  const xs: number[][] = [];
  const ys: number[] = [];

  const binCounts: Record<CleaningAction, number> = {
    keep: 0,
    impute: 0,
    cap: 0,
    drop: 0,
  };

  let keptWithNaiveFill = 0;
  let keptWithOutlier = 0;
  let undecided = 0;

  for (const row of rows) {
    const action = row.playerAction;
    if (action === null) {
      undecided += 1;
      continue;
    }

    binCounts[action] += 1;
    if (action === "drop") continue;

    let naiveFilled = false;
    let stillOutlying = false;

    const vector = FEATURES.map((feature) => {
      const raw = row.features[feature];

      if (raw === null) {
        // Only Impute addresses missingness. Keep and Cap leave a bare 0.
        if (action === "impute") return stats.median[feature];
        naiveFilled = true;
        return NAIVE_FILL;
      }

      if (action === "cap") {
        return clamp(raw, stats.low[feature], stats.high[feature]);
      }

      // Keep and Impute pass values through untouched, outliers included.
      return raw;
    });

    // Count only genuinely corrupted readings that were left uncapped.
    //
    // An earlier version flagged any kept value outside the 5th–95th percentile
    // band, which by definition catches about 10% of perfectly good rows — enough
    // to accuse the ideal pipeline of outlier contamination on every seed. Being
    // in the tail of a distribution is not the same as being corrupted.
    if (row.isOutlier && action !== "cap") stillOutlying = true;

    if (naiveFilled) keptWithNaiveFill += 1;
    if (stillOutlying) keptWithOutlier += 1;

    xs.push(vector);
    ys.push(row.label);
  }

  const kept = xs.length;

  // Class balance, before and after the player's pipeline. Because blanks favour
  // the healthy class, dropping them shifts this — which is measurable here
  // without training anything.
  const decided = rows.filter((row) => row.playerAction !== null);
  const sourcePositives = decided.filter((row) => row.label === 1).length;
  const keptPositives = ys.filter((y) => y === 1).length;

  const sourcePositiveShare =
    decided.length > 0 ? sourcePositives / decided.length : 0;
  const keptPositiveShare = kept > 0 ? keptPositives / kept : 0;

  return {
    xs,
    ys,
    kept,
    dropped: binCounts.drop,
    keptWithNaiveFill,
    keptWithOutlier,
    binCounts,
    undecided,
    keptPositiveShare,
    sourcePositiveShare,
    balanceDrift: keptPositiveShare - sourcePositiveShare,
  };
}

/**
 * A strong pipeline: impute blanks, cap outliers, keep clean rows, and never drop.
 *
 * "Never drop" is a measured conclusion, not a stylistic preference. Dropping the
 * doubly-corrupted rows, on the grounds that no single action repairs them,
 * scores WORSE on every shipped seed (0.792/0.858/0.860/0.825/0.813) than simply
 * imputing them (0.860/0.865/0.885/0.875/0.868). Those rows carry a blank, and
 * blanks are class-skewed, so dropping even a handful starts bending the class
 * balance — which costs more than the spike you failed to cap.
 *
 * (Dropping rows whose ONLY problem is a spike is a different matter: it lands
 * either side of capping them, from 0.795 to 0.905. Spikes themselves hit both
 * classes at close to the same rate, but a spike-only row is one with no blank,
 * and blanks cluster on healthy rows — so spike-only rows lean blighted (21
 * healthy against 45 blighted across the shipped seeds, a majority on every
 * one). Dropping them all costs sample size AND tips the balance toward healthy,
 * by 2–8 points on the shipped seeds — under MAX_BALANCE_DRIFT, so it is never
 * called selection bias, but it is not free either. That is the honest version
 * of "remove what is physically impossible".)
 *
 * For a row that is both blank AND out of range there genuinely is no right
 * answer: imputing fixes the blank and keeps the spike, capping fixes the spike
 * and leaves a 0 where the blank was. Across seeds the two swap places
 * (impute-both 0.860/0.865/0.885/0.875/0.868 against cap-both
 * 0.830/0.895/0.917/0.880/0.828). That is the spec's "no single right answer —
 * only tradeoffs", as a number rather than a slogan.
 */
export function idealAction(row: Row): CleaningAction {
  // Blanks first: they're the biased ones, so repairing them matters most.
  if (row.isNull) return "impute";
  if (row.isOutlier) return "cap";
  return "keep";
}

// ── The model ──────────────────────────────────────────────────────────────

/**
 * A small dense classifier, built with seeded initializers so that identical
 * data produces identical results. Without the seeds, every retrain would move
 * the accuracy meter for reasons that had nothing to do with the player's
 * cleaning — and the tests could not assert anything.
 */
export function buildModel(seed = 1): tf.LayersModel {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 8,
      activation: "relu",
      inputShape: [FEATURES.length],
      kernelInitializer: tf.initializers.glorotNormal({ seed }),
      biasInitializer: "zeros",
    }),
  );
  model.add(
    tf.layers.dense({
      units: 1,
      activation: "sigmoid",
      kernelInitializer: tf.initializers.glorotNormal({ seed: seed + 1 }),
      biasInitializer: "zeros",
    }),
  );
  model.compile({
    optimizer: tf.train.adam(0.05),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

export const TRAIN_EPOCHS = 60;
export const TRAIN_BATCH = 16;

/** Held-out accuracy from raw sigmoid outputs. */
export function accuracyFromPredictions(
  predictions: ArrayLike<number>,
  labels: number[],
): number {
  if (labels.length === 0) return 0;
  let correct = 0;
  for (let index = 0; index < labels.length; index += 1) {
    const predicted = (predictions[index] ?? 0) >= 0.5 ? 1 : 0;
    if (predicted === labels[index]) correct += 1;
  }
  return correct / labels.length;
}

/** The held-out set as a matrix, ready for `predict`. */
export function testMatrix(dataset: Dataset): {
  xs: number[][];
  ys: number[];
} {
  return {
    xs: dataset.test.map((row) => FEATURES.map((f) => row.features[f])),
    ys: dataset.test.map((row) => row.label),
  };
}

// ── Evaluation ─────────────────────────────────────────────────────────────

export type Outcome =
  | "win"
  | "starved"
  | "selection-bias"
  | "unhandled-nulls"
  | "outlier-contamination"
  | "incomplete"
  /** Every row decided, nothing structurally wrong, but no model has been
   *  trained on THIS pipeline yet — so there is no accuracy to judge. */
  | "untrained"
  | "near-miss";

export interface Evaluation {
  /** Held-out accuracy of a model trained on this exact pipeline; null if none. */
  accuracy: number | null;
  timePenalty: number;
  /** Accuracy minus the time penalty; null when there is no accuracy. */
  score: number | null;
  /** The clock the time penalty was charged against, in seconds. */
  secondsElapsed: number;
  kept: number;
  dropped: number;
  keptWithNaiveFill: number;
  keptWithOutlier: number;
  binCounts: Record<CleaningAction, number>;
  keptPositiveShare: number;
  sourcePositiveShare: number;
  balanceDrift: number;
  outcome: Outcome;
  failure: NamedFailure | null;
}

/**
 * How far the surviving class balance may drift before it's called bias.
 *
 * Dropping blank rows shifts this hard, because blanks favour the healthy class.
 */
export const MAX_BALANCE_DRIFT = 0.12;

export function timePenaltyFor(secondsElapsed: number): number {
  return Math.min(
    MAX_TIME_PENALTY,
    Math.max(0, secondsElapsed) * TIME_PENALTY_PER_SECOND,
  );
}

/**
 * Score the pipeline and name the failure — honestly.
 *
 * Ordering is diagnostic. Starvation comes first because with too few rows the
 * accuracy number is not measuring the player's cleaning at all, it's measuring
 * sample size. Only once the model has enough data do the "you left dirt in"
 * diagnoses mean anything.
 *
 * `accuracy` is null when no model has been trained on this exact pipeline. The
 * structural failures above don't need one — class balance and leftover dirt are
 * measured from the pipeline itself — so they still fire, and their copy simply
 * omits the accuracy. A win or a near-miss is a claim ABOUT the accuracy, so
 * without one the verdict is "untrained" rather than a guess.
 */
export function evaluate(
  pipeline: PipelineResult,
  accuracy: number | null,
  secondsElapsed: number,
): Evaluation {
  const timePenalty = timePenaltyFor(secondsElapsed);
  const score = accuracy === null ? null : clamp(accuracy - timePenalty, 0, 1);

  const nullShare = pipeline.kept > 0 ? pipeline.keptWithNaiveFill / pipeline.kept : 0;
  const outlierShare = pipeline.kept > 0 ? pipeline.keptWithOutlier / pipeline.kept : 0;
  const percent = (value: number) => `${Math.round(value * 100)}%`;
  const measured =
    accuracy === null ? "" : ` Held-out accuracy ${percent(accuracy)}.`;

  let outcome: Outcome;
  let failure: NamedFailure | null = null;

  if (pipeline.undecided > 0) {
    outcome = "incomplete";
  } else if (pipeline.kept < MIN_ROWS) {
    outcome = "starved";
    failure = {
      name: "Data starvation",
      detail: `You dropped ${pipeline.dropped} of ${
        pipeline.kept + pipeline.dropped
      } rows, leaving ${pipeline.kept} to train on.${measured} Below about ${MIN_ROWS} rows the model has too few examples to learn the pattern, no matter how clean they are — dropping is the most expensive way to fix a row.`,
    };
  } else if (Math.abs(pipeline.balanceDrift) > MAX_BALANCE_DRIFT) {
    outcome = "selection-bias";
    // Blanks explain only a drift AWAY from healthy. A drift toward healthy
    // came from drops that took blighted rows out of proportion (a code-lane
    // rule like "drop moisture < 0.45" does it), and blaming blanks on
    // blighted plots there would state the generator backwards.
    const why =
      pipeline.balanceDrift < 0
        ? "Blank readings are far more common on healthy plots, so dropping them threw away that class specifically."
        : "The rows you dropped were blighted out of proportion, so healthy plots are now over-represented.";
    failure = {
      name: "Selection bias",
      detail: `Your ${pipeline.kept} surviving rows are ${percent(
        pipeline.keptPositiveShare,
      )} healthy, but the data you started from was ${percent(
        pipeline.sourcePositiveShare,
      )} healthy. ${why} The model learned a world that doesn't exist.${measured}`,
    };
  } else if (nullShare > UNHANDLED_NULL_SHARE) {
    outcome = "unhandled-nulls";
    failure = {
      name: "Unhandled missing values",
      detail: `${pipeline.keptWithNaiveFill} of your ${pipeline.kept} training rows still had a blank cell, and the model read every blank as 0. Zero is not "unknown" — it's a specific wrong measurement, and it pulled the boundary toward it.${measured}`,
    };
  } else if (outlierShare > UNCAPPED_OUTLIER_SHARE) {
    outcome = "outlier-contamination";
    failure = {
      name: "Outlier contamination",
      detail: `${pipeline.keptWithOutlier} of your ${pipeline.kept} training rows still carried an impossible reading, somewhere between ${OUTLIER_MIN} and ${OUTLIER_MAX} on a sensor that reads ${SENSOR_MIN} to ${SENSOR_MAX}. Every neuron sums weight × input, so one spike swamps the rest of its row, and the gradient it sends back scales with it — a handful of glitches pulls the weights further than dozens of ordinary rows. Cap clamps them to the column's fence instead.${measured}`,
    };
  } else if (accuracy === null || score === null) {
    outcome = "untrained";
  } else if (score >= WIN_ACCURACY) {
    outcome = "win";
  } else {
    outcome = "near-miss";
  }

  return {
    accuracy,
    timePenalty,
    score,
    secondsElapsed: Math.max(0, secondsElapsed),
    kept: pipeline.kept,
    dropped: pipeline.dropped,
    keptWithNaiveFill: pipeline.keptWithNaiveFill,
    keptWithOutlier: pipeline.keptWithOutlier,
    binCounts: pipeline.binCounts,
    keptPositiveShare: pipeline.keptPositiveShare,
    sourcePositiveShare: pipeline.sourcePositiveShare,
    balanceDrift: pipeline.balanceDrift,
    outcome,
    failure,
  };
}

/**
 * Rounds this game ships — curated, not arbitrary.
 *
 * Screened so that on every one: a strong pipeline clears the bar even with the
 * time penalty at its cap, keeping everything is caught as unhandled blanks,
 * dropping the blank rows is caught as selection bias, dropping everything
 * starves the model, and keeping the spikes is caught as outlier contamination
 * AND genuinely costs held-out accuracy (5–12 points against capping them). The
 * fences also contain the whole 0–1 sensor range on every one, so Cap never
 * alters a clean reading. `data-detox.test.ts` re-checks all of it.
 *
 * Seed 4101 was cut. Its strongest pipeline reached only 0.745, and — worse —
 * dropping every dirty row happened to leave a barely-balanced sample that WON,
 * which would have taught the opposite of the lesson on one round in six.
 *
 * Seeds 4103 and 4104 were cut when outliers became spikes. On 4103 keeping the
 * spikes scored the same as capping them (0.845 against 0.843), which would have
 * made "Outlier contamination" a verdict the meter contradicts; on 4104 the
 * strong pipeline fell to 0.733, under the bar once the clock ran. 4108 and 4100
 * replace them — 4100 rather than 4115, which passed everything else but has so
 * many blanks (79% of its healthy rows) that the pooled blank rate across the
 * rotation drifted off the configured one. 4102 stays first on purpose: the Concept Library quotes its
 * class balance.
 */
export const ROUND_SEEDS = [4102, 4105, 4106, 4108, 4100] as const;

export function seedForRound(round: number): number {
  return ROUND_SEEDS[(Math.max(1, Math.floor(round)) - 1) % ROUND_SEEDS.length]!;
}

// ── Reveal the math ────────────────────────────────────────────────────────

export const MATH_EQUATION = String.raw`\textbf{impute:}\quad x_{ij} \leftarrow \mathrm{median}_i\!\left(\{x_{ik} : x_{ik} \ne \varnothing\}\right)
\qquad
\textbf{drop:}\quad \text{row removed from } X
\\[1.2em]
\textbf{cap:}\quad x_{ij} \leftarrow \min\!\left(\max\left(x_{ij},\, Q_{1,i} - ${FENCE_K}\,\mathrm{IQR}_i\right),\, Q_{3,i} + ${FENCE_K}\,\mathrm{IQR}_i\right),
\quad \mathrm{IQR}_i = Q_{3,i} - Q_{1,i}
\\[1.2em]
\textbf{keep:}\quad x_{ij} \leftarrow \begin{cases} 0 & x_{ij} = \varnothing \\ x_{ij} & \text{otherwise} \end{cases}`;

export const MATH_CODE = `// The cap bounds are Tukey's fences, fitted once per column on the
// dirty data (blanks skipped):
//   low  = Q1 - ${FENCE_K} * (Q3 - Q1)
//   high = Q3 + ${FENCE_K} * (Q3 - Q1)

// Your four bins are these four transformations. Nothing else.
const vector = FEATURES.map((feature) => {
  const raw = row.features[feature];

  if (raw === null) {
    // Only Impute addresses missingness.
    if (action === 'impute') return stats.median[feature];
    // Keep and Cap leave a bare 0 — a specific wrong number,
    // not "unknown".
    return 0;
  }

  if (action === 'cap') {
    return clamp(raw, stats.low[feature], stats.high[feature]);
  }

  // Keep and Impute pass values through untouched, outliers included.
  return raw;
});

// 'drop' never reaches here: the row is excluded from X entirely.`;

export const MATH_NOTES = `x_ij is feature i of row j, and ∅ is a missing value. The medians and the quartiles Q1 and Q3 are computed from the dirty data with blanks skipped — the same limited view you'd have in real life. Quartiles rather than the 5th and 95th percentiles on purpose: about one reading in ten is a spike, so the 95th percentile would sit among the spikes and a cap there would clamp almost nothing, while a quartile can't be dragged by anything short of a quarter of the column. Labels come from the clean measurements, so the dirt genuinely hides signal: imputing puts a plausible value back, and capping stops an impossible reading from swamping the rest of its row. About ${Math.round(
  LABEL_NOISE * 100,
)}% of labels are wrong on purpose, so roughly ${Math.round(
  (1 - LABEL_NOISE) * 100,
)}% is the ceiling for any pipeline.`;
