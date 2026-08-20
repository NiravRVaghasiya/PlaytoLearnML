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
 * ── Why the dirt is recoverable ─────────────────────────────────────────────
 * Labels are generated from CLEAN features, and only then are values corrupted.
 * So the dirt is genuinely noise obscuring a real signal, and median-imputing a
 * null or capping an outlier actually recovers information. If labels were
 * derived from the dirty values, cleaning would destroy signal rather than
 * restore it, and the game would teach the opposite of its lesson.
 *
 * ── The four actions are orthogonal on purpose ───────────────────────────────
 *   Keep   — use the row as-is. A null becomes 0, which is not "unknown", it's a
 *            specific wrong number. This is the mistake the game is built to show.
 *   Impute — fill nulls with the column median. Does nothing about outliers.
 *   Cap    — clamp every value into the [p5, p95] range. Does nothing about nulls
 *            (you cannot clamp a missing value), so nulls still become 0.
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
/** How far an outlier is thrown out of range. */
export const OUTLIER_FACTOR = 7;
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
 * Set at 0.70 for real headroom. A strong pipeline scores 0.787–0.875 across the
 * shipped seeds, and `score` is accuracy MINUS a time penalty of up to 0.04: at
 * 0.75 the weakest seed became unwinnable the moment the clock moved, and at 0.80
 * two of them did.
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

/** Percentile bounds used by Cap. */
export const CAP_LOW = 0.05;
export const CAP_HIGH = 0.95;

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

    // Missing not at random: blanks favour the healthy class. Dropping them
    // therefore drops mostly one class — see the NULL_RATE_* docblock.
    const nullRate = label === 1 ? NULL_RATE_HEALTHY : NULL_RATE_BLIGHTED;
    if (nullDraw < nullRate) {
      observed[nullVictim] = null;
      isNull = true;
    }
    if (outlierDraw < OUTLIER_RATE && observed[outlierVictim] !== null) {
      // +2 guarantees the value lands unambiguously outside the 0–1 range even
      // when the clean reading is near zero.
      observed[outlierVictim] = clean[outlierVictim] * OUTLIER_FACTOR + 2;
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
 * skipped — which is what an analyst actually has to work with. Outliers drag
 * these statistics around a little, and that's part of the lesson.
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

    median[feature] = quantile(values, 0.5);
    low[feature] = quantile(values, CAP_LOW);
    high[feature] = quantile(values, CAP_HIGH);
  }

  return { median, low, high };
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
 * "Never drop" is a measured conclusion, not a stylistic preference. An earlier
 * version dropped the doubly-corrupted rows on the grounds that no single action
 * repairs them — and it scored WORSE on every seed (0.745/0.813/0.830) than simply
 * imputing them (0.840/0.843/0.868). Because blanks are class-skewed, dropping even
 * a handful of rows starts bending the class balance, and that costs more than the
 * outlier you failed to cap.
 *
 * For a row that is both blank AND out of range there genuinely is no right
 * answer: imputing fixes the blank and keeps the outlier, capping fixes the
 * outlier and leaves a 0 where the blank was. Across seeds the two swap places
 * (impute-both 0.840/0.843/0.868 against cap-both 0.813/0.828/0.890). That is the
 * spec's "no single right answer — only tradeoffs", as a number rather than a
 * slogan.
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
  | "near-miss";

export interface Evaluation {
  accuracy: number;
  timePenalty: number;
  score: number;
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
 */
export function evaluate(
  pipeline: PipelineResult,
  accuracy: number,
  secondsElapsed: number,
): Evaluation {
  const timePenalty = timePenaltyFor(secondsElapsed);
  const score = clamp(accuracy - timePenalty, 0, 1);

  const nullShare = pipeline.kept > 0 ? pipeline.keptWithNaiveFill / pipeline.kept : 0;
  const outlierShare = pipeline.kept > 0 ? pipeline.keptWithOutlier / pipeline.kept : 0;
  const percent = (value: number) => `${Math.round(value * 100)}%`;

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
      } rows, leaving ${pipeline.kept} to train on. Held-out accuracy ${percent(
        accuracy,
      )}. Below about ${MIN_ROWS} rows the model has too few examples to learn the pattern, no matter how clean they are — dropping is the most expensive way to fix a row.`,
    };
  } else if (Math.abs(pipeline.balanceDrift) > MAX_BALANCE_DRIFT) {
    outcome = "selection-bias";
    const lost = pipeline.balanceDrift < 0 ? "healthy" : "blighted";
    failure = {
      name: "Selection bias",
      detail: `Your ${pipeline.kept} surviving rows are ${percent(
        pipeline.keptPositiveShare,
      )} healthy, but the data you started from was ${percent(
        pipeline.sourcePositiveShare,
      )} healthy. Blank readings are far more common on ${lost} plots, so dropping them threw away that class specifically. The model learned a world that doesn't exist. Held-out accuracy ${percent(
        accuracy,
      )}.`,
    };
  } else if (nullShare > UNHANDLED_NULL_SHARE) {
    outcome = "unhandled-nulls";
    failure = {
      name: "Unhandled missing values",
      detail: `${pipeline.keptWithNaiveFill} of your ${pipeline.kept} training rows still had a blank cell, and the model read every blank as 0. Zero is not "unknown" — it's a specific wrong measurement, and it pulled the boundary toward it. Held-out accuracy ${percent(
        accuracy,
      )}.`,
    };
  } else if (outlierShare > UNCAPPED_OUTLIER_SHARE) {
    outcome = "outlier-contamination";
    failure = {
      name: "Outlier contamination",
      detail: `${pipeline.keptWithOutlier} of your ${pipeline.kept} training rows carried a value ${OUTLIER_FACTOR}× out of range. A handful of extreme rows drags the fit further than dozens of ordinary ones. Held-out accuracy ${percent(
        accuracy,
      )}.`,
    };
  } else if (score >= WIN_ACCURACY) {
    outcome = "win";
  } else {
    outcome = "near-miss";
  }

  return {
    accuracy,
    timePenalty,
    score,
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
 * Screened so that on every one: a strong pipeline clears the bar, keeping
 * everything is caught as unhandled blanks, dropping the blank rows is caught as
 * selection bias, and dropping everything starves the model.
 *
 * Seed 4101 was cut. Its strongest pipeline reached only 0.745, and — worse —
 * dropping every dirty row happened to leave a barely-balanced sample that WON,
 * which would have taught the opposite of the lesson on one round in six.
 */
export const ROUND_SEEDS = [4102, 4103, 4104, 4105, 4106] as const;

export function seedForRound(round: number): number {
  return ROUND_SEEDS[(Math.max(1, Math.floor(round)) - 1) % ROUND_SEEDS.length]!;
}

// ── Reveal the math ────────────────────────────────────────────────────────

export const MATH_EQUATION = String.raw`\textbf{impute:}\quad x_{ij} \leftarrow \mathrm{median}_i\!\left(\{x_{ik} : x_{ik} \ne \varnothing\}\right)
\qquad
\textbf{cap:}\quad x_{ij} \leftarrow \min\!\left(\max\left(x_{ij},\, q_{0.05}\right),\, q_{0.95}\right)
\\[1.2em]
\textbf{keep:}\quad x_{ij} \leftarrow \begin{cases} 0 & x_{ij} = \varnothing \\ x_{ij} & \text{otherwise} \end{cases}
\qquad
\textbf{drop:}\quad \text{row removed from } X`;

export const MATH_CODE = `// Your four bins are these four transformations. Nothing else.
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

export const MATH_NOTES = `x_ij is feature i of row j, and ∅ is a missing value. The medians and the 5th/95th percentiles are computed from the dirty data with blanks skipped — the same limited view you'd have in real life, which is why a few extreme values nudge those statistics around. Labels come from the clean measurements, so imputing and capping genuinely recover signal rather than inventing it. About ${Math.round(
  LABEL_NOISE * 100,
)}% of labels are wrong on purpose, so roughly ${Math.round(
  (1 - LABEL_NOISE) * 100,
)}% is the ceiling for any pipeline.`;
