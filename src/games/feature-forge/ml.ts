import * as tf from "@tensorflow/tfjs";
import type { NamedFailure } from "@/engine/types";
import { clamp, seededRandom } from "@/lib/utils";

/**
 * Feature Forge — the real feature engineering.
 *
 * The claim: representation matters more than algorithm choice on tabular data.
 * That is only teachable if the model is genuinely held still, so it is. Every
 * score in this game comes from the SAME logistic regression — same architecture,
 * same optimiser, same learning rate, same epochs, same seed. The only thing that
 * ever differs between two scores is the matrix handed to it.
 *
 * Which makes the central claim checkable rather than assertable, and the tests
 * check it: a deliberately bigger model on the baseline columns does NOT beat the
 * fixed logistic regression on well-forged features.
 *
 * ── The baseline is standardised on purpose ─────────────────────────────────
 * A tempting baseline is "raw columns, untouched", where income runs to 250,000
 * and a signup timestamp is 1.7 billion. Gradient descent handles that so badly
 * that any transform looks like genius, and the lesson collapses into "remember to
 * scale things". So the baseline here already standardises every raw column. Lift
 * has to come from changing what the columns MEAN — binning a non-monotonic
 * effect, one-hot encoding a category that was pretending to be a number, pulling
 * a weekday out of a timestamp — not from fixing the units.
 */

// ── Columns ───────────────────────────────────────────────────────────────

export type Dtype = "numeric" | "categorical" | "timestamp";

export interface Column {
  name: string;
  dtype: Dtype;
  description: string;
  /**
   * True for the column that encodes the outcome.
   *
   * A refund is issued BECAUSE a customer churned, so it is only knowable after
   * the fact. Including it produces a near-perfect score and a model that cannot
   * be deployed. Marked here so the diagnosis can be exact rather than heuristic.
   */
  leaky?: boolean;
  /** Kept out of the baseline matrix: it is a raw timestamp, useless unscaled. */
  excludeFromBaseline?: boolean;
}

export const COLUMNS: readonly Column[] = [
  {
    name: "signup_ts",
    dtype: "timestamp",
    description:
      "Unix seconds when they signed up. Meaningless as a number — the signal is which day of the week it was.",
    excludeFromBaseline: true,
  },
  {
    name: "age",
    dtype: "numeric",
    description:
      "Customer age. The effect is not a straight line: the youngest and oldest both churn more.",
  },
  {
    name: "income",
    dtype: "numeric",
    description: "Annual income. Long-tailed, so its scale swamps everything else.",
  },
  {
    name: "household",
    dtype: "numeric",
    description: "People in the household. Only means something next to income.",
  },
  {
    name: "city_code",
    dtype: "categorical",
    description:
      "City, stored as 0–5. The numbers imply an order that does not exist.",
  },
  {
    name: "support_tickets",
    dtype: "numeric",
    description: "Tickets opened in the last year. Straightforwardly bad news.",
  },
  {
    name: "refund_issued",
    dtype: "numeric",
    description:
      "Whether a refund was issued. Refunds happen AFTER a customer leaves.",
    leaky: true,
  },
] as const;

export function columnByName(name: string): Column | undefined {
  return COLUMNS.find((column) => column.name === name);
}

export const COLUMN_INDEX: Record<string, number> = Object.fromEntries(
  COLUMNS.map((column, index) => [column.name, index]),
);

export interface Row {
  values: number[];
  churned: 0 | 1;
}

export interface Dataset {
  train: Row[];
  validation: Row[];
}

export const TRAIN_ROWS = 1200;
export const VALIDATION_ROWS = 800;

/** Weekend signups (days 0 and 6) churn markedly more. */
const WEEKEND_CHURN_BOOST = 1.25;
/**
 * The first signup day, at midnight UTC and on a day `(ts // 86400) % 7` calls 0.
 *
 * Both lanes read the weekday with that expression — `day_of_week` here and the
 * Python starter's `(df.signup_ts // 86400) % 7` — so the generator's day 0 has
 * to BE day 0 by that arithmetic. (It is not a calendar Sunday; the convention is
 * the one the code uses, and nothing else depends on it.) The previous base,
 * 1_700_000_000, was 22:13 on a day the expression numbers 5, so the weekend
 * effect landed on days 5 and 6 while the starter told the player to look at 0
 * and 6, and signups late in the day bled into the next day's column.
 */
const BASE_TS = 1_700_092_800;

/** The ages either side of which churn rises: the two arms of the U. */
export const AGE_YOUNG = 30;
export const AGE_OLD = 68;
/** Per-head income below which churn rises. */
export const PER_HEAD_THRESHOLD = 20_000;

/**
 * Generate the customer table.
 *
 * Each column's relationship to churn is chosen so that a specific transform is
 * the thing that unlocks it:
 *
 *   signup_ts       weekday effect      -> needs day-of-week extraction
 *   age             U-shaped            -> needs binning
 *   city_code       one city is bad     -> needs one-hot
 *   income+household per-head threshold -> a ratio SOUNDS right, and measures nothing
 *   support_tickets monotonic           -> already usable standardised
 *   refund_issued   caused BY churn     -> leakage
 *
 * The per-head row is the one worth reading twice. Churn JUMPS below a per-head
 * income: the effect is a step. A ratio column gets one weight like any other,
 * so the most the model can do with income / household is a straight-line
 * response to it — dividing changes the axis, not the shape. And the baseline
 * can already slope a line across that same boundary, because "income /
 * household < t" is the same set of customers as "income − t × household < 0",
 * a line in the two raw columns. Neither can make the step; both can only slope
 * across it. So the ratio transform measures no lift here, and the game says so
 * rather than rewarding it. Catching the step itself would take a cut on the
 * ratio, which this forge does not offer. The tests measure that as well — a
 * cut at the threshold does lift the score — which is why the copy never claims
 * the baseline already had the whole effect.
 */
function makeRow(random: () => number): Row {
  const dayOfWeek = Math.floor(random() * 7);
  const signupTs = BASE_TS + dayOfWeek * 86_400 + Math.floor(random() * 86_400);

  const age = 18 + Math.floor(random() * 62);
  const income = Math.round(15_000 + Math.exp(random() * 2.8) * 12_000);
  const household = 1 + Math.floor(random() * 6);
  const cityCode = Math.floor(random() * 6);
  const supportTickets = Math.floor(random() * 21);

  // Log-odds of churning, built from the real effects.
  let logit = -1.15;

  // Weekend: days 0 and 6 of the `(ts // 86400) % 7` week — see BASE_TS.
  if (dayOfWeek === 0 || dayOfWeek === 6) logit += WEEKEND_CHURN_BOOST;

  // U-shape in age: both tails churn.
  //
  // The thresholds sit near quintile boundaries of a uniform 18–80 range on
  // purpose. An earlier version used 26 and 64, which fall in the MIDDLE of the
  // first and fourth bins — so binning mixed churny and calm customers into the
  // same bin and the transform measured no lift at all. An effect only becomes
  // reachable by binning if the bins can actually isolate it.
  if (age < AGE_YOUNG) logit += 1.15;
  else if (age > AGE_OLD) logit += 1.25;

  // One city is genuinely worse. Nothing ordinal about which.
  if (cityCode === 3) logit += 1.7;

  // Income only matters per head, and as a step at a threshold. See the header
  // note on why one weight on a ratio cannot catch a step, and adds nothing.
  const perHead = income / household;
  if (perHead < PER_HEAD_THRESHOLD) logit += 1.1;

  // Straightforward and monotonic.
  logit += (supportTickets - 10) * 0.075;

  const probability = 1 / (1 + Math.exp(-logit));
  const churned: 0 | 1 = random() < probability ? 1 : 0;

  // Consequence, not cause: only issued to people who already left.
  const refundIssued = churned === 1 ? (random() < 0.87 ? 1 : 0) : random() < 0.02 ? 1 : 0;

  return {
    values: [
      signupTs,
      age,
      income,
      household,
      cityCode,
      supportTickets,
      refundIssued,
    ],
    churned,
  };
}

export function generateDataset(seed: number): Dataset {
  const trainRandom = seededRandom(seed);
  const validationRandom = seededRandom(seed + 4241);

  return {
    train: Array.from({ length: TRAIN_ROWS }, () => makeRow(trainRandom)),
    validation: Array.from({ length: VALIDATION_ROWS }, () =>
      makeRow(validationRandom),
    ),
  };
}

// ── Transforms ────────────────────────────────────────────────────────────

export type Transform =
  | "standardise"
  | "log_scale"
  | "bin"
  | "one_hot"
  | "day_of_week"
  | "ratio";

export interface TransformSpec {
  id: Transform;
  label: string;
  /** How many source columns it consumes. */
  arity: number;
  /** Which dtypes it will accept. */
  accepts: readonly Dtype[];
  blurb: string;
  /** What it does to the matrix, in one clause. */
  mechanism: string;
}

export const BIN_COUNT = 5;
export const CITY_LEVELS = 6;

export const TRANSFORMS: readonly TransformSpec[] = [
  {
    id: "standardise",
    label: "Standardise",
    arity: 1,
    // Accepts categorical on purpose. Standardising city_code is a real and common
    // mistake, and it has to be POSSIBLE for the baseline to contain it — that is
    // the thing one-hot encoding is there to fix. Rejecting it here would silently
    // drop the column from the baseline and remove the lesson.
    accepts: ["numeric", "categorical"],
    blurb: "Centre and scale to unit variance.",
    mechanism:
      "one column in, one out — changes the units, never the shape of the relationship",
  },
  {
    id: "log_scale",
    label: "Log + scale",
    arity: 1,
    accepts: ["numeric"],
    blurb: "log1p, then standardise. For long tails.",
    mechanism:
      "compresses a long tail so the top few percent stop dominating the gradient",
  },
  {
    id: "bin",
    label: "Bin",
    arity: 1,
    accepts: ["numeric"],
    blurb: `Cut into ${BIN_COUNT} quantile bins, one-hot encoded.`,
    mechanism: `${BIN_COUNT} columns out — lets a linear model fit a curve it otherwise cannot`,
  },
  {
    id: "one_hot",
    label: "One-hot",
    arity: 1,
    accepts: ["categorical"],
    blurb: "One column per category.",
    mechanism:
      "removes the false ordering that storing a category as a number implies",
  },
  {
    id: "day_of_week",
    label: "Day of week",
    arity: 1,
    accepts: ["timestamp"],
    blurb: "Extract the weekday, one-hot encoded.",
    mechanism:
      "pulls a cyclical fact out of a number that was monotonic and meaningless",
  },
  {
    id: "ratio",
    label: "Ratio",
    arity: 2,
    accepts: ["numeric"],
    blurb: "Divide the first by the second, then scale.",
    mechanism:
      "creates a quantity neither column contains on its own",
  },
] as const;

export function transformById(id: Transform): TransformSpec {
  return TRANSFORMS.find((spec) => spec.id === id) ?? TRANSFORMS[0]!;
}

export interface Feature {
  id: string;
  sourceCols: string[];
  transform: Transform;
}

export function describeFeature(feature: Feature): string {
  const spec = transformById(feature.transform);
  return feature.transform === "ratio"
    ? `${feature.sourceCols[0]} / ${feature.sourceCols[1]}`
    : `${spec.label.toLowerCase()}(${feature.sourceCols.join(", ")})`;
}

/** Is this a legal pairing of transform and columns? */
export function isValidFeature(feature: Feature): boolean {
  const spec = transformById(feature.transform);
  if (feature.sourceCols.length !== spec.arity) return false;
  return feature.sourceCols.every((name) => {
    const column = columnByName(name);
    return column !== undefined && spec.accepts.includes(column.dtype);
  });
}

/**
 * Why a proposed feature cannot be forged, in words — or null if it can.
 *
 * `isValidFeature` answers yes or no, which is all the visual lane needs: it
 * disables the columns a transform will not take. The code lane has no such
 * guard rails, so a typo'd column or a timestamp handed to `bin` has to come
 * back as a sentence the player can act on, not as a score that silently did
 * not move.
 */
export function featureProblem(
  transform: string,
  sourceCols: readonly unknown[],
  existing: readonly Feature[] = [],
): string | null {
  const spec = TRANSFORMS.find((entry) => entry.id === transform);
  if (!spec) {
    return `unknown transform "${transform}" — try ${TRANSFORMS.map(
      (entry) => entry.id,
    ).join(", ")}`;
  }
  if (sourceCols.length !== spec.arity) {
    return `${spec.id} takes ${spec.arity} column${
      spec.arity === 1 ? "" : "s"
    }, got ${sourceCols.length}`;
  }
  for (const name of sourceCols) {
    const column =
      typeof name === "string" ? columnByName(name) : undefined;
    if (!column) {
      return `no column called ${JSON.stringify(name)} — the table has ${COLUMNS.map(
        (entry) => entry.name,
      ).join(", ")}`;
    }
    if (!spec.accepts.includes(column.dtype)) {
      return `${spec.id} needs a ${spec.accepts.join(" or ")} column, and ${
        column.name
      } is a ${column.dtype}`;
    }
  }
  const id = `${spec.id}:${sourceCols.join("+")}`;
  if (existing.some((feature) => feature.id === id)) {
    return `${id} is already in the forge — a duplicate adds width and nothing else`;
  }
  return null;
}

export function usesLeakyColumn(feature: Feature): boolean {
  return feature.sourceCols.some((name) => columnByName(name)?.leaky === true);
}

// ── Building the design matrix ────────────────────────────────────────────

interface ColumnStats {
  mean: number;
  sd: number;
  /** Quantile cut points for binning. */
  cuts: number[];
}

/**
 * Fit the numbers a transform needs — from the TRAINING rows only.
 *
 * This matters more than it looks. Standardising with statistics computed over
 * train and validation together, or binning on quantiles of the whole table, leaks
 * information about the validation set into the features. The lift would be real
 * on paper and unreproducible in production. So every statistic here is fitted on
 * train and then APPLIED to validation, which is what a real pipeline does.
 */
function fitStats(rows: Row[], columnIndex: number): ColumnStats {
  const values = rows.map((row) => row.values[columnIndex] ?? 0);
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  const variance =
    values.reduce((total, value) => total + (value - mean) ** 2, 0) /
    values.length;

  const sorted = [...values].sort((a, b) => a - b);
  const cuts = Array.from({ length: BIN_COUNT - 1 }, (_, index) => {
    const q = (index + 1) / BIN_COUNT;
    return sorted[Math.floor(q * (sorted.length - 1))] ?? 0;
  });

  return { mean, sd: Math.sqrt(Math.max(variance, 1e-9)), cuts };
}

export interface DesignMatrix {
  xs: number[][];
  ys: number[];
  /** One label per column of `xs`, for the importance bars. */
  columnNames: string[];
}

interface FittedTransform {
  names: string[];
  apply: (row: Row) => number[];
}

function fitTransform(
  feature: Feature,
  trainRows: Row[],
): FittedTransform | null {
  if (!isValidFeature(feature)) return null;

  const indices = feature.sourceCols.map((name) => COLUMN_INDEX[name] ?? 0);
  const label = describeFeature(feature);

  switch (feature.transform) {
    case "standardise": {
      const stats = fitStats(trainRows, indices[0]!);
      return {
        names: [label],
        apply: (row) => [
          ((row.values[indices[0]!] ?? 0) - stats.mean) / stats.sd,
        ],
      };
    }

    case "log_scale": {
      const logged = trainRows.map((row) => ({
        ...row,
        values: row.values.map((value, index) =>
          index === indices[0] ? Math.log1p(Math.max(0, value)) : value,
        ),
      }));
      const stats = fitStats(logged, indices[0]!);
      return {
        names: [label],
        apply: (row) => [
          (Math.log1p(Math.max(0, row.values[indices[0]!] ?? 0)) - stats.mean) /
            stats.sd,
        ],
      };
    }

    case "bin": {
      const stats = fitStats(trainRows, indices[0]!);
      return {
        names: Array.from({ length: BIN_COUNT }, (_, bin) => `${label}[${bin}]`),
        apply: (row) => {
          const value = row.values[indices[0]!] ?? 0;
          let bin = 0;
          while (bin < stats.cuts.length && value > stats.cuts[bin]!) bin += 1;
          return Array.from({ length: BIN_COUNT }, (_, index) =>
            index === bin ? 1 : 0,
          );
        },
      };
    }

    case "one_hot": {
      return {
        names: Array.from(
          { length: CITY_LEVELS },
          (_, level) => `${label}=${level}`,
        ),
        apply: (row) => {
          const value = Math.round(row.values[indices[0]!] ?? 0);
          return Array.from({ length: CITY_LEVELS }, (_, level) =>
            level === value ? 1 : 0,
          );
        },
      };
    }

    case "day_of_week": {
      return {
        names: Array.from({ length: 7 }, (_, day) => `${label}=${day}`),
        apply: (row) => {
          const seconds = row.values[indices[0]!] ?? 0;
          const day = Math.floor(seconds / 86_400) % 7;
          return Array.from({ length: 7 }, (_, index) =>
            index === day ? 1 : 0,
          );
        },
      };
    }

    case "ratio": {
      const ratios = trainRows.map((row) => ({
        ...row,
        values: [
          (row.values[indices[0]!] ?? 0) /
            Math.max(1e-9, row.values[indices[1]!] ?? 1),
        ],
      }));
      const stats = fitStats(ratios, 0);
      return {
        names: [label],
        apply: (row) => [
          ((row.values[indices[0]!] ?? 0) /
            Math.max(1e-9, row.values[indices[1]!] ?? 1) -
            stats.mean) /
            stats.sd,
        ],
      };
    }
  }
}

/** The baseline: every non-leaky, non-timestamp column, standardised. */
export function baselineFeatures(): Feature[] {
  return COLUMNS.filter(
    (column) =>
      !column.leaky &&
      !column.excludeFromBaseline &&
      (column.dtype === "numeric" || column.dtype === "categorical"),
  ).map((column) => ({
    id: `baseline-${column.name}`,
    sourceCols: [column.name],
    transform: "standardise" as Transform,
  }));
}

/**
 * Build train and validation matrices from a feature list.
 *
 * `standardise` is applied to categorical columns in the baseline too, which is
 * exactly the mistake the game wants the player to find: city_code standardised is
 * still a number pretending to have an order.
 */
export function buildMatrices(
  dataset: Dataset,
  features: Feature[],
): { train: DesignMatrix; validation: DesignMatrix } | null {
  const fitted = features
    .map((feature) => fitTransform(feature, dataset.train))
    .filter((entry): entry is FittedTransform => entry !== null);

  if (fitted.length === 0) return null;

  const columnNames = fitted.flatMap((entry) => entry.names);
  const build = (rows: Row[]): DesignMatrix => ({
    xs: rows.map((row) => fitted.flatMap((entry) => entry.apply(row))),
    ys: rows.map((row) => row.churned),
    columnNames,
  });

  return { train: build(dataset.train), validation: build(dataset.validation) };
}

// ── The fixed model ───────────────────────────────────────────────────────

/**
 * Every constant here is frozen, and that is the point of the game.
 *
 * If the model could change, a score change would be ambiguous — was it the
 * features or the model? Holding all of this still is what makes "representation
 * did that" a claim the player can trust.
 */
export const MODEL_SEED = 8123;
/**
 * Tuned for convergence AND for a retrain the player will sit through.
 *
 * Every forged feature triggers a retrain, so wall time is a design constraint,
 * not an implementation detail — the Definition of Done asks for the metric to
 * move within about a second. The cost of a fit is dominated by how many
 * sequential optimiser steps it takes (on WebGL each step is a round of kernel
 * dispatches), not by the arithmetic, which for a 1,200 × 24 logistic
 * regression is trivial.
 *
 * History: 200 epochs over batches of 32 was ~7,400 steps and ~21 seconds; 150
 * epochs over batches of 256 was ~750 steps and still several seconds per forge.
 * A logistic regression's loss is convex and every column is standardised or
 * one-hot, so full-batch Adam gets there in 100 steps. While tuning this was
 * checked against a fully converged Newton fit of the same model: the same 63.1%
 * baseline, and lifts within a validation row or two of it. The tests pin those
 * lifts.
 */
export const TRAIN_EPOCHS = 100;
export const TRAIN_BATCH = TRAIN_ROWS;
export const LEARNING_RATE = 0.2;

/** How a model is trained. The fixed model's is frozen along with the model. */
export interface TrainingSchedule {
  epochs: number;
  batchSize: number;
}

export const FIXED_SCHEDULE: TrainingSchedule = {
  epochs: TRAIN_EPOCHS,
  batchSize: TRAIN_BATCH,
};

/**
 * The fancier model keeps the longer mini-batch schedule the game used to give
 * every model: ~750 steps. More steps can only help a network that size, so the
 * "good features beat a bigger model" comparison stays on the conservative side.
 */
export const FANCIER_SCHEDULE: TrainingSchedule = { epochs: 150, batchSize: 256 };
const FANCIER_LEARNING_RATE = 0.25;

/** Logistic regression. One weight per column, one bias, sigmoid. */
export function buildFixedModel(inputWidth: number): tf.LayersModel {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 1,
      activation: "sigmoid",
      inputShape: [inputWidth],
      kernelInitializer: tf.initializers.glorotNormal({ seed: MODEL_SEED }),
      biasInitializer: "zeros",
    }),
  );
  model.compile({
    optimizer: tf.train.adam(LEARNING_RATE),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

/**
 * A deliberately fancier model, for the comparison the core intuition rests on.
 *
 * Used only by the tests — never to score the player. Its whole job is to lose
 * to good features.
 */
export function buildFancierModel(inputWidth: number): tf.LayersModel {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 32,
      activation: "relu",
      inputShape: [inputWidth],
      kernelInitializer: tf.initializers.heNormal({ seed: MODEL_SEED }),
      biasInitializer: tf.initializers.constant({ value: 0.05 }),
    }),
  );
  model.add(
    tf.layers.dense({
      units: 16,
      activation: "relu",
      kernelInitializer: tf.initializers.heNormal({ seed: MODEL_SEED + 7 }),
      biasInitializer: tf.initializers.constant({ value: 0.05 }),
    }),
  );
  model.add(
    tf.layers.dense({
      units: 1,
      activation: "sigmoid",
      kernelInitializer: tf.initializers.glorotNormal({ seed: MODEL_SEED + 13 }),
      biasInitializer: "zeros",
    }),
  );
  model.compile({
    optimizer: tf.train.adam(FANCIER_LEARNING_RATE),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

/** Which schedule a model factory trains on. */
export function scheduleFor(
  modelFactory: (width: number) => tf.LayersModel,
): TrainingSchedule {
  return modelFactory === buildFancierModel ? FANCIER_SCHEDULE : FIXED_SCHEDULE;
}

export function accuracyFrom(
  predictions: ArrayLike<number>,
  labels: number[],
): number {
  if (labels.length === 0) return 0;
  let correct = 0;
  for (let index = 0; index < labels.length; index += 1) {
    if (((predictions[index] ?? 0) >= 0.5 ? 1 : 0) === labels[index]) {
      correct += 1;
    }
  }
  return correct / labels.length;
}

export interface FitResult {
  validationAccuracy: number;
  trainAccuracy: number;
  /** |weight| per matrix column, for the importance bars. */
  importances: number[];
  columnNames: string[];
  width: number;
}

export interface FitOptions {
  /** Defaults to the factory's own schedule — see `scheduleFor`. */
  schedule?: TrainingSchedule;
  /**
   * Checked after every epoch. When it returns true the fit stops early and its
   * result is meaningless — used when the player has reset the forge and nobody
   * is waiting for this score any more, so the dead fit stops competing with the
   * live one for the GPU.
   */
  shouldStop?: () => boolean;
}

/** Train the fixed model on a design matrix and score it. */
export async function fitAndScore(
  matrices: { train: DesignMatrix; validation: DesignMatrix },
  modelFactory: (width: number) => tf.LayersModel = buildFixedModel,
  { schedule = scheduleFor(modelFactory), shouldStop }: FitOptions = {},
): Promise<FitResult> {
  const width = matrices.train.xs[0]?.length ?? 0;
  const model = modelFactory(width);

  const xs = tf.tensor2d(matrices.train.xs);
  const ys = tf.tensor2d(matrices.train.ys.map((y) => [y]));

  // Disposed in `finally` whatever happens: a failed fit (a lost WebGL context,
  // out of memory on a phone) must not strand its tensors, because the caller
  // recovers and the next forge fits again.
  try {
    await model.fit(xs, ys, {
      epochs: schedule.epochs,
      batchSize: schedule.batchSize,
      shuffle: false,
      verbose: 0,
      callbacks: shouldStop
        ? {
            onEpochEnd: () => {
              if (shouldStop()) model.stopTraining = true;
            },
          }
        : undefined,
    });

    const score = (matrix: DesignMatrix) =>
      tf.tidy(() => {
        const input = tf.tensor2d(matrix.xs);
        return accuracyFrom(
          (model.predict(input) as tf.Tensor).dataSync(),
          matrix.ys,
        );
      });

    const validationAccuracy = score(matrices.validation);
    const trainAccuracy = score(matrices.train);

    // Only meaningful for the linear model — a stack of layers has no single
    // weight per input column.
    const importances = tf.tidy(() => {
      const kernel = model.getWeights()[0];
      if (!kernel || kernel.shape[1] !== 1) {
        return new Array<number>(width).fill(0);
      }
      return Array.from((kernel.arraySync() as number[][]).map((row) =>
        Math.abs(row[0] ?? 0),
      ));
    });

    return {
      validationAccuracy,
      trainAccuracy,
      importances,
      columnNames: matrices.train.columnNames,
      width,
    };
  } finally {
    xs.dispose();
    ys.dispose();
    const optimizer = model.optimizer;
    model.dispose();
    optimizer?.dispose();
  }
}

// ── Scoring the forge ─────────────────────────────────────────────────────

/** Lift needed over baseline to call the forge a success. */
export const TARGET_LIFT = 0.06;
/** A forge this close to baseline has achieved nothing. */
export const NO_LIFT_BAND = 0.01;

/**
 * Feature sets the game recognises as "legendary" (spec: legendary combos).
 *
 * Not arbitrary flair — each is a transform that unlocks a relationship no amount
 * of model tuning can recover from the raw column, and the bonus is there to mark
 * the difference between changing units and changing meaning.
 *
 * Membership is earned by MEASUREMENT, which is why income-per-head is not on
 * the list even though "a ratio creates a quantity neither column contains"
 * sounds like the most legendary transform of all. Churn here jumps below a
 * per-head threshold, and one weight on income / household can only answer a
 * step with a straight line — which the baseline's weights on income and
 * household can already slope across the same boundary. Forged, the ratio moves
 * validation accuracy by less than the no-lift band (the tests pin this), so
 * rewarding it would teach the player to trust a feature's name over its
 * measured effect.
 */
export interface LegendaryCombo {
  id: string;
  label: string;
  why: string;
  /** What is still hidden in the table while this combo is missing. */
  hint: string;
  matches: (features: Feature[]) => boolean;
}

/** Is there a feature with this transform over exactly these columns, in order? */
const has = (
  features: Feature[],
  transform: Transform,
  ...columns: string[]
): boolean =>
  features.some(
    (feature) =>
      feature.transform === transform &&
      feature.sourceCols.length === columns.length &&
      columns.every((name, index) => feature.sourceCols[index] === name),
  );

export const LEGENDARY_COMBOS: readonly LegendaryCombo[] = [
  {
    id: "weekday",
    label: "Weekday reveal",
    why: "A timestamp is monotonic and meaningless; the weekday inside it is neither.",
    hint: "a timestamp hiding a weekday",
    matches: (features) => has(features, "day_of_week", "signup_ts"),
  },
  {
    id: "age-curve",
    label: "Age curve",
    why: "Both tails churn. Binning lets a straight-line model fit a U.",
    hint: "an effect that bends at both ends",
    matches: (features) => has(features, "bin", "age"),
  },
  {
    id: "city-unordered",
    label: "Order removed",
    why: "city_code 5 is not five times city_code 1. One-hot says so.",
    hint: "a category stored as a number",
    matches: (features) => has(features, "one_hot", "city_code"),
  },
] as const;

/** "a, b and c" — for copy that lists the combos still missing. */
export function listPhrases(phrases: readonly string[]): string {
  if (phrases.length <= 1) return phrases[0] ?? "";
  return `${phrases.slice(0, -1).join(", ")} and ${phrases[phrases.length - 1]}`;
}

/** Spelled out for copy: there are only ever a handful of combos. */
export function countWord(count: number): string {
  return (
    ["no", "one", "two", "three", "four", "five", "six"][count] ?? String(count)
  );
}

export function legendariesIn(features: Feature[]): LegendaryCombo[] {
  return LEGENDARY_COMBOS.filter((combo) => combo.matches(features));
}

export type Outcome = "forged" | "leakage" | "no-lift" | "short" | "unforged";

export interface Evaluation {
  outcome: Outcome;
  baselineScore: number;
  currentScore: number;
  lift: number;
  legendary: LegendaryCombo[];
  score: number;
  failure: NamedFailure | null;
}

const percent = (value: number) => `${Math.round(value * 100)}%`;
const points = (value: number) => `${(value * 100).toFixed(1)}%`;
const signed = (value: number) =>
  `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)} points`;

/**
 * Judge the forge.
 *
 * Leakage is checked first and unconditionally, even when the score is excellent —
 * especially when the score is excellent. A leaky feature set produces the best
 * number in the game and the least useful model in the world, and a game that
 * rewarded it would be teaching the single most expensive mistake in applied ML.
 */
export function evaluateForge({
  features,
  baselineScore,
  currentScore,
  submitted,
}: {
  features: Feature[];
  baselineScore: number;
  currentScore: number;
  submitted: boolean;
}): Evaluation {
  const lift = currentScore - baselineScore;
  const legendary = legendariesIn(features);
  const leaky = features.filter(usesLeakyColumn);

  const base = { baselineScore, currentScore, lift, legendary };

  if (features.length === 0) {
    return {
      ...base,
      outcome: "unforged",
      score: 0,
      failure: null,
    };
  }

  if (leaky.length > 0) {
    const names = leaky.map(describeFeature).join(", ");
    return {
      ...base,
      outcome: "leakage",
      score: 0,
      failure: {
        name: "Leakage",
        detail: `${points(
          currentScore,
        )} validation accuracy, and none of it is real. ${names} ${
          leaky.length === 1 ? "uses" : "use"
        } refund_issued, which is only ever set AFTER a customer has already left — 87% of churners have one and 2% of everyone else does. So the feature is very nearly the answer written on the back of the card. The score is the highest you will see in this game and the model is worth nothing: at prediction time, for a customer who has not churned yet, that column is always zero. Take it out of the forge.`,
      },
    };
  }

  if (!submitted) {
    return { ...base, outcome: "unforged", score: 0, failure: null };
  }

  if (lift >= TARGET_LIFT) {
    // The spec calls the legendary bonus a multiplier; here it is added, which
    // is easier to read off the score: 5 points per legendary combo, capped at
    // 20. It is only paid once the lift has cleared the target, so it can raise
    // a winning forge's score but never turn a short forge into a win.
    const bonus = Math.min(0.2, legendary.length * 0.05);
    return {
      ...base,
      outcome: "forged",
      score: clamp(0.55 + (lift / 0.2) * 0.45 + bonus, 0, 1),
      failure: null,
    };
  }

  if (Math.abs(lift) < NO_LIFT_BAND) {
    const unitsOnly = features.every((feature) =>
      ["standardise", "log_scale"].includes(feature.transform),
    );
    return {
      ...base,
      outcome: "no-lift",
      score: clamp(0.2 + lift, 0, 1),
      failure: {
        name: "No lift",
        detail: `${points(currentScore)} against a baseline of ${points(
          baselineScore,
        )} — ${signed(lift)}, which is nothing. ${
          unitsOnly
            ? "Every feature in the forge is a scaling transform. Scaling changes the units a column is measured in and leaves the relationship between that column and churn exactly as it was — and the baseline is already standardised, so there was nothing left to fix. A linear model cannot get more out of a straight line by being handed the same straight line in different units."
            : `The transforms in the forge are not giving the model anything it could not already compute from the baseline columns. The lift comes from changing what a column MEANS: ${listPhrases(
                LEGENDARY_COMBOS.map((combo) => combo.hint),
              )}.`
        }`,
      },
    };
  }

  return {
    ...base,
    outcome: "short",
    score: clamp(0.2 + (lift / TARGET_LIFT) * 0.3, 0, 1),
    failure: {
      name: lift < 0 ? "Worse than baseline" : "Not enough lift",
      detail:
        lift < 0
          ? `${points(currentScore)} against a baseline of ${points(
              baselineScore,
            )} — ${signed(
              lift,
            )}. Something in the forge is actively costing the model accuracy. The usual reason is width: every bin and every one-hot level is another column and another weight to fit from the same ${TRAIN_ROWS} rows. ${
              legendary.length === 0
                ? "None of the transforms here has unlocked a relationship yet, so the extra columns are all cost and no benefit."
                : `You have ${legendary.length} genuinely useful transform${
                    legendary.length === 1 ? "" : "s"
                  }, so try removing the ones that are not earning their width.`
            }`
          : `${points(currentScore)} against a baseline of ${points(
              baselineScore,
            )} — ${signed(lift)}, where ${percent(
              TARGET_LIFT,
            )} is needed. Real progress, not enough of it. ${(() => {
              const missing = LEGENDARY_COMBOS.filter(
                (combo) => !legendary.includes(combo),
              );
              if (legendary.length === 0) {
                return `None of the ${countWord(
                  LEGENDARY_COMBOS.length,
                )} relationships that need a genuine change of representation has been unlocked yet.`;
              }
              if (missing.length === 0) {
                return `You have found all ${countWord(
                  LEGENDARY_COMBOS.length,
                )} legendary relationships, so the shortfall is width: try removing the features that are not earning their columns.`;
              }
              return `You have found ${listPhrases(
                legendary.map((combo) => combo.label),
              )}. There ${
                missing.length === 1 ? "is one more" : `are ${countWord(missing.length)} more`
              } relationship${
                missing.length === 1 ? "" : "s"
              } in this table that no rescaling can reach: ${listPhrases(
                missing.map((combo) => combo.hint),
              )}.`;
            })()}`,
    },
  };
}

// ── Reveal the math ───────────────────────────────────────────────────────

export const MATH_EQUATION = String.raw`\hat{y} = \sigma\!\left(w^{\top}\phi(x) + b\right)
\\[1.2em]
\text{the model } (w, b, \sigma) \text{ is fixed} \quad\Longrightarrow\quad \text{only } \phi \text{ can change the score}`;

export const MATH_CODE = `// The model. It never changes — not the width, not the optimiser,
// not the seed. This is what makes a score difference attributable
// to the features and nothing else.
export function buildFixedModel(inputWidth) {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 1,                    // logistic regression, no hidden layer
    activation: 'sigmoid',
    inputShape: [inputWidth],    // <- the ONLY thing your features change
    kernelInitializer: tf.initializers.glorotNormal({ seed: ${MODEL_SEED} }),
  }));
  model.compile({
    optimizer: tf.train.adam(${LEARNING_RATE}),
    loss: 'binaryCrossentropy',
  });
  return model;
}

// Statistics are fitted on TRAIN and applied to validation. Fitting them
// on both would leak the validation set into the features and report a
// lift that does not exist outside this notebook.
const stats = fitStats(trainRows, columnIndex);
const value = (row[columnIndex] - stats.mean) / stats.sd;`;

export const MATH_NOTES = `A model of this shape can only add things up. It multiplies each column by a weight and sums. That is the entire hypothesis space, and it is why the columns you hand it decide what it can possibly learn.

Standardising a column changes its units. It divides by a constant, so the weight simply multiplies by that constant and the model ends up in the same place — which is why a forge full of scaling transforms produces no lift over a baseline that was already standardised. Nothing about the relationship changed.

Binning is different in kind. Cutting age into ${BIN_COUNT} ranges and one-hot encoding them gives the model ${BIN_COUNT} independent weights, so it can score the under-${AGE_YOUNG}s and the over-${AGE_OLD}s as high risk while the middle stays low. No single weight on raw age can do that, because a single weight can only say "more age, more churn" or "less age, more churn". The transform did not add information — it made information reachable.

One-hot encoding removes a claim rather than adding one. Storing city as 0 to 5 tells the model that city 5 is five times city 1, which is false. Splitting it into six columns withdraws that claim.

A ratio is the instructive near miss. Income of 60,000 is comfortable for one person and thin for six, and churn here really does jump when income per head falls below ${PER_HEAD_THRESHOLD.toLocaleString(
  "en-US",
)}. That jump is a step. A ratio column gets one weight, like every other column, so the most the model can do with income / household is a straight-line response: the lower it is, the more churn. Dividing changed the axis, not the shape. And the baseline already had a line across that same boundary, because "income / household < ${PER_HEAD_THRESHOLD.toLocaleString(
  "en-US",
)}" picks out exactly the customers for whom "income − ${PER_HEAD_THRESHOLD.toLocaleString(
  "en-US",
)} × household < 0", and a weight on income and a weight on household can slope across that line. Neither the ratio nor the baseline can make the step; each can only slope across it. So the ratio measures no lift. Catching the step itself would take a cut on the ratio, which this forge does not offer. The lesson is to judge a feature by what it lets the model express that it could not before, not by how clever it sounds.

The one thing worth more than all of this is knowing which columns you are allowed to use at all. refund_issued predicts churn almost perfectly and is worthless, because it is caused by churn. That is leakage, and it is the most expensive mistake in applied machine learning precisely because it looks like the best result you have ever had.`;
