import * as tf from "@tensorflow/tfjs";
import { beforeAll, describe, expect, it } from "vitest";
import {
  BIN_COUNT,
  CITY_LEVELS,
  COLUMNS,
  LEGENDARY_COMBOS,
  NO_LIFT_BAND,
  TARGET_LIFT,
  TRAIN_ROWS,
  VALIDATION_ROWS,
  baselineFeatures,
  buildFancierModel,
  buildFixedModel,
  buildMatrices,
  columnByName,
  describeFeature,
  evaluateForge,
  fitAndScore,
  generateDataset,
  isValidFeature,
  legendariesIn,
  usesLeakyColumn,
  type Feature,
  type Transform,
} from "./ml";

beforeAll(async () => {
  await tf.ready();
});

const DATA_SEED = 5309;
const dataset = generateDataset(DATA_SEED);

const f = (transform: Transform, ...sourceCols: string[]): Feature => ({
  id: `${transform}:${sourceCols.join("+")}`,
  transform,
  sourceCols,
});

/** The four transforms that unlock a relationship no rescaling can reach. */
const FORGED = [
  ...baselineFeatures(),
  f("day_of_week", "signup_ts"),
  f("bin", "age"),
  f("one_hot", "city_code"),
  f("ratio", "income", "household"),
];

async function score(
  features: Feature[],
  factory?: typeof buildFancierModel,
): Promise<number> {
  const matrices = buildMatrices(dataset, features);
  expect(matrices, `features produced no matrix: ${features.map(describeFeature).join(", ")}`).not.toBeNull();
  const result = await fitAndScore(matrices!, factory);
  return result.validationAccuracy;
}

describe("the customer table", () => {
  it("is deterministic for a seed", () => {
    expect(generateDataset(DATA_SEED).train).toEqual(
      generateDataset(DATA_SEED).train,
    );
  });

  it("holds validation apart from training", () => {
    expect(dataset.train).toHaveLength(TRAIN_ROWS);
    expect(dataset.validation).toHaveLength(VALIDATION_ROWS);
    const seen = new Set(dataset.train.map((row) => row.values.join(",")));
    expect(
      dataset.validation.filter((row) => seen.has(row.values.join(","))),
    ).toHaveLength(0);
  });

  it("keeps churn balanced enough for accuracy to mean something", () => {
    const rate =
      dataset.validation.filter((row) => row.churned === 1).length /
      dataset.validation.length;
    expect(rate).toBeGreaterThan(0.35);
    expect(rate).toBeLessThan(0.65);
  });

  it("marks exactly one column as leaky, and it is a consequence of churn", () => {
    const leaky = COLUMNS.filter((column) => column.leaky);
    expect(leaky).toHaveLength(1);
    expect(leaky[0]!.name).toBe("refund_issued");

    // The correlation is what makes it dangerous: it is nearly the answer.
    const index = COLUMNS.findIndex((column) => column.leaky);
    const churners = dataset.train.filter((row) => row.churned === 1);
    const staying = dataset.train.filter((row) => row.churned === 0);
    const rateAmong = (rows: typeof churners) =>
      rows.filter((row) => row.values[index] === 1).length / rows.length;

    expect(rateAmong(churners)).toBeGreaterThan(0.8);
    expect(rateAmong(staying)).toBeLessThan(0.06);
  });
});

describe("features and transforms", () => {
  it("rejects a transform given the wrong number of columns", () => {
    expect(isValidFeature(f("ratio", "income"))).toBe(false);
    expect(isValidFeature(f("ratio", "income", "household"))).toBe(true);
    expect(isValidFeature(f("bin", "age", "income"))).toBe(false);
  });

  it("rejects a transform given the wrong dtype", () => {
    expect(isValidFeature(f("day_of_week", "age"))).toBe(false);
    expect(isValidFeature(f("day_of_week", "signup_ts"))).toBe(true);
    expect(isValidFeature(f("one_hot", "age"))).toBe(false);
    expect(isValidFeature(f("one_hot", "city_code"))).toBe(true);
  });

  it("lets a category be standardised, because that is the mistake being taught", () => {
    // If this were rejected, city_code would be silently missing from the baseline
    // and one-hot encoding would have nothing to fix.
    expect(isValidFeature(f("standardise", "city_code"))).toBe(true);
    expect(columnByName("city_code")!.dtype).toBe("categorical");
  });

  it("spots a leaky column wherever it appears", () => {
    expect(usesLeakyColumn(f("standardise", "refund_issued"))).toBe(true);
    expect(usesLeakyColumn(f("ratio", "income", "refund_issued"))).toBe(true);
    expect(usesLeakyColumn(f("bin", "age"))).toBe(false);
  });

  it("names a feature the way the forge shows it", () => {
    expect(describeFeature(f("ratio", "income", "household"))).toBe(
      "income / household",
    );
    expect(describeFeature(f("bin", "age"))).toBe("bin(age)");
  });
});

describe("the design matrix", () => {
  it("puts the baseline's mistake in it: city_code as a number", () => {
    const baseline = baselineFeatures();
    const cityFeature = baseline.find((feature) =>
      feature.sourceCols.includes("city_code"),
    );
    expect(cityFeature, "city_code missing from the baseline").toBeDefined();
    expect(cityFeature!.transform).toBe("standardise");

    const matrices = buildMatrices(dataset, baseline)!;
    expect(matrices.train.xs[0]).toHaveLength(baseline.length);
  });

  it("excludes the raw timestamp and the leaky column from the baseline", () => {
    const names = baselineFeatures().flatMap((feature) => feature.sourceCols);
    expect(names).not.toContain("signup_ts");
    expect(names).not.toContain("refund_issued");
  });

  it("widens by the right amount per transform", () => {
    const widthOf = (feature: Feature) =>
      buildMatrices(dataset, [feature])!.train.xs[0]!.length;

    expect(widthOf(f("standardise", "age"))).toBe(1);
    expect(widthOf(f("log_scale", "income"))).toBe(1);
    expect(widthOf(f("ratio", "income", "household"))).toBe(1);
    expect(widthOf(f("bin", "age"))).toBe(BIN_COUNT);
    expect(widthOf(f("one_hot", "city_code"))).toBe(CITY_LEVELS);
    expect(widthOf(f("day_of_week", "signup_ts"))).toBe(7);
  });

  it("one-hots to exactly one live column per row", () => {
    const matrices = buildMatrices(dataset, [f("one_hot", "city_code")])!;
    for (const row of matrices.validation.xs) {
      expect(row.reduce((total, value) => total + value, 0)).toBe(1);
    }
  });

  it("fits statistics on training rows and APPLIES them to validation", () => {
    // The subtle one. Standardising with statistics from the whole table would leak
    // the validation set into the features and report a lift that does not exist
    // outside this run. Train should centre near zero; validation should not be
    // forced to.
    const matrices = buildMatrices(dataset, [f("standardise", "income")])!;
    const meanOf = (rows: number[][]) =>
      rows.reduce((total, row) => total + (row[0] ?? 0), 0) / rows.length;

    expect(Math.abs(meanOf(matrices.train.xs))).toBeLessThan(1e-6);
    // Validation has its own mean, close to but not exactly zero.
    expect(Math.abs(meanOf(matrices.validation.xs))).toBeGreaterThan(0);
    expect(Math.abs(meanOf(matrices.validation.xs))).toBeLessThan(0.3);
  });

  it("returns nothing when no feature is valid", () => {
    expect(buildMatrices(dataset, [])).toBeNull();
    expect(buildMatrices(dataset, [f("one_hot", "age")])).toBeNull();
  });

  it("labels every matrix column, for the importance bars", () => {
    const matrices = buildMatrices(dataset, [
      f("bin", "age"),
      f("one_hot", "city_code"),
    ])!;
    expect(matrices.train.columnNames).toHaveLength(BIN_COUNT + CITY_LEVELS);
    expect(new Set(matrices.train.columnNames).size).toBe(
      BIN_COUNT + CITY_LEVELS,
    );
  });
});

describe("the model is held still", () => {
  it("is a logistic regression: one weight per column plus a bias", () => {
    for (const width of [1, 5, 24]) {
      const model = buildFixedModel(width);
      expect(model.countParams()).toBe(width + 1);
      expect(model.outputs[0]!.shape[1]).toBe(1);
      model.dispose();
    }
  });

  it("builds identically twice, so a score change is never the model", () => {
    const a = buildFixedModel(6);
    const b = buildFixedModel(6);
    const weightsOf = (model: tf.LayersModel) =>
      model.getWeights().map((tensor) => Array.from(tensor.dataSync()));
    expect(weightsOf(a)).toEqual(weightsOf(b));
    a.dispose();
    b.dispose();
  });

  it("leaks no tensors", () => {
    const before = tf.memory().numTensors;
    const model = buildFixedModel(8);
    const optimizer = model.optimizer;
    model.dispose();
    optimizer?.dispose();
    expect(tf.memory().numTensors).toBe(before);
  });
});

describe("representation changes the score — measured", () => {
  // These train the real fixed model. The numbers in the comments are what was
  // measured while tuning.

  it("lifts well past the target when all four relationships are unlocked", async () => {
    const baseline = await score(baselineFeatures()); // measured 0.635
    const forged = await score(FORGED); // measured 0.706

    expect(forged - baseline).toBeGreaterThanOrEqual(TARGET_LIFT);
    expect(legendariesIn(FORGED)).toHaveLength(LEGENDARY_COMBOS.length);
  }, 300000);

  it("gives no lift for transforms that only change units", async () => {
    // The baseline is already standardised, so rescaling it again is a no-op in
    // everything but arithmetic. This is the "No lift" lesson as a measurement.
    const baseline = await score(baselineFeatures());
    const rescaled = await score([
      ...baselineFeatures(),
      f("log_scale", "income"),
      f("standardise", "age"),
    ]);

    expect(Math.abs(rescaled - baseline)).toBeLessThan(TARGET_LIFT);
  }, 300000);

  it("makes leakage look like the best result in the game", async () => {
    const baseline = await score(baselineFeatures());
    const leaky = await score([
      ...baselineFeatures(),
      f("standardise", "refund_issued"),
    ]); // measured 0.910

    expect(leaky - baseline).toBeGreaterThan(0.2);
    expect(leaky).toBeGreaterThan(0.85);
  }, 300000);

  it("beats a fancier model given the raw columns — the core intuition", async () => {
    // The claim the game is built on: representation matters more than algorithm
    // choice on tabular data. A 32x16 relu network on the baseline columns against
    // a plain logistic regression on forged features.
    const fancierOnBaseline = await score(baselineFeatures(), buildFancierModel);
    const fixedOnForged = await score(FORGED);

    expect(fixedOnForged).toBeGreaterThan(fancierOnBaseline);
    expect(fixedOnForged - fancierOnBaseline).toBeGreaterThan(0.03);
  }, 300000);

  it("shows the fancier model overfitting where good features do not", async () => {
    // Worth stating: the bigger model does not merely lose, it loses by memorising.
    const fancy = await fitAndScore(
      buildMatrices(dataset, baselineFeatures())!,
      buildFancierModel,
    );
    const forged = await fitAndScore(buildMatrices(dataset, FORGED)!);

    expect(fancy.trainAccuracy - fancy.validationAccuracy).toBeGreaterThan(0.05);
    expect(
      Math.abs(forged.trainAccuracy - forged.validationAccuracy),
    ).toBeLessThan(0.05);
  }, 300000);

  it("reports one importance per matrix column for the linear model", async () => {
    const result = await fitAndScore(buildMatrices(dataset, FORGED)!);
    expect(result.importances).toHaveLength(result.width);
    expect(result.columnNames).toHaveLength(result.width);
    expect(result.importances.every((value) => value >= 0)).toBe(true);
  }, 300000);
});

describe("legendary combos", () => {
  it("recognises each one only when its own transform is present", () => {
    expect(legendariesIn([f("day_of_week", "signup_ts")])).toHaveLength(1);
    expect(legendariesIn([f("bin", "age")])).toHaveLength(1);
    expect(legendariesIn([f("one_hot", "city_code")])).toHaveLength(1);
    expect(legendariesIn([f("ratio", "income", "household")])).toHaveLength(1);
    expect(legendariesIn(baselineFeatures())).toHaveLength(0);
  });

  it("does not count a scaling transform as legendary", () => {
    expect(legendariesIn([f("standardise", "age")])).toHaveLength(0);
    expect(legendariesIn([f("log_scale", "income")])).toHaveLength(0);
  });

  it("gives every combo a reason, not just a name", () => {
    for (const combo of LEGENDARY_COMBOS) {
      expect(combo.why.length, combo.id).toBeGreaterThan(30);
    }
  });
});

describe("naming the failure", () => {
  const baseline = 0.635;

  it("names Leakage first, even when the score is the best in the game", () => {
    // Especially then. A game that rewarded this would be teaching the most
    // expensive mistake in applied ML.
    const evaluation = evaluateForge({
      features: [...baselineFeatures(), f("standardise", "refund_issued")],
      baselineScore: baseline,
      currentScore: 0.91,
      submitted: true,
    });

    expect(evaluation.outcome).toBe("leakage");
    expect(evaluation.failure?.name).toBe("Leakage");
    expect(evaluation.score, "leakage must score nothing").toBe(0);
    expect(evaluation.failure?.detail).toMatch(/refund_issued/);
    expect(evaluation.failure?.detail).toMatch(/after/i);
  });

  it("names Leakage even before the forge is submitted", () => {
    const evaluation = evaluateForge({
      features: [f("standardise", "refund_issued")],
      baselineScore: baseline,
      currentScore: 0.91,
      submitted: false,
    });
    expect(evaluation.outcome).toBe("leakage");
  });

  it("names No lift when only the units changed", () => {
    const evaluation = evaluateForge({
      features: [...baselineFeatures(), f("log_scale", "income")],
      baselineScore: baseline,
      currentScore: baseline + 0.004,
      submitted: true,
    });

    expect(evaluation.outcome).toBe("no-lift");
    expect(evaluation.failure?.name).toBe("No lift");
    expect(evaluation.failure?.detail).toMatch(/units/i);
    expect(evaluation.failure?.detail).toMatch(/already standardised/i);
  });

  it("distinguishes a near miss from no lift at all", () => {
    const evaluation = evaluateForge({
      features: [...baselineFeatures(), f("bin", "age")],
      baselineScore: baseline,
      currentScore: baseline + 0.03,
      submitted: true,
    });

    expect(evaluation.outcome).toBe("short");
    expect(evaluation.failure?.name).toBe("Not enough lift");
    // It should say how many relationships remain, having found one.
    expect(evaluation.failure?.detail).toMatch(/Age curve/);
  });

  it("names a regression as worse than baseline, and blames width", () => {
    const evaluation = evaluateForge({
      features: [...baselineFeatures(), f("bin", "age")],
      baselineScore: baseline,
      currentScore: baseline - 0.05,
      submitted: true,
    });

    expect(evaluation.outcome).toBe("short");
    expect(evaluation.failure?.name).toBe("Worse than baseline");
    expect(evaluation.failure?.detail).toMatch(/width/i);
  });

  it("forges successfully past the target, and pays for legendary combos", () => {
    const plain = evaluateForge({
      features: [...baselineFeatures(), f("standardise", "support_tickets")],
      baselineScore: baseline,
      currentScore: baseline + TARGET_LIFT + 0.01,
      submitted: true,
    });
    const legendary = evaluateForge({
      features: FORGED,
      baselineScore: baseline,
      currentScore: baseline + TARGET_LIFT + 0.01,
      submitted: true,
    });

    expect(plain.outcome).toBe("forged");
    expect(legendary.outcome).toBe("forged");
    expect(legendary.score).toBeGreaterThan(plain.score);
    expect(legendary.legendary).toHaveLength(LEGENDARY_COMBOS.length);
  });

  it("reports nothing before anything is forged", () => {
    const evaluation = evaluateForge({
      features: [],
      baselineScore: baseline,
      currentScore: baseline,
      submitted: false,
    });
    expect(evaluation.outcome).toBe("unforged");
    expect(evaluation.failure).toBeNull();
  });

  it("keeps every score inside the unit interval", () => {
    for (const current of [0, 0.3, 0.635, 0.7, 0.95, 1]) {
      for (const submitted of [true, false]) {
        const evaluation = evaluateForge({
          features: FORGED,
          baselineScore: baseline,
          currentScore: current,
          submitted,
        });
        expect(evaluation.score).toBeGreaterThanOrEqual(0);
        expect(evaluation.score).toBeLessThanOrEqual(1);
      }
    }
  });

  it("caps the legendary bonus so it cannot outweigh the lift", () => {
    const evaluation = evaluateForge({
      features: FORGED,
      baselineScore: baseline,
      currentScore: baseline + TARGET_LIFT,
      submitted: true,
    });
    expect(evaluation.score).toBeLessThan(1);
    expect(NO_LIFT_BAND).toBeLessThan(TARGET_LIFT);
  });
});
