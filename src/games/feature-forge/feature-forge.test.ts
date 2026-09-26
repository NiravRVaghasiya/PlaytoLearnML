import * as tf from "@tensorflow/tfjs";
import { beforeAll, describe, expect, it } from "vitest";
import {
  AGE_OLD,
  AGE_YOUNG,
  BIN_COUNT,
  CITY_LEVELS,
  COLUMNS,
  COLUMN_INDEX,
  FANCIER_SCHEDULE,
  FIXED_SCHEDULE,
  LEGENDARY_COMBOS,
  MATH_NOTES,
  NO_LIFT_BAND,
  PER_HEAD_THRESHOLD,
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
  featureProblem,
  fitAndScore,
  generateDataset,
  isValidFeature,
  legendariesIn,
  scheduleFor,
  usesLeakyColumn,
  type DesignMatrix,
  type Feature,
  type Row,
  type Transform,
} from "./ml";
import { whyCardFor } from "./why-cards";

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

/** The transforms that unlock a relationship no rescaling can reach. */
const LEGENDARY = [
  f("day_of_week", "signup_ts"),
  f("bin", "age"),
  f("one_hot", "city_code"),
];
const FORGED = [...baselineFeatures(), ...LEGENDARY];

/** Income per head: sounds legendary, measures nothing here. */
const PER_HEAD = f("ratio", "income", "household");

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

  it("lifts well past the target when every legendary relationship is unlocked", async () => {
    const baseline = await score(baselineFeatures()); // measured 0.631
    const forged = await score(FORGED); // measured 0.711

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

describe("the weekday lives where both lanes look for it", () => {
  it("puts the weekend effect on days 0 and 6 of (ts // 86400) % 7", () => {
    // The base timestamp used to be 22:13 on a day that arithmetic calls 5, so
    // the effect sat on days 5 and 6 while the Python starter pointed at 0 and 6.
    const index = COLUMN_INDEX.signup_ts!;
    const rate = (day: number) => {
      const rows = dataset.train.filter(
        (row) => Math.floor(row.values[index]! / 86_400) % 7 === day,
      );
      return rows.filter((row) => row.churned === 1).length / rows.length;
    };
    const weekend = Math.min(rate(0), rate(6));
    for (const day of [1, 2, 3, 4, 5]) {
      expect(rate(day), `day ${day}`).toBeLessThan(weekend);
    }
  });
});

describe("income per head: a feature judged by what it measures", () => {
  it("is a threshold that is also a straight line in the raw columns", () => {
    // The claim the copy makes, checked row by row: "income / household < t"
    // and "income - t * household < 0" pick out exactly the same customers.
    const income = COLUMN_INDEX.income!;
    const household = COLUMN_INDEX.household!;
    for (const row of [...dataset.train, ...dataset.validation]) {
      const ratio = row.values[income]! / row.values[household]! < PER_HEAD_THRESHOLD;
      const line =
        row.values[income]! - PER_HEAD_THRESHOLD * row.values[household]! < 0;
      expect(ratio).toBe(line);
    }
  });

  it("is not legendary, in either order", () => {
    expect(legendariesIn([PER_HEAD])).toHaveLength(0);
    expect(legendariesIn([f("ratio", "household", "income")])).toHaveLength(0);
  });

  it("measures no lift, on its own or on top of the legendary set", async () => {
    // Pinned because the copy says so: if the data ever changes so that the
    // ratio does earn its column, this fails and the copy has to change too.
    const baseline = await score(baselineFeatures());
    const alone = await score([...baselineFeatures(), PER_HEAD]);
    const forged = await score(FORGED);
    const withRatio = await score([...FORGED, PER_HEAD]);

    expect(Math.abs(alone - baseline)).toBeLessThan(NO_LIFT_BAND);
    expect(Math.abs(withRatio - forged)).toBeLessThan(NO_LIFT_BAND);
  }, 300000);

  it("is a step the baseline only partly reaches: a cut on the ratio still lifts", async () => {
    // The copy used to say the model "could draw it before you divided
    // anything". It can lean a line across the boundary, not make the step, so
    // a 0/1 column for the threshold itself — the cut this forge does not
    // offer — adds real accuracy on top of the legendary set where the ratio
    // adds none. Pinned at DATA_SEED only: with 800 validation rows the size of
    // this lift is noisy across seeds, which is why the copy never quotes it.
    const income = COLUMN_INDEX.income!;
    const household = COLUMN_INDEX.household!;
    const cut = (row: Row) =>
      row.values[income]! / row.values[household]! < PER_HEAD_THRESHOLD ? 1 : 0;
    const withCut = (matrix: DesignMatrix, rows: Row[]): DesignMatrix => ({
      xs: matrix.xs.map((xs, index) => [...xs, cut(rows[index]!)]),
      ys: matrix.ys,
      columnNames: [...matrix.columnNames, "income / household < threshold"],
    });

    const matrices = buildMatrices(dataset, FORGED)!;
    const forged = (await fitAndScore(matrices)).validationAccuracy;
    const cutScore = (
      await fitAndScore({
        train: withCut(matrices.train, dataset.train),
        validation: withCut(matrices.validation, dataset.validation),
      })
    ).validationAccuracy;

    expect(cutScore - forged).toBeGreaterThan(NO_LIFT_BAND);
  }, 300000);

  it("is explained as a straight line against a step, never as already drawn", () => {
    const threshold = PER_HEAD_THRESHOLD.toLocaleString("en-US");
    const context = {
      baselineScore: 0.63,
      trainScore: 0.72,
      features: [...LEGENDARY, PER_HEAD],
      legendary: LEGENDARY_COMBOS.slice(),
      leaked: false,
    };
    const cardFor = (previousScore: number, currentScore: number) =>
      whyCardFor({
        kind: "forged-feature",
        feature: PER_HEAD,
        previousScore,
        currentScore,
        ...context,
      });

    const flat = cardFor(0.71, 0.7025);
    expect(flat.body).toMatch(/step/);
    expect(flat.body).toMatch(/straight line/);
    expect(flat.body).toMatch(/adds nothing measurable/);
    expect(flat.body).toContain(threshold);
    expect(flat.body).toMatch(/cut on the ratio/);

    // A drop beyond the band is named as the cost it was, not "nothing".
    const costly = cardFor(0.71, 0.69);
    expect(costly.body).toMatch(/step/);
    expect(costly.body).not.toMatch(/adds nothing measurable/);
    expect(costly.body).toContain("cost 2.0 points");

    // A ratio that genuinely lifted the score would not get this explanation.
    expect(cardFor(0.7, 0.72).body).not.toMatch(/straight line/);

    for (const text of [flat.body, costly.body, MATH_NOTES]) {
      expect(text).not.toMatch(/before you divided|does not need to|can already draw/);
    }
    expect(MATH_NOTES).toMatch(/step/);
  });
});

describe("the legendary transforms compound", () => {
  it("are worth more together than the sum of their separate lifts", async () => {
    // The legendary WhyCard says so; this is the measurement behind it.
    const baseline = await score(baselineFeatures());
    let separate = 0;
    for (const feature of LEGENDARY) {
      separate += (await score([...baselineFeatures(), feature])) - baseline;
    }
    const together = (await score(FORGED)) - baseline;
    expect(together).toBeGreaterThan(separate);
  }, 300000);
});

describe("the fixed model trains fast enough to feel live", () => {
  it("takes full-batch steps, a hundred of them", () => {
    // The Definition of Done asks for the metric to move within about a second
    // of a forge. Sequential optimiser steps are what a fit costs on WebGL, and
    // this schedule is ~7× fewer of them than the mini-batch one it replaced.
    const steps =
      FIXED_SCHEDULE.epochs * Math.ceil(TRAIN_ROWS / FIXED_SCHEDULE.batchSize);
    expect(steps).toBeLessThanOrEqual(100);
    expect(scheduleFor(buildFixedModel)).toBe(FIXED_SCHEDULE);
    expect(scheduleFor(buildFancierModel)).toBe(FANCIER_SCHEDULE);
  });

  it("stops early and leaks nothing when told its result is no longer wanted", async () => {
    const before = tf.memory().numTensors;
    let epochs = 0;
    await fitAndScore(buildMatrices(dataset, baselineFeatures())!, undefined, {
      shouldStop: () => {
        epochs += 1;
        return true;
      },
    });
    expect(epochs).toBe(1);
    expect(tf.memory().numTensors).toBe(before);
  });
});

describe("why a feature cannot be forged, in words", () => {
  it("names an unknown transform and lists the real ones", () => {
    expect(featureProblem("binn", ["age"])).toMatch(/unknown transform "binn".*bin/);
  });

  it("names an unknown column and lists the table", () => {
    expect(featureProblem("bin", ["agee"])).toMatch(/no column called "agee".*age/);
  });

  it("names the wrong arity", () => {
    expect(featureProblem("ratio", ["income"])).toMatch(/ratio takes 2 columns, got 1/);
  });

  it("names a dtype the transform will not take", () => {
    expect(featureProblem("bin", ["city_code"])).toMatch(
      /bin needs a numeric column, and city_code is a categorical/,
    );
    expect(featureProblem("one_hot", ["city"])).toMatch(/no column called "city"/);
  });

  it("names a duplicate", () => {
    expect(featureProblem("bin", ["age"], [f("bin", "age")])).toMatch(/already in the forge/);
  });

  it("says nothing for a legal feature", () => {
    expect(featureProblem("bin", ["age"])).toBeNull();
    expect(featureProblem("ratio", ["income", "household"])).toBeNull();
  });
});

describe("failure copy names what is actually missing", () => {
  const baseline = 0.63;

  it("lists only the relationships still to find", () => {
    const evaluation = evaluateForge({
      features: [...baselineFeatures(), f("bin", "age")],
      baselineScore: baseline,
      currentScore: baseline + 0.03,
      submitted: true,
    });
    const missing = LEGENDARY_COMBOS.filter((combo) => combo.id !== "age-curve");
    for (const combo of missing) expect(evaluation.failure!.detail).toContain(combo.hint);
    expect(evaluation.failure!.detail).not.toContain(
      LEGENDARY_COMBOS.find((combo) => combo.id === "age-curve")!.hint,
    );
    expect(evaluation.failure!.detail).not.toMatch(/four/);
  });

  it("describes the ages the generator really uses", () => {
    expect(MATH_NOTES).toContain(`under-${AGE_YOUNG}s`);
    expect(MATH_NOTES).toContain(`over-${AGE_OLD}s`);
    expect(MATH_NOTES).not.toMatch(/under-26s|over-64s/);
  });
});
