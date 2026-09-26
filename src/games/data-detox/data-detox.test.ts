import * as tf from "@tensorflow/tfjs";
import { renderHook } from "@testing-library/react";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import {
  EMPTY_PROGRESSION,
  HIGH_SCORE_THRESHOLD,
  createMemoryAdapter,
  useProgression,
} from "@/engine/progression";
import {
  BATCH_SIZE,
  FEATURES,
  FENCE_K,
  LABEL_NOISE,
  MAX_BALANCE_DRIFT,
  MAX_TIME_PENALTY,
  MIN_ROWS,
  NULL_RATE_BLIGHTED,
  NULL_RATE_HEALTHY,
  OUTLIER_MAX,
  OUTLIER_MIN,
  ROUND_SEEDS,
  SENSOR_MAX,
  SENSOR_MIN,
  TEST_ROWS,
  TRAIN_BATCH,
  TRAIN_EPOCHS,
  TRAIN_ROWS,
  WIN_ACCURACY,
  accuracyFromPredictions,
  applyPipeline,
  buildModel,
  columnStats,
  evaluate,
  generateDataset,
  idealAction,
  isOutOfRange,
  labelFor,
  seedForRound,
  testMatrix,
  timePenaltyFor,
  type CleaningAction,
  type Dataset,
  type Row,
} from "./ml";
import { createCodeApi, useAttemptClock, useDataDetoxStore } from "./store";
import { whyCardFor } from "./why-cards";

beforeAll(async () => {
  await tf.ready();
});

const withAction = (rows: Row[], choose: (row: Row) => CleaningAction): Row[] =>
  rows.map((row) => ({ ...row, playerAction: choose(row) }));

/**
 * Train once and return held-out accuracy, disposing everything.
 *
 * Mirrors what `index.tsx` does through `useModel`, so the numbers these tests
 * assert are the numbers the player sees.
 */
async function trainAndScore(rows: Row[], dataset: Dataset): Promise<number> {
  const pipeline = applyPipeline(rows, columnStats(dataset.train));
  if (pipeline.kept === 0) return 0;

  const model = buildModel(1);
  const xs = tf.tensor2d(pipeline.xs);
  const ys = tf.tensor2d(pipeline.ys.map((y) => [y]));

  await model.fit(xs, ys, {
    epochs: TRAIN_EPOCHS,
    batchSize: TRAIN_BATCH,
    shuffle: false,
    verbose: 0,
  });

  const test = testMatrix(dataset);
  const predictions = tf.tidy(() => {
    const input = tf.tensor2d(test.xs);
    return (model.predict(input) as tf.Tensor).dataSync();
  });
  const accuracy = accuracyFromPredictions(predictions, test.ys);

  xs.dispose();
  ys.dispose();
  const optimizer = model.optimizer;
  model.dispose();
  optimizer?.dispose();

  return accuracy;
}

// ═══════════════════════════════════════════════════════════════════════════
// THE LESSON
//
// "Garbage in = garbage out; cleaning decisions directly move model quality, and
// there's no single right answer — only tradeoffs."
//
// The load-bearing design choice is that missingness is NOT random: blanks favour
// the healthy class. That's what makes dropping harmful for a reason that survives
// more data and simpler models. These assertions are what stop the game from
// teaching "just throw the mess away".
// ═══════════════════════════════════════════════════════════════════════════

describe("the lesson holds on every shipped round", () => {
  it.each([...ROUND_SEEDS])(
    "seed %i teaches the intended lesson",
    async (seed) => {
      const dataset = generateDataset(seed);
      const stats = columnStats(dataset.train);

      const pipelineFor = (choose: (row: Row) => CleaningAction) =>
        applyPipeline(withAction(dataset.train, choose), stats);

      // 1. A strong pipeline wins: no dirt left, no bias, accuracy over the bar.
      const idealPipeline = pipelineFor(idealAction);
      const idealAccuracy = await trainAndScore(
        withAction(dataset.train, idealAction),
        dataset,
      );
      const ideal = evaluate(idealPipeline, idealAccuracy, 0);

      expect(ideal.outcome, `ideal on seed ${seed}`).toBe("win");
      expect(ideal.keptWithNaiveFill).toBe(0);
      expect(Math.abs(ideal.balanceDrift)).toBeLessThanOrEqual(
        MAX_BALANCE_DRIFT,
      );
      expect(ideal.score!).toBeGreaterThanOrEqual(WIN_ACCURACY);

      // 2. Keeping everything is garbage in: the blanks became zeros.
      const keepAll = evaluate(
        pipelineFor(() => "keep"),
        await trainAndScore(
          withAction(dataset.train, () => "keep"),
          dataset,
        ),
        0,
      );
      expect(keepAll.outcome, `keep-all on seed ${seed}`).toBe(
        "unhandled-nulls",
      );
      expect(keepAll.keptWithNaiveFill).toBeGreaterThan(0);

      // 3. THE TRAP: dropping the rows with blanks skews the class balance,
      //    because blanks aren't spread evenly across the classes. This is the
      //    precise case — it drops ONLY for missingness, keeps the sample size
      //    comfortably above starvation, and still bends the distribution, so the
      //    bias can't be mistaken for a sample-size effect.
      const dropNulls = (row: Row): CleaningAction =>
        row.isNull ? "drop" : row.isOutlier ? "cap" : "keep";
      const biased = evaluate(
        pipelineFor(dropNulls),
        await trainAndScore(withAction(dataset.train, dropNulls), dataset),
        0,
      );

      expect(biased.outcome, `drop-nulls on seed ${seed}`).toBe(
        "selection-bias",
      );
      expect(biased.kept, `drop-nulls kept on seed ${seed}`).toBeGreaterThan(
        MIN_ROWS,
      );
      expect(Math.abs(biased.balanceDrift)).toBeGreaterThan(MAX_BALANCE_DRIFT);
      // The surviving sample under-represents the healthy class specifically.
      expect(biased.balanceDrift).toBeLessThan(0);

      // 4. Dropping everything starves the model outright — no training needed
      //    to say so, since there is nothing left to train on.
      const dropAll = evaluate(
        pipelineFor(() => "drop"),
        null,
        0,
      );
      expect(dropAll.outcome, `drop-all on seed ${seed}`).toBe("starved");
      expect(dropAll.kept).toBeLessThan(MIN_ROWS);

      // 5. And the ideal pipeline genuinely beats the naive one on accuracy —
      //    the metric has to agree with the diagnosis.
      expect(idealAccuracy).toBeGreaterThan(keepAll.accuracy!);

      // 6. Leaving the spikes in is caught as outlier contamination, AND it
      //    genuinely costs accuracy. This is the check the first design failed:
      //    with stretched outliers and percentile caps, keeping them scored the
      //    same as capping them, so the named failure contradicted the meter.
      const keepSpikes = (row: Row): CleaningAction =>
        row.isNull ? "impute" : "keep";
      const spiked = evaluate(
        pipelineFor(keepSpikes),
        await trainAndScore(withAction(dataset.train, keepSpikes), dataset),
        0,
      );
      expect(spiked.outcome, `keep-spikes on seed ${seed}`).toBe(
        "outlier-contamination",
      );
      expect(
        idealAccuracy - spiked.accuracy!,
        `capping beats keeping the spikes on seed ${seed}`,
      ).toBeGreaterThanOrEqual(0.03);

      // 7. The clock can't take the round, or the second star, from a strong
      //    pipeline — however long the player spends reading.
      const slow = evaluate(idealPipeline, idealAccuracy, Number.MAX_SAFE_INTEGER);
      expect(slow.timePenalty).toBe(MAX_TIME_PENALTY);
      expect(slow.outcome, `ideal with max penalty on seed ${seed}`).toBe("win");
      expect(slow.score!).toBeGreaterThanOrEqual(HIGH_SCORE_THRESHOLD);
    },
    240000,
  );

  it("keeps the numbers the Concept Library quotes about round 1 true", () => {
    // src/lib/concepts.ts: "drop the rows with blanks and the 39 remaining rows
    // are 33% healthy, where the data you started from was 53% healthy".
    const dataset = generateDataset(seedForRound(1));
    const pipeline = applyPipeline(
      withAction(dataset.train, (row) =>
        row.isNull ? "drop" : row.isOutlier ? "cap" : "keep",
      ),
      columnStats(dataset.train),
    );
    expect(pipeline.kept).toBe(39);
    expect(Math.round(pipeline.keptPositiveShare * 100)).toBe(33);
    expect(Math.round(pipeline.sourcePositiveShare * 100)).toBe(53);
  });

  it.each([...ROUND_SEEDS])(
    "seed %i: Cap clamps every spike just past the real readings, and never touches a clean one",
    (seed) => {
      const dataset = generateDataset(seed);
      const stats = columnStats(dataset.train);

      for (const feature of FEATURES) {
        // The fences contain the whole sensor range, so capping a clean
        // reading is a genuine no-op (the "it did nothing" card is true)...
        expect(stats.low[feature]).toBeLessThanOrEqual(SENSOR_MIN);
        expect(stats.high[feature]).toBeGreaterThanOrEqual(SENSOR_MAX);
        // ...and a capped spike is "still high, no longer absurd".
        expect(stats.high[feature]).toBeLessThan(2);
      }

      const capped = applyPipeline(withAction(dataset.train, () => "cap"), stats);
      for (const vector of capped.xs) {
        for (const value of vector) {
          expect(value).toBeLessThan(2);
        }
      }
      // Every spike on a row without a blank is counted as fixed by Cap, and
      // that is now true of the values, not just the flag.
      expect(capped.keptWithOutlier).toBe(0);
    },
  );

  it.each([...ROUND_SEEDS])(
    "seed %i: rows whose only problem is a spike lean blighted, so dropping them tips the balance toward healthy",
    (seed) => {
      // Spikes land on both classes alike, but a spike-ONLY row has no blank,
      // and blanks are MNAR on healthy rows. The drop card and idealAction's
      // docblock both rest on this; they used to say the opposite.
      const dataset = generateDataset(seed);
      const healthy = dataset.train.filter((row) => row.label === 1).length;
      const blighted = TRAIN_ROWS - healthy;
      const spikeOnly = dataset.train.filter((row) => row.isOutlier && !row.isNull);
      const spikeHealthy = spikeOnly.filter((row) => row.label === 1).length;
      const spikeBlighted = spikeOnly.length - spikeHealthy;

      expect(spikeBlighted).toBeGreaterThan(spikeHealthy);
      expect(spikeBlighted / blighted).toBeGreaterThan(spikeHealthy / healthy);

      const dropSpikeOnly = applyPipeline(
        withAction(dataset.train, (row) =>
          row.isOutlier && !row.isNull ? "drop" : idealAction(row),
        ),
        columnStats(dataset.train),
      );
      // "2–8 points toward healthy, under MAX_BALANCE_DRIFT" (ml.ts).
      expect(dropSpikeOnly.balanceDrift).toBeGreaterThan(0.02);
      expect(dropSpikeOnly.balanceDrift).toBeLessThan(0.08);
      expect(dropSpikeOnly.balanceDrift).toBeLessThanOrEqual(MAX_BALANCE_DRIFT);
    },
  );

  it("keeps the pooled spike-only split idealAction's docblock quotes true", () => {
    let spikeHealthy = 0;
    let spikeBlighted = 0;
    for (const seed of ROUND_SEEDS) {
      for (const row of generateDataset(seed).train) {
        if (!row.isOutlier || row.isNull) continue;
        if (row.label === 1) spikeHealthy += 1;
        else spikeBlighted += 1;
      }
    }
    expect([spikeHealthy, spikeBlighted]).toEqual([21, 45]);
  });

  it("is deterministic, so the meter only moves when the cleaning changes", async () => {
    // Without seeded initializers every retrain would jiggle the accuracy for
    // reasons unrelated to the player's decisions, and nothing here could be
    // asserted.
    const dataset = generateDataset(ROUND_SEEDS[0]!);
    const rows = withAction(dataset.train, idealAction);
    const first = await trainAndScore(rows, dataset);
    const second = await trainAndScore(rows, dataset);
    expect(first).toBe(second);
  }, 120000);

  it("leaks no tensors across repeated retrains", async () => {
    // Data Detox retrains on every batch, so a per-retrain leak would compound
    // faster here than anywhere else in the product.
    const dataset = generateDataset(ROUND_SEEDS[0]!);
    const before = tf.memory().numTensors;

    for (const choose of [
      idealAction,
      () => "keep" as CleaningAction,
      () => "impute" as CleaningAction,
    ]) {
      await trainAndScore(withAction(dataset.train, choose), dataset);
    }

    expect(tf.memory().numTensors).toBe(before);
  }, 180000);
});

// ═══════════════════════════════════════════════════════════════════════════
// The generative process
// ═══════════════════════════════════════════════════════════════════════════

describe("generateDataset", () => {
  it("is deterministic for a seed", () => {
    expect(generateDataset(11).train).toEqual(generateDataset(11).train);
    expect(generateDataset(11).train).not.toEqual(generateDataset(12).train);
  });

  it("produces the configured sizes", () => {
    const dataset = generateDataset(ROUND_SEEDS[0]!);
    expect(dataset.train).toHaveLength(TRAIN_ROWS);
    expect(dataset.test).toHaveLength(TEST_ROWS);
  });

  it("keeps the held-out set completely clean", () => {
    // It stands in for the world you deploy to, so it must not be corrupted.
    for (const row of generateDataset(ROUND_SEEDS[0]!).test) {
      for (const feature of FEATURES) {
        const value = row.features[feature];
        expect(value).not.toBeNull();
        expect(Number.isFinite(value)).toBe(true);
        expect(value).toBeGreaterThanOrEqual(0);
        expect(value).toBeLessThanOrEqual(1);
      }
    }
  });

  it("puts blanks far more often on healthy rows than blighted ones", () => {
    // The entire mechanism rests on this. Pooled over every shipped round.
    let healthyBlank = 0;
    let healthyTotal = 0;
    let blightedBlank = 0;
    let blightedTotal = 0;

    for (const seed of ROUND_SEEDS) {
      for (const row of generateDataset(seed).train) {
        if (row.label === 1) {
          healthyTotal += 1;
          if (row.isNull) healthyBlank += 1;
        } else {
          blightedTotal += 1;
          if (row.isNull) blightedBlank += 1;
        }
      }
    }

    const healthyRate = healthyBlank / healthyTotal;
    const blightedRate = blightedBlank / blightedTotal;

    expect(healthyRate).toBeGreaterThan(blightedRate * 2);
    expect(healthyRate).toBeCloseTo(NULL_RATE_HEALTHY, 1);
    expect(blightedRate).toBeCloseTo(NULL_RATE_BLIGHTED, 1);
  });

  it("marks exactly the rows it corrupted", () => {
    for (const row of generateDataset(ROUND_SEEDS[1]!).train) {
      const hasBlank = FEATURES.some((f) => row.features[f] === null);
      expect(row.isNull).toBe(hasBlank);
    }
  });

  it("throws outliers well out of range", () => {
    const outliers = generateDataset(ROUND_SEEDS[0]!).train.filter(
      (row) => row.isOutlier,
    );
    expect(outliers.length).toBeGreaterThan(0);
    for (const row of outliers) {
      const values = FEATURES.map((f) => row.features[f]).filter(
        (v): v is number => v !== null,
      );
      expect(Math.max(...values)).toBeGreaterThan(1);
    }
  });

  it("spikes exactly one reading per outlier row, inside the spike range", () => {
    // isOutOfRange is what the belt highlights, so it must flag the spikes and
    // nothing else: clean readings live in 0–1, spikes in OUTLIER_MIN–MAX.
    for (const seed of ROUND_SEEDS) {
      for (const row of generateDataset(seed).train) {
        const spiked = FEATURES.filter((f) => isOutOfRange(row.features[f]));
        expect(spiked.length, `${seed} ${row.id}`).toBe(row.isOutlier ? 1 : 0);
        for (const feature of spiked) {
          const value = row.features[feature]!;
          expect(value).toBeGreaterThanOrEqual(OUTLIER_MIN);
          expect(value).toBeLessThanOrEqual(OUTLIER_MAX);
        }
        for (const feature of FEATURES) {
          const value = row.features[feature];
          if (value === null || spiked.includes(feature)) continue;
          expect(value).toBeGreaterThanOrEqual(SENSOR_MIN);
          expect(value).toBeLessThanOrEqual(SENSOR_MAX);
        }
      }
    }
  });

  it("starts every row undecided", () => {
    for (const row of generateDataset(ROUND_SEEDS[0]!).train) {
      expect(row.playerAction).toBeNull();
    }
  });

  it("labels from clean values, with about LABEL_NOISE flipped", () => {
    // Labels must come from the clean measurements, or cleaning would destroy
    // signal instead of recovering it.
    const clean = { moisture: 0.9, temperature: 0.1, nitrogen: 0.9 };
    expect(labelFor(clean, 1)).toBe(1);
    expect(labelFor(clean, 0)).toBe(0); // forced flip
    expect(LABEL_NOISE).toBeGreaterThan(0);
    expect(LABEL_NOISE).toBeLessThan(0.2);
  });

  it("cycles rounds through the vetted seeds", () => {
    expect(seedForRound(1)).toBe(ROUND_SEEDS[0]);
    expect(seedForRound(ROUND_SEEDS.length + 1)).toBe(ROUND_SEEDS[0]);
    expect(seedForRound(0)).toBe(ROUND_SEEDS[0]);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// The pipeline
// ═══════════════════════════════════════════════════════════════════════════

describe("applyPipeline", () => {
  const stats = {
    median: { moisture: 0.5, temperature: 0.5, nitrogen: 0.5 },
    low: { moisture: 0.1, temperature: 0.1, nitrogen: 0.1 },
    high: { moisture: 0.9, temperature: 0.9, nitrogen: 0.9 },
  };

  const row = (
    features: Partial<Record<(typeof FEATURES)[number], number | null>>,
    action: CleaningAction,
    label: 0 | 1 = 1,
  ): Row => ({
    id: "r",
    features: { moisture: 0.5, temperature: 0.5, nitrogen: 0.5, ...features },
    isNull: Object.values(features).some((v) => v === null),
    isOutlier: Object.values(features).some((v) => v !== null && v! > 1),
    playerAction: action,
    label,
  });

  it("Keep turns a blank into a literal 0", () => {
    const result = applyPipeline([row({ moisture: null }, "keep")], stats);
    expect(result.xs[0]![0]).toBe(0);
    expect(result.keptWithNaiveFill).toBe(1);
  });

  it("Impute fills a blank with the column median", () => {
    const result = applyPipeline([row({ moisture: null }, "impute")], stats);
    expect(result.xs[0]![0]).toBe(0.5);
    expect(result.keptWithNaiveFill).toBe(0);
  });

  it("Cap cannot fill a blank — it still becomes 0", () => {
    // Capping addresses range, not missingness. Conflating the two would hide a
    // real tradeoff.
    const result = applyPipeline([row({ moisture: null }, "cap")], stats);
    expect(result.xs[0]![0]).toBe(0);
    expect(result.keptWithNaiveFill).toBe(1);
  });

  it("Cap clamps an extreme value to the column's fence", () => {
    const result = applyPipeline([row({ moisture: 6 }, "cap")], stats);
    expect(result.xs[0]![0]).toBe(0.9);
    expect(result.keptWithOutlier).toBe(0);
  });

  it("Impute leaves an extreme value untouched", () => {
    const result = applyPipeline([row({ moisture: 6 }, "impute")], stats);
    expect(result.xs[0]![0]).toBe(6);
    expect(result.keptWithOutlier).toBe(1);
  });

  it("Drop removes the row from the matrix entirely", () => {
    const result = applyPipeline([row({}, "drop")], stats);
    expect(result.xs).toHaveLength(0);
    expect(result.kept).toBe(0);
    expect(result.dropped).toBe(1);
  });

  it("counts undecided rows without training on them", () => {
    const undecided: Row = { ...row({}, "keep"), playerAction: null };
    const result = applyPipeline([undecided], stats);
    expect(result.undecided).toBe(1);
    expect(result.kept).toBe(0);
  });

  it("measures class balance drift against the decided rows", () => {
    const rows: Row[] = [
      { ...row({}, "keep", 1), id: "a" },
      { ...row({}, "keep", 1), id: "b" },
      { ...row({}, "drop", 1), id: "c" },
      { ...row({}, "keep", 0), id: "d" },
    ];
    const result = applyPipeline(rows, stats);
    // Source: 3 of 4 healthy = 0.75. Kept: 2 of 3 healthy = 0.667.
    expect(result.sourcePositiveShare).toBeCloseTo(0.75, 6);
    expect(result.keptPositiveShare).toBeCloseTo(2 / 3, 6);
    expect(result.balanceDrift).toBeCloseTo(2 / 3 - 0.75, 6);
  });

  it("keeps labels aligned with their feature rows", () => {
    const rows: Row[] = [
      { ...row({}, "drop", 1), id: "a" },
      { ...row({ moisture: 0.2 }, "keep", 0), id: "b" },
      { ...row({ moisture: 0.8 }, "keep", 1), id: "c" },
    ];
    const result = applyPipeline(rows, stats);
    expect(result.xs).toHaveLength(2);
    expect(result.ys).toEqual([0, 1]);
    expect(result.xs[0]![0]).toBe(0.2);
    expect(result.xs[1]![0]).toBe(0.8);
  });
});

describe("columnStats", () => {
  it("ignores blanks when computing medians", () => {
    const rows: Row[] = [1, 2, 3, null].map((value, index) => ({
      id: `r${index}`,
      features: {
        moisture: value as number | null,
        temperature: 0.5,
        nitrogen: 0.5,
      },
      isNull: value === null,
      isOutlier: false,
      playerAction: null,
      label: 1,
    }));
    // Median of [1, 2, 3], not of [0, 1, 2, 3].
    expect(columnStats(rows).median.moisture).toBe(2);
  });

  it("puts the cap bounds at Tukey's fences, which a spike can't drag", () => {
    const column = (values: number[]): Row[] =>
      values.map((value, index) => ({
        id: `r${index}`,
        features: { moisture: value, temperature: 0.5, nitrogen: 0.5 },
        isNull: false,
        isOutlier: value > 1,
        playerAction: null,
        label: 1,
      }));

    // 0, 0.1, …, 1.0: Q1 = 0.25, Q3 = 0.75, IQR = 0.5.
    const even = Array.from({ length: 11 }, (_, i) => i / 10);
    const plain = columnStats(column(even));
    expect(plain.low.moisture).toBeCloseTo(0.25 - FENCE_K * 0.5, 10);
    expect(plain.high.moisture).toBeCloseTo(0.75 + FENCE_K * 0.5, 10);

    // Replace the top reading with a spike of 90. A 95th percentile would
    // move with it; the fences barely notice, because the quartiles don't.
    const spiked = columnStats(column([...even.slice(0, -1), 90]));
    expect(spiked.high.moisture).toBeCloseTo(plain.high.moisture, 10);
    expect(spiked.high.moisture).toBeLessThan(2);
  });
});

describe("idealAction", () => {
  it("imputes blanks, caps extremes, keeps clean rows, and never drops", () => {
    const base = { moisture: 0.5, temperature: 0.5, nitrogen: 0.5 };
    const make = (isNull: boolean, isOutlier: boolean): Row => ({
      id: "r",
      features: base,
      isNull,
      isOutlier,
      playerAction: null,
      label: 1,
    });

    expect(idealAction(make(true, false))).toBe("impute");
    expect(idealAction(make(false, true))).toBe("cap");
    expect(idealAction(make(false, false))).toBe("keep");
    // Both dirty: blanks are the biased ones, so repairing them wins. And
    // dropping is never right here — measured, not assumed.
    expect(idealAction(make(true, true))).toBe("impute");
  });
});

describe("timePenaltyFor", () => {
  it("is small, capped, and never negative", () => {
    expect(timePenaltyFor(0)).toBe(0);
    expect(timePenaltyFor(-50)).toBe(0);
    expect(timePenaltyFor(10)).toBeLessThan(0.01);
    expect(timePenaltyFor(100000)).toBe(MAX_TIME_PENALTY);
  });

  it("cannot make a strong pipeline unwinnable", () => {
    // The cap has to leave room under the win bar, or reading the explanations
    // would cost you the round.
    expect(MAX_TIME_PENALTY).toBeLessThan(0.8 - WIN_ACCURACY + 0.01);
  });
});

describe("accuracyFromPredictions", () => {
  it("thresholds at 0.5", () => {
    expect(accuracyFromPredictions([0.9, 0.1, 0.6], [1, 0, 1])).toBe(1);
    expect(accuracyFromPredictions([0.4, 0.4], [1, 1])).toBe(0);
    expect(accuracyFromPredictions([0.5], [1])).toBe(1);
  });

  it("returns 0 for an empty set rather than dividing by zero", () => {
    expect(accuracyFromPredictions([], [])).toBe(0);
  });
});

describe("evaluate ordering", () => {
  const stats = columnStats(generateDataset(ROUND_SEEDS[0]!).train);
  const dataset = generateDataset(ROUND_SEEDS[0]!);

  it("reports incomplete before anything else", () => {
    const rows = dataset.train.map((row, index) => ({
      ...row,
      playerAction: index === 0 ? null : ("drop" as CleaningAction),
    }));
    const result = evaluate(applyPipeline(rows, stats), 0, 0);
    expect(result.outcome).toBe("incomplete");
    expect(result.failure).toBeNull();
  });

  it("reports starvation before bias, because the accuracy is meaningless", () => {
    const rows = withAction(dataset.train, () => "drop");
    const result = evaluate(applyPipeline(rows, stats), 0.4, 0);
    expect(result.outcome).toBe("starved");
  });

  it("subtracts the time penalty from the score, not from the accuracy", () => {
    const rows = withAction(dataset.train, idealAction);
    const result = evaluate(applyPipeline(rows, stats), 0.9, 100000);
    expect(result.accuracy).toBe(0.9);
    expect(result.timePenalty).toBe(MAX_TIME_PENALTY);
    expect(result.score!).toBeCloseTo(0.9 - MAX_TIME_PENALTY, 10);
    expect(result.secondsElapsed).toBe(100000);
  });

  it("won't call a pipeline with no trained model a win or a near-miss", () => {
    // Previously `accuracy ?? 0` turned "never trained" into a near-miss whose
    // copy blamed "the guesswork itself".
    const rows = withAction(dataset.train, idealAction);
    const result = evaluate(applyPipeline(rows, stats), null, 0);
    expect(result.outcome).toBe("untrained");
    expect(result.failure).toBeNull();
    expect(result.accuracy).toBeNull();
    expect(result.score).toBeNull();
  });

  it("still names structural failures without a model, and quotes no accuracy", () => {
    // Balance and leftover dirt are measured from the pipeline, not the model.
    const biased = evaluate(
      applyPipeline(
        withAction(dataset.train, (row) =>
          row.isNull ? "drop" : row.isOutlier ? "cap" : "keep",
        ),
        stats,
      ),
      null,
      0,
    );
    expect(biased.outcome).toBe("selection-bias");
    expect(biased.failure!.detail).not.toMatch(/accuracy/i);

    const withModel = evaluate(
      applyPipeline(withAction(dataset.train, () => "keep"), stats),
      0.62,
      0,
    );
    expect(withModel.outcome).toBe("unhandled-nulls");
    expect(withModel.failure!.detail).toMatch(/Held-out accuracy 62%/);
  });

  it("blames spikes on the inputs, not on a squared error the model doesn't use", () => {
    const result = evaluate(
      applyPipeline(
        withAction(dataset.train, (row) => (row.isNull ? "impute" : "keep")),
        stats,
      ),
      0.79,
      0,
    );
    expect(result.outcome).toBe("outlier-contamination");
    expect(result.failure!.detail).not.toMatch(/squared/i);
    expect(result.failure!.detail).toMatch(
      new RegExp(`${OUTLIER_MIN} and ${OUTLIER_MAX}`),
    );
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Store
// ═══════════════════════════════════════════════════════════════════════════

describe("store", () => {
  const store = () => useDataDetoxStore.getState();

  /** Stand in for the component's retrain: claim a ticket, report on it. */
  const train = (accuracy: number) => {
    const ticket = store().beginTraining();
    expect(store().reportAccuracy(accuracy, ticket.id)).toBe(true);
    return ticket;
  };

  const decideAll = (choose: (row: Row) => CleaningAction) => {
    for (const row of store().rows) store().setRowAction(row.id, choose(row));
  };

  beforeEach(() => {
    useProgression.setState({
      ...EMPTY_PROGRESSION,
      hydrated: true,
      syncError: null,
      lastGain: null,
    });
    useProgression.getState().setAdapter(createMemoryAdapter());
    useDataDetoxStore.getState().startRound(1);
    useDataDetoxStore.getState().setLane("visual");
  });

  it("starts on round 1 with a full belt and no model yet", () => {
    const state = useDataDetoxStore.getState();
    expect(state.round).toBe(1);
    expect(state.seed).toBe(ROUND_SEEDS[0]);
    expect(state.rows).toHaveLength(TRAIN_ROWS);
    expect(state.pipeline.undecided).toBe(TRAIN_ROWS);
    expect(state.modelAccuracy).toBeNull();
    expect(state.cursor).toBe(0);
  });

  it("sorting a row advances the belt and updates the pipeline", () => {
    useDataDetoxStore.getState().sortRow("keep");
    const state = useDataDetoxStore.getState();
    expect(state.rows[0]!.playerAction).toBe("keep");
    expect(state.pipeline.kept).toBe(1);
    expect(state.pipeline.undecided).toBe(TRAIN_ROWS - 1);
    expect(state.cursor).toBe(1);
  });

  it("derives the pipeline from the real transformations", () => {
    useDataDetoxStore.getState().sortRow("drop");
    const state = useDataDetoxStore.getState();
    const expected = applyPipeline(state.rows, state.stats);
    expect(state.pipeline.kept).toBe(expected.kept);
    expect(state.pipeline.balanceDrift).toBeCloseTo(expected.balanceDrift, 12);
  });

  it("asks for a retrain every BATCH_SIZE decisions", () => {
    expect(useDataDetoxStore.getState().needsRetrain()).toBe(false);
    for (let i = 0; i < BATCH_SIZE - 1; i += 1) {
      useDataDetoxStore.getState().sortRow("keep");
    }
    expect(useDataDetoxStore.getState().needsRetrain()).toBe(false);
    useDataDetoxStore.getState().sortRow("keep");
    expect(useDataDetoxStore.getState().needsRetrain()).toBe(true);
  });

  it("does not ask for a retrain while one is in flight", () => {
    for (let i = 0; i < BATCH_SIZE; i += 1) {
      useDataDetoxStore.getState().sortRow("keep");
    }
    useDataDetoxStore.getState().beginTraining();
    expect(useDataDetoxStore.getState().needsRetrain()).toBe(false);
  });

  it("records a reported accuracy and resets the batch counter", () => {
    for (let i = 0; i < BATCH_SIZE; i += 1) {
      useDataDetoxStore.getState().sortRow("keep");
    }
    train(0.77);

    const state = useDataDetoxStore.getState();
    expect(state.modelAccuracy).toBe(0.77);
    expect(state.training).toBe(false);
    expect(state.sinceRetrain).toBe(0);
    expect(state.retrains).toBe(1);
  });

  it("still retrains on rows sorted while a fit was running", () => {
    // The old reportAccuracy zeroed the counter, so rows 71 and 72 sorted
    // during the retrain started at row 70 were never trained on — and Score
    // judged a 70-row model as if it were the final pipeline.
    const sortIdeal = () => store().sortRow(idealAction(store().rows[store().cursor]!));
    for (let i = 0; i < TRAIN_ROWS - 2; i += 1) sortIdeal();
    const ticket = store().beginTraining();
    const trainedOn = store().pipelineVersion;
    sortIdeal();
    sortIdeal();

    expect(store().reportAccuracy(0.8, ticket.id)).toBe(true);
    expect(store().trainedVersion).toBe(trainedOn);
    expect(store().sinceRetrain).toBe(2);
    expect(store().isTrained()).toBe(false);
    // The belt is empty and the model is behind it: retrain again.
    expect(store().needsRetrain()).toBe(true);
    // And Score refuses to pass the 70-row model off as this pipeline.
    expect(store().check().outcome).toBe("untrained");
  });

  it("drops a result that lands after Retry, instead of writing it into the fresh run", () => {
    for (let i = 0; i < BATCH_SIZE; i += 1) store().sortRow("keep");
    const ticket = store().beginTraining();
    store().reset();

    expect(store().reportAccuracy(0.84, ticket.id)).toBe(false);
    expect(store().modelAccuracy).toBeNull();
    expect(store().training).toBe(false);
    expect(store().retrains).toBe(0);
  });

  it("drops a result that lands after the round changed", () => {
    for (let i = 0; i < BATCH_SIZE; i += 1) store().sortRow("keep");
    const ticket = store().beginTraining();
    store().newRound();

    expect(store().reportAccuracy(0.84, ticket.id)).toBe(false);
    expect(store().round).toBe(2);
    expect(store().modelAccuracy).toBeNull();
  });

  it("keeps only the newest of two overlapping retrains", () => {
    for (let i = 0; i < BATCH_SIZE; i += 1) store().sortRow("keep");
    const first = store().beginTraining();
    const second = store().beginTraining();

    // The first fit was stopped part-way; its number must not reach the meter
    // even though it finishes first.
    expect(store().reportAccuracy(0.51, first.id)).toBe(false);
    expect(store().modelAccuracy).toBeNull();
    expect(store().training).toBe(true);

    expect(store().reportAccuracy(0.77, second.id)).toBe(true);
    expect(store().modelAccuracy).toBe(0.77);
    expect(store().training).toBe(false);
  });

  it("clears the training flag when the current fit is abandoned, and only then", () => {
    for (let i = 0; i < BATCH_SIZE; i += 1) store().sortRow("keep");
    const first = store().beginTraining();
    const second = store().beginTraining();

    store().abandonTraining(first.id); // stale — must not cancel the second
    expect(store().training).toBe(true);
    store().abandonTraining(second.id);
    expect(store().training).toBe(false);
    expect(store().modelAccuracy).toBeNull();
    // The decisions still count toward a retrain...
    expect(store().sinceRetrain).toBe(BATCH_SIZE);
    // ...but the automatic one won't hammer a pipeline whose fit just failed,
    // which is how a broken backend would otherwise retrain in a loop.
    expect(store().needsRetrain()).toBe(false);
    // The next decision is a new pipeline, so it tries again.
    store().sortRow("keep");
    expect(store().needsRetrain()).toBe(true);
  });

  it("snapshots the pipeline a ticket trains on", () => {
    for (let i = 0; i < BATCH_SIZE; i += 1) store().sortRow("keep");
    const ticket = store().beginTraining();
    const snapshot = store().pipeline;
    store().sortRow("drop");
    expect(ticket.xs).toBe(snapshot.xs);
    expect(ticket.xs).toHaveLength(BATCH_SIZE);
    expect(ticket.test.xs).toHaveLength(TEST_ROWS);
  });

  it("lets the code lane re-decide a row without double-counting the batch", () => {
    useDataDetoxStore.getState().sortRow("keep");
    const after = useDataDetoxStore.getState().sinceRetrain;
    const id = useDataDetoxStore.getState().rows[0]!.id;

    useDataDetoxStore.getState().setRowAction(id, "impute");
    expect(useDataDetoxStore.getState().sinceRetrain).toBe(after);
    expect(useDataDetoxStore.getState().rows[0]!.playerAction).toBe("impute");
  });

  it("treats re-deciding a row the same way as no change at all", () => {
    // So running the same snippet twice doesn't make the model it trained stale.
    decideAll(idealAction);
    train(0.86);
    const version = store().pipelineVersion;
    decideAll(idealAction);
    expect(store().pipelineVersion).toBe(version);
    expect(store().isTrained()).toBe(true);
  });

  it("marks the model stale when a decision changes after it trained", () => {
    decideAll(idealAction);
    train(0.86);
    store().setRowAction(store().rows[0]!.id, "drop");
    expect(store().isTrained()).toBe(false);
    expect(store().check().outcome).not.toBe("win");
  });

  it("skips without deciding, so a row can be revisited", () => {
    useDataDetoxStore.getState().skip();
    expect(useDataDetoxStore.getState().cursor).toBe(1);
    expect(useDataDetoxStore.getState().rows[0]!.playerAction).toBeNull();
    expect(useDataDetoxStore.getState().pipeline.undecided).toBe(TRAIN_ROWS);
  });

  it("returns to skipped rows once the belt runs out", () => {
    const state = useDataDetoxStore.getState();
    state.skip(); // leave row 0 undecided
    for (let i = 0; i < TRAIN_ROWS - 1; i += 1) {
      useDataDetoxStore.getState().sortRow("keep");
    }
    // Only row 0 is left, and the cursor should have wrapped back to it.
    expect(useDataDetoxStore.getState().pipeline.undecided).toBe(1);
    expect(useDataDetoxStore.getState().cursor).toBe(0);
  });

  it("refuses to declare a win while rows remain on the belt", () => {
    useDataDetoxStore.getState().sortRow("keep");
    train(0.95);
    const evaluation = useDataDetoxStore.getState().check();
    expect(evaluation.outcome).toBe("incomplete");
    expect(useDataDetoxStore.getState().won).toBe(false);
    expect(useProgression.getState().xp).toBe(0);
  });

  it("won't score a finished pipeline before any model has trained on it", () => {
    decideAll(idealAction);
    const evaluation = store().check();
    expect(evaluation.outcome).toBe("untrained");
    expect(evaluation.failure).toBeNull();
    expect(store().won).toBe(false);
    expect(useProgression.getState().xp).toBe(0);
  });

  it("wins on a clean pipeline and awards XP once", () => {
    decideAll(idealAction);
    // Stand in for the component's retrain with a plausible held-out accuracy.
    train(0.86);

    const evaluation = useDataDetoxStore.getState().check();
    expect(evaluation.outcome).toBe("win");
    expect(useDataDetoxStore.getState().won).toBe(true);

    const xp = useProgression.getState().xp;
    expect(xp).toBeGreaterThan(0);
    useDataDetoxStore.getState().check();
    expect(useProgression.getState().xp).toBe(xp);
    // Checking again can't un-clear it either.
    expect(useDataDetoxStore.getState().won).toBe(true);
  });

  it("names Selection bias when the player drops the rows with blanks", () => {
    decideAll((row) => (row.isNull ? "drop" : row.isOutlier ? "cap" : "keep"));
    train(0.62);

    const evaluation = useDataDetoxStore.getState().check();
    expect(evaluation.outcome).toBe("selection-bias");
    expect(useDataDetoxStore.getState().failure?.name).toBe("Selection bias");
    expect(useProgression.getState().xp).toBe(0);
  });

  it("names Data starvation when almost everything is dropped, with no model needed", () => {
    decideAll(() => "drop");

    const evaluation = useDataDetoxStore.getState().check();
    expect(evaluation.outcome).toBe("starved");
    expect(useDataDetoxStore.getState().failure?.name).toBe("Data starvation");
  });

  it("credits the code lane for the third star when the check comes from api.check()", () => {
    decideAll(idealAction);
    train(0.86);
    useDataDetoxStore.getState().check("code");

    expect(useProgression.getState().games["data-detox"]?.codeLaneCleared).toBe(
      true,
    );
    expect(useProgression.getState().games["data-detox"]?.stars).toBe(3);
  });

  it("credits the lane the check came from, not the tab that happens to be open", () => {
    // The rail's Score button is visible from the code tab too; pressing it
    // there is not a code-lane clear.
    store().setLane("code");
    decideAll(idealAction);
    train(0.86);
    store().check("visual");

    const record = useProgression.getState().games["data-detox"];
    expect(record?.completed).toBe(true);
    expect(record?.codeLaneCleared).toBe(false);
    expect(record?.stars).toBe(2);
  });

  it("records one clear per cleaning per lane, however often Score is pressed", () => {
    const record = () => useProgression.getState().games["data-detox"];
    decideAll(idealAction);
    train(0.86);

    store().check();
    store().check();
    expect(record()?.playCount).toBe(1);

    // The code lane is a separate clear (it earns the third star) — once.
    store().check("code");
    store().check("code");
    expect(record()?.playCount).toBe(2);
    expect(record()?.codeLaneCleared).toBe(true);

    // A different cleaning is a different attempt, and counts again.
    const clean = store().rows.find((row) => !row.isNull && !row.isOutlier)!;
    store().setRowAction(clean.id, "cap");
    train(0.86);
    store().check();
    expect(record()?.playCount).toBe(3);

    // So is the same cleaning after Retry.
    store().reset();
    decideAll(idealAction);
    train(0.86);
    store().check();
    expect(record()?.playCount).toBe(4);
  });

  it("calls a retrain 'the same pipeline' only when no decision changed", () => {
    decideAll(idealAction);
    train(0.86);

    // The starter comment's suggestion, from the code lane: drop the blanks.
    // Every row was already decided, so no NEW decision was made — the old
    // card read "Same pipeline, trained again" for a pipeline 33 rows shorter.
    const blanks = store().rows.filter((row) => row.isNull).length;
    decideAll((row) => (row.isNull ? "drop" : idealAction(row)));
    expect(store().sinceRetrain).toBe(0);
    expect(store().needsRetrain()).toBe(true);

    train(0.61);
    expect(store().whyCard!.body).not.toMatch(/Same pipeline/);
    expect(store().whyCard!.tone).toBe("warn");

    // Back again, and it counts every re-decision it took in.
    decideAll(idealAction);
    train(0.86);
    expect(store().whyCard!.body).toContain(`Your last ${blanks} decisions`);

    // Nothing changed since that one: now it IS the same pipeline.
    train(0.86);
    expect(store().whyCard!.body).toMatch(/^Same pipeline, trained again/);
  });

  it("resumes the clock from the seconds already charged, not the wall-clock gap", () => {
    store().setElapsed(30);
    // The store outlives the page: the player left for the Concept Library
    // ten minutes ago and has just come back.
    useDataDetoxStore.setState({ startedAt: Date.now() - 10 * 60 * 1000 });
    store().resumeClock();
    expect(Math.round((Date.now() - store().startedAt) / 1000)).toBe(30);
  });

  it("only runs the clock while the game is mounted and its tab is visible", () => {
    vi.useFakeTimers();
    let visibility: DocumentVisibilityState = "visible";
    const spy = vi
      .spyOn(document, "visibilityState", "get")
      .mockImplementation(() => visibility);
    try {
      store().reset();
      const first = renderHook(() => useAttemptClock());
      vi.advanceTimersByTime(20_000);
      expect(store().timeElapsed).toBe(20);

      // Off to a Concept Library page for five minutes.
      first.unmount();
      vi.advanceTimersByTime(5 * 60_000);
      const second = renderHook(() => useAttemptClock());
      vi.advanceTimersByTime(10_000);
      expect(store().timeElapsed).toBe(30);

      // A hidden tab doesn't tick, and coming back doesn't charge the gap.
      visibility = "hidden";
      document.dispatchEvent(new Event("visibilitychange"));
      vi.advanceTimersByTime(60_000);
      expect(store().timeElapsed).toBe(30);
      visibility = "visible";
      document.dispatchEvent(new Event("visibilitychange"));
      vi.advanceTimersByTime(5_000);
      expect(store().timeElapsed).toBe(35);
      second.unmount();
    } finally {
      spy.mockRestore();
      vi.useRealTimers();
    }
  });

  it("attaches a distinct why-card to every action", () => {
    const keys = new Set<string>();
    const record = () => {
      const card = useDataDetoxStore.getState().whyCard;
      expect(card).not.toBeNull();
      keys.add(card!.key);
    };

    record();
    useDataDetoxStore.getState().sortRow("keep");
    record();
    useDataDetoxStore.getState().sortRow("drop");
    record();
    train(0.7);
    record();
    useDataDetoxStore.getState().check();
    record();

    expect(keys.size).toBe(5);
  });

  it("resets decisions but keeps the same rows, and restarts the clock", () => {
    const ids = useDataDetoxStore.getState().rows.map((r) => r.id);
    useDataDetoxStore.getState().sortRow("drop");
    train(0.5);
    useDataDetoxStore.getState().setElapsed(250);
    const before = useDataDetoxStore.getState().startedAt;
    useDataDetoxStore.getState().reset();

    const state = useDataDetoxStore.getState();
    expect(state.rows.map((r) => r.id)).toEqual(ids);
    expect(state.pipeline.undecided).toBe(TRAIN_ROWS);
    expect(state.modelAccuracy).toBeNull();
    expect(state.retrains).toBe(0);
    expect(state.timeElapsed).toBe(0);
    // The component's clock counts from startedAt, so Retry really restarts it.
    expect(state.startedAt).toBeGreaterThanOrEqual(before);
  });

  it("charges the time penalty for the clock that ran, and reports those seconds", () => {
    decideAll(idealAction);
    train(0.86);
    store().setElapsed(60);
    const evaluation = store().check();
    expect(evaluation.secondsElapsed).toBe(60);
    expect(evaluation.timePenalty).toBeCloseTo(timePenaltyFor(60), 12);
  });

  it("advances to the next vetted seed", () => {
    useDataDetoxStore.getState().newRound();
    expect(useDataDetoxStore.getState().round).toBe(2);
    expect(useDataDetoxStore.getState().seed).toBe(ROUND_SEEDS[1]);
  });

  it("exposes a clean held-out matrix shaped for predict", () => {
    const test = useDataDetoxStore.getState().testSet();
    expect(test.xs).toHaveLength(TEST_ROWS);
    expect(test.xs[0]).toHaveLength(FEATURES.length);
    expect(test.ys).toHaveLength(TEST_ROWS);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// The code lane's api
// ═══════════════════════════════════════════════════════════════════════════

describe("code lane api", () => {
  const store = () => useDataDetoxStore.getState();
  /** idealAction, written the way a snippet would, against the rows it sees. */
  const rule = (row: { isNull: boolean; isOutlier: boolean }): CleaningAction =>
    row.isNull ? "impute" : row.isOutlier ? "cap" : "keep";

  beforeEach(() => {
    useProgression.setState({
      ...EMPTY_PROGRESSION,
      hydrated: true,
      syncError: null,
      lastGain: null,
    });
    useProgression.getState().setAdapter(createMemoryAdapter());
    store().startRound(1);
  });

  it("applies a rule to every row through the same store action as the bins", () => {
    const api = createCodeApi(async () => null);
    api.forEachRow(rule);
    const expected = applyPipeline(
      withAction(store().rows, idealAction),
      store().stats,
    );
    expect(store().pipeline.xs).toEqual(expected.xs);
    expect(api.blanksLeft()).toBe(0);
    expect(api.extremesLeft()).toBe(expected.keptWithOutlier);
  });

  it("rejects a bad decision without applying any of the rule", () => {
    const api = createCodeApi(async () => null);
    let calls = 0;
    expect(() =>
      api.forEachRow(() => {
        calls += 1;
        return (calls > 5 ? "delete" : "keep") as CleaningAction;
      }),
    ).toThrow(/expected one of 'keep', 'impute', 'cap', 'drop'/);
    // All-or-nothing: the five valid decisions before it were not applied.
    expect(store().pipeline.undecided).toBe(TRAIN_ROWS);
  });

  it("rejects a forEachRow argument that isn't a function", () => {
    const api = createCodeApi(async () => null);
    expect(() =>
      api.forEachRow("impute" as unknown as () => CleaningAction),
    ).toThrow(TypeError);
  });

  it("names an unknown row id or action in setAction", () => {
    const api = createCodeApi(async () => null);
    expect(() => api.setAction("row-999", "keep")).toThrow(/no row with id/);
    expect(() =>
      api.setAction("row-0", "fix" as CleaningAction),
    ).toThrow(/expected one of/);
    expect(store().pipeline.undecided).toBe(TRAIN_ROWS);
    api.setAction("row-0", "drop");
    expect(store().rows[0]!.playerAction).toBe("drop");
  });

  it("says why when retrain has nothing to train on or was cancelled", async () => {
    const api = createCodeApi(async () => null);
    api.forEachRow(() => "drop");
    await expect(api.retrain()).rejects.toThrow(/nothing to train on/);

    store().reset();
    api.forEachRow(() => "keep");
    await expect(api.retrain()).rejects.toThrow(/cancelled/);
  });

  it("returns the accuracy the component's retrain produced", async () => {
    const api = createCodeApi(async () => 0.81);
    api.forEachRow(rule);
    await expect(api.retrain()).resolves.toBe(0.81);
  });

  it("scores from the code lane, and reports no accuracy for a stale model", () => {
    const api = createCodeApi(async () => null);
    api.forEachRow(rule);
    const ticket = store().beginTraining();
    store().reportAccuracy(0.86, ticket.id);
    expect(api.accuracy()).toBe(0.86);

    expect(api.check().outcome).toBe("win");
    expect(useProgression.getState().games["data-detox"]?.codeLaneCleared).toBe(
      true,
    );

    api.setAction("row-0", "drop");
    expect(api.accuracy()).toBeNull();
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Why-cards
// ═══════════════════════════════════════════════════════════════════════════

describe("why-cards", () => {
  const stats = columnStats(generateDataset(ROUND_SEEDS[0]!).train);
  const dataset = generateDataset(ROUND_SEEDS[0]!);

  const evaluationFor = (choose: (row: Row) => CleaningAction, accuracy = 0.6) =>
    evaluate(applyPipeline(withAction(dataset.train, choose), stats), accuracy, 0);

  /** The pipeline after sorting the first `count` rows with `choose`. */
  const partial = (count: number, choose: (row: Row) => CleaningAction) =>
    applyPipeline(
      dataset.train.map((row, index) => ({
        ...row,
        playerAction: index < count ? choose(row) : null,
      })),
      stats,
    );

  it("explains that Keep turns a blank into a zero", () => {
    const blankRow = dataset.train.find((row) => row.isNull)!;
    const card = whyCardFor({
      kind: "sorted",
      action: "keep",
      row: blankRow,
      pipeline: applyPipeline([{ ...blankRow, playerAction: "keep" }], stats),
      stats,
    });
    expect(card.body).toMatch(/reads 0|becomes 0/i);
    expect(card.body).toMatch(/not "unknown"|isn't "unknown"/i);
    expect(card.tone).toBe("warn");
  });

  it("warns that dropping is not neutral", () => {
    const row = dataset.train[0]!;
    const card = whyCardFor({
      kind: "sorted",
      action: "drop",
      row,
      pipeline: applyPipeline([{ ...row, playerAction: "drop" }], stats),
      stats,
    });
    expect(card.body).toMatch(/costs you data/i);
    expect(card.conceptHref).toBeTruthy();
  });

  it("counts the rows that can still train after a drop, not just those kept so far", () => {
    // Dropping the very first row used to read "0 rows left" in the starvation
    // tone, with 71 rows still on the belt.
    const pipeline = partial(1, () => "drop");
    const early = whyCardFor({
      kind: "sorted",
      action: "drop",
      row: dataset.train[0]!,
      pipeline,
      stats,
    });
    expect(early.title).toContain(`${TRAIN_ROWS - 1} rows can still train`);
    expect(early.tone).toBe("warn");

    // Starvation tone only once it is genuinely unavoidable.
    const starving = partial(TRAIN_ROWS - MIN_ROWS + 1, () => "drop");
    const late = whyCardFor({
      kind: "sorted",
      action: "drop",
      row: dataset.train[TRAIN_ROWS - MIN_ROWS]!,
      pipeline: starving,
      stats,
    });
    expect(starving.kept + starving.undecided).toBeLessThan(MIN_ROWS);
    expect(late.tone).toBe("bad");
  });

  it("names selection bias and cites both class balances", () => {
    const card = whyCardFor({
      kind: "checked",
      evaluation: evaluationFor((row) =>
        row.isNull ? "drop" : row.isOutlier ? "cap" : "keep",
      ),
    });
    expect(card.title).toMatch(/selection bias/i);
    expect(card.tone).toBe("bad");
    expect(card.body).toMatch(/\d+% healthy.*\d+% healthy/);
  });

  it("says what a spike-only drop really does to the class balance, and quotes the drift", () => {
    // Every spike-only row dropped, everything else cleaned ideally — the
    // drift is what the rail shows as that card appears.
    const pipeline = applyPipeline(
      withAction(dataset.train, (row) =>
        row.isOutlier && !row.isNull ? "drop" : idealAction(row),
      ),
      stats,
    );
    const row = dataset.train.find((candidate) => candidate.isOutlier && !candidate.isNull)!;
    const card = whyCardFor({ kind: "sorted", action: "drop", row, pipeline, stats });

    expect(pipeline.balanceDrift).toBeGreaterThan(0);
    expect(card.body).not.toMatch(/without bending/);
    expect(card.body).toMatch(/lean blighted/);
    expect(card.body).toMatch(/tips the balance toward healthy/);
    expect(card.body).toContain(`drift is +${Math.round(pipeline.balanceDrift * 100)}%`);
  });

  it("doesn't blame blanks for a selection bias that leans healthy", () => {
    // A label-blind code-lane rule — drop the low-moisture rows — removes
    // blighted plots out of proportion (moisture pushes toward healthy). The
    // copy used to say blanks are "far more common on blighted plots", which
    // states the generator backwards.
    const round2 = generateDataset(ROUND_SEEDS[1]!);
    const evaluation = evaluate(
      applyPipeline(
        withAction(round2.train, (row) =>
          row.features.moisture !== null && row.features.moisture < 0.45
            ? "drop"
            : idealAction(row),
        ),
        columnStats(round2.train),
      ),
      0.8,
      0,
    );
    expect(evaluation.outcome).toBe("selection-bias");
    expect(evaluation.balanceDrift).toBeGreaterThan(MAX_BALANCE_DRIFT);
    expect(evaluation.kept).toBeGreaterThanOrEqual(MIN_ROWS);
    expect(evaluation.failure!.detail).not.toMatch(/common on blighted plots/);
    expect(evaluation.failure!.detail).toMatch(/blighted out of proportion/);
    const card = whyCardFor({ kind: "checked", evaluation });
    expect(card.body).not.toMatch(/Blanks aren't spread evenly/);
    expect(card.body).toMatch(/blighted rows out of proportion/);

    // The ordinary trap — dropping the blanks — still blames the blanks.
    const blanksDropped = evaluationFor((row) =>
      row.isNull ? "drop" : row.isOutlier ? "cap" : "keep",
    );
    expect(blanksDropped.balanceDrift).toBeLessThan(0);
    expect(blanksDropped.failure!.detail).toMatch(/far more common on healthy plots/);
  });

  it("names data starvation and says why dropping is expensive", () => {
    const card = whyCardFor({
      kind: "checked",
      evaluation: evaluationFor(() => "drop"),
    });
    expect(card.title).toMatch(/starvation/i);
    expect(card.tone).toBe("bad");
    expect(card.body).toMatch(new RegExp(String(MIN_ROWS)));
  });

  it("names unhandled missing values for a keep-everything pipeline", () => {
    const card = whyCardFor({
      kind: "checked",
      evaluation: evaluationFor(() => "keep"),
    });
    expect(card.title).toMatch(/missing values/i);
    expect(card.tone).toBe("bad");
  });

  it("names outlier contamination without blaming squared error", () => {
    const card = whyCardFor({
      kind: "checked",
      evaluation: evaluationFor((row) => (row.isNull ? "impute" : "keep")),
    });
    expect(card.title).toMatch(/outlier contamination/i);
    expect(card.tone).toBe("bad");
    expect(card.body).not.toMatch(/squared/i);
    expect(card.conceptLabel).toBe("Outliers are not automatically errors");
  });

  it("asks for a retrain rather than guessing when nothing has trained", () => {
    const card = whyCardFor({
      kind: "checked",
      evaluation: evaluate(
        applyPipeline(withAction(dataset.train, idealAction), stats),
        null,
        0,
      ),
    });
    expect(card.title).toMatch(/train the model/i);
    expect(card.body).toMatch(/api\.retrain\(\)/);
    expect(card.body).not.toMatch(/guesswork/i);
  });

  it("says when it was the clock, not the cleaning, that missed the bar", () => {
    const evaluation = evaluate(
      applyPipeline(withAction(dataset.train, idealAction), stats),
      WIN_ACCURACY + 0.01,
      Number.MAX_SAFE_INTEGER,
    );
    expect(evaluation.outcome).toBe("near-miss");
    const card = whyCardFor({ kind: "checked", evaluation });
    expect(card.body).toMatch(/time penalty took it back under/);
  });

  it("reports whether a retrain helped or hurt", () => {
    const pipeline = applyPipeline(withAction(dataset.train, idealAction), stats);
    const up = whyCardFor({
      kind: "retrained",
      accuracy: 0.8,
      previous: 0.7,
      pipeline,
      decisions: BATCH_SIZE,
    });
    const down = whyCardFor({
      kind: "retrained",
      accuracy: 0.6,
      previous: 0.7,
      pipeline,
      decisions: BATCH_SIZE,
    });
    expect(up.tone).toBe("good");
    expect(down.tone).toBe("warn");
    expect(up.title).toMatch(/\+10%/);
    expect(down.title).toMatch(/−10%/);
  });

  it("counts the decisions a retrain actually took in", () => {
    // A code-lane retrain after the whole belt is one retrain over 72
    // decisions; the copy used to say "your last 10" regardless.
    const pipeline = applyPipeline(withAction(dataset.train, idealAction), stats);
    const card = whyCardFor({
      kind: "retrained",
      accuracy: 0.8,
      previous: 0.7,
      pipeline,
      decisions: TRAIN_ROWS,
    });
    expect(card.body).toContain(`Your last ${TRAIN_ROWS} decisions`);
  });

  it("quotes the actual impossible reading when a spiked row is kept", () => {
    const outlierRow = dataset.train.find(
      (row) => row.isOutlier && !row.isNull,
    )!;
    const spike = FEATURES.map((f) => outlierRow.features[f]).find((v) =>
      isOutOfRange(v),
    )!;
    const card = whyCardFor({
      kind: "sorted",
      action: "keep",
      row: outlierRow,
      pipeline: applyPipeline([{ ...outlierRow, playerAction: "keep" }], stats),
      stats,
    });
    expect(card.body).toContain(spike.toFixed(2));
    expect(card.body).toContain(`${SENSOR_MIN} to ${SENSOR_MAX}`);
    expect(card.body).not.toMatch(/squared/i);
  });

  it("quotes where Cap actually put a spike", () => {
    const outlierRow = dataset.train.find(
      (row) => row.isOutlier && !row.isNull,
    )!;
    const feature = FEATURES.find((f) => isOutOfRange(outlierRow.features[f]))!;
    const card = whyCardFor({
      kind: "sorted",
      action: "cap",
      row: outlierRow,
      pipeline: applyPipeline([{ ...outlierRow, playerAction: "cap" }], stats),
      stats,
    });
    expect(card.body).toContain(outlierRow.features[feature]!.toFixed(2));
    expect(card.body).toContain(stats.high[feature].toFixed(2));
    expect(card.tone).toBe("good");
  });
});
