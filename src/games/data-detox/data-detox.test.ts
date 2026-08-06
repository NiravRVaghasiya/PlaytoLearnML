import * as tf from "@tensorflow/tfjs";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import {
  EMPTY_PROGRESSION,
  createMemoryAdapter,
  useProgression,
} from "@/engine/progression";
import {
  BATCH_SIZE,
  FEATURES,
  LABEL_NOISE,
  MAX_BALANCE_DRIFT,
  MAX_TIME_PENALTY,
  MIN_ROWS,
  NULL_RATE_BLIGHTED,
  NULL_RATE_HEALTHY,
  OUTLIER_FACTOR,
  ROUND_SEEDS,
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
  labelFor,
  seedForRound,
  testMatrix,
  timePenaltyFor,
  type CleaningAction,
  type Dataset,
  type Row,
} from "./ml";
import { useDataDetoxStore } from "./store";
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
      expect(ideal.score).toBeGreaterThanOrEqual(WIN_ACCURACY);

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

      // 4. Dropping everything starves the model outright.
      const dropAll = evaluate(
        pipelineFor(() => "drop"),
        0,
        0,
      );
      expect(dropAll.outcome, `drop-all on seed ${seed}`).toBe("starved");
      expect(dropAll.kept).toBeLessThan(MIN_ROWS);

      // 5. And the ideal pipeline genuinely beats the naive one on accuracy —
      //    the metric has to agree with the diagnosis.
      expect(idealAccuracy).toBeGreaterThan(keepAll.accuracy);
    },
    180000,
  );

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

  it("Cap clamps an extreme value into the percentile band", () => {
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
    expect(result.score).toBeCloseTo(0.9 - MAX_TIME_PENALTY, 10);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Store
// ═══════════════════════════════════════════════════════════════════════════

describe("store", () => {
  beforeEach(() => {
    useProgression.setState({
      ...EMPTY_PROGRESSION,
      hydrated: true,
      syncError: null,
      lastGain: null,
    });
    useProgression.getState().setAdapter(createMemoryAdapter());
    useDataDetoxStore.getState().startRound(1);
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
    useDataDetoxStore.getState().beginTraining();
    useDataDetoxStore.getState().reportAccuracy(0.77);

    const state = useDataDetoxStore.getState();
    expect(state.modelAccuracy).toBe(0.77);
    expect(state.training).toBe(false);
    expect(state.sinceRetrain).toBe(0);
    expect(state.retrains).toBe(1);
  });

  it("lets the code lane re-decide a row without double-counting the batch", () => {
    useDataDetoxStore.getState().sortRow("keep");
    const after = useDataDetoxStore.getState().sinceRetrain;
    const id = useDataDetoxStore.getState().rows[0]!.id;

    useDataDetoxStore.getState().setRowAction(id, "impute");
    expect(useDataDetoxStore.getState().sinceRetrain).toBe(after);
    expect(useDataDetoxStore.getState().rows[0]!.playerAction).toBe("impute");
  });

  it("skips without deciding, so a row can be revisited", () => {
    useDataDetoxStore.getState().skip();
    expect(useDataDetoxStore.getState().cursor).toBe(1);
    expect(useDataDetoxStore.getState().rows[0]!.playerAction).toBeNull();
    expect(useDataDetoxStore.getState().pipeline.undecided).toBe(TRAIN_ROWS);
  });

  it("returns to skipped rows once the belt runs out", () => {
    const store = useDataDetoxStore.getState();
    store.skip(); // leave row 0 undecided
    for (let i = 0; i < TRAIN_ROWS - 1; i += 1) {
      useDataDetoxStore.getState().sortRow("keep");
    }
    // Only row 0 is left, and the cursor should have wrapped back to it.
    expect(useDataDetoxStore.getState().pipeline.undecided).toBe(1);
    expect(useDataDetoxStore.getState().cursor).toBe(0);
  });

  it("refuses to declare a win while rows remain on the belt", () => {
    useDataDetoxStore.getState().sortRow("keep");
    useDataDetoxStore.getState().reportAccuracy(0.95);
    const evaluation = useDataDetoxStore.getState().check();
    expect(evaluation.outcome).toBe("incomplete");
    expect(useDataDetoxStore.getState().won).toBe(false);
    expect(useProgression.getState().xp).toBe(0);
  });

  it("wins on a clean pipeline and awards XP once", () => {
    const rows = useDataDetoxStore.getState().rows;
    for (const row of rows) {
      useDataDetoxStore.getState().setRowAction(row.id, idealAction(row));
    }
    // Stand in for the component's retrain with a plausible held-out accuracy.
    useDataDetoxStore.getState().reportAccuracy(0.86);

    const evaluation = useDataDetoxStore.getState().check();
    expect(evaluation.outcome).toBe("win");
    expect(useDataDetoxStore.getState().won).toBe(true);

    const xp = useProgression.getState().xp;
    expect(xp).toBeGreaterThan(0);
    useDataDetoxStore.getState().check();
    expect(useProgression.getState().xp).toBe(xp);
  });

  it("names Selection bias when the player drops the rows with blanks", () => {
    const rows = useDataDetoxStore.getState().rows;
    for (const row of rows) {
      useDataDetoxStore
        .getState()
        .setRowAction(
          row.id,
          row.isNull ? "drop" : row.isOutlier ? "cap" : "keep",
        );
    }
    useDataDetoxStore.getState().reportAccuracy(0.62);

    const evaluation = useDataDetoxStore.getState().check();
    expect(evaluation.outcome).toBe("selection-bias");
    expect(useDataDetoxStore.getState().failure?.name).toBe("Selection bias");
    expect(useProgression.getState().xp).toBe(0);
  });

  it("names Data starvation when almost everything is dropped", () => {
    const rows = useDataDetoxStore.getState().rows;
    for (const row of rows) {
      useDataDetoxStore.getState().setRowAction(row.id, "drop");
    }
    useDataDetoxStore.getState().reportAccuracy(0.3);

    const evaluation = useDataDetoxStore.getState().check();
    expect(evaluation.outcome).toBe("starved");
    expect(useDataDetoxStore.getState().failure?.name).toBe("Data starvation");
  });

  it("credits the code lane for the third star", () => {
    useDataDetoxStore.getState().setLane("code");
    const rows = useDataDetoxStore.getState().rows;
    for (const row of rows) {
      useDataDetoxStore.getState().setRowAction(row.id, idealAction(row));
    }
    useDataDetoxStore.getState().reportAccuracy(0.86);
    useDataDetoxStore.getState().check();

    expect(useProgression.getState().games["data-detox"]?.codeLaneCleared).toBe(
      true,
    );
    expect(useProgression.getState().games["data-detox"]?.stars).toBe(3);
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
    useDataDetoxStore.getState().reportAccuracy(0.7);
    record();
    useDataDetoxStore.getState().check();
    record();

    expect(keys.size).toBe(5);
  });

  it("resets decisions but keeps the same rows", () => {
    const ids = useDataDetoxStore.getState().rows.map((r) => r.id);
    useDataDetoxStore.getState().sortRow("drop");
    useDataDetoxStore.getState().reportAccuracy(0.5);
    useDataDetoxStore.getState().reset();

    const state = useDataDetoxStore.getState();
    expect(state.rows.map((r) => r.id)).toEqual(ids);
    expect(state.pipeline.undecided).toBe(TRAIN_ROWS);
    expect(state.modelAccuracy).toBeNull();
    expect(state.retrains).toBe(0);
    expect(state.timeElapsed).toBe(0);
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
// Why-cards
// ═══════════════════════════════════════════════════════════════════════════

describe("why-cards", () => {
  const stats = columnStats(generateDataset(ROUND_SEEDS[0]!).train);
  const dataset = generateDataset(ROUND_SEEDS[0]!);

  const evaluationFor = (choose: (row: Row) => CleaningAction, accuracy = 0.6) =>
    evaluate(applyPipeline(withAction(dataset.train, choose), stats), accuracy, 0);

  it("explains that Keep turns a blank into a zero", () => {
    const blankRow = dataset.train.find((row) => row.isNull)!;
    const card = whyCardFor({
      kind: "sorted",
      action: "keep",
      row: blankRow,
      pipeline: applyPipeline([{ ...blankRow, playerAction: "keep" }], stats),
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
    });
    expect(card.body).toMatch(/costs you data/i);
    expect(card.conceptHref).toBeTruthy();
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

  it("reports whether a retrain helped or hurt", () => {
    const pipeline = applyPipeline(withAction(dataset.train, idealAction), stats);
    const up = whyCardFor({
      kind: "retrained",
      accuracy: 0.8,
      previous: 0.7,
      pipeline,
    });
    const down = whyCardFor({
      kind: "retrained",
      accuracy: 0.6,
      previous: 0.7,
      pipeline,
    });
    expect(up.tone).toBe("good");
    expect(down.tone).toBe("warn");
    expect(up.title).toMatch(/\+10%/);
    expect(down.title).toMatch(/−10%/);
  });

  it("mentions the outlier factor when an extreme value is kept", () => {
    const outlierRow = dataset.train.find(
      (row) => row.isOutlier && !row.isNull,
    )!;
    const card = whyCardFor({
      kind: "sorted",
      action: "keep",
      row: outlierRow,
      pipeline: applyPipeline([{ ...outlierRow, playerAction: "keep" }], stats),
    });
    expect(card.body).toMatch(new RegExp(`${OUTLIER_FACTOR}`));
  });
});
