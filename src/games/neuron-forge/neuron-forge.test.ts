import * as tf from "@tensorflow/tfjs";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import {
  HIGH_SCORE_THRESHOLD,
  createMemoryAdapter,
  useProgression,
} from "@/engine/progression";
import {
  ACTIVATIONS,
  EFFICIENCY_WEIGHT,
  FLAT_OUTPUT_SPREAD,
  containsMinimalSolution,
  cutsBeforeFirstBend,
  diagnoseCollapse,
  isScored,
  isSingleCut,
  outputSpread,
  scoreOf,
  shortOf,
  MAX_NEURONS_PER_LAYER,
  MODEL_SEED,
  PATTERNS,
  TEST_POINTS,
  TRAIN_BATCH,
  TRAIN_EPOCHS,
  TRAIN_POINTS,
  accuracyFromPredictions,
  architectureOf,
  buildModel,
  evaluate,
  generateDataset,
  isEffectivelyLinear,
  nonlinearLayerCount,
  patternById,
  surfaceGrid,
  SURFACE_RESOLUTION,
  toMatrix,
  MAX_LAYERS,
  type Activation,
  type Collapse,
  type Layer,
  type PatternId,
} from "./ml";
import {
  SLUG,
  budgetRemaining,
  canAddLayer,
  canGrowLayer,
  useNeuronForgeStore,
} from "./store";
import { whyCardFor } from "./why-cards";

beforeAll(async () => {
  await tf.ready();
});

const hidden = (spec: Array<[number, Activation]>): Layer[] =>
  spec.map(([neurons, activation]) => ({ neurons, activation }));

interface Diagnosed {
  accuracy: number;
  collapse: Collapse | null;
  spread: number;
}

/**
 * Fits are deterministic (seeded data, seeded weights, no shuffling), so each
 * architecture is trained once per file however many tests ask about it.
 */
const diagnosed = new Map<string, Promise<Diagnosed>>();

/**
 * Train the architecture on the puzzle exactly as the game does, then run the
 * same post-training check the trainer runs.
 */
function trainAndDiagnose(
  pattern: PatternId,
  layers: readonly Layer[],
): Promise<Diagnosed> {
  const key = `${pattern}|${JSON.stringify(layers)}`;
  let pending = diagnosed.get(key);
  if (!pending) {
    pending = fitAndDiagnose(pattern, [...layers]);
    diagnosed.set(key, pending);
  }
  return pending;
}

async function fitAndDiagnose(
  pattern: PatternId,
  layers: Layer[],
): Promise<Diagnosed> {
  const dataset = generateDataset(pattern, 7001);
  const train = toMatrix(dataset.train);
  const test = toMatrix(dataset.test);

  const architecture = architectureOf(layers);
  const model = buildModel(architecture, MODEL_SEED);
  const xs = tf.tensor2d(train.xs);
  const ys = tf.tensor2d(train.ys.map((y) => [y]));
  await model.fit(xs, ys, {
    epochs: TRAIN_EPOCHS,
    batchSize: TRAIN_BATCH,
    shuffle: false,
    verbose: 0,
  });

  const predictions = tf.tidy(() => {
    const input = tf.tensor2d(test.xs);
    return (model.predict(input) as tf.Tensor).dataSync();
  });
  const accuracy = accuracyFromPredictions(predictions, test.ys);
  const collapse = diagnoseCollapse(model, architecture, train.xs, predictions);

  xs.dispose();
  ys.dispose();
  const optimizer = model.optimizer;
  model.dispose();
  optimizer?.dispose();
  return { accuracy, collapse, spread: outputSpread(predictions) };
}

async function trainAndScore(
  pattern: PatternId,
  layers: readonly Layer[],
): Promise<number> {
  return (await trainAndDiagnose(pattern, layers)).accuracy;
}

/**
 * How far each rung of the ladder must sit from its puzzle's target.
 *
 * The same code measured 1–3 points apart across TF.js backends (WebGL vs CPU,
 * Chromium vs Node). A target within a point of a rung decides the lesson by
 * float rounding on the player's GPU, so the targets are placed with room on
 * both sides and this holds them there. 2.5 points is 12–13 of the 500
 * held-out points.
 */
const LADDER_MARGIN = 0.025;

describe("datasets", () => {
  it("is deterministic for a given seed", () => {
    const a = generateDataset("circle", 7001);
    const b = generateDataset("circle", 7001);
    expect(a.train).toEqual(b.train);
    expect(a.test).toEqual(b.test);
  });

  it("holds the test set out from the training set", () => {
    const { train, test } = generateDataset("xor", 7001);
    expect(train).toHaveLength(TRAIN_POINTS);
    expect(test).toHaveLength(TEST_POINTS);
    const trainKeys = new Set(train.map((p) => `${p.x},${p.y}`));
    const overlap = test.filter((p) => trainKeys.has(`${p.x},${p.y}`));
    expect(overlap).toHaveLength(0);
  });

  it("keeps every point inside the plotted square", () => {
    for (const pattern of PATTERNS) {
      for (const point of generateDataset(pattern.id, 7001).train) {
        expect(point.x).toBeGreaterThanOrEqual(-1);
        expect(point.x).toBeLessThanOrEqual(1);
        expect(point.y).toBeGreaterThanOrEqual(-1);
        expect(point.y).toBeLessThanOrEqual(1);
      }
    }
  });

  it("balances the classes, so accuracy is a meaningful metric", () => {
    // Without this, a lazy all-one-class model could score well and the whole
    // capacity lesson would be unmeasurable.
    for (const pattern of PATTERNS) {
      const { test } = generateDataset(pattern.id, 7001);
      const positive = test.filter((p) => p.label === 1).length / test.length;
      expect(positive).toBeGreaterThan(0.4);
      expect(positive).toBeLessThan(0.6);
    }
  });

  it("makes every puzzle harder than guessing one class", () => {
    for (const pattern of PATTERNS) {
      const { test } = generateDataset(pattern.id, 7001);
      const positive = test.filter((p) => p.label === 1).length / test.length;
      const majority = Math.max(positive, 1 - positive);
      expect(pattern.target).toBeGreaterThan(majority);
    }
  });
});

describe("architecture", () => {
  it("totals the neurons across layers", () => {
    expect(architectureOf(hidden([[4, "relu"], [3, "tanh"]])).totalNeurons).toBe(7);
    expect(architectureOf([]).totalNeurons).toBe(0);
  });

  it("treats an empty network as linear", () => {
    expect(isEffectivelyLinear(architectureOf([]))).toBe(true);
  });

  it("treats an all-linear stack as linear however deep it is", () => {
    const deep = architectureOf(
      hidden([[8, "linear"], [8, "linear"], [8, "linear"]]),
    );
    expect(deep.totalNeurons).toBe(24);
    expect(isEffectivelyLinear(deep)).toBe(true);
    expect(nonlinearLayerCount(deep)).toBe(0);
  });

  it("stops being linear as soon as one activation bends", () => {
    const mixed = architectureOf(hidden([[8, "linear"], [4, "relu"]]));
    expect(isEffectivelyLinear(mixed)).toBe(false);
    expect(nonlinearLayerCount(mixed)).toBe(1);
  });

  it("offers a linear activation, so the lesson is reachable", () => {
    expect(ACTIVATIONS).toContain("linear");
  });
});

describe("the surface grid", () => {
  it("covers the square once at the stated resolution", () => {
    const grid = surfaceGrid();
    expect(grid).toHaveLength(SURFACE_RESOLUTION * SURFACE_RESOLUTION);
    expect(grid[0]).toEqual([-1, -1]);
    expect(grid[grid.length - 1]).toEqual([1, 1]);
  });
});

describe("naming the failure", () => {
  const circle = patternById("circle");
  const straight = patternById("linear");

  it("does not call a linear model a failure on a linear puzzle", () => {
    // A straight cut is the *right* answer here. Naming it a failure would
    // teach the opposite of the lesson.
    const result = evaluate({
      pattern: straight,
      architecture: architectureOf([]),
      accuracy: 0.96,
      loss: 0.1,
    });
    expect(result.outcome).toBe("win");
    expect(result.failure).toBeNull();
  });

  it("names 'No non-linearity' for a linear stack on a curved puzzle", () => {
    const result = evaluate({
      pattern: circle,
      architecture: architectureOf(hidden([[8, "linear"], [8, "linear"]])),
      accuracy: 0.44,
      loss: 0.7,
    });
    expect(result.outcome).toBe("no-nonlinearity");
    expect(result.failure?.name).toBe("No non-linearity");
    // The fix must point at the activation, not at the neuron count.
    expect(result.failure?.detail).toMatch(/activation/i);
  });

  it("names 'Insufficient capacity' when the shape is right but too small", () => {
    const result = evaluate({
      pattern: circle,
      architecture: architectureOf(hidden([[2, "relu"]])),
      accuracy: 0.71,
      loss: 0.5,
    });
    expect(result.outcome).toBe("insufficient-capacity");
    expect(result.failure?.name).toBe("Insufficient capacity");
  });

  it("prefers the non-linearity diagnosis over the capacity one", () => {
    // Both are technically true of an empty network on a circle, but only one
    // has a fix the player can act on.
    const result = evaluate({
      pattern: circle,
      architecture: architectureOf([]),
      accuracy: 0.44,
      loss: 0.7,
    });
    expect(result.failure?.name).toBe("No non-linearity");
  });

  it("stops advising more neurons once the budget is spent", () => {
    const result = evaluate({
      pattern: patternById("spiral"),
      architecture: architectureOf(hidden([[8, "tanh"], [8, "tanh"], [4, "tanh"]])),
      accuracy: 0.8,
      loss: 0.4,
    });
    expect(result.outcome).toBe("insufficient-capacity");
    expect(result.failure?.detail).toMatch(/budget cap/i);
    expect(result.failure?.detail).not.toMatch(/budget to spend/i);
  });

  it("reports nothing before training has run", () => {
    const result = evaluate({
      pattern: circle,
      architecture: architectureOf(hidden([[4, "relu"]])),
      accuracy: null,
      loss: Number.NaN,
    });
    expect(result.outcome).toBe("untrained");
    expect(result.failure).toBeNull();
    expect(result.score).toBe(0);
  });
});

describe("scoring", () => {
  const circle = patternById("circle");

  it("rewards solving with fewer neurons", () => {
    const lean = evaluate({
      pattern: circle,
      architecture: architectureOf(hidden([[4, "relu"]])),
      accuracy: 0.93,
      loss: 0.2,
    });
    const bloated = evaluate({
      pattern: circle,
      architecture: architectureOf(hidden([[8, "relu"]])),
      accuracy: 0.93,
      loss: 0.2,
    });
    expect(lean.score).toBeGreaterThan(bloated.score);
    expect(lean.efficiency).toBeGreaterThan(bloated.efficiency);
  });

  it("never lets the efficiency bonus swing the result more than its weight", () => {
    const spent = evaluate({
      pattern: circle,
      architecture: architectureOf(hidden([[8, "relu"]])),
      accuracy: 1,
      loss: 0,
    });
    expect(spent.efficiency).toBeGreaterThanOrEqual(1 - EFFICIENCY_WEIGHT);
  });

  it("counts reaching the target as a win regardless of neurons spent", () => {
    // Regression guard: gating the win on the efficiency-scaled score made every
    // minimal solution lose, because spending 16 of a 20 budget caps the product
    // at 0.74 however accurate the network is.
    for (const pattern of PATTERNS) {
      const result = evaluate({
        pattern,
        architecture: architectureOf(
          hidden([[MAX_NEURONS_PER_LAYER, "relu"], [MAX_NEURONS_PER_LAYER, "relu"]]),
        ),
        accuracy: pattern.target,
        loss: 0.1,
      });
      expect(result.outcome, `${pattern.id} at exactly its target`).toBe("win");
      expect(result.solved).toBe(true);
    }
  });
});

describe("the capacity ladder", () => {
  // The lesson, measured. Each puzzle must be solvable by its stated minimal
  // architecture and NOT solvable one step below it — otherwise "more capacity
  // isn't free" is just a claim in the copy.
  //
  // These train real networks, hence the timeouts.

  it("solves a linear puzzle with no hidden layer at all", async () => {
    const pattern = patternById("linear");
    expect(pattern.minimalLayers).toEqual([]);
    const accuracy = await trainAndScore("linear", pattern.minimalLayers);
    expect(accuracy).toBeGreaterThanOrEqual(pattern.target);
  }, 120000);

  it("cannot solve a circle with any number of linear neurons", async () => {
    // 24 neurons, three layers, all linear: the composition is still one matrix.
    const wide = await trainAndScore(
      "circle",
      hidden([[8, "linear"], [8, "linear"], [8, "linear"]]),
    );
    const none = await trainAndScore("circle", []);
    expect(wide).toBeLessThan(patternById("circle").target);
    // And it is no better than having no hidden layer whatsoever — this is the
    // sharpest form of the lesson: those 24 neurons bought literally nothing.
    expect(Math.abs(wide - none)).toBeLessThan(0.1);
  }, 240000);

  it("needs four relu neurons for a circle, and three is not enough", async () => {
    const pattern = patternById("circle");
    // The stated minimal solution itself, so the copy and the check agree.
    expect(pattern.minimalLayers).toEqual(hidden([[4, "relu"]]));
    const enough = await trainAndScore("circle", pattern.minimalLayers);
    const tooFew = await trainAndScore("circle", hidden([[3, "relu"]]));
    expect(enough).toBeGreaterThanOrEqual(pattern.target + LADDER_MARGIN);
    expect(tooFew).toBeLessThanOrEqual(pattern.target - LADDER_MARGIN);
  }, 240000);

  it("needs four relu neurons for XOR, and three is not enough", async () => {
    const pattern = patternById("xor");
    expect(pattern.minimalLayers).toEqual(hidden([[4, "relu"]]));
    const enough = await trainAndScore("xor", pattern.minimalLayers);
    const tooFew = await trainAndScore("xor", hidden([[3, "relu"]]));
    expect(enough).toBeGreaterThanOrEqual(pattern.target + LADDER_MARGIN);
    expect(tooFew).toBeLessThanOrEqual(pattern.target - LADDER_MARGIN);
  }, 240000);

  it("needs depth for a spiral: one wide layer plateaus, two stacked clear it", async () => {
    const pattern = patternById("spiral");
    expect(pattern.minimalLayers).toEqual(hidden([[8, "relu"], [8, "relu"]]));
    const deep = await trainAndScore("spiral", pattern.minimalLayers);
    const wide = await trainAndScore("spiral", hidden([[8, "relu"]]));
    expect(deep).toBeGreaterThanOrEqual(pattern.target + LADDER_MARGIN);
    expect(wide).toBeLessThanOrEqual(pattern.target - LADDER_MARGIN);
    // Same activation, same width, only depth differs.
    expect(deep).toBeGreaterThan(wide);
  }, 240000);
});

describe("a network that stopped learning", () => {
  // The misdiagnosis this guards against: a deep narrow relu stack collapses to
  // a constant and used to be called "Insufficient capacity", sending the player
  // to add neurons to a network whose neurons were never the problem.

  it("names Dying ReLU when a relu layer dies, not Insufficient capacity", async () => {
    // The whole spiral budget in three layers: strictly more expressive than
    // the 8→8 that solves it, measured to end with layer 3 dead.
    const layers = hidden([[8, "relu"], [8, "relu"], [4, "relu"]]);
    const { accuracy, collapse, spread } = await trainAndDiagnose("spiral", layers);

    expect(spread).toBeLessThan(FLAT_OUTPUT_SPREAD);
    expect(collapse?.kind).toBe("dead-relu");
    expect(collapse?.layer).toBe(2);

    const result = evaluate({
      pattern: patternById("spiral"),
      architecture: architectureOf(layers),
      accuracy,
      loss: 0.6932,
      collapse,
    });
    expect(result.outcome).toBe("dead-network");
    expect(result.failure?.name).toBe("Dying ReLU");
    expect(result.failure?.detail).toMatch(/hidden layer 3/);
    // The false claim the old diagnosis made about this network.
    expect(result.failure?.detail).not.toMatch(/can bend the boundary/i);
    // It holds 8→8, so "not a lack of capacity" is a promise it can keep.
    expect(result.failure?.detail).toMatch(/not a lack of capacity/);

    // Worded as measured. The check covers the training and held-out points;
    // the heatmap covers the whole square, and this network's surface still
    // varies in the spiral's empty corners.
    expect(result.failure?.detail).toMatch(/every training and held-out point/);
    expect(result.failure?.detail).not.toMatch(/whole plane/i);
    const card = whyCardFor({
      kind: "trained",
      evaluation: result,
      pattern: patternById("spiral"),
      architecture: architectureOf(layers),
      collapse,
    });
    expect(card.body).toMatch(/every training and held-out point/);
    expect(card.body).not.toMatch(/single colou?r/i);
  }, 240000);

  it("names saturation, not a dead relu, when a tanh stack pins at its limits", async () => {
    const layers = hidden([[8, "tanh"], [8, "tanh"], [4, "tanh"]]);
    const { accuracy, collapse } = await trainAndDiagnose("spiral", layers);
    expect(collapse?.kind).toBe("saturated");
    expect(collapse?.activation).toBe("tanh");

    const result = evaluate({
      pattern: patternById("spiral"),
      architecture: architectureOf(layers),
      accuracy,
      loss: 0.6932,
      collapse,
    });
    expect(result.failure?.name).toBe("Saturated activations");
    expect(result.failure?.detail).toMatch(/Every tanh unit in hidden layer 3/);
    expect(result.failure?.detail).not.toMatch(/relu unit/i);
  }, 240000);

  it("finds nothing wrong with a network that learned", async () => {
    const { collapse, spread } = await trainAndDiagnose(
      "circle",
      hidden([[4, "relu"]]),
    );
    expect(spread).toBeGreaterThan(FLAT_OUTPUT_SPREAD);
    expect(collapse).toBeNull();
  }, 240000);

  it("measures spread as the range of the predictions", () => {
    expect(outputSpread([0.2, 0.9, 0.5])).toBeCloseTo(0.7, 10);
    expect(outputSpread([0.47, 0.47, 0.47])).toBe(0);
  });

  it("still calls an all-linear stack 'No non-linearity' first", () => {
    // The linear diagnosis has the more actionable fix, so it outranks a flat
    // output on a puzzle that needs a bend.
    const result = evaluate({
      pattern: patternById("circle"),
      architecture: architectureOf(hidden([[4, "linear"], [4, "linear"]])),
      accuracy: 0.44,
      loss: 0.69,
      collapse: { kind: "flat", layer: null, activation: null, output: 0.5 },
    });
    expect(result.outcome).toBe("no-nonlinearity");
  });
});

describe("a network that holds the minimal solution", () => {
  // The misdiagnosis this guards against: from the winning 8→8, "Add a layer"
  // and + give 8→8→3 — measured at 0.654 with the output still varying, so no
  // collapse is found — and it was called "Insufficient capacity", for a
  // network that strictly contains the one that solves the puzzle.

  it("knows which networks contain the minimal solution", () => {
    const holds = (pattern: PatternId, spec: Array<[number, Activation]>) =>
      containsMinimalSolution(architectureOf(hidden(spec)), patternById(pattern));

    // Minimal solution, wider, or deeper behind it: all contain it.
    expect(holds("xor", [[4, "relu"]])).toBe(true);
    expect(holds("circle", [[8, "relu"]])).toBe(true);
    expect(holds("xor", [[4, "relu"], [2, "relu"]])).toBe(true);
    expect(holds("xor", [[4, "relu"], [4, "relu"], [2, "relu"]])).toBe(true);
    expect(holds("spiral", [[8, "relu"], [8, "relu"], [3, "relu"]])).toBe(true);
    expect(holds("spiral", [[8, "relu"], [8, "relu"], [4, "tanh"]])).toBe(true);
    // Logistic regression is inside every network.
    expect(holds("linear", [])).toBe(true);
    expect(holds("linear", [[1, "relu"]])).toBe(true);

    // Too narrow, too shallow, the wrong activation, or a narrow layer in front.
    expect(holds("circle", [[1, "relu"]])).toBe(false);
    expect(holds("circle", [[3, "relu"]])).toBe(false);
    expect(holds("spiral", [[8, "relu"]])).toBe(false);
    expect(holds("spiral", [[8, "relu"], [4, "relu"], [8, "relu"]])).toBe(false);
    expect(holds("circle", [[4, "tanh"]])).toBe(false);
    expect(holds("circle", [[4, "linear"], [4, "relu"]])).toBe(false);
    expect(holds("circle", [])).toBe(false);
  });

  it.each<[PatternId, Array<[number, Activation]>]>([
    ["xor", [[4, "relu"], [2, "relu"]]],
    ["xor", [[4, "relu"], [4, "relu"], [2, "relu"]]],
    ["spiral", [[8, "relu"], [8, "relu"], [3, "relu"]]],
  ])(
    "names %s %j an optimisation failure, not insufficient capacity",
    async (id, spec) => {
      const pattern = patternById(id);
      const layers = hidden(spec);
      const architecture = architectureOf(layers);
      const { accuracy, collapse } = await trainAndDiagnose(id, layers);

      // Measured to miss without collapsing — the case the old code misnamed.
      expect(accuracy).toBeLessThan(pattern.target);
      expect(collapse).toBeNull();

      const result = evaluate({ pattern, architecture, accuracy, loss: 0.5, collapse });
      expect(result.outcome).toBe("optimisation-failure");
      expect(result.failure?.name).toBe("Optimisation failure");
      expect(result.failure?.detail).toMatch(/capacity is not what is missing/);
      expect(result.failure?.detail).toMatch(/Drop back to/);
      expect(result.failure?.detail).not.toMatch(/budget to spend|can bend the boundary/i);

      const card = whyCardFor({
        kind: "trained",
        evaluation: result,
        pattern,
        architecture,
        collapse,
      });
      expect(card.title).toMatch(/^Optimisation failure — \d+(\.\d)?% of the \d+% needed/);
      // Not dead, so none of the flat-output copy.
      expect(card.body).not.toMatch(/same answer|P\(class B\)/);
      expect(card.conceptHref).toBe("/concepts/gradient-descent");
      expect(card.conceptLabel).toBe("Local versus global minima");
    },
    240000,
  );

  it.each<[PatternId, Array<[number, Activation]>]>([
    ["circle", [[1, "relu"]]],
    ["circle", [[3, "relu"]]],
    ["spiral", [[8, "relu"]]],
  ])(
    "still names %s %j insufficient capacity",
    async (id, spec) => {
      const pattern = patternById(id);
      const architecture = architectureOf(hidden(spec));
      const { accuracy, collapse } = await trainAndDiagnose(id, hidden(spec));
      expect(accuracy).toBeLessThan(pattern.target);
      expect(collapse).toBeNull();
      const result = evaluate({ pattern, architecture, accuracy, loss: 0.5, collapse });
      expect(result.outcome).toBe("insufficient-capacity");
      expect(result.failure?.name).toBe("Insufficient capacity");
    },
    240000,
  );

  it("does not tell the minimal solution itself to drop back to itself", () => {
    // Not measured to happen on any backend, but the copy must still be true if
    // a device's rounding ever lands the circle's 4 relu under the bar.
    const result = evaluate({
      pattern: patternById("circle"),
      architecture: architectureOf(hidden([[4, "relu"]])),
      accuracy: 0.8,
      loss: 0.4,
    });
    expect(result.outcome).toBe("optimisation-failure");
    expect(result.failure?.detail).not.toMatch(/Drop back/);
    expect(result.failure?.detail).toMatch(/change the arrangement/);
  });

  it("only promises 'not a lack of capacity' for a dead network that has it", () => {
    const flat = { kind: "dead-relu" as const, layer: 1, activation: "relu" as const, output: 0.5 };
    const small = evaluate({
      pattern: patternById("circle"),
      architecture: architectureOf(hidden([[2, "relu"], [2, "relu"]])),
      accuracy: 0.502,
      loss: 0.69,
      collapse: flat,
    });
    expect(small.failure?.name).toBe("Dying ReLU");
    expect(small.failure?.detail).not.toMatch(/not a lack of capacity/);
  });
});

describe("the Score readout", () => {
  it("shows a score only for a run that was judged", () => {
    const pattern = patternById("circle");
    const architecture = architectureOf(hidden([[4, "relu"]]));
    const judged = (input: { accuracy: number | null; epochsRun?: number }) =>
      isScored(evaluate({ pattern, architecture, loss: 0.3, ...input }));

    expect(isScored(null)).toBe(false);
    // Stopped, or the fit never produced a measurement: "not scored".
    expect(judged({ accuracy: 0.9, epochsRun: 51 })).toBe(false);
    expect(judged({ accuracy: null })).toBe(false);
    // A win and a named failure both carry a real score.
    expect(judged({ accuracy: 0.91 })).toBe(true);
    expect(
      isScored(
        evaluate({
          pattern,
          architecture: architectureOf(hidden([[2, "relu"]])),
          accuracy: 0.676,
          loss: 0.55,
        }),
      ),
    ).toBe(true);
  });
});

describe("one neuron is one straight cut", () => {
  it("counts the cuts in front of the first bend", () => {
    expect(cutsBeforeFirstBend(architectureOf(hidden([[4, "relu"]])))).toBe(4);
    expect(
      cutsBeforeFirstBend(architectureOf(hidden([[1, "relu"], [8, "relu"]]))),
    ).toBe(1);
    // A narrow linear layer in front narrows every cut after it.
    expect(
      cutsBeforeFirstBend(architectureOf(hidden([[1, "linear"], [8, "relu"]]))),
    ).toBe(1);
    expect(
      cutsBeforeFirstBend(architectureOf(hidden([[2, "linear"], [8, "relu"]]))),
    ).toBe(2);
    expect(isSingleCut(architectureOf([]))).toBe(false);
    expect(isSingleCut(architectureOf(hidden([[1, "linear"]])))).toBe(false);
  });

  it("does not claim a single-neuron layer bends the boundary", () => {
    // The default "Add a layer" state: one relu neuron on the circle, measured
    // at 0.608 with a single straight edge on the surface.
    const architecture = architectureOf(hidden([[1, "relu"]]));
    const result = evaluate({
      pattern: patternById("circle"),
      architecture,
      accuracy: 0.608,
      loss: 0.588,
    });
    expect(result.outcome).toBe("insufficient-capacity");
    expect(result.failure?.detail).not.toMatch(/can bend the boundary/i);
    expect(result.failure?.detail).toMatch(/straight lines/i);

    const card = whyCardFor({
      kind: "trained",
      evaluation: result,
      pattern: patternById("circle"),
      architecture,
      collapse: null,
    });
    expect(card.body).not.toMatch(/bending|curve in the surface/i);
  });

  it("says so as soon as the one-neuron layer is added, before training", () => {
    const card = whyCardFor({
      kind: "architecture-changed",
      architecture: architectureOf(hidden([[1, "relu"]])),
      pattern: patternById("circle"),
      change: "layer-added",
    });
    expect(card.body).toMatch(/still a straight line/i);
  });
});

describe("a stopped run", () => {
  it("is measured but not judged, and scores nothing", () => {
    // Stopping the circle's known solution half way used to be called
    // "Insufficient capacity" — for the architecture that solves it.
    const result = evaluate({
      pattern: patternById("circle"),
      architecture: architectureOf(hidden([[4, "relu"]])),
      accuracy: 0.9,
      loss: 0.3,
      epochsRun: 51,
    });
    expect(result.outcome).toBe("stopped");
    expect(result.failure).toBeNull();
    expect(result.solved).toBe(false);
    expect(result.score).toBe(0);
    expect(result.epochsRun).toBe(51);
  });

  it("is scored normally once every epoch has run", () => {
    const result = evaluate({
      pattern: patternById("circle"),
      architecture: architectureOf(hidden([[4, "relu"]])),
      accuracy: 0.9,
      loss: 0.2,
      epochsRun: TRAIN_EPOCHS,
    });
    expect(result.outcome).toBe("win");
  });
});

describe("numbers in the copy", () => {
  it("never writes a miss as reaching the bar", () => {
    expect(shortOf(0.848, 0.85)).toBe("84.8%");
    expect(shortOf(0.849, 0.85)).toBe("84.9%");
    expect(shortOf(0.8, 0.85)).toBe("80%");

    const result = evaluate({
      pattern: patternById("circle"),
      architecture: architectureOf(hidden([[2, "relu"]])),
      accuracy: 0.848,
      loss: 0.4,
    });
    expect(result.failure?.detail).toMatch(
      /^84\.8% on held-out points, short of the 85%/,
    );
  });

  it("scores at the precision the Score readout shows", () => {
    // The circle's measured minimal solution: 0.912 held-out at 4 of 8 neurons.
    // Unrounded that is 0.798 — displayed as "80%" and denied the 80% star.
    const result = evaluate({
      pattern: patternById("circle"),
      architecture: architectureOf(hidden([[4, "relu"]])),
      accuracy: 0.912,
      loss: 0.18,
    });
    expect(result.score).toBe(0.8);
    expect(result.score).toBeGreaterThanOrEqual(HIGH_SCORE_THRESHOLD);
    // Rounding is to the displayed whole percent, never a free bump past it.
    expect(scoreOf(0.79, 1)).toBe(0.79);
    expect(scoreOf(0.7949, 1)).toBe(0.79);
  });

  it("leads every result card with the outcome and its number", () => {
    // The title is what screen readers are told. It has to carry the result.
    const pattern = patternById("circle");
    const architecture = architectureOf(hidden([[4, "relu"]]));
    const won = whyCardFor({
      kind: "trained",
      evaluation: evaluate({ pattern, architecture, accuracy: 0.912, loss: 0.18 }),
      pattern,
      architecture,
      collapse: null,
    });
    expect(won.title).toMatch(/^Solved: 91% held-out/);

    const linear = architectureOf(hidden([[4, "linear"]]));
    const lost = whyCardFor({
      kind: "trained",
      evaluation: evaluate({
        pattern,
        architecture: linear,
        accuracy: 0.44,
        loss: 0.69,
      }),
      pattern,
      architecture: linear,
      collapse: null,
    });
    expect(lost.title).toMatch(/^No non-linearity — stuck at 44%/);
  });

  it("links result cards to Concept Library sections that exist", () => {
    const pattern = patternById("circle");
    const architecture = architectureOf(hidden([[2, "relu"]]));
    const card = whyCardFor({
      kind: "trained",
      evaluation: evaluate({ pattern, architecture, accuracy: 0.676, loss: 0.55 }),
      pattern,
      architecture,
      collapse: null,
    });
    expect(card.conceptHref).toBe("/concepts/decision-boundaries");
    expect(card.conceptLabel).toBe("Bias and capacity");
  });
});

describe("budgets", () => {
  it("leaves room for the stated minimal solution on every puzzle", () => {
    for (const pattern of PATTERNS) {
      expect(pattern.budget).toBeGreaterThan(0);
      expect(pattern.budget).toBeLessThanOrEqual(3 * MAX_NEURONS_PER_LAYER);
    }
  });

  it("climbs in difficulty across the puzzle order", () => {
    for (let index = 1; index < PATTERNS.length; index += 1) {
      expect(PATTERNS[index]!.budget).toBeGreaterThanOrEqual(
        PATTERNS[index - 1]!.budget,
      );
    }
  });
});

describe("memory", () => {
  it("leaks no tensors when a model is built and disposed", () => {
    const before = tf.memory().numTensors;
    const model = buildModel(architectureOf(hidden([[4, "relu"]])), MODEL_SEED);
    const optimizer = model.optimizer;
    model.dispose();
    optimizer?.dispose();
    expect(tf.memory().numTensors).toBe(before);
  });
});

describe("the store's budget invariants", () => {
  // The budget is enforced by making illegal states unreachable, rather than by
  // failing the player after the fact. These check that no sequence of edits —
  // from either lane — can produce an over-budget architecture.
  const store = useNeuronForgeStore;

  beforeEach(() => {
    store.getState().setPattern("linear");
    store.getState().reset();
  });

  it("starts empty, which is a legal architecture", () => {
    expect(store.getState().layers).toEqual([]);
    expect(budgetRemaining(store.getState())).toBe(patternById("linear").budget);
  });

  it("refuses to add a layer once the budget is spent", () => {
    store.getState().setPattern("linear"); // budget 6
    for (let index = 0; index < MAX_LAYERS; index += 1) {
      store.getState().addLayer();
    }
    store.getState().setNeurons(0, MAX_NEURONS_PER_LAYER);
    expect(architectureOf(store.getState().layers).totalNeurons).toBeLessThanOrEqual(
      patternById("linear").budget,
    );
    expect(canAddLayer(store.getState())).toBe(false);
  });

  it("clamps a neuron increase to the remaining budget", () => {
    store.getState().setPattern("circle"); // budget 8
    store.getState().addLayer();
    store.getState().addLayer();
    // Ask for far more than the cap allows on the first layer.
    store.getState().setNeurons(0, 999);
    const total = architectureOf(store.getState().layers).totalNeurons;
    expect(total).toBeLessThanOrEqual(patternById("circle").budget);
    // The second layer still has its neuron: growth is capped, not stolen.
    expect(store.getState().layers[1]!.neurons).toBeGreaterThanOrEqual(1);
  });

  it("never lets canGrowLayer disagree with what setNeurons will do", () => {
    store.getState().setPattern("linear"); // budget 6
    store.getState().addLayer();
    for (let attempt = 0; attempt < 20; attempt += 1) {
      const before = store.getState().layers[0]!.neurons;
      const allowed = canGrowLayer(store.getState(), 0);
      store.getState().setNeurons(0, before + 1);
      const after = store.getState().layers[0]!.neurons;
      expect(after > before, `growth at ${before} neurons`).toBe(allowed);
    }
  });

  it("clamps an architecture arriving from the code lane", () => {
    store.getState().setPattern("circle"); // budget 8
    // Deliberately illegal: too many layers, too many neurons each.
    store.getState().applyArchitecture([
      { neurons: 99, activation: "relu" },
      { neurons: 99, activation: "relu" },
      { neurons: 99, activation: "relu" },
      { neurons: 99, activation: "relu" },
    ]);
    const { layers } = store.getState();
    expect(layers.length).toBeLessThanOrEqual(MAX_LAYERS);
    expect(architectureOf(layers).totalNeurons).toBeLessThanOrEqual(
      patternById("circle").budget,
    );
    for (const layer of layers) {
      expect(layer.neurons).toBeLessThanOrEqual(MAX_NEURONS_PER_LAYER);
      expect(layer.neurons).toBeGreaterThanOrEqual(1);
    }
  });

  it("clears stale results whenever the architecture changes", () => {
    // Otherwise the metric would show an accuracy belonging to a network the
    // player has already edited away from.
    store.getState().addLayer();
    store.getState().finishTraining({ accuracy: 0.99, surface: null });
    expect(store.getState().accuracy).toBe(0.99);

    store.getState().setNeurons(0, 2);
    expect(store.getState().accuracy).toBeNull();
    expect(store.getState().lastEvaluation).toBeNull();
    expect(store.getState().lossHistory).toEqual([]);
  });

  it("resets the architecture when the pattern changes", () => {
    store.getState().setPattern("spiral");
    store.getState().addLayer();
    store.getState().setNeurons(0, 8);
    store.getState().setPattern("linear");
    // A 8-neuron design would be over the linear puzzle's budget of 6.
    expect(store.getState().layers).toEqual([]);
  });

  it("names the failure on the store after training, not just in the return", () => {
    store.getState().setPattern("circle");
    store.getState().finishTraining({ accuracy: 0.44, surface: null });
    expect(store.getState().failure?.name).toBe("No non-linearity");
    expect(store.getState().won).toBe(false);
  });

  it("marks a win and keeps no failure when the target is met", () => {
    store.getState().setPattern("circle");
    store.getState().addLayer();
    store.getState().setActivation(0, "relu");
    store.getState().setNeurons(0, 4);
    store.getState().finishTraining({ accuracy: 0.93, surface: null });
    expect(store.getState().won).toBe(true);
    expect(store.getState().failure).toBeNull();
  });

  it("drops a result that belongs to a superseded run", () => {
    store.getState().setPattern("circle");
    store.getState().addLayer();
    const current = store.getState().beginTraining();
    const stale = store
      .getState()
      .finishTraining({ accuracy: 0.99, surface: null }, current - 1);
    expect(stale).toBeNull();
    expect(store.getState().training).toBe(true);
    expect(store.getState().accuracy).toBeNull();

    const applied = store
      .getState()
      .finishTraining({ accuracy: 0.5, surface: null }, current);
    expect(applied?.outcome).toBe("insufficient-capacity");
    expect(store.getState().training).toBe(false);
  });

  it("retries without wiping the architecture the failure told you to change", () => {
    store.getState().setPattern("circle");
    store.getState().addLayer();
    store.getState().setActivation(0, "linear");
    store.getState().setNeurons(0, 4);
    store.getState().finishTraining({ accuracy: 0.44, surface: null });
    expect(store.getState().failure?.name).toBe("No non-linearity");

    store.getState().retry();
    expect(store.getState().layers).toEqual([{ neurons: 4, activation: "linear" }]);
    expect(store.getState().failure).toBeNull();
    expect(store.getState().accuracy).toBeNull();
  });

  it("does not score a stopped run or bank progress for it", () => {
    useProgression.getState().setAdapter(createMemoryAdapter());
    useProgression.setState({ xp: 0, games: {}, badges: [], lastGain: null });
    store.getState().setPattern("circle");
    store.getState().addLayer();
    store.getState().setNeurons(0, 4);
    store
      .getState()
      .finishTraining({ accuracy: 0.95, surface: null, epochsRun: 51 });

    expect(store.getState().won).toBe(false);
    expect(store.getState().failure).toBeNull();
    expect(store.getState().whyCard?.title).toMatch(/^Stopped at epoch 51 of 160/);
    expect(useProgression.getState().games[SLUG]).toBeUndefined();
  });

  it("describes a code-lane edit by what it changed", () => {
    store.getState().setPattern("circle");
    store.getState().applyArchitecture([{ neurons: 4, activation: "relu" }]);
    store.getState().applyArchitecture([{ neurons: 4, activation: "tanh" }]);
    expect(store.getState().whyCard?.body).toMatch(/^New activation/);
    store.getState().applyArchitecture([
      { neurons: 4, activation: "tanh" },
      { neurons: 2, activation: "tanh" },
    ]);
    expect(store.getState().whyCard?.body).toMatch(/^Another layer/);
  });
});

describe("code-lane credit", () => {
  const store = useNeuronForgeStore;

  beforeEach(() => {
    useProgression.getState().setAdapter(createMemoryAdapter());
    useProgression.setState({ xp: 0, games: {}, badges: [], lastGain: null });
    store.getState().setPattern("circle");
    store.getState().addLayer();
    store.getState().setNeurons(0, 4);
  });

  it("is not earned by clicking Train while the code tab happens to be open", () => {
    store.getState().setLane("code");
    store.getState().finishTraining({ accuracy: 0.93, surface: null });
    expect(useProgression.getState().games[SLUG]?.completed).toBe(true);
    expect(useProgression.getState().games[SLUG]?.codeLaneCleared).toBe(false);
    store.getState().setLane("visual");
  });

  it("is earned by a solve that api.train() started", () => {
    store
      .getState()
      .finishTraining({ accuracy: 0.93, surface: null, fromCode: true });
    expect(useProgression.getState().games[SLUG]?.codeLaneCleared).toBe(true);
  });
});
