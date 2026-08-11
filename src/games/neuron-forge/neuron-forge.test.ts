import * as tf from "@tensorflow/tfjs";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import {
  ACTIVATIONS,
  EFFICIENCY_WEIGHT,
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
  type Layer,
  type PatternId,
} from "./ml";
import {
  budgetRemaining,
  canAddLayer,
  canGrowLayer,
  useNeuronForgeStore,
} from "./store";

beforeAll(async () => {
  await tf.ready();
});

const hidden = (spec: Array<[number, Activation]>): Layer[] =>
  spec.map(([neurons, activation]) => ({ neurons, activation }));

/** Train the architecture on the puzzle exactly as the game does. */
async function trainAndScore(
  pattern: PatternId,
  layers: Layer[],
): Promise<number> {
  const dataset = generateDataset(pattern, 7001);
  const train = toMatrix(dataset.train);
  const test = toMatrix(dataset.test);

  const model = buildModel(architectureOf(layers), MODEL_SEED);
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

  xs.dispose();
  ys.dispose();
  const optimizer = model.optimizer;
  model.dispose();
  optimizer?.dispose();
  return accuracy;
}

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
    const accuracy = await trainAndScore("linear", []);
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
    const enough = await trainAndScore("circle", hidden([[4, "relu"]]));
    const tooFew = await trainAndScore("circle", hidden([[3, "relu"]]));
    expect(enough).toBeGreaterThanOrEqual(pattern.target);
    expect(tooFew).toBeLessThan(pattern.target);
  }, 240000);

  it("needs four relu neurons for XOR, and three is not enough", async () => {
    const pattern = patternById("xor");
    const enough = await trainAndScore("xor", hidden([[4, "relu"]]));
    const tooFew = await trainAndScore("xor", hidden([[3, "relu"]]));
    expect(enough).toBeGreaterThanOrEqual(pattern.target);
    expect(tooFew).toBeLessThan(pattern.target);
  }, 240000);

  it("needs depth for a spiral: one wide layer plateaus, two stacked clear it", async () => {
    const pattern = patternById("spiral");
    const deep = await trainAndScore("spiral", hidden([[8, "relu"], [8, "relu"]]));
    const wide = await trainAndScore("spiral", hidden([[8, "relu"]]));
    expect(deep).toBeGreaterThanOrEqual(pattern.target);
    expect(wide).toBeLessThan(pattern.target);
    // Same activation, same width, only depth differs.
    expect(deep).toBeGreaterThan(wide);
  }, 240000);
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
});
