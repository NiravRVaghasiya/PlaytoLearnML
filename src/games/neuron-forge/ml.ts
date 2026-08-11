import * as tf from "@tensorflow/tfjs";
import type { NamedFailure } from "@/engine/types";
import { clamp, seededRandom } from "@/lib/utils";

/**
 * Neuron Forge — the real machine learning.
 *
 * A TensorFlow Playground remix: the player composes an architecture and a real
 * `tf.sequential` model trains on a 2-D toy dataset. The layer list the player
 * builds is passed straight to `buildModel`, so "player action = algorithm" is
 * literal — there is no simulation of a network anywhere in this game.
 *
 * ── What the game is actually about ─────────────────────────────────────────
 * The core intuition is that capacity is not just a neuron count: "depth, width,
 * AND activation determine what patterns a network can even represent." The
 * sharpest demonstration of that is the linear activation. A stack of linear
 * layers is itself a linear map, so a 3-layer, 24-neuron network with linear
 * activations cannot separate a circle — no matter how much of the budget it
 * spends. Adding neurons does nothing; changing one dropdown fixes it.
 *
 * That gives two failures with different names and different fixes:
 *   - Insufficient capacity  — the shape is right, there isn't enough of it
 *   - No non-linearity       — there is plenty of it, and it's the wrong shape
 *
 * Diagnosing those the same way would send the player to the wrong control.
 */

export type Activation = "relu" | "tanh" | "sigmoid" | "linear";

export const ACTIVATIONS: readonly Activation[] = [
  "relu",
  "tanh",
  "sigmoid",
  "linear",
] as const;

/** Activations that can actually bend a decision boundary. */
export const NONLINEAR_ACTIVATIONS: readonly Activation[] = [
  "relu",
  "tanh",
  "sigmoid",
] as const;

export interface Layer {
  neurons: number;
  activation: Activation;
}

export interface Architecture {
  layers: Layer[];
  totalNeurons: number;
}

export type PatternId = "linear" | "circle" | "xor" | "spiral";

export interface Point2D {
  x: number;
  y: number;
  label: 0 | 1;
}

export interface Dataset {
  train: Point2D[];
  test: Point2D[];
}

// ── Tuning ─────────────────────────────────────────────────────────────────

export const TRAIN_POINTS = 220;
export const TEST_POINTS = 500;
/** Label noise, kept low: this game is about representability, not noise. */
export const LABEL_NOISE = 0.02;

export const MAX_LAYERS = 3;
export const MAX_NEURONS_PER_LAYER = 8;
export const MIN_NEURONS_PER_LAYER = 1;

export const TRAIN_EPOCHS = 160;
export const TRAIN_BATCH = 32;
export const LEARNING_RATE = 0.06;

/** Fraction of the score that spending the whole budget costs. */
export const EFFICIENCY_WEIGHT = 0.25;

/**
 * Weight-initialization seed, fixed so the game is deterministic.
 *
 * Not arbitrary. Small relu networks on these puzzles settle into one of a few
 * distinct local optima, and which one depends entirely on the initial weights:
 * across seeds, XOR with 3–4 neurons returned the same small set of scores
 * (~0.89 / ~0.81 / ~0.72) in different orders. Seed 3 is the one measured to
 * give a non-decreasing capacity ladder on every puzzle, so adding a neuron
 * never *looks* like it made things worse. `capacity-ladder` in the test file
 * pins that down, because a different seed would quietly break the lesson.
 */
export const MODEL_SEED = 3;

export interface PatternSpec {
  id: PatternId;
  name: string;
  /** One line on what makes this shape hard. */
  hint: string;
  /** Neuron budget (spec: "a compute budget caps total neurons"). */
  budget: number;
  /** Held-out accuracy that counts as solved. */
  target: number;
  /** Smallest hidden-layer arrangement known to solve it. For the copy. */
  minimalSolution: string;
  /** True when no linear model can represent it. */
  needsNonlinearity: boolean;
}

/**
 * The escalating puzzles (spec: linear → circle → spiral → XOR, reordered to
 * linear → circle → XOR → spiral so difficulty climbs monotonically — XOR needs
 * strictly less capacity than a spiral).
 */
export const PATTERNS: readonly PatternSpec[] = [
  {
    id: "linear",
    name: "Straight line",
    hint: "One straight cut separates these. You may not need a hidden layer at all.",
    budget: 6,
    target: 0.94,
    minimalSolution: "no hidden layers",
    needsNonlinearity: false,
  },
  {
    id: "circle",
    name: "Circle",
    hint: "No straight line can enclose a disc. This needs a bend.",
    budget: 8,
    target: 0.9,
    minimalSolution: "one hidden layer of 4 relu neurons",
    needsNonlinearity: true,
  },
  {
    id: "xor",
    name: "XOR quadrants",
    hint: "Two opposite corners share a class. One line can never do that.",
    budget: 10,
    target: 0.88,
    minimalSolution: "one hidden layer of 4 relu neurons",
    needsNonlinearity: true,
  },
  {
    id: "spiral",
    name: "Spiral",
    hint: "Two interleaved arms. This needs real depth, not just width.",
    budget: 20,
    target: 0.85,
    minimalSolution: "two hidden layers of 8 relu neurons",
    needsNonlinearity: true,
  },
] as const;

export function patternById(id: PatternId): PatternSpec {
  return PATTERNS.find((pattern) => pattern.id === id) ?? PATTERNS[0]!;
}

// ── The datasets ───────────────────────────────────────────────────────────

function gaussian(random: () => number): number {
  const u = Math.max(random(), Number.EPSILON);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}

/** Radius² that splits [-1,1]² into equal halves, so the circle is balanced. */
const CIRCLE_R2 = 2 / Math.PI;

/**
 * How far each spiral arm winds.
 *
 * Chosen by measurement so that width alone is not enough but depth is — the
 * whole point of the level. At 2.5π, held-out accuracy goes 4 neurons 0.66 →
 * 8 neurons 0.79 → two layers of 8 0.93, so a single wide layer plateaus below
 * the 0.85 target and stacking clears it.
 *
 * 2π was rejected as too easy (one layer of 8 reached 0.97, so depth was never
 * required) and 3π because the ladder stopped being monotonic there — 8 neurons
 * scored *below* 4, which would teach that adding capacity hurts.
 */
const SPIRAL_SWEEP = 2.5 * Math.PI;
const SPIRAL_INNER_RADIUS = 0.2;
const SPIRAL_JITTER = 0.035;

function samplePoint(
  pattern: PatternId,
  random: () => number,
  index: number,
  count: number,
): Point2D {
  if (pattern === "spiral") {
    // Parametric: two arms offset by π, so the classes interleave.
    const arm = index % 2;
    const t = (Math.floor(index / 2) / Math.max(1, count / 2)) * SPIRAL_SWEEP;
    const radius =
      SPIRAL_INNER_RADIUS + (t / SPIRAL_SWEEP) * (1 - SPIRAL_INNER_RADIUS);
    const angle = t + arm * Math.PI;
    const clean: 0 | 1 = arm as 0 | 1;
    return {
      x: clamp(radius * Math.cos(angle) + gaussian(random) * SPIRAL_JITTER, -1, 1),
      y: clamp(radius * Math.sin(angle) + gaussian(random) * SPIRAL_JITTER, -1, 1),
      label: random() < LABEL_NOISE ? ((1 - clean) as 0 | 1) : clean,
    };
  }

  const x = random() * 2 - 1;
  const y = random() * 2 - 1;

  let clean: 0 | 1;
  if (pattern === "linear") clean = 0.8 * x + 0.6 * y > 0 ? 1 : 0;
  else if (pattern === "circle") clean = x * x + y * y < CIRCLE_R2 ? 1 : 0;
  else clean = x * y > 0 ? 1 : 0; // xor

  return {
    x,
    y,
    label: random() < LABEL_NOISE ? ((1 - clean) as 0 | 1) : clean,
  };
}

export function generateDataset(pattern: PatternId, seed: number): Dataset {
  const trainRandom = seededRandom(seed);
  const testRandom = seededRandom(seed + 5501);

  const train: Point2D[] = [];
  for (let index = 0; index < TRAIN_POINTS; index += 1) {
    train.push(samplePoint(pattern, trainRandom, index, TRAIN_POINTS));
  }

  const test: Point2D[] = [];
  for (let index = 0; index < TEST_POINTS; index += 1) {
    test.push(samplePoint(pattern, testRandom, index, TEST_POINTS));
  }

  return { train, test };
}

// ── The architecture ───────────────────────────────────────────────────────

export function architectureOf(layers: Layer[]): Architecture {
  return {
    layers: layers.map((layer) => ({ ...layer })),
    totalNeurons: layers.reduce((total, layer) => total + layer.neurons, 0),
  };
}

/**
 * True when the network can only represent a linear function of its inputs.
 *
 * Either there are no hidden layers at all, or every hidden layer uses a linear
 * activation — in which case the composition collapses to a single linear map and
 * the neuron count is irrelevant. This is what makes "No non-linearity" a
 * different diagnosis from "Insufficient capacity".
 */
export function isEffectivelyLinear(architecture: Architecture): boolean {
  if (architecture.layers.length === 0) return true;
  return architecture.layers.every((layer) => layer.activation === "linear");
}

/** Hidden layers that contribute a bend. */
export function nonlinearLayerCount(architecture: Architecture): number {
  return architecture.layers.filter((layer) => layer.activation !== "linear")
    .length;
}

/**
 * Build the model the player described.
 *
 * Seeded initializers, so identical architectures train identically. Without that
 * the decision surface would shift between runs for reasons the player did not
 * cause, and no assertion here could hold.
 */
export function buildModel(architecture: Architecture, seed = 1): tf.LayersModel {
  const model = tf.sequential();

  architecture.layers.forEach((layer, index) => {
    model.add(
      tf.layers.dense({
        units: Math.max(1, Math.round(layer.neurons)),
        activation: layer.activation,
        ...(index === 0 ? { inputShape: [2] } : {}),
        kernelInitializer: initializerFor(layer.activation, seed + index * 13),
        // A small positive bias keeps relu units on the active side of the
        // hinge at step 0. With zeros, a unit whose initial weights point away
        // from every training point outputs 0 for all of them, receives zero
        // gradient, and stays dead for the whole run — so the player adds a
        // neuron and accuracy does not move.
        biasInitializer:
          layer.activation === "relu"
            ? tf.initializers.constant({ value: 0.05 })
            : "zeros",
      }),
    );
  });

  model.add(
    tf.layers.dense({
      units: 1,
      activation: "sigmoid",
      ...(architecture.layers.length === 0 ? { inputShape: [2] } : {}),
      kernelInitializer: tf.initializers.glorotNormal({ seed: seed + 997 }),
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
 * Match the initializer to the activation.
 *
 * Glorot assumes an activation that is roughly linear near zero, which is true
 * of tanh and sigmoid and false of relu — relu discards half its input, so
 * Glorot's variance leaves relu layers under-scaled and prone to dying. He
 * initialization is the correct choice there.
 */
function initializerFor(
  activation: Activation,
  seed: number,
): ReturnType<typeof tf.initializers.heNormal> {
  return activation === "relu"
    ? tf.initializers.heNormal({ seed })
    : tf.initializers.glorotNormal({ seed });
}

export function toMatrix(points: Point2D[]): {
  xs: number[][];
  ys: number[];
} {
  return {
    xs: points.map((point) => [point.x, point.y]),
    ys: points.map((point) => point.label),
  };
}

export function accuracyFromPredictions(
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

/** Grid of input points for the decision-surface heatmap. */
export const SURFACE_RESOLUTION = 34;

export function surfaceGrid(): number[][] {
  const grid: number[][] = [];
  for (let row = 0; row < SURFACE_RESOLUTION; row += 1) {
    for (let column = 0; column < SURFACE_RESOLUTION; column += 1) {
      grid.push([
        -1 + (column / (SURFACE_RESOLUTION - 1)) * 2,
        -1 + (row / (SURFACE_RESOLUTION - 1)) * 2,
      ]);
    }
  }
  return grid;
}

// ── Evaluation ─────────────────────────────────────────────────────────────

export type Outcome =
  | "win"
  | "no-nonlinearity"
  | "insufficient-capacity"
  | "untrained"
  | "near-miss";

export interface Evaluation {
  pattern: PatternId;
  accuracy: number;
  loss: number;
  totalNeurons: number;
  budget: number;
  /** 1 when the architecture is at its most efficient. */
  efficiency: number;
  score: number;
  solved: boolean;
  outcome: Outcome;
  failure: NamedFailure | null;
}

export interface EvaluateInput {
  pattern: PatternSpec;
  architecture: Architecture;
  /** Held-out accuracy after training, or null if never trained. */
  accuracy: number | null;
  loss: number;
}

/**
 * Score the architecture and name the failure — honestly.
 *
 * "No non-linearity" is checked before "Insufficient capacity" and only for
 * patterns that genuinely require a bend. On a linearly-separable puzzle a linear
 * model is the *right* answer, so calling it a failure there would teach the
 * opposite of the lesson.
 */
export function evaluate({
  pattern,
  architecture,
  accuracy,
  loss,
}: EvaluateInput): Evaluation {
  const { totalNeurons } = architecture;
  const budget = pattern.budget;

  // Spending less of the budget scores better (spec: "efficiency bonus for
  // fewer neurons"). An empty architecture gets no bonus for being empty.
  const efficiency = clamp(
    1 - EFFICIENCY_WEIGHT * (totalNeurons / Math.max(1, budget)),
    1 - EFFICIENCY_WEIGHT,
    1,
  );

  const solved = accuracy !== null && accuracy >= pattern.target;
  const score = accuracy === null ? 0 : clamp(accuracy * efficiency, 0, 1);
  const percent = (value: number) => `${Math.round(value * 100)}%`;

  let outcome: Outcome;
  let failure: NamedFailure | null = null;

  if (accuracy === null) {
    outcome = "untrained";
  } else if (solved) {
    // Reaching the target *is* the win (spec: "solve the pattern under the
    // neuron budget; efficiency bonus for fewer neurons"). Efficiency scales the
    // score afterwards, it does not gate the win — gating on it would fail every
    // minimal solution measured here, since spending 16 of a 20 budget caps the
    // product at 0.74 no matter how accurate the network is.
    outcome = "win";
  } else if (pattern.needsNonlinearity && isEffectivelyLinear(architecture)) {
    outcome = "no-nonlinearity";
    failure = {
      name: "No non-linearity",
      detail:
        architecture.layers.length === 0
          ? `With no hidden layer the model is a single straight cut, and ${percent(
              accuracy,
            )} is the best a straight cut can do on this shape. Neurons won't help until there's a layer to put them in.`
          : `${totalNeurons} neurons across ${architecture.layers.length} layer${
              architecture.layers.length === 1 ? "" : "s"
            }, every one of them linear — so the whole stack collapses back to a single straight cut. Stuck at ${percent(
              accuracy,
            )}. Adding neurons cannot fix this; changing one activation can.`,
    };
  } else if (!solved) {
    outcome = "insufficient-capacity";
    const remaining = budget - totalNeurons;
    failure = {
      name: "Insufficient capacity",
      detail: `${percent(accuracy)} on held-out points, short of the ${percent(
        pattern.target,
      )} this shape needs. ${totalNeurons} neuron${
        totalNeurons === 1 ? "" : "s"
      } across ${architecture.layers.length} hidden layer${
        architecture.layers.length === 1 ? "" : "s"
      } can bend the boundary, just not far enough. ${
        remaining >= 2
          ? `${pattern.minimalSolution} is enough, and you still have ${remaining} neuron${
              remaining === 1 ? "" : "s"
            } of budget to spend.`
          : // Near the cap, "add more" is not available and would be the wrong
            // advice anyway: the same neurons arranged differently, or with a
            // different activation, can be worth much more than extra ones.
            `You are at the budget cap, so extra neurons are not the answer — ${pattern.minimalSolution} solves this, so what is left to change is how they are arranged: depth, and which activation bends the boundary.`
      }`,
    };
  } else {
    outcome = "near-miss";
  }

  return {
    pattern: pattern.id,
    accuracy: accuracy ?? 0,
    loss,
    totalNeurons,
    budget,
    efficiency,
    score,
    solved,
    outcome,
    failure,
  };
}

// ── Reveal the math ────────────────────────────────────────────────────────

export const MATH_EQUATION = String.raw`h^{(1)} = \sigma\!\left(W^{(1)}x + b^{(1)}\right)
\qquad
h^{(2)} = \sigma\!\left(W^{(2)}h^{(1)} + b^{(2)}\right)
\qquad
\hat{y} = \mathrm{sigmoid}\!\left(w^\top h^{(L)} + b\right)
\\[1.2em]
\text{if } \sigma(z) = z: \quad W^{(2)}\left(W^{(1)}x + b^{(1)}\right) + b^{(2)} = \underbrace{W^{(2)}W^{(1)}}_{\text{one matrix}}x + \text{const}`;

export const MATH_CODE = `// Your layer list, handed straight to TensorFlow.js.
export function buildModel(architecture) {
  const model = tf.sequential();

  architecture.layers.forEach((layer, index) => {
    model.add(tf.layers.dense({
      units: layer.neurons,
      activation: layer.activation,   // <- this is the important one
      ...(index === 0 ? { inputShape: [2] } : {}),
    }));
  });

  // One output neuron: probability of class 1.
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({
    optimizer: tf.train.adam(${LEARNING_RATE}),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}`;

export const MATH_NOTES = `σ is the activation. The second line is the whole reason activation matters: if σ is the identity, two stacked layers multiply out to a single matrix, so the network is a straight cut however many neurons you give it. Any bend in the decision boundary comes from σ being non-linear — relu, tanh or sigmoid — and depth then composes those bends into more intricate shapes. That is why a linearly-separable puzzle needs no hidden layer, a circle needs one bend, and a spiral needs bends applied to bends.`;
