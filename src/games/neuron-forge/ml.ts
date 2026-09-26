import * as tf from "@tensorflow/tfjs";
import type { NamedFailure } from "@/engine/types";
import { clamp, gaussian, seededRandom } from "@/lib/utils";

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
 * Two more exist because capacity is not the only thing that can run out: a
 * network that has enough of the right shape can still fail to train. It is
 * named after what was measured — a dead or saturated layer ("Dying ReLU",
 * "Saturated activations"), or, when it holds the minimal solution and still
 * misses, "Optimisation failure" — never "Insufficient capacity".
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
 * put each puzzle's stated minimal solution clearly above its target and the
 * rung below it clearly under. It is NOT a promise that every extra neuron
 * helps — XOR with 5 relu scores 0.882, a point below 4's 0.892 — which is why
 * the targets sit several points clear of both rungs rather than between two
 * neighbouring widths. `capacity-ladder` in the test file pins the margins down,
 * because a different seed would quietly break the lesson.
 */
export const MODEL_SEED = 3;

export interface PatternSpec {
  id: PatternId;
  name: string;
  /** One line on what makes this shape hard. */
  hint: string;
  /** Neuron budget (spec: "a compute budget caps total neurons"). */
  budget: number;
  /**
   * Held-out accuracy that counts as solved.
   *
   * Set in the middle of the cliff between the minimal solution and the rung
   * below it, not just under the minimal solution. These networks are chaotic in
   * float rounding: the same code measured 1–3 points apart on the WebGL and CPU
   * backends, and Chromium's CPU backend differs from Node's (spiral 8×8 scored
   * 0.854 in one and 0.930 in the other). A target a point below the measured
   * winner is a coin toss on a real player's GPU.
   *
   * What is protected is the relu ladder — each puzzle's minimal solution and
   * the rung below it. The tightest margin measured on any backend is the
   * spiral's single layer of 8 relu, 0.79 against 0.82; the unit tests hold 2.5
   * points either side in Node and the playthrough 2 in Chromium. Nothing else
   * is: tanh, sigmoid and narrow second layers can land within a point or two of
   * a target (spiral 8 tanh → 8 tanh 0.816 against 0.82, circle 4 relu → 2 relu
   * 0.854 against 0.85), where backend rounding can decide the result.
   */
  target: number;
  /** Smallest hidden-layer arrangement known to solve it. For the copy. */
  minimalSolution: string;
  /**
   * The same arrangement as layers — the one the capacity-ladder tests train.
   * `containsMinimalSolution` compares the player's network against it.
   */
  minimalLayers: readonly Layer[];
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
    minimalLayers: [],
    needsNonlinearity: false,
  },
  {
    id: "circle",
    name: "Circle",
    hint: "No straight line can enclose a disc. This needs a bend.",
    budget: 8,
    // 3 relu 0.68–0.70, 4 relu 0.91–0.92 across backends.
    target: 0.85,
    minimalSolution: "one hidden layer of 4 relu neurons",
    minimalLayers: [{ neurons: 4, activation: "relu" }],
    needsNonlinearity: true,
  },
  {
    id: "xor",
    name: "XOR quadrants",
    hint: "Two opposite corners share a class. One line can never do that.",
    budget: 10,
    // 3 relu 0.73–0.74, 4 and 5 relu 0.876–0.892 across backends.
    target: 0.84,
    minimalSolution: "one hidden layer of 4 relu neurons",
    minimalLayers: [{ neurons: 4, activation: "relu" }],
    needsNonlinearity: true,
  },
  {
    id: "spiral",
    name: "Spiral",
    hint: "Two interleaved arms. This needs real depth, not just width.",
    budget: 20,
    // One layer of 8 relu 0.78–0.79; two layers of 8 0.854–0.930.
    target: 0.82,
    minimalSolution: "two hidden layers of 8 relu neurons",
    minimalLayers: [
      { neurons: 8, activation: "relu" },
      { neurons: 8, activation: "relu" },
    ],
    needsNonlinearity: true,
  },
] as const;

export function patternById(id: PatternId): PatternSpec {
  return PATTERNS.find((pattern) => pattern.id === id) ?? PATTERNS[0]!;
}

// ── The datasets ───────────────────────────────────────────────────────────

/** Radius² that splits [-1,1]² into equal halves, so the circle is balanced. */
const CIRCLE_R2 = 2 / Math.PI;

/**
 * How far each spiral arm winds.
 *
 * Chosen by measurement so that width alone is not enough but depth is — the
 * whole point of the level. At 2.5π, held-out accuracy goes 4 neurons 0.66 →
 * 8 neurons 0.79 → two layers of 8 0.93, so a single wide layer plateaus below
 * the 0.82 target and stacking clears it.
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
 * How many independent straight cuts the network makes before its first bend.
 *
 * Each neuron in the first non-linear layer sees one projection of the input — a
 * single straight cut through the plane — and linear layers in front of it can
 * only reduce how many different projections there are. Everything downstream
 * works with those cuts. So with ONE cut, every later layer sees a single number,
 * and the decision boundary can only be straight lines parallel to that cut:
 * one neuron is still a straight line, however deep the stack behind it.
 *
 * Infinity when there is no non-linear layer (that case is `isEffectivelyLinear`).
 */
export function cutsBeforeFirstBend(architecture: Architecture): number {
  let cuts = Number.POSITIVE_INFINITY;
  for (const layer of architecture.layers) {
    cuts = Math.min(cuts, layer.neurons);
    if (layer.activation !== "linear") return cuts;
  }
  return Number.POSITIVE_INFINITY;
}

/** True when a bend exists but is fed by a single cut, so no curve can form. */
export function isSingleCut(architecture: Architecture): boolean {
  return (
    !isEffectivelyLinear(architecture) && cutsBeforeFirstBend(architecture) === 1
  );
}

/**
 * True when the network can represent everything the puzzle's minimal solution
 * can — so a miss is not a lack of capacity, whatever else it is.
 *
 * A prefix check, deliberately conservative. The first layers must use the
 * minimal solution's activation and be at least as wide (extra units can be
 * zeroed out); every layer after them needs one unit, because on inputs bounded
 * to [-1, 1]² a single unit can carry the minimal network's answer through
 * unchanged (relu(z + c) = z + c once the bias c keeps z + c positive). A narrow
 * layer in front or in the middle — 8→4→8 on the spiral — does not count: it
 * may well be able to, but that is not something this check can promise.
 */
export function containsMinimalSolution(
  architecture: Architecture,
  pattern: PatternSpec,
): boolean {
  const minimal = pattern.minimalLayers;
  if (architecture.layers.length < minimal.length) return false;
  return architecture.layers.every((layer, index) => {
    const required = minimal[index];
    if (!required) return layer.neurons >= 1;
    return (
      layer.activation === required.activation && layer.neurons >= required.neurons
    );
  });
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

// ── A network that stopped learning ───────────────────────────────────────

/**
 * Below this spread in P(class B) across the held-out points, the network is
 * giving every held-out point the same answer.
 *
 * Measured, not guessed: every healthy network trained here spreads its
 * predictions by at least 0.14 (a linear model on the circle, which cannot do
 * better than the base rate), while every collapsed one spread by under 1e-4 —
 * exactly 0 for a dead relu layer, 8e-6 for a saturated tanh stack. Three orders
 * of magnitude separate the two, so the threshold is not a judgement call.
 */
export const FLAT_OUTPUT_SPREAD = 1e-3;
/** A layer whose every unit moves less than this across the training set is flat. */
const FLAT_UNIT_RANGE = 1e-3;
/** Past this fraction of the way to its limit, a tanh or sigmoid unit is pinned. */
const SATURATED = 0.99;

export type CollapseKind = "dead-relu" | "saturated" | "flat";

export interface Collapse {
  kind: CollapseKind;
  /** 0-based index of the first hidden layer that went flat, null if none did. */
  layer: number | null;
  activation: Activation | null;
  /**
   * The one probability the network now outputs, for every training and
   * held-out point. Measured on those points only: the heatmap covers the whole
   * square, and a collapsed network can still vary where no data lies (the
   * spiral's empty corners).
   */
  output: number;
}

export function outputSpread(predictions: ArrayLike<number>): number {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < predictions.length; index += 1) {
    const value = predictions[index] ?? 0;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return predictions.length === 0 ? 0 : max - min;
}

/**
 * Did training kill the network? And if so, where, and how?
 *
 * This is the failure "Insufficient capacity" used to be mistaken for. A deep,
 * narrow relu stack — 8→8→4 on the spiral, the whole budget — is strictly more
 * expressive than the 8→8 that solves it, yet it can end up answering the same
 * probability everywhere. Blaming capacity there sends the player to add neurons
 * to a network whose neurons are not the problem.
 *
 * The check is a direct measurement, not an inference from the score: push the
 * training points through each hidden layer and find the first one whose every
 * unit outputs the same value for every point. For relu that value is zero — the
 * units are dead, and a relu at zero passes back zero gradient, so nothing below
 * can be trained either. For tanh and sigmoid it is the activation's limit, where
 * the slope, and so the gradient, is all but zero.
 *
 * Returns null when the output still varies: a network that learned something is
 * judged on its accuracy, never on this.
 */
export function diagnoseCollapse(
  model: tf.LayersModel,
  architecture: Architecture,
  xs: number[][],
  predictions: ArrayLike<number>,
): Collapse | null {
  if (predictions.length === 0 || outputSpread(predictions) >= FLAT_OUTPUT_SPREAD) {
    return null;
  }

  let sum = 0;
  for (let index = 0; index < predictions.length; index += 1) {
    sum += predictions[index] ?? 0;
  }
  const output = sum / predictions.length;

  // Assigned from inside tidy() rather than returned, because tidy only returns
  // tensor containers and this is plain data.
  let found: Omit<Collapse, "output"> | null = null;
  tf.tidy(() => {
    let hidden: tf.Tensor = tf.tensor2d(xs);
    for (let index = 0; index < architecture.layers.length; index += 1) {
      const layer = architecture.layers[index]!;
      const dense = model.layers[index];
      if (!dense) break;
      hidden = dense.apply(hidden) as tf.Tensor;
      // Dying and saturating are things an activation does. A linear layer has
      // neither failure of its own, so the question is asked of the next bend.
      if (layer.activation === "linear") continue;

      const range = hidden.max(0).sub(hidden.min(0)).max().dataSync()[0] ?? 0;
      if (range >= FLAT_UNIT_RANGE) continue;

      let kind: CollapseKind = "flat";
      if (layer.activation === "relu") {
        // relu outputs are never negative, so a maximum of zero is all zeros.
        if ((hidden.max().dataSync()[0] ?? 0) <= 0) kind = "dead-relu";
      } else {
        // Distance of the least-saturated unit from its activation's midpoint:
        // tanh spans ±1 around 0, sigmoid 0–1 around 0.5.
        const middle = layer.activation === "tanh" ? 0 : 0.5;
        const half = layer.activation === "tanh" ? 1 : 0.5;
        const leastPinned = hidden.sub(middle).abs().min().dataSync()[0] ?? 0;
        if (leastPinned > half * SATURATED) kind = "saturated";
      }
      found = { kind, layer: index, activation: layer.activation };
      return;
    }
  });

  // TypeScript cannot see the assignment inside the callback, so it would
  // otherwise narrow `found` to its initial null.
  const flat = found as Omit<Collapse, "output"> | null;
  return flat
    ? { ...flat, output }
    : { kind: "flat", layer: null, activation: null, output };
}

// ── Evaluation ─────────────────────────────────────────────────────────────

export type Outcome =
  | "win"
  | "no-nonlinearity"
  | "dead-network"
  | "optimisation-failure"
  | "insufficient-capacity"
  | "stopped"
  | "untrained";

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
  /** Epochs the scored fit ran: `TRAIN_EPOCHS` unless it was stopped early. */
  epochsRun: number;
}

export interface EvaluateInput {
  pattern: PatternSpec;
  architecture: Architecture;
  /** Held-out accuracy after training, or null if never trained. */
  accuracy: number | null;
  loss: number;
  /** From `diagnoseCollapse`: set when the network ended up answering one value. */
  collapse?: Collapse | null;
  /**
   * Epochs the fit actually ran. Fewer than `TRAIN_EPOCHS` means the player
   * pressed Stop, and a half-trained network is not scored: judging it would
   * name a capacity failure for an architecture that was never given the chance.
   */
  epochsRun?: number;
}

/**
 * True when the evaluation carries a real score. A stopped or unfinished run
 * scores 0 by rule, not by measurement, so the Score readout must not show it
 * as "0%" beside a card that says "not scored".
 */
export function isScored(evaluation: Evaluation | null): evaluation is Evaluation {
  return (
    evaluation !== null &&
    evaluation.outcome !== "stopped" &&
    evaluation.outcome !== "untrained"
  );
}

/** Whole percent, the way the metric readouts show it. */
const percent = (value: number) => `${Math.round(value * 100)}%`;

/**
 * A shortfall, written so it can never read as reaching the bar.
 *
 * 0.848 against a 0.85 target rounds to "85%, short of the 85% this shape
 * needs", which contradicts itself. When whole-percent rounding would hide the
 * miss, show one decimal, rounded down so it stays below the target.
 */
export function shortOf(value: number, target: number): string {
  if (Math.round(value * 100) < Math.round(target * 100)) return percent(value);
  return `${(Math.floor(value * 1000) / 10).toFixed(1)}%`;
}

/**
 * The score, rounded to the whole percent the Score readout displays.
 *
 * Mastery's second star is a threshold on this number. Unrounded, the circle's
 * minimal solution scores 0.798 — shown as "80%" beside a star that asks for 80%
 * and is then withheld. Scoring at the precision the player reads keeps the star
 * and the readout in agreement.
 */
export function scoreOf(accuracy: number, efficiency: number): number {
  return clamp(Math.round(accuracy * efficiency * 100) / 100, 0, 1);
}

const plural = (count: number, word: string) =>
  `${count} ${word}${count === 1 ? "" : "s"}`;

/** `minimalSolution` is written lower-case, to sit mid-sentence. */
const capitalise = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);

/**
 * Score the architecture and name the failure — honestly.
 *
 * The diagnoses are checked most-specific first, because each one points at a
 * different control and a wrong name sends the player to the wrong one:
 *
 *   1. No non-linearity — only for patterns that genuinely require a bend. On a
 *      linearly-separable puzzle a linear model is the *right* answer, so calling
 *      it a failure there would teach the opposite of the lesson.
 *   2. A collapsed network (Dying ReLU, saturation) — the architecture could
 *      represent more; training killed it. Measured by `diagnoseCollapse`.
 *   3. Optimisation failure — it did not collapse, and it contains the minimal
 *      solution (`containsMinimalSolution`), so it has the capacity; training
 *      ended somewhere worse. From the winning 8→8, "Add a layer" and + give
 *      8→8→3, measured at 0.654 with the output still varying: calling that
 *      "Insufficient capacity" would send the player to add neurons to a network
 *      that already holds the answer.
 *   4. Insufficient capacity — what is left: it trained, it does not hold the
 *      minimal solution, and it fell short.
 *
 * A stopped run is measured but never judged, and scores nothing.
 */
export function evaluate({
  pattern,
  architecture,
  accuracy,
  loss,
  collapse = null,
  epochsRun,
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

  const stopped = epochsRun !== undefined && epochsRun < TRAIN_EPOCHS;
  const solved = accuracy !== null && !stopped && accuracy >= pattern.target;
  const score = accuracy === null || stopped ? 0 : scoreOf(accuracy, efficiency);

  let outcome: Outcome;
  let failure: NamedFailure | null = null;

  if (accuracy === null) {
    outcome = "untrained";
  } else if (stopped) {
    // Measured, reported, not judged — see `epochsRun`.
    outcome = "stopped";
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
  } else if (collapse !== null) {
    // Checked before capacity: a dead network is an optimisation failure, and
    // "not enough neurons" would send the player to the wrong control.
    outcome = "dead-network";
    failure = collapseFailure(collapse, accuracy, pattern, architecture);
  } else if (containsMinimalSolution(architecture, pattern)) {
    outcome = "optimisation-failure";
    failure = optimisationFailure(accuracy, pattern, architecture);
  } else {
    outcome = "insufficient-capacity";
    const remaining = budget - totalNeurons;
    const shape = isSingleCut(architecture)
      ? // One cut in front of the first bend: the boundary is straight lines,
        // so "can bend, just not far enough" would describe a curve that is not
        // on screen.
        `Your first hidden layer makes a single cut, so every layer after it sees one number and the boundary can only be straight lines parallel to that cut — curves come from combining several cuts.`
      : `${plural(totalNeurons, "neuron")} across ${plural(
          architecture.layers.length,
          "hidden layer",
        )} can bend the boundary, just not far enough.`;
    failure = {
      name: "Insufficient capacity",
      detail: `${shortOf(accuracy, pattern.target)} on held-out points, short of the ${percent(
        pattern.target,
      )} this shape needs. ${shape} ${
        remaining >= 2
          ? `${capitalise(pattern.minimalSolution)} is enough, and you still have ${plural(
              remaining,
              "neuron",
            )} of budget to spend.`
          : // Near the cap, "add more" is not available and would be the wrong
            // advice anyway: the same neurons arranged differently, or with a
            // different activation, can be worth much more than extra ones.
            `You are at the budget cap, so extra neurons are not the answer — ${pattern.minimalSolution} solves this, so what is left to change is how they are arranged: width, depth, and which activation bends the boundary.`
      }`,
    };
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
    epochsRun: accuracy === null ? 0 : (epochsRun ?? TRAIN_EPOCHS),
  };
}

/**
 * Name a collapse after what was actually measured in the layer.
 *
 * "Dying ReLU" only when a relu layer's every unit is at zero for every training
 * point; "Saturated activations" only when a tanh or sigmoid layer is pinned at
 * its limit. Anything else gets the plain description, because naming a
 * mechanism that did not happen is the thing the pedagogy contract forbids.
 */
function collapseFailure(
  collapse: Collapse,
  accuracy: number,
  pattern: PatternSpec,
  architecture: Architecture,
): NamedFailure {
  // Measured on the training and held-out points only — see `Collapse.output`.
  const everywhere = `${percent(accuracy)} on held-out points, because the network gives every training and held-out point the same answer: P(class B) = ${collapse.output.toFixed(
    2,
  )}.`;
  const where =
    collapse.layer === null ? "" : `hidden layer ${collapse.layer + 1}`;
  // "Not a lack of capacity" is only a promise this network can keep when it
  // holds the minimal solution; a small one that died may be short of both.
  const fix = containsMinimalSolution(architecture, pattern)
    ? `This is the optimiser failing, not a lack of capacity: fewer, wider layers train where this one stalled, and ${pattern.minimalSolution} solves the puzzle.`
    : `This is the optimiser failing before capacity is even tested: fewer, wider layers train where this one stalled, and ${pattern.minimalSolution} solves the puzzle.`;

  if (collapse.kind === "dead-relu") {
    return {
      name: "Dying ReLU",
      detail: `${everywhere} Every relu unit in ${where} outputs zero for every training point. A relu at zero passes back zero gradient, so nothing above it gets a signal and nothing can revive it. ${fix}`,
    };
  }
  if (collapse.kind === "saturated") {
    return {
      name: "Saturated activations",
      detail: `${everywhere} Every ${collapse.activation} unit in ${where} is pinned at its limit for every training point, where its slope is almost zero — so almost no gradient flows back through it. ${fix}`,
    };
  }
  return {
    name: "Training collapsed",
    detail: `${everywhere} ${
      where === "" ? "The output" : `The output of ${where}`
    } stopped depending on the input, so the loss is stuck at the coin-flip value. ${fix}`,
  };
}

/**
 * A network with the capacity that still missed — see `containsMinimalSolution`.
 *
 * The output still varies, so none of the flat-surface copy applies: this is a
 * network that trained, just not well enough. The advice is the minimal
 * solution, which is measured to clear the bar, rather than more neurons.
 */
function optimisationFailure(
  accuracy: number,
  pattern: PatternSpec,
  architecture: Architecture,
): NamedFailure {
  const minimal = pattern.minimalLayers.length;
  const extra = architecture.layers.length - minimal;
  const isMinimal =
    extra === 0 &&
    architecture.layers.every(
      (layer, index) => layer.neurons === pattern.minimalLayers[index]?.neurons,
    );

  const holds = isMinimal
    ? `This is the arrangement measured to clear this bar: ${pattern.minimalSolution}.`
    : minimal === 0
      ? `A model with ${pattern.minimalSolution} clears this, and your layers can carry its one straight cut through to the output, so this network can draw everything that model can.`
      : extra === 0
        ? `This network is at least as wide as ${pattern.minimalSolution}, which clears this bar, so it can represent everything that network can.`
        : `${
            minimal === 1 ? "Its first layer is" : `Its first ${minimal} layers are`
          } at least as wide as ${pattern.minimalSolution}, which clears this bar, and ${
            extra === 1 ? "the layer after" : `the ${extra} layers after`
          } can pass that answer straight through — so this network can represent everything that one can.`;

  const why =
    extra > 0
      ? ` Every layer past the minimal solution is one more place for the gradient to fade or a unit to stop responding, which makes this more likely.`
      : "";

  const advice = isMinimal
    ? `The weights start from the same place every run, so this exact network will land here again: change the arrangement to change where training ends.`
    : `Drop back to ${pattern.minimalSolution}; it clears this puzzle.`;

  return {
    name: "Optimisation failure",
    detail: `${shortOf(accuracy, pattern.target)} on held-out points, short of the ${percent(
      pattern.target,
    )} this shape needs — and capacity is not what is missing. ${holds} Training ended somewhere worse than a solution this network contains.${why} ${advice}`,
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
