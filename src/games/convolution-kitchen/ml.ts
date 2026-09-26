import * as tf from "@tensorflow/tfjs";
import type { NamedFailure } from "@/engine/types";
import { clamp, seededRandom } from "@/lib/utils";

/**
 * Convolution Kitchen — filters, feature maps, pooling.
 *
 * The player designs 3x3 kernels and stacks them into layers. A FIXED linear
 * classifier sits on top, so the detection score can only move because the
 * features moved — the same discipline Feature Forge uses, and for the same
 * reason: if the model could grow, "my filters got better" and "my model got
 * bigger" would be indistinguishable.
 *
 * ── Why the dishes are stripes ──────────────────────────────────────────────
 * Four classes, built so that the lessons are provable rather than asserted:
 *
 *   vertical            all vertical stripes
 *   horizontal          all horizontal stripes
 *   vertical-over       vertical stripes above, horizontal below
 *   horizontal-over     horizontal stripes above, vertical below
 *
 * Every image has the same mean brightness (up to noise), so a filter that only
 * averages pixels cannot separate anything — which makes "you built a blur" a
 * fact about this dataset rather than a hunch.
 *
 * Every image has random stripe phase, so no fixed template matches. That is what
 * puts the raw-pixel baseline at chance and makes convolution necessary rather
 * than merely helpful.
 *
 * And the last two classes contain exactly the same amount of vertical and
 * horizontal texture as each other. After the pipeline's global pooling, a single
 * conv layer produces one number per kernel — "how much of this pattern is in the
 * picture" — and those numbers are identical for the two of them by construction.
 * So one layer cannot beat roughly three classes out of four no matter which
 * kernels go in it, and the way past that ceiling is a SECOND layer that turns
 * "where the vertical texture is" into a number before the pooling throws the
 * position away. That is hierarchy — edges, then textures, then arrangement — and
 * here it is forced by the data instead of claimed in a caption.
 *
 * ── Deviations from the spec, flagged ───────────────────────────────────────
 * 1. No webcam. The spec offers it as optional. A camera prompt in a teaching
 *    site is a privacy cost with no pedagogical return, and worse, an arbitrary
 *    photo would destroy the symmetry argument above — the ceiling is only
 *    provable because the dataset was built to make it provable.
 * 2. `stride` and `padding` are fixed at 1 and "valid" rather than exposed on
 *    `Kernel` as the spec's data model suggests. Stride above 1 is pooling wearing
 *    a different hat, and it would give the player two controls for one idea.
 *    "Valid" is the honest choice at 3x3: padding invents a border of zeros, and
 *    an invented border is an invented edge, which is a lie a detector will
 *    happily report. Both are explained in the math drawer instead.
 * 3. Layer 2 is DEPTHWISE. A standard conv layer mixes input channels, so its
 *    kernel is 3x3xC and cannot be drawn as one 3x3 grid of numbers — which is
 *    precisely the point at which hand-designing stops being possible. Depthwise
 *    separable convolution is a real building block (MobileNet), it keeps layer 2
 *    hand-designable, and the thing it gives up is exactly what "Let it learn"
 *    exists to demonstrate.
 */

// ── The kitchen's dishes ──────────────────────────────────────────────────

export const IMAGE_SIZE = 24;
export const PIXELS = IMAGE_SIZE * IMAGE_SIZE;

/** Stripe period in pixels, jittered per plate within this range. */
export const PERIOD_MIN = 3.5;
export const PERIOD_MAX = 5;

export type ClassId = 0 | 1 | 2 | 3;

export interface DishSpec {
  id: ClassId;
  label: string;
  short: string;
  /** What a cook has to notice to name it. */
  tell: string;
}

export const DISHES: readonly DishSpec[] = [
  {
    id: 0,
    label: "All vertical",
    short: "vertical",
    tell: "Vertical stripes, edge to edge.",
  },
  {
    id: 1,
    label: "All horizontal",
    short: "horizontal",
    tell: "Horizontal stripes, edge to edge.",
  },
  {
    id: 2,
    label: "Vertical on top",
    short: "V over H",
    tell: "Vertical stripes above, horizontal below.",
  },
  {
    id: 3,
    label: "Horizontal on top",
    short: "H over V",
    tell: "Horizontal stripes above, vertical below.",
  },
] as const;

export const CLASS_COUNT = DISHES.length;
/** Accuracy from guessing. The floor any real filter has to beat. */
export const CHANCE_RATE = 1 / CLASS_COUNT;

export interface Sample {
  /** Row-major, length PIXELS, roughly 0..1. */
  pixels: Float32Array;
  label: ClassId;
  /** Where the two halves meet, or null for the uniform classes. */
  splitRow: number | null;
}

export interface Dataset {
  train: Sample[];
  validation: Sample[];
}

export const TRAIN_SIZE = 600;
export const VALIDATION_SIZE = 400;

/**
 * Stripe value at a coordinate.
 *
 * A sinusoid rather than a hard square wave, and that is a measured decision
 * rather than an aesthetic one. Thresholded stripes quantise the phase: however
 * finely you randomise the offset, the pattern snaps to one of four pixel
 * arrangements, and a linear classifier on raw pixels simply memorises the four —
 * it scored 65% that way, which would have made "the model cannot do this without
 * convolution" a false claim. A smooth profile makes sub-pixel phase produce
 * genuinely different pixel values, and the raw-pixel baseline drops to chance
 * where it belongs.
 */
function stripe(coordinate: number, phase: number, period: number): number {
  return 0.5 + 0.5 * Math.sin((2 * Math.PI * (coordinate + phase)) / period);
}

/**
 * One plated dish.
 *
 * The nuisance variation is not decoration. Random phase is what stops a fixed
 * pixel template from working; random brightness and contrast are what stop the
 * mean and the range from carrying the answer. Together they leave orientation and
 * arrangement as the only signal in the image, which is the whole design.
 */
function makeSample(label: ClassId, random: () => number): Sample {
  const pixels = new Float32Array(PIXELS);
  const periodA = PERIOD_MIN + random() * (PERIOD_MAX - PERIOD_MIN);
  const periodB = PERIOD_MIN + random() * (PERIOD_MAX - PERIOD_MIN);
  const phaseA = random() * periodA;
  const phaseB = random() * periodB;
  const brightness = (random() - 0.5) * 0.2;
  const contrast = 0.8 + random() * 0.2;
  const noise = 0.06;

  // Randomised so the seam itself is not a landmark: a 3x3 kernel that straddles
  // a boundary at a fixed row would leak the answer to a single layer.
  const splitRow =
    label === 2 || label === 3
      ? Math.floor(IMAGE_SIZE / 2 - 2 + random() * 5)
      : null;

  for (let row = 0; row < IMAGE_SIZE; row += 1) {
    for (let col = 0; col < IMAGE_SIZE; col += 1) {
      const topHalf = splitRow === null || row < splitRow;
      let vertical: boolean;
      if (label === 0) vertical = true;
      else if (label === 1) vertical = false;
      else if (label === 2) vertical = topHalf;
      else vertical = !topHalf;

      // Vertical stripes vary along x; horizontal stripes vary along y.
      const phase = topHalf ? phaseA : phaseB;
      const period = topHalf ? periodA : periodB;
      const base = vertical
        ? stripe(col, phase, period)
        : stripe(row, phase, period);

      const value =
        base * contrast +
        brightness +
        (random() - 0.5) * 2 * noise;
      pixels[row * IMAGE_SIZE + col] = clamp(value, 0, 1);
    }
  }

  return { pixels, label, splitRow };
}

/** A balanced, seeded kitchen. */
export function generateDataset(seed = 4711): Dataset {
  const random = seededRandom(seed);
  const build = (count: number): Sample[] => {
    const samples: Sample[] = [];
    for (let index = 0; index < count; index += 1) {
      samples.push(makeSample((index % CLASS_COUNT) as ClassId, random));
    }
    // Shuffle so a batch is never one class.
    for (let index = samples.length - 1; index > 0; index -= 1) {
      const swap = Math.floor(random() * (index + 1));
      [samples[index], samples[swap]] = [samples[swap]!, samples[index]!];
    }
    return samples;
  };

  return { train: build(TRAIN_SIZE), validation: build(VALIDATION_SIZE) };
}

// ── Kernels ───────────────────────────────────────────────────────────────

export const KERNEL_SIZE = 3;
export const KERNEL_CELLS = KERNEL_SIZE * KERNEL_SIZE;
export const WEIGHT_MIN = -2;
export const WEIGHT_MAX = 2;

export interface Kernel {
  id: string;
  label: string;
  /** Row-major, length 9, integers in WEIGHT_MIN..WEIGHT_MAX. */
  weights: number[];
}

export interface KernelPreset {
  label: string;
  hint: string;
  weights: readonly number[];
}

/**
 * Starting points, not answers.
 *
 * The two blurs are in the list deliberately. They are the most natural thing a
 * newcomer reaches for — a 3x3 average feels like "look at the neighbourhood" —
 * and on this dataset they are provably worthless. Hiding them would hide the
 * mistake worth making.
 */
export const KERNEL_PRESETS: readonly KernelPreset[] = [
  {
    label: "Vertical edge",
    hint: "Left column against right column. Fires on up-and-down boundaries.",
    weights: [1, 0, -1, 2, 0, -2, 1, 0, -1],
  },
  {
    label: "Horizontal edge",
    hint: "Top row against bottom row. Fires on side-to-side boundaries.",
    weights: [1, 2, 1, 0, 0, 0, -1, -2, -1],
  },
  {
    label: "Diagonal edge",
    hint: "One corner against the other.",
    weights: [2, 1, 0, 1, 0, -1, 0, -1, -2],
  },
  {
    label: "Centre spot",
    hint: "Centre against its ring. Fires on small bright dots.",
    weights: [-1, -1, -1, -1, 2, -1, -1, -1, -1],
  },
  {
    label: "Pass through",
    hint: "Centre only. Hands the map on unchanged — useful in layer 2.",
    weights: [0, 0, 0, 0, 1, 0, 0, 0, 0],
  },
  {
    label: "Blur",
    hint: "Averages the neighbourhood. Every weight the same sign.",
    weights: [1, 1, 1, 1, 1, 1, 1, 1, 1],
  },
  {
    label: "Sharpen",
    hint: "Centre boosted, ring subtracted.",
    weights: [0, -1, 0, -1, 2, -1, 0, -1, 0],
  },
  {
    label: "Blank",
    hint: "All zeros. Outputs nothing at all.",
    weights: [0, 0, 0, 0, 0, 0, 0, 0, 0],
  },
] as const;

export const presetByLabel = (label: string): KernelPreset | undefined =>
  KERNEL_PRESETS.find((preset) => preset.label === label);

let kernelCounter = 0;
export function makeKernel(preset: KernelPreset | string = "Blank"): Kernel {
  const found =
    typeof preset === "string" ? presetByLabel(preset) : preset;
  const chosen = found ?? KERNEL_PRESETS.at(-1)!;
  kernelCounter += 1;
  return {
    id: `k${kernelCounter}`,
    label: chosen.label,
    weights: [...chosen.weights],
  };
}

/** For tests and reproducible fixtures. */
export function resetKernelIds(): void {
  kernelCounter = 0;
}

export const kernelSum = (kernel: Kernel): number =>
  kernel.weights.reduce((total, weight) => total + weight, 0);

export const kernelMagnitude = (kernel: Kernel): number =>
  kernel.weights.reduce((total, weight) => total + Math.abs(weight), 0);

/**
 * True when every non-zero weight shares a sign.
 *
 * Such a kernel computes a weighted average of nine pixels, and nothing else. On
 * this dataset that is a measurement of brightness, which was randomised on
 * purpose — so this predicate is not a style opinion, it identifies filters that
 * provably carry no signal.
 */
export function isSingleSigned(kernel: Kernel): boolean {
  const positives = kernel.weights.filter((weight) => weight > 0).length;
  const negatives = kernel.weights.filter((weight) => weight < 0).length;
  return positives === 0 || negatives === 0;
}

/** All zeros. Outputs a constant, so it cannot even measure brightness. */
export const isBlank = (kernel: Kernel): boolean =>
  kernel.weights.every((weight) => weight === 0);

/** A named preset if the weights match one exactly, else null. */
export function describeKernel(kernel: Kernel): string {
  const match = KERNEL_PRESETS.find((preset) =>
    preset.weights.every((weight, index) => weight === kernel.weights[index]),
  );
  return match ? match.label : "Custom";
}

// ── Layers ────────────────────────────────────────────────────────────────

export type PoolType = "none" | "max" | "avg";

export interface PoolSpec {
  id: PoolType;
  label: string;
  hint: string;
}

export const POOL_TYPES: readonly PoolSpec[] = [
  {
    id: "none",
    label: "No pooling",
    hint: "Keep the map at full size. The next layer sees three pixels.",
  },
  {
    id: "max",
    label: "Max 2x2",
    hint: "Strongest response in each 2x2 block. Asks: is it present anywhere?",
  },
  {
    id: "avg",
    label: "Average 2x2",
    hint: "Mean response in each 2x2 block. Asks: how much of it is there?",
  },
] as const;

export interface Layer {
  id: string;
  kernels: Kernel[];
  pool: PoolType;
}

export const MAX_LAYERS = 2;
export const MAX_KERNELS_PER_LAYER = 4;
/** Total filters across the stack. The spec scores "fewest" filters. */
export const FILTER_BUDGET = 6;

let layerCounter = 0;
export function makeLayer(kernels: Kernel[], pool: PoolType = "avg"): Layer {
  layerCounter += 1;
  return { id: `L${layerCounter}`, kernels, pool };
}

export function resetLayerIds(): void {
  layerCounter = 0;
}

export const filtersUsed = (layers: readonly Layer[]): number =>
  layers.reduce((total, layer) => total + layer.kernels.length, 0);

/**
 * Channels leaving the stack.
 *
 * Layer 2 is depthwise, so it multiplies rather than replaces: every layer-1
 * channel is filtered by every layer-2 kernel.
 */
export const outputChannels = (layers: readonly Layer[]): number =>
  layers
    .filter((layer) => layer.kernels.length > 0)
    .reduce((total, layer) => total * layer.kernels.length, 1);

/**
 * What "Let it learn" depends on: how many kernels each layer holds and how it
 * pools. Not the weights — those are exactly what gradient descent replaces — so
 * a learned result stays valid through weight and preset edits and goes stale the
 * moment a filter, a layer or a pooling choice changes.
 */
export const shapeKey = (layers: readonly Layer[]): string =>
  layers
    .filter((layer) => layer.kernels.length > 0)
    .map((layer) => `${layer.kernels.length}:${layer.pool}`)
    .join("|");

/** Human-readable channel names, in the order TF emits them. */
export function channelLabels(layers: readonly Layer[]): string[] {
  if (layers.length === 0) return [];
  const first = layers[0]!.kernels.map((kernel) => kernel.label);
  if (layers.length === 1) return first;
  const second = layers[1]!.kernels.map((kernel) => kernel.label);
  // depthwiseConv2d emits [in0 x mult0, in0 x mult1, ..., in1 x mult0, ...].
  return first.flatMap((outer) => second.map((inner) => `${outer} → ${inner}`));
}

/** Map size after the stack, and how many input pixels one output cell sees. */
export interface StackGeometry {
  sizes: number[];
  receptiveField: number;
  finalSize: number;
}

export function stackGeometry(layers: readonly Layer[]): StackGeometry {
  let size = IMAGE_SIZE;
  let field = 1;
  let stride = 1;
  const sizes: number[] = [];

  for (const layer of layers) {
    if (layer.kernels.length === 0) continue;
    size = size - (KERNEL_SIZE - 1);
    field += (KERNEL_SIZE - 1) * stride;
    if (layer.pool !== "none") {
      size = Math.floor(size / 2);
      field += stride;
      stride *= 2;
    }
    sizes.push(size);
  }

  return { sizes, receptiveField: field, finalSize: size };
}

// ── Running the stack ─────────────────────────────────────────────────────

/**
 * Forward pass, as tensors.
 *
 * `tf.conv2d` for layer 1 and `tf.depthwiseConv2d` for layer 2 — these are the
 * real ops, not a hand-rolled loop pretending to be them. `convolveOne` below
 * exists separately for the sliding-window explainer, where the point is to show
 * the nine multiplies, and it is checked against this path in the tests so the
 * explanation cannot drift from the implementation.
 */
function forward(input: tf.Tensor4D, layers: readonly Layer[]): tf.Tensor4D {
  let current = input;

  layers.forEach((layer, index) => {
    if (layer.kernels.length === 0) return;

    const inChannels = current.shape[3];
    let convolved: tf.Tensor4D;

    if (index === 0) {
      // [3, 3, inChannels=1, outChannels=K]
      const filter = tf.tensor4d(
        buildConvFilter(layer.kernels, inChannels),
        [KERNEL_SIZE, KERNEL_SIZE, inChannels, layer.kernels.length],
      );
      convolved = tf.conv2d(current, filter, 1, "valid");
    } else {
      // Depthwise: [3, 3, inChannels, channelMultiplier=K]
      const filter = tf.tensor4d(
        buildConvFilter(layer.kernels, inChannels),
        [KERNEL_SIZE, KERNEL_SIZE, inChannels, layer.kernels.length],
      );
      convolved = tf.depthwiseConv2d(current, filter, 1, "valid");
    }

    let activated: tf.Tensor4D = tf.relu(convolved);

    if (layer.pool === "max") {
      activated = tf.maxPool(activated, 2, 2, "valid");
    } else if (layer.pool === "avg") {
      activated = tf.avgPool(activated, 2, 2, "valid");
    }

    current = activated;
  });

  return current;
}

/**
 * Weights laid out the way TF wants them: [row][col][inChannel][kernel].
 *
 * The same kernel is copied across every input channel, which is what makes the
 * depthwise layer hand-designable — one 3x3 grid applies to all of them.
 */
function buildConvFilter(
  kernels: readonly Kernel[],
  inChannels: number,
): number[][][][] {
  const filter: number[][][][] = [];
  for (let row = 0; row < KERNEL_SIZE; row += 1) {
    const rowValues: number[][][] = [];
    for (let col = 0; col < KERNEL_SIZE; col += 1) {
      const colValues: number[][] = [];
      for (let channel = 0; channel < inChannels; channel += 1) {
        colValues.push(
          kernels.map((kernel) => kernel.weights[row * KERNEL_SIZE + col]!),
        );
      }
      rowValues.push(colValues);
    }
    filter.push(rowValues);
  }
  return filter;
}

function toBatch(samples: readonly Sample[]): tf.Tensor4D {
  const data = new Float32Array(samples.length * PIXELS);
  samples.forEach((sample, index) => {
    data.set(sample.pixels, index * PIXELS);
  });
  return tf.tensor4d(data, [samples.length, IMAGE_SIZE, IMAGE_SIZE, 1]);
}

function toLabels(samples: readonly Sample[]): tf.Tensor2D {
  const data = new Float32Array(samples.length * CLASS_COUNT);
  samples.forEach((sample, index) => {
    data[index * CLASS_COUNT + sample.label] = 1;
  });
  return tf.tensor2d(data, [samples.length, CLASS_COUNT]);
}

/**
 * Global average pooling, which is part of the FIXED pipeline.
 *
 * This is the line that makes the ceiling provable. After it, a one-layer stack
 * has produced exactly one number per kernel — how much of that pattern is in the
 * picture — and no set of kernels can make those numbers differ between two
 * classes built to contain the same amount of everything. Move the pooling and the
 * proof moves with it, which is why the player cannot touch this one.
 */
function globalAverage(maps: tf.Tensor4D): tf.Tensor2D {
  return tf.mean(maps, [1, 2]) as tf.Tensor2D;
}

/** Feature vectors for a set of samples, as plain arrays. */
export function extractFeatures(
  samples: readonly Sample[],
  layers: readonly Layer[],
): number[][] {
  if (filtersUsed(layers) === 0) return samples.map(() => []);

  return tf.tidy(() => {
    const batch = toBatch(samples);
    const pooled = globalAverage(forward(batch, layers));
    return pooled.arraySync() as number[][];
  });
}

/** One feature map, for the gallery. Values are post-ReLU, pre-pooling. */
export interface FeatureMap {
  label: string;
  size: number;
  /** Row-major, length size*size. */
  values: number[];
  max: number;
  mean: number;
}

export function featureMapsFor(
  sample: Sample,
  layers: readonly Layer[],
): FeatureMap[][] {
  const perLayer: FeatureMap[][] = [];
  if (filtersUsed(layers) === 0) return perLayer;

  tf.tidy(() => {
    let current = toBatch([sample]);

    layers.forEach((layer, index) => {
      if (layer.kernels.length === 0) {
        perLayer.push([]);
        return;
      }

      const inChannels = current.shape[3];
      const filter = tf.tensor4d(
        buildConvFilter(layer.kernels, inChannels),
        [KERNEL_SIZE, KERNEL_SIZE, inChannels, layer.kernels.length],
      );
      const convolved =
        index === 0
          ? tf.conv2d(current, filter, 1, "valid")
          : tf.depthwiseConv2d(current, filter, 1, "valid");
      const activated = tf.relu(convolved);

      const size = activated.shape[1];
      const channels = activated.shape[3];
      const flat = activated.arraySync() as number[][][][];
      const labels =
        index === 0
          ? layer.kernels.map((kernel) => kernel.label)
          : channelLabels(layers);

      const maps: FeatureMap[] = [];
      for (let channel = 0; channel < channels; channel += 1) {
        const values: number[] = [];
        let max = 0;
        let total = 0;
        for (let row = 0; row < size; row += 1) {
          for (let col = 0; col < size; col += 1) {
            const value = flat[0]![row]![col]![channel]!;
            values.push(value);
            if (value > max) max = value;
            total += value;
          }
        }
        maps.push({
          label: labels[channel] ?? `channel ${channel + 1}`,
          size,
          values,
          max,
          mean: total / values.length,
        });
      }
      perLayer.push(maps);

      current =
        layer.pool === "max"
          ? tf.maxPool(activated, 2, 2, "valid")
          : layer.pool === "avg"
            ? tf.avgPool(activated, 2, 2, "valid")
            : activated;
    });
  });

  return perLayer;
}

/**
 * One kernel over one 3x3 patch, by hand.
 *
 * Nine multiplies and a sum. This is what `<ImageCanvas>` steps the player
 * through, and the tests check it against `tf.conv2d` on the same patch — an
 * explanation that disagrees with the implementation is worse than no explanation.
 */
export interface WindowResult {
  patch: number[];
  products: number[];
  sum: number;
  activated: number;
}

export function convolveOne(
  sample: Sample,
  kernel: Kernel,
  row: number,
  col: number,
): WindowResult {
  const patch: number[] = [];
  const products: number[] = [];
  let sum = 0;

  for (let dy = 0; dy < KERNEL_SIZE; dy += 1) {
    for (let dx = 0; dx < KERNEL_SIZE; dx += 1) {
      const y = row + dy;
      const x = col + dx;
      const pixel =
        y >= 0 && y < IMAGE_SIZE && x >= 0 && x < IMAGE_SIZE
          ? sample.pixels[y * IMAGE_SIZE + x]!
          : 0;
      const weight = kernel.weights[dy * KERNEL_SIZE + dx]!;
      patch.push(pixel);
      products.push(pixel * weight);
      sum += pixel * weight;
    }
  }

  return { patch, products, sum, activated: Math.max(0, sum) };
}

// ── Cancelling a fit ──────────────────────────────────────────────────────

/**
 * A cooperative stop flag for a training run.
 *
 * Every edit rescores the stack and a Retry throws the whole kitchen away, and in
 * both cases the fit already running is now answering a question nobody is
 * asking. A superseded result was always discarded; the flag stops it spending
 * the GPU first, so the fit that matters is not queued behind the ones that do
 * not. It is read at the start of every batch, which costs nothing — unlike an
 * `onBatchEnd` hook, it never forces the batch's loss off the GPU to be read.
 */
export interface CancelToken {
  cancelled: boolean;
}

/** Thrown by a fit that was cancelled. Callers treat it as "no result", not as an error. */
export class FitCancelledError extends Error {
  constructor() {
    super("The fit was superseded before it finished.");
    this.name = "FitCancelledError";
  }
}

function stopWhenCancelled(model: tf.LayersModel, cancel: CancelToken | undefined) {
  return cancel === undefined
    ? undefined
    : {
        onBatchBegin: () => {
          if (cancel.cancelled) model.stopTraining = true;
        },
      };
}

/**
 * Free a model AND the optimizer it was compiled with.
 *
 * `model.dispose()` only frees an optimizer the model created itself. Every model
 * here is compiled with `tf.train.adam(rate)`, an instance passed in, so the
 * model does not own it and its moment tensors used to outlive every fit — six
 * per score, eight per learn, one WebGL texture each, for every stepper click.
 */
function disposeModel(model: tf.LayersModel): void {
  const optimizer = model.optimizer;
  model.dispose();
  optimizer?.dispose();
}

// ── The fixed classifier ──────────────────────────────────────────────────

export const HEAD_SEED = 1337;
export const HEAD_EPOCHS = 60;
export const HEAD_LEARNING_RATE = 0.08;
export const HEAD_L2 = 1e-3;

/**
 * The head, which the player never gets to change.
 *
 * One dense softmax layer. No hidden units, no depth, no growth: the only way the
 * detection score moves is if the features moved. It is also weak on purpose —
 * a linear classifier cannot compensate for a bad representation, which is exactly
 * the property that makes the score a measurement of the player's filters rather
 * than of the optimiser's patience.
 */
export function buildHead(inputWidth: number): tf.LayersModel {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [inputWidth],
      units: CLASS_COUNT,
      activation: "softmax",
      kernelInitializer: tf.initializers.glorotNormal({ seed: HEAD_SEED }),
      kernelRegularizer: tf.regularizers.l2({ l2: HEAD_L2 }),
    }),
  );
  model.compile({
    optimizer: tf.train.adam(HEAD_LEARNING_RATE),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

export interface ScoreResult {
  /** Validation accuracy. This is the detection score. */
  accuracy: number;
  trainAccuracy: number;
  /** Accuracy per class, so the player can see what is still invisible. */
  perClass: number[];
  /** Predicted-vs-actual counts, validation split. */
  confusion: number[][];
}

function accuracyFrom(
  probabilities: number[][],
  samples: readonly Sample[],
): { accuracy: number; perClass: number[]; confusion: number[][] } {
  const confusion = Array.from({ length: CLASS_COUNT }, () =>
    new Array<number>(CLASS_COUNT).fill(0),
  );

  probabilities.forEach((row, index) => {
    let best = 0;
    for (let candidate = 1; candidate < row.length; candidate += 1) {
      if (row[candidate]! > row[best]!) best = candidate;
    }
    const tally = confusion[samples[index]!.label];
    if (tally !== undefined) tally[best] = (tally[best] ?? 0) + 1;
  });

  const perClass = confusion.map((row, actual) => {
    const total = row.reduce((sum, count) => sum + count, 0);
    return total === 0 ? 0 : row[actual]! / total;
  });
  const correct = confusion.reduce((sum, row, actual) => sum + row[actual]!, 0);

  return { accuracy: correct / probabilities.length, perClass, confusion };
}

/** Fit the fixed head on the player's features and score it. */
export async function scoreStack(
  dataset: Dataset,
  layers: readonly Layer[],
  cancel?: CancelToken,
): Promise<ScoreResult> {
  const width = outputChannels(layers);
  if (filtersUsed(layers) === 0 || width === 0) {
    return {
      accuracy: 0,
      trainAccuracy: 0,
      perClass: new Array<number>(CLASS_COUNT).fill(0),
      confusion: Array.from({ length: CLASS_COUNT }, () =>
        new Array<number>(CLASS_COUNT).fill(0),
      ),
    };
  }

  const trainFeatures = extractFeatures(dataset.train, layers);
  if (cancel?.cancelled) throw new FitCancelledError();
  const validationFeatures = extractFeatures(dataset.validation, layers);

  const model = buildHead(width);
  const xs = tf.tensor2d(trainFeatures);
  const ys = toLabels(dataset.train);

  try {
    await model.fit(xs, ys, {
      epochs: HEAD_EPOCHS,
      batchSize: 64,
      shuffle: false,
      verbose: 0,
      callbacks: stopWhenCancelled(model, cancel),
    });
    if (cancel?.cancelled) throw new FitCancelledError();

    const predict = (features: number[][], samples: readonly Sample[]) =>
      tf.tidy(() => {
        const input = tf.tensor2d(features);
        const output = model.predict(input) as tf.Tensor2D;
        return accuracyFrom(output.arraySync() as number[][], samples);
      });

    const validation = predict(validationFeatures, dataset.validation);
    const train = predict(trainFeatures, dataset.train);

    return {
      accuracy: validation.accuracy,
      trainAccuracy: train.accuracy,
      perClass: validation.perClass,
      confusion: validation.confusion,
    };
  } finally {
    xs.dispose();
    ys.dispose();
    disposeModel(model);
  }
}

/**
 * The same head on raw pixels, as a baseline.
 *
 * Worth having on screen: it is the number that says the model cannot do this at
 * all. Random stripe phase means no fixed pixel template matches, so 576 raw
 * inputs and a linear classifier sit near chance — and every point above chance in
 * this game came from convolution, not from the classifier.
 */
export async function scoreRawPixels(dataset: Dataset): Promise<number> {
  const model = buildHead(PIXELS);
  const xs = tf.tensor2d(
    dataset.train.map((sample) => Array.from(sample.pixels)),
  );
  const ys = toLabels(dataset.train);

  try {
    await model.fit(xs, ys, {
      epochs: HEAD_EPOCHS,
      batchSize: 64,
      shuffle: false,
      verbose: 0,
    });
    return tf.tidy(() => {
      const input = tf.tensor2d(
        dataset.validation.map((sample) => Array.from(sample.pixels)),
      );
      const output = model.predict(input) as tf.Tensor2D;
      return accuracyFrom(output.arraySync() as number[][], dataset.validation)
        .accuracy;
    });
  } finally {
    xs.dispose();
    ys.dispose();
    disposeModel(model);
  }
}

// ── "Let it learn": the same architecture, weights chosen by gradient descent ──

/**
 * The learned reference runs on a subset, and briefly.
 *
 * Training conv weights is orders of magnitude more work than fitting the linear
 * head — measured at 37 seconds for the smallest stack on the full set, against
 * about a second for scoring. It only has to answer one question ("can this shape
 * do it at all?"), and a quarter of the data for a third of the epochs answers it:
 * the two-layer stacks still reach 100% and the one-layer stacks still stall well
 * short of the target, which is the whole content of the comparison.
 */
export const LEARN_EPOCHS = 24;
export const LEARN_SAMPLES = 240;
export const LEARN_LEARNING_RATE = 0.03;

export interface LearnedResult {
  accuracy: number;
  /** The kernels gradient descent settled on, layer by layer. */
  kernels: number[][][];
}

/**
 * Train the same-shaped stack end to end and report what it found.
 *
 * This is the reference run, and it does the same job value iteration does in
 * Agent Academy: it separates "my kernels are bad" from "this architecture cannot
 * do it". If learning the weights of the player's own shape sails past their
 * score, the shape was fine. If it hits the same wall, the shape is the wall — and
 * on this dataset that wall is a provable one.
 *
 * It is also the honest answer to the question the whole game raises. You can
 * hand-draw an edge detector. Nobody hand-draws layer four.
 */
export async function learnStack(
  dataset: Dataset,
  layers: readonly Layer[],
  cancel?: CancelToken,
): Promise<LearnedResult> {
  const shapes = layers
    .filter((layer) => layer.kernels.length > 0)
    .map((layer) => ({ count: layer.kernels.length, pool: layer.pool }));

  if (shapes.length === 0) return { accuracy: 0, kernels: [] };

  const model = tf.sequential();
  shapes.forEach((shape, index) => {
    const common = {
      kernelSize: KERNEL_SIZE,
      padding: "valid" as const,
      activation: "relu" as const,
      useBias: false,
    };
    if (index === 0) {
      model.add(
        tf.layers.conv2d({
          ...common,
          inputShape: [IMAGE_SIZE, IMAGE_SIZE, 1],
          filters: shape.count,
          kernelInitializer: tf.initializers.glorotNormal({
            seed: HEAD_SEED + index,
          }),
        }),
      );
    } else {
      model.add(
        tf.layers.depthwiseConv2d({
          ...common,
          depthMultiplier: shape.count,
          depthwiseInitializer: tf.initializers.glorotNormal({
            seed: HEAD_SEED + index,
          }),
        }),
      );
    }
    if (shape.pool === "max") {
      model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
    } else if (shape.pool === "avg") {
      model.add(tf.layers.averagePooling2d({ poolSize: 2, strides: 2 }));
    }
  });
  model.add(tf.layers.globalAveragePooling2d({}));
  model.add(
    tf.layers.dense({
      units: CLASS_COUNT,
      activation: "softmax",
      kernelInitializer: tf.initializers.glorotNormal({ seed: HEAD_SEED }),
    }),
  );
  model.compile({
    optimizer: tf.train.adam(LEARN_LEARNING_RATE),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  const subset = dataset.train.slice(0, LEARN_SAMPLES);
  const xs = toBatch(subset);
  const ys = toLabels(subset);

  try {
    await model.fit(xs, ys, {
      epochs: LEARN_EPOCHS,
      batchSize: 32,
      shuffle: false,
      verbose: 0,
      callbacks: stopWhenCancelled(model, cancel),
    });
    if (cancel?.cancelled) throw new FitCancelledError();

    const accuracy = tf.tidy(() => {
      const input = toBatch(dataset.validation);
      const output = model.predict(input) as tf.Tensor2D;
      return accuracyFrom(output.arraySync() as number[][], dataset.validation)
        .accuracy;
    });

    // Pull the learned kernels back out, normalised so they can be shown on the
    // same colour scale as the hand-designed ones.
    const kernels: number[][][] = [];
    for (const layer of model.layers) {
      const weights = layer.getWeights();
      if (weights.length === 0) continue;
      const shape = weights[0]!.shape;
      if (shape.length !== 4) continue;
      const values = weights[0]!.arraySync() as number[][][][];
      const count = shape[3]!;
      const perLayer: number[][] = [];
      for (let index = 0; index < count; index += 1) {
        const flat: number[] = [];
        for (let row = 0; row < KERNEL_SIZE; row += 1) {
          for (let col = 0; col < KERNEL_SIZE; col += 1) {
            flat.push(values[row]![col]![0]![index]!);
          }
        }
        const peak = Math.max(...flat.map((value) => Math.abs(value)), 1e-9);
        perLayer.push(flat.map((value) => (value / peak) * WEIGHT_MAX));
      }
      kernels.push(perLayer);
    }

    return { accuracy, kernels };
  } finally {
    xs.dispose();
    ys.dispose();
    disposeModel(model);
  }
}

// ── Judging the kitchen ───────────────────────────────────────────────────

/** Detection score needed to pass. */
export const TARGET_ACCURACY = 0.9;
/**
 * Mean activation below this and the filter is doing nothing.
 *
 * Measured, with a wide margin either side: the live presets sit between 0.80 and
 * 4.17, while "Sharpen" manages 0.0002 and "Centre spot" and "Blank" produce a flat
 * zero. Anything in between would be an arbitrary line; this is not.
 */
export const DEAD_THRESHOLD = 0.01;
/**
 * Feature-map correlation above this and two filters are the same filter.
 *
 * Also measured. Identical kernels correlate at 1.000; a vertical edge detector
 * against a diagonal one lands far below. Note the check is deliberately on signed
 * correlation, not its absolute value: a kernel and its own negation come out
 * anti-correlated before the ReLU and detect OPPOSITE edge polarities after it, so
 * they are two useful filters, not one duplicated. Measured at 0 duplicates for
 * that pair, which is the correct answer.
 */
export const DUPLICATE_THRESHOLD = 0.95;
/**
 * How much of one arrangement dish has to be mistaken for the other before we
 * call it the single-layer ceiling rather than sloppy kernels.
 */
export const CEILING_CONFUSION = 0.5;

/**
 * The two dishes a single layer provably cannot separate.
 *
 * They contain the same amount of vertical and horizontal texture as each other
 * and differ only in where each one sits, so the one-number-per-filter summary a
 * one-layer stack hands to the classifier is identical for both.
 */
export const ARRANGEMENT_CLASSES: readonly ClassId[] = [2, 3] as const;

/**
 * The measured single-layer ceiling — an upper bound, not a target.
 *
 * The symmetry argument alone predicts 75%, and hand-designed kernels land there
 * exactly. Gradient descent given a free hand with a one-layer stack does a little
 * better — measured between 73% and 83% across kernel counts and epoch budgets —
 * because the seam where the two textures meet IS visible to a 3x3 kernel, and two
 * or three rows out of twenty-four is a small but real cue. Worth stating plainly
 * rather than rounding away: the ceiling is a consequence of the global pooling,
 * and the leak past 75% is a consequence of the seam. Nothing single-layer has been
 * measured above this bound, and the target is 90%.
 */
export const ONE_LAYER_CEILING = 0.85;

/**
 * How far past guessing a score has to be before the copy may say something in
 * the stack is being detected.
 *
 * The validation split is VALIDATION_SIZE (400) plates, so a stack at chance
 * wobbles by about two points; ten is five times that. The gap it separates is
 * not a close call either: measured, a blurred layer 1 under only averaging
 * filters scores 23-25%, and under a sign-changing layer-2 filter 49-51%.
 */
export const CHANCE_MARGIN = 0.1;
export const clearlyAboveChance = (accuracy: number): boolean =>
  accuracy >= CHANCE_RATE + CHANCE_MARGIN;

/**
 * The filters above layer 1 that have a sign change, once per label: the ones
 * that can detect something in whatever layer 1 hands them. A pass-through or a
 * blur up there only passes that on.
 */
export function detectorsAboveLayerOne(layers: readonly Layer[]): string[] {
  return [
    ...new Set(
      layers
        .slice(1)
        .flatMap((layer) => layer.kernels)
        .filter((kernel) => !isSingleSigned(kernel))
        .map((kernel) => kernel.label),
    ),
  ];
}

export interface FilterHealth {
  layerIndex: number;
  kernelIndex: number;
  label: string;
  meanActivation: number;
  /** Sum of the weights. Negative is the usual reason a filter is dead. */
  weightSum: number;
  dead: boolean;
  blur: boolean;
  blank: boolean;
}

/** Pearson correlation between two equal-length vectors. */
export function correlation(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length);
  if (n === 0) return 0;
  let sumA = 0;
  let sumB = 0;
  for (let index = 0; index < n; index += 1) {
    sumA += a[index]!;
    sumB += b[index]!;
  }
  const meanA = sumA / n;
  const meanB = sumB / n;

  let covariance = 0;
  let varianceA = 0;
  let varianceB = 0;
  for (let index = 0; index < n; index += 1) {
    const da = a[index]! - meanA;
    const db = b[index]! - meanB;
    covariance += da * db;
    varianceA += da * da;
    varianceB += db * db;
  }
  if (varianceA <= 1e-12 || varianceB <= 1e-12) return 0;
  return covariance / Math.sqrt(varianceA * varianceB);
}

export interface DuplicatePair {
  layerIndex: number;
  a: number;
  b: number;
  labelA: string;
  labelB: string;
  correlation: number;
}

/**
 * Which filters are dead, which are blurs, and which pairs are the same filter.
 *
 * Measured on real activations over a sample of the training set rather than
 * inferred from the weights. Two different-looking kernels can respond
 * identically to this dataset, and two similar-looking ones can differ; only the
 * feature maps know.
 */
export function inspectFilters(
  dataset: Dataset,
  layers: readonly Layer[],
  /**
   * 24 plates is a mean over roughly 11,600 activations per filter, which is far
   * more than this decision needs: the measured gap between a live filter and a
   * dead one is 0.44 against 0.0000, and duplicate correlations come out at 1.000
   * against well under 0.5. Neither is a marginal call, and a bigger sample only
   * costs forward passes on every edit.
   */
  sampleCount = 24,
): { health: FilterHealth[]; duplicates: DuplicatePair[] } {
  const health: FilterHealth[] = [];
  const duplicates: DuplicatePair[] = [];
  if (filtersUsed(layers) === 0) return { health, duplicates };

  const samples = dataset.train.slice(0, sampleCount);
  // One forward pass per sample, reused for every layer.
  const perSample = samples.map((sample) => featureMapsFor(sample, layers));

  layers.forEach((layer, layerIndex) => {
    if (layer.kernels.length === 0) return;

    // Flattened activations per kernel, concatenated across the sample.
    const responses: number[][] = layer.kernels.map(() => []);

    for (const maps of perSample) {
      const layerMaps = maps[layerIndex] ?? [];
      layer.kernels.forEach((_, kernelIndex) => {
        if (layerIndex === 0) {
          responses[kernelIndex]!.push(...(layerMaps[kernelIndex]?.values ?? []));
          return;
        }
        // Layer 2 is depthwise: tf emits output channel `c * K2 + k` for input
        // channel c and kernel k, so one kernel's own responses are every K2-th
        // channel. Comparing without that stride would compare a kernel against
        // itself on a different input channel and call everything a duplicate.
        const step = layer.kernels.length;
        for (
          let channel = kernelIndex;
          channel < layerMaps.length;
          channel += step
        ) {
          responses[kernelIndex]!.push(...layerMaps[channel]!.values);
        }
      });
    }

    layer.kernels.forEach((kernel, kernelIndex) => {
      const values = responses[kernelIndex] ?? [];
      const mean =
        values.length === 0
          ? 0
          : values.reduce((total, value) => total + value, 0) / values.length;
      health.push({
        layerIndex,
        kernelIndex,
        label: kernel.label,
        meanActivation: mean,
        weightSum: kernelSum(kernel),
        dead: mean < DEAD_THRESHOLD,
        // Only layer 1 earns the "blur" diagnosis. The argument for it is that an
        // average of nine PIXELS measures brightness, and brightness was
        // randomised — that argument does not hold one layer up, where the input
        // is an activation map and averaging it is a legitimate way to pass
        // energy on. Flagging a layer-2 averaging kernel would be wrong.
        blur: layerIndex === 0 && isSingleSigned(kernel) && !isBlank(kernel),
        blank: isBlank(kernel),
      });
    });

    for (let a = 0; a < layer.kernels.length; a += 1) {
      for (let b = a + 1; b < layer.kernels.length; b += 1) {
        const value = correlation(responses[a] ?? [], responses[b] ?? []);
        if (value >= DUPLICATE_THRESHOLD) {
          duplicates.push({
            layerIndex,
            a,
            b,
            labelA: layer.kernels[a]!.label,
            labelB: layer.kernels[b]!.label,
            correlation: value,
          });
        }
      }
    }
  });

  return { health, duplicates };
}

export type Outcome =
  | "served"
  | "dead-filters"
  | "blur-only"
  | "duplicate-filters"
  | "one-layer-ceiling"
  | "needs-work"
  | "empty";

export interface Evaluation {
  outcome: Outcome;
  score: ScoreResult;
  /** Detection score the same shape reaches with LEARNED kernels. */
  learned: LearnedResult | null;
  health: FilterHealth[];
  duplicates: DuplicatePair[];
  filters: number;
  /**
   * The normalised score progression sees. Stars are NOT computed here: the
   * engine owns that rule (★2 = points ≥ HIGH_SCORE_THRESHOLD, ★3 = that plus a
   * code-lane clear), and a second rubric in the game is how the tooltip and the
   * awarded stars came to disagree.
   */
  points: number;
  failure: NamedFailure | null;
}

/**
 * What a served stack scores (spec: "the fewest / cleanest filters").
 *
 * Full marks for the leanest stack that can serve — one edge detector under a
 * pass-through and an asymmetric filter, which the tests prove passes — and a
 * fixed deduction for every filter past it and for every duplicated pair, which
 * is a filter paid for twice. The old blend (0.55 + thrift + cleanliness) cleared
 * the second-star threshold on every ordinary win, six filters included, so the
 * star that says "thrift" measured nothing. With these deductions the threshold
 * lands exactly on "four filters or fewer, none duplicated" (`THRIFTY_FILTERS`),
 * a pass never drops below `SERVED_FLOOR`, and no combination sits near the
 * threshold for floating point to round across.
 */
export const LEAN_FILTERS = 3;
export const THRIFTY_FILTERS = LEAN_FILTERS + 1;
export const FILTER_PENALTY = 0.12;
export const DUPLICATE_PENALTY = 0.25;
/** Above every failing score, which are capped at 0.5. */
export const SERVED_FLOOR = 0.55;

export function servedPoints(filters: number, duplicates: number): number {
  const extra = Math.max(0, filters - LEAN_FILTERS);
  return clamp(
    1 - FILTER_PENALTY * extra - DUPLICATE_PENALTY * duplicates,
    SERVED_FLOOR,
    1,
  );
}

const percent = (value: number) => `${Math.round(value * 100)}%`;

/**
 * Judge the kitchen.
 *
 * Order matters. Dead and blur filters are checked before the score, because a
 * filter that never fires is a fact about the filter and it stays a fact whether
 * the score happens to be passable — a player who wins with three live filters and
 * one corpse should be told about the corpse.
 *
 * The ceiling verdict comes last and needs the reference run: "one layer cannot do
 * this" is only worth saying when learning that same one layer's weights also
 * fails, and it is the difference between "design better kernels" and "you need
 * another layer", which are opposite fixes.
 */
export function evaluateKitchen({
  layers,
  score,
  health,
  duplicates,
  learned,
}: {
  layers: readonly Layer[];
  score: ScoreResult;
  health: FilterHealth[];
  duplicates: DuplicatePair[];
  learned: LearnedResult | null;
}): Evaluation {
  const filters = filtersUsed(layers);
  const base = {
    score,
    learned,
    health,
    duplicates,
    filters,
  };

  if (filters === 0) {
    return {
      ...base,
      outcome: "empty",
      points: 0,
      failure: null,
    };
  }

  const dead = health.filter((filter) => filter.dead);
  const blurs = health.filter((filter) => filter.blur);
  const live = health.length - dead.length;

  // Points reward the spec's "fewest / cleanest": a pass first, then thrift. A
  // failing stack scores in proportion to how close it got, below every pass.
  const points =
    score.accuracy >= TARGET_ACCURACY
      ? servedPoints(filters, duplicates.length)
      : clamp((score.accuracy / TARGET_ACCURACY) * 0.5, 0, 0.5);

  if (dead.length > 0) {
    const worst = dead[0]!;
    return {
      ...base,
      outcome: "dead-filters",
      points: Math.min(points, 0.45),
      failure: {
        name: "Dead filters",
        detail: `${
          dead.length === 1 ? "One filter is" : `${dead.length} filters are`
        } outputting nothing. "${worst.label}" in layer ${
          worst.layerIndex + 1
        } has a mean activation of ${worst.meanActivation.toFixed(
          4,
        )} across the training images — after the ReLU, essentially every cell of its feature map is zero. ${
          worst.blank
            ? `Its weights are all zero, so there is nothing for it to respond to.`
            : worst.weightSum < 0
              ? `Its weights sum to ${worst.weightSum}. Pixels here are never negative, so a kernel with a negative total produces a negative answer on almost every patch it sees — and the ReLU turns every one of those into a zero. It is not detecting something rare, it is switched off. Note that this can happen to a perfectly sensible kernel: a centre-against-its-ring detector is looking for small bright dots, and there are no dots in this kitchen.`
              : `Whatever pattern it is looking for does not occur in these images, so its answer never rises above zero.`
        } The head is being handed a column of zeros: ${live} of your ${
          health.length
        } filters ${
          live === 1 ? "is" : "are"
        } actually contributing. This is the most expensive kind of mistake in a real network, because the loss curve looks perfectly healthy while a slice of the model does nothing at all.`,
      },
    };
  }

  /**
   * The blur verdict is about LAYER 1, and it stays the verdict with a second
   * layer on top — layer 1 is still the thing to fix. What changes is the claim
   * the copy may make, and it turns on whether anything above layer 1 has a sign
   * change. If nothing does — one layer, or a pass-through and a blur on top —
   * "every filter you have is an average" is literally true and the score sits at
   * chance. With a layer-2 edge detector it is false: that filter DOES detect
   * something, measured at about half the plates. What it detects it in is a
   * smoothed photograph, and smoothing then detecting is, near enough, one filter
   * — exactly so when layer 1 pools by averaging, since a non-negative kernel
   * passes the ReLU unchanged — so the stack has a second layer and none of what
   * a second layer is for.
   */
  const layerOne = health.filter((filter) => filter.layerIndex === 0);
  const deep = health.some((filter) => filter.layerIndex > 0);
  const detectorsAbove = detectorsAboveLayerOne(layers);
  if (blurs.length > 0 && blurs.length === layerOne.length) {
    const first = blurs[0]!;
    const one = detectorsAbove.length === 1;
    // "Agrees" only if the score really is at chance; the copy does not get to
    // assume it.
    const verdictOnScore = `The detection score ${
      clearlyAboveChance(score.accuracy) ? "is" : "agrees:"
    } ${percent(score.accuracy)} against ${percent(CHANCE_RATE)} for guessing.`;
    const detail =
      detectorsAbove.length === 0
        ? `Every filter you have${
            deep ? ", in both layers," : ""
          } has weights of a single sign, which means every one of them computes a weighted AVERAGE ${
            deep
              ? `and nothing else: layer 1 averages nine pixels, and layer 2 only averages those averages again. "${first.label}" in layer 1 is one of them.`
              : `of nine pixels and nothing else. "${first.label}" is one of them.`
          } An average measures brightness — and every dish in this kitchen was plated with randomised brightness and contrast on purpose, so that number carries no information about which dish it is. ${verdictOnScore} A detector has to measure a DIFFERENCE, which means at least one positive weight and at least one negative weight. Change a sign${
            deep ? " in layer 1" : ""
          } and watch the feature map stop looking like the photograph.`
        : `Every filter in layer 1 has weights of a single sign, which means each one computes a weighted AVERAGE of nine pixels and nothing else. "${
            first.label
          }" is one of them. On its own an average measures brightness, which every plate here randomises — and above it, it means layer 2 is not reading a map of detected patterns at all, only a softened copy of the photograph. Whatever ${detectorsAbove
            .map((label) => `"${label}"`)
            .join(" and ")} ${one ? "finds" : "find"} in that copy, ${
            one ? "it" : "they"
          } could have found in the photograph directly: smoothing and then detecting is, near enough, just detecting. So your second layer is doing a first layer's job, and the stack gets none of what depth is for — ${percent(
            score.accuracy,
          )} against the ${percent(
            TARGET_ACCURACY,
          )} needed. Layer 1 has to measure a DIFFERENCE, which means at least one positive weight and at least one negative weight. Change a sign there and layer 2 finally has something to arrange.`;
    return {
      ...base,
      outcome: "blur-only",
      points: Math.min(points, 0.45),
      failure: { name: "You built a blur", detail },
    };
  }

  if (duplicates.length > 0 && score.accuracy < TARGET_ACCURACY) {
    const pair = duplicates[0]!;
    return {
      ...base,
      outcome: "duplicate-filters",
      points: Math.min(points, 0.5),
      failure: {
        name: "Duplicate filters",
        detail: `"${pair.labelA}" and "${
          pair.labelB
        }" in layer ${
          pair.layerIndex + 1
        } produce feature maps that correlate at ${pair.correlation.toFixed(
          3,
        )}. They are not two filters, they are one filter costed twice: deleting either would change the detection score by almost nothing, and you are ${filters} filters into a budget of ${FILTER_BUDGET}. Two kernels can look quite different on the grid and still respond to the same thing — which is why this is measured on the activations rather than on the weights. Spend the slot on a pattern you are not detecting yet.`,
      },
    };
  }

  if (score.accuracy >= TARGET_ACCURACY) {
    return {
      ...base,
      outcome: "served",
      points,
      failure: null,
    };
  }

  // Is this a design problem or an architecture problem?
  //
  // The evidence is in the confusion matrix, not in a training run: if a
  // single-layer stack is mistaking one arrangement dish for the OTHER
  // arrangement dish, it has hit the ceiling the global pooling imposes, and no
  // choice of kernels moves it. Deciding this from the confusion rather than from
  // `learned` keeps every edit cheap to score — the learned run costs seconds and
  // exists so the player can check this claim, not so the game can make it.
  const singleLayer = layers.filter((layer) => layer.kernels.length > 0).length < 2;
  const [first, second] = ARRANGEMENT_CLASSES as unknown as [ClassId, ClassId];

  // Of the mistakes made on the two arrangement dishes, how many are confusions
  // with each OTHER? Not "how many were wrong" — a stack at the ceiling reads one
  // of the pair perfectly and calls the other one by the same name, so counting
  // raw errors misses it. What identifies the ceiling is that the errors all land
  // on the sibling dish: the stack can see that it is an arrangement, and cannot
  // see which one.
  const crossed =
    (score.confusion[first]?.[second] ?? 0) +
    (score.confusion[second]?.[first] ?? 0);
  const mistakes = ARRANGEMENT_CLASSES.reduce<number>((total, actual) => {
    const row = score.confusion[actual] ?? [];
    return (
      total +
      row.reduce<number>(
        (sum, count, predicted) => (predicted === actual ? sum : sum + count),
        0,
      )
    );
  }, 0);
  const arrangementTotal = ARRANGEMENT_CLASSES.reduce<number>(
    (total, actual) =>
      total +
      (score.confusion[actual]?.reduce<number>((sum, c) => sum + c, 0) ?? 0),
    0,
  );
  const confusedWithEachOther =
    mistakes > 0 && crossed / mistakes >= CEILING_CONFUSION;

  if (singleLayer && confusedWithEachOther) {
    return {
      ...base,
      outcome: "one-layer-ceiling",
      points,
      failure: {
        name: "One layer is not enough",
        detail: `${percent(
          score.accuracy,
        )} is not bad kernel design — it is the ceiling of a single layer on this menu, and you can see it in the per-class scores: "${
          DISHES[0]!.label
        }" and "${DISHES[1]!.label}" are being read at ${percent(
          score.perClass[0] ?? 0,
        )} and ${percent(
          score.perClass[1] ?? 0,
        )}, while "${DISHES[2]!.label}" and "${
          DISHES[3]!.label
        }" sit at ${percent(score.perClass[2] ?? 0)} and ${percent(
          score.perClass[3] ?? 0,
        )} — and the mistakes land on each OTHER: ${Math.round(
          crossed,
        )} of the ${arrangementTotal} plates of those two came back wearing the sibling's name. There is a reason, and it is not your kernels. Those two dishes contain exactly the same amount of vertical and horizontal texture; they differ only in WHERE each one sits. Your stack ends in a global average, so a single layer hands the classifier one number per filter — how much of that pattern is in the picture — and for those two dishes those numbers come out almost exactly the same. The only difference left is the two or three rows where the textures meet, and no choice of 3x3 kernel gets much out of a sliver that thin. ${
          learned === null
            ? `Do not take my word for it: press "Let it learn" and gradient descent will pick kernels for this exact shape instead of you. It has never been measured past ${percent(
                ONE_LAYER_CEILING,
              )} with one layer, against the ${percent(
                TARGET_ACCURACY,
              )} you need.`
            : `You already checked: gradient descent picking its own kernels for this shape reached ${percent(
                learned.accuracy,
              )}. ${
                learned.accuracy < TARGET_ACCURACY
                  ? `Same wall.`
                  : `That run got past it, which no single-layer run behind this verdict ever did — so read the wall as a very steep slope, and the next sentence as the way over it.`
              }`
        } A second layer sees the first layer's map, so a filter there can respond to vertical texture ABOVE horizontal texture — turning a position into a quantity before the average throws the position away. That is what "hierarchical features" means, and this is the smallest example of it that cannot be faked.`,
      },
    };
  }

  return {
    ...base,
    outcome: "needs-work",
    points,
    failure: {
      name: "Not detected yet",
      detail: `${percent(score.accuracy)} against the ${percent(
        TARGET_ACCURACY,
      )} needed. Nothing is structurally wrong: your filters are live, none of them duplicate each other${
        learned === null
          ? ""
          : `, and the same shape with learned kernels reaches ${percent(
              learned.accuracy,
            )}${
              singleLayer
                ? ""
                : ` (its layer 2 learns a separate grid for each layer-1 map, where yours shares one, so read that as a little generous)`
            } — ${
              learned.accuracy >= TARGET_ACCURACY
                ? `so the shape of your stack can do it and the weights are what is left`
                : `so even with the weights chosen for you, this shape comes up short too`
            }`
      }. Read the per-class scores rather than the total: ${DISHES.map(
        (dish, index) => `${dish.short} ${percent(score.perClass[index] ?? 0)}`,
      ).join(", ")}. The dish with the lowest number is the pattern none of your filters responds to yet.`,
    },
  };
}

// ── Reveal the math ───────────────────────────────────────────────────────

/**
 * Rows first, exactly as MATH_CODE indexes `image[y + i][x + j] * kernel[i][j]`:
 * y is the row, x the column, and K(i, j) is the kernel's row i, column j. Written
 * the other way round the equation applies K transposed, which turns the
 * "Vertical edge" preset into a horizontal one. Both pooling rules are shown
 * because the kitchen defaults to the average, not the max.
 */
export const MATH_EQUATION = String.raw`(I * K)(y,x) = \sum_{i=0}^{2}\sum_{j=0}^{2} I(y+i,\; x+j)\, K(i,j)
\\[1em]
A(y,x) = \max\bigl(0,\; (I * K)(y,x)\bigr)
\\[1em]
P_{\max}(y,x) = \max_{\substack{0 \le i < 2 \\ 0 \le j < 2}} A(2y+i,\; 2x+j)
\qquad
P_{\text{avg}}(y,x) = \tfrac{1}{4} \sum_{i=0}^{1}\sum_{j=0}^{1} A(2y+i,\; 2x+j)`;

export const MATH_CODE = `// The whole operation. Nine multiplies, one sum, per output cell.
for (let y = 0; y < height - 2; y++) {
  for (let x = 0; x < width - 2; x++) {
    let sum = 0;
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        sum += image[y + i][x + j] * kernel[i][j];

    map[y][x] = Math.max(0, sum);   // ReLU: negative means "not this pattern"
  }
}

// One kernel, reused at every position. That is the whole trick: a
// 3x3 detector is 9 numbers whether the image is 24 pixels or 24 million,
// and it finds its pattern wherever the pattern happens to be.`;

export const MATH_NOTES = `Convolution is a sliding pattern-matcher, and the sum above is the match score. (Rows first, as in the code: y is the row, x the column, and K(i, j) sits in the kernel's row i, column j.) Where the patch under the kernel looks like the kernel, positive weights land on bright pixels and negative weights land on dark ones, and the total is large. Where it looks like the kernel's opposite, the total is large and negative — which the ReLU then throws away, because "strongly not this pattern" and "not this pattern" are the same answer as far as the next layer is concerned.

That is why a kernel with weights all of one sign cannot detect anything. It has no opposite. Whatever patch you slide it over, the answer is some multiple of the local brightness, and this kitchen randomises brightness on every plate.

Pooling is doing two jobs at once. The obvious one is size: 2x2 pooling quarters the map. The one that matters more is invariance — the stripes in these images sit at a random phase, so a filter's response wobbles as the pattern shifts under it, and a pooled response stops caring exactly where the match happened. Max pooling asks "is this pattern present anywhere in this block", average pooling asks "how much of it is in this block". Those are different questions and on this menu they get different scores.

The stack ends in a global average, one number per filter. That is a real design (it is how modern classifiers finish) and it has a consequence worth sitting with: a single convolution layer can only ever tell the classifier HOW MUCH of each pattern is in the picture, never where. Two of the four dishes here contain identical amounts of vertical and horizontal texture and differ only in arrangement, so to a one-layer stack they look almost exactly alike. The only thing left to tell them by is the two or three rows where the textures meet, and that sliver is not enough: gradient descent, given a free hand with one layer, has never been measured past ${Math.round(
  ONE_LAYER_CEILING * 100,
)}% here. Not short of training — short of any way to see where things are. A second layer looks at the first layer's map, so a filter in it can respond to "vertical texture above horizontal texture", converting a position into a quantity before the average destroys the position. Edges, then textures, then arrangement. That is the whole idea of depth, and it is the reason a 3x3 kernel at layer four can be sensitive to something the size of a face.

The last thing is the honest one. You can draw an edge detector by hand; the presets in this kitchen are the classic ones and they work. Nobody hand-draws layer four. A real conv layer mixes its input channels, so its kernel is 3x3xC numbers and there is no grid to draw it on — and the useful values in it look like nothing in particular. "Let it learn" trains a stack of the same shape and shows you the kernels gradient descent picks instead — its layer 2 learns a separate grid for each layer-1 map, a little more freedom than your shared one. Sometimes it rediscovers your edge detectors, which is a nice moment. Sometimes it does not, and that is a more useful one.`;
