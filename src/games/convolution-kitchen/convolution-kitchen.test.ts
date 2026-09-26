import * as tf from "@tensorflow/tfjs";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import {
  EMPTY_PROGRESSION,
  HIGH_SCORE_THRESHOLD,
  createMemoryAdapter,
  starsFor,
  useProgression,
} from "@/engine/progression";
import {
  ARRANGEMENT_CLASSES,
  CHANCE_RATE,
  CLASS_COUNT,
  DEAD_THRESHOLD,
  DISHES,
  DUPLICATE_THRESHOLD,
  FILTER_BUDGET,
  FitCancelledError,
  IMAGE_SIZE,
  KERNEL_CELLS,
  KERNEL_PRESETS,
  MAX_KERNELS_PER_LAYER,
  MATH_CODE,
  MATH_EQUATION,
  MATH_NOTES,
  MAX_LAYERS,
  ONE_LAYER_CEILING,
  PIXELS,
  TARGET_ACCURACY,
  THRIFTY_FILTERS,
  WEIGHT_MAX,
  WEIGHT_MIN,
  channelLabels,
  clearlyAboveChance,
  convolveOne,
  correlation,
  describeKernel,
  detectorsAboveLayerOne,
  evaluateKitchen,
  extractFeatures,
  featureMapsFor,
  filtersUsed,
  generateDataset,
  inspectFilters,
  isBlank,
  isSingleSigned,
  kernelSum,
  learnStack,
  makeKernel,
  makeLayer,
  outputChannels,
  scoreRawPixels,
  scoreStack,
  servedPoints,
  shapeKey,
  stackGeometry,
  type Layer,
  type PoolType,
  type ScoreResult,
} from "./ml";
import { SLUG, createCodeApi, currentLearned, useKitchenStore } from "./store";
import { whyCardFor, windowCaption } from "./why-cards";

/**
 * Every constant here was measured by a throwaway probe that swept kernel sets,
 * pool types, layer counts and learned-versus-hand-designed weights. The probe is
 * gone; what it found is asserted below so the tuning cannot rot silently.
 *
 * Two assertions carry the whole game and are worth reading first:
 *
 *   "a single layer cannot pass, whatever the kernels" — the ceiling, checked with
 *   hand-designed kernels AND with kernels chosen by gradient descent.
 *
 *   "the same classifier on raw pixels sits near chance" — the premise. If this
 *   ever rises, the detection score stops being a measurement of convolution.
 */

beforeAll(async () => {
  await tf.ready();
}, 60000);

const dataset = generateDataset(4711);

const layer = (pool: PoolType, ...presets: string[]): Layer =>
  makeLayer(
    presets.map((preset) => makeKernel(preset)),
    pool,
  );

/** The stack the game is designed to be solved with: 3 filters, two layers. */
const WINNER: Layer[] = [
  layer("avg", "Vertical edge"),
  layer("avg", "Pass through", "Horizontal edge"),
];

/**
 * Scoring memoised by stack shape.
 *
 * Fitting the head takes a second or two and several tests ask about the same
 * stacks — the verdict tests and the why-card tests in particular re-score the same
 * four configurations. Keying on the kernels and pooling makes those free without
 * weakening anything: `scoreStack` is deterministic given a stack and a seeded
 * dataset, which is itself asserted below.
 */
const scoreCache = new Map<string, Promise<ScoreResult>>();

function keyOf(layers: readonly Layer[]): string {
  return layers
    .map((l) => `${l.pool}:${l.kernels.map((k) => k.weights.join("")).join("|")}`)
    .join("/");
}

function scored(layers: readonly Layer[]): Promise<ScoreResult> {
  const key = keyOf(layers);
  let pending = scoreCache.get(key);
  if (pending === undefined) {
    pending = scoreStack(dataset, layers);
    scoreCache.set(key, pending);
  }
  return pending;
}

describe("the dishes", () => {
  it("are four, on a 24 by 24 plate", () => {
    expect(IMAGE_SIZE).toBe(24);
    expect(PIXELS).toBe(576);
    expect(CLASS_COUNT).toBe(4);
    expect(CHANCE_RATE).toBe(0.25);
    expect(dataset.train).toHaveLength(600);
    expect(dataset.validation).toHaveLength(400);
  });

  it("are balanced across both splits", () => {
    for (const split of [dataset.train, dataset.validation]) {
      const counts = new Array<number>(CLASS_COUNT).fill(0);
      for (const sample of split) {
        counts[sample.label] = (counts[sample.label] ?? 0) + 1;
      }
      for (const count of counts) {
        expect(count).toBe(split.length / CLASS_COUNT);
      }
    }
  });

  it("hide the answer from mean brightness", () => {
    // A filter that only averages pixels has to be provably useless, which means
    // no class may be distinguishable by its overall brightness.
    const means = DISHES.map((dish) => {
      const members = dataset.train.filter((s) => s.label === dish.id);
      const perImage = members.map(
        (s) => s.pixels.reduce((a, b) => a + b, 0) / s.pixels.length,
      );
      return perImage.reduce((a, b) => a + b, 0) / perImage.length;
    });
    const spread = Math.max(...means) - Math.min(...means);
    // Measured at 0.008 against a per-image standard deviation around 0.06.
    expect(spread).toBeLessThan(0.02);
  });

  it("keep every pixel in range", () => {
    for (const sample of dataset.train.slice(0, 40)) {
      for (const pixel of sample.pixels) {
        expect(pixel).toBeGreaterThanOrEqual(0);
        expect(pixel).toBeLessThanOrEqual(1);
      }
    }
  });

  it("split the two arrangement dishes at a randomised row", () => {
    const uniform = dataset.train.filter((s) => s.label === 0 || s.label === 1);
    expect(uniform.every((s) => s.splitRow === null)).toBe(true);

    const arranged = dataset.train.filter((s) => s.label === 2 || s.label === 3);
    const rows = new Set(arranged.map((s) => s.splitRow));
    // A fixed seam would be a landmark a single 3x3 kernel could exploit.
    expect(rows.size).toBeGreaterThan(1);
    expect(arranged.every((s) => s.splitRow !== null)).toBe(true);
  });

  it("is reproducible from its seed", () => {
    const again = generateDataset(4711);
    expect(again.train[0]!.label).toBe(dataset.train[0]!.label);
    expect(Array.from(again.train[0]!.pixels)).toEqual(
      Array.from(dataset.train[0]!.pixels),
    );
  });

  it("names the two dishes a single layer cannot separate", () => {
    expect([...ARRANGEMENT_CLASSES]).toEqual([2, 3]);
  });
});

describe("kernels", () => {
  it("are 3 by 3 with integer weights in range", () => {
    expect(KERNEL_CELLS).toBe(9);
    for (const preset of KERNEL_PRESETS) {
      expect(preset.weights).toHaveLength(KERNEL_CELLS);
      for (const weight of preset.weights) {
        expect(Number.isInteger(weight)).toBe(true);
        expect(weight).toBeGreaterThanOrEqual(WEIGHT_MIN);
        expect(weight).toBeLessThanOrEqual(WEIGHT_MAX);
      }
    }
  });

  it("recognise a preset by its weights, and call anything else Custom", () => {
    expect(describeKernel(makeKernel("Vertical edge"))).toBe("Vertical edge");
    const custom = makeKernel("Vertical edge");
    custom.weights = [...custom.weights];
    custom.weights[4] = 2;
    expect(describeKernel(custom)).toBe("Custom");
  });

  it("identify the kernels that can only measure brightness", () => {
    // The load-bearing predicate behind the "You built a blur" failure.
    expect(isSingleSigned(makeKernel("Blur"))).toBe(true);
    expect(isSingleSigned(makeKernel("Pass through"))).toBe(true);
    expect(isSingleSigned(makeKernel("Blank"))).toBe(true);
    expect(isSingleSigned(makeKernel("Vertical edge"))).toBe(false);
    expect(isSingleSigned(makeKernel("Horizontal edge"))).toBe(false);
    expect(isSingleSigned(makeKernel("Sharpen"))).toBe(false);
  });

  it("give every classic edge detector a zero sum", () => {
    // Which is what makes them ignore flat regions and answer only on changes.
    for (const label of ["Vertical edge", "Horizontal edge", "Diagonal edge"]) {
      expect(kernelSum(makeKernel(label))).toBe(0);
    }
    expect(kernelSum(makeKernel("Blur"))).toBe(9);
    expect(kernelSum(makeKernel("Centre spot"))).toBe(-6);
    expect(isBlank(makeKernel("Blank"))).toBe(true);
  });

  it("offer the blur that will not work, on purpose", () => {
    // Hiding the most natural wrong answer would hide the mistake worth making.
    expect(KERNEL_PRESETS.map((p) => p.label)).toContain("Blur");
  });
});

describe("geometry", () => {
  it("shrinks by two per conv and halves per pool", () => {
    expect(stackGeometry([layer("none", "Vertical edge")]).sizes).toEqual([22]);
    expect(stackGeometry([layer("avg", "Vertical edge")]).sizes).toEqual([11]);
    expect(stackGeometry(WINNER).sizes).toEqual([11, 4]);
  });

  it("grows the receptive field with depth, which is the point of depth", () => {
    expect(stackGeometry([layer("none", "Vertical edge")]).receptiveField).toBe(3);
    expect(stackGeometry([layer("avg", "Vertical edge")]).receptiveField).toBe(4);
    // Two layers with pooling: one output cell now covers 10 of 24 pixels.
    expect(stackGeometry(WINNER).receptiveField).toBe(10);
    // Without the first pool it covers half as much, which is what pooling buys.
    expect(
      stackGeometry([
        layer("none", "Vertical edge"),
        layer("none", "Pass through"),
      ]).receptiveField,
    ).toBe(5);
  });

  it("multiplies channels through the depthwise layer", () => {
    expect(outputChannels([layer("avg", "Vertical edge")])).toBe(1);
    expect(
      outputChannels([layer("avg", "Vertical edge", "Horizontal edge")]),
    ).toBe(2);
    expect(outputChannels(WINNER)).toBe(2);
    expect(
      outputChannels([
        layer("avg", "Vertical edge", "Horizontal edge"),
        layer("avg", "Pass through", "Horizontal edge"),
      ]),
    ).toBe(4);
  });

  it("labels channels in the order tf emits them", () => {
    const labels = channelLabels([
      layer("avg", "Vertical edge", "Horizontal edge"),
      layer("avg", "Pass through", "Diagonal edge"),
    ]);
    // depthwiseConv2d emits input-channel-major: in0 x all multipliers first.
    expect(labels).toEqual([
      "Vertical edge → Pass through",
      "Vertical edge → Diagonal edge",
      "Horizontal edge → Pass through",
      "Horizontal edge → Diagonal edge",
    ]);
  });
});

describe("the convolution itself", () => {
  it("agrees with tf.conv2d at every position", () => {
    // The sliding-window explainer and the real op must not drift apart: the whole
    // credibility of the panel rests on the numbers matching.
    const sample = dataset.train[0]!;
    const kernel = makeKernel("Vertical edge");
    const map = featureMapsFor(sample, [makeLayer([kernel], "none")])[0]![0]!;

    let worst = 0;
    for (let row = 0; row < map.size; row += 1) {
      for (let col = 0; col < map.size; col += 1) {
        worst = Math.max(
          worst,
          Math.abs(
            convolveOne(sample, kernel, row, col).activated -
              map.values[row * map.size + col]!,
          ),
        );
      }
    }
    // Measured at 3.4e-7 — float noise, not disagreement.
    expect(worst).toBeLessThan(1e-4);
  });

  it("multiplies nine pixels by nine weights and sums them", () => {
    const sample = dataset.train[3]!;
    const kernel = makeKernel("Vertical edge");
    const result = convolveOne(sample, kernel, 5, 5);
    expect(result.patch).toHaveLength(9);
    expect(result.products).toHaveLength(9);
    const byHand = result.patch.reduce(
      (total, pixel, index) => total + pixel * kernel.weights[index]!,
      0,
    );
    expect(result.sum).toBeCloseTo(byHand, 10);
  });

  it("applies the ReLU, so a negative answer becomes no answer", () => {
    const sample = dataset.train[0]!;
    const negative = makeKernel("Centre spot");
    const result = convolveOne(sample, negative, 5, 5);
    expect(result.sum).toBeLessThan(0);
    expect(result.activated).toBe(0);
  });

  it("treats outside the picture as zero", () => {
    const sample = dataset.train[0]!;
    const kernel = makeKernel("Blur");
    const corner = convolveOne(sample, kernel, -1, -1);
    // The window spans rows -1..1 and columns -1..1, so only its bottom-right
    // 2x2 — patch indices 4, 5, 7, 8 — is actually inside the picture.
    for (const index of [0, 1, 2, 3, 6]) {
      expect(corner.patch[index], `patch cell ${index}`).toBe(0);
    }
    expect(corner.patch[4]).toBeGreaterThan(0);
  });

  it("produces one feature map per filter, at the right size", () => {
    const maps = featureMapsFor(dataset.train[0]!, WINNER);
    expect(maps).toHaveLength(2);
    expect(maps[0]).toHaveLength(1);
    expect(maps[1]).toHaveLength(2);
    expect(maps[0]![0]!.size).toBe(22);
    expect(maps[0]![0]!.values).toHaveLength(22 * 22);
    // Layer 2 reads the pooled 11x11 map, so its own map is 9x9.
    expect(maps[1]![0]!.size).toBe(9);
  });

  it("hands the classifier one number per channel", () => {
    const features = extractFeatures(dataset.validation.slice(0, 5), WINNER);
    expect(features).toHaveLength(5);
    for (const row of features) {
      expect(row).toHaveLength(outputChannels(WINNER));
    }
  });

  it("gives an empty stack no features at all", () => {
    const features = extractFeatures(dataset.validation.slice(0, 3), []);
    expect(features).toEqual([[], [], []]);
  });
});

describe("the premise: the classifier cannot do this alone", () => {
  it("puts raw pixels near chance", async () => {
    // Measured at 31% against 25% for guessing. This is why the dishes use a
    // smooth stripe profile with a continuous phase: with hard-edged stripes the
    // phase quantises to four arrangements and this number was 65%, which would
    // have made the whole game's claim false.
    const accuracy = await scoreRawPixels(dataset);
    expect(accuracy).toBeGreaterThan(CHANCE_RATE - 0.05);
    expect(accuracy).toBeLessThan(0.45);
  }, 180000);

  it("scores a blur at chance too", async () => {
    const score = await scored([layer("avg", "Blur")]);
    expect(score.accuracy).toBeLessThan(0.35);
  }, 120000);

  it("scores an empty stack at zero rather than crashing", async () => {
    const score = await scored([]);
    expect(score.accuracy).toBe(0);
    expect(score.confusion).toHaveLength(CLASS_COUNT);
  });
});

describe("one layer is capped, whatever the kernels", () => {
  const oneLayerSets = [
    ["Vertical edge"],
    ["Horizontal edge"],
    ["Vertical edge", "Horizontal edge"],
    ["Vertical edge", "Horizontal edge", "Diagonal edge"],
    ["Vertical edge", "Horizontal edge", "Diagonal edge", "Centre spot"],
  ];

  it("never reaches the target with hand-designed kernels", async () => {
    // Every kernel set at average pooling, plus the richest set at each other
    // pooling choice. Sweeping all fifteen combinations added four minutes and no
    // information: the cap is a property of the global average at the end of the
    // stack, and pooling cannot lift it.
    const combinations: Array<[string[], PoolType]> = [
      ...oneLayerSets.map((set) => [set, "avg"] as [string[], PoolType]),
      [oneLayerSets.at(-1)!, "max"],
      [oneLayerSets.at(-1)!, "none"],
    ];

    for (const [set, pool] of combinations) {
      const score = await scored([layer(pool, ...set)]);
      expect(
        score.accuracy,
        `${set.join("+")} with ${pool} pooling passed the target`,
      ).toBeLessThan(TARGET_ACCURACY);
      expect(score.accuracy).toBeLessThanOrEqual(ONE_LAYER_CEILING);
    }
  }, 900000);

  it("gets the two uniform dishes right and confuses the two arrangements", async () => {
    // The signature of the ceiling. Both uniform dishes are easy; the arrangement
    // pair collapses into one label, which is exactly what a global average of a
    // single layer's maps predicts.
    const score = await scoreStack(
      dataset,
      [layer("avg", "Vertical edge", "Horizontal edge")],
    );
    expect(score.perClass[0]).toBeGreaterThan(0.9);
    expect(score.perClass[1]).toBeGreaterThan(0.9);
    expect(Math.min(score.perClass[2]!, score.perClass[3]!)).toBeLessThan(0.2);

    const crossed =
      score.confusion[2]![3]! + score.confusion[3]![2]!;
    const mistakes =
      score.confusion[2]!.reduce((s, c, i) => (i === 2 ? s : s + c), 0) +
      score.confusion[3]!.reduce((s, c, i) => (i === 3 ? s : s + c), 0);
    // Nearly every mistake on those two is a confusion with the sibling.
    expect(crossed / mistakes).toBeGreaterThan(0.9);
  }, 180000);

  it("stays capped even when gradient descent picks the kernels", async () => {
    // The claim is about the architecture, not about the player's taste, so it has
    // to hold when nobody is hand-designing anything. Measured 73-83% across
    // kernel counts; the target is 90%.
    for (const count of [2, 4]) {
      const learned = await learnStack(dataset, [
        makeLayer(
          Array.from({ length: count }, () => makeKernel("Blank")),
          "avg",
        ),
      ]);
      expect(learned.accuracy).toBeGreaterThan(0.6);
      expect(
        learned.accuracy,
        `${count} learned kernels in one layer passed the target`,
      ).toBeLessThan(TARGET_ACCURACY);
      expect(learned.accuracy).toBeLessThanOrEqual(ONE_LAYER_CEILING);
    }
  }, 900000);
});

describe("two layers break the cap", () => {
  it("passes with three filters", async () => {
    const score = await scored(WINNER);
    expect(filtersUsed(WINNER)).toBe(3);
    expect(score.accuracy).toBeGreaterThan(TARGET_ACCURACY);
    for (const value of score.perClass) {
      expect(value).toBeGreaterThan(0.9);
    }
  }, 180000);

  it("passes with four, and the extra filter buys nothing", async () => {
    // Which is the "fewest filters" lesson: more is not better, it is just more.
    const four: Layer[] = [
      layer("avg", "Vertical edge", "Horizontal edge"),
      layer("avg", "Pass through", "Horizontal edge"),
    ];
    const score = await scored(four);
    expect(score.accuracy).toBeGreaterThan(TARGET_ACCURACY);
  }, 180000);

  it("needs an asymmetric filter in layer 2, not just any filter", async () => {
    // A blur in layer 2 cannot tell above from below, so the pair stays confused.
    const symmetric: Layer[] = [
      layer("avg", "Vertical edge", "Horizontal edge"),
      layer("avg", "Pass through", "Blur"),
    ];
    const score = await scored(symmetric);
    expect(score.accuracy).toBeLessThan(TARGET_ACCURACY);
  }, 180000);

  it("needs the pass-through as well, so amounts survive alongside arrangement", async () => {
    const arrangementOnly: Layer[] = [
      layer("avg", "Vertical edge", "Horizontal edge"),
      layer("avg", "Horizontal edge"),
    ];
    const score = await scored(arrangementOnly);
    expect(score.accuracy).toBeLessThan(TARGET_ACCURACY);
  }, 180000);

  it("reaches the target with learned kernels too", async () => {
    const learned = await learnStack(dataset, [
      makeLayer([makeKernel("Blank"), makeKernel("Blank")], "avg"),
      makeLayer([makeKernel("Blank"), makeKernel("Blank")], "avg"),
    ]);
    expect(learned.accuracy).toBeGreaterThan(TARGET_ACCURACY);
    expect(learned.kernels).toHaveLength(2);
    for (const perLayer of learned.kernels) {
      for (const weights of perLayer) {
        expect(weights).toHaveLength(KERNEL_CELLS);
      }
    }
  }, 900000);
});

describe("filter health", () => {
  it("finds the filters that never fire", () => {
    const { health } = inspectFilters(dataset, [
      layer("avg", "Vertical edge", "Centre spot", "Sharpen", "Blank"),
    ]);
    const byLabel = new Map(health.map((filter) => [filter.label, filter]));

    // Measured: live presets sit between 0.44 and 3.97; these three are at zero.
    expect(byLabel.get("Vertical edge")!.dead).toBe(false);
    expect(byLabel.get("Vertical edge")!.meanActivation).toBeGreaterThan(0.4);
    for (const label of ["Centre spot", "Sharpen", "Blank"]) {
      expect(byLabel.get(label)!.dead, `${label} should be dead`).toBe(true);
      expect(byLabel.get(label)!.meanActivation).toBeLessThan(DEAD_THRESHOLD);
    }
  }, 120000);

  it("explains a dead filter by its weight sum", () => {
    const { health } = inspectFilters(dataset, [
      layer("avg", "Vertical edge", "Centre spot"),
    ]);
    const spot = health.find((filter) => filter.label === "Centre spot")!;
    // Negative total plus non-negative pixels plus a ReLU equals silence.
    expect(spot.weightSum).toBe(-6);
    expect(spot.dead).toBe(true);
  }, 120000);

  it("flags a brightness meter in layer 1 but not in layer 2", () => {
    // The argument for the blur diagnosis is that an average of nine PIXELS
    // measures brightness, which is randomised. One layer up the input is an
    // activation map and averaging it is a legitimate thing to do.
    const { health } = inspectFilters(dataset, [
      layer("avg", "Blur"),
      layer("avg", "Blur"),
    ]);
    expect(health.find((f) => f.layerIndex === 0)!.blur).toBe(true);
    expect(health.find((f) => f.layerIndex === 1)!.blur).toBe(false);
  }, 120000);

  it("finds duplicate filters from their activations", () => {
    const { duplicates } = inspectFilters(dataset, [
      layer("avg", "Vertical edge", "Vertical edge"),
    ]);
    expect(duplicates).toHaveLength(1);
    expect(duplicates[0]!.correlation).toBeGreaterThan(DUPLICATE_THRESHOLD);
    expect(duplicates[0]!.correlation).toBeCloseTo(1, 2);
  }, 120000);

  it("does not call a kernel and its negation duplicates", () => {
    // They look identical on the grid and are anti-correlated before the ReLU;
    // afterwards they catch opposite edge polarities, so they are two filters.
    const negated = makeKernel("Vertical edge");
    negated.weights = negated.weights.map((weight) => -weight);
    negated.label = "Vertical edge (negated)";
    const { duplicates, health } = inspectFilters(dataset, [
      makeLayer([makeKernel("Vertical edge"), negated], "avg"),
    ]);
    expect(duplicates).toHaveLength(0);
    for (const filter of health) {
      expect(filter.dead).toBe(false);
    }
  }, 120000);

  it("does not call different detectors duplicates", () => {
    const { duplicates } = inspectFilters(dataset, [
      layer("avg", "Vertical edge", "Horizontal edge", "Diagonal edge"),
    ]);
    expect(duplicates).toHaveLength(0);
  }, 120000);

  it("computes correlation the ordinary way", () => {
    expect(correlation([1, 2, 3], [1, 2, 3])).toBeCloseTo(1, 10);
    expect(correlation([1, 2, 3], [3, 2, 1])).toBeCloseTo(-1, 10);
    expect(correlation([1, 1, 1], [1, 2, 3])).toBe(0);
    expect(correlation([], [])).toBe(0);
  });
});

describe("the verdict", () => {
  const healthCache = new Map<string, ReturnType<typeof inspectFilters>>();
  const inspected = (layers: readonly Layer[]) => {
    const key = keyOf(layers);
    let found = healthCache.get(key);
    if (found === undefined) {
      found = inspectFilters(dataset, layers);
      healthCache.set(key, found);
    }
    return found;
  };

  const judge = async (layers: Layer[], learned = null) => {
    const score = await scored(layers);
    const { health, duplicates } = inspected(layers);
    return evaluateKitchen({ layers, score, health, duplicates, learned });
  };

  it("says nothing about an empty kitchen", async () => {
    const evaluation = await judge([]);
    expect(evaluation.outcome).toBe("empty");
    expect(evaluation.failure).toBeNull();
    expect(evaluation.points).toBe(0);
  });

  it("names the blur", async () => {
    const evaluation = await judge([layer("avg", "Blur", "Blur")]);
    expect(evaluation.outcome).toBe("blur-only");
    expect(evaluation.failure?.name).toBe("You built a blur");
    expect(evaluation.failure?.detail).toMatch(/AVERAGE/);
    expect(evaluation.failure?.detail).toMatch(/brightness/);
  }, 180000);

  it("names dead filters before anything else", async () => {
    // The catalog's declared failure mode for this game, and it is checked ahead
    // of the score: a corpse in the stack is a fact whether or not you passed.
    const evaluation = await judge([layer("avg", "Vertical edge", "Centre spot")]);
    expect(evaluation.outcome).toBe("dead-filters");
    expect(evaluation.failure?.name).toBe("Dead filters");
    expect(evaluation.failure?.detail).toMatch(/-6/);
    expect(evaluation.failure?.detail).toMatch(/ReLU/);
  }, 180000);

  it("names duplicates", async () => {
    const evaluation = await judge([
      layer("avg", "Vertical edge", "Vertical edge"),
    ]);
    expect(evaluation.outcome).toBe("duplicate-filters");
    expect(evaluation.failure?.name).toBe("Duplicate filters");
    expect(evaluation.failure?.detail).toMatch(/correlate at 1\.00/);
  }, 180000);

  it("names the one-layer ceiling, from the confusion matrix", async () => {
    const evaluation = await judge([
      layer("avg", "Vertical edge", "Horizontal edge"),
    ]);
    expect(evaluation.outcome).toBe("one-layer-ceiling");
    expect(evaluation.failure?.name).toBe("One layer is not enough");
    expect(evaluation.failure?.detail).toMatch(/global average/);
    expect(evaluation.failure?.detail).toMatch(/second layer/);
  }, 180000);

  it("serves a winning stack", async () => {
    const evaluation = await judge(WINNER);
    expect(evaluation.outcome).toBe("served");
    expect(evaluation.failure).toBeNull();
    // Three stars once the code lane serves it, by the engine's own rule — the
    // game no longer keeps a second rubric of its own for the tooltip to drift from.
    expect(
      starsFor({ completed: true, bestScore: evaluation.points, codeLaneCleared: true }),
    ).toBe(3);
    expect(evaluation.points).toBeGreaterThan(0.9);
  }, 180000);

  it("rewards thrift", async () => {
    const lean = await judge(WINNER);
    const fat = await judge([
      layer("avg", "Vertical edge", "Horizontal edge", "Diagonal edge"),
      layer("avg", "Pass through", "Horizontal edge", "Vertical edge"),
    ]);
    expect(lean.outcome).toBe("served");
    expect(fat.outcome).toBe("served");
    expect(lean.filters).toBeLessThan(fat.filters);
    expect(lean.points).toBeGreaterThan(fat.points);
  }, 300000);

  it("keeps every failing score below every passing score", async () => {
    const failing = await judge([layer("avg", "Blur", "Blur")]);
    const passing = await judge(WINNER);
    expect(failing.points).toBeLessThan(0.5);
    expect(passing.points).toBeGreaterThan(0.5);
  }, 300000);

  it("carries numbers in every named failure", async () => {
    const configs: Layer[][] = [
      [layer("avg", "Blur", "Blur")],
      [layer("avg", "Vertical edge", "Centre spot")],
      [layer("avg", "Vertical edge", "Vertical edge")],
      [layer("avg", "Vertical edge", "Horizontal edge")],
    ];
    for (const layers of configs) {
      const evaluation = await judge(layers);
      expect(evaluation.failure).not.toBeNull();
      expect(evaluation.failure!.name).not.toMatch(/game over/i);
      expect(evaluation.failure!.detail).toMatch(/\d/);
      expect(evaluation.failure!.detail.length).toBeGreaterThan(150);
    }
  }, 600000);
});

describe("the store", () => {
  beforeEach(() => {
    useKitchenStore.setState({
      layers: [makeLayer([makeKernel("Blur")], "avg")],
      previewIndex: 0,
      windowRow: 8,
      windowCol: 8,
      windowKernel: 0,
      evaluation: null,
      learned: null,
      scoring: false,
      learning: false,
      phase: "designing",
      failure: null,
    });
  });

  it("opens on the mistake everybody makes", () => {
    const state = useKitchenStore.getState();
    expect(state.layers).toHaveLength(1);
    expect(state.layers[0]!.kernels[0]!.label).toBe("Blur");
    expect(state.layers[0]!.pool).toBe("avg");
  });

  it("clamps a weight to the allowed range and relabels the kernel", () => {
    useKitchenStore.getState().setWeight(0, 0, 0, 99);
    expect(useKitchenStore.getState().layers[0]!.kernels[0]!.weights[0]).toBe(
      WEIGHT_MAX,
    );
    useKitchenStore.getState().setWeight(0, 0, 0, -99);
    expect(useKitchenStore.getState().layers[0]!.kernels[0]!.weights[0]).toBe(
      WEIGHT_MIN,
    );
    // Editing away from a preset must stop claiming to be that preset.
    expect(useKitchenStore.getState().layers[0]!.kernels[0]!.label).toBe("Custom");
  });

  it("ignores an out-of-range cell", () => {
    const before = useKitchenStore.getState().layers[0]!.kernels[0]!.weights;
    useKitchenStore.getState().setWeight(0, 0, 99, 1);
    expect(useKitchenStore.getState().layers[0]!.kernels[0]!.weights).toEqual(
      before,
    );
  });

  it("applies a preset by name", () => {
    useKitchenStore.getState().applyPreset(0, 0, "Vertical edge");
    const kernel = useKitchenStore.getState().layers[0]!.kernels[0]!;
    expect(kernel.label).toBe("Vertical edge");
    expect(kernel.weights).toEqual([1, 0, -1, 2, 0, -2, 1, 0, -1]);
  });

  it("ignores an unknown preset", () => {
    useKitchenStore.getState().applyPreset(0, 0, "Cat detector");
    expect(useKitchenStore.getState().layers[0]!.kernels[0]!.label).toBe("Blur");
  });

  it("adds and removes filters, keeping at least one", () => {
    useKitchenStore.getState().addKernel(0, "Vertical edge");
    expect(useKitchenStore.getState().layers[0]!.kernels).toHaveLength(2);
    useKitchenStore.getState().removeKernel(0, 1);
    expect(useKitchenStore.getState().layers[0]!.kernels).toHaveLength(1);
    useKitchenStore.getState().removeKernel(0, 0);
    expect(useKitchenStore.getState().layers[0]!.kernels).toHaveLength(1);
  });

  it("caps a layer at four filters", () => {
    for (let index = 0; index < 10; index += 1) {
      useKitchenStore.getState().addKernel(0, "Vertical edge");
    }
    expect(
      useKitchenStore.getState().layers[0]!.kernels.length,
    ).toBeLessThanOrEqual(MAX_KERNELS_PER_LAYER);
  });

  it("caps the whole stack at the filter budget", () => {
    useKitchenStore.getState().addLayer();
    for (let index = 0; index < 12; index += 1) {
      useKitchenStore.getState().addKernel(0, "Vertical edge");
      useKitchenStore.getState().addKernel(1, "Pass through");
    }
    expect(filtersUsed(useKitchenStore.getState().layers)).toBeLessThanOrEqual(
      FILTER_BUDGET,
    );
  });

  it("adds a second layer with both a pass-through and an asymmetric filter", () => {
    // Those two together are what actually breaks the ceiling, so the default has
    // to be a stack that can win rather than one that cannot.
    useKitchenStore.getState().addLayer();
    const state = useKitchenStore.getState();
    expect(state.layers).toHaveLength(2);
    const labels = state.layers[1]!.kernels.map((kernel) => kernel.label);
    expect(labels).toContain("Pass through");
    expect(labels).toContain("Horizontal edge");
  });

  it("caps the stack at two layers", () => {
    useKitchenStore.getState().addLayer();
    useKitchenStore.getState().addLayer();
    expect(useKitchenStore.getState().layers.length).toBeLessThanOrEqual(
      MAX_LAYERS,
    );
  });

  it("removes a layer but never the last one", () => {
    useKitchenStore.getState().addLayer();
    useKitchenStore.getState().removeLayer(1);
    expect(useKitchenStore.getState().layers).toHaveLength(1);
    useKitchenStore.getState().removeLayer(0);
    expect(useKitchenStore.getState().layers).toHaveLength(1);
  });

  it("changes the pool type", () => {
    useKitchenStore.getState().setPool(0, "max");
    expect(useKitchenStore.getState().layers[0]!.pool).toBe("max");
    useKitchenStore.getState().setPool(0, "none");
    expect(useKitchenStore.getState().layers[0]!.pool).toBe("none");
  });

  it("keeps the sliding window inside the picture", () => {
    useKitchenStore.getState().moveWindow(-5, -5);
    expect(useKitchenStore.getState().windowRow).toBe(0);
    expect(useKitchenStore.getState().windowCol).toBe(0);
    useKitchenStore.getState().moveWindow(999, 999);
    expect(useKitchenStore.getState().windowRow).toBe(IMAGE_SIZE - 3);
    expect(useKitchenStore.getState().windowCol).toBe(IMAGE_SIZE - 3);
  });

  it("keeps the preview inside the validation split", () => {
    useKitchenStore.getState().setPreview(-1);
    expect(useKitchenStore.getState().previewIndex).toBe(0);
    useKitchenStore.getState().setPreview(999999);
    expect(useKitchenStore.getState().previewIndex).toBe(
      dataset.validation.length - 1,
    );
  });

  it("keeps the window's kernel selection valid when filters are removed", () => {
    useKitchenStore.getState().addKernel(0, "Vertical edge");
    useKitchenStore.getState().setWindowKernel(1);
    expect(useKitchenStore.getState().windowKernel).toBe(1);
    useKitchenStore.getState().removeKernel(0, 1);
    expect(useKitchenStore.getState().windowKernel).toBe(0);
  });

  it("scores the opening filter and names the blur", async () => {
    await useKitchenStore.getState().score_();
    const state = useKitchenStore.getState();
    expect(state.evaluation?.outcome).toBe("blur-only");
    expect(state.failure?.name).toBe("You built a blur");
    expect(state.score.accuracy).toBeLessThan(0.35);
    expect(state.baseline).not.toBeNull();
  }, 300000);

  it("serves a winning stack and records progression", async () => {
    useKitchenStore.getState().applyPreset(0, 0, "Vertical edge");
    useKitchenStore.getState().addLayer();
    await useKitchenStore.getState().score_();

    const state = useKitchenStore.getState();
    expect(state.phase).toBe("served");
    expect(state.failure).toBeNull();
    expect(state.score.accuracy).toBeGreaterThan(TARGET_ACCURACY);
  }, 300000);

  it("fits the raw-pixel baseline once and keeps it", async () => {
    await useKitchenStore.getState().score_();
    const first = useKitchenStore.getState().baseline;
    useKitchenStore.getState().applyPreset(0, 0, "Vertical edge");
    await useKitchenStore.getState().score_();
    expect(useKitchenStore.getState().baseline).toBe(first);
  }, 300000);

  it("resets to the opening kitchen", async () => {
    useKitchenStore.getState().applyPreset(0, 0, "Vertical edge");
    useKitchenStore.getState().addLayer();
    useKitchenStore.getState().reset();
    const state = useKitchenStore.getState();
    expect(state.layers).toHaveLength(1);
    expect(state.layers[0]!.kernels[0]!.label).toBe("Blur");
    expect(state.learned).toBeNull();
  });
});

describe("the code lane api", () => {
  beforeEach(() => {
    useKitchenStore.setState({
      layers: [makeLayer([makeKernel("Blur")], "avg")],
      evaluation: null,
      learned: null,
      scoring: false,
      learning: false,
      phase: "designing",
      failure: null,
    });
  });

  it("writes the same state the steppers write", () => {
    const api = createCodeApi();
    api.applyPreset(0, 0, "Vertical edge");
    expect(useKitchenStore.getState().layers[0]!.kernels[0]!.label).toBe(
      "Vertical edge",
    );

    api.setWeights(0, 0, [0, 0, 0, 0, 1, 0, 0, 0, 0]);
    expect(useKitchenStore.getState().layers[0]!.kernels[0]!.weights).toEqual([
      0, 0, 0, 0, 1, 0, 0, 0, 0,
    ]);

    api.addLayer();
    expect(useKitchenStore.getState().layers).toHaveLength(2);
    api.setPool(0, "max");
    expect(useKitchenStore.getState().layers[0]!.pool).toBe("max");
  });

  it("rejects malformed input by name", () => {
    const api = createCodeApi();
    expect(() => api.setWeights(0, 0, [1, 2, 3])).toThrow(/exactly 9/);
    expect(() => api.setPool(0, "sideways")).toThrow(/Unknown pool/);
    expect(() => api.trial([])).rejects.toThrow(/at least one layer/);
  });

  it("scores a trial stack without adopting it", async () => {
    const api = createCodeApi();
    const before = useKitchenStore.getState().layers[0]!.kernels[0]!.label;

    const result = await api.trial([
      { pool: "avg", kernels: ["Vertical edge"] },
      { pool: "avg", kernels: ["Pass through", "Horizontal edge"] },
    ]);
    expect(result.accuracy).toBeGreaterThan(TARGET_ACCURACY);
    expect(result.filters).toBe(3);
    expect(result.channels).toBe(2);

    // The player's own kitchen is untouched.
    expect(useKitchenStore.getState().layers).toHaveLength(1);
    expect(useKitchenStore.getState().layers[0]!.kernels[0]!.label).toBe(before);
  }, 300000);

  it("accepts raw weight arrays in a trial", async () => {
    const api = createCodeApi();
    const result = await api.trial([
      { pool: "avg", kernels: [[1, 0, -1, 2, 0, -2, 1, 0, -1]] },
    ]);
    expect(result.accuracy).toBeGreaterThan(0.5);
    expect(result.filters).toBe(1);
  }, 300000);

  it("rejects a trial kernel of the wrong size", async () => {
    const api = createCodeApi();
    await expect(
      api.trial([{ pool: "avg", kernels: [[1, 2, 3]] }]),
    ).rejects.toThrow(/exactly 9 weights/);
    await expect(
      api.trial([{ pool: "avg", kernels: ["Cat detector"] }]),
    ).rejects.toThrow(/Unknown preset/);
  });

  it("enforces the same layer and kernel caps as the buttons", async () => {
    const api = createCodeApi();
    await expect(
      api.trial([
        { kernels: ["Blur"] },
        { kernels: ["Blur"] },
        { kernels: ["Blur"] },
      ]),
    ).rejects.toThrow(/At most 2 layers/);
    await expect(
      api.trial([
        { kernels: ["Blur", "Blur", "Blur", "Blur", "Blur"] },
      ]),
    ).rejects.toThrow(/At most 4 kernels/);
  });

  it("trains a shape end to end without touching the player's stack", async () => {
    const api = createCodeApi();
    const learned = await api.learn([{ pool: "avg", kernels: 2 }]);
    expect(learned.accuracy).toBeLessThan(TARGET_ACCURACY);
    expect(useKitchenStore.getState().layers).toHaveLength(1);
    expect(useKitchenStore.getState().learned).toBeNull();
  }, 600000);

  it("hands back copies, not live state", () => {
    const api = createCodeApi();
    const layers = api.layers();
    layers[0]!.kernels[0]!.weights[0] = 99;
    expect(
      useKitchenStore.getState().layers[0]!.kernels[0]!.weights[0],
    ).not.toBe(99);
  });

  it("reports feature maps as plain arrays", () => {
    const api = createCodeApi();
    const maps = api.maps(0);
    expect(maps).toHaveLength(1);
    expect(maps[0]![0]!.size).toBe(22);
    expect(Array.isArray(maps[0]![0]!.values)).toBe(true);
  });
});

describe("why-cards", () => {
  const cardFor = async (layers: Layer[]) => {
    const score = await scored(layers);
    const { health, duplicates } = inspectFilters(dataset, layers);
    const evaluation = evaluateKitchen({
      layers,
      score,
      health,
      duplicates,
      learned: null,
    });
    return whyCardFor({ kind: "scored", evaluation, layers, baseline: 0.31 });
  };

  it("opens by naming the trap it has set", () => {
    const card = whyCardFor({ kind: "briefing" });
    expect(card.body).toMatch(/Blur/);
    expect(card.body).toMatch(/randomised brightness/);
  });

  it("explains a one-sided kernel as a brightness meter", () => {
    const card = whyCardFor({
      kind: "weight-changed",
      kernel: makeKernel("Blur"),
      layerIndex: 0,
    });
    expect(card.tone).toBe("warn");
    expect(card.body).toMatch(/no sign change/i);
  });

  it("praises a zero-sum kernel for ignoring flat regions", () => {
    const card = whyCardFor({
      kind: "weight-changed",
      kernel: makeKernel("Vertical edge"),
      layerIndex: 0,
    });
    expect(card.tone).toBe("info");
    expect(card.body).toMatch(/cancel/);
  });

  it("calls out an all-zero kernel", () => {
    const card = whyCardFor({
      kind: "weight-changed",
      kernel: makeKernel("Blank"),
      layerIndex: 0,
    });
    expect(card.tone).toBe("warn");
    expect(card.title).toMatch(/zero/);
  });

  it("says a blur is legitimate in layer 2 but not layer 1", () => {
    const preset = KERNEL_PRESETS.find((p) => p.label === "Blur")!;
    const one = whyCardFor({ kind: "preset-applied", preset, layerIndex: 0 });
    const two = whyCardFor({ kind: "preset-applied", preset, layerIndex: 1 });
    expect(one.body).toMatch(/brightness meter/);
    expect(two.body).toMatch(/legitimate/);
  });

  it("explains what pooling buys, in pixels", () => {
    const card = whyCardFor({
      kind: "pool-changed",
      pool: "avg",
      layerIndex: 0,
      layers: WINNER,
      previousAccuracy: 0.75,
    });
    expect(card.body).toMatch(/10 pixels/);
  });

  it("explains why layer 2 is different in kind", () => {
    const card = whyCardFor({ kind: "layer-added", layers: WINNER });
    expect(card.title).toMatch(/feature maps, not pixels/i);
    expect(card.body).toMatch(/pass-through/);
    expect(card.body).toMatch(/arrangements of layer-1 patterns/);
  });

  it("credits the filters, not the classifier, on a win", async () => {
    const card = await cardFor(WINNER);
    expect(card.tone).toBe("good");
    expect(card.body).toMatch(/reused at every position/);
  }, 300000);

  it("tells the player a feature map should stop looking like the picture", async () => {
    const card = await cardFor([layer("avg", "Blur", "Blur")]);
    expect(card.tone).toBe("bad");
    expect(card.body).toMatch(/not an image/);
  }, 300000);

  it("calls the ceiling a wall rather than a plateau", async () => {
    const card = await cardFor([layer("avg", "Vertical edge", "Horizontal edge")]);
    expect(card.title).toMatch(/wall/);
    expect(card.body).toMatch(/global average/);
  }, 300000);

  it("reads the learned comparison as being about the shape", () => {
    const card = whyCardFor({
      kind: "learned",
      learned: { accuracy: 0.79, kernels: [] },
      layers: [makeLayer([makeKernel("Vertical edge")], "avg")],
      mine: 0.75,
    });
    expect(card.body).toMatch(/shape is the limit/);
  });

  it("explains the three outcomes of the sliding window", () => {
    expect(windowCaption(3.2, 3.2)).toMatch(/looks like the kernel/);
    expect(windowCaption(0, 0)).toMatch(/cancelled|flat/);
    expect(windowCaption(-2.5, 0)).toMatch(/opposite/);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Invariants behind the audit fixes: memory, cancellation, stale results, the
// star rule, and what the code lane's setters promise.
// ═══════════════════════════════════════════════════════════════════════════

/** A few plates are enough to exercise a fit's bookkeeping, and cost almost nothing. */
const tiny = {
  train: dataset.train.slice(0, 16),
  validation: dataset.validation.slice(0, 8),
};

describe("fits clean up after themselves", () => {
  // Earlier tests leave rescores running in the background. A direct score
  // supersedes and cancels all of them, so nothing else is allocating tensors
  // while these count them.
  beforeAll(async () => {
    await useKitchenStore.getState().score_();
  }, 300000);

  it("frees the optimizer along with the model on every score, baseline and learn", async () => {
    // Warm each path once so backend-level caches are not counted as leaks.
    const stack = [layer("avg", "Vertical edge")];
    const shape = [makeLayer([makeKernel("Blank")], "avg")];
    await scoreStack(tiny, stack);
    await scoreRawPixels(tiny);
    await learnStack(tiny, shape);

    const before = tf.memory().numTensors;
    await scoreStack(tiny, stack);
    expect(tf.memory().numTensors, "scoreStack leaked").toBe(before);
    await scoreRawPixels(tiny);
    expect(tf.memory().numTensors, "scoreRawPixels leaked").toBe(before);
    await learnStack(tiny, shape);
    expect(tf.memory().numTensors, "learnStack leaked").toBe(before);
  }, 180000);

  it("stops a cancelled fit, reports it as cancelled, and still frees everything", async () => {
    const stack = [layer("avg", "Vertical edge")];
    await scoreStack(tiny, stack);
    const before = tf.memory().numTensors;

    await expect(
      scoreStack(tiny, stack, { cancelled: true }),
    ).rejects.toBeInstanceOf(FitCancelledError);
    const token = { cancelled: false };
    const learning = learnStack(tiny, [makeLayer([makeKernel("Blank")], "avg")], token);
    token.cancelled = true;
    await expect(learning).rejects.toBeInstanceOf(FitCancelledError);

    expect(tf.memory().numTensors).toBe(before);
  }, 180000);
});

describe("the shape a learned result belongs to", () => {
  it("ignores weights and changes with kernel counts and pooling", () => {
    const base = [layer("avg", "Vertical edge")];
    expect(shapeKey([layer("avg", "Blur")])).toBe(shapeKey(base));
    expect(shapeKey([layer("max", "Vertical edge")])).not.toBe(shapeKey(base));
    expect(shapeKey([layer("avg", "Vertical edge", "Blur")])).not.toBe(shapeKey(base));
    expect(shapeKey(WINNER)).not.toBe(shapeKey(base));
  });
});

describe("the second star, as the score formula decides it", () => {
  it("lands exactly on the thrifty, duplicate-free wins the criterion names", () => {
    for (let filters = 1; filters <= FILTER_BUDGET; filters += 1) {
      for (let duplicates = 0; duplicates <= 3; duplicates += 1) {
        const earns = servedPoints(filters, duplicates) >= HIGH_SCORE_THRESHOLD;
        expect(earns, `${filters} filters, ${duplicates} duplicate pairs`).toBe(
          filters <= THRIFTY_FILTERS && duplicates === 0,
        );
      }
    }
  });

  it("keeps every served stack above every failing one", () => {
    for (let filters = 1; filters <= FILTER_BUDGET; filters += 1) {
      for (let duplicates = 0; duplicates <= 6; duplicates += 1) {
        expect(servedPoints(filters, duplicates)).toBeGreaterThan(0.5);
      }
    }
  });

  it("no longer hands the thrift star to a six-filter win", async () => {
    const fat: Layer[] = [
      layer("avg", "Vertical edge", "Horizontal edge", "Diagonal edge"),
      layer("avg", "Pass through", "Horizontal edge", "Vertical edge"),
    ];
    const score = await scored(fat);
    const { health, duplicates } = inspectFilters(dataset, fat);
    const evaluation = evaluateKitchen({ layers: fat, score, health, duplicates, learned: null });
    expect(evaluation.outcome).toBe("served");
    expect(evaluation.points).toBeLessThan(HIGH_SCORE_THRESHOLD);

    const lean = await scored(WINNER);
    const leanHealth = inspectFilters(dataset, WINNER);
    const leanEvaluation = evaluateKitchen({
      layers: WINNER,
      score: lean,
      health: leanHealth.health,
      duplicates: leanHealth.duplicates,
      learned: null,
    });
    expect(leanEvaluation.points).toBeGreaterThanOrEqual(HIGH_SCORE_THRESHOLD);
  }, 300000);
});

describe("the blur verdict with a second layer on top", () => {
  it("still names the blur, and stops claiming every filter is an average", async () => {
    // What "Add layer 2" on the opening kitchen produces: a Blur under a
    // pass-through and a real edge detector.
    const stack: Layer[] = [
      layer("avg", "Blur"),
      layer("avg", "Pass through", "Horizontal edge"),
    ];
    const score = await scored(stack);
    const { health, duplicates } = inspectFilters(dataset, stack);
    const evaluation = evaluateKitchen({ layers: stack, score, health, duplicates, learned: null });

    expect(evaluation.outcome).toBe("blur-only");
    const detail = evaluation.failure!.detail;
    expect(detail).toMatch(/Every filter in layer 1/);
    expect(detail).not.toMatch(/Every filter you have/);
    expect(detail).toMatch(/"Horizontal edge"/);
    expect(detail).toMatch(/positive weight and at least one negative weight/);
    // It compares against the target, not against guessing, because layer 2 is
    // detecting something and the score is no longer at chance.
    expect(detail).toContain(`against the ${Math.round(TARGET_ACCURACY * 100)}% needed`);

    // Only the filter that can detect anything is credited with finding it, and
    // the verb agrees with how many there are.
    expect(detail).toContain(`Whatever "Horizontal edge" finds in that copy, it could`);
    expect(detail).not.toMatch(/Whatever[^.]*"Pass through"/);

    const card = whyCardFor({ kind: "scored", evaluation, layers: stack, baseline: 0.31 });
    expect(card.title).not.toMatch(/looks like the photograph/);
    expect(card.body).toMatch(/layer 2/);
    // "Which is why this scores X rather than chance" is said because it is true.
    expect(clearlyAboveChance(score.accuracy)).toBe(true);
    expect(card.body).toContain(
      `which is why this scores ${Math.round(score.accuracy * 100)}% rather than ${Math.round(
        CHANCE_RATE * 100,
      )}%`,
    );

    // The same card on a score at chance does not claim layer 2 is finding stripes.
    const atChance = whyCardFor({
      kind: "scored",
      evaluation: { ...evaluation, score: { ...score, accuracy: CHANCE_RATE } },
      layers: stack,
      baseline: 0.31,
    });
    expect(atChance.body).not.toMatch(/which is why this scores/);
    expect(atChance.body).toMatch(/finding almost nothing/);
  }, 300000);

  it("keeps the every-filter wording when nothing above layer 1 has a sign change", async () => {
    // Two clicks from the opening kitchen: Add layer 2, then remove its
    // Horizontal edge. A pass-through detects nothing; it only passes on.
    const stack: Layer[] = [layer("avg", "Blur"), layer("avg", "Pass through")];
    expect(detectorsAboveLayerOne(stack)).toEqual([]);
    const score = await scored(stack);
    const { health, duplicates } = inspectFilters(dataset, stack);
    const evaluation = evaluateKitchen({ layers: stack, score, health, duplicates, learned: null });

    expect(evaluation.outcome).toBe("blur-only");
    // Measured at 23%: no filter anywhere measures a difference, and the score says so.
    expect(clearlyAboveChance(score.accuracy)).toBe(false);
    const detail = evaluation.failure!.detail;
    expect(detail).toMatch(/Every filter you have, in both layers, has weights of a single sign/);
    expect(detail).not.toMatch(/Whatever/);
    expect(detail).toContain(
      `The detection score agrees: ${Math.round(score.accuracy * 100)}% against ${Math.round(
        CHANCE_RATE * 100,
      )}% for guessing.`,
    );

    const card = whyCardFor({ kind: "scored", evaluation, layers: stack, baseline: 0.31 });
    expect(card.key).toBe("blur-only");
    expect(card.title).toMatch(/looks like the photograph/);
    expect(card.body).not.toMatch(/find stripes|which is why this scores/);
  }, 300000);

  it("counts only sign-changing filters above layer 1 as detectors, once each", () => {
    expect(detectorsAboveLayerOne([layer("avg", "Vertical edge")])).toEqual([]);
    expect(
      detectorsAboveLayerOne([
        layer("avg", "Blur"),
        layer("avg", "Pass through", "Blur", "Blank"),
      ]),
    ).toEqual([]);
    expect(
      detectorsAboveLayerOne([
        layer("avg", "Vertical edge"),
        layer("avg", "Pass through", "Horizontal edge", "Horizontal edge"),
      ]),
    ).toEqual(["Horizontal edge"]);
  });
});

describe("the math drawer", () => {
  it("indexes the kernel the same way round as the code beside it", () => {
    // The code multiplies image[y + i][x + j] by kernel[i][j]: rows first.
    expect(MATH_CODE).toContain("image[y + i][x + j] * kernel[i][j]");
    expect(MATH_EQUATION).toContain("I(y+i,\\; x+j)\\, K(i,j)");
    expect(MATH_EQUATION).not.toContain("I(x+i");
  });

  it("shows the average pool the kitchen defaults to, not only the max", () => {
    expect(makeLayer([]).pool).toBe("avg");
    expect(MATH_EQUATION).toMatch(/P_\{\\text\{avg\}\}/);
    expect(MATH_EQUATION).toMatch(/P_\{\\max\}/);
  });

  it("does not call the one-layer ceiling a mathematical impossibility", () => {
    expect(MATH_NOTES).not.toMatch(/mathematically incapable/);
    expect(MATH_NOTES).toContain(`${Math.round(ONE_LAYER_CEILING * 100)}%`);
  });
});

describe("the store's async lifecycle", () => {
  beforeEach(() => {
    useProgression.setState({
      ...EMPTY_PROGRESSION,
      hydrated: true,
      syncError: null,
      lastGain: null,
    });
    useProgression.getState().setAdapter(createMemoryAdapter());
    useKitchenStore.setState({
      layers: [makeLayer([makeKernel("Blur")], "avg")],
      evaluation: null,
      learned: null,
      learnedShape: null,
      scoring: false,
      learning: false,
      phase: "designing",
      failure: null,
      lane: "visual",
      editSource: "visual",
    });
  });

  it("forgets a learned result when the shape changes, and keeps it through weight edits", () => {
    const layers = useKitchenStore.getState().layers;
    useKitchenStore.setState({
      learned: { accuracy: 0.8, kernels: [] },
      learnedShape: shapeKey(layers),
    });
    expect(currentLearned(useKitchenStore.getState())).not.toBeNull();

    useKitchenStore.getState().applyPreset(0, 0, "Vertical edge");
    expect(currentLearned(useKitchenStore.getState())).not.toBeNull();

    useKitchenStore.getState().addKernel(0, "Horizontal edge");
    expect(currentLearned(useKitchenStore.getState())).toBeNull();

    useKitchenStore.getState().removeKernel(0, 1);
    // Back to the shape that was learned, so its result is true again.
    expect(currentLearned(useKitchenStore.getState())).not.toBeNull();

    useKitchenStore.getState().setPool(0, "max");
    expect(currentLearned(useKitchenStore.getState())).toBeNull();
    useKitchenStore.getState().setPool(0, "avg");

    useKitchenStore.getState().addLayer();
    expect(currentLearned(useKitchenStore.getState())).toBeNull();
  });

  it("keeps the failure up while an edit is rescored, and reuses it when nothing changed", async () => {
    await useKitchenStore.getState().score_();
    const first = useKitchenStore.getState().failure;
    expect(first?.name).toBe("You built a blur");

    // An edit no longer blanks the verdict it is about to be judged against.
    useKitchenStore.getState().setWeight(0, 0, 4, 2);
    expect(useKitchenStore.getState().failure).toBe(first);
    expect(useKitchenStore.getState().scoring).toBe(true);

    // Put the weight back and rescore: same verdict, same words, same object —
    // so the shell's alert is not remounted and not re-announced.
    useKitchenStore.getState().setWeight(0, 0, 4, 1);
    await useKitchenStore.getState().score_();
    expect(useKitchenStore.getState().failure).toBe(first);
  }, 300000);

  it("makes one edit of a code-lane setWeights call, not nine", () => {
    let layerChanges = 0;
    const unsubscribe = useKitchenStore.subscribe((state, previous) => {
      if (state.layers !== previous.layers) layerChanges += 1;
    });
    void createCodeApi().setWeights(0, 0, [1, 0, -1, 2, 0, -2, 1, 0, -1]);
    unsubscribe();
    expect(layerChanges).toBe(1);
    expect(useKitchenStore.getState().layers[0]!.kernels[0]!.label).toBe(
      "Vertical edge",
    );
  });

  it("starts learning while a rescore is pending instead of silently doing nothing", async () => {
    useKitchenStore.getState().applyPreset(0, 0, "Vertical edge");
    expect(useKitchenStore.getState().scoring).toBe(true);
    const run = useKitchenStore.getState().letItLearn();
    expect(useKitchenStore.getState().learning).toBe(true);
    // A Retry while it is still waiting on that rescore stops it before it
    // trains: nothing from that run lands afterwards. (Retry mid-fit is below.)
    useKitchenStore.getState().reset();
    await run;
    const state = useKitchenStore.getState();
    expect(state.learning).toBe(false);
    expect(state.learned).toBeNull();
    expect(state.whyCard?.key).not.toMatch(/^learned-/);
    expect(state.layers[0]!.kernels[0]!.label).toBe("Blur");
  }, 300000);

  it("stops a Let it learn that is already fitting when the kitchen is reset", async () => {
    useKitchenStore.setState({
      layers: [makeLayer([makeKernel("Vertical edge")], "avg")],
      // Known, so no baseline fit starts behind the score and muddies the count.
      baseline: 0.3,
    });
    // A direct score supersedes whatever earlier tests left scheduled, so from
    // here on the only thing allocating tensors is the run under test.
    await useKitchenStore.getState().score_();
    const before = tf.memory().numTensors;

    const run = useKitchenStore.getState().letItLearn();
    // Past the flush and into learnStack: its model and batches now exist, and
    // the fit that would take about 20 seconds to finish is under way.
    await vi.waitFor(
      () => expect(tf.memory().numTensors).toBeGreaterThan(before),
      { timeout: 60000, interval: 5 },
    );
    expect(useKitchenStore.getState().learning).toBe(true);

    const resetAt = performance.now();
    useKitchenStore.getState().reset();
    await run;
    // Cancelled at the next batch (measured at about a quarter of a second),
    // rather than left to train to the end for an answer nobody will read.
    expect(performance.now() - resetAt).toBeLessThan(8000);

    const state = useKitchenStore.getState();
    expect(state.learning).toBe(false);
    expect(state.learned).toBeNull();
    expect(state.learnedShape).toBeNull();
    // The cancellation is not reported as an error, and no learned card lands.
    // (The reset's own rescore may already have replaced the briefing.)
    expect(state.whyCard?.key).not.toMatch(/^(learned|error)-/);
    expect(state.layers[0]!.kernels[0]!.label).toBe("Blur");

    // And the cancelled fit freed everything it had allocated.
    await useKitchenStore.getState().score_();
    expect(tf.memory().numTensors).toBe(before);
  }, 300000);

  it("credits the code lane for a serve only when the code lane built it", async () => {
    // The code tab is open, but the stack was built with the visual controls.
    useKitchenStore.getState().setLane("code");
    useKitchenStore.getState().applyPreset(0, 0, "Vertical edge");
    useKitchenStore.getState().addLayer();
    await useKitchenStore.getState().score_();
    expect(useKitchenStore.getState().phase).toBe("served");
    expect(useProgression.getState().games[SLUG]?.completed).toBe(true);
    expect(useProgression.getState().games[SLUG]?.codeLaneCleared).toBe(false);

    // Now build the same stack through the api, from the visual tab.
    useKitchenStore.getState().setLane("visual");
    const api = createCodeApi();
    await api.reset();
    await api.applyPreset(0, 0, "Vertical edge");
    await api.addLayer();
    expect(useKitchenStore.getState().phase).toBe("served");
    expect(api.score().accuracy).toBeGreaterThan(TARGET_ACCURACY);
    expect(useProgression.getState().games[SLUG]?.codeLaneCleared).toBe(true);
  }, 600000);
});

describe("the code lane's setters", () => {
  beforeEach(() => {
    useKitchenStore.setState({
      layers: [makeLayer([makeKernel("Blur")], "avg")],
      evaluation: null,
      learned: null,
      learnedShape: null,
      scoring: false,
      learning: false,
      phase: "designing",
      failure: null,
    });
  });

  it("throw by name on anything the steppers could not have asked for", () => {
    const api = createCodeApi();
    expect(() => api.applyPreset(0, 0, "Sobel")).toThrow(/Unknown preset "Sobel"/);
    expect(() => api.addKernel(0, "Sobel")).toThrow(/Unknown preset "Sobel"/);
    expect(() => api.applyPreset(3, 0, "Blur")).toThrow(/no layer 3/);
    expect(() => api.applyPreset(0, 5, "Blur")).toThrow(/no filter 5/);
    expect(() => api.setWeights(0, 0, [1, 1, 1, 1, Number.NaN, 1, 1, 1, 1])).toThrow(
      /must be numbers/,
    );
    expect(() => api.removeKernel(0, 0)).toThrow(/at least one filter/);
    expect(() => api.removeLayer(0)).toThrow(/at least one layer/);
    expect(() => api.maps(Number.NaN)).toThrow(/image index/);
    // Nothing reached the store.
    expect(useKitchenStore.getState().layers[0]!.kernels[0]!.label).toBe("Blur");
    expect(useKitchenStore.getState().layers).toHaveLength(1);
  });

  it("refuse to overfill a layer or the stack, as the buttons do", () => {
    const api = createCodeApi();
    for (let count = 1; count < MAX_KERNELS_PER_LAYER; count += 1) {
      void api.addKernel(0, "Vertical edge");
    }
    expect(() => api.addKernel(0, "Vertical edge")).toThrow(/full/);
    // Four in layer 1, and layer 2 arrives with its default pair: six of six.
    void api.addLayer();
    expect(() => api.addLayer()).toThrow(/At most 2 layers/);
    expect(filtersUsed(useKitchenStore.getState().layers)).toBe(FILTER_BUDGET);
    expect(() => api.addKernel(1, "Pass through")).toThrow(/filters are spent/);
  });

  it("refuse to edit while Let it learn has the kitchen", () => {
    useKitchenStore.setState({ learning: true });
    expect(() => createCodeApi().applyPreset(0, 0, "Vertical edge")).toThrow(
      /busy with "Let it learn"/,
    );
    useKitchenStore.setState({ learning: false });
  });

  it("cap a learned shape at the same two layers as everything else", async () => {
    await expect(
      createCodeApi().learn([{ kernels: 1 }, { kernels: 1 }, { kernels: 1 }]),
    ).rejects.toThrow(/At most 2 layers/);
  });
});
