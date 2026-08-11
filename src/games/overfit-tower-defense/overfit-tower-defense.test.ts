import * as tf from "@tensorflow/tfjs";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import {
  CORE_MAX_HP,
  DAMAGE_SCALE,
  FEATURE_COUNT,
  INFORMATIVE_FEATURES,
  MAX_COMPLEXITY,
  MAX_TOWERS_PER_TYPE,
  MODEL_SEED,
  NOISE_FEATURES,
  TRAIN_BATCH,
  TRAIN_EPOCHS,
  VALIDATION_POINTS,
  WAVES,
  accuracyFromPredictions,
  buildModel,
  evaluateRun,
  generateDataset,
  parameterCount,
  regularizationOf,
  scoreWave,
  toMatrix,
  waveAt,
  type Tower,
  type Wave,
} from "./ml";
import {
  biasOf,
  gapOf,
  towerCount,
  useTowerDefenseStore,
} from "./store";

beforeAll(async () => {
  await tf.ready();
});

const DATA_SEED = 5150;

const towers = (l1: number, l2: number, dropout: number): Tower[] =>
  [
    { type: "l1" as const, strength: l1 },
    { type: "l2" as const, strength: l2 },
    { type: "dropout" as const, strength: dropout },
  ].filter((tower) => tower.strength > 0);

interface Measured {
  trainAccuracy: number;
  validationAccuracy: number;
  gap: number;
  bias: number;
  /** Mean |weight| on the noise-feature rows of the input layer. */
  noiseWeight: number;
}

/** Train exactly as the game does, then measure both sides. */
async function measure(
  wave: Wave,
  complexity: number,
  towerList: Tower[],
): Promise<Measured> {
  const data = generateDataset(wave.trainPoints, wave.noiseRate, DATA_SEED);
  const train = toMatrix(data.train);
  const validation = toMatrix(data.validation);

  const model = buildModel(complexity, towerList, MODEL_SEED);
  const xs = tf.tensor2d(train.xs);
  const ys = tf.tensor2d(train.ys.map((y) => [y]));
  await model.fit(xs, ys, {
    epochs: TRAIN_EPOCHS,
    batchSize: TRAIN_BATCH,
    shuffle: false,
    verbose: 0,
  });

  const score = (rows: number[][], labels: number[]) =>
    tf.tidy(() => {
      const input = tf.tensor2d(rows);
      return accuracyFromPredictions(
        (model.predict(input) as tf.Tensor).dataSync(),
        labels,
      );
    });

  const trainAccuracy = score(train.xs, train.ys);
  const validationAccuracy = score(validation.xs, validation.ys);

  // Input kernel is [FEATURE_COUNT, units]. Rows past the informative ones are
  // the distractors — how much weight survives on them is what L1 should change.
  const noiseWeight = tf.tidy(() => {
    const kernel = model.getWeights()[0]!;
    const rows = kernel.arraySync() as number[][];
    const noiseRows = rows.slice(INFORMATIVE_FEATURES);
    const total = noiseRows.flat().reduce((sum, w) => sum + Math.abs(w), 0);
    return total / Math.max(1, noiseRows.flat().length);
  });

  xs.dispose();
  ys.dispose();
  const optimizer = model.optimizer;
  model.dispose();
  optimizer?.dispose();

  return {
    trainAccuracy,
    validationAccuracy,
    gap: Math.max(0, trainAccuracy - validationAccuracy),
    bias: Math.max(0, data.achievable - trainAccuracy),
    noiseWeight,
  };
}

describe("the data", () => {
  it("is deterministic for a seed", () => {
    const a = generateDataset(80, 0.1, DATA_SEED);
    const b = generateDataset(80, 0.1, DATA_SEED);
    expect(a.train).toEqual(b.train);
    expect(a.validation).toEqual(b.validation);
  });

  it("holds validation out of training", () => {
    const { train, validation } = generateDataset(90, 0.12, DATA_SEED);
    expect(validation).toHaveLength(VALIDATION_POINTS);
    const seen = new Set(train.map((s) => s.features.join(",")));
    expect(validation.filter((s) => seen.has(s.features.join(",")))).toHaveLength(
      0,
    );
  });

  it("puts the same label noise in validation as in training", () => {
    // Scoring against clean labels would flatter every model and make the gap an
    // artefact of the measurement rather than a property of the model.
    const { validation, achievable } = generateDataset(90, 0.2, DATA_SEED);
    expect(achievable).toBeCloseTo(0.8, 5);
    const flipped = validation.filter((sample) => {
      const [x0, x1] = sample.features as [number, number];
      const clean = x0 * x0 + x1 * x1 < 2 / Math.PI ? 1 : 0;
      return clean !== sample.label;
    });
    // Roughly the stated rate, allowing for sampling noise at n=500.
    expect(flipped.length / validation.length).toBeGreaterThan(0.14);
    expect(flipped.length / validation.length).toBeLessThan(0.26);
  });

  it("keeps the classes balanced, so accuracy means something", () => {
    for (const wave of WAVES) {
      const { validation } = generateDataset(
        wave.trainPoints,
        wave.noiseRate,
        DATA_SEED,
      );
      const positive =
        validation.filter((s) => s.label === 1).length / validation.length;
      expect(positive, `wave ${wave.index}`).toBeGreaterThan(0.4);
      expect(positive, `wave ${wave.index}`).toBeLessThan(0.6);
    }
  });

  it("carries mostly distractor features", () => {
    const { train } = generateDataset(40, 0.1, DATA_SEED);
    expect(train[0]!.features).toHaveLength(FEATURE_COUNT);
    expect(NOISE_FEATURES).toBeGreaterThan(INFORMATIVE_FEATURES);
  });
});

describe("the two error terms are orthogonal", () => {
  // This is the load-bearing property of the whole game: if a memoriser also
  // registered as underfitting, the named failures could not tell the player
  // which way to move, and the two have OPPOSITE fixes.

  it("gives a memoriser a large gap and zero bias", () => {
    const result = scoreWave({
      wave: waveAt(4),
      trainAccuracy: 1,
      validationAccuracy: 0.62,
      achievable: 0.88,
      coreHp: CORE_MAX_HP,
    });
    expect(result.gap).toBeCloseTo(0.38, 5);
    expect(result.bias).toBe(0);
    expect(result.dominant).toBe("overfit");
    expect(result.underfitDamage).toBe(0);
  });

  it("gives an over-constrained model large bias and no gap", () => {
    const result = scoreWave({
      wave: waveAt(5),
      trainAccuracy: 0.65,
      validationAccuracy: 0.65,
      achievable: 0.95,
      coreHp: CORE_MAX_HP,
    });
    expect(result.gap).toBe(0);
    expect(result.bias).toBeCloseTo(0.3, 5);
    expect(result.dominant).toBe("underfit");
    expect(result.overfitDamage).toBe(0);
  });

  it("gives a model that generalises almost no damage of either kind", () => {
    const result = scoreWave({
      wave: waveAt(1),
      trainAccuracy: 0.93,
      validationAccuracy: 0.92,
      achievable: 0.95,
      coreHp: CORE_MAX_HP,
    });
    expect(result.damage).toBeLessThan(2);
    expect(result.survived).toBe(true);
  });

  it("never heals the core when a model beats its ceiling", () => {
    const result = scoreWave({
      wave: waveAt(1),
      trainAccuracy: 1,
      validationAccuracy: 1,
      achievable: 0.95,
      coreHp: 50,
    });
    expect(result.damage).toBeGreaterThanOrEqual(0);
    expect(result.bias).toBe(0);
  });
});

describe("naming the failure", () => {
  const base = {
    coreHp: 0,
    wavesCleared: 3,
    complexity: 32,
    finished: false,
  };

  it("names Overfitting when the gap did the damage, and points at towers", () => {
    const evaluation = evaluateRun({
      ...base,
      towers: [],
      result: scoreWave({
        wave: waveAt(4),
        trainAccuracy: 1,
        validationAccuracy: 0.62,
        achievable: 0.88,
        coreHp: 10,
      }),
    });
    expect(evaluation.outcome).toBe("overfitting");
    expect(evaluation.failure?.name).toBe("Overfitting");
    expect(evaluation.failure?.detail).toMatch(/L2 or dropout/i);
    expect(evaluation.failure?.detail).toMatch(/\d+%/);
  });

  it("names Underfitting when bias did the damage, and points the other way", () => {
    const evaluation = evaluateRun({
      ...base,
      towers: towers(6, 6, 6),
      result: scoreWave({
        wave: waveAt(5),
        trainAccuracy: 0.65,
        validationAccuracy: 0.64,
        achievable: 0.95,
        coreHp: 10,
      }),
    });
    expect(evaluation.outcome).toBe("underfitting");
    expect(evaluation.failure?.name).toBe("Underfitting");
    // The advice must be the OPPOSITE of the overfitting advice.
    expect(evaluation.failure?.detail).toMatch(/Take towers down|raise complexity/i);
    expect(evaluation.failure?.detail).not.toMatch(/add regulari[sz]ation/i);
  });

  it("tells an underfitting player that overfitting was not the problem", () => {
    const evaluation = evaluateRun({
      ...base,
      towers: towers(0, 6, 0),
      result: scoreWave({
        wave: waveAt(5),
        trainAccuracy: 0.66,
        validationAccuracy: 0.65,
        achievable: 0.95,
        coreHp: 5,
      }),
    });
    expect(evaluation.failure?.detail).toMatch(/overfitting was never your problem/i);
  });

  it("reports no failure while the core still stands", () => {
    const evaluation = evaluateRun({
      ...base,
      coreHp: 40,
      towers: [],
      result: scoreWave({
        wave: waveAt(2),
        trainAccuracy: 0.9,
        validationAccuracy: 0.85,
        achievable: 0.92,
        coreHp: 40,
      }),
    });
    expect(evaluation.failure).toBeNull();
    expect(evaluation.outcome).not.toBe("overfitting");
  });

  it("wins only once every wave is behind you", () => {
    const won = evaluateRun({
      ...base,
      coreHp: 55,
      wavesCleared: WAVES.length,
      finished: true,
      towers: towers(0, 1, 0),
      result: scoreWave({
        wave: waveAt(5),
        trainAccuracy: 0.93,
        validationAccuracy: 0.92,
        achievable: 0.95,
        coreHp: 55,
      }),
    });
    expect(won.outcome).toBe("win");
    expect(won.score).toBeGreaterThan(0.5);
  });
});

describe("towers", () => {
  it("sums stacked towers into real regularizer strengths", () => {
    const reg = regularizationOf(towers(2, 3, 1));
    expect(reg.l1).toBeCloseTo(0.008, 6);
    expect(reg.l2).toBeCloseTo(0.036, 6);
    expect(reg.dropout).toBeCloseTo(0.12, 6);
  });

  it("caps dropout below the point where it stops training the model", () => {
    expect(regularizationOf(towers(0, 0, 99)).dropout).toBeLessThanOrEqual(0.72);
  });

  it("counts parameters honestly", () => {
    // units * inputs + units biases + units output weights + 1 output bias
    expect(parameterCount(1)).toBe(FEATURE_COUNT + 3);
    expect(parameterCount(MAX_COMPLEXITY)).toBeGreaterThan(parameterCount(8));
  });
});

describe("the waves move the target", () => {
  it("escalates scarcity, then hands the data back", () => {
    const sizes = WAVES.map((wave) => wave.trainPoints);
    // Waves 1-4 get progressively less data...
    for (let index = 1; index < 4; index += 1) {
      expect(sizes[index]!).toBeLessThan(sizes[index - 1]!);
    }
    // ...and wave 5 restores it, which is what makes leftover towers hurt.
    expect(sizes[4]!).toBeGreaterThan(sizes[3]!);
    expect(WAVES[4]!.underfitEnemies).toBeGreaterThan(WAVES[4]!.overfitEnemies);
    expect(WAVES[3]!.overfitEnemies).toBeGreaterThan(WAVES[3]!.underfitEnemies);
  });

  it("punishes leftover regularization on the final wave hard enough to matter", () => {
    // The trap has to be able to actually kill, or the lesson is decorative.
    const stillFortified = scoreWave({
      wave: waveAt(5),
      trainAccuracy: 0.68,
      validationAccuracy: 0.654,
      achievable: 0.95,
      coreHp: CORE_MAX_HP,
    });
    expect(stillFortified.underfitDamage).toBeGreaterThan(30);
    expect(stillFortified.dominant).toBe("underfit");
  });
});

describe("the bias-variance tradeoff, measured", () => {
  // These train real models. The numbers in the comments are what was measured
  // while tuning; the assertions are loose enough to survive small drift but
  // tight enough that losing the lesson fails the build.

  it("goes from underfitting to overfitting as complexity rises", async () => {
    const wave = waveAt(1); // 400 points, 5% noise, ceiling 0.95
    const tiny = await measure(wave, 1, []); // measured bias 0.390
    const right = await measure(wave, 4, []); // measured bias 0.065, gap 0.027
    const big = await measure(wave, MAX_COMPLEXITY, []); // gap 0.078, bias 0

    expect(tiny.bias).toBeGreaterThan(0.2);
    expect(right.bias).toBeLessThan(tiny.bias);
    expect(big.bias).toBeLessThan(right.bias + 0.02);
    // The gap is what grows on the way up.
    expect(big.gap).toBeGreaterThan(right.gap);
    // And a model too small to fit cannot generalise either.
    expect(tiny.validationAccuracy).toBeLessThan(right.validationAccuracy);
  }, 300000);

  it("overfits badly when data is scarce and nothing restrains it", async () => {
    const wave = waveAt(4); // 90 points, 12% noise
    const bare = await measure(wave, MAX_COMPLEXITY, []);
    // Measured: train 1.000, val 0.620, gap 0.380.
    expect(bare.trainAccuracy).toBeGreaterThan(0.95);
    expect(bare.gap).toBeGreaterThan(0.25);
    expect(bare.bias).toBe(0);
  }, 300000);

  it("trades a little training accuracy for more validation accuracy", async () => {
    // The core intuition, stated as a measurement. If this ever fails, the game
    // is teaching something false.
    const wave = waveAt(5); // 400 points, 5% noise
    const bare = await measure(wave, MAX_COMPLEXITY, []); // train .960 val .882
    const held = await measure(wave, MAX_COMPLEXITY, towers(1, 0, 0)); // .917/.918

    expect(held.trainAccuracy).toBeLessThan(bare.trainAccuracy);
    expect(held.validationAccuracy).toBeGreaterThan(bare.validationAccuracy);
    expect(held.gap).toBeLessThan(bare.gap);
  }, 300000);

  it("helps on the starved wave too, not just the comfortable one", async () => {
    const wave = waveAt(4);
    const bare = await measure(wave, MAX_COMPLEXITY, []); // val 0.620
    const held = await measure(wave, MAX_COMPLEXITY, towers(3, 0, 0)); // val 0.688

    expect(held.validationAccuracy).toBeGreaterThan(bare.validationAccuracy);
    expect(held.gap).toBeLessThan(bare.gap);
  }, 300000);

  it("starves the model into underfitting when over-defended", async () => {
    // "Over-defend and you starve the model into underfitting" — spec.
    const wave = waveAt(5);
    const fortified = await measure(wave, MAX_COMPLEXITY, towers(6, 6, 6));
    // Measured: train 0.680, val 0.654, bias 0.270.
    expect(fortified.bias).toBeGreaterThan(0.15);
    expect(fortified.gap).toBeLessThan(0.06);

    const held = await measure(wave, MAX_COMPLEXITY, towers(1, 0, 0));
    // Too much regularization is worse than the right amount, on BOTH sides.
    expect(fortified.validationAccuracy).toBeLessThan(held.validationAccuracy);
  }, 300000);

  it("makes L1 delete distractor features where L2 only shrinks them", async () => {
    // Justifies the claim in MATH_NOTES. L1's constant-force penalty drives
    // unhelpful weights to exactly zero; L2's shrinks in proportion to size and
    // so rarely arrives.
    const wave = waveAt(4);
    const withL1 = await measure(wave, 16, towers(6, 0, 0));
    const withL2 = await measure(wave, 16, towers(0, 6, 0));
    expect(withL1.noiseWeight).toBeLessThan(withL2.noiseWeight);
  }, 300000);
});

describe("damage tuning", () => {
  it("lets a well-played run survive all five waves", () => {
    // Best-play gaps and biases as measured during tuning.
    const bestPlay: Array<[number, number, number]> = [
      // wave, gap, bias
      [1, 0.056, 0.017],
      [2, 0.06, 0.02],
      [3, 0.08, 0.01],
      [4, 0.212, 0.0],
      [5, 0.0, 0.032],
    ];
    let hp = CORE_MAX_HP;
    for (const [index, gap, bias] of bestPlay) {
      const wave = waveAt(index);
      const data = generateDataset(wave.trainPoints, wave.noiseRate, DATA_SEED);
      const trainAccuracy = data.achievable - bias;
      const result = scoreWave({
        wave,
        trainAccuracy,
        validationAccuracy: trainAccuracy - gap,
        achievable: data.achievable,
        coreHp: hp,
      });
      hp -= result.damage;
    }
    expect(hp).toBeGreaterThan(20);
  });

  it("kills a run that never regularises", () => {
    const neverDefends: Array<[number, number]> = [
      [1, 0.078],
      [2, 0.12],
      [3, 0.285],
      [4, 0.38],
    ];
    let hp = CORE_MAX_HP;
    for (const [index, gap] of neverDefends) {
      const wave = waveAt(index);
      const result = scoreWave({
        wave,
        trainAccuracy: 1,
        validationAccuracy: 1 - gap,
        achievable: 1 - wave.noiseRate,
        coreHp: hp,
      });
      hp -= result.damage;
    }
    expect(hp).toBeLessThanOrEqual(0);
  });

  it("scales damage from the error, with a deadzone for measurement noise", () => {
    expect(DAMAGE_SCALE).toBeGreaterThan(0);
    const clean = scoreWave({
      wave: waveAt(1),
      trainAccuracy: 0.95,
      validationAccuracy: 0.94,
      achievable: 0.95,
      coreHp: CORE_MAX_HP,
    });
    expect(clean.damage).toBe(0);
  });
});

describe("memory", () => {
  it("leaks no tensors building and disposing a model", () => {
    const before = tf.memory().numTensors;
    const model = buildModel(16, towers(1, 1, 1), MODEL_SEED);
    const optimizer = model.optimizer;
    model.dispose();
    optimizer?.dispose();
    expect(tf.memory().numTensors).toBe(before);
  });
});

describe("the store's run flow", () => {
  const store = useTowerDefenseStore;

  beforeEach(() => {
    store.getState().restart();
    store.getState().clearTowers();
    store.getState().setComplexity(12);
  });

  it("starts on wave 1 at full health", () => {
    expect(store.getState().wave).toBe(1);
    expect(store.getState().coreHp).toBe(CORE_MAX_HP);
    expect(store.getState().phase).toBe("tuning");
    expect(store.getState().dataset.train).toHaveLength(WAVES[0]!.trainPoints);
  });

  it("costs the core nothing for a trial", () => {
    // Selecting a configuration on held-out data is the habit being taught; it
    // must not be the thing that kills you.
    store.getState().finishTrial({
      trainAccuracy: 1,
      validationAccuracy: 0.5,
    });
    expect(store.getState().coreHp).toBe(CORE_MAX_HP);
    expect(store.getState().phase).toBe("tuning");
    expect(store.getState().validationAccuracy).toBe(0.5);
  });

  it("takes damage from a gap when a wave is deployed against", () => {
    store.getState().resolveWave({
      trainAccuracy: 1,
      validationAccuracy: 0.6,
    });
    expect(store.getState().coreHp).toBeLessThan(CORE_MAX_HP);
    expect(store.getState().lastResult?.dominant).toBe("overfit");
  });

  it("advances only after a wave has resolved", () => {
    expect(store.getState().phase).toBe("tuning");
    store.getState().nextWave();
    expect(store.getState().wave, "must not skip ahead while tuning").toBe(1);

    store.getState().resolveWave({ trainAccuracy: 0.94, validationAccuracy: 0.93 });
    expect(store.getState().phase).toBe("resolved");
    store.getState().nextWave();
    expect(store.getState().wave).toBe(2);
    expect(store.getState().dataset.train).toHaveLength(WAVES[1]!.trainPoints);
  });

  it("clears stale accuracy when the tuning changes", () => {
    store.getState().finishTrial({ trainAccuracy: 0.9, validationAccuracy: 0.8 });
    expect(store.getState().trainAccuracy).toBe(0.9);

    store.getState().setComplexity(20);
    expect(store.getState().trainAccuracy).toBeNull();
    expect(store.getState().validationAccuracy).toBeNull();
  });

  it("clears stale accuracy when a tower changes", () => {
    store.getState().finishTrial({ trainAccuracy: 0.9, validationAccuracy: 0.8 });
    store.getState().addTower("l2");
    expect(store.getState().trainAccuracy).toBeNull();
  });

  it("caps towers per type", () => {
    for (let index = 0; index < 20; index += 1) store.getState().addTower("l2");
    expect(towerCount(store.getState(), "l2")).toBeLessThanOrEqual(
      MAX_TOWERS_PER_TYPE,
    );
  });

  it("clamps tower counts arriving from the code lane", () => {
    store.getState().setTowers({ l1: 99, l2: -4, dropout: 3 });
    expect(towerCount(store.getState(), "l1")).toBe(MAX_TOWERS_PER_TYPE);
    expect(towerCount(store.getState(), "l2")).toBe(0);
    expect(towerCount(store.getState(), "dropout")).toBe(3);
  });

  it("locks the controls while training", () => {
    store.getState().beginTraining();
    store.getState().setComplexity(31);
    store.getState().addTower("l1");
    expect(store.getState().modelComplexity).toBe(12);
    expect(towerCount(store.getState(), "l1")).toBe(0);
  });

  it("cannot be killed by a single bad wave-1 deployment", () => {
    // Wave 1 fields one overfit attacker, so even a total collapse caps at about
    // 37 damage. The opening wave is meant to be somewhere you can experiment.
    store.getState().resolveWave({ trainAccuracy: 1, validationAccuracy: 0.1 });
    expect(store.getState().coreHp).toBeGreaterThan(0);
    expect(store.getState().phase).toBe("resolved");
  });

  it("ends the run when the core is destroyed, and names the failure", () => {
    // Clear the first three waves cleanly to reach the famine, where four
    // overfit attackers can actually finish the core off.
    for (let wave = 1; wave <= 3; wave += 1) {
      const { achievable } = store.getState().dataset;
      store.getState().resolveWave({
        trainAccuracy: achievable,
        validationAccuracy: achievable,
      });
      store.getState().nextWave();
    }
    expect(store.getState().wave).toBe(4);

    store.getState().resolveWave({ trainAccuracy: 1, validationAccuracy: 0.1 });
    expect(store.getState().coreHp).toBe(0);
    expect(store.getState().phase).toBe("lost");
    expect(store.getState().failure?.name).toBe("Overfitting");
  });

  it("names Underfitting when the final wave punishes leftover towers", () => {
    for (let wave = 1; wave <= 4; wave += 1) {
      const { achievable } = store.getState().dataset;
      store.getState().resolveWave({
        trainAccuracy: achievable,
        validationAccuracy: achievable,
      });
      store.getState().nextWave();
    }
    expect(store.getState().wave).toBe(5);

    // Over-regularised: no gap at all, but nowhere near the ceiling.
    store.getState().resolveWave({ trainAccuracy: 0.3, validationAccuracy: 0.3 });
    expect(store.getState().phase).toBe("lost");
    expect(store.getState().failure?.name).toBe("Underfitting");
  });

  it("wins only after the fifth wave resolves with the core alive", () => {
    for (let wave = 1; wave <= WAVES.length; wave += 1) {
      const { achievable } = store.getState().dataset;
      store.getState().resolveWave({
        trainAccuracy: achievable,
        validationAccuracy: achievable,
      });
      if (wave < WAVES.length) {
        expect(store.getState().phase, `after wave ${wave}`).toBe("resolved");
        store.getState().nextWave();
      }
    }
    expect(store.getState().phase).toBe("won");
    expect(store.getState().wavesCleared).toBe(WAVES.length);
    expect(store.getState().failure).toBeNull();
  });

  it("reports the gap and bias as NaN before anything is measured", () => {
    // So the metric shows "deploy to measure" rather than a confident zero.
    expect(Number.isNaN(gapOf(store.getState()))).toBe(true);
    expect(Number.isNaN(biasOf(store.getState()))).toBe(true);
  });
});
