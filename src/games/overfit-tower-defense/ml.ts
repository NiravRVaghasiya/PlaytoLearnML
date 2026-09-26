import * as tf from "@tensorflow/tfjs";
import type { NamedFailure } from "@/engine/types";
import { clamp, gaussian, seededRandom } from "@/lib/utils";

/**
 * Overfit Tower Defense — the real machine learning.
 *
 * The player sets model complexity and places regularization towers; a real TF.js
 * model is built from exactly those settings and trained on that wave's data. The
 * two things attacking the core are not invented enemies with hit points — they
 * ARE the two error terms:
 *
 *   variance / overfitting  = train accuracy − validation accuracy   (the gap)
 *   bias / underfitting     = achievable accuracy − train accuracy
 *
 * Those are orthogonal by construction, which is what makes the lesson teachable:
 *
 *   - A model that memorises reaches ~100% train accuracy, ABOVE the achievable
 *     ceiling, so its bias term is zero and only the gap punishes it.
 *   - A model that is too constrained cannot reach the ceiling even on data it
 *     trained on, so its bias term is large while its gap is ~0.
 *   - A model that generalises sits at the ceiling on both, so both terms are ~0.
 *
 * So "which enemy is killing me" is a real diagnosis, and the two failures have
 * opposite fixes. That is the entire game.
 *
 * ── Why the target moves ────────────────────────────────────────────────────
 * Each wave changes the TRAINING SET SIZE and the LABEL NOISE, not the enemy
 * mix. The optimal amount of regularization genuinely depends on how much data
 * you have and how noisy it is, so the correct answer moves between waves for a
 * real reason rather than because the game decided to move the goalposts. Wave 5
 * hands back a large clean dataset specifically to punish anyone who left wave
 * 4's heavy regularization switched on.
 */

// ── The data ───────────────────────────────────────────────────────────────

/** Total input features. Only the first two carry any signal. */
/**
 * Six features, two of them real.
 *
 * Started at eight (six noise) and measured: with that many distractors the
 * starved waves became unlearnable rather than merely hard — validation accuracy
 * sat at 0.52 for every complexity and every tower combination, so
 * regularization could only lower training accuracy and never raise validation
 * accuracy. That is the exact opposite of the lesson. Four distractors keep L1's
 * feature-selection job real while leaving the task solvable from ~90 points.
 */
export const FEATURE_COUNT = 6;
export const INFORMATIVE_FEATURES = 2;
/** The other four are pure noise — which is what gives L1 something to do. */
export const NOISE_FEATURES = FEATURE_COUNT - INFORMATIVE_FEATURES;

export const VALIDATION_POINTS = 500;

/** Radius² splitting [-1,1]² into equal halves, so classes are balanced. */
const CIRCLE_R2 = 2 / Math.PI;

export interface Sample {
  features: number[];
  label: 0 | 1;
}

export interface Dataset {
  train: Sample[];
  validation: Sample[];
  /**
   * The best accuracy any model can expect, given the label noise. Used as the
   * reference for bias: falling short of this on your OWN training data is
   * underfitting.
   */
  achievable: number;
}

function sample(random: () => number, noiseRate: number): Sample {
  const x0 = random() * 2 - 1;
  const x1 = random() * 2 - 1;
  // A circle, so complexity genuinely matters: no linear model can draw it.
  const clean: 0 | 1 = x0 * x0 + x1 * x1 < CIRCLE_R2 ? 1 : 0;

  const features = [x0, x1];
  for (let index = 0; index < NOISE_FEATURES; index += 1) {
    features.push(gaussian(random) * 0.6);
  }

  return {
    features,
    label: random() < noiseRate ? ((1 - clean) as 0 | 1) : clean,
  };
}

export function generateDataset(
  trainPoints: number,
  noiseRate: number,
  seed: number,
): Dataset {
  const trainRandom = seededRandom(seed);
  const validationRandom = seededRandom(seed + 8809);

  const train: Sample[] = [];
  for (let index = 0; index < trainPoints; index += 1) {
    train.push(sample(trainRandom, noiseRate));
  }

  // Validation carries the SAME label noise. Scoring against clean labels would
  // flatter every model and make the gap an artefact of the measurement.
  const validation: Sample[] = [];
  for (let index = 0; index < VALIDATION_POINTS; index += 1) {
    validation.push(sample(validationRandom, noiseRate));
  }

  return { train, validation, achievable: 1 - noiseRate };
}

// ── The player's controls ──────────────────────────────────────────────────

export const MIN_COMPLEXITY = 1;
export const MAX_COMPLEXITY = 32;
export const DEFAULT_COMPLEXITY = 12;

export type TowerType = "l1" | "l2" | "dropout";

export const TOWER_TYPES: readonly TowerType[] = ["l1", "l2", "dropout"] as const;

export interface Tower {
  type: TowerType;
  /** Number of stacked towers of this type. */
  strength: number;
}

/**
 * Six of each type. Not a limit on over-defending — six of everything is plenty
 * to starve the model into underfitting, which is the real punishment for it —
 * but the point where dropout reaches its 0.72 ceiling, and one pip per tower
 * still fits the palette.
 */
export const MAX_TOWERS_PER_TYPE = 6;

/** Per-tower contribution. Tuned by measurement — see the probe in the tests. */
export const L1_PER_TOWER = 0.004;
export const L2_PER_TOWER = 0.012;
export const DROPOUT_PER_TOWER = 0.12;

export interface Regularization {
  l1: number;
  l2: number;
  dropout: number;
}

export function regularizationOf(towers: Tower[]): Regularization {
  const count = (type: TowerType) =>
    towers
      .filter((tower) => tower.type === type)
      .reduce((total, tower) => total + tower.strength, 0);

  return {
    l1: count("l1") * L1_PER_TOWER,
    l2: count("l2") * L2_PER_TOWER,
    // Above ~0.7 a dropout layer stops training rather than regularising it.
    dropout: Math.min(count("dropout") * DROPOUT_PER_TOWER, 0.72),
  };
}

export function totalTowers(towers: Tower[]): number {
  return towers.reduce((total, tower) => total + tower.strength, 0);
}

// ── The model ─────────────────────────────────────────────────────────────

export const TRAIN_EPOCHS = 120;
export const TRAIN_BATCH = 32;
export const LEARNING_RATE = 0.01;
/** Fixed so the same build always trains the same way. */
export const MODEL_SEED = 4242;

/** Both accuracies are re-measured this often while a model trains. */
export const SAMPLE_EVERY = 10;

/** How a fit ended, read from the last epoch `useModel`'s `train` resolved with. */
export type FitEnd =
  | { kind: "complete" }
  | { kind: "stopped"; epochsRun: number }
  | { kind: "failed" };

/**
 * Null means the fit threw or never ran — `useModel` reports why in its
 * `error`. That is not the player pressing Stop, and calling it "stopped at
 * epoch 0" would tell them they did something they did not.
 */
export function fitEnd(last: { epoch: number } | null): FitEnd {
  if (last === null) return { kind: "failed" };
  const epochsRun = last.epoch + 1;
  return epochsRun < TRAIN_EPOCHS
    ? { kind: "stopped", epochsRun }
    : { kind: "complete" };
}

/**
 * Build the model the player configured.
 *
 * `complexity` is the hidden width, and the regularizers come straight from the
 * towers. There is no fudge factor anywhere in here: if the player's settings are
 * bad, the numbers get worse on their own.
 */
export function buildModel(
  complexity: number,
  towers: Tower[],
  seed = MODEL_SEED,
): tf.LayersModel {
  const units = clamp(Math.round(complexity), MIN_COMPLEXITY, MAX_COMPLEXITY);
  const { l1, l2, dropout } = regularizationOf(towers);

  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      units,
      activation: "relu",
      inputShape: [FEATURE_COUNT],
      kernelInitializer: tf.initializers.heNormal({ seed }),
      // Small positive bias: relu units initialised into their flat region get
      // no gradient and never recover, which would make added capacity do
      // nothing and break the complexity axis of this game.
      biasInitializer: tf.initializers.constant({ value: 0.05 }),
      kernelRegularizer:
        l1 > 0 || l2 > 0 ? tf.regularizers.l1l2({ l1, l2 }) : undefined,
    }),
  );

  if (dropout > 0) {
    model.add(tf.layers.dropout({ rate: dropout, seed }));
  }

  model.add(
    tf.layers.dense({
      units: 1,
      activation: "sigmoid",
      kernelInitializer: tf.initializers.glorotNormal({ seed: seed + 17 }),
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

/** Trainable parameter count — the honest measure of "how big is this model". */
export function parameterCount(complexity: number): number {
  const units = clamp(Math.round(complexity), MIN_COMPLEXITY, MAX_COMPLEXITY);
  return units * FEATURE_COUNT + units + units + 1;
}

export function toMatrix(samples: Sample[]): { xs: number[][]; ys: number[] } {
  return {
    xs: samples.map((item) => item.features),
    ys: samples.map((item) => item.label),
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

// ── The waves ─────────────────────────────────────────────────────────────

export interface Wave {
  index: number;
  name: string;
  /** How much training data this wave gives you. */
  trainPoints: number;
  /** Label noise in this wave's data. */
  noiseRate: number;
  /** How hard the gap is punished this wave. */
  overfitEnemies: number;
  /** How hard the bias is punished this wave. */
  underfitEnemies: number;
  /** What the player should notice. Shown before the wave. */
  briefing: string;
}

/**
 * Five waves. The data changes, so the right answer changes with it.
 *
 * The shape of the run is deliberate: waves 3 and 4 starve the model until
 * regularization is the only way through, and wave 5 then hands back plenty of
 * clean data so that the same towers now cause underfitting. A player who never
 * dials anything back loses on the last wave, which is the point.
 *
 * ── Why the starved waves still field underfit attackers ────────────────────
 * With one underfit attacker against four overfit ones, the cheapest way through
 * the famine was to destroy the model: a network that answers 50% everywhere has
 * no gap at all, and its bias cost a quarter of what the best-generalising model
 * paid in gap (15 damage against 32). The game rewarded a coin flip over the
 * model with the best validation accuracy — the opposite of its lesson.
 *
 * So waves 3 and 4 carry two and three underfit attackers. Measured over 84
 * configurations per wave, the least-damage model on every wave is now within a
 * point of the best validation accuracy, and every coin-flip model takes more
 * damage than the best real one. The overfit attackers still outnumber them,
 * because scarce data is where memorising is the danger.
 *
 * ── Damage is the two error terms, not validation accuracy alone ─────────────
 * By design, and the copy says so: the overfit attackers feed on the gap even
 * when validation accuracy holds still, so two models with the SAME validation
 * accuracy can take about 20 HP apart on the famine wave. Measured at
 * complexity 12: no towers, train 0.956 / validation 0.592, takes 57.7; every
 * tower up, 0.733 / 0.590, takes 36.7. The first one memorised its mislabelled
 * points — training accuracy above the 0.88 ceiling — and that is charged as
 * gap whether or not it cost anything on validation.
 *
 * A pure validation-shortfall formula was measured on the same grid and
 * rejected. Making an unregularised 32-unit model lose while best play keeps
 * the 43 HP the second star needs takes at least 5 HP per validation point on
 * some wave (this formula is under 2), and even then best play clears the star
 * line by 1–9 HP — so the 1–3 points of backend drift would decide runs.
 */
export const WAVES: readonly Wave[] = [
  {
    index: 1,
    name: "Skirmish",
    trainPoints: 400,
    noiseRate: 0.05,
    overfitEnemies: 1,
    underfitEnemies: 2,
    briefing:
      "Plenty of clean data. A reasonably sized model generalises here with no help at all — try it before you build anything.",
  },
  {
    index: 2,
    name: "Thinning supply",
    trainPoints: 250,
    noiseRate: 0.08,
    overfitEnemies: 2,
    underfitEnemies: 1,
    briefing:
      "Less data and noisier labels. The same model that was fine last wave now has room to start memorising.",
  },
  {
    index: 3,
    name: "Scarcity",
    trainPoints: 150,
    noiseRate: 0.1,
    overfitEnemies: 3,
    underfitEnemies: 2,
    briefing:
      "150 points, one in ten mislabelled. A big unregularised model will start treating those mistakes as patterns.",
  },
  {
    index: 4,
    name: "Famine",
    trainPoints: 90,
    noiseRate: 0.12,
    overfitEnemies: 4,
    underfitEnemies: 3,
    briefing:
      "90 points and 12% of the labels wrong. This is where an unregularised model hurts most: it memorises the mistakes, and the gap costs more than any other wave — but strangle it and the underfit attackers are waiting too.",
  },
  {
    index: 5,
    name: "The flood",
    trainPoints: 400,
    noiseRate: 0.05,
    overfitEnemies: 1,
    underfitEnemies: 4,
    briefing:
      "Supply lines restored: 400 points, clean labels. Every tower you needed for the famine is now just holding the model back.",
  },
] as const;

export function waveAt(index: number): Wave {
  return WAVES[clamp(index - 1, 0, WAVES.length - 1)] ?? WAVES[0]!;
}

export const CORE_MAX_HP = 100;
/** Scales an error fraction into hit points. */
export const DAMAGE_SCALE = 42;
/** Errors below this are treated as noise in the measurement, not damage. */
export const DAMAGE_DEADZONE = 0.02;

// ── Scoring a wave ────────────────────────────────────────────────────────

export type Outcome = "win" | "overfitting" | "underfitting" | "untrained";

export interface WaveResult {
  wave: number;
  trainAccuracy: number;
  validationAccuracy: number;
  achievable: number;
  /** train − validation. Variance. */
  gap: number;
  /** achievable − train. Bias. */
  bias: number;
  overfitDamage: number;
  underfitDamage: number;
  damage: number;
  /** Which error term did more harm. */
  dominant: "overfit" | "underfit" | "balanced";
  survived: boolean;
}

export interface ScoreWaveInput {
  wave: Wave;
  trainAccuracy: number;
  validationAccuracy: number;
  achievable: number;
  coreHp: number;
}

export function scoreWave({
  wave,
  trainAccuracy,
  validationAccuracy,
  achievable,
  coreHp,
}: ScoreWaveInput): WaveResult {
  // Both clamped at zero: a model can land above the achievable ceiling by luck,
  // and negative "damage" would heal the core, which would be nonsense.
  const gap = Math.max(0, trainAccuracy - validationAccuracy);
  const bias = Math.max(0, achievable - trainAccuracy);

  const overfitDamage =
    wave.overfitEnemies * Math.max(0, gap - DAMAGE_DEADZONE) * DAMAGE_SCALE;
  const underfitDamage =
    wave.underfitEnemies * Math.max(0, bias - DAMAGE_DEADZONE) * DAMAGE_SCALE;

  const damage = overfitDamage + underfitDamage;
  const margin = Math.max(overfitDamage, underfitDamage) * 0.25;

  return {
    wave: wave.index,
    trainAccuracy,
    validationAccuracy,
    achievable,
    gap,
    bias,
    overfitDamage,
    underfitDamage,
    damage,
    dominant:
      Math.abs(overfitDamage - underfitDamage) <= margin
        ? "balanced"
        : overfitDamage > underfitDamage
          ? "overfit"
          : "underfit",
    survived: coreHp - damage > 0,
  };
}

/**
 * Which failure to NAME for a wave, given which error did the harm.
 *
 * A "balanced" wave is still one of the two. Falling through to overfitting on
 * a tie told players with eighteen towers up and bias as the larger term to add
 * regularization — the opposite of the fix. The larger damage term decides.
 */
export function failureSide(result: WaveResult): "overfit" | "underfit" {
  if (result.dominant !== "balanced") return result.dominant;
  return result.underfitDamage > result.overfitDamage ? "underfit" : "overfit";
}

export interface Evaluation {
  outcome: Outcome;
  /** 0–1, from surviving core HP and how many waves were cleared. */
  score: number;
  wavesCleared: number;
  coreHp: number;
  failure: NamedFailure | null;
}

/** Surviving matters more than surviving prettily, hence the weighting. */
export const WAVES_WEIGHT = 0.65;
export const HP_WEIGHT = 0.35;

/**
 * The run score: waves cleared, then core health.
 *
 * HP is counted in the whole points the core readout shows. Mastery's second
 * star is a threshold on this score, and scoring 42.85 HP — displayed as "43" —
 * below a star text that asks for 43 would be a promise broken by rounding.
 */
export function runScore(wavesCleared: number, coreHp: number): number {
  const waveFraction = clamp(wavesCleared, 0, WAVES.length) / WAVES.length;
  const hpFraction = clamp(Math.round(coreHp) / CORE_MAX_HP, 0, 1);
  return clamp(waveFraction * WAVES_WEIGHT + hpFraction * HP_WEIGHT, 0, 1);
}

/** Core HP a finished run needs for its score to reach `threshold`. */
export function hpForScore(threshold: number): number {
  return Math.ceil(((threshold - WAVES_WEIGHT) / HP_WEIGHT) * CORE_MAX_HP);
}

const percent = (value: number) => `${Math.round(value * 100)}%`;

/**
 * Name the failure by which error term actually did the damage.
 *
 * The two names carry OPPOSITE advice, so getting this wrong is worse than
 * saying nothing: telling an underfitting player to add regularization would
 * push them further from a working model.
 */
export function evaluateRun({
  result,
  coreHp,
  wavesCleared,
  complexity,
  towers,
  finished,
}: {
  result: WaveResult | null;
  coreHp: number;
  wavesCleared: number;
  complexity: number;
  towers: Tower[];
  finished: boolean;
}): Evaluation {
  if (result === null) {
    return {
      outcome: "untrained",
      score: 0,
      wavesCleared,
      coreHp,
      failure: null,
    };
  }

  const score = runScore(wavesCleared, coreHp);

  if (coreHp > 0 && finished) {
    return { outcome: "win", score, wavesCleared, coreHp, failure: null };
  }

  if (coreHp > 0) {
    // Mid-run, still standing: not a failure, just not finished.
    return { outcome: "untrained", score, wavesCleared, coreHp, failure: null };
  }

  const { l1, l2, dropout } = regularizationOf(towers);
  const regDescription = [
    l1 > 0 ? `L1 ${l1.toFixed(3)}` : null,
    l2 > 0 ? `L2 ${l2.toFixed(3)}` : null,
    dropout > 0 ? `dropout ${dropout.toFixed(2)}` : null,
  ]
    .filter(Boolean)
    .join(", ");

  if (failureSide(result) === "underfit") {
    return {
      outcome: "underfitting",
      score,
      wavesCleared,
      coreHp,
      failure: {
        name: "Underfitting",
        detail: `The core fell to the underfit wave. Your model managed only ${percent(
          result.trainAccuracy,
        )} on the data it trained on, when ${percent(
          result.achievable,
        )} was achievable — it is not memorising anything, it cannot even fit what it was shown. ${
          regDescription === ""
            ? `${complexity} hidden units is too small a model for this shape.`
            : `${regDescription} across ${totalTowers(
                towers,
              )} towers is holding it too tightly. Take towers down, or raise complexity.`
        } The train/validation gap was only ${percent(
          result.gap,
        )}, so overfitting was never your problem here.`,
      },
    };
  }

  return {
    outcome: "overfitting",
    score,
    wavesCleared,
    coreHp,
    failure: {
      name: "Overfitting",
      detail: `The core fell to the overfit wave. ${percent(
        result.trainAccuracy,
      )} on the ${
        result.trainAccuracy > result.achievable ? "training data — above the " : "training data, against a "
      }${percent(result.achievable)} ceiling — but only ${percent(
        result.validationAccuracy,
      )} on data it had never seen, a gap of ${percent(result.gap)}. ${
        parameterCount(complexity)
      } parameters memorised the mislabelled points. ${
        regDescription === ""
          ? "No regularization towers are up: L2 or dropout would cost you a little training accuracy and buy back much more validation accuracy."
          : `${regDescription} was not enough for this little data.`
      }`,
    },
  };
}

// ── Reveal the math ───────────────────────────────────────────────────────

export const MATH_EQUATION = String.raw`\text{minimise}\quad \underbrace{\frac{1}{n}\sum_{i=1}^{n} \mathcal{L}\big(f(x_i),\,y_i\big)}_{\text{fit the data}} \;+\; \underbrace{\lambda_1\lVert W\rVert_1 \;+\; \lambda_2\lVert W\rVert_2^2}_{\text{towers: keep it simple}}
\\[1.4em]
\text{gap} = \mathrm{acc}_{\text{train}} - \mathrm{acc}_{\text{val}}
\qquad
\text{bias} = \mathrm{acc}_{\text{achievable}} - \mathrm{acc}_{\text{train}}`;

export const MATH_CODE = `// Your complexity slider and your towers, handed to TensorFlow.js.
export function buildModel(complexity, towers) {
  const { l1, l2, dropout } = regularizationOf(towers);
  const model = tf.sequential();

  model.add(tf.layers.dense({
    units: complexity,              // <- the complexity slider
    activation: 'relu',
    inputShape: [${FEATURE_COUNT}],
    // L1 and L2 towers become a penalty on the weights themselves:
    kernelRegularizer: tf.regularizers.l1l2({ l1, l2 }),
  }));

  // Dropout towers switch off a fraction of units on every training step,
  // so no single unit can become load-bearing.
  if (dropout > 0) {
    model.add(tf.layers.dropout({ rate: dropout }));
  }

  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({
    optimizer: tf.train.adam(${LEARNING_RATE}),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
}`;

export const MATH_NOTES = `The first term wants to fit your ${FEATURE_COUNT} features to the labels. The second term wants the weights to stay small, and does not care about the labels at all. Regularization is that tug-of-war: λ decides who wins.

L2 (‖W‖²) pushes every weight toward zero with a force that weakens as the weight shrinks. L1 (‖W‖₁) pushes with the same force however small the weight already is, so weights that aren't earning their place are driven close to zero — which is why L1 quiets the ${NOISE_FEATURES} pure-noise features far more than L2 does. Close to zero, not exactly: an exact zero needs a solver that can land on it, and Adam keeps stepping back and forth across it instead.

The two readouts underneath are the diagnosis. A gap means the model learned things about your training set that were not true of the world. Bias means it never learned the training set in the first place. Both go up when you get it wrong, and they go up in response to opposite mistakes — which is why one meter could never tell you what to do next.`;
