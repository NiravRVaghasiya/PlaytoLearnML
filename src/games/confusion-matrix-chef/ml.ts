import type { NamedFailure } from "@/engine/types";
import { clamp, gaussian, seededRandom } from "@/lib/utils";

/**
 * Confusion Matrix Chef — the real evaluation maths.
 *
 * This is the one game in the set that does not train anything, and that is the
 * spec's design, not a shortcut: "pre-scored samples; metrics recomputed on
 * threshold change (no retrain needed)". The lesson lives entirely downstream of
 * the model. A classifier hands you a score per case, and *you* still have to
 * choose the cutoff — that choice is not part of training, it is a business
 * decision about which kind of mistake you would rather make.
 *
 * So there is no TF.js here. Adding a model would train something, throw away
 * everything except its scores, and teach nothing extra.
 *
 * ── Where the scores come from ──────────────────────────────────────────────
 * Each case gets a latent logit z, normal with mean ±d/2 by class and unit
 * variance, then score = sigmoid(z). That is the shape a calibrated classifier
 * actually produces, and it has a property worth having: for two unit-variance
 * normals separated by d, the difference of a positive and a negative draw is
 * N(d, 2), so
 *
 *     AUC = P(z⁺ > z⁻) = Φ(d / √2)
 *
 * and because sigmoid is monotone, the AUC of the scores equals the AUC of the
 * logits. That gives a closed form to test the generator against, so "separability
 * 2.2" is a checkable claim rather than a magic number.
 */

export interface Sample {
  /** Model score in (0,1). */
  score: number;
  /** 1 = the thing you are trying to catch. */
  trueLabel: 0 | 1;
}

export interface ConfusionMatrix {
  truePositives: number;
  falsePositives: number;
  trueNegatives: number;
  falseNegatives: number;
}

export interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  /** True-negative rate. The other half of the ROC axes. */
  specificity: number;
  /** Mean of recall and specificity — accuracy's honest cousin. */
  balancedAccuracy: number;
  falsePositiveRate: number;
}

export type MetricKey =
  | "accuracy"
  | "precision"
  | "recall"
  | "f1"
  | "balancedAccuracy";

export const SAMPLE_COUNT = 800;

/** Threshold granularity of the slider. */
export const THRESHOLD_STEP = 0.01;
export const DEFAULT_THRESHOLD = 0.5;

/**
 * Every cutoff the slider can reach, 0.00 to 1.00.
 *
 * Built from integer steps rather than by adding 0.01 a hundred times, so each
 * stop is exactly the number the slider reports (0.07, not 0.07000000000000001).
 */
export const SLIDER_THRESHOLDS: readonly number[] = Array.from(
  { length: Math.round(1 / THRESHOLD_STEP) + 1 },
  (_, step) => step / Math.round(1 / THRESHOLD_STEP),
);

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/** Φ, via the Abramowitz–Stegun erf approximation. Used only to state the AUC. */
export function normalCdf(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const poly =
    t *
    (0.319381530 +
      t *
        (-0.356563782 +
          t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const upperTail = (Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)) * poly;
  return x >= 0 ? 1 - upperTail : upperTail;
}

/** The AUC this separability implies, before any sampling noise. */
export function expectedAuc(separability: number): number {
  return normalCdf(separability / Math.SQRT2);
}

export function generateSamples(
  prevalence: number,
  separability: number,
  seed: number,
  count = SAMPLE_COUNT,
): Sample[] {
  const random = seededRandom(seed);
  const positives = Math.round(count * prevalence);

  const samples: Sample[] = [];
  for (let index = 0; index < count; index += 1) {
    const trueLabel: 0 | 1 = index < positives ? 1 : 0;
    const mean = (trueLabel === 1 ? 1 : -1) * (separability / 2);
    samples.push({
      score: sigmoid(mean + gaussian(random)),
      trueLabel,
    });
  }

  // Shuffle so the order carries no information — the table and any top-k view
  // would otherwise be sorted by label.
  for (let index = samples.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(random() * (index + 1));
    [samples[index], samples[swap]] = [samples[swap]!, samples[index]!];
  }

  return samples;
}

/**
 * The confusion matrix at a cutoff.
 *
 * `score >= threshold` is flagged. Using >= rather than > matters at the extremes:
 * at threshold 0 every case is flagged, which is the "flag everything" corner the
 * accuracy paradox lives next to, and it should be reachable.
 */
export function confusionAt(
  samples: Sample[],
  threshold: number,
): ConfusionMatrix {
  let truePositives = 0;
  let falsePositives = 0;
  let trueNegatives = 0;
  let falseNegatives = 0;

  for (const sample of samples) {
    const flagged = sample.score >= threshold;
    if (sample.trueLabel === 1) {
      if (flagged) truePositives += 1;
      else falseNegatives += 1;
    } else if (flagged) falsePositives += 1;
    else trueNegatives += 1;
  }

  return { truePositives, falsePositives, trueNegatives, falseNegatives };
}

/**
 * Metrics from a matrix.
 *
 * Every zero-denominator case is defined explicitly rather than left as NaN,
 * because those cases are exactly where the lesson is:
 *
 *   - Flag nothing, and precision has no denominator. Reported as 0, not 1: a
 *     classifier that never fires has not achieved perfect precision, it has
 *     declined to make a prediction. Calling that 1.0 would hand the player a
 *     perfect score for doing nothing, which is the paradox this game is about.
 *   - No positives in the data at all, and recall has no denominator. Reported as
 *     0 for the same reason.
 */
export function metricsOf(matrix: ConfusionMatrix): Metrics {
  const {
    truePositives: tp,
    falsePositives: fp,
    trueNegatives: tn,
    falseNegatives: fn,
  } = matrix;

  const total = tp + fp + tn + fn;
  const flagged = tp + fp;
  const actualPositives = tp + fn;
  const actualNegatives = tn + fp;

  const accuracy = total === 0 ? 0 : (tp + tn) / total;
  const precision = flagged === 0 ? 0 : tp / flagged;
  const recall = actualPositives === 0 ? 0 : tp / actualPositives;
  const specificity = actualNegatives === 0 ? 0 : tn / actualNegatives;
  const f1 =
    precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);

  return {
    accuracy,
    precision,
    recall,
    f1,
    specificity,
    balancedAccuracy: (recall + specificity) / 2,
    falsePositiveRate: actualNegatives === 0 ? 0 : fp / actualNegatives,
  };
}

export function metricValue(metrics: Metrics, key: MetricKey): number {
  return metrics[key];
}

export const METRIC_LABELS: Record<MetricKey, string> = {
  accuracy: "Accuracy",
  precision: "Precision",
  recall: "Recall",
  f1: "F1",
  balancedAccuracy: "Balanced accuracy",
};

/** A metric's name mid-sentence: "recall", "balanced accuracy" — but "F1". */
export function metricName(metric: MetricKey): string {
  return metric === "f1" ? METRIC_LABELS.f1 : METRIC_LABELS[metric].toLowerCase();
}

// ── ROC ───────────────────────────────────────────────────────────────────

export interface RocPoint {
  threshold: number;
  falsePositiveRate: number;
  truePositiveRate: number;
}

/**
 * The ROC curve, swept over every distinct cutoff.
 *
 * Walks the samples in descending score order so the curve is exact rather than
 * sampled on a grid — a grid would round off the corners at the ends, which is
 * where the interesting thresholds are.
 */
export function rocCurve(samples: Sample[]): RocPoint[] {
  const sorted = [...samples].sort((a, b) => b.score - a.score);
  const positives = sorted.filter((s) => s.trueLabel === 1).length;
  const negatives = sorted.length - positives;
  if (positives === 0 || negatives === 0) return [];

  const points: RocPoint[] = [
    { threshold: 1.000001, falsePositiveRate: 0, truePositiveRate: 0 },
  ];

  let truePositives = 0;
  let falsePositives = 0;
  for (let index = 0; index < sorted.length; index += 1) {
    const sample = sorted[index]!;
    if (sample.trueLabel === 1) truePositives += 1;
    else falsePositives += 1;

    // Only emit a point once the score changes, or ties would produce a
    // staircase that misrepresents the curve.
    const next = sorted[index + 1];
    if (next && next.score === sample.score) continue;

    points.push({
      threshold: sample.score,
      falsePositiveRate: falsePositives / negatives,
      truePositiveRate: truePositives / positives,
    });
  }

  return points;
}

/** Area under the ROC curve, by trapezoid over the exact curve. */
export function auc(samples: Sample[]): number {
  const curve = rocCurve(samples);
  let area = 0;
  for (let index = 1; index < curve.length; index += 1) {
    const previous = curve[index - 1]!;
    const current = curve[index]!;
    area +=
      ((current.falsePositiveRate - previous.falsePositiveRate) *
        (current.truePositiveRate + previous.truePositiveRate)) /
      2;
  }
  return area;
}

/** Where the player currently sits on the curve. */
export function rocPointAt(samples: Sample[], threshold: number): RocPoint {
  const metrics = metricsOf(confusionAt(samples, threshold));
  return {
    threshold,
    falsePositiveRate: metrics.falsePositiveRate,
    truePositiveRate: metrics.recall,
  };
}

// ── The critic's scenarios ────────────────────────────────────────────────

export interface Constraint {
  metric: MetricKey;
  /** Minimum acceptable value. */
  floor: number;
}

export interface Scenario {
  id: string;
  index: number;
  /** What the critic calls this shift. */
  name: string;
  /** What a positive case is. */
  positiveLabel: string;
  /**
   * The plural, spelled out. Appending "s" gives "contaminated dishs", and this
   * label sits in the always-visible matrix and brief.
   */
  positiveLabelPlural: string;
  /** What flagging one does. */
  flagAction: string;
  prevalence: number;
  separability: number;
  /** The metric the shell shows as the headline number. */
  primary: MetricKey;
  constraints: readonly Constraint[];
  /** The critic's brief: the cost of each error, in plain language. */
  brief: string;
  /** Cost of a false positive, in words. */
  falsePositiveCost: string;
  /** Cost of a false negative, in words. */
  falseNegativeCost: string;
}

/**
 * Which way to move the cutoff to raise a given metric.
 *
 * Recall and precision are unambiguous and opposite. F1 and balanced accuracy are
 * composites, so the direction depends on which half is currently the weaker one —
 * computed from the live metrics rather than hardcoded per scenario, because the
 * answer genuinely changes with where the player is standing.
 */
export function directionToRaise(
  metric: MetricKey,
  metrics: Metrics,
): "lower" | "raise" {
  switch (metric) {
    case "recall":
      return "lower";
    case "precision":
      return "raise";
    case "f1":
      return metrics.recall < metrics.precision ? "lower" : "raise";
    case "balancedAccuracy":
    case "accuracy":
      return metrics.recall < metrics.specificity ? "lower" : "raise";
  }
}

/** The metric a given one trades against, where that trade is crisp. */
export function opposingMetric(metric: MetricKey): MetricKey | null {
  if (metric === "recall") return "precision";
  if (metric === "precision") return "recall";
  return null;
}

/**
 * Four shifts, ordered so the lesson builds.
 *
 * They are not difficulty tiers — they are four different *cost structures*, and
 * each one has its optimum in a different place. Shift 2 is the trap: at 6%
 * prevalence, flagging nothing scores 94% accuracy, which is the accuracy paradox
 * with a number attached.
 */
export const SCENARIOS: readonly Scenario[] = [
  {
    id: "spam-orders",
    index: 1,
    name: "Prank orders",
    positiveLabel: "prank order",
    positiveLabelPlural: "prank orders",
    flagAction: "refuse the order",
    prevalence: 0.35,
    separability: 2.0,
    primary: "precision",
    constraints: [
      { metric: "precision", floor: 0.9 },
      { metric: "recall", floor: 0.35 },
    ],
    brief:
      "A third of tonight's tickets are pranks. Refusing a real customer's dinner is far worse than cooking one prank — but I still want most pranks caught.",
    falsePositiveCost: "a paying customer is turned away and does not come back",
    falseNegativeCost: "the kitchen wastes one meal",
  },
  {
    id: "allergen-screen",
    index: 2,
    name: "Allergen screening",
    positiveLabel: "contaminated dish",
    positiveLabelPlural: "contaminated dishes",
    flagAction: "pull the dish and remake it",
    prevalence: 0.06,
    separability: 2.2,
    primary: "recall",
    constraints: [
      { metric: "recall", floor: 0.9 },
      { metric: "precision", floor: 0.2 },
    ],
    brief:
      "Only 6% of dishes are contaminated, so you can hit 94% accuracy by declaring the kitchen spotless and going home. One missed dish puts someone in hospital. Remaking a clean dish costs me four pounds.",
    falsePositiveCost: "one dish is remade for nothing",
    falseNegativeCost: "a customer is hospitalised",
  },
  {
    id: "kitchen-qa",
    index: 3,
    name: "Plating standards",
    positiveLabel: "badly plated dish",
    positiveLabelPlural: "badly plated dishes",
    flagAction: "send it back to be replated",
    prevalence: 0.5,
    separability: 2.0,
    primary: "f1",
    constraints: [{ metric: "f1", floor: 0.8 }],
    brief:
      "Half the plates need work and the two mistakes cost me about the same: replating a fine dish wastes a minute, letting a bad one out costs a review. Balance them.",
    falsePositiveCost: "a good dish is delayed a minute",
    falseNegativeCost: "a bad plate reaches the dining room",
  },
  {
    id: "fraud-review",
    index: 4,
    name: "Card fraud review",
    positiveLabel: "fraudulent payment",
    positiveLabelPlural: "fraudulent payments",
    flagAction: "hold the payment for review",
    prevalence: 0.12,
    separability: 2.4,
    primary: "balancedAccuracy",
    constraints: [
      { metric: "recall", floor: 0.75 },
      { metric: "precision", floor: 0.5 },
    ],
    brief:
      "12% of card payments are fraud. I have one person on reviews, so more than half of what you hold has to be real fraud or they are wasting their shift — and I still need three quarters of the fraud caught. Both errors hurt here.",
    falsePositiveCost: "an honest customer's card is held up",
    falseNegativeCost: "the restaurant absorbs the chargeback",
  },
] as const;

export function scenarioAt(index: number): Scenario {
  return SCENARIOS[clamp(index - 1, 0, SCENARIOS.length - 1)] ?? SCENARIOS[0]!;
}

/** Best accuracy obtainable by ignoring the scores and guessing one class. */
export function majorityBaseline(prevalence: number): number {
  return Math.max(prevalence, 1 - prevalence);
}

// ── Judging a shift ───────────────────────────────────────────────────────

export type Outcome =
  | "win"
  | "accuracy-paradox"
  | "wrong-side"
  | "overshoot"
  | "missed"
  | "unscored";

export interface ConstraintResult extends Constraint {
  achieved: number;
  met: boolean;
}

export interface RoundResult {
  scenario: string;
  threshold: number;
  matrix: ConfusionMatrix;
  metrics: Metrics;
  constraints: ConstraintResult[];
  outcome: Outcome;
  score: number;
  /**
   * On a win: the lowest and highest value the shift's headline metric takes
   * across every slider cutoff that satisfies the brief. The score is where the
   * served cutoff sits inside that range. Null when the brief was not met.
   */
  primaryRange: { min: number; max: number } | null;
  failure: NamedFailure | null;
}

const percent = (value: number) => `${Math.round(value * 100)}%`;
const points = (value: number) => `${(value * 100).toFixed(1)}%`;
const lower = metricName;

/**
 * How imbalanced the data must be before "Accuracy paradox" is the right name.
 *
 * At 80% majority class, guessing gets you 80% and the number starts doing real
 * damage to a reader's judgement. Below that, high accuracy is not yet a lie.
 */
export const PARADOX_BASELINE = 0.8;
/** How near-perfect the opposing metric must be to call it an overshoot. */
export const OVERSHOOT_LEVEL = 0.95;

/**
 * Share of a shift's score that meeting the brief is worth on its own.
 *
 * The rest is earned by WHERE inside the winning band the cutoff sits: how far
 * the shift's headline metric was pushed, as a fraction of how far the brief
 * allowed. Measured, the old split (65% for the floors plus 35% of F1) put every
 * possible winning week above the 80% second-star bar — the lowest winning
 * cutoffs on all four shifts still averaged 85% — so the star said nothing.
 * With this split, a week served at the bottom of every band scores 60% and one
 * served at the top scores 100%; the second star asks for the better half.
 */
export const BRIEF_MET_SHARE = 0.6;

/** Does this set of metrics satisfy every floor in the brief? */
export function meetsBrief(scenario: Scenario, metrics: Metrics): boolean {
  return scenario.constraints.every(
    (constraint) => metricValue(metrics, constraint.metric) >= constraint.floor,
  );
}

/**
 * Every slider cutoff that satisfies the brief, with the headline metric there.
 *
 * A real sweep over the 101 stops — the same search the code lane's starter
 * snippet runs. It is what lets the judge say "no cutoff satisfies both" only
 * when that is true, rather than inferring it from which way two metrics point.
 * 101 × 800 comparisons, well under a millisecond.
 */
export function winningBand(
  scenario: Scenario,
  samples: Sample[],
): { threshold: number; primary: number }[] {
  const band: { threshold: number; primary: number }[] = [];
  for (const threshold of SLIDER_THRESHOLDS) {
    const metrics = metricsOf(confusionAt(samples, threshold));
    if (meetsBrief(scenario, metrics)) {
      band.push({ threshold, primary: metricValue(metrics, scenario.primary) });
    }
  }
  return band;
}

/**
 * Which error this brief can least afford — read from the BRIEF, never from
 * whichever floor happened to break.
 *
 * On the prank shift (headline: precision) the dear error is the false positive,
 * a paying customer turned away. On the allergen shift (headline: recall) it is
 * the false negative, a customer in hospital. Getting this backwards is how a
 * judge ends up calling a hospitalisation "the cheaper mistake".
 *
 * Composite headlines (F1, balanced accuracy) put a floor under both errors, so
 * there the dear one is whichever is currently breaking its floor: too little
 * precision means too many false positives, too little recall too many misses.
 */
export function dearErrorFor(
  scenario: Scenario,
  failing: MetricKey,
): "falsePositive" | "falseNegative" {
  if (scenario.primary === "precision") return "falsePositive";
  if (scenario.primary === "recall") return "falseNegative";
  return failing === "precision" ? "falsePositive" : "falseNegative";
}

/**
 * Judge the player's cutoff against the critic's brief.
 *
 * Four failures, ordered by how fundamental the misunderstanding is:
 *
 *   1. "Accuracy paradox" — accuracy is at or above what you would get by
 *      ignoring the model entirely, while the metric that matters has collapsed.
 *      Checked first because a player in this state is reading the wrong number,
 *      and no amount of nudging the threshold fixes a wrong scoreboard.
 *   2. "Wrong side of the tradeoff" — a floor is short while the metric it
 *      trades against, which is NOT the one the critic is paying for, is nearly
 *      perfect. The right scoreboard, the wrong direction.
 *   3. "Overshot the tradeoff" — the headline metric itself is nearly perfect,
 *      and the floor on the other one broke to pay for it. The right direction,
 *      pushed too far. Kept apart from (2) because the advice about which error
 *      is cheap is the opposite: telling an allergen player who has flagged
 *      every dish to "let the cheaper mistake happen" would name a
 *      hospitalisation as the cheap one.
 *   4. "Target band missed" — outside the band, short of it or past it. Not a
 *      conceptual error, so it gets directional advice rather than a lecture,
 *      and the direction is read from where the band actually is.
 */
export function judgeRound({
  scenario,
  samples,
  threshold,
}: {
  scenario: Scenario;
  samples: Sample[];
  threshold: number;
}): RoundResult {
  const matrix = confusionAt(samples, threshold);
  const metrics = metricsOf(matrix);

  const constraints: ConstraintResult[] = scenario.constraints.map(
    (constraint) => {
      const achieved = metricValue(metrics, constraint.metric);
      return { ...constraint, achieved, met: achieved >= constraint.floor };
    },
  );

  const met = constraints.filter((constraint) => constraint.met).length;
  const satisfied = met === constraints.length;
  const constraintScore = met / Math.max(1, constraints.length);
  const primaryValue = metricValue(metrics, scenario.primary);

  if (satisfied) {
    // Where inside the band this cutoff sits, on the metric the brief is about.
    // A cutoff off the slider grid (the code lane can set any value) may sit
    // past the grid's best, so the position is clamped; an empty grid band
    // means the player found a winner the grid cannot, which is the top.
    const band = winningBand(scenario, samples);
    const values = band.map((entry) => entry.primary);
    const min = values.length > 0 ? Math.min(...values) : primaryValue;
    const max = values.length > 0 ? Math.max(...values) : primaryValue;
    const position =
      max - min < 1e-9 ? 1 : clamp((primaryValue - min) / (max - min), 0, 1);
    return {
      scenario: scenario.id,
      threshold,
      matrix,
      metrics,
      constraints,
      outcome: "win",
      score: clamp(BRIEF_MET_SHARE + (1 - BRIEF_MET_SHARE) * position, 0, 1),
      primaryRange: { min, max },
      failure: null,
    };
  }

  const base = {
    scenario: scenario.id,
    threshold,
    matrix,
    metrics,
    constraints,
    score: clamp(constraintScore * BRIEF_MET_SHARE, 0, 1),
    primaryRange: null,
  };

  const baseline = majorityBaseline(scenario.prevalence);
  const failed = constraints.filter((constraint) => !constraint.met);
  const flagged = matrix.truePositives + matrix.falsePositives;

  // 1. Reading the wrong number entirely.
  //
  // Gated on the BASELINE being deceptive, not just on accuracy being high.
  // On balanced data, flagging nothing also scores accuracy equal to the
  // baseline — but that baseline is 50%, and nobody is fooled by 50%. The
  // paradox is a property of imbalance, so it is only named when the data is
  // actually imbalanced enough for accuracy to flatter a useless classifier.
  //
  // Recall is the metric checked rather than the scenario's primary, because the
  // paradox always takes the same form: the positives are being missed.
  if (
    baseline >= PARADOX_BASELINE &&
    metrics.accuracy >= baseline - 0.01 &&
    metrics.recall < 0.5
  ) {
    return {
      ...base,
      outcome: "accuracy-paradox",
      failure: {
        name: "Accuracy paradox",
        detail: `${percent(
          metrics.accuracy,
        )} accuracy — and you would get ${percent(
          baseline,
        )} by ignoring the scores and calling every case ${
          scenario.prevalence < 0.5 ? "clean" : "suspect"
        }. ${
          flagged === 0
            ? `You flagged nothing at all, so ${lower(
                scenario.primary,
              )} is ${percent(primaryValue)}.`
            : `You flagged ${flagged} of ${samples.length} cases and caught ${
                matrix.truePositives
              } of ${matrix.truePositives + matrix.falseNegatives} ${
                scenario.positiveLabelPlural
              }, so ${lower(scenario.primary)} is ${percent(primaryValue)}.`
        } Accuracy counts both classes equally and ${percent(
          1 - scenario.prevalence,
        )} of these cases are negative, so it barely moves when you miss every single ${
          scenario.positiveLabel
        }. It is the wrong scoreboard for this shift.`,
      },
    };
  }

  // 2 and 3. One metric all but perfect, the floor on the other broken.
  //
  // Judged on the FAILED constraint rather than the scenario's headline metric,
  // so a player at 97% precision and 14% recall on the prank shift is not told
  // they are merely a little short. Which of the two failures it is depends on
  // whether the near-perfect metric is the one the brief is paying for.
  const lopsided = failed
    .map((constraint) => {
      const opposing = opposingMetric(constraint.metric);
      if (opposing === null) return null;
      const opposingValue = metricValue(metrics, opposing);
      return opposingValue >= OVERSHOOT_LEVEL
        ? { constraint, opposing, opposingValue }
        : null;
    })
    .find((entry) => entry !== null);

  if (lopsided) {
    const { constraint, opposing, opposingValue } = lopsided;
    const move =
      directionToRaise(constraint.metric, metrics) === "lower"
        ? "Lower"
        : "Raise";
    const shortfall = `${lower(constraint.metric)} is ${percent(
      constraint.achieved,
    )}, short of the ${percent(constraint.floor)} floor`;

    // The error that is piling up is the one the broken floor counts: a
    // precision floor counts false positives, a recall floor counts misses.
    const pilingIsFalsePositive = constraint.metric === "precision";
    const pilingCount = pilingIsFalsePositive
      ? matrix.falsePositives
      : matrix.falseNegatives;
    const pilingCost = pilingIsFalsePositive
      ? scenario.falsePositiveCost
      : scenario.falseNegativeCost;
    const pilingNoun = pilingIsFalsePositive
      ? `false alarm${pilingCount === 1 ? "" : "s"}`
      : `miss${pilingCount === 1 ? "" : "es"}`;

    if (opposing === scenario.primary) {
      // 3. The right metric, pushed past what the brief needs.
      const primaryFloor = scenario.constraints.find(
        (entry) => entry.metric === scenario.primary,
      )?.floor;
      const protectedCost =
        opposing === "precision"
          ? scenario.falsePositiveCost
          : scenario.falseNegativeCost;
      return {
        ...base,
        outcome: "overshoot",
        failure: {
          name: "Overshot the tradeoff",
          detail: `${METRIC_LABELS[opposing]} ${percent(
            opposingValue,
          )} is more than this brief needs${
            primaryFloor === undefined
              ? ""
              : ` — it asks for ${percent(primaryFloor)}`
          } — and ${shortfall}. You protected against the right error (here ${protectedCost}) so hard that the other one piled up: ${pilingCount} ${pilingNoun}, each meaning ${pilingCost}. ${move} the threshold and give back some ${lower(
            opposing,
          )}; the brief asks for enough of it, not all of it.`,
        },
      };
    }

    // 2. The metric the critic is NOT paying for, bought with the one they are.
    const dear = dearErrorFor(scenario, constraint.metric);
    const dearer =
      dear === "falsePositive"
        ? scenario.falsePositiveCost
        : scenario.falseNegativeCost;
    const cheaper =
      dear === "falsePositive"
        ? scenario.falseNegativeCost
        : scenario.falsePositiveCost;
    const composite = opposingMetric(scenario.primary) === null;

    return {
      ...base,
      outcome: "wrong-side",
      failure: {
        name: "Wrong side of the tradeoff",
        detail: composite
          ? // "Both errors hurt here": no error is the cheap one, so the copy
            // says which floor is breaking rather than ranking the two.
            `${METRIC_LABELS[opposing]} ${percent(
              opposingValue,
            )} is all but perfect while ${shortfall}. This brief puts a floor under both errors and you have spent everything on one side of it: ${pilingCount} ${pilingNoun}, each meaning ${pilingCost}. ${move} the threshold and accept more of the other error — ${cheaper} — until both floors hold.`
          : `${METRIC_LABELS[opposing]} ${percent(
              opposingValue,
            )} is all but perfect while ${shortfall}. You bought the error the critic does not mind at the price of the one that matters: here ${dearer}, whereas ${cheaper}. ${move} the threshold and let the cheaper mistake happen more often.`,
      },
    };
  }

  // 4. Right idea, short of the band.
  const shortfalls = failed
    .map(
      (constraint) =>
        `${lower(constraint.metric)} ${points(
          constraint.achieved,
        )} against a ${percent(constraint.floor)} floor`,
    )
    .join(", and ");

  // Flag nothing and there is no trade to be on the wrong side of yet. Named
  // first because the direction heuristics below would disagree here — recall
  // says lower, and a precision of 0 (no denominator) says "raise", which
  // cannot raise anything — and that disagreement used to be reported as proof
  // that no cutoff could work, on shifts that have a winning band.
  if (flagged === 0) {
    return {
      ...base,
      outcome: "missed",
      failure: {
        name: "Target band missed",
        detail: `${shortfalls}. You flagged nothing at all, so precision has no denominator — it is reported as 0, not as perfect — and nothing is being traded yet. Lower the threshold until the model starts flagging cases, then watch which floor moves first.`,
      },
    };
  }

  const directions = new Set(
    failed.map((constraint) => directionToRaise(constraint.metric, metrics)),
  );

  let advice: string;
  if (directions.size > 1) {
    // Claim infeasibility only after actually looking for a cutoff.
    const band = winningBand(scenario, samples);
    if (band.length === 0) {
      advice =
        "The two shortfalls want the threshold moved in opposite directions, and no cutoff on the slider satisfies both — this model is not separating the cases well enough for this brief, and the honest answer is to say so rather than to keep sliding.";
    } else {
      const nearest = band.reduce((best, entry) =>
        Math.abs(entry.threshold - threshold) <
        Math.abs(best.threshold - threshold)
          ? entry
          : best,
      );
      advice = `The two shortfalls pull the threshold in opposite directions from here, yet cutoffs that satisfy both do exist, ${
        nearest.threshold > threshold ? "above" : "below"
      } yours. Move toward them one notch at a time and watch which cell of the matrix pays.`;
    }
  } else {
    // Said from where the winning cutoffs actually are, not from the path the
    // player took: "not far enough along it" read backwards to a player who had
    // pushed PAST the band (prank shift at 0.93, one notch between two
    // overshoots), and the direction is the band's, which cannot be wrong.
    const band = winningBand(scenario, samples);
    if (band.length === 0) {
      advice = `${
        directions.has("lower") ? "Lower" : "Raise"
      } the threshold and watch which cell of the matrix pays for it — though no cutoff on the slider meets every floor of this brief, so another floor will give way first.`;
    } else {
      const nearest = band.reduce((best, entry) =>
        Math.abs(entry.threshold - threshold) <
        Math.abs(best.threshold - threshold)
          ? entry
          : best,
      );
      const below = nearest.threshold < threshold;
      const side = below ? "below" : "above";
      const all = band.every((entry) =>
        below ? entry.threshold < threshold : entry.threshold > threshold,
      );
      // The headline floor already holding is the overshoot's lesson, one
      // notch short of its 95% bar: say it here too.
      const headline = scenario.constraints.find(
        (constraint) => constraint.metric === scenario.primary,
      );
      const headlineHolds =
        headline !== undefined && primaryValue >= headline.floor;
      advice = `${
        below ? "Lower" : "Raise"
      } the threshold and watch which cell of the matrix pays for it. ${
        all
          ? `Every cutoff that meets this brief is ${side} yours.`
          : `The nearest cutoff that meets this brief is ${side} yours.`
      }${
        headlineHolds && headline
          ? ` ${METRIC_LABELS[scenario.primary]} already clears its ${percent(
              headline.floor,
            )} floor: the brief asks for enough of it, not all of it.`
          : ""
      }`;
    }
  }

  return {
    ...base,
    outcome: "missed",
    failure: {
      name: "Target band missed",
      detail: `${shortfalls}. ${advice}`,
    },
  };
}

// ── Reveal the math ───────────────────────────────────────────────────────

export const MATH_EQUATION = String.raw`\text{flag } i \iff s_i \ge t
\\[1em]
\mathrm{precision} = \frac{TP}{TP+FP}
\qquad
\mathrm{recall} = \frac{TP}{TP+FN}
\\[1em]
F_1 = 2\cdot\frac{\mathrm{precision}\cdot\mathrm{recall}}{\mathrm{precision}+\mathrm{recall}}
\qquad
\mathrm{accuracy} = \frac{TP+TN}{TP+FP+TN+FN}`;

export const MATH_CODE = `// The whole game. No model is retrained when you move the slider —
// the scores are fixed, and the threshold decides what to do with them.
export function confusionAt(samples, threshold) {
  let TP = 0, FP = 0, TN = 0, FN = 0;

  for (const { score, trueLabel } of samples) {
    const flagged = score >= threshold;   // <- the slider
    if (trueLabel === 1) flagged ? TP++ : FN++;
    else                 flagged ? FP++ : TN++;
  }

  return { TP, FP, TN, FN };
}

// Note which denominators each metric uses:
const precision = TP / (TP + FP);   // of what you flagged, how much was right
const recall    = TP / (TP + FN);   // of what mattered, how much you caught
const accuracy  = (TP + TN) / (TP + FP + TN + FN);   // counts TN, which is why
                                                     // it lies on rare positives`;

export const MATH_NOTES = `Precision and recall share a numerator and differ in the denominator, and that is the entire tradeoff. Raising the cutoff removes cases from the flagged pile, lowest scores first. Any true positives among them lower recall — it can never rise when the pile shrinks. Precision usually climbs, because with a model that ranks well the lowest-scoring flagged cases are the likeliest to be false alarms; but it only climbs when the removed cases were right less often than the pile as a whole, so a notch here and there can move it the other way. Over the whole range you cannot buy one without paying in the other.

Accuracy is the odd one out because TN is in its numerator. When 94% of cases are negative, a classifier that flags nothing collects all of those true negatives and reports 94% — while catching none of the thing you built it for. That is not a rounding artefact, it is what accuracy means on imbalanced data.

F1 is the harmonic mean of precision and recall, which is why it cannot be gamed by pushing one to 1.0: the harmonic mean of 1.0 and 0.05 is 0.095, not 0.525. It is the right summary when both errors cost about the same, and the wrong one when they do not — which is why the critic keeps changing the brief instead of asking for F1 every night.

The ROC curve is every threshold at once, plotted as recall against false-positive rate. Moving the slider slides one point along a fixed curve. The curve is a property of the model; the point on it is your decision.`;
