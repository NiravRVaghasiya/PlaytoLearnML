import { beforeEach, describe, expect, it } from "vitest";
import {
  EMPTY_PROGRESSION,
  HIGH_SCORE_THRESHOLD,
  createMemoryAdapter,
  useProgression,
} from "@/engine/progression";
import {
  BRIEF_MET_SHARE,
  OVERSHOOT_LEVEL,
  PARADOX_BASELINE,
  SAMPLE_COUNT,
  SCENARIOS,
  SLIDER_THRESHOLDS,
  auc,
  confusionAt,
  dearErrorFor,
  directionToRaise,
  expectedAuc,
  generateSamples,
  judgeRound,
  majorityBaseline,
  metricValue,
  metricsOf,
  opposingMetric,
  rocCurve,
  rocPointAt,
  scenarioAt,
  winningBand,
  type Sample,
  type Scenario,
} from "./ml";
import { createJsExecutor } from "@/engine/useCodeLane";
import { SLUG, createCodeApi, useChefStore } from "./store";
import { whyCardFor } from "./why-cards";
import { STARTER_CODE } from "./CodeLane";

const DATA_SEED = 9310;

/** Same construction the store uses, so the tests measure the real shifts. */
const samplesFor = (scenario: Scenario): Sample[] =>
  generateSamples(
    scenario.prevalence,
    scenario.separability,
    DATA_SEED + scenario.index * 37,
  );

/** Every threshold the slider can reach. */
const THRESHOLDS = Array.from({ length: 101 }, (_, step) => step / 100);

function winnableThresholds(scenario: Scenario): number[] {
  const samples = samplesFor(scenario);
  return THRESHOLDS.filter((threshold) => {
    const metrics = metricsOf(confusionAt(samples, threshold));
    return scenario.constraints.every(
      (constraint) => metricValue(metrics, constraint.metric) >= constraint.floor,
    );
  });
}

describe("the score generator", () => {
  it("is deterministic for a seed", () => {
    expect(generateSamples(0.3, 2, 11)).toEqual(generateSamples(0.3, 2, 11));
  });

  it("produces the requested prevalence", () => {
    const samples = generateSamples(0.06, 2.2, 5);
    const positives = samples.filter((s) => s.trueLabel === 1).length;
    expect(positives).toBe(Math.round(SAMPLE_COUNT * 0.06));
  });

  it("keeps every score inside the open unit interval", () => {
    for (const sample of generateSamples(0.5, 3, 7)) {
      expect(sample.score).toBeGreaterThan(0);
      expect(sample.score).toBeLessThan(1);
    }
  });

  it("hits the AUC its separability predicts", () => {
    // The generator's contract: two unit-variance normals separated by d give
    // AUC = Φ(d/√2), and sigmoid is monotone so the scores inherit it. This is
    // what makes "separability 2.2" a checkable claim instead of a magic number.
    for (const separability of [1, 1.6, 2, 2.4, 3]) {
      const samples = generateSamples(0.4, separability, 4242);
      expect(
        auc(samples),
        `separability ${separability}`,
      ).toBeCloseTo(expectedAuc(separability), 1);
    }
  });

  it("orders the samples so position carries no label information", () => {
    // Unshuffled, every positive would sit at the front and any table view would
    // give the answer away.
    const samples = generateSamples(0.5, 2, 3);
    const firstHalfPositives = samples
      .slice(0, samples.length / 2)
      .filter((s) => s.trueLabel === 1).length;
    expect(firstHalfPositives).toBeGreaterThan(samples.length * 0.2);
    expect(firstHalfPositives).toBeLessThan(samples.length * 0.3);
  });
});

describe("the confusion matrix", () => {
  const samples = generateSamples(0.3, 2, 21);

  it("accounts for every case at every threshold", () => {
    for (const threshold of THRESHOLDS) {
      const m = confusionAt(samples, threshold);
      expect(
        m.truePositives + m.falsePositives + m.trueNegatives + m.falseNegatives,
      ).toBe(samples.length);
    }
  });

  it("flags everything at zero and nothing above one", () => {
    const all = confusionAt(samples, 0);
    expect(all.falseNegatives).toBe(0);
    expect(all.trueNegatives).toBe(0);

    const none = confusionAt(samples, 1.01);
    expect(none.truePositives).toBe(0);
    expect(none.falsePositives).toBe(0);
  });

  it("catches monotonically more as the threshold falls", () => {
    let previous = -1;
    for (const threshold of [...THRESHOLDS].reverse()) {
      const { truePositives } = confusionAt(samples, threshold);
      expect(truePositives).toBeGreaterThanOrEqual(previous);
      previous = truePositives;
    }
  });
});

describe("the metrics", () => {
  it("computes the textbook definitions", () => {
    const metrics = metricsOf({
      truePositives: 30,
      falsePositives: 10,
      trueNegatives: 50,
      falseNegatives: 10,
    });
    expect(metrics.accuracy).toBeCloseTo(80 / 100, 6);
    expect(metrics.precision).toBeCloseTo(30 / 40, 6);
    expect(metrics.recall).toBeCloseTo(30 / 40, 6);
    expect(metrics.specificity).toBeCloseTo(50 / 60, 6);
    expect(metrics.f1).toBeCloseTo(0.75, 6);
  });

  it("scores a classifier that flags nothing as zero precision, not perfect", () => {
    // The whole game turns on this. Vacuous precision of 1.0 would hand a perfect
    // headline to a player who did nothing at all, which is the exact deception
    // the accuracy paradox is about.
    const metrics = metricsOf({
      truePositives: 0,
      falsePositives: 0,
      trueNegatives: 94,
      falseNegatives: 6,
    });
    expect(metrics.precision).toBe(0);
    expect(metrics.recall).toBe(0);
    expect(metrics.f1).toBe(0);
    // And accuracy still reads 94%, which is the point.
    expect(metrics.accuracy).toBeCloseTo(0.94, 6);
  });

  it("uses a harmonic mean for F1, so one perfect half cannot carry it", () => {
    const lopsided = metricsOf({
      truePositives: 1,
      falsePositives: 0,
      trueNegatives: 80,
      falseNegatives: 19,
    });
    expect(lopsided.precision).toBe(1);
    expect(lopsided.recall).toBeCloseTo(0.05, 6);
    // Arithmetic mean would be 0.525. Harmonic mean is 0.095.
    expect(lopsided.f1).toBeCloseTo(0.0952, 3);
    expect(lopsided.f1).toBeLessThan(0.11);
  });

  it("reports balanced accuracy as the mean of recall and specificity", () => {
    const metrics = metricsOf({
      truePositives: 0,
      falsePositives: 0,
      trueNegatives: 94,
      falseNegatives: 6,
    });
    // Flag nothing: recall 0, specificity 1, so balanced accuracy is 0.5 —
    // correctly reporting a coin flip where plain accuracy said 94%.
    expect(metrics.balancedAccuracy).toBeCloseTo(0.5, 6);
  });
});

describe("the ROC curve", () => {
  const samples = generateSamples(0.25, 2, 31);

  it("runs from the origin to the far corner", () => {
    const curve = rocCurve(samples);
    expect(curve[0]!.falsePositiveRate).toBe(0);
    expect(curve[0]!.truePositiveRate).toBe(0);
    expect(curve[curve.length - 1]!.falsePositiveRate).toBeCloseTo(1, 6);
    expect(curve[curve.length - 1]!.truePositiveRate).toBeCloseTo(1, 6);
  });

  it("never goes backwards on either axis", () => {
    const curve = rocCurve(samples);
    for (let index = 1; index < curve.length; index += 1) {
      expect(curve[index]!.falsePositiveRate).toBeGreaterThanOrEqual(
        curve[index - 1]!.falsePositiveRate,
      );
      expect(curve[index]!.truePositiveRate).toBeGreaterThanOrEqual(
        curve[index - 1]!.truePositiveRate,
      );
    }
  });

  it("beats the diagonal for a model with real signal", () => {
    expect(auc(samples)).toBeGreaterThan(0.85);
  });

  it("puts the player's point on the curve it was drawn from", () => {
    // The slider slides a point along a fixed curve; if the point were computed
    // differently from the curve they would drift apart.
    const curve = rocCurve(samples);
    for (const threshold of [0.2, 0.5, 0.8]) {
      const point = rocPointAt(samples, threshold);
      const nearest = curve.reduce((best, candidate) =>
        Math.abs(candidate.falsePositiveRate - point.falsePositiveRate) <
        Math.abs(best.falsePositiveRate - point.falsePositiveRate)
          ? candidate
          : best,
      );
      expect(
        Math.abs(nearest.truePositiveRate - point.truePositiveRate),
      ).toBeLessThan(0.02);
    }
  });

  it("declines to draw a curve when one class is missing", () => {
    expect(rocCurve([{ score: 0.5, trueLabel: 1 }])).toEqual([]);
  });
});

describe("the critic's shifts are all winnable", () => {
  // The load-bearing test. Three of these four bands were originally unreachable
  // or a single threshold wide; the assertions below are what stop that
  // regressing when a floor or a separability gets nudged.

  it.each(SCENARIOS.map((scenario) => [scenario.name, scenario] as const))(
    "%s has a findable winning band",
    (_name, scenario) => {
      const winnable = winnableThresholds(scenario);
      expect(winnable.length, "no threshold satisfies this brief").toBeGreaterThan(
        0,
      );
      // Wide enough to find by sliding with live feedback, rather than by luck.
      expect(winnable.length, "band too narrow to find").toBeGreaterThanOrEqual(8);
    },
  );

  it("puts the winning band in a contiguous run for each shift", () => {
    // A split band would mean the metrics are not behaving monotonically, which
    // would make the slider's feedback misleading.
    for (const scenario of SCENARIOS) {
      const winnable = winnableThresholds(scenario);
      const span =
        Math.round((winnable[winnable.length - 1]! - winnable[0]!) * 100) + 1;
      expect(winnable.length, scenario.name).toBe(span);
    }
  });

  it("needs a different cutoff for the precision shift than the recall shift", () => {
    // "The right metric depends on the cost of each error type" has to be true of
    // the actual numbers: if one threshold cleared both briefs, the game would be
    // teaching that there is a universally correct cutoff.
    const precisionShift = winnableThresholds(scenarioAt(1));
    const recallShift = winnableThresholds(scenarioAt(2));
    const overlap = precisionShift.filter((threshold) =>
      recallShift.includes(threshold),
    );
    expect(overlap).toHaveLength(0);
    expect(Math.min(...precisionShift)).toBeGreaterThan(Math.max(...recallShift));
  });

  it("never lets accuracy alone identify the winning cutoff", () => {
    // If maximising accuracy happened to win the shift, the paradox would never
    // bite and the game would have no argument to make.
    for (const scenario of SCENARIOS.filter(
      (entry) => majorityBaseline(entry.prevalence) >= PARADOX_BASELINE,
    )) {
      const samples = samplesFor(scenario);
      const best = THRESHOLDS.reduce(
        (winner, threshold) => {
          const accuracy = metricsOf(confusionAt(samples, threshold)).accuracy;
          return accuracy > winner.accuracy ? { threshold, accuracy } : winner;
        },
        { threshold: 0, accuracy: -1 },
      );
      expect(
        winnableThresholds(scenario),
        `${scenario.name}: the most accurate cutoff also wins the brief`,
      ).not.toContain(best.threshold);
    }
  });
});

describe("naming the failure", () => {
  const allergen = scenarioAt(2);
  const plating = scenarioAt(3);
  const pranks = scenarioAt(1);

  it("names the Accuracy paradox for flagging nothing on rare positives", () => {
    const result = judgeRound({
      scenario: allergen,
      samples: samplesFor(allergen),
      threshold: 1.01,
    });
    expect(result.outcome).toBe("accuracy-paradox");
    expect(result.failure?.name).toBe("Accuracy paradox");
    // It must state the baseline, or the number means nothing.
    expect(result.failure?.detail).toMatch(/94%/);
    expect(result.failure?.detail).toMatch(/ignoring the scores/i);
    expect(result.metrics.accuracy).toBeGreaterThan(0.9);
    expect(result.metrics.recall).toBe(0);
  });

  it("does NOT name the paradox on balanced data, where 50% fools nobody", () => {
    // Flagging nothing here also matches the majority baseline, but that baseline
    // is a coin flip. Calling it a paradox would misdiagnose the mistake.
    const result = judgeRound({
      scenario: plating,
      samples: samplesFor(plating),
      threshold: 1.01,
    });
    expect(result.outcome).not.toBe("accuracy-paradox");
    expect(result.metrics.accuracy).toBeCloseTo(0.5, 1);
  });

  it("names an overshoot when the headline metric is pushed until the other floor breaks", () => {
    // Prank shift at a high cutoff: precision is ~97%, which SATISFIES the shift's
    // headline metric, while recall collapses and fails the brief. Judging on the
    // headline metric would have called this a near miss — but it is not the
    // wrong side either: precision IS what the critic pays for. It is the right
    // side, pushed too far, and the copy must not call a turned-away customer
    // the cheap mistake.
    const result = judgeRound({
      scenario: pranks,
      samples: samplesFor(pranks),
      threshold: 0.9,
    });
    expect(result.metrics.precision).toBeGreaterThan(OVERSHOOT_LEVEL);
    expect(result.outcome).toBe("overshoot");
    expect(result.failure?.name).toBe("Overshot the tradeoff");
    expect(result.failure?.detail).toMatch(/lower the threshold/i);
    expect(result.failure?.detail).toContain(
      `protected against the right error (here ${pranks.falsePositiveCost})`,
    );
    expect(result.failure?.detail).not.toMatch(/cheaper mistake/i);
  });

  it("never calls a hospitalisation the cheaper mistake on the allergen shift", () => {
    // The finding that prompted the split: flag every dish (t=0, the natural
    // direction on a recall shift) and precision breaks its floor. The old copy
    // then said "a customer is hospitalised … let the cheaper mistake happen".
    for (const threshold of [0, 0.1, 0.36]) {
      const result = judgeRound({
        scenario: allergen,
        samples: samplesFor(allergen),
        threshold,
      });
      expect(result.outcome, `t=${threshold}`).toBe("overshoot");
      expect(result.metrics.recall).toBeGreaterThanOrEqual(OVERSHOOT_LEVEL);
      expect(result.failure?.detail).toMatch(/raise the threshold/i);
      expect(result.failure?.detail).toContain(
        `(here ${allergen.falseNegativeCost})`,
      );
      expect(result.failure?.detail).not.toMatch(/cheaper mistake/i);
    }
  });

  it("names the wrong side when the metric the critic does not pay for is bought", () => {
    // The mirror image, and the advice must reverse with it: recall is near
    // perfect on the PRECISION shift. Here the cost ordering comes from the
    // brief — refusing a customer is the dear error, a wasted meal the cheap one.
    const result = judgeRound({
      scenario: pranks,
      samples: samplesFor(pranks),
      threshold: 0.02,
    });
    expect(result.metrics.recall).toBeGreaterThan(OVERSHOOT_LEVEL);
    expect(result.outcome).toBe("wrong-side");
    expect(result.failure?.name).toBe("Wrong side of the tradeoff");
    expect(result.failure?.detail).toMatch(/raise the threshold/i);
    expect(result.failure?.detail).toContain(
      `here ${pranks.falsePositiveCost}, whereas ${pranks.falseNegativeCost}`,
    );
  });

  it("does not rank the two errors on a shift where both hurt", () => {
    // Fraud's brief: "Both errors hurt here." Flag every payment and the
    // precision floor breaks; the copy says which floor, not which error is cheap.
    const fraud = scenarioAt(4);
    const result = judgeRound({
      scenario: fraud,
      samples: samplesFor(fraud),
      threshold: 0.02,
    });
    expect(result.outcome).toBe("wrong-side");
    expect(result.failure?.detail).toMatch(/floor under both errors/i);
    expect(result.failure?.detail).not.toMatch(/does not mind|cheaper mistake/i);
    expect(result.failure?.detail).toMatch(/raise the threshold/i);
  });

  it("reads the dear error from the brief, not from the floor that broke", () => {
    expect(dearErrorFor(pranks, "recall")).toBe("falsePositive");
    expect(dearErrorFor(pranks, "precision")).toBe("falsePositive");
    expect(dearErrorFor(allergen, "precision")).toBe("falseNegative");
    expect(dearErrorFor(allergen, "recall")).toBe("falseNegative");
    // Composite headline: whichever floor is breaking.
    expect(dearErrorFor(scenarioAt(4), "precision")).toBe("falsePositive");
    expect(dearErrorFor(scenarioAt(4), "recall")).toBe("falseNegative");
  });

  it("names flagging nothing for what it is, on a shift that has a winning band", () => {
    // Prank shift, cutoff at the top: nothing flagged, precision reported 0.
    // The old judge read "precision wants the cutoff raised, recall wants it
    // lowered" as proof no cutoff could work — on a shift winnable at 0.69-0.79.
    expect(winnableThresholds(pranks).length).toBeGreaterThan(0);
    for (const threshold of [0.98, 0.99, 1]) {
      const result = judgeRound({
        scenario: pranks,
        samples: samplesFor(pranks),
        threshold,
      });
      expect(result.matrix.truePositives + result.matrix.falsePositives).toBe(0);
      expect(result.outcome).toBe("missed");
      expect(result.failure?.detail).toMatch(/flagged nothing/i);
      expect(result.failure?.detail).toMatch(/lower the threshold/i);
      expect(result.failure?.detail).not.toMatch(/no cutoff/i);
    }
  });

  it("claims no cutoff satisfies both only after a sweep finds none", () => {
    // Hand-built so that at t=0.93 only two negatives are flagged: precision
    // and recall are both 0, pulling in opposite directions, with something
    // flagged. Whether a band exists then decides the copy, not the directions.
    const samples: Sample[] = [
      ...Array.from({ length: 10 }, () => ({ score: 0.9, trueLabel: 1 as const })),
      ...Array.from({ length: 2 }, () => ({ score: 0.95, trueLabel: 0 as const })),
      ...Array.from({ length: 10 }, () => ({ score: 0.5, trueLabel: 0 as const })),
    ];
    const brief = (floor: number): Scenario => ({
      ...pranks,
      prevalence: 10 / 22,
      primary: "precision",
      constraints: [
        { metric: "precision", floor },
        { metric: "recall", floor },
      ],
    });

    // Feasible: t in (0.5, 0.9] flags all 10 positives and 2 negatives, 83% precision.
    const feasible = judgeRound({ scenario: brief(0.8), samples, threshold: 0.93 });
    expect(winningBand(brief(0.8), samples).length).toBeGreaterThan(0);
    expect(feasible.outcome).toBe("missed");
    expect(feasible.failure?.detail).toMatch(/do exist, below yours/i);
    expect(feasible.failure?.detail).not.toMatch(/no cutoff/i);

    // Infeasible: 83% is the best precision any cutoff with recall reaches.
    const infeasible = judgeRound({ scenario: brief(0.9), samples, threshold: 0.93 });
    expect(winningBand(brief(0.9), samples)).toHaveLength(0);
    expect(infeasible.failure?.detail).toMatch(/no cutoff on the slider satisfies both/i);
  });

  it("names a near miss as a near miss, with a direction", () => {
    const result = judgeRound({
      scenario: pranks,
      samples: samplesFor(pranks),
      threshold: 0.65,
    });
    expect(result.outcome).toBe("missed");
    expect(result.failure?.name).toBe("Target band missed");
    expect(result.failure?.detail).toMatch(/threshold/i);
    expect(result.failure?.detail).toMatch(/\d+\.\d%/);
  });

  it("does not tell a player who has gone past the band they are not far enough", () => {
    // Prank shift at 0.93: precision 93.8%, just under the overshoot bar, one
    // notch between two "Overshot the tradeoff" verdicts. It used to read
    // "You are on the right side of the trade, just not far enough along it."
    const result = judgeRound({
      scenario: pranks,
      samples: samplesFor(pranks),
      threshold: 0.93,
    });
    expect(result.metrics.precision).toBeLessThan(OVERSHOOT_LEVEL);
    expect(result.outcome).toBe("missed");
    expect(result.failure?.detail).toMatch(/lower the threshold/i);
    expect(result.failure?.detail).toContain("Every cutoff that meets this brief is below yours.");
    expect(result.failure?.detail).toContain(
      "Precision already clears its 90% floor: the brief asks for enough of it, not all of it.",
    );
    expect(result.failure?.detail).not.toMatch(/far enough/i);
  });

  it("points a one-way miss at where the winning cutoffs really are, on every shift", () => {
    // Every slider stop judged "missed" with something flagged and one
    // direction to go: the direction and the side must be the band's.
    let checked = 0;
    for (const scenario of SCENARIOS) {
      const samples = samplesFor(scenario);
      const band = winningBand(scenario, samples).map((entry) => entry.threshold);
      expect(band.length, scenario.name).toBeGreaterThan(0);
      for (const threshold of SLIDER_THRESHOLDS) {
        const result = judgeRound({ scenario, samples, threshold });
        const detail = result.failure?.detail ?? "";
        if (result.outcome !== "missed" || /flagged nothing|opposite directions/.test(detail)) {
          continue;
        }
        checked += 1;
        const label = `${scenario.name} at t=${threshold}`;
        const below = band.every((cutoff) => cutoff < threshold);
        const above = band.every((cutoff) => cutoff > threshold);
        expect(below || above, label).toBe(true);
        expect(detail, label).toMatch(below ? /Lower the threshold/ : /Raise the threshold/);
        expect(detail, label).toContain(
          `Every cutoff that meets this brief is ${below ? "below" : "above"} yours.`,
        );
        expect(detail, label).not.toMatch(/far enough/i);
        // The headline note appears exactly when the headline floor holds.
        const headline = scenario.constraints.find(
          (constraint) => constraint.metric === scenario.primary,
        );
        expect(/already clears its/.test(detail), label).toBe(
          headline !== undefined &&
            metricValue(result.metrics, scenario.primary) >= headline.floor,
        );
      }
    }
    expect(checked).toBeGreaterThan(100);
  });

  it("wins inside the band, with no failure attached", () => {
    for (const scenario of SCENARIOS) {
      const band = winnableThresholds(scenario);
      const middle = band[Math.floor(band.length / 2)]!;
      const result = judgeRound({
        scenario,
        samples: samplesFor(scenario),
        threshold: middle,
      });
      expect(result.outcome, `${scenario.name} at t=${middle}`).toBe("win");
      expect(result.failure).toBeNull();
      expect(result.constraints.every((constraint) => constraint.met)).toBe(true);
      expect(result.score).toBeGreaterThan(0.65);
    }
  });

  it("reports every constraint, met or not, so the brief is auditable", () => {
    const result = judgeRound({
      scenario: allergen,
      samples: samplesFor(allergen),
      threshold: 0.75,
    });
    expect(result.constraints).toHaveLength(allergen.constraints.length);
    for (const constraint of result.constraints) {
      expect(constraint.achieved).toBeGreaterThanOrEqual(0);
      expect(constraint.met).toBe(constraint.achieved >= constraint.floor);
    }
  });

  it("keeps the score inside the unit interval everywhere", () => {
    for (const scenario of SCENARIOS) {
      for (const threshold of THRESHOLDS) {
        const { score } = judgeRound({
          scenario,
          samples: samplesFor(scenario),
          threshold,
        });
        expect(score).toBeGreaterThanOrEqual(0);
        expect(score).toBeLessThanOrEqual(1);
      }
    }
  });
});

describe("threshold advice", () => {
  it("sends recall down and precision up", () => {
    const metrics = metricsOf({
      truePositives: 20,
      falsePositives: 5,
      trueNegatives: 60,
      falseNegatives: 15,
    });
    expect(directionToRaise("recall", metrics)).toBe("lower");
    expect(directionToRaise("precision", metrics)).toBe("raise");
  });

  it("sends composite metrics toward whichever half is weaker", () => {
    const recallStarved = metricsOf({
      truePositives: 5,
      falsePositives: 0,
      trueNegatives: 60,
      falseNegatives: 35,
    });
    expect(directionToRaise("f1", recallStarved)).toBe("lower");

    const precisionStarved = metricsOf({
      truePositives: 39,
      falsePositives: 50,
      trueNegatives: 10,
      falseNegatives: 1,
    });
    expect(directionToRaise("f1", precisionStarved)).toBe("raise");
  });

  it("pairs recall with precision and leaves composites unpaired", () => {
    expect(opposingMetric("recall")).toBe("precision");
    expect(opposingMetric("precision")).toBe("recall");
    expect(opposingMetric("f1")).toBeNull();
    expect(opposingMetric("accuracy")).toBeNull();
  });
});

describe("the shifts as a sequence", () => {
  it("makes each shift a different cost structure, not a difficulty tier", () => {
    const primaries = SCENARIOS.map((scenario) => scenario.primary);
    expect(new Set(primaries).size).toBe(SCENARIOS.length);
  });

  it("includes at least one shift imbalanced enough for the paradox to bite", () => {
    const imbalanced = SCENARIOS.filter(
      (scenario) => majorityBaseline(scenario.prevalence) >= PARADOX_BASELINE,
    );
    expect(imbalanced.length).toBeGreaterThan(0);
  });

  it("states the cost of both error types in every brief", () => {
    for (const scenario of SCENARIOS) {
      expect(scenario.falsePositiveCost.length, scenario.name).toBeGreaterThan(10);
      expect(scenario.falseNegativeCost.length, scenario.name).toBeGreaterThan(10);
      expect(scenario.brief.length, scenario.name).toBeGreaterThan(40);
    }
  });

  it("clamps scenarioAt to the real range", () => {
    expect(scenarioAt(0).index).toBe(1);
    expect(scenarioAt(99).index).toBe(SCENARIOS.length);
  });
});

describe("the store's shift flow", () => {
  const store = useChefStore;

  beforeEach(() => {
    store.getState().restart();
  });

  it("starts on shift one, cutoff at the midpoint", () => {
    expect(store.getState().scenarioIndex).toBe(1);
    expect(store.getState().threshold).toBe(0.5);
    expect(store.getState().phase).toBe("tuning");
    expect(store.getState().servedResult).toBeNull();
    expect(store.getState().clearedScores).toEqual([]);
  });

  it("hands the same samples the tests measure against", () => {
    // The store and the test helper must agree, or the winnable bands verified
    // above would not be the bands the player experiences.
    expect(store.getState().samples).toEqual(samplesFor(scenarioAt(1)));
  });

  it("clamps the threshold to the unit interval", () => {
    store.getState().setThreshold(5);
    expect(store.getState().threshold).toBe(1);
    store.getState().setThreshold(-3);
    expect(store.getState().threshold).toBe(0);
  });

  it("retracts a served verdict when the cutoff moves", () => {
    // The critic judged a cutoff. Once the slider moves, this is not that cutoff,
    // and leaving the old verdict on screen would attribute it to the new one.
    store.getState().setThreshold(0.99);
    store.getState().serve();
    expect(store.getState().servedResult).not.toBeNull();
    expect(store.getState().failure).not.toBeNull();

    store.getState().setThreshold(0.5);
    expect(store.getState().servedResult).toBeNull();
    expect(store.getState().failure).toBeNull();
  });

  it("keeps the verdict after a win, so the cleared shift can be read", () => {
    const band = winnableThresholds(scenarioAt(1));
    store.getState().setThreshold(band[Math.floor(band.length / 2)]!);
    store.getState().serve();
    expect(store.getState().phase).toBe("cleared");

    // Sliding after a win must not silently un-clear the shift.
    store.getState().setThreshold(0.1);
    expect(store.getState().phase).toBe("cleared");
    expect(store.getState().servedResult).not.toBeNull();
  });

  it("counts attempts and does not bank a score for a miss", () => {
    store.getState().setThreshold(0.99);
    store.getState().serve();
    store.getState().setThreshold(0.98);
    store.getState().serve();
    expect(store.getState().attempts).toBe(2);
    expect(store.getState().clearedScores).toEqual([]);
    expect(store.getState().phase).toBe("tuning");
  });

  it("advances only after a shift is cleared", () => {
    store.getState().nextShift();
    expect(store.getState().scenarioIndex, "must not skip while tuning").toBe(1);

    const band = winnableThresholds(scenarioAt(1));
    store.getState().setThreshold(band[0]!);
    store.getState().serve();
    store.getState().nextShift();
    expect(store.getState().scenarioIndex).toBe(2);
    expect(store.getState().threshold).toBe(0.5);
    expect(store.getState().attempts).toBe(0);
  });

  it("gives each shift its own samples", () => {
    const first = store.getState().samples;
    const band = winnableThresholds(scenarioAt(1));
    store.getState().setThreshold(band[0]!);
    store.getState().serve();
    store.getState().nextShift();
    expect(store.getState().samples).not.toEqual(first);
    expect(store.getState().samples).toEqual(samplesFor(scenarioAt(2)));
  });

  it("completes only after the fourth shift is signed off", () => {
    for (let shift = 1; shift <= SCENARIOS.length; shift += 1) {
      const band = winnableThresholds(scenarioAt(shift));
      store.getState().setThreshold(band[Math.floor(band.length / 2)]!);
      const result = store.getState().serve();
      expect(result.outcome, `shift ${shift}`).toBe("win");
      if (shift < SCENARIOS.length) {
        expect(store.getState().phase).toBe("cleared");
        store.getState().nextShift();
      }
    }
    expect(store.getState().phase).toBe("complete");
    expect(store.getState().clearedScores).toHaveLength(SCENARIOS.length);
    expect(store.getState().failure).toBeNull();
  });

  it("requires a different cutoff on shift two than the one that won shift one", () => {
    // The lesson, as a state-machine fact: carrying shift one's answer forward
    // fails shift two.
    const shiftOneBand = winnableThresholds(scenarioAt(1));
    const carried = shiftOneBand[Math.floor(shiftOneBand.length / 2)]!;
    store.getState().setThreshold(carried);
    store.getState().serve();
    store.getState().nextShift();

    store.getState().setThreshold(carried);
    const result = store.getState().serve();
    expect(result.outcome).not.toBe("win");
    expect(store.getState().failure).not.toBeNull();
  });
});

describe("scoring a win by where it sits in the band", () => {
  // The second star is "best score ≥ HIGH_SCORE_THRESHOLD". Under the old
  // formula every winning week cleared it — the worst winning cutoff on every
  // shift still averaged 85% — so the star measured nothing but finishing.

  const bandScores = (scenario: Scenario) =>
    winnableThresholds(scenario).map(
      (threshold) =>
        judgeRound({ scenario, samples: samplesFor(scenario), threshold }).score,
    );

  it("spans the whole range from meeting the brief to its best cutoff", () => {
    for (const scenario of SCENARIOS) {
      const scores = bandScores(scenario);
      expect(Math.min(...scores), scenario.name).toBeCloseTo(BRIEF_MET_SHARE, 6);
      expect(Math.max(...scores), scenario.name).toBeCloseTo(1, 6);
    }
  });

  it("makes the second star something a finished week can miss, and can earn", () => {
    const worstWeek =
      SCENARIOS.reduce((total, scenario) => total + Math.min(...bandScores(scenario)), 0) /
      SCENARIOS.length;
    const bestWeek =
      SCENARIOS.reduce((total, scenario) => total + Math.max(...bandScores(scenario)), 0) /
      SCENARIOS.length;
    expect(worstWeek).toBeLessThan(HIGH_SCORE_THRESHOLD);
    expect(bestWeek).toBeGreaterThanOrEqual(HIGH_SCORE_THRESHOLD);
  });

  it("scores higher wherever the headline metric is higher inside the band", () => {
    for (const scenario of SCENARIOS) {
      const samples = samplesFor(scenario);
      const results = winnableThresholds(scenario).map((threshold) =>
        judgeRound({ scenario, samples, threshold }),
      );
      for (const a of results) {
        for (const b of results) {
          const pa = metricValue(a.metrics, scenario.primary);
          const pb = metricValue(b.metrics, scenario.primary);
          if (pa > pb + 1e-12) expect(a.score, scenario.name).toBeGreaterThan(b.score);
        }
      }
    }
  });

  it("reports the headline metric's range across the band on a win, and only then", () => {
    const scenario = scenarioAt(2);
    const samples = samplesFor(scenario);
    const band = winningBand(scenario, samples);
    const win = judgeRound({ scenario, samples, threshold: band[0]!.threshold });
    expect(win.primaryRange).toEqual({
      min: Math.min(...band.map((entry) => entry.primary)),
      max: Math.max(...band.map((entry) => entry.primary)),
    });
    expect(judgeRound({ scenario, samples, threshold: 0 }).primaryRange).toBeNull();
  });

  it("uses exactly the slider's stops for the band", () => {
    expect(SLIDER_THRESHOLDS).toHaveLength(101);
    expect(SLIDER_THRESHOLDS[7]).toBe(0.07);
    expect(SLIDER_THRESHOLDS[100]).toBe(1);
    for (const scenario of SCENARIOS) {
      expect(
        winningBand(scenario, samplesFor(scenario)).map((entry) => entry.threshold),
      ).toEqual(winnableThresholds(scenario));
    }
  });
});

describe("the why-cards tell the truth about the numbers they print", () => {
  it("says precision rose when loosening raised it, and fell when it fell", () => {
    // "Every case you add to the pile … can only hurt precision" was printed
    // next to numbers showing precision going up (prank 0.91 → 0.90). The card
    // now reports what this notch did, so check it against every notch.
    let rose = 0;
    let fell = 0;
    for (const scenario of SCENARIOS) {
      const samples = samplesFor(scenario);
      for (let step = 100; step > 0; step -= 1) {
        const previous = SLIDER_THRESHOLDS[step]!;
        const threshold = SLIDER_THRESHOLDS[step - 1]!;
        const before = metricsOf(confusionAt(samples, previous));
        const after = metricsOf(confusionAt(samples, threshold));
        const card = whyCardFor({
          kind: "threshold-moved",
          scenario,
          samples,
          threshold,
          previous,
        });
        expect(card.body).not.toMatch(/can only hurt precision/i);
        const flaggedBefore = confusionAt(samples, previous);
        if (flaggedBefore.truePositives + flaggedBefore.falsePositives === 0) continue;
        if (card.title.includes("nothing crossed")) continue;
        if (after.precision > before.precision) {
          rose += 1;
          expect(card.body, `${scenario.name} ${previous}→${threshold}`).toMatch(
            /so precision rose/,
          );
        } else if (after.precision < before.precision) {
          fell += 1;
          expect(card.body, `${scenario.name} ${previous}→${threshold}`).toMatch(
            /so precision fell/,
          );
        }
      }
    }
    // Both directions genuinely happen, which is why the card cannot assert one.
    expect(rose).toBeGreaterThan(0);
    expect(fell).toBeGreaterThan(0);
  });

  it("never tells an allergen player that recall is not what the critic pays for", () => {
    const allergen = scenarioAt(2);
    const result = judgeRound({
      scenario: allergen,
      samples: samplesFor(allergen),
      threshold: 0,
    });
    const card = whyCardFor({
      kind: "served",
      scenario: allergen,
      result,
      attempts: 1,
      complete: false,
    });
    expect(card.body).not.toMatch(/not the one the critic is paying for/i);
    expect(card.body).toContain(`the expensive mistake is the miss — ${allergen.falseNegativeCost}`);
  });

  it("names the metric the critic IS paying for on the wrong side", () => {
    const pranks = scenarioAt(1);
    const result = judgeRound({
      scenario: pranks,
      samples: samplesFor(pranks),
      threshold: 0.02,
    });
    const card = whyCardFor({
      kind: "served",
      scenario: pranks,
      result,
      attempts: 1,
      complete: false,
    });
    expect(card.body).toMatch(/Recall is nearly perfect, and it is not the one the critic is paying for — precision is/);
    expect(card.body).toContain(
      `On this shift ${pranks.falsePositiveCost}, while ${pranks.falseNegativeCost}`,
    );
  });

  it("spells every plural out rather than appending an s", () => {
    // "contaminated dishs", "badly plated dishs" used to sit in the matrix.
    expect(SCENARIOS.map((scenario) => scenario.positiveLabelPlural)).toEqual([
      "prank orders",
      "contaminated dishes",
      "badly plated dishes",
      "fraudulent payments",
    ]);
    for (const scenario of SCENARIOS) {
      const naive = `${scenario.positiveLabel}s`;
      if (naive === scenario.positiveLabelPlural) continue;
      const samples = samplesFor(scenario);
      const released = whyCardFor({
        kind: "threshold-moved",
        scenario,
        samples,
        threshold: 0.9,
        previous: 0.1,
      });
      const judged = [0.5, 0.8, 0.99].map(
        (threshold) =>
          judgeRound({ scenario, samples, threshold }).failure?.detail ?? "",
      );
      for (const text of [released.body, ...judged]) {
        expect(text, scenario.name).not.toContain(naive);
      }
    }
  });
});

describe("the store cannot double-count, un-clear, or wipe a shift", () => {
  const store = useChefStore;

  /** Park the cutoff in the middle of the current shift's band. */
  const aimAtBand = () => {
    const band = winnableThresholds(scenarioAt(store.getState().scenarioIndex));
    store.getState().setThreshold(band[Math.floor(band.length / 2)]!);
  };

  beforeEach(() => {
    useProgression.setState({
      ...EMPTY_PROGRESSION,
      hydrated: true,
      syncError: null,
      lastGain: null,
    });
    useProgression.getState().setAdapter(createMemoryAdapter());
    store.getState().restart();
    store.getState().setLane("visual");
  });

  it("banks a cleared shift once, however many times it is served", () => {
    // Running the starter snippet twice used to read "2 of 4 signed off" while
    // still on shift one.
    aimAtBand();
    store.getState().serve();
    store.getState().serve();
    store.getState().serve("code");
    expect(store.getState().clearedScores).toHaveLength(1);
    expect(store.getState().phase).toBe("cleared");
    expect(store.getState().attempts).toBe(1);
  });

  it("does not un-clear a shift when a worse cutoff is served after it", () => {
    aimAtBand();
    const won = store.getState().serve();
    store.getState().setThreshold(0.2);
    const judged = store.getState().serve();

    // The caller still gets an honest verdict on the cutoff it asked about…
    expect(judged.outcome).not.toBe("win");
    expect(judged.threshold).toBe(0.2);
    // …but the state keeps the shift signed off with its winning judgement.
    expect(store.getState().phase).toBe("cleared");
    expect(store.getState().servedResult).toEqual(won);
    expect(store.getState().failure).toBeNull();
    expect(store.getState().clearedScores).toEqual([won.score]);
  });

  it("records a finished week with progression exactly once", () => {
    for (let shift = 1; shift <= SCENARIOS.length; shift += 1) {
      aimAtBand();
      store.getState().serve();
      if (shift < SCENARIOS.length) store.getState().nextShift();
    }
    expect(store.getState().phase).toBe("complete");
    expect(store.getState().clearedScores).toHaveLength(SCENARIOS.length);
    const played = useProgression.getState().games[SLUG]?.playCount;
    expect(played).toBe(1);

    store.getState().serve();
    store.getState().serve("code");
    expect(useProgression.getState().games[SLUG]?.playCount).toBe(played);
  });

  it("retries the current shift and keeps the ones already signed off", () => {
    aimAtBand();
    store.getState().serve();
    store.getState().nextShift();
    store.getState().setThreshold(1);
    store.getState().serve();
    expect(store.getState().failure).not.toBeNull();

    store.getState().retryShift();
    expect(store.getState().scenarioIndex).toBe(2);
    expect(store.getState().clearedScores).toHaveLength(1);
    expect(store.getState().failure).toBeNull();
    expect(store.getState().servedResult).toBeNull();
    expect(store.getState().threshold).toBe(0.5);
    expect(store.getState().phase).toBe("tuning");
  });

  it("replaces a retried shift's score rather than appending it, keeping the best", () => {
    const band = winnableThresholds(scenarioAt(1));
    store.getState().setThreshold(band[band.length - 1]!);
    const best = store.getState().serve().score;
    store.getState().retryShift();
    store.getState().setThreshold(band[0]!);
    const worse = store.getState().serve().score;
    expect(worse).toBeLessThan(best);
    expect(store.getState().clearedScores).toEqual([best]);
  });

  it("counts the third star from the lane that cleared, not the tab that is open", () => {
    // Every shift served from the visual button, with the Code tab showing.
    store.getState().setLane("code");
    for (let shift = 1; shift <= SCENARIOS.length; shift += 1) {
      aimAtBand();
      store.getState().serve("visual");
      if (shift < SCENARIOS.length) store.getState().nextShift();
    }
    expect(useProgression.getState().games[SLUG]?.codeLaneCleared).toBe(false);
  });

  it("awards the code-lane clear when a shift is cleared through the api", () => {
    // Shift one cleared by the script, the rest by hand in the visual lane —
    // exactly what running the starter snippet and then playing on does.
    const api = createCodeApi();
    aimAtBand();
    expect(api.serve().outcome).toBe("win");
    expect(store.getState().codeLaneWin).toBe(true);
    store.getState().nextShift();
    for (let shift = 2; shift <= SCENARIOS.length; shift += 1) {
      aimAtBand();
      store.getState().serve("visual");
      if (shift < SCENARIOS.length) store.getState().nextShift();
    }
    expect(useProgression.getState().games[SLUG]?.codeLaneCleared).toBe(true);
  });

  it("treats a click event handed to serve as a visual-lane serve", () => {
    aimAtBand();
    // What `onClick={serve}` would pass.
    (store.getState().serve as (source: unknown) => unknown)({ type: "click" });
    expect(store.getState().codeLaneWin).toBe(false);
  });

  it("runs the starter snippet to the top of the band, and can run it twice", async () => {
    const run = async () => {
      const logs: string[] = [];
      await createJsExecutor<ReturnType<typeof createCodeApi>>()(STARTER_CODE, {
        api: createCodeApi(),
        log: (...args: unknown[]) => logs.push(args.map(String).join(" ")),
        checkBudget: () => {},
      });
      return logs;
    };

    const first = await run();
    expect(first.some((line) => /verdict: win/.test(line))).toBe(true);
    // It pushes the shift's headline metric as far as the band allows, which
    // is exactly what the score rewards.
    expect(store.getState().servedResult?.score).toBeCloseTo(1, 9);
    expect(store.getState().codeLaneWin).toBe(true);

    await run();
    expect(store.getState().clearedScores).toHaveLength(1);
    expect(store.getState().phase).toBe("cleared");
  });

  it("rejects bad script input with a named error instead of corrupting state", () => {
    const api = createCodeApi();
    const before = store.getState().threshold;
    expect(() => api.setThreshold(Number.NaN)).toThrow(/finite number/);
    expect(() => api.setThreshold("0.5" as unknown as number)).toThrow(/finite number/);
    expect(() => api.setThreshold(1.5)).toThrow(/between 0 and 1/);
    expect(() => api.metricsAt(Number.POSITIVE_INFINITY)).toThrow(/metricsAt/);
    expect(() => api.matrixAt(-0.1)).toThrow(/matrixAt/);
    expect(store.getState().threshold).toBe(before);
    // And the store itself ignores a NaN from any other caller.
    store.getState().setThreshold(Number.NaN);
    expect(store.getState().threshold).toBe(before);
  });
});
