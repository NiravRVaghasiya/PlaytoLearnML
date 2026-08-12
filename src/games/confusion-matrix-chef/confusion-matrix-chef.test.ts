import { beforeEach, describe, expect, it } from "vitest";
import {
  OVERSHOOT_LEVEL,
  PARADOX_BASELINE,
  SAMPLE_COUNT,
  SCENARIOS,
  auc,
  confusionAt,
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
  type Sample,
  type Scenario,
} from "./ml";
import { useChefStore } from "./store";

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

  it("names the wrong side of the tradeoff when precision is bought with recall", () => {
    // Prank shift at a high cutoff: precision is ~97%, which SATISFIES the shift's
    // headline metric, while recall collapses and fails the brief. Judging on the
    // headline metric would have called this a near miss.
    const result = judgeRound({
      scenario: pranks,
      samples: samplesFor(pranks),
      threshold: 0.9,
    });
    expect(result.metrics.precision).toBeGreaterThan(OVERSHOOT_LEVEL);
    expect(result.outcome).toBe("wrong-side");
    expect(result.failure?.name).toBe("Wrong side of the tradeoff");
    expect(result.failure?.detail).toMatch(/lower the threshold/i);
  });

  it("names the wrong side when recall is bought with precision", () => {
    // The mirror image, and the advice must reverse with it.
    const result = judgeRound({
      scenario: pranks,
      samples: samplesFor(pranks),
      threshold: 0.02,
    });
    expect(result.metrics.recall).toBeGreaterThan(OVERSHOOT_LEVEL);
    expect(result.outcome).toBe("wrong-side");
    expect(result.failure?.detail).toMatch(/raise the threshold/i);
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
