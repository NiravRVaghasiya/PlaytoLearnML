import { beforeEach, describe, expect, it } from "vitest";
import { createElement } from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import {
  EMPTY_PROGRESSION,
  createMemoryAdapter,
  useProgression,
} from "@/engine/progression";
import { seededRandom } from "@/lib/utils";
import {
  CAPACITY_TOLERANCE,
  KNOT_COUNTS,
  MIN_PARAMS,
  NOISE_RATE,
  OVERFIT_GAP,
  PENALTY_PER_EXTRA_PARAM,
  ROUND_SEEDS,
  SPREAD,
  TEST_SIZE,
  TRAIN_SIZE,
  WIN_SCORE,
  accuracyOf,
  boundaryAt,
  capacityCeiling,
  capacityOf,
  classifyPoint,
  complexityCostOf,
  evaluate,
  fitKnots,
  generateTestPoints,
  generateTrainPoints,
  independentCeiling,
  penaltyFor,
  resampleParams,
  scoreFor,
  seedForRound,
  trueBoundary,
  type BoundaryType,
  type Point,
} from "./ml";
import { useSortItStore } from "./store";
import { whyCardFor } from "./why-cards";
import { STARTER_CODE, createSortItApi } from "./CodeLane";
import { VisualLane } from "./VisualLane";
import { clientToUser, nearestKnotIndex, type ScreenMatrix } from "./field";

const flat = (type: BoundaryType) =>
  Array.from({ length: KNOT_COUNTS[type] }, () => 0.5);

const fittedFor = (type: BoundaryType, train: Point[]) =>
  fitKnots(train, flat(type));

// ═══════════════════════════════════════════════════════════════════════════
// THE LESSON
//
// This is the most important block in the file. Sort-It Arcade exists to teach
// one sentence: "a simpler boundary that misses a few points often generalises
// better than a perfect-but-complex one." If these assertions fail, the game is
// teaching something false and must not ship — regardless of whether the UI
// works.
// ═══════════════════════════════════════════════════════════════════════════

describe("the lesson holds on every shipped round", () => {
  it.each([...ROUND_SEEDS])("seed %i teaches the intended lesson", (seed) => {
    const train = generateTrainPoints(seed);
    const test = generateTestPoints(seed);

    const line = evaluate(train, test, fittedFor("line", train));
    const curve = evaluate(train, test, fittedFor("curve", train));
    const wiggle = evaluate(train, test, fittedFor("wiggle", train));

    // 1. The minimum-capacity model must never be accused of overfitting. Two
    //    parameters cannot memorise 200 points; a gap there is variance.
    expect(line.outcome, `line on seed ${seed}`).not.toBe("overfit");
    // ...and "Fit it for me" on a line must be named for what it is.
    expect(line.outcome, `fitted line on seed ${seed}`).toBe("underfit");

    // 1b. No straight line may clear the round — searched exhaustively, not by
    //     the optimizer's own line, which stalls short of the best one. Round 1
    //     used to be a seed where a hand-placed line reached exactly 80% and
    //     won, with a card saying two parameters "followed the real boundary".
    //     The margin covers placements finer than the search grid: a grid four
    //     times finer never found more than one extra point on these seeds.
    const lineCeiling = independentCeiling(train, KNOT_COUNTS.line);
    expect(lineCeiling, `line ceiling on seed ${seed}`).toBeLessThan(
      WIN_SCORE - 0.015,
    );

    // 2. The mid-capacity model is the intended answer, so it must be winnable.
    expect(curve.outcome, `curve on seed ${seed}`).toBe("win");
    expect(curve.score).toBeGreaterThanOrEqual(WIN_SCORE);

    // 3. The over-capacity model must overfit: better on train, worse on
    //    held-out data. This is the trap the game is built around.
    expect(wiggle.outcome, `wiggle on seed ${seed}`).toBe("overfit");
    expect(wiggle.trainAccuracy).toBeGreaterThan(curve.trainAccuracy);
    expect(wiggle.testAccuracy).toBeLessThan(curve.testAccuracy);
    expect(wiggle.generalizationGap).toBeGreaterThanOrEqual(OVERFIT_GAP);

    // 4. And it must "quietly lose points" (spec's wording) once the
    //    complexity penalty is applied.
    expect(wiggle.score).toBeLessThan(curve.score);

    // 5. The strongest form of 4: even at the ceiling of what 25 parameters can
    //    fit on these points, the score stays under the win bar. Without this
    //    the wiggle can squeak a win on a lucky fit, rewarding precisely the
    //    behaviour the game exists to punish. A real browser playthrough found
    //    exactly that at score 0.802 against a 0.80 bar.
    const wiggleCeiling =
      capacityCeiling(train, flat("wiggle")) - penaltyFor(flat("wiggle"));
    expect(wiggleCeiling, `wiggle ceiling on seed ${seed}`).toBeLessThan(
      WIN_SCORE,
    );
  });

  /**
   * The lesson has to hold on the routes players actually take, not only from a
   * flat start. Round 5 used to be a seed where the flat-start curve won by
   * five thousandths while about 40% of ordinary 5-parameter fits were branded
   * "memorised the noise" — with training accuracy below the true boundary's.
   */
  it.each([...ROUND_SEEDS])(
    "seed %i: the curve is judged the same from any reasonable start",
    (seed) => {
      const train = generateTrainPoints(seed);
      const test = generateTestPoints(seed);
      const random = seededRandom(seed * 3 + 1);
      const outcomes: string[] = [];

      for (let i = 0; i < 40; i += 1) {
        // Hand-like: roughly the true shape, as a player tracing it would.
        const nearTruth = [0, 0.25, 0.5, 0.75, 1].map(
          (x) => trueBoundary(x) + (random() - 0.5) * 0.1,
        );
        outcomes.push(evaluate(train, test, fitKnots(train, nearTruth)).outcome);
        // Arbitrary: wherever the handles happen to be when Fit is pressed.
        const arbitrary = Array.from({ length: KNOT_COUNTS.curve }, () =>
          random(),
        );
        outcomes.push(evaluate(train, test, fitKnots(train, arbitrary)).outcome);
      }

      const overfit = outcomes.filter((o) => o === "overfit").length;
      expect(overfit, `curves branded overfit on seed ${seed}`).toBeLessThan(
        outcomes.length * 0.05,
      );
      // A curve stalled in a local optimum is unfitted, never "underfit": five
      // parameters clear every shipped round.
      expect(outcomes, `seed ${seed}`).not.toContain("underfit");
    },
  );

  it("names overfitting with numbers that back it up", () => {
    const seed = ROUND_SEEDS[0]!;
    const train = generateTrainPoints(seed);
    const test = generateTestPoints(seed);
    const result = evaluate(train, test, fittedFor("wiggle", train));

    expect(result.failure).not.toBeNull();
    expect(result.failure!.name).toBe("Overfitting");
    // The claim must be evidenced, not asserted.
    expect(result.failure!.detail).toMatch(/\d+%/);
    expect(result.trainAccuracy).toBeGreaterThan(result.testAccuracy);
  });

  it("names underfitting only when the model class is genuinely maxed out", () => {
    const seed = ROUND_SEEDS[0]!;
    const train = generateTrainPoints(seed);
    const test = generateTestPoints(seed);

    // A fitted line: at its ceiling, and that ceiling is poor → underfitting.
    const fitted = evaluate(train, test, fittedFor("line", train));
    expect(fitted.outcome).toBe("underfit");
    expect(fitted.failure!.name).toBe("Underfitting");
    expect(fitted.bestAtThisComplexity).toBeLessThan(WIN_SCORE);

    // An unfitted flat line is the same model class, with the same poor
    // ceiling, and scores worse still — but it is NOT underfitting: it simply
    // hasn't been fitted. Calling that underfitting would teach the wrong word.
    const unfitted = evaluate(train, test, [0.5, 0.5]);
    expect(unfitted.bestAtThisComplexity).toBe(fitted.bestAtThisComplexity);
    expect(unfitted.trainAccuracy).toBeLessThan(fitted.trainAccuracy);
    expect(unfitted.outcome).not.toBe("underfit");
    expect(unfitted.failure).toBeNull();
  });

  it.each([...ROUND_SEEDS])(
    "seed %i: the most carefully placed line is still named Underfitting",
    (seed) => {
      // A fixed 78% cut-off used to sit exactly on the best line on three
      // rounds, so a player who hand-tuned the line to its limit there got an
      // unnamed near-miss while a sloppier line got the name.
      const train = generateTrainPoints(seed);
      const test = generateTestPoints(seed);
      let best: number[] = [0.5, 0.5];
      let bestAccuracy = 0;
      for (let a = 0; a <= 100; a += 1) {
        for (let b = 0; b <= 100; b += 1) {
          const accuracy = accuracyOf(train, [a / 100, b / 100]);
          if (accuracy > bestAccuracy) {
            bestAccuracy = accuracy;
            best = [a / 100, b / 100];
          }
        }
      }
      const result = evaluate(train, test, best);
      expect(result.outcome, `line ${best} at ${bestAccuracy}`).toBe("underfit");
      expect(result.failure?.name).toBe("Underfitting");
    },
  );

  it("names a line at the class's exact ceiling", () => {
    // Round 6: the line [0.67, 0.34] sorts 78% — every line's best there.
    const seed = 20260809;
    const train = generateTrainPoints(seed);
    const line = [0.67, 0.34];
    expect(accuracyOf(train, line)).toBe(
      independentCeiling(train, KNOT_COUNTS.line),
    );
    const result = evaluate(train, generateTestPoints(seed), line);
    expect(result.failure?.detail).toMatch(
      /^78% train, and 78% is the best 2 parameters can do/,
    );
  });

  it("cannot reach the noise ceiling honestly", () => {
    // ~NOISE_RATE of labels are wrong by construction, so no boundary should
    // approach 100% on held-out data. If one did, the noise isn't working and
    // the overfitting lesson would be a lie.
    for (const seed of ROUND_SEEDS) {
      const train = generateTrainPoints(seed);
      const test = generateTestPoints(seed);
      const best = evaluate(train, test, fittedFor("wiggle", train));
      expect(best.testAccuracy).toBeLessThan(1 - NOISE_RATE / 2);
    }
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// The generative process
// ═══════════════════════════════════════════════════════════════════════════

describe("dataset generation", () => {
  it("is deterministic for a seed", () => {
    expect(generateTrainPoints(42)).toEqual(generateTrainPoints(42));
    expect(generateTrainPoints(42)).not.toEqual(generateTrainPoints(43));
  });

  it("produces the configured sizes", () => {
    expect(generateTrainPoints(1)).toHaveLength(TRAIN_SIZE);
    expect(generateTestPoints(1)).toHaveLength(TEST_SIZE);
  });

  it("draws train and test independently", () => {
    const train = generateTrainPoints(7);
    const test = generateTestPoints(7);
    const trainKeys = new Set(train.map((p) => `${p.x},${p.y}`));
    const overlap = test.filter((p) => trainKeys.has(`${p.x},${p.y}`));
    expect(overlap).toHaveLength(0);
  });

  it("keeps every point inside the field without clamping", () => {
    for (const point of generateTrainPoints(3)) {
      expect(point.x).toBeGreaterThanOrEqual(0);
      expect(point.x).toBeLessThanOrEqual(1);
      expect(point.y).toBeGreaterThan(0);
      expect(point.y).toBeLessThan(1);
    }
  });

  it("scatters points within SPREAD of the true boundary", () => {
    for (const point of generateTrainPoints(5)) {
      expect(Math.abs(point.y - trueBoundary(point.x))).toBeLessThanOrEqual(
        SPREAD + 1e-9,
      );
    }
  });

  it("flips labels at approximately NOISE_RATE", () => {
    // Pooled over every shipped round for a tight estimate.
    let total = 0;
    let flipped = 0;
    for (const seed of ROUND_SEEDS) {
      for (const point of generateTrainPoints(seed)) {
        total += 1;
        const clean = point.y > trueBoundary(point.x) ? 1 : 0;
        if (clean !== point.label) flipped += 1;
      }
    }
    expect(flipped / total).toBeGreaterThan(NOISE_RATE * 0.6);
    expect(flipped / total).toBeLessThan(NOISE_RATE * 1.4);
  });

  it("cycles round numbers through the vetted seeds", () => {
    expect(seedForRound(1)).toBe(ROUND_SEEDS[0]);
    expect(seedForRound(ROUND_SEEDS.length + 1)).toBe(ROUND_SEEDS[0]);
    expect(seedForRound(0)).toBe(ROUND_SEEDS[0]);
    expect(seedForRound(2)).toBe(ROUND_SEEDS[1]);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// The model
// ═══════════════════════════════════════════════════════════════════════════

describe("boundaryAt", () => {
  it("returns the knot heights at the knot positions", () => {
    const params = [0.2, 0.6, 0.4];
    expect(boundaryAt(params, 0)).toBeCloseTo(0.2, 10);
    expect(boundaryAt(params, 0.5)).toBeCloseTo(0.6, 10);
    expect(boundaryAt(params, 1)).toBeCloseTo(0.4, 10);
  });

  it("interpolates linearly between knots", () => {
    expect(boundaryAt([0, 1], 0.25)).toBeCloseTo(0.25, 10);
    expect(boundaryAt([0.2, 0.6, 0.4], 0.25)).toBeCloseTo(0.4, 10);
  });

  it("is constant for two equal knots — a horizontal line", () => {
    for (const x of [0, 0.3, 0.77, 1]) {
      expect(boundaryAt([0.42, 0.42], x)).toBeCloseTo(0.42, 10);
    }
  });

  it("clamps x outside the field", () => {
    expect(boundaryAt([0.1, 0.9], -1)).toBeCloseTo(0.1, 10);
    expect(boundaryAt([0.1, 0.9], 5)).toBeCloseTo(0.9, 10);
  });

  it("degrades gracefully on empty or single-knot params", () => {
    expect(boundaryAt([], 0.5)).toBe(0.5);
    expect(boundaryAt([0.3], 0.9)).toBe(0.3);
  });
});

describe("classifyPoint", () => {
  it("puts points above the boundary in class 1 and below in class 0", () => {
    const params = [0.5, 0.5];
    expect(classifyPoint(params, { x: 0.5, y: 0.9 })).toBe(1);
    expect(classifyPoint(params, { x: 0.5, y: 0.1 })).toBe(0);
  });

  it("treats a point exactly on the boundary as class 1", () => {
    expect(classifyPoint([0.5, 0.5], { x: 0.5, y: 0.5 })).toBe(1);
  });

  it("follows a sloped boundary", () => {
    const params = [0, 1]; // rises left to right
    expect(classifyPoint(params, { x: 0.1, y: 0.5 })).toBe(1);
    expect(classifyPoint(params, { x: 0.9, y: 0.5 })).toBe(0);
  });
});

describe("accuracyOf", () => {
  const points: Point[] = [
    { id: "a", x: 0.2, y: 0.9, label: 1, predictedSide: 1 },
    { id: "b", x: 0.4, y: 0.1, label: 0, predictedSide: 0 },
    { id: "c", x: 0.6, y: 0.8, label: 1, predictedSide: 1 },
    { id: "d", x: 0.8, y: 0.2, label: 0, predictedSide: 0 },
  ];

  it("scores a perfect separator at 1", () => {
    expect(accuracyOf(points, [0.5, 0.5])).toBe(1);
  });

  it("scores an inverted separator at 0", () => {
    // Boundary above everything: all points predicted 0, half are labelled 1.
    expect(accuracyOf(points, [1.1, 1.1])).toBe(0.5);
    // Boundary below everything: all predicted 1.
    expect(accuracyOf(points, [-0.1, -0.1])).toBe(0.5);
  });

  it("returns 0 for no points rather than dividing by zero", () => {
    expect(accuracyOf([], [0.5, 0.5])).toBe(0);
  });

  it("is a real computation, not a stored field", () => {
    // Points carry a stale `predictedSide`; accuracy must ignore it and
    // recompute from geometry.
    const stale = points.map((p) => ({ ...p, predictedSide: 1 as const }));
    expect(accuracyOf(stale, [0.5, 0.5])).toBe(1);
    expect(accuracyOf(stale, [1.1, 1.1])).toBe(0.5);
  });
});

describe("complexity and scoring", () => {
  it("counts parameters", () => {
    expect(complexityCostOf([0.5, 0.5])).toBe(2);
    expect(complexityCostOf(flat("wiggle"))).toBe(KNOT_COUNTS.wiggle);
  });

  it("charges nothing for the simplest boundary", () => {
    expect(penaltyFor(flat("line"))).toBe(0);
  });

  it("charges per extra parameter", () => {
    expect(penaltyFor(flat("curve"))).toBeCloseTo(
      PENALTY_PER_EXTRA_PARAM * (KNOT_COUNTS.curve - MIN_PARAMS),
      10,
    );
    expect(penaltyFor(flat("wiggle"))).toBeGreaterThan(
      penaltyFor(flat("curve")),
    );
  });

  it("computes score as accuracy minus penalty", () => {
    expect(scoreFor(0.9, flat("line"))).toBeCloseTo(0.9, 10);
    expect(scoreFor(0.9, flat("curve"))).toBeCloseTo(
      0.9 - penaltyFor(flat("curve")),
      10,
    );
  });

  it("clamps score into 0..1", () => {
    expect(scoreFor(0.01, flat("wiggle"))).toBe(0);
    expect(scoreFor(1, flat("line"))).toBe(1);
  });
});

describe("resampleParams", () => {
  it("preserves a straight line's shape at any resolution", () => {
    const resampled = resampleParams([0.2, 0.8], 5);
    expect(resampled).toHaveLength(5);
    resampled.forEach((height, index) => {
      expect(height).toBeCloseTo(0.2 + (0.8 - 0.2) * (index / 4), 10);
    });
  });

  it("keeps the endpoints", () => {
    const resampled = resampleParams([0.1, 0.7, 0.3], 9);
    expect(resampled[0]).toBeCloseTo(0.1, 10);
    expect(resampled.at(-1)).toBeCloseTo(0.3, 10);
  });

  it("leaves classification unchanged when only resolution changes", () => {
    const train = generateTrainPoints(ROUND_SEEDS[0]!);
    const coarse = [0.3, 0.7];
    const fine = resampleParams(coarse, 25);
    expect(accuracyOf(train, fine)).toBeCloseTo(accuracyOf(train, coarse), 10);
  });
});

describe("fitKnots", () => {
  it("never makes training accuracy worse", () => {
    const train = generateTrainPoints(ROUND_SEEDS[1]!);
    for (const type of ["line", "curve", "wiggle"] as BoundaryType[]) {
      const start = flat(type);
      const fitted = fitKnots(train, start);
      expect(accuracyOf(train, fitted)).toBeGreaterThanOrEqual(
        accuracyOf(train, start),
      );
    }
  });

  it("meaningfully improves on an unfitted boundary", () => {
    const train = generateTrainPoints(ROUND_SEEDS[0]!);
    const start = flat("curve");
    const fitted = fitKnots(train, start);
    expect(accuracyOf(train, fitted)).toBeGreaterThan(
      accuracyOf(train, start) + 0.1,
    );
  });

  it("preserves the parameter count — it fits, it doesn't add capacity", () => {
    const train = generateTrainPoints(ROUND_SEEDS[0]!);
    expect(fitKnots(train, flat("curve"))).toHaveLength(KNOT_COUNTS.curve);
  });

  it("is deterministic", () => {
    const train = generateTrainPoints(ROUND_SEEDS[0]!);
    expect(fitKnots(train, flat("curve"))).toEqual(
      fitKnots(train, flat("curve")),
    );
  });

  it("gives higher capacity a higher ceiling", () => {
    const train = generateTrainPoints(ROUND_SEEDS[0]!);
    expect(capacityOf(train, flat("wiggle"))).toBeGreaterThan(
      capacityOf(train, flat("line")),
    );
  });

  it("copes with an empty point set", () => {
    expect(fitKnots([], [0.5, 0.5])).toEqual([0.5, 0.5]);
  });

  it("fits exactly what a full recount after every candidate would", () => {
    // fitKnots only recounts the points a knot can move. That is a speed-up,
    // not a change of optimizer: every seed above was screened against the
    // full-recount version below, so the two must agree knot for knot.
    const reference = (points: Point[], params: number[]) => {
      const fitted = [...params];
      for (let pass = 0; pass < 3; pass += 1) {
        let improved = false;
        for (let index = 0; index < fitted.length; index += 1) {
          const original = fitted[index]!;
          let bestHeight = original;
          let bestAccuracy = accuracyOf(points, fitted);
          for (let step = 0; step <= 40; step += 1) {
            fitted[index] = step / 40;
            const candidate = accuracyOf(points, fitted);
            if (candidate > bestAccuracy) {
              bestAccuracy = candidate;
              bestHeight = step / 40;
            }
          }
          fitted[index] = bestHeight;
          if (bestHeight !== original) improved = true;
        }
        if (!improved) break;
      }
      return fitted;
    };

    const random = seededRandom(99);
    for (const seed of [ROUND_SEEDS[0]!, ROUND_SEEDS[5]!]) {
      const train = generateTrainPoints(seed);
      for (const type of ["line", "curve", "wiggle"] as BoundaryType[]) {
        const starts = [
          flat(type),
          Array.from({ length: KNOT_COUNTS[type] }, () => 0),
          Array.from({ length: KNOT_COUNTS[type] }, () => random()),
        ];
        for (const start of starts) {
          expect(fitKnots(train, start), `${type} on ${seed}`).toEqual(
            reference(train, start),
          );
        }
      }
    }
  });
});

describe("evaluate", () => {
  const train = generateTrainPoints(ROUND_SEEDS[0]!);
  const test = generateTestPoints(ROUND_SEEDS[0]!);

  it("reports the parameter count and penalty it used", () => {
    const result = evaluate(train, test, flat("curve"));
    expect(result.paramCount).toBe(KNOT_COUNTS.curve);
    expect(result.penalty).toBeCloseTo(penaltyFor(flat("curve")), 10);
  });

  it("computes the gap as train minus test", () => {
    const params = fittedFor("wiggle", train);
    const result = evaluate(train, test, params);
    expect(result.generalizationGap).toBeCloseTo(
      result.trainAccuracy - result.testAccuracy,
      10,
    );
  });

  it("refuses to call a two-parameter model overfitting", () => {
    // Construct a line with a deliberately large gap and confirm the guard
    // holds: capacity, not gap alone, licenses the word.
    const rigged: Point[] = train.map((p, i) => ({
      ...p,
      label: (i % 2 === 0 ? 1 : 0) as 0 | 1,
    }));
    const result = evaluate(rigged, test, [0.5, 0.5]);
    expect(result.paramCount).toBe(MIN_PARAMS);
    expect(result.outcome).not.toBe("overfit");
  });

  it("reports bestAtThisComplexity as the class's ceiling, whatever the start", () => {
    const params = flat("curve");
    const result = evaluate(train, test, params);
    expect(result.bestAtThisComplexity).toBeCloseTo(
      capacityCeiling(train, params),
      10,
    );
    // Never below what the optimizer reaches from the player's own knots, nor
    // below the player's own accuracy — it can't contradict a visible number.
    expect(result.bestAtThisComplexity).toBeGreaterThanOrEqual(
      capacityOf(train, params),
    );
    expect(result.bestAtThisComplexity).toBeGreaterThanOrEqual(
      result.trainAccuracy,
    );
    expect(result.bestAtThisComplexity).toBeGreaterThanOrEqual(
      result.trainAccuracy - CAPACITY_TOLERANCE,
    );
    // And the same capacity reports the same ceiling from a poor start.
    const stalled = evaluate(train, test, fitKnots(train, [0, 0, 0, 0, 0]));
    expect(stalled.bestAtThisComplexity).toBeGreaterThanOrEqual(
      independentCeiling(train, KNOT_COUNTS.curve),
    );
  });

  it("does not call a curve stalled in a local optimum underfitting", () => {
    // Round 6, every handle pressed to the bottom, then "Fit it for me": the
    // greedy fit stalls at 75% and used to be told "75% is the best 5
    // parameters can do" — on a round where 5 parameters reach 88%.
    const seed = 20260809;
    expect(ROUND_SEEDS).toContain(seed);
    const roundTrain = generateTrainPoints(seed);
    const roundTest = generateTestPoints(seed);
    const stalled = fitKnots(roundTrain, [0, 0, 0, 0, 0]);
    const result = evaluate(roundTrain, roundTest, stalled);

    // As low as a line gets — but five parameters can clear this round.
    expect(result.trainAccuracy).toBeLessThan(WIN_SCORE);
    expect(result.bestAtThisComplexity).toBeGreaterThanOrEqual(WIN_SCORE);
    expect(result.outcome).not.toBe("underfit");
    expect(result.failure).toBeNull();
    expect(result.bestAtThisComplexity).toBeGreaterThanOrEqual(
      capacityOf(roundTrain, flat("curve")),
    );
    // The near-miss copy quotes the real ceiling and says it's reachable.
    const card = whyCardFor({ kind: "checked", evaluation: result });
    expect(card.body).toContain(
      `${Math.round(result.bestAtThisComplexity * 100)}%`,
    );
    expect(card.body).toMatch(/still room/);
  });

  it("finds the best straight line exactly, on the classifier's own arithmetic", () => {
    // The specialised line search must agree with `accuracyOf` — the thing the
    // player's metric uses — on the very same grid, or the ceiling it quotes
    // would be a different model's.
    let best = 0;
    for (let a = 0; a <= 200; a += 1) {
      for (let b = 0; b <= 200; b += 1) {
        best = Math.max(best, accuracyOf(train, [a / 200, b / 200]));
      }
    }
    expect(independentCeiling(train, KNOT_COUNTS.line)).toBeCloseTo(best, 12);
  });

  it("returns no failure on a win", () => {
    const result = evaluate(train, test, fittedFor("curve", train));
    expect(result.outcome).toBe("win");
    expect(result.failure).toBeNull();
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Store: the live-feedback binding
// ═══════════════════════════════════════════════════════════════════════════

describe("store", () => {
  beforeEach(() => {
    useProgression.setState({
      ...EMPTY_PROGRESSION,
      hydrated: true,
      syncError: null,
      lastGain: null,
    });
    useProgression.getState().setAdapter(createMemoryAdapter());
    useSortItStore.getState().startRound(1);
  });

  it("starts on round 1 with a flat line and no verdict", () => {
    const state = useSortItStore.getState();
    expect(state.round).toBe(1);
    expect(state.seed).toBe(ROUND_SEEDS[0]);
    expect(state.boundary.type).toBe("line");
    expect(state.boundary.params).toEqual([0.5, 0.5]);
    expect(state.testAccuracy).toBeNull();
    expect(state.failure).toBeNull();
    expect(state.won).toBe(false);
  });

  it("derives the metric from the real classifier, not a stored number", () => {
    useSortItStore.getState().setKnot(0, 0.3);
    const state = useSortItStore.getState();

    // Independent recomputation must match exactly.
    const expected = accuracyOf(state.trainPoints, state.boundary.params);
    expect(state.accuracy).toBeCloseTo(expected, 12);

    // And each point's recorded side must match the classifier.
    for (const point of state.trainPoints) {
      expect(point.predictedSide).toBe(
        classifyPoint(state.boundary.params, point),
      );
    }
  });

  it("moves the metric when the player moves the boundary", () => {
    const before = useSortItStore.getState().accuracy;
    useSortItStore.getState().setKnot(0, 0.05);
    useSortItStore.getState().setKnot(1, 0.95);
    expect(useSortItStore.getState().accuracy).not.toBe(before);
  });

  it("tracks misclassified points consistently with accuracy", () => {
    useSortItStore.getState().autoFit();
    const state = useSortItStore.getState();
    const total = state.trainPoints.length;
    expect(state.misclassifiedIds).toHaveLength(
      Math.round((1 - state.accuracy) * total),
    );
  });

  it("clamps knot heights into the field", () => {
    useSortItStore.getState().setKnot(0, 5);
    expect(useSortItStore.getState().boundary.params[0]).toBe(1);
    useSortItStore.getState().setKnot(0, -5);
    expect(useSortItStore.getState().boundary.params[0]).toBe(0);
  });

  it("ignores out-of-range knot indices", () => {
    const before = useSortItStore.getState().boundary.params;
    useSortItStore.getState().setKnot(99, 0.5);
    useSortItStore.getState().setKnot(-1, 0.5);
    expect(useSortItStore.getState().boundary.params).toEqual(before);
  });

  it("nudges relative to the current height", () => {
    useSortItStore.getState().setKnot(0, 0.5);
    useSortItStore.getState().nudgeKnot(0, 0.1);
    expect(useSortItStore.getState().boundary.params[0]).toBeCloseTo(0.6, 10);
  });

  it("changes capacity, updates the penalty, and preserves the shape", () => {
    useSortItStore.getState().setKnot(0, 0.2);
    useSortItStore.getState().setKnot(1, 0.8);
    const before = useSortItStore.getState();
    const shapeBefore = boundaryAt(before.boundary.params, 0.5);

    useSortItStore.getState().setBoundaryType("curve");
    const after = useSortItStore.getState();

    expect(after.boundary.type).toBe("curve");
    expect(after.boundary.params).toHaveLength(KNOT_COUNTS.curve);
    expect(after.boundary.complexityCost).toBe(KNOT_COUNTS.curve);
    expect(after.penalty).toBeGreaterThan(before.penalty);
    expect(boundaryAt(after.boundary.params, 0.5)).toBeCloseTo(shapeBefore, 10);
    // Same boundary shape means the same accuracy — capacity alone changes.
    expect(after.accuracy).toBeCloseTo(before.accuracy, 10);
  });

  it("does nothing when the capacity is already selected", () => {
    const before = useSortItStore.getState().boundary;
    useSortItStore.getState().setBoundaryType("line");
    expect(useSortItStore.getState().boundary).toBe(before);
  });

  it("hides the held-out score until the player checks", () => {
    expect(useSortItStore.getState().testAccuracy).toBeNull();
    useSortItStore.getState().autoFit();
    // Fitting must not leak the test set — that would make it a second
    // training set, which is the mistake the game teaches against.
    expect(useSortItStore.getState().testAccuracy).toBeNull();

    useSortItStore.getState().check();
    expect(useSortItStore.getState().testAccuracy).not.toBeNull();
  });

  it("re-hides the verdict after the boundary changes again", () => {
    useSortItStore.getState().check();
    expect(useSortItStore.getState().testAccuracy).not.toBeNull();
    useSortItStore.getState().setKnot(0, 0.4);
    expect(useSortItStore.getState().testAccuracy).toBeNull();
    expect(useSortItStore.getState().failure).toBeNull();
  });

  it("wins with the mid-capacity fit and awards XP once", () => {
    useSortItStore.getState().setBoundaryType("curve");
    useSortItStore.getState().autoFit();
    const evaluation = useSortItStore.getState().check();

    expect(evaluation.outcome).toBe("win");
    expect(useSortItStore.getState().won).toBe(true);
    expect(useSortItStore.getState().failure).toBeNull();

    const xpAfterFirst = useProgression.getState().xp;
    expect(xpAfterFirst).toBeGreaterThan(0);
    expect(useProgression.getState().games["sort-it-arcade"]?.stars).toBe(2);

    // Checking the same boundary again must not pay — or count — twice.
    useSortItStore.getState().check();
    expect(useProgression.getState().xp).toBe(xpAfterFirst);
    expect(useProgression.getState().games["sort-it-arcade"]?.playCount).toBe(1);
    expect(useSortItStore.getState().won).toBe(true);
  });

  it("names the overfit failure when the player over-fits, and awards nothing", () => {
    useSortItStore.getState().setBoundaryType("wiggle");
    useSortItStore.getState().autoFit();
    const evaluation = useSortItStore.getState().check();

    expect(evaluation.outcome).toBe("overfit");
    const failure = useSortItStore.getState().failure;
    expect(failure?.name).toBe("Overfitting");
    expect(failure?.detail).toContain("%");
    expect(useSortItStore.getState().won).toBe(false);
    expect(useProgression.getState().xp).toBe(0);
  });

  it("names round 1's curve-then-wiggle route overfitting", () => {
    // The browser playthrough wins with the curve, then runs a wiggle fit on
    // top of it from the code lane and expects the named failure. On most
    // rounds that route is an honest near-miss; round 1 must stay the one
    // where it isn't, or the playthrough's contract-#4 check has nothing to see.
    useSortItStore.getState().setBoundaryType("curve");
    useSortItStore.getState().autoFit();
    expect(useSortItStore.getState().check().outcome).toBe("win");
    useSortItStore.getState().setBoundaryType("wiggle");
    useSortItStore.getState().autoFit();
    expect(useSortItStore.getState().check("code").outcome).toBe("overfit");
    expect(useSortItStore.getState().failure?.name).toBe("Overfitting");
  });

  it("credits the code lane for the third star", () => {
    const api = createSortItApi();
    useSortItStore.getState().setLane("code");
    api.setBoundaryType("curve");
    api.autoFit();
    api.check();

    expect(
      useProgression.getState().games["sort-it-arcade"]?.codeLaneCleared,
    ).toBe(true);
    expect(useProgression.getState().games["sort-it-arcade"]?.stars).toBe(3);
  });

  it("credits the lane the clearing call came from, not the visible tab", () => {
    // The rail's "Check generalization" button stays on screen in the code
    // tab. Pressing it there is a visual-lane clear.
    useSortItStore.getState().setLane("code");
    useSortItStore.getState().setBoundaryType("curve");
    useSortItStore.getState().autoFit();
    expect(useSortItStore.getState().check().outcome).toBe("win");

    const progress = useProgression.getState().games["sort-it-arcade"];
    expect(progress?.completed).toBe(true);
    expect(progress?.codeLaneCleared).toBe(false);
    expect(progress?.stars).toBe(2);
  });

  it("never stores a NaN knot, whatever it is handed", () => {
    const before = [...useSortItStore.getState().boundary.params];
    const accuracyBefore = useSortItStore.getState().accuracy;
    const { setKnot, nudgeKnot } = useSortItStore.getState();

    setKnot(1, Number.NaN);
    setKnot(1, undefined as unknown as number);
    setKnot(0.5, 0.2);
    nudgeKnot(0, Number.NaN);

    const state = useSortItStore.getState();
    expect(state.boundary.params).toEqual(before);
    expect(state.boundary.params.every(Number.isFinite)).toBe(true);
    expect(state.accuracy).toBe(accuracyBefore);
  });

  it("ignores a capacity that isn't one, including prototype keys", () => {
    const before = useSortItStore.getState().boundary;
    useSortItStore
      .getState()
      .setBoundaryType("toString" as unknown as BoundaryType);
    expect(useSortItStore.getState().boundary).toBe(before);
  });

  it("stops vouching for a cleared round once the boundary moves", () => {
    useSortItStore.getState().setBoundaryType("curve");
    useSortItStore.getState().autoFit();
    useSortItStore.getState().check();
    expect(useSortItStore.getState().won).toBe(true);

    // Knock a handle to the floor: the old "good" verdict no longer applies.
    useSortItStore.getState().setKnot(2, 0);
    expect(useSortItStore.getState().won).toBe(false);
    expect(useSortItStore.getState().testAccuracy).toBeNull();
  });

  it("leaves the verdict alone when nothing actually moves", () => {
    useSortItStore.getState().setBoundaryType("curve");
    useSortItStore.getState().autoFit();
    useSortItStore.getState().check();
    const verdict = useSortItStore.getState().lastEvaluation;
    const height = useSortItStore.getState().boundary.params[1]!;

    // A press that didn't drag, and a re-fit of an already-fitted curve.
    useSortItStore.getState().setKnot(1, height);
    useSortItStore.getState().autoFit();

    const state = useSortItStore.getState();
    expect(state.won).toBe(true);
    expect(state.testAccuracy).not.toBeNull();
    expect(state.lastEvaluation).toBe(verdict);
    expect(state.whyCard?.title).toMatch(/already fitted/i);
  });

  it("attaches a why-card to every action", () => {
    const keys = new Set<string>();
    const record = () => {
      const card = useSortItStore.getState().whyCard;
      expect(card).not.toBeNull();
      keys.add(card!.key);
    };

    record();
    useSortItStore.getState().setKnot(0, 0.2);
    record();
    useSortItStore.getState().setBoundaryType("curve");
    record();
    useSortItStore.getState().autoFit();
    record();
    useSortItStore.getState().check();
    record();

    // Distinct keys mean the card visibly refreshes for each new action.
    expect(keys.size).toBe(5);
  });

  it("resets the boundary but keeps the round's points", () => {
    const pointsBefore = useSortItStore.getState().trainPoints;
    useSortItStore.getState().setBoundaryType("curve");
    useSortItStore.getState().autoFit();
    useSortItStore.getState().reset();

    const after = useSortItStore.getState();
    expect(after.boundary.params.every((height) => height === 0.5)).toBe(true);
    expect(after.boundary.type).toBe("curve");
    expect(after.trainPoints.map((p) => p.id)).toEqual(
      pointsBefore.map((p) => p.id),
    );
    expect(after.testAccuracy).toBeNull();
    expect(after.won).toBe(false);
  });

  it("advances to the next vetted seed on a new round", () => {
    useSortItStore.getState().newRound();
    const after = useSortItStore.getState();
    expect(after.round).toBe(2);
    expect(after.seed).toBe(ROUND_SEEDS[1]);
    expect(after.boundary.type).toBe("line");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Why-cards
// ═══════════════════════════════════════════════════════════════════════════

describe("why-cards", () => {
  const train = generateTrainPoints(ROUND_SEEDS[0]!);
  const test = generateTestPoints(ROUND_SEEDS[0]!);

  it("names overfitting and cites both accuracies", () => {
    const evaluation = evaluate(train, test, fittedFor("wiggle", train));
    const card = whyCardFor({ kind: "checked", evaluation });

    expect(card.title).toMatch(/overfit/i);
    expect(card.tone).toBe("bad");
    expect(card.body).toMatch(/\d+%.*\d+%/);
    expect(card.conceptHref).toBeTruthy();
  });

  it("names underfitting and cites the ceiling", () => {
    const evaluation = evaluate(train, test, fittedFor("line", train));
    const card = whyCardFor({ kind: "checked", evaluation });

    expect(card.title).toMatch(/underfit/i);
    expect(card.tone).toBe("bad");
    expect(card.body).toMatch(/ceiling/i);
  });

  it("celebrates a win with the generalisation gap", () => {
    const evaluation = evaluate(train, test, fittedFor("curve", train));
    const card = whyCardFor({ kind: "checked", evaluation });

    expect(card.tone).toBe("good");
    expect(card.title).toMatch(/cleared/i);
  });

  it("quotes every gap as the distance between the two figures it shows", () => {
    // Round 1's wiggle is 95.5% / 77.25%: shown as 96% and 77%, so the drop
    // must read 19 points, not the raw gap's rounded "18%".
    for (const seed of ROUND_SEEDS) {
      const roundTrain = generateTrainPoints(seed);
      const roundTest = generateTestPoints(seed);
      const overfit = whyCardFor({
        kind: "checked",
        evaluation: evaluate(roundTrain, roundTest, fittedFor("wiggle", roundTrain)),
      });
      const drop = /^(\d+)% on the points you fitted, (\d+)% on fresh ones — a drop of (\d+) points\./.exec(
        overfit.body,
      );
      expect(drop, overfit.body).not.toBeNull();
      expect(Number(drop![1]) - Number(drop![2])).toBe(Number(drop![3]));

      const win = whyCardFor({
        kind: "checked",
        evaluation: evaluate(roundTrain, roundTest, fittedFor("curve", roundTrain)),
      });
      const gap = /^(\d+)% on your points and (\d+)% on ones it never saw: (?:only a (\d+)-point difference|no difference at all)\./.exec(
        win.body,
      );
      expect(gap, win.body).not.toBeNull();
      expect(Math.abs(Number(gap![1]) - Number(gap![2]))).toBe(
        Number(gap![3] ?? 0),
      );
    }
  });

  it("never rounds a near-miss up to the bar it missed", () => {
    // Round 1's fitted curve with its first handle pulled to 0.06: 82% train
    // minus 0.024 is 0.796, which used to title itself "80% — needs 80%".
    const params = [...fittedFor("curve", train)];
    params[0] = 0.06;
    const evaluation = evaluate(train, test, params);
    expect(evaluation.outcome).toBe("near-miss");
    expect(evaluation.score).toBeCloseTo(0.796, 9);

    const card = whyCardFor({ kind: "checked", evaluation });
    expect(card.title).toBe("79% — needs 80%");
  });

  it("distinguishes a helpful move from a harmful one", () => {
    const better = whyCardFor({
      kind: "boundary-moved",
      accuracy: 0.8,
      accuracyDelta: 0.05,
      misclassified: 40,
      total: 200,
    });
    const worse = whyCardFor({
      kind: "boundary-moved",
      accuracy: 0.7,
      accuracyDelta: -0.05,
      misclassified: 60,
      total: 200,
    });

    expect(better.tone).toBe("good");
    expect(worse.tone).toBe("warn");
    expect(better.key).not.toBe(worse.key);
  });

  it("explains a no-op move rather than claiming progress", () => {
    const card = whyCardFor({
      kind: "boundary-moved",
      accuracy: 0.75,
      accuracyDelta: 0,
      misclassified: 50,
      total: 200,
    });
    expect(card.title).toMatch(/no change/i);
  });

  it("warns when capacity increases and reassures when it drops", () => {
    const up = whyCardFor({
      kind: "complexity-changed",
      from: "curve",
      to: "wiggle",
      params: KNOT_COUNTS.wiggle,
      penalty: penaltyFor(flat("wiggle")),
    });
    const down = whyCardFor({
      kind: "complexity-changed",
      from: "wiggle",
      to: "line",
      params: KNOT_COUNTS.line,
      penalty: 0,
    });

    expect(up.tone).toBe("warn");
    expect(down.tone).toBe("good");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Code lane: argument checks and the starter snippet
// ═══════════════════════════════════════════════════════════════════════════

describe("code lane api", () => {
  beforeEach(() => {
    useProgression.setState({
      ...EMPTY_PROGRESSION,
      hydrated: true,
      syncError: null,
      lastGain: null,
    });
    useProgression.getState().setAdapter(createMemoryAdapter());
    useSortItStore.getState().startRound(1);
  });

  /** Run a snippet the way the JavaScript lane does: `api` and `log` in scope. */
  const runSnippet = (code: string, api = createSortItApi()) => {
    const logs: string[] = [];
    const fn = new Function("api", "log", "checkBudget", code) as (
      api: unknown,
      log: (...args: unknown[]) => void,
      checkBudget: () => void,
    ) => unknown;
    fn(api, (...args) => logs.push(args.join(" ")), () => {});
    return logs;
  };

  it("names a missing height instead of storing NaN", () => {
    // `api.setKnot(1)` used to store NaN, and the next visual-lane render threw
    // on `yScale(NaN).toFixed` and took the whole route down.
    const api = createSortItApi();
    const before = [...useSortItStore.getState().boundary.params];

    expect(() =>
      (api.setKnot as (index: number) => void)(1),
    ).toThrow(TypeError);
    expect(() => api.setKnot(1, Number.NaN)).toThrow(/height must be a number/);
    expect(() => api.setKnot(1, 1.5)).toThrow(RangeError);
    expect(() => api.setKnot(7, 0.5)).toThrow(/index must be a whole number from 0 to 1/);
    expect(() => api.setKnot(0.5, 0.5)).toThrow(RangeError);

    expect(useSortItStore.getState().boundary.params).toEqual(before);
  });

  it("rejects prototype keys as boundary types", () => {
    const api = createSortItApi();
    expect(() =>
      api.setBoundaryType("toString" as unknown as BoundaryType),
    ).toThrow(/unknown boundary type "toString"/);
    expect(() =>
      api.setBoundaryType(3 as unknown as BoundaryType),
    ).toThrow(TypeError);
    expect(useSortItStore.getState().boundary.type).toBe("line");
  });

  it("still accepts every valid call", () => {
    const api = createSortItApi();
    api.setBoundaryType("curve");
    api.setKnot(4, 1);
    expect(api.knots()[4]).toBe(1);
    api.setKnot(0, 0);
    expect(api.knots()[0]).toBe(0);
  });

  it("credits the code lane when the starter snippet re-checks a visual clear", () => {
    // Clear round 1 from the rail, then run the snippet as-is. Its autoFit()
    // is a no-op on an already-fitted curve, so the board is unchanged — but
    // this is the first clear called from the code lane, and it earns ★3.
    useSortItStore.getState().setBoundaryType("curve");
    useSortItStore.getState().autoFit();
    useSortItStore.getState().check();
    expect(useProgression.getState().games["sort-it-arcade"]?.stars).toBe(2);

    runSnippet(STARTER_CODE);
    const progress = useProgression.getState().games["sort-it-arcade"];
    expect(progress?.codeLaneCleared).toBe(true);
    expect(progress?.stars).toBe(3);
    expect(progress?.playCount).toBe(2);

    // Once credited, neither lane re-counts the same boundary.
    runSnippet(STARTER_CODE);
    useSortItStore.getState().check();
    expect(useProgression.getState().games["sort-it-arcade"]?.playCount).toBe(2);
  });

  it("clears round 1 with the starter snippet, and a second run counts nothing twice", () => {
    const first = runSnippet(STARTER_CODE);
    expect(first.join("\n")).toMatch(/verdict win/);
    const progress = useProgression.getState().games["sort-it-arcade"];
    expect(progress?.codeLaneCleared).toBe(true);
    expect(progress?.playCount).toBe(1);
    const xp = useProgression.getState().xp;

    const second = runSnippet(STARTER_CODE);
    expect(second.join("\n")).toMatch(/verdict win/);
    expect(useSortItStore.getState().won).toBe(true);
    expect(useProgression.getState().games["sort-it-arcade"]?.playCount).toBe(1);
    expect(useProgression.getState().xp).toBe(xp);
  });

  it.each(ROUND_SEEDS.map((seed, index) => [index + 1, seed] as const))(
    "round %i (seed %i): the snippet's closing comment tells the truth",
    (round) => {
      useSortItStore.getState().startRound(round);
      runSnippet(STARTER_CODE);
      const curve = useSortItStore.getState().lastEvaluation!;
      expect(curve.outcome).toBe("win");

      // "Try it with 'wiggle'": run again on the fitted curve, as a player
      // editing one word would. The wiggle refines the curve it inherits.
      const wiggleCode = STARTER_CODE.replace(
        "api.setBoundaryType('curve');",
        "api.setBoundaryType('wiggle');",
      );
      expect(wiggleCode).not.toBe(STARTER_CODE);
      runSnippet(wiggleCode);
      const wiggle = useSortItStore.getState().lastEvaluation!;
      expect(wiggle.paramCount).toBe(KNOT_COUNTS.wiggle);
      expect(wiggle.trainAccuracy).toBeGreaterThan(curve.trainAccuracy);
      expect(wiggle.generalizationGap).toBeGreaterThan(curve.generalizationGap);
      expect(wiggle.outcome).not.toBe("win");
      // Too costly, never too simple.
      expect(wiggle.outcome).not.toBe("underfit");
      expect(wiggle.score).toBeLessThan(curve.score);
      // Named only when the gap backs it — a near-miss here is honest.
      expect(wiggle.outcome === "overfit").toBe(
        wiggle.generalizationGap >= OVERFIT_GAP,
      );

      // "Add api.reset() before api.autoFit()": a flat-start wiggle.
      const flatCode = wiggleCode.replace(
        "api.autoFit();",
        "api.reset();\napi.autoFit();",
      );
      expect(flatCode).not.toBe(wiggleCode);
      runSnippet(flatCode);
      expect(useSortItStore.getState().failure?.name).toBe("Overfitting");
    },
  );
});

// ═══════════════════════════════════════════════════════════════════════════
// Pointer geometry
// ═══════════════════════════════════════════════════════════════════════════

/**
 * The matrix a browser paints a `viewBox="0 0 100 100"` SVG with, for a box of
 * `width`×`height` CSS px at (left, top), under the default "xMidYMid meet":
 * uniform scale by the shorter side, centred along the longer one.
 */
function meetMatrix(left: number, top: number, width: number, height: number) {
  const scale = Math.min(width, height) / 100;
  const e = left + (width - 100 * scale) / 2;
  const f = top + (height - 100 * scale) / 2;
  const matrix: ScreenMatrix = {
    a: scale,
    b: 0,
    c: 0,
    d: scale,
    e,
    f,
    inverse: () => ({ a: 1 / scale, b: 0, c: 0, d: 1 / scale, e: -e / scale, f: -f / scale }),
  };
  return {
    matrix,
    /** Where viewBox point (x, y) is painted on screen. */
    toClient: (x: number, y: number) => ({
      clientX: e + x * scale,
      clientY: f + y * scale,
    }),
  };
}

describe("pointer geometry", () => {
  it("maps a pointer through the painted transform, letterboxing included", () => {
    // 1024×768: the field renders 622 wide and 824 tall, so the square content
    // sits in a band 101 px down. Mapping by the bounding box put a drag to
    // height 90 at height 80.
    const { matrix, toClient } = meetMatrix(40, 120, 622, 824);
    for (const [x, y] of [
      [6, 10],
      [50, 50],
      [94, 94],
    ] as const) {
      const { clientX, clientY } = toClient(x, y);
      const view = clientToUser(clientX, clientY, matrix);
      expect(view!.x).toBeCloseTo(x, 9);
      expect(view!.y).toBeCloseTo(y, 9);
    }
  });

  it("returns null without a usable transform", () => {
    expect(clientToUser(10, 10, null)).toBeNull();
    const degenerate: ScreenMatrix = {
      a: 0, b: 0, c: 0, d: 0, e: 0, f: 0,
      inverse: () => ({ a: Number.NaN, b: 0, c: 0, d: Number.NaN, e: 0, f: 0 }),
    };
    expect(clientToUser(10, 10, degenerate)).toBeNull();
  });

  it("gives every wiggle knot its own column, so a press can't hit a neighbour", () => {
    const count = KNOT_COUNTS.wiggle;
    for (let index = 0; index < count; index += 1) {
      expect(nearestKnotIndex(index / (count - 1), count)).toBe(index);
    }
    // Column edges sit halfway between knots.
    const half = 0.5 / (count - 1);
    expect(nearestKnotIndex(half * 0.98, count)).toBe(0);
    expect(nearestKnotIndex(half * 1.02, count)).toBe(1);
    expect(nearestKnotIndex(-3, count)).toBe(0);
    expect(nearestKnotIndex(7, count)).toBe(count - 1);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Visual lane: pointer and keyboard wiring
// ═══════════════════════════════════════════════════════════════════════════

describe("visual lane", () => {
  beforeEach(() => {
    useProgression.setState({
      ...EMPTY_PROGRESSION,
      hydrated: true,
      syncError: null,
      lastGain: null,
    });
    useProgression.getState().setAdapter(createMemoryAdapter());
    useSortItStore.getState().startRound(1);
  });

  /** Give jsdom the transform a real browser would have painted. */
  function mountField(box = { left: 40, top: 120, width: 622, height: 1309 }) {
    render(createElement(VisualLane));
    const svg = screen.getByRole("group", { name: "Classification field" });
    const { matrix, toClient } = meetMatrix(box.left, box.top, box.width, box.height);
    Object.assign(svg, {
      getScreenCTM: () => matrix,
      setPointerCapture: () => {},
      hasPointerCapture: () => false,
      releasePointerCapture: () => {},
    });
    // Field (0–1) → viewBox, matching the lane's d3 scales (PAD = 6).
    const toView = (x: number, y: number) => ({
      x: 6 + x * 88,
      y: 94 - y * 88,
    });
    const press = (type: "pointerDown" | "pointerMove" | "pointerUp", x: number, y: number) => {
      const v = toView(x, y);
      const { clientX, clientY } = toClient(v.x, v.y);
      act(() => {
        fireEvent[type](svg, { clientX, clientY, isPrimary: true, button: 0, pointerId: 1 });
      });
    };
    /** A finger at field (x, y), shifted `dy` CSS px down the screen. */
    const touch = (
      type: "pointerDown" | "pointerMove" | "pointerUp",
      x: number,
      y: number,
      dy = 0,
    ) => {
      const v = toView(x, y);
      const { clientX, clientY } = toClient(v.x, v.y);
      act(() => {
        fireEvent[type](svg, {
          clientX,
          clientY: clientY + dy,
          isPrimary: true,
          button: 0,
          pointerId: 7,
          pointerType: "touch",
        });
      });
    };
    return { svg, press, touch };
  }

  /** A 360×740 phone: the field renders about 294 px square. */
  const PHONE_FIELD = { left: 33, top: 260, width: 294, height: 294 };

  it("lets a thumb swipe across the field without editing a cleared fit", () => {
    // The field is most of a phone's width, so a scroll that starts on it is
    // the ordinary case. Moving a handle on touch-down turned every such
    // swipe into an edit that took a cleared round's Next button away.
    act(() => {
      useSortItStore.getState().setBoundaryType("curve");
      useSortItStore.getState().autoFit();
      useSortItStore.getState().check();
    });
    expect(useSortItStore.getState().won).toBe(true);
    const before = [...useSortItStore.getState().boundary.params];

    const { touch } = mountField(PHONE_FIELD);
    // Knot 2 (x = 0.25) sits well above y = 0.1, so this lands off-handle.
    expect(Math.abs(before[1]! - 0.1) * 88 * 2.94).toBeGreaterThan(22);
    touch("pointerDown", 0.3, 0.1);
    for (let step = 1; step <= 6; step += 1) {
      touch("pointerMove", 0.3, 0.1, -20 * step);
    }
    touch("pointerUp", 0.3, 0.1, -120);

    const state = useSortItStore.getState();
    expect(state.boundary.params).toEqual(before);
    expect(state.won).toBe(true);
    expect(state.testAccuracy).not.toBeNull();
  });

  it("still moves the nearest handle on a touch tap, wobble and all", () => {
    const { touch } = mountField(PHONE_FIELD);
    touch("pointerDown", 0.2, 0.15);
    // Fingers are never perfectly still; a few pixels is still a tap.
    touch("pointerMove", 0.2, 0.15, 4);
    expect(useSortItStore.getState().boundary.params[0]).toBe(0.5);
    touch("pointerUp", 0.2, 0.15, 4);

    const params = useSortItStore.getState().boundary.params;
    expect(params[0]).toBeCloseTo(0.15, 6);
    expect(params[1]).toBe(0.5);
    expect(document.activeElement).toBe(
      screen.getByRole("slider", { name: "Boundary handle 1 of 2" }),
    );
  });

  it("still drags a handle a finger lands on", () => {
    const { touch } = mountField(PHONE_FIELD);
    touch("pointerDown", 1, 0.5);
    touch("pointerMove", 1, 0.9);
    touch("pointerUp", 1, 0.9);
    expect(useSortItStore.getState().boundary.params[1]).toBeCloseTo(0.9, 6);
  });

  it("drags the handle under the pointer to the height under the pointer", () => {
    const { press } = mountField();
    // Handle 2 (x = 1) starts at 0.5; drag it to 0.9 on a letterboxed field.
    press("pointerDown", 1, 0.5);
    press("pointerMove", 1, 0.9);
    press("pointerUp", 1, 0.9);
    const params = useSortItStore.getState().boundary.params;
    expect(params[1]).toBeCloseTo(0.9, 6);
    expect(params[0]).toBe(0.5);
  });

  it("moves handle i, not i + 1, when 25 handles share the width", () => {
    act(() => useSortItStore.getState().setBoundaryType("wiggle"));
    const { press } = mountField();
    const count = KNOT_COUNTS.wiggle;
    for (const index of [5, 12, 23]) {
      const x = index / (count - 1);
      press("pointerDown", x, 0.5);
      press("pointerMove", x, 0.2);
      press("pointerUp", x, 0.2);
      const params = useSortItStore.getState().boundary.params;
      expect(params[index], `handle ${index}`).toBeCloseTo(0.2, 6);
      expect(params[index + 1], `handle ${index + 1}`).toBe(0.5);
    }
  });

  it("moves the nearest handle with a single tap — no drag needed", () => {
    const { press } = mountField();
    press("pointerDown", 0.2, 0.15);
    press("pointerUp", 0.2, 0.15);
    const params = useSortItStore.getState().boundary.params;
    expect(params[0]).toBeCloseTo(0.15, 6);
    expect(params[1]).toBe(0.5);
    // And focus follows, so the arrow keys continue on the handle that moved.
    expect(document.activeElement).toBe(
      screen.getByRole("slider", { name: "Boundary handle 1 of 2" }),
    );
  });

  it("grabs a handle without jumping it when pressed on the handle itself", () => {
    const { press } = mountField();
    const before = useSortItStore.getState().whyCard;
    press("pointerDown", 0, 0.51);
    press("pointerUp", 0, 0.51);
    expect(useSortItStore.getState().boundary.params[0]).toBe(0.5);
    expect(useSortItStore.getState().whyCard).toBe(before);
  });

  it("follows the ARIA slider pattern: Home is the bottom, End the top", () => {
    render(createElement(VisualLane));
    const handle = screen.getByRole("slider", { name: "Boundary handle 1 of 2" });
    act(() => {
      fireEvent.keyDown(handle, { key: "End" });
    });
    expect(useSortItStore.getState().boundary.params[0]).toBe(1);
    expect(handle.getAttribute("aria-valuenow")).toBe("100");
    act(() => {
      fireEvent.keyDown(handle, { key: "Home" });
    });
    expect(useSortItStore.getState().boundary.params[0]).toBe(0);
    expect(handle.getAttribute("aria-valuenow")).toBe("0");
  });
});

