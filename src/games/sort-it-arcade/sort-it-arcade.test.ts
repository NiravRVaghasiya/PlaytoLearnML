import { beforeEach, describe, expect, it } from "vitest";
import {
  EMPTY_PROGRESSION,
  createMemoryAdapter,
  useProgression,
} from "@/engine/progression";
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
  UNDERFIT_ACCURACY,
  WIN_SCORE,
  accuracyOf,
  boundaryAt,
  capacityOf,
  classifyPoint,
  complexityCostOf,
  evaluate,
  fitKnots,
  generateTestPoints,
  generateTrainPoints,
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
      capacityOf(train, flat("wiggle")) - penaltyFor(flat("wiggle"));
    expect(wiggleCeiling, `wiggle ceiling on seed ${seed}`).toBeLessThan(
      WIN_SCORE,
    );
  });

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
    expect(fitted.trainAccuracy).toBeLessThan(UNDERFIT_ACCURACY);

    // An unfitted flat line scores just as badly but is NOT underfitting — it
    // simply hasn't been fitted. Calling that underfitting would teach the
    // wrong word.
    const unfitted = evaluate(train, test, [0.5, 0.5]);
    expect(unfitted.trainAccuracy).toBeLessThan(UNDERFIT_ACCURACY);
    expect(unfitted.outcome).not.toBe("underfit");
    expect(unfitted.failure).toBeNull();
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

  it("reports bestAtThisComplexity as the fitted ceiling", () => {
    const params = flat("curve");
    const result = evaluate(train, test, params);
    expect(result.bestAtThisComplexity).toBeCloseTo(
      capacityOf(train, params),
      10,
    );
    expect(result.bestAtThisComplexity).toBeGreaterThanOrEqual(
      result.trainAccuracy - CAPACITY_TOLERANCE,
    );
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

    // Checking the same boundary again must not pay twice.
    useSortItStore.getState().check();
    expect(useProgression.getState().xp).toBe(xpAfterFirst);
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

  it("credits the code lane for the third star", () => {
    useSortItStore.getState().setLane("code");
    useSortItStore.getState().setBoundaryType("curve");
    useSortItStore.getState().autoFit();
    useSortItStore.getState().check();

    expect(
      useProgression.getState().games["sort-it-arcade"]?.codeLaneCleared,
    ).toBe(true);
    expect(useProgression.getState().games["sort-it-arcade"]?.stars).toBe(3);
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
