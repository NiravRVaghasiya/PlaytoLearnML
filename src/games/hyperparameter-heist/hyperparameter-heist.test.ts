import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HIGH_SCORE_THRESHOLD, useProgression } from "@/engine/progression";
import { createJsExecutor } from "@/engine/useCodeLane";
import {
  BASE_SCORE,
  BATCH_SIZE,
  BAYESIAN_SEED_TRIALS,
  BUDGET,
  CRACK_RATE_RUNS,
  CRACK_THRESHOLD,
  DIALS,
  DIAL_COUNT,
  GRID_TRAP_DISTINCT,
  LEARNING_RATE,
  MOMENTUM,
  WEIGHT_DECAY,
  bestTrial,
  crackScore,
  describePoint,
  dialInfluence,
  distinctValuesTried,
  dominantDial,
  evaluateRun,
  expectedImprovement,
  fitSurrogate,
  gridBest,
  gridLevels,
  gridPoints,
  haltonPoints,
  latestCrackScoring,
  objectiveAt,
  randomCrackRate,
  randomPoints,
  runStrategy,
  standardNormalCdf,
  suggestNext,
  temperatureOf,
  toRealValue,
  toUnitValue,
  type Strategy,
  type Trial,
} from "./ml";
import {
  RANDOM_SEED,
  bestObjective,
  crackedOn,
  lastTemperature,
  randomSeedFor,
  triesLeft,
  triesUsed,
  useHeistStore,
} from "./store";
import { whyCardFor } from "./why-cards";
import { STARTER_CODE, createCodeApi } from "./code-api";

/** How many runs to average a stochastic strategy over. */
const RUNS = 120;

function bestObjectives(strategy: Strategy, runs = RUNS): number[] {
  return Array.from({ length: runs }, (_, index) => {
    const trials = runStrategy(strategy, BUDGET, (index + 1) * 7919);
    return bestTrial(trials)!.objectiveValue;
  });
}

function crackRate(strategy: Strategy, runs = RUNS): number {
  let cracked = 0;
  for (let index = 0; index < runs; index += 1) {
    const trials = runStrategy(strategy, BUDGET, (index + 1) * 7919);
    if (bestTrial(trials)!.objectiveValue >= CRACK_THRESHOLD) cracked += 1;
  }
  return cracked / runs;
}

const mean = (values: number[]) =>
  values.reduce((total, value) => total + value, 0) / values.length;

const asTrials = (points: number[][], source: Strategy | "manual" = "manual"): Trial[] =>
  points.map((params, index) => ({
    params,
    objectiveValue: objectiveAt(params),
    index: index + 1,
    source,
  }));

/**
 * A point that opens the safe, and one that nearly does.
 *
 * Found by sweep rather than written down. Hardcoded coordinates would silently
 * stop being winning points the moment the surface is retuned, and the test would
 * then be asserting something about the wrong place.
 */
const CRACKING_POINT = (() => {
  for (const point of haltonPoints(6000)) {
    if (objectiveAt(point) >= CRACK_THRESHOLD) return point;
  }
  throw new Error("the surface has no point above the crack threshold");
})();

const NEAR_MISS_POINT = (() => {
  let best: number[] | null = null;
  for (const point of haltonPoints(6000)) {
    const value = objectiveAt(point);
    if (value < CRACK_THRESHOLD && value > 0.75) {
      if (best === null || value > objectiveAt(best)) best = point;
    }
  }
  if (best === null) throw new Error("the surface has no near miss");
  return best;
})();

/** Cold, and reliably so: wrong rate, no momentum, heavy decay. */
const HOPELESS_POINT = [0.99, 0.02, 1, 1];

describe("the dials", () => {
  it("round-trips between unit and real values", () => {
    for (const dial of DIALS) {
      for (const unit of [0, 0.13, 0.5, 0.87, 1]) {
        expect(toUnitValue(dial, toRealValue(dial, unit)), dial.name).toBeCloseTo(
          unit,
          8,
        );
      }
    }
  });

  it("spans the stated range at the ends", () => {
    for (const dial of DIALS) {
      expect(toRealValue(dial, 0)).toBeCloseTo(dial.min, 8);
      expect(toRealValue(dial, 1)).toBeCloseTo(dial.max, 8);
    }
  });

  it("puts the learning rate on a log scale, so a grid is even in decades", () => {
    const lr = DIALS[LEARNING_RATE]!;
    // The midpoint of four decades is two decades up, not the arithmetic middle.
    expect(toRealValue(lr, 0.5)).toBeCloseTo(0.01, 6);
    expect(toRealValue(lr, 0.5)).not.toBeCloseTo((lr.min + lr.max) / 2, 3);
  });

  it("clamps out-of-range unit coordinates rather than extrapolating", () => {
    expect(toRealValue(DIALS[MOMENTUM]!, 5)).toBeCloseTo(0.99, 8);
    expect(toRealValue(DIALS[MOMENTUM]!, -2)).toBeCloseTo(0, 8);
  });
});

describe("the objective surface", () => {
  it("is deterministic — retrying a combination cannot pay off", () => {
    // If re-evaluating gave a different number, the game would teach that search
    // is about luck rather than about coverage.
    const point = [0.4, 0.8, 0.2, 0.1];
    expect(objectiveAt(point)).toBe(objectiveAt(point));
  });

  it("stays inside the unit interval everywhere", () => {
    for (const point of haltonPoints(400)) {
      const value = objectiveAt(point);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
  });

  it("has low effective dimensionality — the premise of the whole game", () => {
    // Bergstra & Bengio's argument only applies when some dials barely matter. If
    // this ever stops being true, random search stops beating grid search and the
    // game is teaching something false.
    const influence = dialInfluence();
    const carrying = influence.filter((value) => value > 0.15).length;
    const flat = influence.filter((value) => value < 0.06).length;

    expect(carrying, "no dial dominates").toBeGreaterThanOrEqual(1);
    expect(flat, "no dial is near-flat").toBeGreaterThanOrEqual(1);
    expect(influence[LEARNING_RATE]!).toBeGreaterThan(influence[BATCH_SIZE]!);
    expect(influence[LEARNING_RATE]!).toBeGreaterThan(influence[WEIGHT_DECAY]!);
  });

  it("names the learning rate as the dominant dial, by measurement", () => {
    // dominantDial is measured, not hardcoded, so the Grid trap diagnosis follows
    // the surface if it is ever retuned.
    expect(dominantDial()).toBe(LEARNING_RATE);
  });

  it("moves the learning rate's optimum as momentum rises", () => {
    // The interaction that makes the surface non-separable — and makes
    // one-dial-at-a-time reasoning wrong.
    const bestLrAt = (momentumUnit: number) => {
      let best = { unit: 0, value: -1 };
      for (let step = 0; step <= 200; step += 1) {
        const unit = step / 200;
        const value = objectiveAt([unit, momentumUnit, 0, 0]);
        if (value > best.value) best = { unit, value };
      }
      return best.unit;
    };

    const slowLr = bestLrAt(0);
    const fastLr = bestLrAt(1);
    expect(fastLr, "more momentum should want a smaller rate").toBeLessThan(slowLr);
  });

  it("puts its optimum above the crack threshold, but not trivially so", () => {
    let best = -1;
    for (const point of haltonPoints(4000)) {
      best = Math.max(best, objectiveAt(point));
    }
    expect(best).toBeGreaterThan(CRACK_THRESHOLD);
    // A coin-flip baseline must be well below the bar, or the safe opens itself.
    expect(BASE_SCORE).toBeLessThan(CRACK_THRESHOLD - 0.3);
  });

  it("maps objective values onto safecracking temperatures in order", () => {
    expect(temperatureOf(0.5)).toBe("freezing");
    expect(temperatureOf(0.65)).toBe("cold");
    expect(temperatureOf(0.78)).toBe("warm");
    expect(temperatureOf(0.86)).toBe("hot");
    expect(temperatureOf(CRACK_THRESHOLD)).toBe("cracked");
  });
});

describe("grid search", () => {
  it("takes the finest full-factorial grid the budget allows", () => {
    expect(gridLevels(16, 4)).toBe(2);
    expect(gridLevels(81, 4)).toBe(3);
    expect(gridLevels(15, 4)).toBe(1);
    expect(gridPoints(BUDGET)).toHaveLength(BUDGET);
  });

  it("never spends a try on a duplicate point", () => {
    const seen = new Set(gridPoints(BUDGET).map((point) => point.join(",")));
    expect(seen.size).toBe(BUDGET);
  });

  it("avoids the corners of the search space", () => {
    for (const point of gridPoints(BUDGET)) {
      for (const coordinate of point) {
        expect(coordinate).toBeGreaterThan(0);
        expect(coordinate).toBeLessThan(1);
      }
    }
  });

  it("sees only two learning rates despite spending the entire budget", () => {
    // The heart of the lesson, as a number.
    const grid = runStrategy("grid", BUDGET, 0);
    expect(grid).toHaveLength(BUDGET);
    expect(distinctValuesTried(grid, LEARNING_RATE)).toBe(gridLevels(BUDGET));
    expect(distinctValuesTried(grid, LEARNING_RATE)).toBe(2);
  });

  it("cannot crack the safe at all", () => {
    const grid = runStrategy("grid", BUDGET, 0);
    expect(bestTrial(grid)!.objectiveValue).toBeLessThan(CRACK_THRESHOLD);
  });
});

describe("random search", () => {
  it("explores far more values of the dial that matters", () => {
    const random = runStrategy("random", BUDGET, 12345);
    const grid = runStrategy("grid", BUDGET, 0);
    expect(distinctValuesTried(random, LEARNING_RATE)).toBeGreaterThan(
      distinctValuesTried(grid, LEARNING_RATE) * 3,
    );
  });

  it("is reproducible for a seed and different across seeds", () => {
    expect(randomPoints(8, 42)).toEqual(randomPoints(8, 42));
    expect(randomPoints(8, 42)).not.toEqual(randomPoints(8, 43));
  });

  it("beats grid search on the same budget, on average", () => {
    // The spec's claim, measured over many runs rather than asserted. Any single
    // run is luck; the average is the argument.
    const gridBest = bestTrial(runStrategy("grid", BUDGET, 0))!.objectiveValue;
    const randomMean = mean(bestObjectives("random"));
    expect(randomMean).toBeGreaterThan(gridBest + 0.05);
  });

  it("crosses the line grid cannot, which is the part that matters", () => {
    // A margin on the mean understates it. The grid's best point sits in the warm
    // band and stays there; random gets past the threshold most runs. Being close
    // is not the same as opening the safe.
    const gridBest = bestTrial(runStrategy("grid", BUDGET, 0))!.objectiveValue;
    expect(gridBest).toBeLessThan(CRACK_THRESHOLD);
    expect(gridBest).toBeGreaterThan(0.75);
    expect(crackRate("random")).toBeGreaterThan(0.6);
  });

  it("cracks the safe most of the time, where grid never does", () => {
    const rate = crackRate("random");
    expect(rate).toBeGreaterThan(0.6);
    expect(rate).toBeLessThan(1);
  });
});

describe("the Gaussian process surrogate", () => {
  const trials = asTrials(haltonPoints(10, 1));

  it("interpolates the points it was given", () => {
    const surrogate = fitSurrogate(trials)!;
    for (const trial of trials) {
      const { mean: predicted } = surrogate.predict(trial.params);
      expect(predicted).toBeCloseTo(trial.objectiveValue, 2);
    }
  });

  it("is nearly certain at observed points and unsure far away", () => {
    const surrogate = fitSurrogate(trials)!;
    const atObserved = surrogate.predict(trials[0]!.params).sd;
    // A corner far from any Halton point in the interior.
    const farAway = surrogate.predict([0.99, 0.01, 0.99, 0.01]).sd;
    expect(atObserved).toBeLessThan(farAway);
  });

  it("reports the best value it has seen", () => {
    const surrogate = fitSurrogate(trials)!;
    expect(surrogate.best).toBeCloseTo(
      Math.max(...trials.map((trial) => trial.objectiveValue)),
      10,
    );
  });

  it("survives duplicate points instead of dividing by zero", () => {
    // A degenerate Gram matrix has no Cholesky factor. The fallback must not throw.
    const duplicated = asTrials([
      [0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5],
    ]);
    const surrogate = fitSurrogate(duplicated);
    expect(surrogate).not.toBeNull();
    const prediction = surrogate!.predict([0.4, 0.4, 0.4, 0.4]);
    expect(Number.isFinite(prediction.mean)).toBe(true);
    expect(Number.isFinite(prediction.sd)).toBe(true);
  });

  it("returns nothing for no observations", () => {
    expect(fitSurrogate([])).toBeNull();
  });

  it("computes the normal CDF closely enough for an acquisition function", () => {
    expect(standardNormalCdf(0)).toBeCloseTo(0.5, 4);
    expect(standardNormalCdf(1.96)).toBeCloseTo(0.975, 3);
    expect(standardNormalCdf(-1.96)).toBeCloseTo(0.025, 3);
  });
});

describe("expected improvement", () => {
  const trials = asTrials(haltonPoints(8, 1));

  it("is never negative", () => {
    const surrogate = fitSurrogate(trials)!;
    for (const candidate of haltonPoints(200, 40)) {
      expect(expectedImprovement(surrogate, candidate)).toBeGreaterThanOrEqual(0);
    }
  });

  it("is near zero where the surrogate is confident and mediocre", () => {
    // An observed point that scored badly: nothing to gain and nothing to learn.
    const poor = asTrials([
      [0.05, 0.5, 0.5, 0.5],
      [0.95, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5],
      [0.3, 0.5, 0.5, 0.5],
    ]);
    const surrogate = fitSurrogate(poor)!;
    const atObserved = expectedImprovement(surrogate, poor[0]!.params);
    const unexplored = expectedImprovement(surrogate, [0.42, 0.9, 0.1, 0.1]);
    expect(atObserved).toBeLessThan(unexplored);
  });

  it("rewards uncertainty, not just a high predicted mean", () => {
    // Two candidates with similar predicted means: the less-explored one should
    // score higher. This is the exploration term doing its job.
    const surrogate = fitSurrogate(trials)!;
    let bothTerms = 0;
    for (const candidate of haltonPoints(300, 90)) {
      const { mean: predicted, sd } = surrogate.predict(candidate);
      const ei = expectedImprovement(surrogate, candidate);
      if (predicted < surrogate.best && ei > 0 && sd > 0) bothTerms += 1;
    }
    // Points predicted WORSE than the incumbent still get positive EI, which can
    // only come from the uncertainty term.
    expect(bothTerms).toBeGreaterThan(0);
  });
});

describe("Bayesian search", () => {
  it("spreads its first tries out instead of modelling one data point", () => {
    const first = suggestNext([]);
    expect(Number.isNaN(first.expectedImprovement)).toBe(true);

    const seeded: Trial[] = [];
    for (let index = 0; index < BAYESIAN_SEED_TRIALS; index += 1) {
      const point = suggestNext(seeded).point;
      seeded.push({
        params: point,
        objectiveValue: objectiveAt(point),
        index: index + 1,
        source: "bayesian",
      });
    }
    const distinct = new Set(seeded.map((trial) => trial.params.join(","))).size;
    expect(distinct).toBe(BAYESIAN_SEED_TRIALS);
  });

  it("starts modelling once it has enough observations", () => {
    const seeded = asTrials(haltonPoints(BAYESIAN_SEED_TRIALS, 1), "bayesian");
    const suggestion = suggestNext(seeded);
    expect(Number.isFinite(suggestion.expectedImprovement)).toBe(true);
    expect(Number.isFinite(suggestion.predictedMean)).toBe(true);
    expect(suggestion.expectedImprovement).toBeGreaterThanOrEqual(0);
  });

  it("never suggests re-evaluating a point it already tried", () => {
    // The objective is deterministic, so a repeat try is a wasted try.
    const trials = asTrials(haltonPoints(10, 1), "bayesian");
    const suggestion = suggestNext(trials);
    for (const trial of trials) {
      const identical = trial.params.every(
        (value, index) => Math.abs(value - (suggestion.point[index] ?? 0)) < 1e-6,
      );
      expect(identical).toBe(false);
    }
  });

  it("beats random search on the same budget, on average", () => {
    const randomMean = mean(bestObjectives("random"));
    const bayesianMean = mean(bestObjectives("bayesian"));
    expect(bayesianMean).toBeGreaterThan(randomMean);
  });

  it("cracks the safe more reliably than random", () => {
    expect(crackRate("bayesian")).toBeGreaterThan(crackRate("random"));
  });

  it("wastes fewer tries getting there", () => {
    const firstCrack = (strategy: Strategy) => {
      const totals: number[] = [];
      for (let index = 0; index < RUNS; index += 1) {
        const trials = runStrategy(strategy, BUDGET, (index + 1) * 7919);
        const first = trials.find(
          (trial) => trial.objectiveValue >= CRACK_THRESHOLD,
        );
        if (first) totals.push(first.index);
      }
      return mean(totals);
    };
    expect(firstCrack("bayesian")).toBeLessThanOrEqual(firstCrack("random"));
  });

  it("gets more resolution on the dominant dial than grid does", () => {
    const bayesian = runStrategy("bayesian", BUDGET, 12345);
    const grid = runStrategy("grid", BUDGET, 0);
    expect(distinctValuesTried(bayesian, LEARNING_RATE)).toBeGreaterThan(
      distinctValuesTried(grid, LEARNING_RATE) * 3,
    );
  });
});

describe("the strategies in order", () => {
  it("ranks grid worst, then random, then Bayesian", () => {
    // The single claim the game exists to make.
    const gridBest = bestTrial(runStrategy("grid", BUDGET, 0))!.objectiveValue;
    const randomMean = mean(bestObjectives("random"));
    const bayesianMean = mean(bestObjectives("bayesian"));

    expect(gridBest).toBeLessThan(randomMean);
    expect(randomMean).toBeLessThan(bayesianMean);
  });

  it("spends exactly the budget, whichever strategy is used", () => {
    for (const strategy of ["grid", "random", "bayesian"] as Strategy[]) {
      expect(runStrategy(strategy, BUDGET, 7).length, strategy).toBe(BUDGET);
    }
  });

  it("numbers its trials in order from one", () => {
    const trials = runStrategy("bayesian", BUDGET, 3);
    expect(trials.map((trial) => trial.index)).toEqual(
      Array.from({ length: BUDGET }, (_, index) => index + 1),
    );
  });
});

describe("naming the failure", () => {
  it("names Grid trap when the dominant dial was barely sampled", () => {
    const grid = runStrategy("grid", BUDGET, 0);
    const evaluation = evaluateRun({ trials: grid, budget: BUDGET, finished: true });

    expect(evaluation.outcome).toBe("grid-trap");
    expect(evaluation.failure?.name).toBe("Grid trap");
    // It must name the dial and both numbers that make the argument.
    expect(evaluation.failure?.detail).toMatch(/learning rate/);
    expect(evaluation.failure?.detail).toMatch(/2 distinct/);
    expect(evaluation.failure?.detail).toMatch(/at random/);
  });

  it("names Budget exhausted when coverage was fine but the safe stayed shut", () => {
    // Well-spread tries that all sit in a mediocre band: not a coverage failure.
    const spread = asTrials(
      Array.from({ length: BUDGET }, (_, index) => [
        0.02 + (index / BUDGET) * 0.28,
        0.05,
        0.5,
        0.95,
      ]),
    );
    const evaluation = evaluateRun({
      trials: spread,
      budget: BUDGET,
      finished: true,
    });

    expect(distinctValuesTried(spread, LEARNING_RATE)).toBeGreaterThan(
      GRID_TRAP_DISTINCT,
    );
    expect(evaluation.outcome).toBe("budget-exhausted");
    expect(evaluation.failure?.name).toBe("Budget exhausted");
    expect(evaluation.failure?.detail).toMatch(/not a coverage failure/i);
  });

  it("reports no failure while tries remain", () => {
    const partial = asTrials(haltonPoints(4, 1));
    const evaluation = evaluateRun({
      trials: partial,
      budget: BUDGET,
      finished: false,
    });
    expect(evaluation.outcome).toBe("searching");
    expect(evaluation.failure).toBeNull();
  });

  it("cracks as soon as a try clears the threshold, budget spent or not", () => {
    const lucky = asTrials([CRACKING_POINT]);
    expect(lucky[0]!.objectiveValue).toBeGreaterThanOrEqual(CRACK_THRESHOLD);
    const evaluation = evaluateRun({
      trials: lucky,
      budget: BUDGET,
      finished: false,
    });
    expect(evaluation.outcome).toBe("cracked");
    expect(evaluation.failure).toBeNull();
  });

  it("scores an early crack above a late one", () => {
    const early = evaluateRun({
      trials: asTrials([CRACKING_POINT, HOPELESS_POINT, HOPELESS_POINT]),
      budget: BUDGET,
      finished: true,
    });
    const late = evaluateRun({
      trials: asTrials([
        ...Array.from({ length: 12 }, () => HOPELESS_POINT),
        CRACKING_POINT,
      ]),
      budget: BUDGET,
      finished: true,
    });

    expect(early.outcome).toBe("cracked");
    expect(late.outcome).toBe("cracked");
    expect(early.score).toBeGreaterThan(late.score);
  });

  it("gives a near miss more credit than a hopeless run", () => {
    const near = evaluateRun({
      trials: asTrials([NEAR_MISS_POINT]),
      budget: BUDGET,
      finished: true,
    });
    const hopeless = evaluateRun({
      trials: asTrials([HOPELESS_POINT]),
      budget: BUDGET,
      finished: true,
    });
    expect(near.score).toBeGreaterThan(hopeless.score);
  });

  it("keeps every score inside the unit interval", () => {
    for (const strategy of ["grid", "random", "bayesian"] as Strategy[]) {
      for (let seed = 1; seed <= 20; seed += 1) {
        const evaluation = evaluateRun({
          trials: runStrategy(strategy, BUDGET, seed * 101),
          budget: BUDGET,
          finished: true,
        });
        expect(evaluation.score).toBeGreaterThanOrEqual(0);
        expect(evaluation.score).toBeLessThanOrEqual(1);
      }
    }
  });

  it("handles an empty run without inventing a best", () => {
    const evaluation = evaluateRun({ trials: [], budget: BUDGET, finished: false });
    expect(evaluation.best).toBeNull();
    expect(evaluation.score).toBe(0);
  });
});

describe("describing a point", () => {
  it("reads out every dial with its own formatting", () => {
    const text = describePoint([0.5, 0.5, 0.5, 0.5]);
    for (const dial of DIALS) {
      expect(text).toContain(dial.short);
    }
    expect(DIAL_COUNT).toBe(4);
  });
});

describe("the store's heist flow", () => {
  const store = useHeistStore;

  beforeEach(() => {
    store.getState().reset();
  });

  it("starts with a full budget, centred dials and nothing known", () => {
    expect(store.getState().trials).toEqual([]);
    expect(store.getState().budget).toBe(BUDGET);
    expect(store.getState().dials).toEqual(new Array(DIAL_COUNT).fill(0.5));
    expect(store.getState().phase).toBe("cracking");
    expect(store.getState().revealed, "the surface must start hidden").toBe(false);
    expect(Number.isNaN(bestObjective(store.getState()))).toBe(true);
  });

  it("spends exactly one try per attempt", () => {
    store.getState().tryCurrent();
    expect(triesUsed(store.getState())).toBe(1);
    expect(triesLeft(store.getState())).toBe(BUDGET - 1);
  });

  it("records the objective at the dials it was given", () => {
    store.getState().setDials([0.3, 0.7, 0.2, 0.1]);
    const trial = store.getState().tryCurrent()!;
    expect(trial.params).toEqual([0.3, 0.7, 0.2, 0.1]);
    expect(trial.objectiveValue).toBeCloseTo(objectiveAt([0.3, 0.7, 0.2, 0.1]), 12);
  });

  it("clamps dials into range rather than extrapolating", () => {
    store.getState().setDial(0, 5);
    expect(store.getState().dials[0]).toBe(1);
    store.getState().setDials([-1, 2, 0.5, 0.5]);
    expect(store.getState().dials[0]).toBe(0);
    expect(store.getState().dials[1]).toBe(1);
  });

  it("keeps the surface hidden until the run is over", () => {
    // Revealing it early would hand over the answer; this is the load-bearing
    // information asymmetry of the whole game.
    store.getState().setDials([0.9, 0.1, 0.5, 0.5]);
    store.getState().tryCurrent();
    expect(store.getState().revealed).toBe(false);

    store.getState().runToBudget("grid");
    expect(triesLeft(store.getState())).toBe(0);
    expect(store.getState().revealed).toBe(true);
  });

  it("reveals the surface the moment the safe opens", () => {
    store.getState().setDials(CRACKING_POINT);
    store.getState().tryCurrent();
    expect(store.getState().phase).toBe("cracked");
    expect(store.getState().revealed).toBe(true);
    expect(store.getState().failure).toBeNull();
  });

  it("refuses further tries once the run has ended", () => {
    store.getState().setDials(CRACKING_POINT);
    store.getState().tryCurrent();
    const spent = triesUsed(store.getState());
    expect(store.getState().tryCurrent()).toBeNull();
    expect(triesUsed(store.getState())).toBe(spent);
  });

  it("never spends more than the budget", () => {
    for (let attempt = 0; attempt < BUDGET * 2; attempt += 1) {
      store.getState().setDials([Math.random(), 0.05, 1, 1]);
      store.getState().tryCurrent();
    }
    expect(triesUsed(store.getState())).toBeLessThanOrEqual(BUDGET);
  });

  it("lays a strategy out over the whole budget, not the remainder", () => {
    // Re-planning a grid for the tries that are left would change its resolution
    // mid-run and make any comparison meaningless.
    store.getState().setDials([0.9, 0.1, 0.5, 0.5]);
    store.getState().tryCurrent();
    store.getState().runToBudget("grid");

    const gridTail = gridPoints(BUDGET).slice(1);
    const recorded = store.getState().trials.slice(1).map((trial) => trial.params);
    expect(recorded).toEqual(gridTail);
  });

  it("busts the run when the grid spends everything without cracking", () => {
    store.getState().runToBudget("grid");
    expect(store.getState().phase).toBe("busted");
    expect(store.getState().failure?.name).toBe("Grid trap");
    expect(store.getState().revealed).toBe(true);
  });

  it("cracks the safe with Bayesian search inside the budget", () => {
    store.getState().setStrategy("bayesian");
    store.getState().runToBudget("bayesian");
    expect(store.getState().phase).toBe("cracked");
    expect(bestObjective(store.getState())).toBeGreaterThanOrEqual(
      CRACK_THRESHOLD,
    );
  });

  it("moves the dials to the suggestion when following the hint", () => {
    store.getState().setStrategy("bayesian");
    const suggested = suggestNext(store.getState().trials).point;
    const trial = store.getState().trySuggestion()!;
    expect(trial.params).toEqual(suggested);
    expect(store.getState().dials).toEqual(suggested);
  });

  it("reports the temperature of the last try, not of the dials", () => {
    // Reading the dials would be a free objective evaluation.
    expect(lastTemperature(store.getState())).toBeNull();
    store.getState().setDials(CRACKING_POINT);
    store.getState().tryCurrent();
    expect(lastTemperature(store.getState())).toBe("cracked");
  });

  it("scores an early crack higher than a late one, through the store", () => {
    store.getState().setDials(CRACKING_POINT);
    store.getState().tryCurrent();
    const early = store.getState().evaluation!.score;

    store.getState().reset();
    for (let attempt = 0; attempt < 10; attempt += 1) {
      store.getState().setDials(HOPELESS_POINT);
      store.getState().tryCurrent();
    }
    store.getState().setDials(CRACKING_POINT);
    store.getState().tryCurrent();
    const late = store.getState().evaluation!.score;

    expect(early).toBeGreaterThan(late);
  });

  it("resets to a fresh safe", () => {
    store.getState().runToBudget("random");
    store.getState().reset();
    expect(store.getState().trials).toEqual([]);
    expect(store.getState().revealed).toBe(false);
    expect(store.getState().phase).toBe("cracking");
    expect(store.getState().failure).toBeNull();
  });
});

describe("the visual lane's random run", () => {
  const store = useHeistStore;

  beforeEach(() => {
    store.getState().reset();
  });

  it("opens the safe on the first attempt, as random search usually does", () => {
    // The old fixed seed was one of the ~20% of draws that never crack, so every
    // player's first random run lost to the grid while the copy said it won.
    const trials = runStrategy("random", BUDGET, randomSeedFor(0));
    expect(bestTrial(trials)!.objectiveValue).toBeGreaterThanOrEqual(CRACK_THRESHOLD);
    expect(bestTrial(trials)!.objectiveValue).toBeGreaterThan(gridBest(BUDGET));
  });

  it("draws afresh on each attempt, reproducibly", () => {
    const attempt = store.getState().attempt;
    store.getState().runToBudget("random");
    const first = store.getState().trials.map((trial) => trial.params);
    expect(first[0]).toEqual(randomPoints(BUDGET, randomSeedFor(attempt))[0]);

    store.getState().reset();
    expect(store.getState().attempt).toBe(attempt + 1);
    store.getState().runToBudget("random");
    const second = store.getState().trials.map((trial) => trial.params);
    expect(second[0]).not.toEqual(first[0]);
    expect(randomSeedFor(0)).toBe(RANDOM_SEED);
  });

  it("stops spending the moment the safe opens", () => {
    // A tuner with a target stops too, and every try it did not need is score.
    store.getState().setStrategy("bayesian");
    store.getState().runToBudget("bayesian");
    const state = store.getState();
    expect(state.phase).toBe("cracked");
    expect(crackedOn(state)).toBe(state.trials.length);
    expect(triesLeft(state)).toBeGreaterThan(0);
    // The budget readout and the score agree about which try opened it.
    expect(state.evaluation!.score).toBeCloseTo(
      crackScore(crackedOn(state)!, BUDGET),
      12,
    );
  });
});

describe("the random run's WhyCard", () => {
  // Found by search rather than written down, like CRACKING_POINT.
  const UNLUCKY_SEED = (() => {
    for (let seed = 1; seed < 5000; seed += 1) {
      const trials = runStrategy("random", BUDGET, seed);
      if (bestTrial(trials)!.objectiveValue < CRACK_THRESHOLD) return seed;
    }
    throw new Error("every random run cracks");
  })();

  const cardFor = (trials: Trial[], cracked: boolean) =>
    whyCardFor({
      kind: "strategy-run",
      strategy: "random",
      trials,
      added: trials,
      budget: BUDGET,
      cracked,
    });

  it("does not claim a win for a draw that lost", () => {
    const card = cardFor(runStrategy("random", BUDGET, UNLUCKY_SEED), false);
    expect(card.body).not.toMatch(/It wins/);
    expect(card.body).toMatch(/unlucky/i);
    expect(card.tone).toBe("warn");
  });

  it("quotes the crack rate it measured, not one it assumed", () => {
    let cracked = 0;
    for (let seed = 1; seed <= CRACK_RATE_RUNS; seed += 1) {
      const best = bestTrial(runStrategy("random", BUDGET, seed))!;
      if (best.objectiveValue >= CRACK_THRESHOLD) cracked += 1;
    }
    expect(randomCrackRate(BUDGET)).toBe(cracked / CRACK_RATE_RUNS);
    expect(randomCrackRate(BUDGET)).toBeGreaterThan(0.6);

    const card = cardFor(runStrategy("random", BUDGET, UNLUCKY_SEED), false);
    expect(card.body).toContain(
      `${Math.round(randomCrackRate(BUDGET) * 100)}% open the safe`,
    );
    expect(card.body).toContain(`${CRACK_RATE_RUNS} simulated`);
  });

  it("compares against the grid's real best, whichever side it lands", () => {
    const card = cardFor(runStrategy("random", BUDGET, UNLUCKY_SEED), false);
    expect(card.body).toContain(`${(gridBest(BUDGET) * 100).toFixed(1)}%`);
  });

  it("credits a random run that did open the safe", () => {
    const trials = runStrategy("random", BUDGET, randomSeedFor(0));
    const first = trials.findIndex((trial) => trial.objectiveValue >= CRACK_THRESHOLD);
    const card = cardFor(trials.slice(0, first + 1), true);
    expect(card.body).toMatch(/open/);
    expect(card.body).toMatch(/It wins/);
    expect(card.tone).toBe("good");
  });

  it("calls a first-try crack luck, not a strategy winning", () => {
    // A run stops at the crack, so some draws open the safe on try 1 having
    // seen one learning rate. That card used to say "no more resolution than
    // the grid" and then "It wins because…" in the same breath.
    const seed = (() => {
      for (let candidate = 1; candidate < 5000; candidate += 1) {
        const first = runStrategy("random", BUDGET, candidate)[0]!;
        if (first.objectiveValue >= CRACK_THRESHOLD) return candidate;
      }
      throw new Error("no random run cracks on its first try");
    })();
    const card = cardFor(runStrategy("random", BUDGET, seed).slice(0, 1), true);

    expect(card.title).toBe("Random spent 1 try and saw 1 learning rate");
    expect(card.body).toMatch(/opened on try 1/);
    expect(card.body).toMatch(/lucky/i);
    expect(card.body).not.toMatch(/It wins/);
    expect(card.body).not.toMatch(/no more resolution/);
    expect(card.body).toContain(
      `${Math.round(randomCrackRate(BUDGET) * 100)}% open the safe`,
    );
    expect(card.body).toContain("15 tries to spare");
  });

  it("never credits the strategy on a crack that saw no more learning rates than the grid", () => {
    // Every attempt the visual lane can draw, driven through the real store so
    // the card is judged on the truncated run it actually receives.
    const store = useHeistStore;
    const levels = gridLevels(BUDGET);
    let early = 0;
    let credited = 0;
    for (let attempt = 0; attempt < 200; attempt += 1) {
      store.getState().reset();
      store.getState().setStrategy("random");
      store.getState().runToBudget("random");
      const state = store.getState();
      if (state.phase !== "cracked") continue;
      const body = state.whyCard!.body;
      if (distinctValuesTried(state.trials, LEARNING_RATE) <= levels) {
        early += 1;
        expect(body, `attempt ${state.attempt}`).not.toMatch(/It wins/);
        expect(body, `attempt ${state.attempt}`).toMatch(/lucky/i);
      } else {
        credited += 1;
        expect(body, `attempt ${state.attempt}`).toMatch(/It wins/);
      }
    }
    // Both branches are genuinely exercised by the draws a player can get.
    expect(early).toBeGreaterThan(0);
    expect(credited).toBeGreaterThan(0);
  });
});

describe("Bayesian simulations are genuinely different runs", () => {
  it("varies its warm-up with the seed, so an average is an average", () => {
    // The warm-up was the same four Halton points for every seed, so sixty
    // "runs" were four distinct sequences and two distinct results.
    const runs = Array.from({ length: 60 }, (_, index) =>
      runStrategy("bayesian", BUDGET, (index + 1) * 7919),
    );
    const openings = new Set(
      runs.map((trials) =>
        trials
          .slice(0, BAYESIAN_SEED_TRIALS)
          .map((trial) => trial.params.map((value) => value.toFixed(4)).join(","))
          .join("|"),
      ),
    );
    const bests = new Set(
      runs.map((trials) => bestTrial(trials)!.objectiveValue.toFixed(6)),
    );
    expect(openings.size).toBe(60);
    expect(bests.size).toBeGreaterThan(20);
  });

  it("keeps the visual lane's hint identical on every safe", () => {
    // The hint box and "Spend up to … on bayesian" must agree with each other.
    expect(suggestNext([]).point).toEqual(haltonPoints(BAYESIAN_SEED_TRIALS, 1)[0]);
  });

  it("hands back a copy of its pick, never a shared candidate", () => {
    const seeded = asTrials(haltonPoints(BAYESIAN_SEED_TRIALS, 1), "bayesian");
    const first = suggestNext(seeded);
    first.point[0] = 42;
    expect(suggestNext(seeded).point[0]).not.toBe(42);
  });
});

describe("non-finite input cannot poison a run", () => {
  const store = useHeistStore;

  beforeEach(() => {
    store.getState().reset();
  });

  it("puts a NaN dial back to the centre instead of trying it", () => {
    store.getState().setDials([0.3, 0.5, Number.NaN, 0.5]);
    expect(store.getState().dials).toEqual([0.3, 0.5, 0.5, 0.5]);
    store.getState().setDial(0, Number.NaN);
    expect(store.getState().dials[0]).toBe(0.3);
    const trial = store.getState().tryCurrent()!;
    expect(Number.isFinite(trial.objectiveValue)).toBe(true);
  });

  it("never lets a NaN reading become the best", () => {
    const poisoned: Trial = {
      params: [0.5, 0.5, 0.5, 0.5],
      objectiveValue: Number.NaN,
      index: 1,
      source: "manual",
    };
    const crack: Trial = {
      params: CRACKING_POINT,
      objectiveValue: objectiveAt(CRACKING_POINT),
      index: 2,
      source: "manual",
    };
    expect(bestTrial([poisoned, crack])).toBe(crack);
    const evaluation = evaluateRun({
      trials: [poisoned, crack],
      budget: BUDGET,
      finished: false,
    });
    expect(evaluation.outcome).toBe("cracked");
  });
});

describe("scoring and the star criteria", () => {
  it("states the try that still earns the high-score star, from the formula", () => {
    const latest = latestCrackScoring(HIGH_SCORE_THRESHOLD, BUDGET);
    expect(crackScore(latest, BUDGET)).toBeGreaterThanOrEqual(HIGH_SCORE_THRESHOLD);
    expect(crackScore(latest + 1, BUDGET)).toBeLessThan(HIGH_SCORE_THRESHOLD);
    expect(latest).toBe(9);
  });

  it("scores a crack on the first try as perfect and falls with every try", () => {
    expect(crackScore(1, BUDGET)).toBe(1);
    for (let tryNumber = 2; tryNumber <= BUDGET; tryNumber += 1) {
      expect(crackScore(tryNumber, BUDGET)).toBeLessThan(
        crackScore(tryNumber - 1, BUDGET),
      );
    }
  });
});

describe("the Grid trap names numbers it measured", () => {
  it("reports the dominant dial's swing in points, and the real resolution ratio", () => {
    const evaluation = evaluateRun({
      trials: runStrategy("grid", BUDGET, 0),
      budget: BUDGET,
      finished: true,
    });
    const swing = Math.round(dialInfluence()[LEARNING_RATE]! * 100);
    expect(evaluation.failure!.detail).toContain(`by ${swing} points`);
    expect(evaluation.failure!.detail).toContain(
      `${BUDGET / gridLevels(BUDGET)} times the resolution`,
    );
    // It used to call that absolute swing a percentage of the range.
    expect(evaluation.failure!.detail).not.toMatch(/of this safe's range/);
  });
});

describe("the code-lane star follows the action, not the tab", () => {
  const store = useHeistStore;
  const recorded: Array<{ lane: string; codeLaneCleared: boolean }> = [];
  const realRecord = useProgression.getState().recordResult;

  beforeEach(() => {
    recorded.length = 0;
    useProgression.setState({
      recordResult: vi.fn((result: Parameters<typeof realRecord>[0]) => {
        recorded.push({ lane: result.lane, codeLaneCleared: result.codeLaneCleared === true });
        return realRecord(result);
      }),
    });
    store.getState().reset();
  });

  afterEach(() => {
    useProgression.setState({ recordResult: realRecord });
  });

  it("does not award it for a rail click while the code tab is open", () => {
    store.getState().setLane("code");
    store.getState().setDials(CRACKING_POINT);
    store.getState().tryCurrent();
    expect(recorded).toEqual([{ lane: "visual", codeLaneCleared: false }]);
  });

  it("awards it for a crack made by a code-lane call", () => {
    const api = createCodeApi();
    api.setDials(CRACKING_POINT);
    api.try();
    expect(store.getState().phase).toBe("cracked");
    expect(recorded).toEqual([{ lane: "code", codeLaneCleared: true }]);
  });

  it("records a crack once, however often the verb is called again", () => {
    const api = createCodeApi();
    api.setDials(CRACKING_POINT);
    api.try();
    expect(api.try()).toBeNull();
    expect(api.runStrategy("random")).toEqual([]);
    expect(recorded).toHaveLength(1);
  });
});

describe("the code-lane api refuses bad input by name", () => {
  const store = useHeistStore;

  beforeEach(() => {
    store.getState().reset();
  });

  it("rejects dials that are not four finite unit coordinates", () => {
    const api = createCodeApi();
    expect(() => api.setDials([0.3, 0.5, Number.NaN, 0.5])).toThrow(/finite number/);
    expect(() => api.setDials([0.3, 0.5])).toThrow(/4 unit coordinates/);
    expect(() => api.setDials([0.3, 0.5, 32, 0.5])).toThrow(/\[0, 1\]/);
    expect(() => api.setDials("centre")).toThrow(/4 unit coordinates/);
    expect(store.getState().dials).toEqual(new Array(DIAL_COUNT).fill(0.5));
    expect(triesUsed(store.getState())).toBe(0);
  });

  it("rejects a real value its dial cannot hold", () => {
    const api = createCodeApi();
    expect(() => api.setDialValue(0, -1)).toThrow(/learning rate runs from/);
    expect(() => api.setDialValue(2, 512)).toThrow(/batch size runs from/);
    expect(() => api.setDialValue(7, 0.1)).toThrow(/no dial 7/);
    expect(() => api.setDialValue(0, Number.NaN)).toThrow(/finite number/);
    api.setDialValue(0, 0.01);
    expect(store.getState().dials[0]).toBeCloseTo(0.5, 8);
  });

  it("will not simulate a strategy it does not know", () => {
    const api = createCodeApi();
    // "Random" used to run Bayesian search under the typo's label.
    expect(() => api.simulate("Random", 1)).toThrow(/unknown strategy "Random"/);
    expect(() => api.runStrategy("grd")).toThrow(/unknown strategy/);
    expect(() => api.simulate("random", 1.5)).toThrow(/whole-number seed/);
    expect(() => api.simulate("random", -3)).toThrow(/whole-number seed/);
    expect(api.simulate("random", 3).strategy).toBe("random");
  });

  it("hands back copies, so a snippet cannot edit the store behind its back", () => {
    const api = createCodeApi();
    const trial = api.try()!;
    trial.params[0] = 0.99;
    expect(store.getState().trials[0]!.params[0]).toBe(0.5);
    const planned = api.runStrategy("grid");
    planned[0]!.params[0] = 0.01;
    expect(store.getState().trials[1]!.params[0]).not.toBe(0.01);
  });

  it("exposes the visual lane's random seed for this attempt", () => {
    const api = createCodeApi();
    expect(api.randomSeed()).toBe(randomSeedFor(store.getState().attempt));
    store.getState().runToBudget("random");
    expect(store.getState().trials[0]!.params).toEqual(
      api.randomPoints(api.randomSeed())[0],
    );
  });
});

describe("the starter snippet", () => {
  it("runs to the end, pauses between runs, and prints the three rows", async () => {
    const logs: string[] = [];
    let budgetChecks = 0;
    await createJsExecutor<ReturnType<typeof createCodeApi>>()(STARTER_CODE, {
      api: createCodeApi(),
      log: (...args: unknown[]) => logs.push(args.map(String).join(" ")),
      checkBudget: () => {
        budgetChecks += 1;
      },
    });
    // A result row starts with the padded strategy name; "grid at this
    // budget:" does not.
    const rowOf = (name: string) =>
      logs.find((line) => line.startsWith(`${name.padEnd(10)} `));
    for (const strategy of ["grid", "random", "bayesian"]) {
      expect(rowOf(strategy), strategy).toBeDefined();
    }
    // It yields to the page (and so can be stopped) rather than running a
    // hundred searches in one long task.
    expect(budgetChecks).toBeGreaterThan(40);
    const crackColumn = (name: string) =>
      Number(
        rowOf(name)!
          .trim()
          .split(/\s+/)[2]!
          .replace("%", ""),
      );
    expect(crackColumn("grid")).toBe(0);
    expect(crackColumn("bayesian")).toBeGreaterThan(crackColumn("random"));
  }, 120000);
});
