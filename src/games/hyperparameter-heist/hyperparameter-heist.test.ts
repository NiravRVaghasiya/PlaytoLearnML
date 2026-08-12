import { beforeEach, describe, expect, it } from "vitest";
import {
  BASE_SCORE,
  BATCH_SIZE,
  BAYESIAN_SEED_TRIALS,
  BUDGET,
  CRACK_THRESHOLD,
  DIALS,
  DIAL_COUNT,
  GRID_TRAP_DISTINCT,
  LEARNING_RATE,
  MOMENTUM,
  WEIGHT_DECAY,
  bestTrial,
  describePoint,
  dialInfluence,
  distinctValuesTried,
  dominantDial,
  evaluateRun,
  expectedImprovement,
  fitSurrogate,
  gridLevels,
  gridPoints,
  haltonPoints,
  objectiveAt,
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
  bestObjective,
  lastTemperature,
  triesLeft,
  triesUsed,
  useHeistStore,
} from "./store";

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
