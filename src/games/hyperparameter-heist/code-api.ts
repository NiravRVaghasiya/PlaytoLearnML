import {
  BUDGET,
  CRACK_THRESHOLD,
  DIALS,
  DIAL_COUNT,
  STRATEGIES,
  bestTrial,
  describePoint,
  dialInfluence,
  distinctValuesTried,
  expectedImprovement,
  fitSurrogate,
  gridLevels,
  gridPoints,
  isStrategy,
  objectiveAt,
  randomPoints,
  runStrategy,
  suggestNext,
  toRealValue,
  toUnitValue,
  type Trial,
  type UnitPoint,
} from "./ml";
import { randomSeedFor, useHeistStore } from "./store";

/**
 * Hyperparameter Heist — the code lane.
 *
 * `api.setDials` and `api.try` write the same state the sliders and the Try button
 * write, and `api.runStrategy` is the same routine the toggle runs (CLAUDE.md
 * two-lane rule).
 *
 * The starter snippet is the experiment the visual lane cannot run: all three
 * strategies, over many seeds, with the crack rates printed. One run of anything is
 * luck, and the whole claim of this game is about averages — so the code lane is
 * where the claim is actually settled. `api.simulate` runs strategies against the
 * objective WITHOUT spending the player's budget, because a comparison that cost
 * tries would be self-defeating.
 *
 * ── Two rules the api keeps ─────────────────────────────────────────────────
 * 1. Bad input is refused with a sentence, not absorbed. A NaN dial used to be
 *    spent as a try, score NaN, and become an incumbent no later crack could
 *    displace; a mistyped strategy name silently ran Bayesian search under the
 *    typo's label. Every verb now checks its arguments first.
 * 2. Everything handed back is a copy. The store's trials are its state, and a
 *    snippet that edited `t.params` would have been editing it behind zustand's
 *    back.
 */

export const STARTER_CODE = `// One run of anything is luck. Settle it over many.
// (A few hundred simulated searches is real work, so the loop pauses every
// couple of runs to let the page repaint — see pause() at the bottom.)

const RUNS = 60;
log('budget', api.budget(), '| dials', api.dials().length,
    '| opens at', pct(api.crackThreshold()));
log('grid at this budget:', api.gridLevels(), 'settings per dial =',
    Math.pow(api.gridLevels(), api.dials().length), 'tries');
log('');
log('strategy    best(mean)   crack rate   lr values seen');

for (const strategy of ['grid', 'random', 'bayesian']) {
  let total = 0, cracked = 0, lrSeen = 0;

  // Grid is deterministic, so one run is the whole story.
  const runs = strategy === 'grid' ? 1 : RUNS;

  for (let seed = 1; seed <= runs; seed++) {
    const r = api.simulate(strategy, seed * 7919);
    total += r.best;
    if (r.best >= api.crackThreshold()) cracked++;
    lrSeen += r.distinctLearningRates;
    if (seed % 2 === 0) await pause();
  }

  log(strategy.padEnd(10),
      pct(total / runs).padStart(9),
      pct(cracked / runs).padStart(12),
      (lrSeen / runs).toFixed(1).padStart(14));
}

log('');
log('Same budget every time. The third column is the argument;');
log('the fourth column is the reason.');

function pct(x) { return (x * 100).toFixed(1) + '%'; }
function pause() { checkBudget(); return new Promise((r) => setTimeout(r, 0)); }`;

/** A unit-cube point from a snippet, or a clear error saying what is wrong. */
export function unitPointFrom(point: unknown, verb: string): UnitPoint {
  if (!Array.isArray(point) || point.length !== DIAL_COUNT) {
    throw new Error(
      `${verb} needs an array of ${DIAL_COUNT} unit coordinates, one per dial (${DIALS.map(
        (dial) => dial.short,
      ).join(", ")})`,
    );
  }
  return point.map((value, index) => {
    const dial = DIALS[index]!;
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error(
        `${verb}: ${dial.name} must be a finite number, got ${String(value)}`,
      );
    }
    if (value < 0 || value > 1) {
      throw new Error(
        `${verb}: ${dial.name} is a unit coordinate in [0, 1], got ${value} — use setDialValue(${index}, …) for a real value`,
      );
    }
    return value;
  });
}

/** A trial for the snippet: a copy, never the store's own object. */
function trialCopy(trial: Trial | null) {
  return trial === null
    ? null
    : {
        index: trial.index,
        params: [...trial.params],
        objectiveValue: trial.objectiveValue,
        source: trial.source,
      };
}

/**
 * The code lane's api, as a plain object over the store.
 *
 * Lives outside the component so its argument checks and copies can be tested
 * without rendering anything, the same way the other games test theirs. Every
 * verb reads the store when it is called, so one object per mount is enough.
 */
export function createCodeApi() {
  return {
    /** Set every dial, in unit coordinates. Same state the sliders write. */
    setDials: (point: unknown) => {
      useHeistStore.getState().setDials(unitPointFrom(point, "setDials"));
    },
    /** Set one dial by its real value, e.g. setDialValue(0, 0.01). */
    setDialValue: (index: unknown, value: unknown) => {
      const dial =
        typeof index === "number" && Number.isInteger(index)
          ? DIALS[index]
          : undefined;
      if (!dial) {
        throw new Error(
          `no dial ${String(index)}; dials are numbered 0 to ${DIALS.length - 1}`,
        );
      }
      if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new Error(`setDialValue needs a finite number, got ${String(value)}`);
      }
      // Out of range is refused rather than clamped: on a log dial, zero or a
      // negative value has no position at all, and a silent clamp would hide
      // a units mistake (a batch size of 512, a rate of 10).
      if (value < dial.min || value > dial.max) {
        throw new Error(
          `${dial.name} runs from ${dial.min} to ${dial.max}; got ${value}`,
        );
      }
      useHeistStore.getState().setDial(index as number, toUnitValue(dial, value));
    },

    /** Spend one try at the current dials. Costs budget. */
    try: () => trialCopy(useHeistStore.getState().tryCurrent("code")),
    /** Spend one try where Expected Improvement points. Costs budget. */
    trySuggestion: () => trialCopy(useHeistStore.getState().trySuggestion("code")),
    /**
     * Spend the remaining tries with one strategy, stopping when the safe
     * opens. Costs budget.
     */
    runStrategy: (strategy: unknown) => {
      if (!isStrategy(strategy)) {
        throw new Error(
          `unknown strategy ${JSON.stringify(strategy)} — try ${STRATEGIES.join(", ")}`,
        );
      }
      return useHeistStore
        .getState()
        .runToBudget(strategy, "code")
        .map((trial) => trialCopy(trial)!);
    },
    reset: () => useHeistStore.getState().reset(),

    /**
     * Run a strategy against the objective without touching the player's budget.
     *
     * This is what makes the comparison possible at all: settling "does random
     * beat grid" needs dozens of runs, and charging tries for them would make the
     * question unaskable.
     */
    simulate: (strategy: unknown, seed: unknown = 1) => {
      if (!isStrategy(strategy)) {
        throw new Error(
          `unknown strategy ${JSON.stringify(strategy)} — try ${STRATEGIES.join(", ")}`,
        );
      }
      if (
        typeof seed !== "number" ||
        !Number.isSafeInteger(seed) ||
        seed < 0
      ) {
        throw new Error(
          `simulate needs a whole-number seed of 0 or more, got ${String(seed)}`,
        );
      }
      const trials = runStrategy(strategy, BUDGET, seed);
      const best = bestTrial(trials)!;
      const firstCrack = trials.find(
        (trial) => trial.objectiveValue >= CRACK_THRESHOLD,
      );
      return {
        strategy,
        best: best.objectiveValue,
        bestAt: describePoint(best.params),
        firstCrackOn: firstCrack?.index ?? null,
        distinctLearningRates: distinctValuesTried(trials, 0),
        trials: trials.map((trial) => ({
          index: trial.index,
          params: [...trial.params],
          objectiveValue: trial.objectiveValue,
        })),
      };
    },

    /** Read the objective anywhere, for free. Analysis, not a try. */
    objectiveAt: (point: unknown) =>
      objectiveAt(unitPointFrom(point, "objectiveAt")),
    /** How much each dial moves the objective on its own. */
    influence: () =>
      dialInfluence().map((value, index) => ({
        dial: DIALS[index]!.name,
        influence: value,
      })),

    /** Where Bayesian search would look next, given the tries so far. */
    suggestion: () => {
      const suggestion = suggestNext(useHeistStore.getState().trials);
      return {
        point: [...suggestion.point],
        expectedImprovement: suggestion.expectedImprovement,
        predictedMean: suggestion.predictedMean,
        predictedSd: suggestion.predictedSd,
        describe: describePoint(suggestion.point),
      };
    },
    /** The surrogate's belief at a point, given the tries so far. */
    surrogateAt: (point: unknown) => {
      const unit = unitPointFrom(point, "surrogateAt");
      const surrogate = fitSurrogate(useHeistStore.getState().trials);
      if (surrogate === null) return null;
      const { mean, sd } = surrogate.predict(unit);
      return {
        mean,
        sd,
        expectedImprovement: expectedImprovement(surrogate, unit),
      };
    },

    gridPoints: () => gridPoints(BUDGET).map((point) => [...point]),
    randomPoints: (seed: unknown = 1) => {
      if (typeof seed !== "number" || !Number.isSafeInteger(seed) || seed < 0) {
        throw new Error(
          `randomPoints needs a whole-number seed of 0 or more, got ${String(seed)}`,
        );
      }
      return randomPoints(BUDGET, seed).map((point) => [...point]);
    },
    /**
     * The seed the visual lane's Random strategy draws from on this attempt,
     * so `api.randomPoints(api.randomSeed())` is exactly the run it will make.
     */
    randomSeed: () => randomSeedFor(useHeistStore.getState().attempt),
    gridLevels: () => gridLevels(BUDGET),

    dials: () =>
      DIALS.map((dial, index) => ({
        index,
        name: dial.name,
        short: dial.short,
        min: dial.min,
        max: dial.max,
        scale: dial.scale,
        current: toRealValue(dial, useHeistStore.getState().dials[index] ?? 0.5),
      })),
    trials: () =>
      useHeistStore.getState().trials.map((trial) => ({
        index: trial.index,
        params: [...trial.params],
        objectiveValue: trial.objectiveValue,
        source: trial.source,
      })),
    best: () => {
      const best = bestTrial(useHeistStore.getState().trials);
      return best === null
        ? null
        : {
            objectiveValue: best.objectiveValue,
            params: [...best.params],
            index: best.index,
            describe: describePoint(best.params),
          };
    },
    triesLeft: () => {
      const { budget, trials } = useHeistStore.getState();
      return Math.max(0, budget - trials.length);
    },
    budget: () => useHeistStore.getState().budget,
    crackThreshold: () => CRACK_THRESHOLD,
    phase: () => useHeistStore.getState().phase,
    evaluation: () => {
      const evaluation = useHeistStore.getState().evaluation;
      return evaluation === null
        ? null
        : {
            outcome: evaluation.outcome,
            score: evaluation.score,
            triesUsed: evaluation.triesUsed,
            budget: evaluation.budget,
            best: evaluation.best ? trialCopy(evaluation.best) : null,
            failure: evaluation.failure ? { ...evaluation.failure } : null,
          };
    },
  };
}
