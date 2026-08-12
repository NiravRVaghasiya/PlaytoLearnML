"use client";

import { useMemo } from "react";
import { Play, RotateCcw } from "lucide-react";
import { Button, CodeEditor } from "@/components";
import { useCodeLane } from "@/engine/useCodeLane";
import {
  BUDGET,
  CRACK_THRESHOLD,
  DIALS,
  bestTrial,
  describePoint,
  dialInfluence,
  distinctValuesTried,
  expectedImprovement,
  fitSurrogate,
  gridLevels,
  gridPoints,
  objectiveAt,
  randomPoints,
  runStrategy,
  suggestNext,
  toRealValue,
  toUnitValue,
  type Strategy,
} from "./ml";
import { useHeistStore } from "./store";

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
 */

const STARTER_CODE = `// One run of anything is luck. Settle it over many.

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
  }

  log(strategy.padEnd(10),
      pct(total / runs).padStart(9),
      pct(cracked / runs).padStart(12),
      (lrSeen / runs).toFixed(1).padStart(14));
}

log('');
log('Same budget every time. The third column is the argument;');
log('the fourth column is the reason.');

function pct(x) { return (x * 100).toFixed(1) + '%'; }`;

export function CodeLane() {
  const store = useHeistStore;

  const api = useMemo(
    () => ({
      /** Set every dial, in unit coordinates. Same state the sliders write. */
      setDials: (point: number[]) => {
        if (!Array.isArray(point)) {
          throw new Error("setDials needs an array of unit coordinates");
        }
        store.getState().setDials(point);
      },
      /** Set one dial by its real value, e.g. setDialValue(0, 0.01). */
      setDialValue: (index: number, value: number) => {
        const dial = DIALS[index];
        if (!dial) throw new Error(`no dial ${index}; there are ${DIALS.length}`);
        if (!Number.isFinite(value)) {
          throw new Error("setDialValue needs a finite number");
        }
        store.getState().setDial(index, toUnitValue(dial, value));
      },

      /** Spend one try at the current dials. Costs budget. */
      try: () => store.getState().tryCurrent(),
      /** Spend one try where Expected Improvement points. Costs budget. */
      trySuggestion: () => store.getState().trySuggestion(),
      /** Spend every remaining try with one strategy. Costs budget. */
      runStrategy: (strategy: Strategy) => {
        if (!["grid", "random", "bayesian"].includes(strategy)) {
          throw new Error(`unknown strategy "${strategy}"`);
        }
        return store.getState().runToBudget(strategy);
      },
      reset: () => store.getState().reset(),

      /**
       * Run a strategy against the objective without touching the player's budget.
       *
       * This is what makes the comparison possible at all: settling "does random
       * beat grid" needs dozens of runs, and charging tries for them would make the
       * question unaskable.
       */
      simulate: (strategy: Strategy, seed = 1) => {
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
      objectiveAt: (point: number[]) => objectiveAt(point),
      /** How much each dial moves the objective on its own. */
      influence: () =>
        dialInfluence().map((value, index) => ({
          dial: DIALS[index]!.name,
          influence: value,
        })),

      /** Where Bayesian search would look next, given the tries so far. */
      suggestion: () => {
        const suggestion = suggestNext(store.getState().trials);
        return {
          point: [...suggestion.point],
          expectedImprovement: suggestion.expectedImprovement,
          predictedMean: suggestion.predictedMean,
          predictedSd: suggestion.predictedSd,
          describe: describePoint(suggestion.point),
        };
      },
      /** The surrogate's belief at a point, given the tries so far. */
      surrogateAt: (point: number[]) => {
        const surrogate = fitSurrogate(store.getState().trials);
        if (surrogate === null) return null;
        const { mean, sd } = surrogate.predict(point);
        return {
          mean,
          sd,
          expectedImprovement: expectedImprovement(surrogate, point),
        };
      },

      gridPoints: () => gridPoints(BUDGET).map((point) => [...point]),
      randomPoints: (seed = 1) =>
        randomPoints(BUDGET, seed).map((point) => [...point]),
      gridLevels: () => gridLevels(BUDGET),

      dials: () =>
        DIALS.map((dial, index) => ({
          index,
          name: dial.name,
          short: dial.short,
          min: dial.min,
          max: dial.max,
          scale: dial.scale,
          current: toRealValue(dial, store.getState().dials[index] ?? 0.5),
        })),
      trials: () =>
        store.getState().trials.map((trial) => ({
          index: trial.index,
          params: [...trial.params],
          objectiveValue: trial.objectiveValue,
          source: trial.source,
        })),
      best: () => {
        const best = bestTrial(store.getState().trials);
        return best === null
          ? null
          : {
              objectiveValue: best.objectiveValue,
              params: [...best.params],
              index: best.index,
              describe: describePoint(best.params),
            };
      },
      triesLeft: () =>
        Math.max(0, store.getState().budget - store.getState().trials.length),
      budget: () => store.getState().budget,
      crackThreshold: () => CRACK_THRESHOLD,
      phase: () => store.getState().phase,
      evaluation: () => store.getState().evaluation,
    }),
    [store],
  );

  const lane = useCodeLane({ initialCode: STARTER_CODE, api, maxRunMs: 60000 });

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="primary"
          size="sm"
          onClick={() => void lane.run()}
          disabled={lane.running}
          icon={<Play className="size-4" />}
        >
          {lane.running ? "Running…" : "Run"}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={lane.reset}
          icon={<RotateCcw className="size-4" />}
        >
          Restore snippet
        </Button>
        {lane.dirty ? (
          <span className="font-mono text-xs text-text-muted">edited</span>
        ) : null}
      </div>

      <CodeEditor
        className="min-h-0 flex-1"
        label="Cracking script"
        value={lane.code}
        onChange={lane.setCode}
        language="javascript"
        error={lane.error}
        rows={18}
        hint="api.simulate · api.setDials · api.setDialValue · api.try · api.trySuggestion · api.runStrategy · api.objectiveAt · api.influence · api.suggestion · api.surrogateAt · api.gridPoints · api.dials · api.best · api.triesLeft"
      />

      <section aria-label="Script output" className="min-h-24">
        <h2 className="mb-1 text-xs font-semibold tracking-wide text-text-muted uppercase">
          Output
        </h2>
        {lane.logs.length === 0 ? (
          <p className="font-mono text-xs text-text-muted">
            Run the script to see output here. `api.simulate` costs you nothing —
            only `api.try` and `api.runStrategy` spend the budget.
          </p>
        ) : (
          <ul className="max-h-40 overflow-auto rounded-md border border-border bg-bg p-2 font-mono text-xs whitespace-pre">
            {lane.logs.map((entry, index) => (
              <li
                key={index}
                className={
                  entry.level === "error" ? "text-wrong" : "text-text-muted"
                }
              >
                {entry.message}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
