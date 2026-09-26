"use client";

import { create } from "zustand";
import { useProgression } from "@/engine/progression";
import type { Lane, NamedFailure } from "@/engine/types";
import type { WhyCardContent } from "@/components";
import { clamp } from "@/lib/utils";
import {
  BUDGET,
  CRACK_THRESHOLD,
  DIAL_COUNT,
  DIALS,
  bestTrial,
  evaluateRun,
  gridPoints,
  objectiveAt,
  randomPoints,
  suggestNext,
  temperatureOf,
  toUnitValue,
  type Evaluation,
  type Strategy,
  type Suggestion,
  type Temperature,
  type Trial,
  type UnitPoint,
} from "./ml";
import { whyCardFor } from "./why-cards";

export const SLUG = "hyperparameter-heist";

/**
 * Base seed for the visual lane's random strategy.
 *
 * Each attempt at the safe draws from `RANDOM_SEED + attempt`, so a run is
 * reproducible (the code lane can replay it with `api.randomPoints(api.randomSeed())`)
 * but "Crack it again" is a genuinely new draw rather than the same sixteen points.
 *
 * The base is chosen so the FIRST attempt's draw is typical of random search on
 * this budget — it opens the safe, on try 5, which is the median first crack over
 * a thousand seeds. The previous base, 60613, was one of the ~20% of seeds that
 * never crack, so every player's first random run lost to the grid while the
 * copy said random wins. The tests pin both halves: this draw cracks, and the
 * copy for a draw that does not is honest about it.
 */
export const RANDOM_SEED = 60641;

/** The seed the visual lane's random strategy uses on a given attempt. */
export function randomSeedFor(attempt: number): number {
  return RANDOM_SEED + attempt;
}

export type Phase = "cracking" | "cracked" | "busted";

/**
 * Hyperparameter Heist state.
 *
 * The spec's data model is:
 *
 *   HyperParam { name, range, currentValue }
 *   Trial      { params{}, objectiveValue }
 *   GameState  { params[], trials[], budget, mode, best }
 *
 * `HyperParam`'s name and range live in `DIALS` as module constants — they never
 * change, so storing them per-game would be a copy that can drift. Only
 * `currentValue` is state, held as `dials`.
 *
 * `best` is not stored either: it is `bestTrial(trials)`, and a stored copy could
 * disagree with the trial list it is supposed to summarise.
 */
export interface HeistState {
  /** Current dial positions, in unit-cube coordinates. */
  dials: UnitPoint;
  trials: Trial[];
  budget: number;
  /** Which strategy the toggle is showing (spec: `mode`). */
  strategy: Strategy | "manual";
  /**
   * Whether the true objective surface is on show.
   *
   * False while cracking: you cannot see a response surface before you have
   * sampled it, and drawing it would hand over the answer. Revealed when the run
   * ends, which is the moment it teaches the most — you get to see what your tries
   * were covering.
   */
  revealed: boolean;

  phase: Phase;
  evaluation: Evaluation | null;
  failure: NamedFailure | null;
  whyCard: WhyCardContent | null;
  /** Which tab is on show. NOT what earns the code-lane star — see `source`. */
  lane: Lane;
  /**
   * How many times the safe has been reset this session. Drives the random
   * strategy's seed, so each attempt is a fresh, reproducible draw.
   */
  attempt: number;

  setLane: (lane: Lane) => void;
  setDial: (index: number, unit: number) => void;
  /** Set every dial at once — used by suggestions and the code lane. */
  setDials: (point: UnitPoint) => void;
  setStrategy: (strategy: Strategy | "manual") => void;
  /**
   * Spend one try at the current dial positions.
   *
   * `source` is the lane the ACTION came from, which is what the progression
   * service is told: a crack from a code-lane `api.try()` earns the code-lane
   * star, a click on the rail's button does not, whichever tab happens to be
   * open at the time.
   */
  tryCurrent: (source?: Lane) => Trial | null;
  /** Spend one try wherever the acquisition function points. */
  trySuggestion: (source?: Lane) => Trial | null;
  /**
   * Spend the remaining tries with one strategy, stopping as soon as the safe
   * opens — a tuner with a target stops too, and the tries it did not need are
   * exactly what the score rewards.
   */
  runToBudget: (strategy: Strategy, source?: Lane) => Trial[];
  reveal: () => void;
  reset: () => void;
}

// ── selectors: primitives and stable references only ─────────────────────

export function triesUsed(state: HeistState): number {
  return state.trials.length;
}

export function triesLeft(state: HeistState): number {
  return Math.max(0, state.budget - state.trials.length);
}

export function bestObjective(state: HeistState): number {
  return bestTrial(state.trials)?.objectiveValue ?? Number.NaN;
}

export function lastTemperature(state: HeistState): Temperature | null {
  const last = state.trials[state.trials.length - 1];
  return last ? temperatureOf(last.objectiveValue) : null;
}

/** The try that opened the safe, or null while it is still shut. */
export function crackedOn(state: HeistState): number | null {
  return (
    state.trials.find((trial) => trial.objectiveValue >= CRACK_THRESHOLD)
      ?.index ?? null
  );
}

/**
 * NOT exported as a selector.
 *
 * `suggestNext` fits a GP and scans 512 candidates, and it returns a fresh object.
 * Running that inside a zustand selector would refit on every store read AND break
 * React's cached-snapshot contract. Components call it inside a `useMemo` keyed on
 * the trial list.
 */
export function suggestionFor(trials: Trial[]): Suggestion {
  return suggestNext(trials);
}

const centre = (): UnitPoint => new Array<number>(DIAL_COUNT).fill(0.5);

/**
 * A dial position the store can hold. Non-finite input falls back to the centre
 * rather than reaching `objectiveAt` — a NaN dial scores NaN, and a NaN try used
 * to become an incumbent no genuine crack could displace.
 */
const toDial = (unit: number | undefined): number =>
  typeof unit === "number" && Number.isFinite(unit) ? clamp(unit, 0, 1) : 0.5;

function freshRun() {
  return {
    dials: centre(),
    trials: [] as Trial[],
    budget: BUDGET,
    strategy: "manual" as Strategy | "manual",
    revealed: false,
    phase: "cracking" as Phase,
    evaluation: null,
    failure: null,
  };
}

/** Fold new trials in and work out where that leaves the run. */
function settle(state: HeistState, incoming: Trial[]) {
  const trials = [...state.trials, ...incoming];
  const finished = trials.length >= state.budget;
  const evaluation = evaluateRun({
    trials,
    budget: state.budget,
    finished,
  });

  const cracked = evaluation.outcome === "cracked";
  const phase: Phase = cracked ? "cracked" : finished ? "busted" : "cracking";

  return {
    trials,
    evaluation,
    failure: evaluation.failure,
    phase,
    // The reveal is the payoff for a finished run, either way.
    revealed: state.revealed || cracked || finished,
    cracked,
    finished,
  };
}

/** Tell the progression service about a crack, crediting the lane it came from. */
function recordCrack(score: number, source: Lane) {
  useProgression.getState().recordResult({
    slug: SLUG,
    score,
    lane: source,
    completed: true,
    codeLaneCleared: source === "code",
  });
}

export const useHeistStore = create<HeistState>((set, get) => ({
  ...freshRun(),
  whyCard: whyCardFor({ kind: "briefing" }),
  lane: "visual" as Lane,
  attempt: 0,

  setLane: (lane) => set({ lane }),

  setDial: (index, unit) => {
    const state = get();
    if (!Number.isInteger(index) || index < 0 || index >= DIAL_COUNT) return;
    if (!Number.isFinite(unit)) return;
    const dials = [...state.dials];
    dials[index] = clamp(unit, 0, 1);
    if (dials[index] === state.dials[index]) return;
    set({ dials });
  },

  setDials: (point) => {
    set({
      dials: Array.from({ length: DIAL_COUNT }, (_, index) =>
        toDial(point[index]),
      ),
    });
  },

  setStrategy: (strategy) => {
    const state = get();
    if (strategy === state.strategy) return;
    set({
      strategy,
      whyCard: whyCardFor({
        kind: "strategy-changed",
        strategy,
        budget: state.budget,
        triesLeft: triesLeft(state),
      }),
    });
  },

  tryCurrent: (source = "visual") => {
    const state = get();
    if (state.phase !== "cracking" || triesLeft(state) <= 0) return null;

    const trial: Trial = {
      params: [...state.dials],
      objectiveValue: objectiveAt(state.dials),
      index: state.trials.length + 1,
      source: state.strategy === "bayesian" ? "bayesian" : "manual",
    };

    const next = settle(state, [trial]);
    set({
      ...next,
      whyCard: whyCardFor({
        kind: "tried",
        trial,
        previousBest:
          bestTrial(state.trials)?.objectiveValue ?? Number.NEGATIVE_INFINITY,
        triesLeft: state.budget - next.trials.length,
        cracked: next.cracked,
        finished: next.finished,
        trials: next.trials,
        budget: state.budget,
      }),
    });

    // Only on the transition into "cracked": the phase guard above means a second
    // call cannot reach here, so a re-run snippet cannot record the crack twice.
    if (next.cracked) recordCrack(next.evaluation.score, source);

    return trial;
  },

  trySuggestion: (source = "visual") => {
    const state = get();
    if (state.phase !== "cracking" || triesLeft(state) <= 0) return null;
    const suggestion = suggestNext(state.trials);
    get().setDials(suggestion.point);
    return get().tryCurrent(source);
  },

  runToBudget: (strategy, source = "visual") => {
    const state = get();
    if (state.phase !== "cracking") return [];
    const remaining = triesLeft(state);
    if (remaining <= 0) return [];

    // Grid and random are laid out for the WHOLE budget and then truncated to
    // what is left. Re-planning a grid for the remaining tries would quietly
    // change its resolution mid-run and make the comparison meaningless.
    const planned: UnitPoint[] =
      strategy === "grid"
        ? gridPoints(state.budget).slice(state.trials.length)
        : strategy === "random"
          ? randomPoints(state.budget, randomSeedFor(state.attempt)).slice(
              state.trials.length,
            )
          : [];

    const incoming: Trial[] = [];
    const opens = (trial: Trial) => trial.objectiveValue >= CRACK_THRESHOLD;
    if (strategy === "bayesian") {
      // Sequential by nature: each suggestion depends on everything before it.
      let sofar = [...state.trials];
      for (let step = 0; step < remaining; step += 1) {
        const point = suggestNext(sofar).point;
        const trial: Trial = {
          params: point,
          objectiveValue: objectiveAt(point),
          index: sofar.length + 1,
          source: "bayesian",
        };
        sofar = [...sofar, trial];
        incoming.push(trial);
        if (opens(trial)) break;
      }
    } else {
      for (const [step, point] of planned.slice(0, remaining).entries()) {
        const trial: Trial = {
          params: point,
          objectiveValue: objectiveAt(point),
          index: state.trials.length + step + 1,
          source: strategy,
        };
        incoming.push(trial);
        // Stop at the crack. Spending the rest would record tries the score
        // already ignores, and the budget readout would disagree with it.
        if (opens(trial)) break;
      }
    }

    const next = settle(state, incoming);
    set({
      ...next,
      strategy,
      dials: incoming[incoming.length - 1]?.params ?? state.dials,
      whyCard: whyCardFor({
        kind: "strategy-run",
        strategy,
        trials: next.trials,
        added: incoming,
        budget: state.budget,
        cracked: next.cracked,
      }),
    });

    if (next.cracked) recordCrack(next.evaluation.score, source);

    return incoming;
  },

  reveal: () => set({ revealed: true }),

  reset: () =>
    set((state) => ({
      ...freshRun(),
      attempt: state.attempt + 1,
      whyCard: whyCardFor({ kind: "briefing" }),
    })),
}));

/** Where a real-world value sits on its dial, for the code lane's convenience. */
export function unitFor(dialIndex: number, value: number): number {
  const dial = DIALS[dialIndex];
  return dial ? toUnitValue(dial, value) : 0.5;
}

export { BUDGET, CRACK_THRESHOLD };
