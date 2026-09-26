"use client";

import { create } from "zustand";
import { useProgression } from "@/engine/progression";
import type { Lane, NamedFailure } from "@/engine/types";
import type { WhyCardContent } from "@/components";
import { seededRandom } from "@/lib/utils";
import {
  ACTIONS,
  COMPETENCE_RATE,
  DEFAULT_EPSILON,
  DEFAULT_REWARDS,
  EPISODE_BATCH,
  MAX_EPISODES,
  MAX_STEPS,
  REWARD_KNOBS,
  START,
  cellAt,
  emptyQTable,
  evaluatePolicy,
  evaluateRun,
  farmValue,
  greedyAction,
  positionOf,
  roundTripValue,
  runEpisodes,
  solveOptimal,
  stateOf,
  stepEnvironment,
  trainFresh,
  type Action,
  type Evaluation,
  type EpisodeRecord,
  type OptimalPolicy,
  type PolicyReport,
  type Position,
  type QTable,
  type RewardConfig,
} from "./ml";
import { whyCardFor } from "./why-cards";

export const SLUG = "agent-academy";

/** Base seed. Batches advance it by episode count so runs stay reproducible. */
export const TRAINING_SEED = 20260805;

/**
 * Enough episodes to be worth diagnosing.
 *
 * A single batch of 100 is not evidence of anything — at epsilon 0.3 the maze is
 * solved by episode 200 on most seeds, so naming a failure after 100 would be
 * crying wolf. Reward hacking is exempt: that verdict comes from value iteration
 * on the reward function itself, so it is just as true before training as after.
 */
export const DIAGNOSE_AFTER = 2 * EPISODE_BATCH;

export type Phase = "designing" | "graduated" | "failed";

/**
 * Agent Academy state.
 *
 * The spec's data model is:
 *
 *   Grid      { cells[][], rewards{} }
 *   Agent     { qTable{}, epsilon, policy }
 *   GameState { grid, agent, episode, cumulativeReward }
 *
 * `Grid.cells` is not in state because it never changes — it is the `GRID`
 * constant in ml.ts, and a per-store copy would be the same 28 cells again with
 * a chance of drifting. `rewards` does live here: it is the thing the player
 * writes, and it is the whole game.
 *
 * `Agent.policy` is not stored separately either. A policy is `argmax` over the
 * Q-table, so keeping both would be keeping the same information twice and
 * inviting them to disagree. `trail` holds the greedy walk the policy produces,
 * which is the part the grid actually needs to draw.
 *
 * `cumulativeReward` is not a running total: `history` holds per-episode rewards
 * because that is what the reward curve plots, and a sum of it is one `reduce`
 * away whenever anybody wants it.
 *
 * ── Why changing anything resets the agent ──────────────────────────────────
 * A Q-value is an estimate of future reward UNDER A PARTICULAR REWARD FUNCTION.
 * Change the rewards and every number in the table is now an answer to a question
 * nobody asked, so the table has to go. Epsilon resets it too, which is a design
 * choice rather than a requirement: it means every run is one clean experiment
 * with one variable changed, which is exactly the reasoning the game is trying to
 * teach. Mixing two epsilons inside one training history would make "epsilon 0
 * never solves this" unprovable from the player's own screen.
 */
export interface AcademyState {
  rewards: RewardConfig;
  epsilon: number;

  qTable: QTable;
  /**
   * How many times each (state, action) pair has been taken, same shape as the
   * table. The Q-value panel's "never tried" and the heatmap's untinted cells
   * read this rather than guessing from a value of 0 — see `TrainOptions.visits`.
   */
  visits: QTable;
  history: EpisodeRecord[];
  episodesUsed: number;

  /**
   * The best policy the current rewards admit, from value iteration.
   *
   * Recomputed on every reward change, not on demand: it costs microseconds, it
   * never depends on training, and having it always present is what lets the
   * visual lane show what a reward function ASKS for next to what an agent
   * actually learned.
   */
  optimal: OptimalPolicy;
  /** Greedy policy measured after the last batch. */
  report: PolicyReport;
  /** The greedy walk from the start, for the grid to draw. */
  trail: Position[];

  training: boolean;
  phase: Phase;
  evaluation: Evaluation | null;
  failure: NamedFailure | null;
  whyCard: WhyCardContent | null;
  lane: Lane;
  /** Q-value overlay on the grid. Off by default so the maze reads first. */
  showHeatmap: boolean;
  /** Optimal-policy arrows on the grid. */
  showOptimal: boolean;

  setLane: (lane: Lane) => void;
  setReward: (key: keyof RewardConfig, value: number) => void;
  setRewards: (rewards: Partial<RewardConfig>) => void;
  setEpsilon: (epsilon: number) => void;
  toggleHeatmap: () => void;
  toggleOptimal: () => void;
  /**
   * Run a batch of episodes and fold the result back in.
   *
   * `source` is the lane the click came from, so a graduation earns the code-lane
   * star only when the code lane actually trained it — not whenever that tab
   * happens to be open. Once the agent graduates the visual lane swaps Train for
   * "New academy", and the code lane's `train` refuses by name (see
   * `createCodeApi().train`), so re-running a snippet cannot re-record or
   * un-graduate a finished run.
   */
  train: (episodes?: number, source?: Lane) => void;
  /** Throw the agent away, keep the reward function and epsilon. */
  forget: () => void;
  /** Back to the starting reward function and epsilon. */
  reset: () => void;
}

// ── selectors: primitives and stable references only ─────────────────────

export function goalRate(state: AcademyState): number {
  return state.report.goalRate;
}

export function meanEpisodeReward(state: AcademyState): number {
  return state.report.meanReward;
}

export function episodesLeft(state: AcademyState): number {
  return Math.max(0, MAX_EPISODES - state.episodesUsed);
}

export function rewardsAreSatisfiable(state: AcademyState): boolean {
  return state.optimal.reachesGoal;
}

/**
 * The greedy walk from the start under the current table.
 *
 * Stops on a repeated (state, action) pair as well as on the step cap, because a
 * farming policy loops forever and the grid only needs to show the loop once.
 */
export function greedyTrail(qTable: QTable, rewards: RewardConfig): Position[] {
  const random = seededRandom(1);
  const trail: Position[] = [{ ...START }];
  const seen = new Set<string>();
  let position = { ...START };

  for (let step = 0; step < MAX_STEPS; step += 1) {
    const state = stateOf(position);
    const action = greedyAction(qTable, state, random);
    const key = `${state}:${action}`;
    if (seen.has(key)) break;
    seen.add(key);

    const result = stepEnvironment(position, action, rewards);
    position = result.next;
    trail.push({ ...position });
    if (result.done) break;
  }

  return trail;
}

/** The optimal policy's walk, for the "what you asked for" overlay. */
export function optimalTrail(
  optimal: OptimalPolicy,
  rewards: RewardConfig,
): Position[] {
  const trail: Position[] = [{ ...START }];
  const seen = new Set<number>([stateOf(START)]);
  let position = { ...START };

  for (let step = 0; step < MAX_STEPS; step += 1) {
    const result = stepEnvironment(
      position,
      optimal.policy[stateOf(position)]!,
      rewards,
    );
    position = result.next;
    trail.push({ ...position });
    if (result.done) break;
    if (seen.has(stateOf(position))) break;
    seen.add(stateOf(position));
  }

  return trail;
}

/** Per-cell best value and its action, for the heatmap. Stable given a table. */
export interface CellValue {
  state: number;
  row: number;
  col: number;
  best: number;
  action: Action;
  visited: boolean;
}

export function cellValues(qTable: QTable, visits?: QTable): CellValue[] {
  return qTable.map((row, state) => {
    let best = -Infinity;
    let action: Action = 0;
    for (const candidate of ACTIONS) {
      const value = row[candidate] ?? 0;
      if (value > best) {
        best = value;
        action = candidate;
      }
    }
    const position = positionOf(state);
    return {
      state,
      row: position.row,
      col: position.col,
      best,
      action,
      // Tracked, not inferred: with a step cost of 0 a tried action can learn a
      // value of exactly 0. The fallback keeps the old reading for callers that
      // only have a table.
      visited: visits
        ? (visits[state] ?? []).some((count) => count > 0)
        : row.some((value) => value !== 0),
    };
  });
}

/** Anything that is not literally "code" — a click event, say — is the visual lane. */
const laneOf = (source: unknown): Lane => (source === "code" ? "code" : "visual");

function freshAgent(rewards: RewardConfig) {
  const qTable = emptyQTable();
  return {
    qTable,
    visits: emptyQTable(),
    history: [] as EpisodeRecord[],
    episodesUsed: 0,
    report: evaluatePolicy(qTable, rewards),
    trail: greedyTrail(qTable, rewards),
    evaluation: null,
    failure: null,
    phase: "designing" as Phase,
  };
}

export const useAcademyStore = create<AcademyState>((set, get) => ({
  rewards: { ...DEFAULT_REWARDS },
  epsilon: DEFAULT_EPSILON,
  optimal: solveOptimal(DEFAULT_REWARDS),
  ...freshAgent(DEFAULT_REWARDS),
  whyCard: whyCardFor({ kind: "briefing" }),
  lane: "visual" as Lane,
  showHeatmap: false,
  showOptimal: false,
  training: false,

  setLane: (lane) => set({ lane }),

  toggleHeatmap: () => set({ showHeatmap: !get().showHeatmap }),
  toggleOptimal: () => set({ showOptimal: !get().showOptimal }),

  setReward: (key, value) => {
    get().setRewards({ [key]: value });
  },

  setRewards: (partial) => {
    const state = get();
    if (state.training) return;

    const rewards = { ...state.rewards, ...partial };
    const optimal = solveOptimal(rewards);
    const changedKeys = (Object.keys(partial) as Array<keyof RewardConfig>).filter(
      (key) => partial[key] !== state.rewards[key],
    );
    if (changedKeys.length === 0) return;

    set({
      rewards,
      optimal,
      ...freshAgent(rewards),
      whyCard: whyCardFor({
        kind: "reward-changed",
        keys: changedKeys,
        rewards,
        optimal,
        hadEpisodes: state.episodesUsed,
      }),
    });
  },

  setEpsilon: (epsilon) => {
    const state = get();
    if (state.training || epsilon === state.epsilon) return;

    set({
      epsilon,
      ...freshAgent(state.rewards),
      whyCard: whyCardFor({
        kind: "epsilon-changed",
        epsilon,
        previous: state.epsilon,
        hadEpisodes: state.episodesUsed,
      }),
    });
  },

  train: (episodes = EPISODE_BATCH, source) => {
    const state = get();
    if (state.training) return;

    // A NaN or fractional count from the code lane used to reach the counter and
    // leave it at NaN for good; the api rejects those by name, and this is the
    // floor under it.
    const requested = Math.floor(episodes);
    if (!Number.isFinite(requested) || requested < 1) return;
    const budget = Math.min(requested, MAX_EPISODES - state.episodesUsed);
    if (budget <= 0) return;
    const lane = laneOf(source);

    set({ training: true });

    // The table is mutated in place by runEpisodes, which is how Q-learning
    // works; a fresh array is handed to the store so React sees a new reference.
    const qTable = state.qTable.map((row) => [...row]);
    const visits = state.visits.map((row) => [...row]);
    const batch = runEpisodes({
      qTable,
      visits,
      rewards: state.rewards,
      epsilon: state.epsilon,
      episodes: budget,
      // Seeded from the episode count so continuing a run never replays the same
      // random stream, and so the whole run is reproducible from the start.
      random: seededRandom(TRAINING_SEED + state.episodesUsed),
    });

    const episodesUsed = state.episodesUsed + budget;
    const offset = state.episodesUsed;
    const history = [
      ...state.history,
      ...batch.map((record) => ({ ...record, index: record.index + offset })),
    ];

    const evaluation = evaluateRun({
      qTable,
      rewards: state.rewards,
      epsilon: state.epsilon,
      episodesUsed,
    });

    // Reward hacking is provable from the reward function alone, so it is named
    // immediately. Everything else waits until there is enough training to be
    // fair about it.
    const showFailure =
      evaluation.outcome === "reward-hacking" || episodesUsed >= DIAGNOSE_AFTER;
    const won = evaluation.outcome === "competent";

    set({
      qTable,
      visits,
      history,
      episodesUsed,
      report: evaluation.report,
      trail: greedyTrail(qTable, state.rewards),
      training: false,
      evaluation,
      failure: showFailure ? evaluation.failure : null,
      phase: won ? "graduated" : showFailure && evaluation.failure ? "failed" : "designing",
      whyCard: whyCardFor({
        kind: "trained",
        evaluation,
        batch,
        episodesUsed,
        epsilon: state.epsilon,
        rewards: state.rewards,
        // The why-card waits for the same evidence the failure strip does, or it
        // would name a verdict the strip is deliberately holding back.
        diagnosed: showFailure,
      }),
    });

    if (won) {
      useProgression.getState().recordResult({
        slug: SLUG,
        score: evaluation.score,
        lane,
        completed: true,
        codeLaneCleared: lane === "code",
      });
    }
  },

  forget: () => {
    const state = get();
    if (state.training) return;
    set({
      ...freshAgent(state.rewards),
      whyCard: whyCardFor({ kind: "forgot" }),
    });
  },

  reset: () => {
    set({
      rewards: { ...DEFAULT_REWARDS },
      epsilon: DEFAULT_EPSILON,
      optimal: solveOptimal(DEFAULT_REWARDS),
      ...freshAgent(DEFAULT_REWARDS),
      whyCard: whyCardFor({ kind: "briefing" }),
    });
  },
}));

/**
 * What the code lane can do, and nothing more.
 *
 * Every verb here is a store action the visual lane also calls, so a snippet and
 * a slider drag are the same operation. `report()` returns a plain snapshot
 * rather than store state, so a Python dict conversion cannot reach in and mutate
 * anything.
 */
export interface AcademyCodeApi {
  setReward: (key: string, value: number) => void;
  setEpsilon: (epsilon: number) => void;
  train: (episodes?: number) => void;
  forget: () => void;
  rewards: () => RewardConfig;
  epsilon: () => number;
  episodes: () => number;
  report: () => PolicyReport & { episodesUsed: number; outcome: string };
  /** Value iteration on the CURRENT rewards — what they ask for, before training. */
  optimal: () => {
    reachesGoal: boolean;
    behaviour: string;
    valueAtStart: number;
    valueAtPellet: number;
    episodeReward: number;
    steps: number;
    pelletsEaten: number;
  };
  /** Per-cell best Q-value, row-major, for printing the table. */
  values: () => number[];
  /**
   * Train a throwaway agent and report on it, WITHOUT touching the player's
   * agent or spending their episode budget.
   *
   * This is the verb that makes the code lane worth having. "ε 0 never solves
   * this" is a claim about a distribution, and the visual lane can only ever show
   * one run at a time — so the experiment that actually settles it has to be
   * cheap and repeatable. Charging the player's budget for it would make the
   * question unaskable.
   */
  simulate: (options?: {
    rewards?: Partial<RewardConfig>;
    epsilon?: number;
    episodes?: number;
    seed?: number;
  }) => PolicyReport & { solved: boolean };
  /** Value iteration on any reward function. Free, and needs no training. */
  solve: (rewards?: Partial<RewardConfig>) => {
    reachesGoal: boolean;
    behaviour: string;
    valueAtStart: number;
    valueAtPellet: number;
    episodeReward: number;
    steps: number;
    pelletsEaten: number;
    farmValue: number;
    roundTripValue: number;
  };
}

const REWARD_KEYS: ReadonlyArray<keyof RewardConfig> = [
  "goal",
  "trap",
  "step",
  "pellet",
  "bump",
];

function rewardKey(key: unknown): keyof RewardConfig {
  const match = REWARD_KEYS.find((candidate) => candidate === key);
  if (match === undefined) {
    throw new Error(
      `Unknown reward "${String(key)}". Try one of: ${REWARD_KEYS.join(", ")}.`,
    );
  }
  return match;
}

/**
 * A partial reward function from the code lane, checked before it is used.
 *
 * `simulate({ rewards: { cheese: 3 } })` used to spread the typo straight in, so
 * the experiment silently ran on the player's current rewards and "proved"
 * something about a reward function nobody wrote. Throwaway experiments may go
 * past the sliders' ranges — that is what they are for — but every key has to be
 * real and every value a finite number.
 */
function checkedRewards(partial: unknown): Partial<RewardConfig> {
  if (partial === undefined) return {};
  if (typeof partial !== "object" || partial === null || Array.isArray(partial)) {
    throw new Error("rewards must be an object, e.g. { pellet: 3 }.");
  }
  const checked: Partial<RewardConfig> = {};
  for (const [key, value] of Object.entries(partial)) {
    const match = rewardKey(key);
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error(`Reward "${key}" must be a finite number.`);
    }
    checked[match] = value;
  }
  return checked;
}

export function createCodeApi(): AcademyCodeApi {
  return {
    setReward: (key, value) => {
      const match = rewardKey(key);
      if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new Error(`Reward "${key}" must be a finite number.`);
      }
      // The player's own rewards are what the sliders show, so they stay inside
      // the sliders' ranges; a slider pinned at its end while the store holds
      // something else would be one store telling two stories.
      const knob = REWARD_KNOBS.find((candidate) => candidate.key === match);
      if (knob && (value < knob.min || value > knob.max)) {
        throw new Error(
          `Reward "${key}" must be between ${knob.min} and ${knob.max}. api.simulate and api.solve take any value without adopting it.`,
        );
      }
      useAcademyStore.getState().setReward(match, value);
    },
    setEpsilon: (epsilon) => {
      if (
        typeof epsilon !== "number" ||
        !Number.isFinite(epsilon) ||
        epsilon < 0 ||
        epsilon > 1
      ) {
        throw new Error("Epsilon must be a number between 0 and 1.");
      }
      useAcademyStore.getState().setEpsilon(epsilon);
    },
    train: (episodes) => {
      if (
        episodes !== undefined &&
        (typeof episodes !== "number" || !Number.isFinite(episodes) || episodes < 1)
      ) {
        throw new Error(
          `Episodes must be a number of at least 1 (got ${String(episodes)}).`,
        );
      }
      const state = useAcademyStore.getState();
      if (state.phase === "graduated") {
        throw new Error(
          "This agent has already graduated. api.forget() starts a fresh one on the same rules.",
        );
      }
      if (state.episodesUsed >= MAX_EPISODES) {
        throw new Error(
          `All ${MAX_EPISODES} episodes are spent. api.forget() starts a fresh agent.`,
        );
      }
      state.train(episodes === undefined ? undefined : Math.floor(episodes), "code");
    },
    forget: () => useAcademyStore.getState().forget(),
    rewards: () => ({ ...useAcademyStore.getState().rewards }),
    epsilon: () => useAcademyStore.getState().epsilon,
    episodes: () => useAcademyStore.getState().episodesUsed,
    report: () => {
      const state = useAcademyStore.getState();
      return {
        ...state.report,
        episodesUsed: state.episodesUsed,
        outcome: state.evaluation?.outcome ?? "untrained",
      };
    },
    optimal: () => {
      const { optimal } = useAcademyStore.getState();
      return {
        reachesGoal: optimal.reachesGoal,
        behaviour: optimal.behaviour,
        valueAtStart: optimal.valueAtStart,
        valueAtPellet: optimal.valueAtPellet,
        episodeReward: optimal.episodeReward,
        steps: optimal.steps,
        pelletsEaten: optimal.pelletsEaten,
      };
    },
    values: () =>
      useAcademyStore
        .getState()
        .qTable.map((row, state) =>
          cellAt(positionOf(state)) === "wall" ? Number.NaN : Math.max(...row),
        ),

    simulate: ({ rewards, epsilon, episodes = 300, seed = 1 } = {}) => {
      const state = useAcademyStore.getState();
      const merged = { ...state.rewards, ...checkedRewards(rewards) };
      const useEpsilon = epsilon ?? state.epsilon;
      if (
        typeof useEpsilon !== "number" ||
        !Number.isFinite(useEpsilon) ||
        useEpsilon < 0 ||
        useEpsilon > 1
      ) {
        throw new Error("Epsilon must be a number between 0 and 1.");
      }
      if (
        typeof episodes !== "number" ||
        !Number.isFinite(episodes) ||
        episodes < 1 ||
        episodes > 20000
      ) {
        throw new Error("Episodes must be between 1 and 20000.");
      }
      if (typeof seed !== "number" || !Number.isFinite(seed)) {
        throw new Error("seed must be a finite number.");
      }
      const report = trainFresh(merged, useEpsilon, Math.floor(episodes), seed)
        .report;
      return { ...report, solved: report.goalRate >= COMPETENCE_RATE };
    },

    solve: (rewards) => {
      const merged = {
        ...useAcademyStore.getState().rewards,
        ...checkedRewards(rewards),
      };
      const optimal = solveOptimal(merged);
      return {
        reachesGoal: optimal.reachesGoal,
        behaviour: optimal.behaviour,
        valueAtStart: optimal.valueAtStart,
        valueAtPellet: optimal.valueAtPellet,
        episodeReward: optimal.episodeReward,
        steps: optimal.steps,
        pelletsEaten: optimal.pelletsEaten,
        farmValue: farmValue(merged),
        roundTripValue: roundTripValue(merged),
      };
    },
  };
}
