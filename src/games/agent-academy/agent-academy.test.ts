import { beforeEach, describe, expect, it } from "vitest";
import {
  EMPTY_PROGRESSION,
  HIGH_SCORE_THRESHOLD,
  createMemoryAdapter,
  starsFor,
  useProgression,
} from "@/engine/progression";
import { seededRandom } from "@/lib/utils";
import {
  ACTIONS,
  COLS,
  COMPETENCE_RATE,
  DEFAULT_EPSILON,
  DEFAULT_REWARDS,
  DISCOUNT,
  EPISODE_BATCH,
  GOAL,
  GRID,
  LEARNING_RATE,
  LOW_EPSILON,
  MAX_EPISODES,
  MAX_STEPS,
  PELLET,
  REFERENCE_EPSILON,
  REWARD_KNOBS,
  ROWS,
  START,
  STATE_COUNT,
  TRAP,
  MATH_NOTES,
  cellAt,
  competentScore,
  discountAfter,
  emptyQTable,
  episodesForScore,
  evaluatePolicy,
  evaluateRun,
  farmValue,
  greedyAction,
  isTerminal,
  isWall,
  move,
  positionOf,
  roundTripValue,
  runEpisodes,
  solveOptimal,
  stateOf,
  stepEnvironment,
  stepsFromStart,
  straightExitValue,
  trainFresh,
  type Action,
  type Outcome,
  type RewardConfig,
} from "./ml";
import {
  DIAGNOSE_AFTER,
  SLUG,
  cellValues,
  createCodeApi,
  greedyTrail,
  optimalTrail,
  useAcademyStore,
} from "./store";
import { whyCardFor } from "./why-cards";

/**
 * Every constant in this game was chosen by measurement, not by taste. A
 * throwaway probe swept six layouts, three learning rates, four step caps, three
 * slip probabilities and seven epsilons over twelve seeds each; the probe is gone
 * and what it found is asserted here, so the tuning cannot silently rot.
 *
 * The assertions worth reading are the ones about what does NOT work: epsilon 0
 * failing at every episode budget, and a raised cheese reward that no amount of
 * exploration can rescue. Those two facts are the game.
 */

const SEEDS = [7, 11, 23, 41, 97, 4242, 606, 1234, 5, 88, 300, 512] as const;

const withRewards = (partial: Partial<RewardConfig>): RewardConfig => ({
  ...DEFAULT_REWARDS,
  ...partial,
});

/** How many of the twelve seeds reach competence. */
function solvedCount(
  rewards: RewardConfig,
  epsilon: number,
  episodes: number,
): number {
  return SEEDS.filter(
    (seed) =>
      trainFresh(rewards, epsilon, episodes, seed).report.goalRate >=
      COMPETENCE_RATE,
  ).length;
}

function meanOf(
  rewards: RewardConfig,
  epsilon: number,
  episodes: number,
  pick: (report: ReturnType<typeof trainFresh>["report"]) => number,
): number {
  const values = SEEDS.map((seed) =>
    pick(trainFresh(rewards, epsilon, episodes, seed).report),
  );
  return values.reduce((total, value) => total + value, 0) / values.length;
}

describe("the maze", () => {
  it("is the 4x7 grid the comments describe", () => {
    expect(ROWS).toBe(4);
    expect(COLS).toBe(7);
    expect(STATE_COUNT).toBe(28);
    expect(GRID.flat().filter((cell) => cell === "wall")).toHaveLength(0);
  });

  it("puts the cheese near and the exit far", () => {
    // These three distances ARE the difficulty curve. The cheese has to be found
    // immediately and the exit has to take real searching.
    expect(stepsFromStart(PELLET)).toBe(2);
    expect(stepsFromStart(TRAP)).toBe(2);
    expect(stepsFromStart(GOAL)).toBe(9);
  });

  it("holds exactly one of each special cell", () => {
    const counts = GRID.flat().reduce<Record<string, number>>((tally, cell) => {
      tally[cell] = (tally[cell] ?? 0) + 1;
      return tally;
    }, {});
    expect(counts.start).toBe(1);
    expect(counts.goal).toBe(1);
    expect(counts.trap).toBe(1);
    expect(counts.pellet).toBe(1);
  });

  it("gives every episode room to find a nine-step exit", () => {
    expect(MAX_STEPS).toBeGreaterThan(stepsFromStart(GOAL) * 4);
  });

  it("round-trips states and positions", () => {
    for (let state = 0; state < STATE_COUNT; state += 1) {
      expect(stateOf(positionOf(state))).toBe(state);
    }
  });

  it("treats off-grid as wall", () => {
    expect(isWall({ row: -1, col: 0 })).toBe(true);
    expect(isWall({ row: 0, col: COLS })).toBe(true);
    expect(isWall(START)).toBe(false);
  });

  it("ends the episode on the exit and the pit, nowhere else", () => {
    expect(isTerminal(stateOf(GOAL))).toBe(true);
    expect(isTerminal(stateOf(TRAP))).toBe(true);
    expect(isTerminal(stateOf(START))).toBe(false);
    expect(isTerminal(stateOf(PELLET))).toBe(false);
  });
});

describe("the environment", () => {
  it("moves in the four directions the labels claim", () => {
    const from = { row: 2, col: 3 };
    expect(move(from, 0)).toEqual({ row: 1, col: 3 });
    expect(move(from, 1)).toEqual({ row: 2, col: 4 });
    expect(move(from, 2)).toEqual({ row: 3, col: 3 });
    expect(move(from, 3)).toEqual({ row: 2, col: 2 });
  });

  it("charges the bump and stays put, without also charging a step", () => {
    const result = stepEnvironment(START, 0, DEFAULT_REWARDS);
    expect(result.outcome).toBe("bumped");
    expect(result.next).toEqual(START);
    expect(result.reward).toBe(DEFAULT_REWARDS.bump);
    expect(result.done).toBe(false);
  });

  it("pays the step alongside the exit and the pit", () => {
    const beforeGoal = { row: GOAL.row, col: GOAL.col - 1 };
    const atGoal = stepEnvironment(beforeGoal, 1, DEFAULT_REWARDS);
    expect(atGoal.outcome).toBe("goal");
    expect(atGoal.done).toBe(true);
    expect(atGoal.reward).toBe(DEFAULT_REWARDS.step + DEFAULT_REWARDS.goal);

    const beforeTrap = { row: TRAP.row - 1, col: TRAP.col };
    const atTrap = stepEnvironment(beforeTrap, 2, DEFAULT_REWARDS);
    expect(atTrap.outcome).toBe("trap");
    expect(atTrap.done).toBe(true);
    expect(atTrap.reward).toBe(DEFAULT_REWARDS.step + DEFAULT_REWARDS.trap);
  });

  it("pays the cheese every single time, which is what makes farming possible", () => {
    const beside = { row: PELLET.row, col: PELLET.col - 1 };
    for (let visit = 0; visit < 5; visit += 1) {
      const result = stepEnvironment(beside, 1, DEFAULT_REWARDS);
      expect(result.outcome).toBe("moved");
      expect(result.reward).toBe(
        DEFAULT_REWARDS.step + DEFAULT_REWARDS.pellet,
      );
    }
  });

  it("is deterministic — no slip, on purpose", () => {
    // Slip was measured and rejected: at slip 0.05 an epsilon-ZERO agent solved
    // the maze on 10 of 10 seeds, because wandering into new states is itself
    // exploration. The exploration lesson is worth more than a risky trap.
    const first = stepEnvironment(START, 1, DEFAULT_REWARDS);
    for (let repeat = 0; repeat < 50; repeat += 1) {
      expect(stepEnvironment(START, 1, DEFAULT_REWARDS)).toEqual(first);
    }
  });
});

describe("the starting reward function", () => {
  it("makes a round trip to the cheese exactly break even", () => {
    // The single most load-bearing number in the game. Zero is still better than
    // every alternative on a board where moving costs you, so a greedy agent
    // stalls there — and one notch up the slider makes it a profit.
    expect(roundTripValue(DEFAULT_REWARDS)).toBeCloseTo(0, 10);
  });

  it("still asks for the exit", () => {
    const optimal = solveOptimal(DEFAULT_REWARDS);
    expect(optimal.reachesGoal).toBe(true);
    expect(optimal.behaviour).toBe("exit");
    expect(optimal.steps).toBe(9);
    expect(optimal.episodeReward).toBeCloseTo(16.5, 6);
  });

  it("starts the player below the exploration they need", () => {
    expect(DEFAULT_EPSILON).toBeLessThanOrEqual(LOW_EPSILON);
    expect(solvedCount(DEFAULT_REWARDS, DEFAULT_EPSILON, 300)).toBe(0);
  });

  it("exposes a knob for every reward, with the default inside its range", () => {
    const keys = REWARD_KNOBS.map((knob) => knob.key).sort();
    expect(keys).toEqual(
      (Object.keys(DEFAULT_REWARDS) as Array<keyof RewardConfig>).sort(),
    );
    for (const knob of REWARD_KNOBS) {
      const value = DEFAULT_REWARDS[knob.key];
      expect(value).toBeGreaterThanOrEqual(knob.min);
      expect(value).toBeLessThanOrEqual(knob.max);
    }
  });
});

describe("Q-learning", () => {
  it("uses the learning rate and discount the math drawer prints", () => {
    expect(LEARNING_RATE).toBe(0.4);
    expect(DISCOUNT).toBe(0.95);
  });

  it("starts from a table of zeros", () => {
    const qTable = emptyQTable();
    expect(qTable).toHaveLength(STATE_COUNT);
    expect(qTable.every((row) => row.length === ACTIONS.length)).toBe(true);
    expect(qTable.flat().every((value) => value === 0)).toBe(true);
  });

  it("applies the plain update rule, and drops the bootstrap on terminals", () => {
    const qTable = emptyQTable();
    const beforeGoal = { row: GOAL.row, col: GOAL.col - 1 };
    const state = stateOf(beforeGoal);

    // Seed the goal cell with a value that MUST be ignored: the episode ends
    // there, so there is no next state to look ahead to.
    qTable[stateOf(GOAL)] = [999, 999, 999, 999];

    runEpisodes({
      qTable,
      rewards: DEFAULT_REWARDS,
      epsilon: 0,
      episodes: 0,
      random: seededRandom(1),
    });

    // One manual step, to check the arithmetic exactly.
    const result = stepEnvironment(beforeGoal, 1, DEFAULT_REWARDS);
    const expected = 0 + LEARNING_RATE * (result.reward + 0 - 0);
    const table = emptyQTable();
    table[stateOf(GOAL)] = [999, 999, 999, 999];
    const bootstrap = result.done
      ? 0
      : DISCOUNT * Math.max(...table[stateOf(result.next)]!);
    expect(bootstrap).toBe(0);
    table[state]![1] = 0 + LEARNING_RATE * (result.reward + bootstrap - 0);
    expect(table[state]![1]).toBeCloseTo(expected, 10);
    expect(table[state]![1]).toBeCloseTo(LEARNING_RATE * 19.5, 10);
  });

  it("breaks ties at random so a greedy agent gets a fair first move", () => {
    // With a zero table every action ties. Deterministic tie-breaking would make
    // an epsilon-zero agent walk into a wall forever, which would be "stuck" for a
    // reason that has nothing to do with exploration.
    const qTable = emptyQTable();
    const random = seededRandom(3);
    const seen = new Set<Action>();
    for (let draw = 0; draw < 200; draw += 1) {
      seen.add(greedyAction(qTable, stateOf(START), random));
    }
    expect(seen.size).toBe(ACTIONS.length);
  });

  it("prefers a strictly better action without any randomness", () => {
    const qTable = emptyQTable();
    qTable[stateOf(START)] = [0, 5, 0, 0];
    for (let draw = 0; draw < 20; draw += 1) {
      expect(greedyAction(qTable, stateOf(START), seededRandom(draw))).toBe(1);
    }
  });

  it("records one history entry per episode, capped at the step limit", () => {
    const qTable = emptyQTable();
    const history = runEpisodes({
      qTable,
      rewards: DEFAULT_REWARDS,
      epsilon: 0.3,
      episodes: 12,
      random: seededRandom(5),
    });
    expect(history).toHaveLength(12);
    expect(history.map((record) => record.index)).toEqual([
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    ]);
    for (const record of history) {
      expect(record.steps).toBeGreaterThan(0);
      expect(record.steps).toBeLessThanOrEqual(MAX_STEPS);
      expect(["goal", "trap", "timeout"]).toContain(record.ending);
    }
  });

  it("learns nothing when asked not to", () => {
    const qTable = emptyQTable();
    runEpisodes({
      qTable,
      rewards: DEFAULT_REWARDS,
      epsilon: 0.5,
      episodes: 30,
      random: seededRandom(9),
      learn: false,
    });
    expect(qTable.flat().every((value) => value === 0)).toBe(true);
  });

  it("is reproducible from a seed", () => {
    const a = trainFresh(DEFAULT_REWARDS, 0.3, 200, 77);
    const b = trainFresh(DEFAULT_REWARDS, 0.3, 200, 77);
    expect(a.qTable).toEqual(b.qTable);
    expect(a.report).toEqual(b.report);
  });
});

describe("exploration is what the game is about", () => {
  it("never solves the maze at epsilon 0, at any episode budget", () => {
    // The load-bearing negative result. If this ever passes, the "No exploration"
    // failure is a lie and the epsilon slider is decoration.
    for (const episodes of [100, 300, 600, MAX_EPISODES]) {
      expect(
        solvedCount(DEFAULT_REWARDS, 0, episodes),
        `epsilon 0 solved something at ${episodes} episodes`,
      ).toBe(0);
    }
  });

  it("does not solve it at epsilon 0.05 within a normal budget either", () => {
    expect(solvedCount(DEFAULT_REWARDS, 0.05, 300)).toBe(0);
    expect(solvedCount(DEFAULT_REWARDS, 0.05, 600)).toBeLessThanOrEqual(1);
  });

  it("solves it on every seed from epsilon 0.2 up", () => {
    for (const epsilon of [0.2, 0.3, 0.4, 0.5]) {
      expect(
        solvedCount(DEFAULT_REWARDS, epsilon, 300),
        `epsilon ${epsilon} failed a seed`,
      ).toBe(SEEDS.length);
    }
  });

  it("gets there at epsilon 0.1, but needs three times the episodes", () => {
    // The subtler half of the lesson: too little exploration is not always fatal,
    // it is expensive. Scoring rewards fewer episodes, so this costs the player.
    expect(solvedCount(DEFAULT_REWARDS, 0.1, 300)).toBeLessThan(SEEDS.length / 2);
    expect(solvedCount(DEFAULT_REWARDS, 0.1, 900)).toBe(SEEDS.length);
  });

  it("shows the stalled agent pacing by the cheese rather than losing ground", () => {
    const pellets = meanOf(
      DEFAULT_REWARDS,
      0,
      300,
      (report) => report.meanPellets,
    );
    const reward = meanOf(DEFAULT_REWARDS, 0, 300, (report) => report.meanReward);
    const timeouts = meanOf(
      DEFAULT_REWARDS,
      0,
      300,
      (report) => report.timeoutRate,
    );

    // 40 bites in 80 steps is the signature of an agent oscillating on and off
    // the cheese, and the reward is exactly 0 because a round trip breaks even.
    expect(pellets).toBeCloseTo(MAX_STEPS / 2, 1);
    expect(reward).toBeCloseTo(0, 6);
    expect(timeouts).toBe(1);
  });

  it("reaches the exit in nine steps for +16.5 once it has learned", () => {
    const report = trainFresh(DEFAULT_REWARDS, 0.3, 300, 7).report;
    expect(report.goalRate).toBe(1);
    expect(report.meanSteps).toBe(9);
    // 20 for the exit, +1 for the cheese it passes on the way, -0.5 x 9 moves.
    expect(report.meanReward).toBeCloseTo(16.5, 6);
    expect(report.meanPellets).toBe(1);
  });
});

describe("value iteration: what a reward function actually asks for", () => {
  it("agrees with the closed-form farming value at the cheese", () => {
    // Standing on the cheese, the loop is step off then step back on. Value
    // iteration has to land on the same geometric series, or one of them is wrong.
    const rewards = withRewards({ pellet: 3 });
    const optimal = solveOptimal(rewards);
    expect(optimal.behaviour).toBe("farm");
    expect(optimal.valueAtPellet).toBeCloseTo(farmValue(rewards), 6);
  });

  it("bounds the optimal value by the discounted best single reward", () => {
    const optimal = solveOptimal(DEFAULT_REWARDS);
    expect(optimal.valueAtStart).toBeLessThan(DEFAULT_REWARDS.goal);
    expect(optimal.valueAtStart).toBeGreaterThan(0);
  });

  it("gives terminal states no value, because the episode is over", () => {
    const optimal = solveOptimal(DEFAULT_REWARDS);
    expect(optimal.values[stateOf(GOAL)]).toBe(0);
    expect(optimal.values[stateOf(TRAP)]).toBe(0);
  });

  it("stops asking for the exit once the cheese pays a big enough profit", () => {
    expect(solveOptimal(withRewards({ pellet: 1 })).reachesGoal).toBe(true);
    expect(solveOptimal(withRewards({ pellet: 2 })).reachesGoal).toBe(true);
    // Between 2 and 2.5 the farming loop overtakes the exit.
    expect(solveOptimal(withRewards({ pellet: 2.5 })).reachesGoal).toBe(false);
    expect(solveOptimal(withRewards({ pellet: 3 })).reachesGoal).toBe(false);
    expect(solveOptimal(withRewards({ pellet: 6 })).reachesGoal).toBe(false);
  });

  it("stops asking for the exit when moving pays", () => {
    expect(solveOptimal(withRewards({ step: 0 })).reachesGoal).toBe(true);
    expect(solveOptimal(withRewards({ step: 0.5 })).reachesGoal).toBe(false);
    expect(solveOptimal(withRewards({ step: 0.5 })).behaviour).toBe("farm");
  });

  it("stops asking for the exit when the exit pays nothing", () => {
    const optimal = solveOptimal(withRewards({ goal: 0 }));
    expect(optimal.reachesGoal).toBe(false);
  });

  it("asks for nothing in particular when nothing is paid", () => {
    const optimal = solveOptimal({
      goal: 0,
      trap: 0,
      step: 0,
      pellet: 0,
      bump: 0,
    });
    expect(optimal.reachesGoal).toBe(false);
    expect(optimal.behaviour).toBe("wander");
    expect(optimal.valueAtStart).toBe(0);
  });

  it("still asks for the exit even at the steepest step cost the slider allows", () => {
    // Worth pinning: this is why the pit-suicide case is NOT labelled reward
    // hacking. The exit remains optimal; it is the search that fails.
    const optimal = solveOptimal(withRewards({ step: -2, trap: 0 }));
    expect(optimal.reachesGoal).toBe(true);
    expect(optimal.valueAtStart).toBeGreaterThan(
      -2 * (1 + DISCOUNT) + 0, // beats walking into the pit two steps away
    );
  });
});

describe("the diagnosis", () => {
  const diagnose = (
    rewards: RewardConfig,
    epsilon: number,
    episodes: number,
    seed = 7,
  ) => {
    const { qTable } = trainFresh(rewards, epsilon, episodes, seed);
    return evaluateRun({ qTable, rewards, epsilon, episodesUsed: episodes });
  };

  const outcomesFor = (
    rewards: RewardConfig,
    epsilon: number,
    episodes: number,
  ): Outcome[] =>
    SEEDS.map((seed) => diagnose(rewards, epsilon, episodes, seed).outcome);

  it("says untrained before anything has run", () => {
    const evaluation = evaluateRun({
      qTable: emptyQTable(),
      rewards: DEFAULT_REWARDS,
      epsilon: 0.3,
      episodesUsed: 0,
    });
    expect(evaluation.outcome).toBe("untrained");
    expect(evaluation.score).toBe(0);
    expect(evaluation.failure).toBeNull();
  });

  it("passes a competent agent", () => {
    expect(outcomesFor(DEFAULT_REWARDS, 0.3, 300).every((o) => o === "competent")).toBe(
      true,
    );
  });

  it("names reward hacking, and proves it rather than inferring it", () => {
    const evaluation = diagnose(withRewards({ pellet: 3 }), 0.3, 400);
    expect(evaluation.outcome).toBe("reward-hacking");
    expect(evaluation.failure?.name).toBe("Reward hacking");
    expect(evaluation.optimal.reachesGoal).toBe(false);
    // The verdict must not need a training run to justify itself.
    expect(evaluation.explorer).toBeNull();
    expect(evaluation.patient).toBeNull();
    expect(evaluation.failure?.detail).toContain("solved exactly");
  });

  it("refuses to award a win for hacked rewards, even on a lucky seed", () => {
    // Measured: with step +0.5, one seed in twelve stumbles into a goal-reaching
    // policy anyway. Checking hacking first is what stops that counting.
    const rewards = withRewards({ step: 0.5 });
    expect(solveOptimal(rewards).reachesGoal).toBe(false);
    const outcomes = outcomesFor(rewards, 0.3, 400);
    expect(outcomes.every((outcome) => outcome === "reward-hacking")).toBe(true);
    expect(outcomes).not.toContain("competent");
  });

  it("names no exploration when more exploring on the same budget would have worked", () => {
    for (const epsilon of [0, 0.05, 0.1]) {
      const evaluation = diagnose(DEFAULT_REWARDS, epsilon, 300);
      expect(evaluation.outcome, `epsilon ${epsilon}`).toBe("no-exploration");
      expect(evaluation.failure?.name).toBe("No exploration");
      expect(evaluation.explorer!.goalRate).toBeGreaterThanOrEqual(
        COMPETENCE_RATE,
      );
    }
  });

  it("names impatience, not epsilon, when the epsilon is already fine", () => {
    // The distinction that stops the game giving lazy advice: at epsilon 0.15 and
    // 100 episodes BOTH changes would work, and training on is the smaller one.
    const evaluation = diagnose(DEFAULT_REWARDS, 0.15, 100);
    expect(evaluation.outcome).toBe("under-trained");
    expect(evaluation.failure?.name).toBe("Not competent yet");
    expect(evaluation.patient!.goalRate).toBeGreaterThanOrEqual(COMPETENCE_RATE);
  });

  it("names the local optimum when the cheese pays but the exit is still optimal", () => {
    // cheese 2 is the interesting middle: value iteration still prefers the exit,
    // so this is NOT reward hacking, and no epsilon rescues it either.
    const rewards = withRewards({ pellet: 2 });
    expect(solveOptimal(rewards).reachesGoal).toBe(true);
    expect(roundTripValue(rewards)).toBeGreaterThan(0);

    const evaluation = diagnose(rewards, 0.3, 400);
    expect(evaluation.outcome).toBe("unlearnable-rewards");
    expect(evaluation.failure?.name).toBe("Local optimum");
    expect(evaluation.explorer!.goalRate).toBeLessThan(COMPETENCE_RATE);
  });

  it("names quitting when a steep step cost makes the pit the cheapest exit", () => {
    const rewards = withRewards({ trap: 0, step: -1 });
    const evaluation = diagnose(rewards, 0.3, 400);
    expect(evaluation.outcome).toBe("unlearnable-rewards");
    expect(evaluation.failure?.name).toBe("Quitting beats trying");
    expect(evaluation.report.trapRate).toBeGreaterThanOrEqual(0.4);
    // And it must NOT be called reward hacking: the exit is still worth more.
    expect(evaluation.optimal.reachesGoal).toBe(true);
  });

  it("carries the numbers that justify every verdict", () => {
    const cases: Array<[RewardConfig, number, number]> = [
      [withRewards({ pellet: 3 }), 0.3, 400],
      [DEFAULT_REWARDS, 0, 300],
      [DEFAULT_REWARDS, 0.15, 100],
      [withRewards({ pellet: 2 }), 0.3, 400],
      [withRewards({ trap: 0, step: -1 }), 0.3, 400],
    ];
    for (const [rewards, epsilon, episodes] of cases) {
      const evaluation = diagnose(rewards, epsilon, episodes);
      expect(evaluation.failure).not.toBeNull();
      expect(evaluation.failure!.name).not.toMatch(/game over/i);
      // A named failure with no numbers in it is a contract violation.
      expect(evaluation.failure!.detail).toMatch(/-?\d/);
      expect(evaluation.failure!.detail.length).toBeGreaterThan(120);
    }
  });

  it("scores competence higher the fewer episodes it took", () => {
    const quick = diagnose(DEFAULT_REWARDS, 0.3, 200);
    const slow = diagnose(DEFAULT_REWARDS, 0.3, 900);
    expect(quick.outcome).toBe("competent");
    expect(slow.outcome).toBe("competent");
    expect(quick.score).toBeGreaterThan(slow.score);
    expect(quick.score).toBeLessThanOrEqual(1);
    expect(slow.score).toBeGreaterThanOrEqual(0.6);
  });

  it("keeps a failing score below any passing score", () => {
    const failing = diagnose(DEFAULT_REWARDS, 0, 300);
    const passing = diagnose(DEFAULT_REWARDS, 0.3, MAX_EPISODES);
    expect(failing.score).toBeLessThan(passing.score);
    expect(failing.score).toBeLessThan(0.45);
  });

  it("uses the reference epsilon it advertises", () => {
    expect(REFERENCE_EPSILON).toBeGreaterThan(LOW_EPSILON);
    expect(solvedCount(DEFAULT_REWARDS, REFERENCE_EPSILON, EPISODE_BATCH)).toBe(
      SEEDS.length,
    );
  });
});

describe("evaluatePolicy", () => {
  it("measures the greedy policy without changing it", () => {
    const { qTable } = trainFresh(DEFAULT_REWARDS, 0.3, 300, 7);
    const snapshot = qTable.map((row) => [...row]);
    evaluatePolicy(qTable, DEFAULT_REWARDS);
    expect(qTable).toEqual(snapshot);
  });

  it("returns rates that sum to one", () => {
    const { report } = trainFresh(DEFAULT_REWARDS, 0.1, 200, 11);
    expect(report.goalRate + report.trapRate + report.timeoutRate).toBeCloseTo(
      1,
      10,
    );
  });
});

describe("the store", () => {
  beforeEach(() => {
    useAcademyStore.getState().reset();
  });

  it("starts with the default rewards, a low epsilon and no agent", () => {
    const state = useAcademyStore.getState();
    expect(state.rewards).toEqual(DEFAULT_REWARDS);
    expect(state.epsilon).toBe(DEFAULT_EPSILON);
    expect(state.episodesUsed).toBe(0);
    expect(state.history).toHaveLength(0);
    expect(state.qTable.flat().every((value) => value === 0)).toBe(true);
    expect(state.optimal.reachesGoal).toBe(true);
    expect(state.failure).toBeNull();
    expect(state.phase).toBe("designing");
  });

  it("trains a batch and keeps the history contiguous", () => {
    const store = useAcademyStore.getState();
    store.setEpsilon(0.3);
    useAcademyStore.getState().train();
    useAcademyStore.getState().train();

    const state = useAcademyStore.getState();
    expect(state.episodesUsed).toBe(2 * EPISODE_BATCH);
    expect(state.history).toHaveLength(2 * EPISODE_BATCH);
    expect(state.history.map((record) => record.index)).toEqual(
      Array.from({ length: 2 * EPISODE_BATCH }, (_, index) => index + 1),
    );
  });

  it("does not reuse the same random stream across batches", () => {
    useAcademyStore.getState().setEpsilon(0.3);
    useAcademyStore.getState().train();
    const first = useAcademyStore
      .getState()
      .history.slice(0, EPISODE_BATCH)
      .map((record) => record.reward);
    useAcademyStore.getState().train();
    const second = useAcademyStore
      .getState()
      .history.slice(EPISODE_BATCH)
      .map((record) => record.reward);
    expect(second).not.toEqual(first);
  });

  it("throws the agent away when a reward changes, and says why", () => {
    useAcademyStore.getState().setEpsilon(0.3);
    useAcademyStore.getState().train();
    expect(useAcademyStore.getState().episodesUsed).toBe(EPISODE_BATCH);

    useAcademyStore.getState().setReward("pellet", 3);
    const state = useAcademyStore.getState();
    expect(state.episodesUsed).toBe(0);
    expect(state.history).toHaveLength(0);
    expect(state.qTable.flat().every((value) => value === 0)).toBe(true);
    expect(state.rewards.pellet).toBe(3);
    // Value iteration is recomputed immediately, so the editor can warn before
    // the player spends a single episode.
    expect(state.optimal.reachesGoal).toBe(false);
    expect(state.whyCard?.tone).toBe("bad");
  });

  it("throws the agent away when epsilon changes, so each run is one experiment", () => {
    useAcademyStore.getState().setEpsilon(0.3);
    useAcademyStore.getState().train();
    useAcademyStore.getState().setEpsilon(0.5);
    expect(useAcademyStore.getState().episodesUsed).toBe(0);
    expect(useAcademyStore.getState().epsilon).toBe(0.5);
  });

  it("ignores a no-op change", () => {
    useAcademyStore.getState().setEpsilon(0.3);
    useAcademyStore.getState().train();
    const before = useAcademyStore.getState();
    before.setEpsilon(0.3);
    before.setReward("goal", DEFAULT_REWARDS.goal);
    const after = useAcademyStore.getState();
    expect(after.episodesUsed).toBe(EPISODE_BATCH);
    expect(after.history).toBe(before.history);
  });

  it("forgets the agent but keeps the rules", () => {
    useAcademyStore.getState().setReward("pellet", 3);
    useAcademyStore.getState().setEpsilon(0.3);
    useAcademyStore.getState().train();
    useAcademyStore.getState().forget();

    const state = useAcademyStore.getState();
    expect(state.episodesUsed).toBe(0);
    expect(state.rewards.pellet).toBe(3);
    expect(state.epsilon).toBe(0.3);
  });

  it("names reward hacking on the first batch, without waiting for evidence", () => {
    // Provable from the reward function alone, so it does not have to wait like
    // the other verdicts do.
    useAcademyStore.getState().setReward("pellet", 3);
    useAcademyStore.getState().setEpsilon(0.3);
    useAcademyStore.getState().train();

    const state = useAcademyStore.getState();
    expect(state.episodesUsed).toBeLessThan(DIAGNOSE_AFTER);
    expect(state.evaluation?.outcome).toBe("reward-hacking");
    expect(state.failure?.name).toBe("Reward hacking");
    expect(state.phase).toBe("failed");
  });

  it("holds back the other verdicts until there is enough training to be fair", () => {
    useAcademyStore.getState().setEpsilon(0);
    useAcademyStore.getState().train();

    // One batch at epsilon 0 is already doomed, but saying so after 100 episodes
    // would be crying wolf — at epsilon 0.3 the maze is solved by episode 200.
    expect(useAcademyStore.getState().evaluation?.outcome).toBe("no-exploration");
    expect(useAcademyStore.getState().failure).toBeNull();

    useAcademyStore.getState().train();
    expect(useAcademyStore.getState().episodesUsed).toBe(DIAGNOSE_AFTER);
    expect(useAcademyStore.getState().failure?.name).toBe("No exploration");
  });

  it("graduates and records progression", () => {
    useAcademyStore.getState().setEpsilon(0.3);
    for (let batch = 0; batch < 3; batch += 1) {
      useAcademyStore.getState().train();
      if (useAcademyStore.getState().phase === "graduated") break;
    }
    const state = useAcademyStore.getState();
    expect(state.phase).toBe("graduated");
    expect(state.report.goalRate).toBeGreaterThanOrEqual(COMPETENCE_RATE);
    expect(state.evaluation?.score).toBeGreaterThan(0.6);
  });

  it("never spends more than the episode ceiling", () => {
    useAcademyStore.getState().setEpsilon(0);
    for (let batch = 0; batch < MAX_EPISODES / EPISODE_BATCH + 4; batch += 1) {
      useAcademyStore.getState().train();
    }
    expect(useAcademyStore.getState().episodesUsed).toBe(MAX_EPISODES);
    expect(useAcademyStore.getState().history).toHaveLength(MAX_EPISODES);
  });

  it("clamps a partial final batch to the ceiling", () => {
    useAcademyStore.getState().setEpsilon(0);
    useAcademyStore.getState().train(MAX_EPISODES - 30);
    useAcademyStore.getState().train(EPISODE_BATCH);
    expect(useAcademyStore.getState().episodesUsed).toBe(MAX_EPISODES);
  });
});

describe("derived views", () => {
  it("draws a trail that stops when the walk starts repeating", () => {
    const trail = greedyTrail(emptyQTable(), DEFAULT_REWARDS);
    expect(trail.length).toBeGreaterThan(1);
    expect(trail.length).toBeLessThanOrEqual(MAX_STEPS + 1);
    expect(trail[0]).toEqual(START);
  });

  it("draws the winning trail as nine steps to the exit", () => {
    const { qTable } = trainFresh(DEFAULT_REWARDS, 0.3, 300, 7);
    const trail = greedyTrail(qTable, DEFAULT_REWARDS);
    expect(trail).toHaveLength(10);
    expect(trail.at(-1)).toEqual(GOAL);
  });

  it("draws the optimal trail to the exit at the defaults", () => {
    const optimal = solveOptimal(DEFAULT_REWARDS);
    const trail = optimalTrail(optimal, DEFAULT_REWARDS);
    expect(trail.at(-1)).toEqual(GOAL);
    expect(trail).toHaveLength(10);
  });

  it("draws the optimal trail as a loop when the cheese is over-paid", () => {
    const rewards = withRewards({ pellet: 3 });
    const trail = optimalTrail(solveOptimal(rewards), rewards);
    expect(trail.at(-1)).not.toEqual(GOAL);
    expect(
      trail.some((p) => p.row === PELLET.row && p.col === PELLET.col),
    ).toBe(true);
  });

  it("reports one cell value per state, flagged by whether it was visited", () => {
    const untrained = cellValues(emptyQTable());
    expect(untrained).toHaveLength(STATE_COUNT);
    expect(untrained.every((cell) => !cell.visited)).toBe(true);
    expect(untrained.every((cell) => cell.best === 0)).toBe(true);

    const { qTable } = trainFresh(DEFAULT_REWARDS, 0.3, 300, 7);
    const trained = cellValues(qTable);
    expect(trained.filter((cell) => cell.visited).length).toBeGreaterThan(20);
    for (const cell of trained) {
      expect(cell.row).toBe(positionOf(cell.state).row);
      expect(cell.col).toBe(positionOf(cell.state).col);
    }
  });

  it("puts the highest learned value beside the exit", () => {
    const { qTable } = trainFresh(DEFAULT_REWARDS, 0.3, 300, 7);
    const values = cellValues(qTable);
    const best = values.reduce((top, cell) => (cell.best > top.best ? cell : top));
    // Adjacent to the exit — one step from collecting +20, so the highest Q-value
    // in the table. That is value propagating backwards from where the reward is.
    expect(
      Math.abs(best.row - GOAL.row) + Math.abs(best.col - GOAL.col),
    ).toBe(1);
    expect(best.best).toBeGreaterThan(DEFAULT_REWARDS.goal * 0.9);
  });
});

describe("the code lane api", () => {
  beforeEach(() => {
    useAcademyStore.getState().reset();
  });

  it("writes the same state the sliders write", () => {
    const api = createCodeApi();
    api.setReward("pellet", 2.5);
    api.setEpsilon(0.3);
    expect(useAcademyStore.getState().rewards.pellet).toBe(2.5);
    expect(useAcademyStore.getState().epsilon).toBe(0.3);

    api.train(200);
    expect(useAcademyStore.getState().episodesUsed).toBe(200);
    expect(api.episodes()).toBe(200);
    expect(api.report().episodesUsed).toBe(200);
  });

  it("rejects a reward that does not exist, by name", () => {
    const api = createCodeApi();
    expect(() => api.setReward("cheese", 3)).toThrow(/Unknown reward/);
    expect(() => api.setReward("pellet", Number.NaN)).toThrow(/finite/);
  });

  it("rejects an epsilon outside 0 to 1", () => {
    const api = createCodeApi();
    expect(() => api.setEpsilon(-0.1)).toThrow(/between 0 and 1/);
    expect(() => api.setEpsilon(1.5)).toThrow(/between 0 and 1/);
  });

  it("simulates without spending the player's episodes or touching their agent", () => {
    const api = createCodeApi();
    api.setEpsilon(0.3);
    api.train(100);
    const before = useAcademyStore.getState();
    const snapshot = before.qTable.map((row) => [...row]);

    const result = api.simulate({ epsilon: 0.5, episodes: 400, seed: 3 });
    expect(result.solved).toBe(true);

    const after = useAcademyStore.getState();
    expect(after.episodesUsed).toBe(100);
    expect(after.qTable).toEqual(snapshot);
    expect(after.epsilon).toBe(0.3);
  });

  it("simulates other reward functions without adopting them", () => {
    const api = createCodeApi();
    const hacked = api.simulate({
      rewards: { pellet: 3 },
      epsilon: 0.5,
      episodes: 400,
      seed: 3,
    });
    expect(hacked.solved).toBe(false);
    expect(hacked.meanPellets).toBeGreaterThan(5);
    expect(useAcademyStore.getState().rewards.pellet).toBe(
      DEFAULT_REWARDS.pellet,
    );
  });

  it("solves arbitrary reward functions for free", () => {
    const api = createCodeApi();
    expect(api.solve().reachesGoal).toBe(true);
    expect(api.solve({ pellet: 3 }).reachesGoal).toBe(false);
    expect(api.solve({ pellet: 3 }).roundTripValue).toBeCloseTo(2, 6);
    expect(useAcademyStore.getState().rewards.pellet).toBe(
      DEFAULT_REWARDS.pellet,
    );
  });

  it("guards the simulate budget", () => {
    const api = createCodeApi();
    expect(() => api.simulate({ episodes: 0 })).toThrow(/between 1 and 20000/);
    expect(() => api.simulate({ epsilon: 2 })).toThrow(/between 0 and 1/);
  });

  it("hands back copies, not live state", () => {
    const api = createCodeApi();
    const rewards = api.rewards();
    rewards.goal = 9999;
    expect(useAcademyStore.getState().rewards.goal).toBe(DEFAULT_REWARDS.goal);
  });

  it("reports a NaN for wall cells so a printed grid lines up", () => {
    const api = createCodeApi();
    const values = api.values();
    expect(values).toHaveLength(STATE_COUNT);
    // This maze has no walls, so nothing should be NaN.
    expect(values.every((value) => Number.isFinite(value))).toBe(true);
  });
});

describe("why-cards", () => {
  it("returns a stable key per event so the card can animate on change", () => {
    const briefing = whyCardFor({ kind: "briefing" });
    expect(whyCardFor({ kind: "briefing" }).key).toBe(briefing.key);
  });

  it("warns before a single episode is spent when the rewards are broken", () => {
    const rewards = withRewards({ pellet: 3 });
    const card = whyCardFor({
      kind: "reward-changed",
      keys: ["pellet"],
      rewards,
      optimal: solveOptimal(rewards),
      hadEpisodes: 0,
    });
    expect(card.tone).toBe("bad");
    expect(card.body).toMatch(/never leaves/);
  });

  it("stays neutral when a reward change keeps the exit optimal", () => {
    const rewards = withRewards({ goal: 30 });
    const card = whyCardFor({
      kind: "reward-changed",
      keys: ["goal"],
      rewards,
      optimal: solveOptimal(rewards),
      hadEpisodes: 100,
    });
    expect(card.tone).toBe("info");
    expect(card.body).toMatch(/100 episodes/);
  });

  it("calls out epsilon 0 specifically", () => {
    const card = whyCardFor({
      kind: "epsilon-changed",
      epsilon: 0,
      previous: 0.3,
      hadEpisodes: 0,
    });
    expect(card.tone).toBe("warn");
    expect(card.title).toMatch(/ε 0/);
  });

  it("explains a graduation in terms of what the rewards asked for", () => {
    const rewards = DEFAULT_REWARDS;
    const { qTable, history } = trainFresh(rewards, 0.3, 300, 7);
    const evaluation = evaluateRun({
      qTable,
      rewards,
      epsilon: 0.3,
      episodesUsed: 300,
    });
    const card = whyCardFor({
      kind: "trained",
      evaluation,
      batch: history,
      episodesUsed: 300,
      epsilon: 0.3,
      rewards,
    });
    expect(card.tone).toBe("good");
    expect(card.body).toMatch(/best possible policy/i);
  });

  it("says the reward went up and the task went unfinished", () => {
    const rewards = withRewards({ pellet: 3 });
    const { qTable, history } = trainFresh(rewards, 0.3, 400, 7);
    const evaluation = evaluateRun({
      qTable,
      rewards,
      epsilon: 0.3,
      episodesUsed: 400,
    });
    const card = whyCardFor({
      kind: "trained",
      evaluation,
      batch: history,
      episodesUsed: 400,
      epsilon: 0.3,
      rewards,
    });
    expect(card.tone).toBe("bad");
    expect(card.body).toMatch(/reward went UP and the task went unfinished/);
  });

  it("has a card for every training outcome", () => {
    const cases: Array<[RewardConfig, number, number]> = [
      [DEFAULT_REWARDS, 0.3, 300],
      [withRewards({ pellet: 3 }), 0.3, 400],
      [DEFAULT_REWARDS, 0, 300],
      [withRewards({ pellet: 2 }), 0.3, 400],
      [DEFAULT_REWARDS, 0.15, 100],
    ];
    const seen = new Set<Outcome>();
    for (const [rewards, epsilon, episodes] of cases) {
      const { qTable, history } = trainFresh(rewards, epsilon, episodes, 7);
      const evaluation = evaluateRun({
        qTable,
        rewards,
        epsilon,
        episodesUsed: episodes,
      });
      seen.add(evaluation.outcome);
      const card = whyCardFor({
        kind: "trained",
        evaluation,
        batch: history,
        episodesUsed: episodes,
        epsilon,
        rewards,
      });
      expect(card.key.length).toBeGreaterThan(0);
      expect(card.body.length).toBeGreaterThan(80);
    }
    expect(seen.size).toBe(cases.length);
  });
});

describe("the cell the maze is built around", () => {
  it("keeps the cheese two steps from the start and on the way out", () => {
    // The cheese has to be ON the shortest route, or passing it would cost a
    // detour and the "+1 on the way past" arithmetic in the metric would be wrong.
    const optimal = solveOptimal(DEFAULT_REWARDS);
    const trail = optimalTrail(optimal, DEFAULT_REWARDS);
    expect(
      trail.some((p) => p.row === PELLET.row && p.col === PELLET.col),
    ).toBe(true);
    expect(trail).toHaveLength(stepsFromStart(GOAL) + 1);
  });

  it("keeps the pit off the shortest route, so avoiding it is free", () => {
    const optimal = solveOptimal(DEFAULT_REWARDS);
    const trail = optimalTrail(optimal, DEFAULT_REWARDS);
    expect(trail.some((p) => p.row === TRAP.row && p.col === TRAP.col)).toBe(
      false,
    );
  });

  it("has a walkable cell for every state, since there are no walls", () => {
    for (let state = 0; state < STATE_COUNT; state += 1) {
      expect(cellAt(positionOf(state))).not.toBe("wall");
    }
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Invariants behind the audit fixes. Each one is a claim some piece of copy or
// some readout makes, checked against the arithmetic that is supposed to back it.
// ═══════════════════════════════════════════════════════════════════════════

describe("what walking straight out is worth", () => {
  it("is the best policy's value when the best policy walks out", () => {
    // At the defaults the optimum IS the nine-move route past the cheese, so the
    // two readouts agree — to the last digit, because they are the same sum.
    expect(solveOptimal(DEFAULT_REWARDS).reachesGoal).toBe(true);
    expect(straightExitValue(DEFAULT_REWARDS)).toBeCloseTo(
      solveOptimal(DEFAULT_REWARDS).valueAtStart,
      6,
    );
    expect(straightExitValue(DEFAULT_REWARDS)).toBeCloseTo(10.52, 2);
  });

  it("stays the exit's own value when farming wins, instead of repeating the farm", () => {
    // The playthrough's reward-hacking setting. V*(start) is the farming value
    // here, which is why it could not be the "walking out" row: the editor
    // printed +19.2 twice and the exit's real +12.4 nowhere.
    const rewards = withRewards({ pellet: 3 });
    const optimal = solveOptimal(rewards);
    expect(optimal.behaviour).toBe("farm");
    expect(optimal.valueAtStart).toBeCloseTo(farmValue(rewards), 4);
    expect(straightExitValue(rewards)).toBeCloseTo(12.42, 2);
    expect(straightExitValue(rewards)).toBeLessThan(farmValue(rewards));
  });

  it("is never worth more than the best policy, whatever the rewards", () => {
    for (const knob of REWARD_KNOBS) {
      for (const value of [knob.min, knob.max]) {
        const rewards = withRewards({ [knob.key]: value });
        expect(straightExitValue(rewards)).toBeLessThanOrEqual(
          solveOptimal(rewards).valueAtStart + 1e-9,
        );
      }
    }
  });
});

describe("the discount the copy quotes", () => {
  it("weights the exit by gamma to the eighth, which is what value iteration does", () => {
    // Goal-only rewards: the only thing V*(start) can be is the exit's payout,
    // discounted by however the backup discounts move nine.
    const goalOnly: RewardConfig = { goal: 20, trap: 0, step: 0, pellet: 0, bump: 0 };
    const moves = stepsFromStart(GOAL);
    expect(moves).toBe(9);
    expect(solveOptimal(goalOnly).valueAtStart).toBeCloseTo(
      20 * discountAfter(moves),
      9,
    );
    expect(discountAfter(moves)).toBeCloseTo(DISCOUNT ** 8, 12);
  });

  it("weights the cheese on move two by gamma, not gamma squared", () => {
    const cheeseOnly: RewardConfig = { goal: 0, trap: 0, step: 0, pellet: 1, bump: 0 };
    expect(straightExitValue(cheeseOnly)).toBeCloseTo(
      discountAfter(stepsFromStart(PELLET)),
      12,
    );
    expect(discountAfter(stepsFromStart(PELLET))).toBeCloseTo(DISCOUNT, 12);
  });

  it("prints those factors in the math notes and the reward-hacking verdict", () => {
    const exit = discountAfter(stepsFromStart(GOAL)).toFixed(2);
    const cheese = discountAfter(stepsFromStart(PELLET)).toFixed(2);
    expect(MATH_NOTES).toContain(`keeps ${exit} of what you pay`);
    expect(MATH_NOTES).toContain(`keeps ${cheese}`);
    // The off-by-one it used to make: gamma^9 = 0.63 and gamma^2 = 0.90.
    expect(MATH_NOTES).not.toContain((DISCOUNT ** 9).toFixed(2));
    expect(MATH_NOTES).not.toContain((DISCOUNT ** 2).toFixed(2));

    const rewards = withRewards({ pellet: 3 });
    const { qTable } = trainFresh(rewards, 0.3, 400, 7);
    const evaluation = evaluateRun({ qTable, rewards, epsilon: 0.3, episodesUsed: 400 });
    expect(evaluation.outcome).toBe("reward-hacking");
    const detail = evaluation.failure!.detail;
    expect(detail).toContain(`discounted to ${exit} of face value`);
    expect(detail).toContain(`+${straightExitValue(rewards).toFixed(1)}`);
  });
});

describe("the second star", () => {
  it("names the last episode count that still scores the high-score threshold", () => {
    const budget = episodesForScore(HIGH_SCORE_THRESHOLD);
    expect(budget).toBeGreaterThan(0);
    expect(budget).toBeLessThan(MAX_EPISODES);
    expect(competentScore(budget)).toBeGreaterThanOrEqual(HIGH_SCORE_THRESHOLD);
    expect(competentScore(budget + 1)).toBeLessThan(HIGH_SCORE_THRESHOLD);
    // And the engine agrees about what those scores are worth.
    expect(
      starsFor({ completed: true, bestScore: competentScore(budget), codeLaneCleared: false }),
    ).toBe(2);
    expect(
      starsFor({ completed: true, bestScore: competentScore(budget + 1), codeLaneCleared: true }),
    ).toBe(1);
  });

  it("is the score evaluateRun awards a graduate", () => {
    const { qTable } = trainFresh(DEFAULT_REWARDS, 0.3, 300, 7);
    const evaluation = evaluateRun({
      qTable,
      rewards: DEFAULT_REWARDS,
      epsilon: 0.3,
      episodesUsed: 300,
    });
    expect(evaluation.outcome).toBe("competent");
    expect(evaluation.score).toBe(competentScore(300));
  });
});

describe("tracking what was actually tried", () => {
  it("counts every step taken, per state and action", () => {
    const qTable = emptyQTable();
    const visits = emptyQTable();
    const records = runEpisodes({
      qTable,
      visits,
      rewards: DEFAULT_REWARDS,
      epsilon: 0.3,
      episodes: 20,
      random: seededRandom(5),
    });
    const steps = records.reduce((total, record) => total + record.steps, 0);
    expect(visits.flat().reduce((total, count) => total + count, 0)).toBe(steps);
  });

  it("marks a tried action as tried even when its value is exactly 0", () => {
    // Step and wall costs of 0 and no cheese: the agent's first moves update to
    // r + gamma*0 = 0 and look, by value alone, untouched.
    const rewards = withRewards({ step: 0, bump: 0, pellet: 0 });
    const qTable = emptyQTable();
    const visits = emptyQTable();
    runEpisodes({
      qTable,
      visits,
      rewards,
      epsilon: 1,
      episodes: 1,
      random: seededRandom(3),
    });
    const start = stateOf(START);
    const triedAtZero = ACTIONS.some(
      (action) => visits[start]![action]! > 0 && qTable[start]![action] === 0,
    );
    expect(triedAtZero).toBe(true);

    const cell = cellValues(qTable, visits).find((entry) => entry.state === start)!;
    expect(cell.visited).toBe(true);
  });

  it("keeps visits in the store and clears them with the agent", () => {
    useAcademyStore.getState().reset();
    useAcademyStore.getState().setEpsilon(0.3);
    useAcademyStore.getState().train();
    const state = useAcademyStore.getState();
    const steps = state.history.reduce((total, record) => total + record.steps, 0);
    expect(state.visits.flat().reduce((total, count) => total + count, 0)).toBe(steps);

    useAcademyStore.getState().forget();
    expect(
      useAcademyStore.getState().visits.flat().every((count) => count === 0),
    ).toBe(true);
  });
});

describe("the store's guards", () => {
  beforeEach(() => {
    useProgression.setState({
      ...EMPTY_PROGRESSION,
      hydrated: true,
      syncError: null,
      lastGain: null,
    });
    useProgression.getState().setAdapter(createMemoryAdapter());
    useAcademyStore.getState().reset();
  });

  const graduate = (train: () => void) => {
    useAcademyStore.getState().setEpsilon(0.3);
    for (let batch = 0; batch < 6; batch += 1) {
      train();
      if (useAcademyStore.getState().phase === "graduated") return;
    }
    throw new Error("expected ε 0.3 to graduate within 600 episodes");
  };

  it("rejects a NaN, a non-number or a sub-one episode count from the code lane", () => {
    const api = createCodeApi();
    expect(() => api.train(Number.NaN)).toThrow(/at least 1/);
    expect(() => api.train(0.5)).toThrow(/at least 1/);
    expect(() => api.train("100" as unknown as number)).toThrow(/at least 1/);
    expect(useAcademyStore.getState().episodesUsed).toBe(0);
  });

  it("never lets a bad count reach the episode counter, whoever sends it", () => {
    useAcademyStore.getState().train(Number.NaN);
    expect(useAcademyStore.getState().episodesUsed).toBe(0);
    useAcademyStore.getState().train(150.7);
    expect(useAcademyStore.getState().episodesUsed).toBe(150);
    expect(useAcademyStore.getState().history).toHaveLength(150);
  });

  it("will not let a re-run snippet train a graduated agent again", () => {
    // The visual lane swaps Train for "New academy" at graduation; the code lane
    // gets the same rule, by name, so a second run cannot un-graduate the first.
    const api = createCodeApi();
    graduate(() => api.train());
    const before = useAcademyStore.getState();
    expect(() => api.train(100)).toThrow(/already graduated/);
    const after = useAcademyStore.getState();
    expect(after.phase).toBe("graduated");
    expect(after.episodesUsed).toBe(before.episodesUsed);
    expect(after.history).toBe(before.history);

    // And forgetting the agent is the documented way to go again.
    api.forget();
    expect(() => api.train(100)).not.toThrow();
  });

  it("credits the code lane only when the code lane trained the agent", () => {
    // The visible tab says code, but the click came from the visual Train button.
    useAcademyStore.getState().setLane("code");
    graduate(() => useAcademyStore.getState().train());
    expect(useProgression.getState().games[SLUG]?.completed).toBe(true);
    expect(useProgression.getState().games[SLUG]?.codeLaneCleared).toBe(false);

    useAcademyStore.getState().reset();
    useAcademyStore.getState().setLane("visual");
    const api = createCodeApi();
    graduate(() => api.train());
    expect(useProgression.getState().games[SLUG]?.codeLaneCleared).toBe(true);
  });

  it("rejects typos and non-numbers in the rewards a simulation runs on", () => {
    const api = createCodeApi();
    expect(() => api.simulate({ rewards: { cheese: 3 } as never })).toThrow(
      /Unknown reward "cheese"/,
    );
    expect(() => api.simulate({ rewards: { pellet: Number.NaN } })).toThrow(/finite/);
    expect(() => api.solve({ goal: "20" as unknown as number })).toThrow(/finite/);
    // Experiments may go past the sliders; the player's own rewards may not.
    expect(api.solve({ pellet: 20 }).reachesGoal).toBe(false);
    expect(() => api.setReward("pellet", 20)).toThrow(/between 0 and 6/);
    expect(useAcademyStore.getState().rewards.pellet).toBe(DEFAULT_REWARDS.pellet);
  });

  it("does not name a verdict on the why-card before the failure strip would", () => {
    useAcademyStore.getState().setEpsilon(0);
    useAcademyStore.getState().train();
    let state = useAcademyStore.getState();
    // The verdict is already computed — it is held back, not missing.
    expect(state.evaluation?.outcome).toBe("no-exploration");
    expect(state.failure).toBeNull();
    expect(state.whyCard?.key).toMatch(/^progress-/);
    expect(state.whyCard?.body).not.toMatch(/Another \d+ episodes will move it/);

    useAcademyStore.getState().train();
    state = useAcademyStore.getState();
    expect(state.episodesUsed).toBe(DIAGNOSE_AFTER);
    expect(state.failure?.name).toBe("No exploration");
    expect(state.whyCard?.key).toMatch(/^stuck-/);
  });
});
