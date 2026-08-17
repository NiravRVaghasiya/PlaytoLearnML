import type { NamedFailure } from "@/engine/types";
import { clamp, seededRandom } from "@/lib/utils";

/**
 * Agent Academy — tabular Q-learning.
 *
 * The player never moves the agent. They write its reward function and set how
 * much it explores, and then watch what those two choices produce. That is the
 * whole point: you shape behaviour through rewards, not through control, and a
 * reward function is a specification you can get wrong in ways that look like
 * success.
 *
 * ── Telling the failures apart, provably ────────────────────────────────────
 * Four different mistakes all look identical on screen — the agent does not
 * finish — and they have different fixes, so guessing between them teaches
 * nothing. This module separates them with two cheap experiments:
 *
 *   `solveOptimal` runs value iteration on the player's reward function. The maze
 *     is 28 states by 4 actions and fully known, so we can compute the BEST
 *     possible policy for those rewards exactly. If the best possible policy does
 *     not leave the maze, no agent could ever be asked to — that is reward
 *     hacking, and it is proved rather than inferred.
 *
 *   a reference training run with generous exploration answers the other half:
 *     given rewards that CAN be satisfied, was the search the problem? If the same
 *     rewards with a high epsilon succeed, the player's epsilon was too low. If
 *     even that fails, the rewards are technically satisfiable but so punishing
 *     that nothing survives long enough to find the exit — a real phenomenon, and
 *     a different fix again.
 *
 * ── Deviations from the spec, flagged ───────────────────────────────────────
 * 1. Tabular, not DQN. The spec offers either. On a 4x7 grid the Q-table is 28
 *    states by 4 actions, so a neural approximator would be slower, noisier, and
 *    would turn the Q-value heatmap — the clearest window into what the agent
 *    believes — into an approximation of itself. There is no tensor computation
 *    here, so no TF.js.
 * 2. `<GridWorld>` is drawn in SVG rather than Phaser, as in Overfit Tower
 *    Defense and for the same reasons: the grid is 28 cells with a token on it,
 *    the Q-heatmap has to overlay those same cells, and an SVG grid is in the
 *    accessibility tree already.
 * 3. The spec suggests "cheese reward too high NEAR A TRAP -> greedy suicidal
 *    behavior", which implies a risky cell. In a deterministic grid, standing
 *    beside a trap costs exactly nothing — measured across every layout and
 *    reward setting tried, the greedy policy's trap rate was 0.00 without
 *    exception. The obvious fix, a slip probability, turned out to explore the
 *    maze all by itself: at slip 0.05 an epsilon-ZERO agent solved the maze in
 *    10 of 10 seeds, which erases the exploration lesson entirely. So the grid
 *    stays deterministic and the trap earns its place a different way: it sits two
 *    steps from the start, where a punishing step cost makes falling in the
 *    cheapest way to end a badly-paid episode. Suicidal behaviour, same lesson,
 *    reachable without stochasticity.
 */

// ── The grid ──────────────────────────────────────────────────────────────

export type Cell = "empty" | "wall" | "trap" | "pellet" | "start" | "goal";

/**
 * The academy maze.
 *
 * Tuned by measurement, not taste. Every constant below was chosen from a sweep
 * over six layouts, three learning rates, four step caps and seven epsilons,
 * scored on twelve seeds each. What this arrangement buys:
 *
 *   - The cheese sits TWO steps from the start, on the natural route to the exit,
 *     so every agent finds it in its first few episodes. It is the obvious habit
 *     to fall into.
 *   - The exit is NINE steps away, far enough that finding it takes real
 *     exploration and near enough that a modest epsilon manages it.
 *   - The trap sits two steps from the start as well — the quick way out.
 *   - No walls. Walls made the search harder without teaching anything, and they
 *     pushed the epsilon a player needs up to 0.5, which is not a number anyone
 *     would recognise as "some exploration".
 */
const LAYOUT = ["S.P....", ".T.....", ".......", "......G"] as const;

const CELL_FROM_CHAR: Record<string, Cell> = {
  ".": "empty",
  "#": "wall",
  T: "trap",
  P: "pellet",
  S: "start",
  G: "goal",
};

export const GRID: Cell[][] = LAYOUT.map((row) =>
  [...row].map((char) => CELL_FROM_CHAR[char] ?? "empty"),
);

export const ROWS = GRID.length;
export const COLS = GRID[0]!.length;
export const STATE_COUNT = ROWS * COLS;

export interface Position {
  row: number;
  col: number;
}

function findCell(kind: Cell): Position {
  for (let row = 0; row < ROWS; row += 1) {
    for (let col = 0; col < COLS; col += 1) {
      if (GRID[row]![col] === kind) return { row, col };
    }
  }
  return { row: 0, col: 0 };
}

export const START = findCell("start");
export const GOAL = findCell("goal");
export const PELLET = findCell("pellet");
export const TRAP = findCell("trap");

export const stateOf = (position: Position): number =>
  position.row * COLS + position.col;
export const positionOf = (state: number): Position => ({
  row: Math.floor(state / COLS),
  col: state % COLS,
});

export const cellAt = (position: Position): Cell =>
  GRID[position.row]?.[position.col] ?? "wall";

export const isWall = (position: Position): boolean =>
  position.row < 0 ||
  position.row >= ROWS ||
  position.col < 0 ||
  position.col >= COLS ||
  cellAt(position) === "wall";

/** True where the episode ends on arrival. Value iteration needs this. */
export const isTerminal = (state: number): boolean => {
  const cell = cellAt(positionOf(state));
  return cell === "goal" || cell === "trap";
};

/** Shortest walkable distance from the start, ignoring rewards. */
export function stepsFromStart(target: Position): number {
  const seen = new Set<number>([stateOf(START)]);
  let frontier: Position[] = [START];
  let distance = 0;

  while (frontier.length > 0) {
    if (frontier.some((p) => p.row === target.row && p.col === target.col)) {
      return distance;
    }
    const next: Position[] = [];
    for (const position of frontier) {
      // Traps end the episode, so you cannot path THROUGH one.
      if (cellAt(position) === "trap" && stateOf(position) !== stateOf(START)) {
        continue;
      }
      for (const action of ACTIONS) {
        const candidate = move(position, action);
        if (isWall(candidate)) continue;
        const key = stateOf(candidate);
        if (seen.has(key)) continue;
        seen.add(key);
        next.push(candidate);
      }
    }
    frontier = next;
    distance += 1;
  }

  return Number.POSITIVE_INFINITY;
}

// ── Actions ───────────────────────────────────────────────────────────────

export type Action = 0 | 1 | 2 | 3;
export const ACTIONS: readonly Action[] = [0, 1, 2, 3] as const;
export const ACTION_LABELS = ["up", "right", "down", "left"] as const;
export const ACTION_ARROWS = ["↑", "→", "↓", "←"] as const;

const DELTAS: ReadonlyArray<Position> = [
  { row: -1, col: 0 },
  { row: 0, col: 1 },
  { row: 1, col: 0 },
  { row: 0, col: -1 },
];

export function move(position: Position, action: Action): Position {
  const delta = DELTAS[action]!;
  return { row: position.row + delta.row, col: position.col + delta.col };
}

// ── The reward function: the thing the player actually writes ─────────────

export interface RewardConfig {
  /** Reaching the exit. Ends the episode. */
  goal: number;
  /** Stepping on a trap. Ends the episode. */
  trap: number;
  /** Charged on every move. Negative makes the agent want to finish. */
  step: number;
  /**
   * Collected every time the agent enters the cheese cell.
   *
   * Deliberately repeatable. A one-shot pellet would need the collected-set in the
   * state, which would blow up a tabular Q-table and make the heatmap unreadable —
   * and the repeatable version is what makes farming possible, which is the more
   * valuable lesson.
   */
  pellet: number;
  /** Walking into a wall. The agent stays put and pays this. */
  bump: number;
}

/**
 * The starting rewards.
 *
 * `pellet: 1` against `step: -0.5` is chosen to the decimal: a round trip to the
 * cheese and back costs two moves and pays one cheese, so it nets EXACTLY zero.
 * That single fact carries both lessons. Zero is still better than every
 * alternative on a board where moving costs you, so a greedy agent that finds the
 * cheese early will pace beside it forever — which is why epsilon matters here.
 * And one nudge up the slider makes the loop pay a profit, at which point pacing
 * beside the cheese becomes genuinely optimal and no amount of exploration will
 * talk the agent out of it.
 */
export const DEFAULT_REWARDS: RewardConfig = {
  goal: 20,
  trap: -20,
  step: -0.5,
  pellet: 1,
  bump: -1,
};

export interface RewardKnob {
  key: keyof RewardConfig;
  label: string;
  min: number;
  max: number;
  step: number;
  blurb: string;
}

export const REWARD_KNOBS: readonly RewardKnob[] = [
  {
    key: "goal",
    label: "Reach the exit",
    min: 0,
    max: 40,
    step: 1,
    blurb: "Paid once, when the episode ends successfully.",
  },
  {
    key: "trap",
    label: "Fall in the pit",
    min: -40,
    max: 0,
    step: 1,
    blurb: "Paid once, and the episode ends there.",
  },
  {
    key: "step",
    label: "Per move",
    min: -2,
    max: 1,
    step: 0.1,
    blurb:
      "Charged on every move. Make this positive and finishing stops being worth it.",
  },
  {
    key: "pellet",
    label: "Eat cheese",
    min: 0,
    max: 6,
    step: 0.5,
    blurb:
      "Paid every time the agent enters that square. It can be eaten again and again.",
  },
  {
    key: "bump",
    label: "Walk into a wall",
    min: -4,
    max: 0,
    step: 0.5,
    blurb: "The agent stays where it is and pays this.",
  },
] as const;

// ── The environment ───────────────────────────────────────────────────────

/**
 * Steps per episode.
 *
 * 80 is roughly nine times the shortest solution, which is enough slack for a
 * wandering agent to stumble on the exit and short enough that a farming agent's
 * episode ends while the reward curve still fits on screen.
 */
export const MAX_STEPS = 80;

export type StepOutcome = "moved" | "bumped" | "goal" | "trap";

export interface StepResult {
  next: Position;
  reward: number;
  done: boolean;
  outcome: StepOutcome;
}

/** One environment transition. Deterministic — see deviation 3 in the header. */
export function stepEnvironment(
  position: Position,
  action: Action,
  rewards: RewardConfig,
): StepResult {
  const target = move(position, action);

  if (isWall(target)) {
    return {
      next: position,
      reward: rewards.bump,
      done: false,
      outcome: "bumped",
    };
  }

  const cell = cellAt(target);

  if (cell === "goal") {
    return {
      next: target,
      reward: rewards.step + rewards.goal,
      done: true,
      outcome: "goal",
    };
  }

  if (cell === "trap") {
    return {
      next: target,
      reward: rewards.step + rewards.trap,
      done: true,
      outcome: "trap",
    };
  }

  return {
    next: target,
    reward: rewards.step + (cell === "pellet" ? rewards.pellet : 0),
    done: false,
    outcome: "moved",
  };
}

// ── Q-learning ────────────────────────────────────────────────────────────

/**
 * Learning rate and discount.
 *
 * alpha 0.4 rather than the textbook 0.1: the environment is deterministic, so
 * there is no observation noise to average away, and a larger step propagates the
 * exit's value back along the corridor in far fewer episodes. Measured over
 * twelve seeds, 0.2 and 0.4 solve the same fraction of runs; 0.4 does it sooner.
 *
 * gamma 0.95 is what makes the agent prefer finishing quickly, and it is also the
 * reason the cheese is tempting: a reward nine steps away is worth 0.63 of its
 * face value, while a reward two steps away keeps 0.90 of it.
 */
export const LEARNING_RATE = 0.4;
export const DISCOUNT = 0.95;

export type QTable = number[][];

export function emptyQTable(): QTable {
  return Array.from({ length: STATE_COUNT }, () =>
    new Array<number>(ACTIONS.length).fill(0),
  );
}

/**
 * Greedy action, ties broken at random.
 *
 * The tie-break matters more than it looks. With a zero-initialised table every
 * action ties on the first visit, so deterministic tie-breaking would make an
 * epsilon-zero agent walk the same direction until it hit a wall — "stuck" for a
 * reason that has nothing to do with exploration. Random tie-breaking gives the
 * greedy agent the fairest possible start, so when it still fails, exploration is
 * genuinely what was missing.
 */
export function greedyAction(
  qTable: QTable,
  state: number,
  random: () => number,
): Action {
  const row = qTable[state] ?? [];
  let best = -Infinity;
  const ties: Action[] = [];

  for (const action of ACTIONS) {
    const value = row[action] ?? 0;
    if (value > best + 1e-12) {
      best = value;
      ties.length = 0;
      ties.push(action);
    } else if (Math.abs(value - best) <= 1e-12) {
      ties.push(action);
    }
  }

  return ties[Math.floor(random() * ties.length)] ?? 0;
}

export interface EpisodeRecord {
  index: number;
  reward: number;
  steps: number;
  /** How the episode ended. */
  ending: "goal" | "trap" | "timeout";
  /** How many times the cheese was eaten. */
  pelletsEaten: number;
}

export interface TrainOptions {
  qTable: QTable;
  rewards: RewardConfig;
  epsilon: number;
  episodes: number;
  random: () => number;
  /** Learn from what happens, or just measure the current policy. */
  learn?: boolean;
}

/**
 * Run episodes, optionally learning from them.
 *
 * The update is the plain Q-learning rule and nothing else:
 *
 *     Q(s,a) <- Q(s,a) + alpha * ( r + gamma * max_a' Q(s',a') - Q(s,a) )
 *
 * Terminal transitions drop the bootstrap term, because there is no next state to
 * look ahead to. Forgetting that is the classic way to make a Q-learner believe
 * the goal is worth far more than it is.
 */
export function runEpisodes({
  qTable,
  rewards,
  epsilon,
  episodes,
  random,
  learn = true,
}: TrainOptions): EpisodeRecord[] {
  const records: EpisodeRecord[] = [];

  for (let index = 0; index < episodes; index += 1) {
    let position = { ...START };
    let total = 0;
    let steps = 0;
    let pelletsEaten = 0;
    let ending: EpisodeRecord["ending"] = "timeout";

    while (steps < MAX_STEPS) {
      const state = stateOf(position);
      const action =
        random() < epsilon
          ? (Math.floor(random() * ACTIONS.length) as Action)
          : greedyAction(qTable, state, random);

      const result = stepEnvironment(position, action, rewards);
      steps += 1;
      total += result.reward;
      if (cellAt(result.next) === "pellet" && result.outcome === "moved") {
        pelletsEaten += 1;
      }

      if (learn) {
        const nextState = stateOf(result.next);
        const bootstrap = result.done
          ? 0
          : DISCOUNT * Math.max(...(qTable[nextState] ?? [0]));
        const current = qTable[state]![action]!;
        qTable[state]![action] =
          current + LEARNING_RATE * (result.reward + bootstrap - current);
      }

      position = result.next;

      if (result.done) {
        ending = result.outcome === "goal" ? "goal" : "trap";
        break;
      }
    }

    records.push({ index: index + 1, reward: total, steps, pelletsEaten, ending });
  }

  return records;
}

export interface PolicyReport {
  /** Fraction of rollouts that reached the exit. */
  goalRate: number;
  trapRate: number;
  timeoutRate: number;
  meanReward: number;
  meanSteps: number;
  meanPellets: number;
}

/** Measure the CURRENT greedy policy without learning from the rollouts. */
export function evaluatePolicy(
  qTable: QTable,
  rewards: RewardConfig,
  seed = 99,
  rollouts = 40,
): PolicyReport {
  const records = runEpisodes({
    qTable,
    rewards,
    epsilon: 0,
    episodes: rollouts,
    random: seededRandom(seed),
    learn: false,
  });

  const rate = (ending: EpisodeRecord["ending"]) =>
    records.filter((record) => record.ending === ending).length / records.length;
  const mean = (pick: (record: EpisodeRecord) => number) =>
    records.reduce((total, record) => total + pick(record), 0) / records.length;

  return {
    goalRate: rate("goal"),
    trapRate: rate("trap"),
    timeoutRate: rate("timeout"),
    meanReward: mean((record) => record.reward),
    meanSteps: mean((record) => record.steps),
    meanPellets: mean((record) => record.pelletsEaten),
  };
}

/** Train a fresh agent from scratch. Used for reference runs and the code lane. */
export function trainFresh(
  rewards: RewardConfig,
  epsilon: number,
  episodes: number,
  seed = 7,
): { qTable: QTable; history: EpisodeRecord[]; report: PolicyReport } {
  const qTable = emptyQTable();
  const history = runEpisodes({
    qTable,
    rewards,
    epsilon,
    episodes,
    random: seededRandom(seed),
  });
  return { qTable, history, report: evaluatePolicy(qTable, rewards) };
}

// ── What the reward function actually asks for ────────────────────────────

export type OptimalBehaviour = "exit" | "farm" | "quit" | "wander";

export interface OptimalPolicy {
  /** V*(s) for every state. */
  values: number[];
  /** The best action in every non-terminal state. */
  policy: Action[];
  /** Does following it from the start reach the exit inside an episode? */
  reachesGoal: boolean;
  /** What it does instead, when it doesn't. */
  behaviour: OptimalBehaviour;
  /** V*(start) — the most this reward function can pay anybody. */
  valueAtStart: number;
  /** V*(cheese) — the best that can be done from the cheese cell. */
  valueAtPellet: number;
  /** Undiscounted episode reward from following it. */
  episodeReward: number;
  steps: number;
  pelletsEaten: number;
}

/**
 * What pacing beside the cheese forever is worth, in closed form.
 *
 * Standing on the cheese, the loop is: step off (`step`), step back on
 * (`step + pellet`), repeat. Discounted, that geometric series is
 *
 *     V = step + gamma*(step + pellet) + gamma^2 * V
 *
 * which rearranges to the expression below. Worth having exactly rather than by
 * simulation, because it is the number that explains the whole game: compare it
 * against `valueAtPellet` from value iteration and you can see precisely how much
 * the agent settled for.
 */
export function farmValue(rewards: RewardConfig): number {
  return (
    (rewards.step + DISCOUNT * (rewards.step + rewards.pellet)) /
    (1 - DISCOUNT * DISCOUNT)
  );
}

/** Undiscounted profit on one round trip to the cheese. Zero at the defaults. */
export function roundTripValue(rewards: RewardConfig): number {
  return rewards.pellet + 2 * rewards.step;
}

/**
 * Value iteration on the player's reward function.
 *
 * The maze is 28 states, 4 actions, deterministic and fully known, so the best
 * achievable policy for ANY reward function is a few microseconds of arithmetic
 * away. That is what makes the reward-hacking verdict a proof instead of a guess:
 * if the optimal policy for the rewards the player wrote does not leave the maze,
 * then no amount of training, exploration or patience was ever going to.
 *
 * Terminal states have value 0 — the episode is over, nothing more is collected.
 */
export function solveOptimal(rewards: RewardConfig): OptimalPolicy {
  const values = new Array<number>(STATE_COUNT).fill(0);

  for (let sweep = 0; sweep < 2000; sweep += 1) {
    let delta = 0;
    for (let state = 0; state < STATE_COUNT; state += 1) {
      const position = positionOf(state);
      if (cellAt(position) === "wall" || isTerminal(state)) continue;

      let best = -Infinity;
      for (const action of ACTIONS) {
        const result = stepEnvironment(position, action, rewards);
        const candidate =
          result.reward +
          (result.done ? 0 : DISCOUNT * values[stateOf(result.next)]!);
        if (candidate > best) best = candidate;
      }

      delta = Math.max(delta, Math.abs(best - values[state]!));
      values[state] = best;
    }
    if (delta < 1e-9) break;
  }

  const policy = new Array<Action>(STATE_COUNT).fill(0);
  for (let state = 0; state < STATE_COUNT; state += 1) {
    const position = positionOf(state);
    if (cellAt(position) === "wall" || isTerminal(state)) continue;

    let best = -Infinity;
    let bestAction: Action = 0;
    for (const action of ACTIONS) {
      const result = stepEnvironment(position, action, rewards);
      const candidate =
        result.reward +
        (result.done ? 0 : DISCOUNT * values[stateOf(result.next)]!);
      if (candidate > best) {
        best = candidate;
        bestAction = action;
      }
    }
    policy[state] = bestAction;
  }

  // Follow it and see what it actually does.
  let position = { ...START };
  let episodeReward = 0;
  let steps = 0;
  let pelletsEaten = 0;
  let reachesGoal = false;
  let hitTrap = false;

  while (steps < MAX_STEPS) {
    const result = stepEnvironment(position, policy[stateOf(position)]!, rewards);
    steps += 1;
    episodeReward += result.reward;
    if (cellAt(result.next) === "pellet" && result.outcome === "moved") {
      pelletsEaten += 1;
    }
    position = result.next;
    if (result.done) {
      reachesGoal = result.outcome === "goal";
      hitTrap = result.outcome === "trap";
      break;
    }
  }

  const behaviour: OptimalBehaviour = reachesGoal
    ? "exit"
    : hitTrap
      ? "quit"
      : pelletsEaten >= 2
        ? "farm"
        : "wander";

  return {
    values,
    policy,
    reachesGoal,
    behaviour,
    valueAtStart: values[stateOf(START)]!,
    valueAtPellet: values[stateOf(PELLET)]!,
    episodeReward,
    steps,
    pelletsEaten,
  };
}

// ── Judging the academy ───────────────────────────────────────────────────

/** Goal rate the greedy policy must reach to count as competent. */
export const COMPETENCE_RATE = 0.9;
/** Exploration used for the reference run that isolates the search. */
export const REFERENCE_EPSILON = 0.4;
/** Epsilon at or below this is "barely exploring", for the diagnosis. */
export const LOW_EPSILON = 0.1;
/** Episodes the player buys per click, and the ceiling. */
export const EPISODE_BATCH = 100;
export const MAX_EPISODES = 1200;
/** Seed for the diagnostic reference runs. Fixed so a verdict is reproducible. */
export const DIAGNOSIS_SEED = 4242;
/**
 * Where the epsilon slider starts.
 *
 * Deliberately too low. Measured over twelve seeds, epsilon 0.05 solves the maze
 * 0 times out of 12 at 300 episodes and 1 out of 12 at 600, so a player who
 * changes nothing meets the exploration failure by default — which is the arc the
 * spec asks for ("tune epsilon to escape bad habits").
 */
export const DEFAULT_EPSILON = 0.05;

export type Outcome =
  | "competent"
  | "reward-hacking"
  | "no-exploration"
  | "under-trained"
  | "unlearnable-rewards"
  | "untrained";

export interface Evaluation {
  outcome: Outcome;
  report: PolicyReport;
  episodesUsed: number;
  score: number;
  failure: NamedFailure | null;
  /** The best policy the player's rewards admit. Always computed. */
  optimal: OptimalPolicy;
  /** Same rewards, same episode budget, generous exploration. */
  explorer: PolicyReport | null;
  /** Same rewards, same epsilon, the full episode budget. */
  patient: PolicyReport | null;
}

const percent = (value: number) => `${Math.round(value * 100)}%`;
const signed = (value: number) =>
  `${value >= 0 ? "+" : ""}${value.toFixed(1)}`;
/** "+3.0", but "nothing at all" for zero, which reads better mid-sentence. */
const paid = (value: number) => (value === 0 ? "nothing at all" : signed(value));

/**
 * Judge what the player's reward function and epsilon produced.
 *
 * The order is deliberate and it is not the obvious one.
 *
 * Reward hacking is checked FIRST and unconditionally, before we even ask whether
 * this agent reached the exit. If the best possible policy for the rewards the
 * player wrote does not leave the maze, then the specification is wrong, and an
 * agent that happens to leave anyway did so by luck — measured, one seed in twelve
 * does exactly that with a positive step reward. Letting that count as a win would
 * teach the opposite of the lesson.
 *
 * Then, for a run that failed, the question is which single change would have
 * fixed it, and the answer is measured rather than assumed:
 *
 *   `explorer` re-runs the same rewards with generous exploration on the SAME
 *     episode budget the player spent. If that reaches competence, the player's
 *     epsilon was the bottleneck — with no more experience than they already paid
 *     for, a bolder agent would have got there.
 *
 *   `patient` re-runs the same rewards at the player's OWN epsilon with the full
 *     budget. If that reaches competence, nothing is wrong except impatience.
 *
 * Trying `patient` before falling back to epsilon advice matters: at epsilon 0.15
 * and 100 episodes, both changes work, and "train a bit longer" is the smaller and
 * more honest correction. Epsilon only gets the blame when the player is genuinely
 * barely exploring.
 */
export function evaluateRun({
  qTable,
  rewards,
  epsilon,
  episodesUsed,
}: {
  qTable: QTable;
  rewards: RewardConfig;
  epsilon: number;
  episodesUsed: number;
}): Evaluation {
  const optimal = solveOptimal(rewards);
  const report = evaluatePolicy(qTable, rewards);

  if (episodesUsed === 0) {
    return {
      outcome: "untrained",
      report,
      episodesUsed,
      score: 0,
      failure: null,
      optimal,
      explorer: null,
      patient: null,
    };
  }

  // 1. Can these rewards be satisfied AT ALL? Arithmetic, not a guess, and asked
  //    before we give anybody credit for finishing.
  if (!optimal.reachesGoal) {
    return {
      outcome: "reward-hacking",
      report,
      episodesUsed,
      score: 0,
      failure: {
        name: "Reward hacking",
        detail: rewardHackingDetail(rewards, report, optimal),
      },
      optimal,
      explorer: null,
      patient: null,
    };
  }

  if (report.goalRate >= COMPETENCE_RATE) {
    // Fewer episodes is better, per the spec's "fewest episodes".
    const efficiency = clamp(1 - episodesUsed / MAX_EPISODES, 0, 1);
    return {
      outcome: "competent",
      report,
      episodesUsed,
      score: clamp(0.6 + 0.4 * efficiency, 0, 1),
      failure: null,
      optimal,
      explorer: null,
      patient: null,
    };
  }

  const progress = clamp(report.goalRate / COMPETENCE_RATE, 0, 1);
  const score = clamp(progress * 0.45, 0, 1);

  // 2. Would more exploration, on the budget already spent, have fixed it?
  const explorer = trainFresh(
    rewards,
    REFERENCE_EPSILON,
    Math.max(episodesUsed, EPISODE_BATCH),
    DIAGNOSIS_SEED,
  ).report;
  const explorerWorks = explorer.goalRate >= COMPETENCE_RATE;

  if (epsilon <= LOW_EPSILON && explorerWorks) {
    return {
      outcome: "no-exploration",
      report,
      episodesUsed,
      score,
      failure: {
        name: "No exploration",
        detail: noExplorationDetail(rewards, report, optimal, explorer, epsilon, episodesUsed),
      },
      optimal,
      explorer,
      patient: null,
    };
  }

  // 3. Would more episodes at their own epsilon have fixed it?
  const patient =
    episodesUsed < MAX_EPISODES
      ? trainFresh(rewards, epsilon, MAX_EPISODES, DIAGNOSIS_SEED).report
      : null;

  if (patient !== null && patient.goalRate >= COMPETENCE_RATE) {
    return {
      outcome: "under-trained",
      report,
      episodesUsed,
      score,
      failure: {
        name: "Not competent yet",
        detail: `${percent(report.goalRate)} of rollouts reach the exit, against the ${percent(
          COMPETENCE_RATE,
        )} needed — and nothing is wrong with the setup. The rewards are satisfiable: solved exactly, the best policy walks out in ${
          optimal.steps
        } steps for ${signed(
          optimal.episodeReward,
        )}. ε ${epsilon.toFixed(2)} is enough exploration too — the same rewards at the same ε reach ${percent(
          patient.goalRate,
        )} by ${MAX_EPISODES} episodes. ${episodesUsed} episodes is simply not enough experience yet for the exit's value to reach back along ${
          optimal.steps
        } cells to the start. ${
          report.trapRate >= 0.2
            ? `Watch the ${percent(
                report.trapRate,
              )} of rollouts ending in the pit as you go — that should fall as the values settle.`
            : report.meanPellets >= 2
              ? `Right now it is still pacing by the cheese, ${report.meanPellets.toFixed(
                  1,
                )} bites an episode. That habit breaks once the exit's value gets far enough back.`
              : `Train more.`
        }`,
      },
      optimal,
      explorer,
      patient,
    };
  }

  // 4. Epsilon above the "barely exploring" line, but still the bottleneck.
  if (explorerWorks) {
    return {
      outcome: "no-exploration",
      report,
      episodesUsed,
      score,
      failure: {
        name: "Not exploring enough",
        detail: `Your rewards are fine — solved exactly, the best policy walks out in ${
          optimal.steps
        } steps for ${signed(
          optimal.episodeReward,
        )}. And patience is not the answer either: at ε ${epsilon.toFixed(
          2,
        )} this reward function still only manages ${percent(
          patient?.goalRate ?? report.goalRate,
        )} by ${MAX_EPISODES} episodes. What does work is exploring harder — the same rewards on the same ${episodesUsed} episodes at ε ${REFERENCE_EPSILON} reach ${percent(
          explorer.goalRate,
        )}. ${
          report.meanPellets >= 2
            ? `At ε ${epsilon.toFixed(
                2,
              )} the agent takes the action it already believes in ${percent(
                1 - epsilon,
              )} of the time, and what it believes in is the cheese two steps from home — ${report.meanPellets.toFixed(
                1,
              )} bites an episode.`
            : `Exploration is not noise for its own sake: a Q-value the agent never revisits keeps whatever it was initialised to, forever.`
        }`,
      },
      optimal,
      explorer,
      patient,
    };
  }

  // 5. Satisfiable on paper, unreachable in practice.
  return {
    outcome: "unlearnable-rewards",
    report,
    episodesUsed,
    score,
    failure: unlearnableFailure(rewards, report, optimal, explorer),
    optimal,
    explorer,
    patient,
  };
}

function rewardHackingDetail(
  rewards: RewardConfig,
  report: PolicyReport,
  optimal: OptimalPolicy,
): string {
  const lucky = report.goalRate >= COMPETENCE_RATE;
  const opening = lucky
    ? `Your agent did reach the exit — and it should not have. It got lucky, and the rewards you wrote pay better for not bothering.`
    : `The agent is not broken and it is not under-trained. It is doing very nearly the best thing available, and the best thing available is not leaving the maze.`;

  const proof = `We can check that without any training at all: the maze is ${STATE_COUNT} states and four actions, so your reward function can be solved exactly. The best policy it admits is worth ${signed(
    optimal.valueAtStart,
  )} from the start, and it ${
    optimal.behaviour === "farm"
      ? `never goes near the exit. It eats the cheese ${optimal.pelletsEaten} times instead, for ${signed(
          optimal.episodeReward,
        )} an episode.`
      : optimal.behaviour === "quit"
        ? `walks straight into the pit, ${optimal.steps} steps in.`
        : `wanders for all ${optimal.steps} steps and never finishes.`
  }`;

  const positiveStep =
    rewards.step >= 0
      ? `Look at what you are paying per move: ${paid(
          rewards.step,
        )}. Moving is not a cost any more, so there is nothing for the agent to finish FOR. `
      : ``;

  const arithmetic =
    optimal.behaviour === "farm"
      ? `${positiveStep}At ${signed(rewards.pellet)} a bite against ${signed(
          rewards.step,
        )} a move, a round trip to the cheese clears ${signed(
          roundTripValue(rewards),
        )} — repeatable, forever. Pacing beside it is worth ${signed(
          farmValue(rewards),
        )}; the exit pays ${paid(
          rewards.goal,
        )} once, nine steps away, discounted to ${(DISCOUNT ** 9).toFixed(
          2,
        )} of face value. ${
          lucky
            ? `The reward curve would have climbed higher if it had never left.`
            : `Your agent worked this out: ${report.meanPellets.toFixed(
                1,
              )} bites an episode, ${percent(
                report.timeoutRate,
              )} of episodes never ending, ${signed(
                report.meanReward,
              )} an episode while doing it.`
        }`
      : optimal.behaviour === "quit"
        ? `At ${signed(rewards.step)} a move and ${paid(
            rewards.trap,
          )} for the pit, quitting two steps in costs less than the nine steps to the exit are worth. ${
            lucky
              ? ``
              : `Your agent agrees: ${percent(
                  report.trapRate,
                )} of rollouts end in the pit.`
          }`
        : `Nothing you are paying for happens at the exit, so the exit is not where the reward is.`;

  return `${opening} ${proof} ${arithmetic} More episodes will not help. More exploration will not help. Change what you are paying for.`;
}

function noExplorationDetail(
  rewards: RewardConfig,
  report: PolicyReport,
  optimal: OptimalPolicy,
  explorer: PolicyReport,
  epsilon: number,
  episodesUsed: number,
): string {
  const never = epsilon === 0;
  return `Your rewards are fine. Solved exactly, the best policy they admit walks out in ${
    optimal.steps
  } steps for ${signed(
    optimal.episodeReward,
  )} — and on the same ${episodesUsed} episodes you just spent, an agent at ε ${REFERENCE_EPSILON} reaches the exit ${percent(
    explorer.goalRate,
  )} of the time. Yours reaches it ${percent(report.goalRate)} of the time, because at ε ${epsilon.toFixed(
    2,
  )} it ${
    never
      ? `never once tries anything it does not already believe in. Not rarely — never.`
      : `almost never tries anything it does not already believe in.`
  } ${
    report.meanPellets >= 2
      ? `It found the cheese two steps from the start, learned that pacing beside it beats losing ${signed(
          rewards.step,
        )} a move going nowhere, and stopped looking — ${report.meanPellets.toFixed(
          1,
        )} bites an episode, ${percent(
          report.timeoutRate,
        )} of episodes never ending. The part worth sitting with: that habit is not even profitable. A round trip to the cheese clears exactly ${signed(
          roundTripValue(rewards),
        )}. Break-even is enough to trap it, because every alternative it has actually tried is worse, and it has not tried many.`
      : `It committed to the first thing that worked and never found out what else was out there.`
  } A Q-value the agent never revisits keeps whatever it was initialised to, forever${
    never ? `, and at ε 0 the set of values it revisits never grows` : ``
  }. Raise ε and train again.`;
}

function unlearnableFailure(
  rewards: RewardConfig,
  report: PolicyReport,
  optimal: OptimalPolicy,
  explorer: PolicyReport,
): NamedFailure {
  const quitting = report.trapRate >= 0.4 || explorer.trapRate >= 0.4;
  const farming = report.meanPellets >= 2 || explorer.meanPellets >= 2;

  const preamble = `This is the subtle one, and it is the most common way a reward function goes wrong in practice. Your rewards CAN be satisfied — solved exactly, the best policy walks out in ${
    optimal.steps
  } steps and is worth ${signed(
    optimal.valueAtStart,
  )} from the start. No agent gets there. Not with more episodes, and not with more exploration: at ε ${REFERENCE_EPSILON} a fresh agent still only finishes ${percent(
    explorer.goalRate,
  )} of the time.`;

  if (quitting) {
    return {
      name: "Quitting beats trying",
      detail: `${preamble} At ${signed(
        rewards.step,
      )} a move, finding the exit means spending nine moves — ${signed(
        9 * rewards.step,
      )} — on the chance of something the agent has no idea is there. Every exploratory step looks like a mistake while it is happening, and the pit two steps from home costs ${paid(
        rewards.trap,
      )}. So it learns the one thing it can see clearly: stop early. ${percent(
        report.trapRate,
      )} of rollouts end in the pit. Being achievable in principle is not the same as being findable. Make the journey less punishing, or make the pit cost more than the search does.`,
    };
  }

  if (farming) {
    return {
      name: "Local optimum",
      detail: `${preamble} The reason is two steps from the start. A round trip to the cheese clears ${signed(
        roundTripValue(rewards),
      )}, so pacing beside it forever is worth ${signed(
        farmValue(rewards),
      )} — a real, positive, reliable profit. Standing in that same square and heading for the exit instead is worth ${signed(
        optimal.valueAtPellet,
      )}, comfortably more. But the agent has no way to know that: it found the profitable habit in its first few episodes, and a policy that already pays will not spend episodes discovering that something better exists. ${report.meanPellets.toFixed(
        1,
      )} bites an episode, ${percent(
        report.timeoutRate,
      )} never ending. This is what people mean by a local optimum, and note that raising ε did not rescue it — once a habit pays, exploration has to survive being interrupted by it. Take the profit out of the loop.`,
    };
  }

  return {
    name: "Reward too distant to find",
    detail: `${preamble} Nothing on the way to the exit pays anything, so there is no gradient to follow — the agent has to stumble on the exit by pure chance before a single Q-value anywhere improves. ${percent(
      report.timeoutRate,
    )} of rollouts simply run out of time. Sparse rewards are the hardest case in reinforcement learning, and the usual fix is not a better agent: it is paying something for progress.`,
  };
}

// ── Reveal the math ───────────────────────────────────────────────────────

export const MATH_EQUATION = String.raw`Q(s,a) \leftarrow Q(s,a) + \alpha\Big[\underbrace{r + \gamma \max_{a'} Q(s',a')}_{\text{what we now think it is worth}} - \underbrace{Q(s,a)}_{\text{what we thought}}\Big]
\\[1.2em]
\pi(s) = \begin{cases} \text{random action} & \text{with probability } \varepsilon \\ \arg\max_a Q(s,a) & \text{otherwise} \end{cases}`;

export const MATH_CODE = `// The entire learning rule. Everything else is bookkeeping.
const bootstrap = done ? 0 : ${DISCOUNT} * Math.max(...qTable[nextState]);
qTable[state][action] +=
  ${LEARNING_RATE} * (reward + bootstrap - qTable[state][action]);

// Terminal transitions drop the bootstrap: there is no next state to look
// ahead to. Leaving it in is the classic way to make an agent believe the
// goal is worth vastly more than it is.

// And the policy — the only place exploration enters:
const action = Math.random() < epsilon
  ? randomAction()                    // try something. anything.
  : argmax(qTable[state]);            // do what you already believe`;

export const MATH_NOTES = `Q(s,a) is the agent's estimate of everything it will collect from here on, if it takes action a now and behaves greedily afterwards. The update nudges that estimate toward what it just observed — the reward it actually got, plus its own estimate of the state it landed in. Nothing in it knows what the maze looks like. It only knows what happened.

The discount γ = ${DISCOUNT} is why the agent prefers finishing sooner: a reward n steps away is worth γⁿ times its face value. The exit is nine steps off, so it keeps ${(
  DISCOUNT ** 9
).toFixed(2)} of what you pay for it, while the cheese two steps away keeps ${(
  DISCOUNT ** 2
).toFixed(
  2,
)}. That gap is the whole reason a nearby mediocre reward can out-compete a distant excellent one.

ε is the entire exploration mechanism, and it is doing something less obvious than "add noise". A Q-value that is never updated stays at whatever it was initialised to. An agent that only ever takes the action it currently believes in will keep confirming that belief and never learn what the alternatives were worth. Exploration is not there to add randomness for its own sake — it is there to generate the data that corrects the table.

Which brings up the part that catches everyone. This agent will find the highest-scoring behaviour available to it, and that is a promise, not a hope. If eating cheese repeatedly pays better than leaving the maze, it will eat cheese repeatedly, and the reward curve will climb beautifully while it does. Nothing has gone wrong with the learning. The reward function said that was the best available behaviour, and it was right.

That is also why this game can tell you which mistake you made rather than guessing. The maze is ${STATE_COUNT} states and four actions, all known, so your reward function can be solved exactly by value iteration — no learning, no randomness, no epsilon. If the best possible policy for your rewards does not leave the maze, the problem is the reward function and it is provable. If it does leave, and your agent doesn't, the problem is the search.`;
