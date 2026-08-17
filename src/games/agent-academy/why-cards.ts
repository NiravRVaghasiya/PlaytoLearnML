import type { WhyCardContent } from "@/components";
import {
  COMPETENCE_RATE,
  DEFAULT_EPSILON,
  DISCOUNT,
  EPISODE_BATCH,
  LOW_EPSILON,
  MAX_EPISODES,
  MAX_STEPS,
  REFERENCE_EPSILON,
  farmValue,
  roundTripValue,
  type EpisodeRecord,
  type Evaluation,
  type OptimalPolicy,
  type RewardConfig,
} from "./ml";

/**
 * "Why did that happen?" copy for Agent Academy.
 *
 * The rule this copy follows: never say "the agent got stuck" when the real
 * sentence is "the agent found something that pays and stopped looking". The
 * first is a description of a symptom the grid already shows; the second is the
 * thing that transfers to every RL problem the player ever meets.
 *
 * It also has to be careful about a distinction the game rests on. There are two
 * completely different reasons a run fails — the rewards asked for the wrong
 * thing, or the search never found what they asked for — and the copy is only
 * allowed to claim the first when value iteration has proved it. "It is behaving
 * optimally" is a strong claim; it gets made when it is true and not otherwise.
 */
export type AcademyEvent =
  | { kind: "briefing" }
  | {
      kind: "reward-changed";
      keys: Array<keyof RewardConfig>;
      rewards: RewardConfig;
      optimal: OptimalPolicy;
      hadEpisodes: number;
    }
  | {
      kind: "epsilon-changed";
      epsilon: number;
      previous: number;
      hadEpisodes: number;
    }
  | {
      kind: "trained";
      evaluation: Evaluation;
      batch: EpisodeRecord[];
      episodesUsed: number;
      epsilon: number;
      rewards: RewardConfig;
    }
  | { kind: "forgot" };

const percent = (value: number) => `${Math.round(value * 100)}%`;
const signed = (value: number) =>
  `${value >= 0 ? "+" : ""}${value.toFixed(1)}`;

const KNOB_NAMES: Record<keyof RewardConfig, string> = {
  goal: "the exit payout",
  trap: "the pit penalty",
  step: "the cost per move",
  pellet: "the cheese",
  bump: "the wall penalty",
};

export function whyCardFor(event: AcademyEvent): WhyCardContent {
  switch (event.kind) {
    case "briefing":
      return {
        key: "briefing",
        title: "You do not get to move the agent",
        body: `You write what it is paid, and you decide how often it tries something new. That is the entire interface. It will then find the highest-scoring behaviour those two choices allow — which is a promise, not a hope, and it is the reason this game is hard. Two things worth knowing before you start: the cheese is two steps away and the exit is nine, and ε starts at ${DEFAULT_EPSILON}, which is not enough.`,
        tone: "info",
      };

    case "reward-changed": {
      const { keys, optimal, hadEpisodes } = event;
      const names = keys.map((key) => KNOB_NAMES[key]).join(" and ");
      const forgot =
        hadEpisodes > 0
          ? `The ${hadEpisodes} episodes of experience are gone, and they had to be: a Q-value is an estimate of future reward under a particular reward function, so changing the rewards makes every number in that table an answer to a question nobody is asking any more. `
          : "";

      if (!optimal.reachesGoal) {
        return {
          key: `reward-broken-${keys.join("-")}-${optimal.valueAtStart.toFixed(2)}`,
          title: `Those rewards no longer ask for the exit`,
          body: `${forgot}You can check this before spending a single episode, and it is worth doing: solved exactly, the best policy your rewards now admit ${
            optimal.behaviour === "farm"
              ? `eats the cheese ${optimal.pelletsEaten} times and never leaves`
              : optimal.behaviour === "quit"
                ? `walks into the pit`
                : `wanders until the episode times out`
          }, for ${signed(
            optimal.valueAtStart,
          )} from the start. Train if you like — a well-behaved agent will now do exactly that.`,
          tone: "bad",
        };
      }

      return {
        key: `reward-${keys.join("-")}-${optimal.valueAtStart.toFixed(2)}`,
        title: `Changed ${names}`,
        body: `${forgot}Solved exactly, the best policy these rewards admit still walks out in ${
          optimal.steps
        } steps for ${signed(optimal.episodeReward)} — so the exit is still what you are asking for. Whether an agent can FIND it is a separate question, and the only way to answer it is to train.`,
        tone: "info",
      };
    }

    case "epsilon-changed": {
      const { epsilon, previous, hadEpisodes } = event;
      const raised = epsilon > previous;
      const forgot =
        hadEpisodes > 0
          ? `Those ${hadEpisodes} episodes are gone — each run is one clean experiment, so the comparison you are about to make means something. `
          : "";

      if (epsilon === 0) {
        return {
          key: "epsilon-zero",
          title: "ε 0 — pure exploitation",
          body: `${forgot}The agent will now only ever take the action it currently believes is best. Nothing else. Everything it has not tried keeps the value it was initialised with, which is 0, forever — because the only way a Q-value changes is if the action gets taken. Worth running, so the failure is yours to look at rather than mine to assert.`,
          tone: "warn",
        };
      }

      return {
        key: `epsilon-${epsilon}`,
        title: `ε ${epsilon.toFixed(2)} — ${percent(
          epsilon,
        )} of moves will be random`,
        body: `${forgot}${
          raised ? "Raised" : "Lowered"
        } from ${previous.toFixed(2)}. The other ${percent(
          1 - epsilon,
        )} of the time it does whatever it already believes in. ${
          epsilon <= LOW_EPSILON
            ? `That is not much room to discover a reward nine steps away, especially with something that already pays two steps from home.`
            : `Those random moves are not noise for its own sake — they are the only way the table gets data about actions the agent does not already favour.`
        }`,
        tone: epsilon <= LOW_EPSILON ? "warn" : "info",
      };
    }

    case "forgot":
      return {
        key: "forgot",
        title: "Agent reset, reward function kept",
        body: `Empty Q-table, same rules. Useful for checking that a result was the reward function talking and not one lucky run.`,
        tone: "info",
      };

    case "trained": {
      const { evaluation, batch, episodesUsed, epsilon, rewards } = event;
      const { report, optimal, outcome } = evaluation;

      const goals = batch.filter((record) => record.ending === "goal").length;
      const traps = batch.filter((record) => record.ending === "trap").length;
      const first = batch[0]?.reward ?? 0;
      const last = batch.at(-1)?.reward ?? 0;

      if (outcome === "competent") {
        return {
          key: `graduated-${episodesUsed}`,
          title: `Graduated in ${episodesUsed} episodes`,
          body: `The greedy policy reaches the exit in ${percent(
            report.goalRate,
          )} of rollouts, ${report.meanSteps.toFixed(
            1,
          )} steps a run, ${signed(
            report.meanReward,
          )} an episode. Compare that with what your rewards actually asked for: solved exactly, the best possible policy takes ${
            optimal.steps
          } steps for ${signed(
            optimal.episodeReward,
          )}. Your agent got there by trial and error, from a table of zeros, with no idea the exit existed until it fell into it. Nothing in it ever saw the maze.`,
          tone: "good",
        };
      }

      if (outcome === "reward-hacking") {
        return {
          key: `hacking-${episodesUsed}-${optimal.valueAtStart.toFixed(2)}`,
          title:
            optimal.behaviour === "quit"
              ? "It learned that quitting pays"
              : "It is doing exactly what you paid for",
          body: `${signed(
            report.meanReward,
          )} an episode and ${percent(
            report.goalRate,
          )} of rollouts reaching the exit. Those two numbers together are the whole lesson: the reward went UP and the task went unfinished, and the agent is not confused about that. ${
            optimal.behaviour === "farm"
              ? `A round trip to the cheese clears ${signed(
                  roundTripValue(rewards),
                )}. Repeat that forever and it is worth ${signed(
                  farmValue(rewards),
                )} — more than the exit, which pays once and is nine steps away.`
              : optimal.behaviour === "quit"
                ? `The pit is two steps from home and the exit is nine, and you have priced them so the pit wins.`
                : `Nothing you are paying for happens at the exit.`
          } This is what reward hacking looks like from the inside, and it is why "the metric went up" is not the same as "the thing I wanted happened".`,
          tone: "bad",
        };
      }

      if (outcome === "no-exploration") {
        return {
          key: `stuck-${episodesUsed}-${epsilon}`,
          title:
            epsilon === 0
              ? "It never tried anything new, so it never found anything new"
              : `ε ${epsilon.toFixed(2)} is not enough room to search`,
          body: `${episodesUsed} episodes, ${percent(
            report.goalRate,
          )} of rollouts reaching the exit, ${report.meanPellets.toFixed(
            1,
          )} bites of cheese a run. Look at the reward curve: it is flat, not falling. The agent is not getting worse — it settled early and has been repeating itself since. On the same ${episodesUsed} episodes, ε ${REFERENCE_EPSILON} finishes ${percent(
            evaluation.explorer?.goalRate ?? 1,
          )} of the time, so the rewards were never the problem here.`,
          tone: "warn",
        };
      }

      if (outcome === "unlearnable-rewards") {
        return {
          key: `unlearnable-${episodesUsed}`,
          title: "Achievable on paper, unreachable in practice",
          body: `Your rewards do ask for the exit — solved exactly, the best policy walks out in ${
            optimal.steps
          } steps for ${signed(
            optimal.episodeReward,
          )}. But no agent finds it, at any ε. ${
            report.trapRate >= 0.4
              ? `${percent(
                  report.trapRate,
                )} of rollouts end in the pit instead: at ${signed(
                  rewards.step,
                )} a move, searching costs more than stopping.`
              : `${report.meanPellets.toFixed(
                  1,
                )} bites of cheese a run instead: the habit two steps from home pays a real profit, and a policy that already pays will not go looking for a better one.`
          } "Correct" and "learnable" are different properties of a reward function, and only one of them can be checked with arithmetic.`,
          tone: "bad",
        };
      }

      // Still training, nothing diagnosably wrong.
      const climbing = last > first;
      return {
        key: `progress-${episodesUsed}`,
        title: `${episodesUsed} episodes · ${goals} of ${batch.length} reached the exit`,
        body: `Episode reward went from ${signed(first)} to ${signed(
          last,
        )} across this batch${
          climbing ? "" : ", which is not yet a trend"
        }. The greedy policy manages ${percent(
          report.goalRate,
        )} against the ${percent(COMPETENCE_RATE)} needed${
          traps > 0 ? `, and ${traps} episodes ended in the pit` : ""
        }. What has to happen next is mechanical: the exit's value has to travel back one cell per successful visit, ${
          optimal.steps
        } cells to the start, discounted by ${DISCOUNT} each hop. ${
          episodesUsed >= MAX_EPISODES
            ? `That is the last of the ${MAX_EPISODES} episodes, though — change something.`
            : `Another ${EPISODE_BATCH} episodes will move it.`
        }`,
        tone: "info",
      };
    }
  }
}

/** Caption under the grid: what the trail on screen actually is. */
export function trailCaption(
  episodesUsed: number,
  report: { goalRate: number; meanSteps: number },
): string {
  if (episodesUsed === 0) {
    return `Untrained. Every Q-value is 0, so the agent has no preference at all and the walk below is what breaking a four-way tie at random looks like.`;
  }
  if (report.goalRate >= COMPETENCE_RATE) {
    return `The greedy policy after ${episodesUsed} episodes: no exploration, just whatever the table now believes.`;
  }
  return `The greedy policy after ${episodesUsed} episodes. It stops where the walk starts repeating itself — a loop is drawn once, and ${
    report.meanSteps >= MAX_STEPS ? "this one never ends" : "this one ends early"
  }.`;
}
