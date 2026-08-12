import type { WhyCardContent } from "@/components";
import {
  BAYESIAN_SEED_TRIALS,
  BUDGET,
  CRACK_THRESHOLD,
  DIALS,
  DIAL_COUNT,
  LEARNING_RATE,
  bestTrial,
  describePoint,
  dialInfluence,
  distinctValuesTried,
  gridLevels,
  temperatureOf,
  type Strategy,
  type Trial,
} from "./ml";

/**
 * "Why did that happen?" copy for Hyperparameter Heist.
 *
 * The rule this copy follows: never praise or blame a single try. One try landing
 * hot is luck; what the player needs feedback on is COVERAGE — how much of each
 * dial they have actually seen. So every card that can reports distinct values
 * tried, because that is the quantity the lesson is about and the one nobody
 * tracks by instinct.
 *
 * It also never calls grid search stupid. A grid is the right tool when you have
 * two dials and budget to spare, and the reason it fails here is specific and
 * arithmetical: k^d grows faster than any budget you will be given.
 */
export type HeistEvent =
  | { kind: "briefing" }
  | {
      kind: "strategy-changed";
      strategy: Strategy | "manual";
      budget: number;
      triesLeft: number;
    }
  | {
      kind: "tried";
      trial: Trial;
      previousBest: number;
      triesLeft: number;
      cracked: boolean;
      finished: boolean;
      trials: Trial[];
      budget: number;
    }
  | {
      kind: "strategy-run";
      strategy: Strategy;
      trials: Trial[];
      added: Trial[];
      budget: number;
      cracked: boolean;
    };

const percent = (value: number) => `${Math.round(value * 100)}%`;
const points = (value: number) => `${(value * 100).toFixed(1)}%`;

const TEMPERATURE_WORDS: Record<string, string> = {
  freezing: "freezing",
  cold: "cold",
  warm: "warm",
  hot: "hot",
  cracked: "open",
};

export function whyCardFor(event: HeistEvent): WhyCardContent {
  switch (event.kind) {
    case "briefing":
      return {
        key: "briefing",
        title: `${DIAL_COUNT} dials, ${BUDGET} tries`,
        body: `The safe opens at ${percent(
          CRACK_THRESHOLD,
        )}. You cannot see the objective surface — nobody tuning a real model can — so every try buys you one reading and nothing else. Before you start guessing, note the arithmetic: a full grid over ${DIAL_COUNT} dials at just ${gridLevels(
          BUDGET,
        )} settings each already costs ${gridLevels(BUDGET) ** DIAL_COUNT} tries. Thoroughness is not free, and in ${DIAL_COUNT} dimensions it is barely affordable.`,
        tone: "info",
      };

    case "strategy-changed": {
      const { strategy, budget, triesLeft } = event;
      if (strategy === "manual") {
        return {
          key: "strategy-manual",
          title: "Cracking by hand",
          body: `Your dials, your choices. Worth doing at least a few of these yourself before handing it to a strategy — the point of the exercise is to feel how little a single reading tells you, and how quickly the tries run out.`,
          tone: "info",
        };
      }

      if (strategy === "grid") {
        const levels = gridLevels(budget);
        return {
          key: "strategy-grid",
          title: `Grid: ${levels} settings per dial, ${levels ** DIAL_COUNT} tries`,
          body: `A full factorial is the thorough option and that is exactly its problem. ${levels}^${DIAL_COUNT} is ${
            levels ** DIAL_COUNT
          }, so the whole budget goes on a grid that samples the ${
            DIALS[LEARNING_RATE]!.name
          } at ${levels} values — the same ${levels} values, over and over, while the other dials cycle around it. ${triesLeft} tries left. Run it and see what it finds.`,
          tone: "warn",
        };
      }

      if (strategy === "random") {
        return {
          key: "strategy-random",
          title: `Random: ${triesLeft} independent draws`,
          body: `Every try samples every dial at a fresh value, so ${triesLeft} tries means up to ${triesLeft} different learning rates rather than ${gridLevels(
            budget,
          )}. That is the whole of the advantage, and it is a big one: the budget goes into resolution instead of into covering combinations of dials that do not matter.`,
          tone: "info",
        };
      }

      return {
        key: "strategy-bayesian",
        title: "Bayesian: model the safe, then pick",
        body: `This fits a Gaussian process to the readings you already have and then tries wherever Expected Improvement is highest — balancing "the model thinks this scores well" against "the model has never looked here". The first ${BAYESIAN_SEED_TRIALS} tries are spread out on purpose, because a surrogate fitted to two points is an expensive random guess. Under a tight budget you are partly paying for that warm-up.`,
        tone: "info",
      };
    }

    case "tried": {
      const {
        trial,
        previousBest,
        triesLeft,
        cracked,
        finished,
        trials,
        budget,
      } = event;
      const temperature = temperatureOf(trial.objectiveValue);
      const improved = trial.objectiveValue > previousBest;
      const lrSeen = distinctValuesTried(trials, LEARNING_RATE);

      if (cracked) {
        return {
          key: `cracked-${trial.index}`,
          title: `Open, on try ${trial.index} of ${budget}`,
          body: `${points(trial.objectiveValue)} at ${describePoint(
            trial.params,
          )}. You got there having sampled ${lrSeen} distinct ${
            DIALS[LEARNING_RATE]!.name
          } value${lrSeen === 1 ? "" : "s"} — a grid would have shown you ${gridLevels(
            budget,
          )} for the same spend. The surface is on show now: have a look at how much of it you never touched, and how little of it was worth touching.`,
          tone: "good",
        };
      }

      if (finished) {
        return {
          key: "out-of-tries",
          title: "Out of tries",
          body: `${points(
            trial.objectiveValue,
          )} on the last one. The true surface is revealed now — the question worth asking is not whether you were unlucky but whether your ${budget} readings were spread across the dial that mattered. You sampled ${lrSeen} ${
            DIALS[LEARNING_RATE]!.name
          } value${lrSeen === 1 ? "" : "s"}.`,
          tone: "bad",
        };
      }

      return {
        key: `try-${trial.index}-${trial.objectiveValue.toFixed(4)}`,
        title: `${points(trial.objectiveValue)} — ${TEMPERATURE_WORDS[temperature]}`,
        body: `${describePoint(trial.params)}. ${
          improved
            ? `Your best so far, and ${points(
                trial.objectiveValue - Math.max(previousBest, 0),
              )} better than the previous one.`
            : `Colder than your best of ${points(previousBest)} — which is still information: it rules out a region.`
        } ${triesLeft} tries left, and you have seen ${lrSeen} distinct ${
          DIALS[LEARNING_RATE]!.name
        } value${lrSeen === 1 ? "" : "s"} so far. ${
          temperature === "hot"
            ? "Hot. Something nearby is worth a try — but do not forget the dials you have barely moved."
            : temperature === "freezing"
              ? "Freezing. A reading this cold usually means the learning rate is orders of magnitude off, not that the other dials are wrong."
              : "Keep track of coverage rather than of near misses."
        }`,
        tone: temperature === "hot" ? "good" : "info",
      };
    }

    case "strategy-run": {
      const { strategy, trials, added, budget, cracked } = event;
      const best = bestTrial(trials)!;
      const lrSeen = distinctValuesTried(trials, LEARNING_RATE);
      const influence = dialInfluence();

      if (strategy === "grid") {
        return {
          key: "grid-run",
          title: `Grid spent ${added.length} tries and saw ${lrSeen} learning rates`,
          body: `Best ${points(best.objectiveValue)}${
            cracked ? " — open." : `, against the ${percent(CRACK_THRESHOLD)} needed.`
          } That is the grid's whole story: ${budget} readings, ${lrSeen} distinct values of the dial that controls ${percent(
            influence[LEARNING_RATE] ?? 0,
          )} of this safe's range. It also spent ${
            budget / Math.max(1, gridLevels(budget))
          } tries varying ${DIALS[2]!.name} and ${
            DIALS[3]!.name
          }, which between them move the objective by under ${percent(
            (influence[2] ?? 0) + (influence[3] ?? 0),
          )}. A full factorial cannot choose to skip those — varying everything equally is what makes it a grid.`,
          tone: cracked ? "good" : "bad",
        };
      }

      if (strategy === "random") {
        return {
          key: "random-run",
          title: `Random spent ${added.length} tries and saw ${lrSeen} learning rates`,
          body: `Best ${points(best.objectiveValue)}${
            cracked ? " — open." : `, against ${percent(CRACK_THRESHOLD)}.`
          } Same budget as the grid, ${
            lrSeen > gridLevels(budget)
              ? `${(lrSeen / Math.max(1, gridLevels(budget))).toFixed(0)}× the resolution`
              : "different coverage"
          } on the dial that decides the answer. Nothing clever happened here — no draw knew about any other. It wins because it stops spending budget on combinations of dials that were never going to matter.`,
          tone: cracked ? "good" : "warn",
        };
      }

      return {
        key: "bayesian-run",
        title: `Bayesian spent ${added.length} tries, ${
          added.length - BAYESIAN_SEED_TRIALS > 0
            ? `${Math.max(0, added.length - BAYESIAN_SEED_TRIALS)} of them modelled`
            : "all of them on warm-up"
        }`,
        body: `Best ${points(best.objectiveValue)}${
          cracked ? " — open." : `, against ${percent(CRACK_THRESHOLD)}.`
        } Watch where the tries went: clustered where the surrogate expected improvement, sparse where earlier readings had already ruled the region out. That is the difference from random — not better luck on any one try, but never spending a try twice on the same conclusion.`,
        tone: cracked ? "good" : "warn",
      };
    }
  }
}
