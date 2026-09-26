import type { WhyCardContent } from "@/components";
import {
  BATCH_SIZE,
  BAYESIAN_SEED_TRIALS,
  BUDGET,
  CRACK_RATE_RUNS,
  CRACK_THRESHOLD,
  DIALS,
  DIAL_COUNT,
  LEARNING_RATE,
  WEIGHT_DECAY,
  bestTrial,
  describePoint,
  dialInfluence,
  distinctValuesTried,
  gridBest,
  gridLevels,
  randomCrackRate,
  temperatureOf,
  type Strategy,
  type Trial,
} from "./ml";

/**
 * The Concept Library page this game's coverage lesson belongs to. Its section
 * "Why learning rate dominates" is about exactly this safe: a log-scaled dial
 * that decides the run, and a grid that sees too few values of it.
 */
const LEARNING_RATE_HREF = "/concepts/learning-rate";

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
/** "4×" for a whole ratio, "3.5×" otherwise — never rounded into a bigger claim. */
const ratioText = (ratio: number) =>
  `${Number.isInteger(ratio) ? ratio : ratio.toFixed(1)}×`;
/** An objective difference, in accuracy points — see `swing` in ml.ts. */
const swing = (value: number) => `${Math.round(value * 100)} points`;
/** "1 try", "3 tries" — a strategy that stops at the crack often spends only one. */
const tries = (count: number) => `${count} ${count === 1 ? "try" : "tries"}`;
const learningRates = (count: number) =>
  `${count} learning rate${count === 1 ? "" : "s"}`;

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
          conceptHref: LEARNING_RATE_HREF,
          conceptLabel: "Why learning rate dominates",
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
          conceptHref: LEARNING_RATE_HREF,
          conceptLabel: "Why learning rate dominates",
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
        // A freezing reading is the learning-rate lesson in miniature, so that
        // card is the one that points at the concept page.
        ...(temperature === "freezing"
          ? {
              conceptHref: LEARNING_RATE_HREF,
              conceptLabel: "Why learning rate dominates",
            }
          : {}),
      };
    }

    case "strategy-run": {
      const { strategy, trials, added, budget, cracked } = event;
      const best = bestTrial(trials)!;
      const lrSeen = distinctValuesTried(trials, LEARNING_RATE);
      const influence = dialInfluence();
      const levels = gridLevels(budget);
      const spare = budget - trials.length;
      const opened = cracked ? ` — open, with ${tries(spare)} to spare.` : "";

      if (strategy === "grid") {
        return {
          key: "grid-run",
          title: `Grid spent ${tries(added.length)} and saw ${learningRates(lrSeen)}`,
          body: `Best ${points(best.objectiveValue)}${
            cracked ? opened : `, against the ${percent(CRACK_THRESHOLD)} needed.`
          } That is the grid's whole story: ${budget} readings, ${lrSeen} distinct values of the dial that swings this safe's objective by ${swing(
            influence[LEARNING_RATE] ?? 0,
          )} on its own. Every one of those tries also had to vary ${
            DIALS[BATCH_SIZE]!.name
          } and ${
            DIALS[WEIGHT_DECAY]!.name
          }, which between them move the objective by only ${swing(
            (influence[BATCH_SIZE] ?? 0) + (influence[WEIGHT_DECAY] ?? 0),
          )}. A full factorial cannot choose to skip those — varying everything equally is what makes it a grid.`,
          tone: cracked ? "good" : "bad",
          conceptHref: LEARNING_RATE_HREF,
          conceptLabel: "Why learning rate dominates",
        };
      }

      if (strategy === "random") {
        // Judged against the numbers, never assumed. A single random draw loses
        // to the grid a real fraction of the time, and a card that said "it wins"
        // over a run that visibly lost would be teaching the reverse of its own
        // lesson — that one run is evidence of anything.
        const grid = gridBest(budget);
        const gridCracks = grid >= CRACK_THRESHOLD;
        const resolution =
          lrSeen > levels
            ? `${ratioText(lrSeen / Math.max(1, levels))} the grid's resolution`
            : "no more resolution than the grid";
        const rate = randomCrackRate(budget);
        const title = `Random spent ${tries(added.length)} and saw ${learningRates(lrSeen)}`;
        const onAverage = `over ${CRACK_RATE_RUNS} simulated from-scratch random runs on this budget, ${percent(
          rate,
        )} open the safe${gridCracks ? "" : ", and a full grid never does"}`;

        // A run stops at the crack, so a draw can open the safe on its first
        // try or two — before it has seen more learning rates than the grid.
        // That is luck, and the card says so instead of crediting the strategy.
        if (cracked && lrSeen <= levels) {
          return {
            key: "random-run-early",
            title,
            body: `Best ${points(best.objectiveValue)}${opened} It opened on try ${
              best.index
            }, having seen ${learningRates(lrSeen)} — no more than the ${levels} a full grid sees in all ${budget} tries — so coverage had no time to matter. That was a lucky draw, not the strategy paying off: ${onAverage}. One run of anything is luck, which is why the code lane settles the comparison over many seeds rather than on one draw.`,
            tone: "good",
            conceptHref: LEARNING_RATE_HREF,
            conceptLabel: "Why learning rate dominates",
          };
        }

        if (cracked) {
          return {
            key: "random-run",
            title,
            body: `Best ${points(best.objectiveValue)}${opened} ${
              gridCracks
                ? `The grid opens it too, but only after spending its whole budget.`
                : `A full grid spends all ${budget} and tops out at ${points(grid)}, having seen ${learningRates(levels)}.`
            } That is ${resolution} on the dial that decides the answer. Nothing clever happened here — no draw knew about any other. It wins because it stops spending budget on combinations of dials that were never going to matter.`,
            tone: "good",
            conceptHref: LEARNING_RATE_HREF,
            conceptLabel: "Why learning rate dominates",
          };
        }

        return {
          key: "random-run-unlucky",
          title,
          body: `Best ${points(best.objectiveValue)}, against ${percent(
            CRACK_THRESHOLD,
          )} — ${
            best.objectiveValue > grid
              ? `still above the grid's best of ${points(grid)}`
              : `below the grid's best of ${points(grid)} this time`
          }, with ${resolution}. This draw was unlucky: ${onAverage}. One run of anything is luck, which is why the code lane settles the comparison over many seeds rather than on one draw.`,
          tone: "warn",
          conceptHref: LEARNING_RATE_HREF,
          conceptLabel: "Why learning rate dominates",
        };
      }

      // Counted by try number, not by how many this batch added: after some
      // manual tries, part of the warm-up has already been spent.
      const modelled = added.filter(
        (trial) => trial.index > BAYESIAN_SEED_TRIALS,
      ).length;
      return {
        key: "bayesian-run",
        title: `Bayesian spent ${tries(added.length)}, ${
          modelled > 0
            ? `${modelled} of them modelled`
            : added.length === 1
              ? "on warm-up"
              : "all of them on warm-up"
        }`,
        body: `Best ${points(best.objectiveValue)}${
          cracked ? opened : `, against ${percent(CRACK_THRESHOLD)}.`
        } ${
          cracked && modelled > 0 && modelled <= 2
            ? `The first ${BAYESIAN_SEED_TRIALS} readings were enough for the surrogate to point at the right region: it opened the safe on modelled try ${modelled}. `
            : "Watch where the tries went: clustered where the surrogate expected improvement, sparse where earlier readings had already ruled the region out. "
        }That is the difference from random — not better luck on any one try, but never spending a try twice on the same conclusion.`,
        tone: cracked ? "good" : "warn",
      };
    }
  }
}
