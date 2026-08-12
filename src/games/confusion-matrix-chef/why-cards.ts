import type { WhyCardContent } from "@/components";
import {
  METRIC_LABELS,
  SCENARIOS,
  confusionAt,
  majorityBaseline,
  metricsOf,
  type RoundResult,
  type Sample,
  type Scenario,
} from "./ml";

/**
 * "Why did that happen?" copy for Confusion Matrix Chef.
 *
 * The rule this copy follows: never report a metric moving without naming the
 * cell of the matrix that paid for it. "Precision went up" is a fact a player can
 * read off the screen; "precision went up because eleven cases left the flagged
 * pile and nine of them were false alarms" is the thing they cannot see and the
 * only version that transfers.
 *
 * It also never calls a metric "better". Higher recall is better on the allergen
 * shift and actively wrong on the prank shift, and the whole game is that the
 * direction of "better" is set by the cost of each error, not by the metric.
 */
export type ChefEvent =
  | { kind: "shift-briefing"; scenario: Scenario }
  | {
      kind: "threshold-moved";
      scenario: Scenario;
      samples: Sample[];
      threshold: number;
      previous: number;
    }
  | {
      kind: "served";
      scenario: Scenario;
      result: RoundResult;
      attempts: number;
      complete: boolean;
    };

const percent = (value: number) => `${Math.round(value * 100)}%`;

export function whyCardFor(event: ChefEvent): WhyCardContent {
  switch (event.kind) {
    case "shift-briefing": {
      const { scenario } = event;
      const baseline = majorityBaseline(scenario.prevalence);
      return {
        key: `brief-${scenario.id}`,
        title: `Shift ${scenario.index} of ${SCENARIOS.length}: ${scenario.name}`,
        body: `${scenario.brief} The scores are already computed — nothing retrains when you move the slider. All you are choosing is where to cut, and that choice is not a modelling question, it is a question about which mistake you would rather make. ${
          baseline >= 0.8
            ? `Watch accuracy on this one: ${percent(
                baseline,
              )} of these cases are negative, so accuracy will look respectable no matter how badly you do at the part that matters.`
            : `The classes are close to balanced here, so accuracy is not lying to you this time — but it still cannot tell you which error you are making.`
        }`,
        tone: "info",
      };
    }

    case "threshold-moved": {
      const { scenario, samples, threshold, previous } = event;
      const now = confusionAt(samples, threshold);
      const before = confusionAt(samples, previous);
      const metrics = metricsOf(now);

      const flaggedDelta =
        now.truePositives + now.falsePositives -
        (before.truePositives + before.falsePositives);
      const caughtDelta = now.truePositives - before.truePositives;
      const alarmDelta = now.falsePositives - before.falsePositives;

      // Nothing crossed the cutoff: worth saying so rather than implying change.
      if (flaggedDelta === 0) {
        return {
          key: `still-${threshold.toFixed(2)}`,
          title: `Cutoff ${threshold.toFixed(2)} — nothing crossed`,
          body: `No case has a score between ${Math.min(
            previous,
            threshold,
          ).toFixed(2)} and ${Math.max(previous, threshold).toFixed(
            2,
          )}, so the matrix is unchanged. Thresholds only matter where cases actually sit; a gap in the score distribution is a range where the decision is stable.`,
          tone: "info",
        };
      }

      const loosened = threshold < previous;
      return {
        key: `moved-${threshold.toFixed(2)}`,
        title: `Cutoff ${threshold.toFixed(2)} — ${
          loosened ? "flagging" : "releasing"
        } ${Math.abs(flaggedDelta)} more case${
          Math.abs(flaggedDelta) === 1 ? "" : "s"
        }`,
        body: loosened
          ? `${Math.abs(flaggedDelta)} more case${
              Math.abs(flaggedDelta) === 1 ? "" : "s"
            } crossed into the flagged pile: ${caughtDelta} of them ${
              caughtDelta === 1 ? "was" : "were"
            } a real ${scenario.positiveLabel} and ${alarmDelta} ${
              alarmDelta === 1 ? "was" : "were"
            } a false alarm. Recall is now ${percent(
              metrics.recall,
            )} and precision ${percent(
              metrics.precision,
            )}. Every case you add to the pile helps recall and can only hurt precision — that is not a coincidence, they share a numerator and differ in the denominator.`
          : `${Math.abs(flaggedDelta)} case${
              Math.abs(flaggedDelta) === 1 ? "" : "s"
            } left the flagged pile: ${Math.abs(
              alarmDelta,
            )} false alarm${Math.abs(alarmDelta) === 1 ? "" : "s"} you no longer raise, and ${Math.abs(
              caughtDelta,
            )} real ${scenario.positiveLabel}${
              Math.abs(caughtDelta) === 1 ? "" : "s"
            } you now miss. Precision is ${percent(
              metrics.precision,
            )} and recall ${percent(
              metrics.recall,
            )}. Accuracy meanwhile reads ${percent(
              metrics.accuracy,
            )}, which on this shift ${
              majorityBaseline(scenario.prevalence) >= 0.8
                ? "tells you almost nothing"
                : "is at least roughly honest"
            }.`,
        tone: "info",
      };
    }

    case "served": {
      const { scenario, result, attempts, complete } = event;

      if (complete) {
        return {
          key: "all-clear",
          title: "Every shift signed off",
          body: `Four briefs, four different cutoffs. Notice that no single threshold would have passed all four: the prank shift needed one near ${"0.7"}–${"0.8"} to protect paying customers, and the allergen shift needed one near ${"0.4"} because a miss put someone in hospital. The model never changed. What changed was the cost of being wrong, and that is the thing you cannot read off a validation score.`,
          tone: "good",
        };
      }

      if (result.outcome === "win") {
        const metricLines = result.constraints
          .map(
            (constraint) =>
              `${METRIC_LABELS[constraint.metric].toLowerCase()} ${percent(
                constraint.achieved,
              )} against a ${percent(constraint.floor)} floor`,
          )
          .join(", ");
        return {
          key: `cleared-${scenario.id}`,
          title: `${scenario.name} signed off`,
          body: `${metricLines}. ${
            attempts === 1
              ? "First cutoff you tried."
              : `Took ${attempts} cutoffs to find the band.`
          } Accuracy on this shift was ${percent(
            result.metrics.accuracy,
          )} — worth noting it was ${
            result.metrics.accuracy >=
            majorityBaseline(scenario.prevalence)
              ? "above"
              : "below"
          } the ${percent(
            majorityBaseline(scenario.prevalence),
          )} you would get for ignoring the model entirely, which is how little accuracy had to say about whether you did the job.`,
          tone: "good",
        };
      }

      if (result.outcome === "accuracy-paradox") {
        return {
          key: `paradox-${scenario.id}`,
          title: "Accuracy is flattering you",
          body: `${percent(
            result.metrics.accuracy,
          )} accuracy, ${percent(
            result.metrics.recall,
          )} recall. Those two numbers describe the same classifier. Accuracy has ${result.matrix.trueNegatives} true negatives in its numerator and only ${
            result.matrix.truePositives
          } true positive${
            result.matrix.truePositives === 1 ? "" : "s"
          }, so it is mostly measuring how good you are at the easy majority class. The ${
            result.matrix.falseNegatives
          } missed ${scenario.positiveLabel}${
            result.matrix.falseNegatives === 1 ? "" : "s"
          } barely register in it — and on this shift, each one ${
            scenario.falseNegativeCost
          }.`,
          tone: "bad",
        };
      }

      if (result.outcome === "wrong-side") {
        return {
          key: `wrong-side-${scenario.id}-${result.threshold.toFixed(2)}`,
          title: "You optimised the wrong error",
          body: `Precision ${percent(
            result.metrics.precision,
          )}, recall ${percent(
            result.metrics.recall,
          )}. One of those is nearly perfect and it is not the one the critic is paying for. On this shift ${
            scenario.falseNegativeCost
          }, while ${
            scenario.falsePositiveCost
          } — so the cheap mistake is the one to make more of. The matrix shows exactly where the trade sits: ${
            result.matrix.falsePositives
          } false alarm${
            result.matrix.falsePositives === 1 ? "" : "s"
          } against ${result.matrix.falseNegatives} miss${
            result.matrix.falseNegatives === 1 ? "" : "es"
          }.`,
          tone: "bad",
        };
      }

      const failedNames = result.constraints
        .filter((constraint) => !constraint.met)
        .map((constraint) => METRIC_LABELS[constraint.metric].toLowerCase())
        .join(" and ");
      return {
        key: `missed-${scenario.id}-${result.threshold.toFixed(2)}`,
        title: `Short on ${failedNames}`,
        body: `${result.constraints
          .map(
            (constraint) =>
              `${METRIC_LABELS[constraint.metric].toLowerCase()} ${percent(
                constraint.achieved,
              )}/${percent(constraint.floor)}${constraint.met ? " ✓" : ""}`,
          )
          .join(", ")}. The matrix is ${result.matrix.truePositives} caught, ${
          result.matrix.falseNegatives
        } missed, ${result.matrix.falsePositives} false alarm${
          result.matrix.falsePositives === 1 ? "" : "s"
        }. Move the cutoff one notch at a time and watch which of those three numbers changes — the one that moves is the one you are trading.`,
        tone: "warn",
      };
    }
  }
}

