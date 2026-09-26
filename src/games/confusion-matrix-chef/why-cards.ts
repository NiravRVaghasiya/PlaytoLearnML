import type { WhyCardContent } from "@/components";
import {
  METRIC_LABELS,
  OVERSHOOT_LEVEL,
  SCENARIOS,
  confusionAt,
  dearErrorFor,
  majorityBaseline,
  metricName,
  metricsOf,
  opposingMetric,
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
const points = (value: number) => `${(value * 100).toFixed(1)}%`;

/** "1 prank order", "3 prank orders" — spelled from the scenario, never by "+s". */
const casesOf = (scenario: Scenario, count: number) =>
  count === 1 ? scenario.positiveLabel : scenario.positiveLabelPlural;

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

      // Precision after loosening is not a law, it is arithmetic: the pile's hit
      // rate moves toward the newcomers' hit rate. Say which way it went THIS
      // time, from the counts, rather than asserting it can only fall — on these
      // very shifts a notch down often raises it (prank 0.91 to 0.90 takes it
      // from 96.4% to 97.4%).
      const beforePrecision = metricsOf(before).precision;
      const beforeFlagged = before.truePositives + before.falsePositives;
      const newcomerRate =
        flaggedDelta > 0 ? Math.max(0, caughtDelta) / flaggedDelta : 0;
      const precisionMoved =
        metrics.precision > beforePrecision + 1e-12
          ? "rose"
          : metrics.precision < beforePrecision - 1e-12
            ? "fell"
            : "held";
      const precisionClause =
        beforeFlagged === 0
          ? `The pile was empty before, so precision is simply how often these newcomers were right: ${percent(
              metrics.precision,
            )}.`
          : `Precision only falls when the newcomers are right less often than the pile already was. These were right ${points(
              newcomerRate,
            )} of the time against the pile's ${points(
              beforePrecision,
            )}, so precision ${precisionMoved}.`;

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
              caughtDelta === 1
                ? `was a real ${scenario.positiveLabel}`
                : `were real ${scenario.positiveLabelPlural}`
            } and ${alarmDelta} ${
              alarmDelta === 1 ? "was a false alarm" : "were false alarms"
            }. Recall is now ${percent(
              metrics.recall,
            )} and precision ${percent(
              metrics.precision,
            )}. Growing the pile can never lower recall — ${
              caughtDelta > 0
                ? `the ${caughtDelta} new catch${caughtDelta === 1 ? "" : "es"} raised it`
                : "none of these were real, so it held"
            }. ${precisionClause} The two share a numerator and differ in the denominator, which is why they usually, but not on every notch, move in opposite directions.`
          : `${Math.abs(flaggedDelta)} case${
              Math.abs(flaggedDelta) === 1 ? "" : "s"
            } left the flagged pile: ${Math.abs(
              alarmDelta,
            )} false alarm${Math.abs(alarmDelta) === 1 ? "" : "s"} you no longer raise, and ${Math.abs(
              caughtDelta,
            )} real ${casesOf(
              scenario,
              Math.abs(caughtDelta),
            )} you now miss. Precision is ${percent(
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
              `${metricName(constraint.metric)} ${percent(
                constraint.achieved,
              )} against a ${percent(constraint.floor)} floor`,
          )
          .join(", ");
        // Where inside the band this cutoff landed, on the headline metric —
        // the number the score is built from, so the second star is legible.
        const range = result.primaryRange;
        const rangeLine =
          range === null || range.max - range.min < 0.0005
            ? ""
            : ` Inside the band, ${metricName(scenario.primary)} ran from ${points(
                range.min,
              )} to ${points(range.max)}; you served ${points(
                result.metrics[scenario.primary],
              )}.`;
        return {
          key: `cleared-${scenario.id}`,
          title: `${scenario.name} signed off`,
          body: `${metricLines}.${rangeLine} ${
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
          } missed ${casesOf(
            scenario,
            result.matrix.falseNegatives,
          )} barely register in it — and on this shift, each one ${
            scenario.falseNegativeCost
          }.`,
          tone: "bad",
        };
      }

      if (result.outcome === "wrong-side" || result.outcome === "overshoot") {
        // Both are "one metric all but perfect, the other's floor broken". The
        // card names which metric the critic is paying for from the BRIEF, so it
        // never tells a player at 100% recall on the allergen shift that recall
        // "is not the one the critic is paying for".
        // The same floor the judge picked: broken, with its opposite metric
        // all but perfect.
        const failing =
          result.constraints.find((constraint) => {
            const opposite = opposingMetric(constraint.metric);
            return (
              !constraint.met &&
              opposite !== null &&
              result.metrics[opposite] >= OVERSHOOT_LEVEL
            );
          })?.metric ?? "precision";
        const perfect = opposingMetric(failing) ?? "recall";
        const dear = dearErrorFor(scenario, failing);
        const dearCost =
          dear === "falsePositive"
            ? scenario.falsePositiveCost
            : scenario.falseNegativeCost;
        const cheapCost =
          dear === "falsePositive"
            ? scenario.falseNegativeCost
            : scenario.falsePositiveCost;
        const trade = `The matrix shows exactly where the trade sits: ${
          result.matrix.falsePositives
        } false alarm${
          result.matrix.falsePositives === 1 ? "" : "s"
        } against ${result.matrix.falseNegatives} miss${
          result.matrix.falseNegatives === 1 ? "" : "es"
        }.`;
        const pair = `Precision ${percent(
          result.metrics.precision,
        )}, recall ${percent(result.metrics.recall)}.`;

        if (result.outcome === "overshoot") {
          return {
            key: `overshoot-${scenario.id}-${result.threshold.toFixed(2)}`,
            title: "Right error, pushed too far",
            body: `${pair} ${METRIC_LABELS[perfect]} is the number this shift is judged on, and it is nearly perfect — but the brief also puts a floor under ${metricName(
              failing,
            )}, and you sold it to get there. On this shift the expensive mistake is the ${
              dear === "falsePositive" ? "false alarm" : "miss"
            } — ${dearCost} — so leaning away from it was right. The other one, where ${cheapCost}, is cheaper but not free. ${trade}`,
            tone: "bad",
          };
        }

        return {
          key: `wrong-side-${scenario.id}-${result.threshold.toFixed(2)}`,
          title: "You optimised the wrong error",
          body:
            opposingMetric(scenario.primary) === null
              ? `${pair} ${METRIC_LABELS[perfect]} is nearly perfect, and this brief needs both — it puts a floor under ${metricName(
                  failing,
                )} too, and that is the one breaking. On this shift ${dearCost}, and you are making that mistake far more often than the brief allows. ${trade}`
              : `${pair} ${METRIC_LABELS[perfect]} is nearly perfect, and it is not the one the critic is paying for — ${metricName(
                  scenario.primary,
                )} is. On this shift ${dearCost}, while ${cheapCost} — so the cheap mistake is the one to make more of. ${trade}`,
          tone: "bad",
        };
      }

      const failedNames = result.constraints
        .filter((constraint) => !constraint.met)
        .map((constraint) => metricName(constraint.metric))
        .join(" and ");
      return {
        key: `missed-${scenario.id}-${result.threshold.toFixed(2)}`,
        title: `Short on ${failedNames}`,
        body: `${result.constraints
          .map(
            (constraint) =>
              `${metricName(constraint.metric)} ${percent(
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

