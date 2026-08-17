import type { WhyCardContent } from "@/components";
import {
  LEGENDARY_COMBOS,
  TARGET_LIFT,
  TRAIN_ROWS,
  columnByName,
  describeFeature,
  transformById,
  usesLeakyColumn,
  type Evaluation,
  type Feature,
  type LegendaryCombo,
  type Transform,
} from "./ml";

/**
 * "Why did that happen?" copy for Feature Forge.
 *
 * The rule this copy follows: when a score moves, say what the model can now
 * express that it could not before. "That feature helped" is a fact the gauge
 * already shows; "five bins give the model five independent weights, so it can
 * score both ends of the age range as risky while the middle stays calm" is the
 * part that transfers to the next dataset.
 *
 * And it never lets a good number pass unexamined. The highest score in this game
 * comes from leakage, so a rising gauge is not automatically good news.
 */
export type ForgeEvent =
  | { kind: "briefing" }
  | { kind: "baseline-ready"; baselineScore: number; trainScore: number }
  | { kind: "transform-picked"; transform: Transform }
  | ({ kind: "forged-feature"; feature: Feature } & ForgeContext)
  | ({ kind: "removed-feature"; feature: Feature } & ForgeContext)
  | ({ kind: "cleared" } & ForgeContext)
  | ({ kind: "retrained" } & ForgeContext)
  | { kind: "submitted"; evaluation: Evaluation };

interface ForgeContext {
  baselineScore: number;
  currentScore: number;
  trainScore: number;
  previousScore: number;
  features: Feature[];
  legendary: LegendaryCombo[];
  leaked: boolean;
}

const percent = (value: number) => `${Math.round(value * 100)}%`;
const points = (value: number) => `${(value * 100).toFixed(1)}%`;
const signed = (value: number) =>
  `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)} points`;

export function whyCardFor(event: ForgeEvent): WhyCardContent {
  switch (event.kind) {
    case "briefing":
      return {
        key: "briefing",
        title: "The model is fixed. Only the features can change.",
        body: `One logistic regression, same weights initialised the same way, same optimiser, same ${TRAIN_ROWS} rows, every single time. Nothing you can do will change the model — so anything that moves the score did it by changing what the columns MEAN. That is the entire experiment. One column in this table is poison; the gauge will love it.`,
        tone: "info",
      };

    case "baseline-ready":
      return {
        key: "baseline-ready",
        title: `Baseline ${points(event.baselineScore)}`,
        body: `Every usable column, standardised, handed to the fixed model. Note that the baseline is ALREADY scaled — so rescaling something again will do nothing, and the lift has to come from somewhere else. Train accuracy is ${points(
          event.trainScore,
        )}, almost the same, which means the model is not memorising: it simply cannot express much with these columns. Beat it by ${percent(
          TARGET_LIFT,
        )}.`,
        tone: "info",
      };

    case "transform-picked": {
      const spec = transformById(event.transform);
      return {
        key: `transform-${event.transform}`,
        title: `${spec.label} — ${spec.arity === 1 ? "one column" : "two columns"}`,
        body: `${spec.blurb} ${
          spec.mechanism.charAt(0).toUpperCase() + spec.mechanism.slice(1)
        }. ${
          event.transform === "standardise" || event.transform === "log_scale"
            ? "Worth knowing before you spend it: the baseline is already standardised, so this changes units and nothing else."
            : "This one changes what the model is able to express, not just the numbers it sees."
        }`,
        tone: "info",
      };
    }

    case "forged-feature": {
      const { feature, baselineScore, currentScore, previousScore, legendary, leaked } =
        event;
      const delta = currentScore - previousScore;
      const spec = transformById(feature.transform);

      if (leaked && usesLeakyColumn(feature)) {
        return {
          key: `leak-${feature.id}`,
          title: "That is not a feature, it is the answer",
          body: `${points(
            currentScore,
          )} — the best number you will see here, and worthless. ${describeFeature(
            feature,
          )} uses refund_issued, which only gets set after a customer has already gone. At prediction time, for someone who has not churned yet, it is always zero. ${
            columnByName("refund_issued")?.description ?? ""
          } Remove it and get the real answer.`,
          tone: "bad",
        };
      }

      const legendaryHit = legendary.find((combo) =>
        combo.matches([feature]),
      );

      if (legendaryHit) {
        return {
          key: `legendary-${feature.id}`,
          title: `Legendary: ${legendaryHit.label}`,
          body: `${legendaryHit.why} ${describeFeature(
            feature,
          )} took the score to ${points(currentScore)}, ${signed(
            delta,
          )} on the last fit and ${signed(
            currentScore - baselineScore,
          )} against baseline. ${
            Math.abs(delta) < 0.015
              ? `A small move on its own — these four relationships compound, and the combination is worth far more than the sum of the parts. Keep going.`
              : `${spec.mechanism.charAt(0).toUpperCase() + spec.mechanism.slice(1)}.`
          }`,
          tone: "good",
        };
      }

      return {
        key: `forged-${feature.id}-${currentScore.toFixed(4)}`,
        title: `${describeFeature(feature)} — ${signed(delta)}`,
        body: `Score ${points(currentScore)}, baseline ${points(
          baselineScore,
        )}. ${
          Math.abs(delta) < 0.005
            ? `Nothing moved, and that is informative: ${spec.mechanism}. If a transform only restates a column the model already had, a linear model has no new way to use it.`
            : delta < 0
              ? `It went DOWN. Every extra column is another weight to estimate from the same ${TRAIN_ROWS} rows, so a transform that adds width without adding meaning costs accuracy.`
              : `${spec.mechanism.charAt(0).toUpperCase() + spec.mechanism.slice(1)}.`
        }`,
        tone: Math.abs(delta) < 0.005 ? "warn" : delta < 0 ? "warn" : "info",
      };
    }

    case "removed-feature": {
      const { feature, currentScore, previousScore } = event;
      const delta = currentScore - previousScore;
      return {
        key: `removed-${feature.id}`,
        title: `Removed ${describeFeature(feature)} — ${signed(delta)}`,
        body:
          delta > 0.005
            ? `The score went UP when that feature came out. It was costing width without earning it — the model had another weight to fit and nothing extra to fit it to.`
            : delta < -0.005
              ? `The score fell by ${points(
                  Math.abs(delta),
                )} without it, so that feature was doing real work.`
              : `No change either way, which means that feature was neither helping nor hurting. Fewer columns for the same score is the better model.`,
        tone: "info",
      };
    }

    case "cleared":
      return {
        key: "cleared",
        title: "Back to the baseline",
        body: `Every forged feature removed. The score is whatever the raw standardised columns can manage — the number you have to beat by ${percent(
          TARGET_LIFT,
        )}.`,
        tone: "info",
      };

    case "retrained": {
      const { currentScore, trainScore } = event;
      return {
        key: `retrained-${currentScore.toFixed(4)}`,
        title: `Refit: ${points(currentScore)}`,
        body: `Same features, same fixed model, same seed — so this is the same number as last time. Train accuracy ${points(
          trainScore,
        )}${
          trainScore - currentScore > 0.06
            ? `, which is well above validation: with this many columns the model has started memorising rows rather than learning the pattern.`
            : `, in step with validation, so nothing is being memorised.`
        }`,
        tone: "info",
      };
    }

    case "submitted": {
      const { evaluation } = event;

      if (evaluation.outcome === "forged") {
        const names = evaluation.legendary.map((combo) => combo.label);
        return {
          key: "forged-win",
          title: `${signed(evaluation.lift)} over baseline`,
          body: `${points(evaluation.currentScore)} against ${points(
            evaluation.baselineScore,
          )}, from the same model that scored the baseline. ${
            names.length > 0
              ? `${names.join(", ")} — ${names.length} of ${
                  LEGENDARY_COMBOS.length
                } relationships that no amount of rescaling could reach.`
              : "Done without any of the four legendary transforms, which is a harder route than it looks."
          } The thing to take away is what did NOT happen: the model never got bigger, never got tuned, never got more data. Representation did all of it.`,
          tone: "good",
        };
      }

      if (evaluation.outcome === "leakage") {
        return {
          key: "leak-submitted",
          title: "The best score in the game, and unusable",
          body: `${points(
            evaluation.currentScore,
          )} — ${signed(
            evaluation.lift,
          )}. A model this good at predicting churn from a refund is only telling you that refunds follow churn. Deployed, it would see zero in that column for every customer who has not left yet, and predict that nobody ever will. Leakage is the most expensive mistake in applied machine learning precisely because it arrives disguised as your best result.`,
          tone: "bad",
        };
      }

      if (evaluation.outcome === "no-lift") {
        return {
          key: "no-lift-submitted",
          title: "Nothing moved",
          body: `${points(evaluation.currentScore)} against ${points(
            evaluation.baselineScore,
          )}. A linear model multiplies each column by a weight and adds them up — that is its whole vocabulary. Handing it the same column in different units gives it nothing new to say. What it cannot do is bend a line, split a false ordering, or divide one column by another, and those are the three things worth forging.`,
          tone: "warn",
        };
      }

      return {
        key: `short-submitted-${evaluation.currentScore.toFixed(4)}`,
        title: `${signed(evaluation.lift)} — short of ${percent(TARGET_LIFT)}`,
        body: `${points(evaluation.currentScore)} against ${points(
          evaluation.baselineScore,
        )}. ${
          evaluation.legendary.length > 0
            ? `${evaluation.legendary
                .map((combo) => combo.label)
                .join(" and ")} found. `
            : ""
        }There are ${
          LEGENDARY_COMBOS.length - evaluation.legendary.length
        } relationship${
          LEGENDARY_COMBOS.length - evaluation.legendary.length === 1 ? "" : "s"
        } left that a rescale cannot touch: a timestamp hiding a weekday, an effect that bends at both ends, a category stored as a number, and a quantity that only exists between two columns.`,
        tone: "warn",
      };
    }
  }
}
