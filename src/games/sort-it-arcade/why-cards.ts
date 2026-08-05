import type { WhyCardContent } from "@/components";
import {
  KNOT_COUNTS,
  NOISE_RATE,
  WIN_SCORE,
  type BoundaryType,
  type Evaluation,
} from "./ml";

/**
 * "Why did that happen?" copy for Sort-It Arcade.
 *
 * Every card is keyed to the player's *last action* (spec §4), and every number
 * in the copy is passed in from a real computation in `ml.ts` — nothing here
 * invents a figure. The `key` field changes whenever the explanation is new,
 * which is what replays the card's fade-in so the player notices.
 *
 * House style: two lines, plain language, and name the mechanism rather than
 * just the outcome. "You gained 4% but paid for 20 more parameters" teaches;
 * "Nice!" does not.
 */

export type SortItEvent =
  | { kind: "reset" }
  | {
      kind: "boundary-moved";
      accuracy: number;
      accuracyDelta: number;
      misclassified: number;
      total: number;
    }
  | {
      kind: "complexity-changed";
      from: BoundaryType;
      to: BoundaryType;
      params: number;
      penalty: number;
    }
  | { kind: "auto-fitted"; accuracy: number; params: number }
  | { kind: "checked"; evaluation: Evaluation };

const CONCEPT_HREF = "/concepts/decision-boundaries";
const OVERFIT_HREF = "/concepts/overfitting";

const percent = (value: number) => `${Math.round(value * 100)}%`;
const signedPercent = (value: number) =>
  `${value >= 0 ? "+" : "−"}${Math.abs(Math.round(value * 100))}%`;

const TYPE_LABEL: Record<BoundaryType, string> = {
  line: "straight line",
  curve: "curve",
  wiggle: "wiggle",
};

export function whyCardFor(event: SortItEvent): WhyCardContent {
  switch (event.kind) {
    case "reset":
      return {
        key: "reset",
        title: "Drag the boundary to sort the points",
        body: `Blue circles belong below the line, orange triangles above it. Accuracy updates as you move. About ${percent(
          NOISE_RATE,
        )} of these labels are deliberately wrong, so nobody can reach 100% — and chasing it is the trap.`,
        tone: "info",
        conceptHref: CONCEPT_HREF,
        conceptLabel: "What is a decision boundary?",
      };

    case "boundary-moved": {
      const { accuracy, accuracyDelta, misclassified, total } = event;

      if (Math.abs(accuracyDelta) < 0.0001) {
        return {
          key: `moved-flat-${accuracy.toFixed(4)}`,
          title: "No change in accuracy",
          body: `Still ${percent(
            accuracy,
          )}, with ${misclassified} of ${total} points on the wrong side. You moved the boundary through empty space — it only matters where points actually are.`,
          tone: "info",
        };
      }

      const improved = accuracyDelta > 0;
      return {
        key: `moved-${accuracy.toFixed(4)}-${accuracyDelta.toFixed(4)}`,
        title: improved
          ? `Accuracy up ${signedPercent(accuracyDelta)}`
          : `Accuracy down ${signedPercent(accuracyDelta)}`,
        body: improved
          ? `${percent(
              accuracy,
            )} now — ${misclassified} of ${total} points still on the wrong side. That move put the boundary between two groups instead of through one.`
          : `${percent(
              accuracy,
            )} now, and ${misclassified} of ${total} points are on the wrong side. The boundary crossed over points it had already sorted correctly.`,
        tone: improved ? "good" : "warn",
      };
    }

    case "complexity-changed": {
      const { from, to, params, penalty } = event;
      const moreCapacity = KNOT_COUNTS[to] > KNOT_COUNTS[from];

      if (!moreCapacity) {
        return {
          key: `complexity-${to}`,
          title: `Back to a ${TYPE_LABEL[to]} — ${params} parameters`,
          body: `Fewer parameters means less to charge you for: the penalty is now ${penalty.toFixed(
            3,
          )}. A simpler boundary usually travels better to data it hasn't seen.`,
          tone: "good",
          conceptHref: CONCEPT_HREF,
        };
      }

      return {
        key: `complexity-${to}`,
        title: `${TYPE_LABEL[to]} selected — ${params} parameters`,
        body: `More bends means you can fit these points more tightly, but it costs ${penalty.toFixed(
          3,
        )} off your score. The question is whether the extra accuracy is real or just noise.`,
        tone: to === "wiggle" ? "warn" : "info",
        conceptHref: OVERFIT_HREF,
        conceptLabel: "Why complexity is a cost",
      };
    }

    case "auto-fitted":
      return {
        key: `fitted-${event.params}-${event.accuracy.toFixed(4)}`,
        title: `Fitted — ${percent(event.accuracy)} on the training points`,
        body: `The optimizer swept every knot to the height that made the most training points correct. That's the same thing you do by dragging, just faster and more stubborn about it.`,
        tone: "info",
        conceptHref: CONCEPT_HREF,
      };

    case "checked": {
      const {
        outcome,
        trainAccuracy,
        testAccuracy,
        generalizationGap,
        paramCount,
        penalty,
        score,
        bestAtThisComplexity,
      } = event.evaluation;

      const key = `checked-${outcome}-${score.toFixed(4)}`;

      switch (outcome) {
        case "overfit":
          return {
            key,
            title: "You overfit",
            body: `${percent(
              trainAccuracy,
            )} on the points you fitted, ${percent(
              testAccuracy,
            )} on fresh ones — a ${percent(
              generalizationGap,
            )} drop. Your ${paramCount} parameters bent around individual mislabelled points, and those bends mean nothing anywhere else.`,
            tone: "bad",
            conceptHref: OVERFIT_HREF,
            conceptLabel: "What is overfitting?",
          };

        case "underfit":
          return {
            key,
            title: "You underfit",
            body: `${percent(
              trainAccuracy,
            )} train, and ${percent(
              bestAtThisComplexity,
            )} is the ceiling for ${paramCount} parameters here. The real boundary bends and yours can't — add capacity rather than nudging this one further.`,
            tone: "bad",
            conceptHref: CONCEPT_HREF,
            conceptLabel: "Bias and capacity",
          };

        case "win":
          return {
            key,
            title: `Round cleared — ${percent(score)}`,
            body: `${percent(trainAccuracy)} on your points and ${percent(
              testAccuracy,
            )} on ones it never saw: only a ${percent(
              Math.abs(generalizationGap),
            )} difference. ${paramCount} parameters was enough to follow the real boundary without tracing the noise.`,
            tone: "good",
            conceptHref: OVERFIT_HREF,
            conceptLabel: "Why simpler generalises",
          };

        case "near-miss":
        default:
          return {
            key,
            title: `${percent(score)} — needs ${percent(WIN_SCORE)}`,
            body: `${percent(trainAccuracy)} accuracy minus ${penalty.toFixed(
              3,
            )} for ${paramCount} parameters. The best a ${paramCount}-parameter boundary can do here is ${percent(
              bestAtThisComplexity,
            )}, so there's still room — keep adjusting, or change capacity.`,
            tone: "warn",
            conceptHref: CONCEPT_HREF,
          };
      }
    }
  }
}
