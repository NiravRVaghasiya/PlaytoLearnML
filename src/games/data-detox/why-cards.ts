import type { WhyCardContent } from "@/components";
import {
  BATCH_SIZE,
  MAX_BALANCE_DRIFT,
  MIN_ROWS,
  OUTLIER_FACTOR,
  WIN_ACCURACY,
  type CleaningAction,
  type Evaluation,
  type PipelineResult,
  type Row,
} from "./ml";

/**
 * "Why did that happen?" copy for Data Detox.
 *
 * Keyed to the player's last action (spec §4). Every number is computed in
 * `ml.ts` — none is invented here.
 *
 * The through-line is the core intuition: cleaning decisions move model quality,
 * and there is no single right answer. So the copy avoids praising or scolding a
 * choice outright; it names the tradeoff the choice just made.
 */

export type DataDetoxEvent =
  | { kind: "reset" }
  | {
      kind: "sorted";
      action: CleaningAction;
      row: Row;
      pipeline: PipelineResult;
    }
  | {
      kind: "retrained";
      accuracy: number;
      previous: number | null;
      pipeline: PipelineResult;
    }
  | { kind: "checked"; evaluation: Evaluation };

const CLEANING_HREF = "/concepts/data-cleaning";
const MISSING_HREF = "/concepts/missing-data";

const percent = (value: number) => `${Math.round(value * 100)}%`;
const signed = (value: number) =>
  `${value >= 0 ? "+" : "−"}${Math.abs(Math.round(value * 100))}%`;

export function whyCardFor(event: DataDetoxEvent): WhyCardContent {
  switch (event.kind) {
    case "reset":
      return {
        key: "reset",
        title: "Sort the rows, then watch the model",
        body: `Blank cells glow red, out-of-range values sit oversized. Impute fills blanks with the column median, Cap clamps extremes, Keep passes the row through untouched, Drop throws it away. Every ${BATCH_SIZE} decisions the model retrains and the accuracy meter moves.`,
        tone: "info",
        conceptHref: CLEANING_HREF,
        conceptLabel: "What data cleaning does",
      };

    case "sorted": {
      const { action, row, pipeline } = event;
      const key = `sorted-${row.id}-${action}`;

      // Name the tradeoff this specific choice just made on this specific row.
      if (action === "drop") {
        return {
          key,
          title: `Dropped — ${pipeline.kept} rows left`,
          body: `That row's label is gone from the training set along with its problem. Dropping is the only action that costs you data, and blank readings aren't spread evenly across the two classes — so dropping them quietly reshapes what the model thinks the world looks like.`,
          tone: pipeline.kept < MIN_ROWS ? "bad" : "warn",
          conceptHref: MISSING_HREF,
          conceptLabel: "Why dropping isn't neutral",
        };
      }

      if (action === "keep" && row.isNull) {
        return {
          key,
          title: "Kept a row with a blank cell",
          body: `The model can't read a blank, so it reads 0. Zero isn't "unknown" — it's a specific, wrong measurement at the bottom of the range, and it will drag the boundary toward it. ${pipeline.keptWithNaiveFill} of your ${pipeline.kept} training rows now carry one.`,
          tone: "warn",
          conceptHref: MISSING_HREF,
        };
      }

      if (action === "keep" && row.isOutlier) {
        return {
          key,
          title: "Kept a row with an extreme value",
          body: `That reading is about ${OUTLIER_FACTOR}× the normal range. A handful of extreme rows pulls a fit further than dozens of ordinary ones, because the error is squared. ${pipeline.keptWithOutlier} of your ${pipeline.kept} rows still carry one.`,
          tone: "warn",
        };
      }

      if (action === "keep") {
        return {
          key,
          title: "Kept a clean row",
          body: `Nothing wrong with it, so nothing to fix — and every untouched row is a row the model learns from without any guesswork on your part.`,
          tone: "good",
        };
      }

      if (action === "impute") {
        return {
          key,
          title: row.isNull
            ? "Imputed — blank filled with the median"
            : "Imputed a row that had no blanks",
          body: row.isNull
            ? `The blank became the column median. That's a guess, and it makes the row slightly less informative than a real measurement — but it keeps the row, and its label, in the training set.${
                row.isOutlier
                  ? " Note this row also has an extreme value, and imputing doesn't touch that."
                  : ""
              }`
            : `Impute only affects blank cells, and this row had none, so nothing changed. Not harmful — just not useful.`,
          tone: row.isNull ? "good" : "info",
          conceptHref: MISSING_HREF,
        };
      }

      // action === "cap"
      return {
        key,
        title: row.isOutlier
          ? "Capped — extreme value clamped into range"
          : "Capped a row with nothing extreme",
        body: row.isOutlier
          ? `The reading was clamped to the 95th percentile: still high, no longer absurd. You keep the row's information about direction without letting one bad sensor dominate the fit.${
              row.isNull
                ? " This row also has a blank, and capping can't fill a blank — it became 0."
                : ""
            }`
          : `Cap clamps values into the 5th–95th percentile band, and everything here was already inside it. Harmless, but it did nothing.`,
        tone: row.isOutlier && !row.isNull ? "good" : row.isNull ? "warn" : "info",
      };
    }

    case "retrained": {
      const { accuracy, previous, pipeline } = event;
      const delta = previous === null ? null : accuracy - previous;

      return {
        key: `retrained-${pipeline.kept}-${accuracy.toFixed(4)}`,
        title:
          delta === null
            ? `Model trained — ${percent(accuracy)} on held-out data`
            : `Retrained — ${percent(accuracy)} (${signed(delta)})`,
        body:
          delta === null
            ? `That's a real network, trained on your ${pipeline.kept} cleaned rows and scored on rows it has never seen. Keep sorting; it retrains every ${BATCH_SIZE} decisions.`
            : delta >= 0
              ? `Your last ${BATCH_SIZE} decisions moved held-out accuracy ${signed(delta)}, trained on ${pipeline.kept} rows. Whatever you did to those rows preserved more signal than it destroyed.`
              : `Held-out accuracy fell ${signed(delta)} on ${pipeline.kept} rows. Something in that batch cost the model more than it gave — check whether you dropped rows, or kept blanks that became zeros.`,
        tone: delta === null ? "info" : delta >= 0 ? "good" : "warn",
        conceptHref: CLEANING_HREF,
      };
    }

    case "checked": {
      const {
        outcome,
        accuracy,
        score,
        timePenalty,
        kept,
        dropped,
        keptWithNaiveFill,
        keptWithOutlier,
        keptPositiveShare,
        sourcePositiveShare,
        binCounts,
      } = event.evaluation;

      const key = `checked-${outcome}-${score.toFixed(4)}`;

      switch (outcome) {
        case "incomplete":
          return {
            key,
            title: "Still rows on the belt",
            body: `Sort every row before scoring — an unfinished pipeline isn't a pipeline. Drop is a valid answer if you can justify it.`,
            tone: "warn",
          };

        case "starved":
          return {
            key,
            title: "Data starvation",
            body: `${kept} rows survived out of ${kept + dropped}. Below about ${MIN_ROWS} the model has too few examples to find the pattern, however spotless they are. Dropping is the most expensive repair available — it fixes the row by deleting the evidence.`,
            tone: "bad",
            conceptHref: MISSING_HREF,
            conceptLabel: "The cost of dropping rows",
          };

        case "selection-bias":
          return {
            key,
            title: "Selection bias",
            body: `Your surviving rows are ${percent(keptPositiveShare)} healthy; the data you started with was ${percent(sourcePositiveShare)} healthy. Blanks aren't spread evenly across the classes, so dropping ${dropped} of them removed one class in particular. The model is now well fitted to a world that doesn't exist.`,
            tone: "bad",
            conceptHref: MISSING_HREF,
            conceptLabel: "Missing not at random",
          };

        case "unhandled-nulls":
          return {
            key,
            title: "Unhandled missing values",
            body: `${keptWithNaiveFill} of ${kept} training rows still had a blank, and the model read every one as 0 — a specific wrong measurement, not "unknown". Held-out accuracy ${percent(accuracy)}. Impute would have filled them with the median instead.`,
            tone: "bad",
            conceptHref: MISSING_HREF,
          };

        case "outlier-contamination":
          return {
            key,
            title: "Outlier contamination",
            body: `${keptWithOutlier} of ${kept} training rows carried a value ${OUTLIER_FACTOR}× out of range. Squared error means a few absurd rows outvote many sensible ones. Held-out accuracy ${percent(accuracy)}.`,
            tone: "bad",
            conceptHref: CLEANING_HREF,
          };

        case "win":
          return {
            key,
            title: `Clean enough — ${percent(score)}`,
            body: `${percent(accuracy)} on held-out data from ${kept} rows${
              timePenalty > 0 ? `, minus ${percent(timePenalty)} for time` : ""
            }. You imputed ${binCounts.impute}, capped ${binCounts.cap}, kept ${binCounts.keep} and dropped ${binCounts.drop} — and the class balance came through intact, within ${percent(MAX_BALANCE_DRIFT)}.`,
            tone: "good",
            conceptHref: CLEANING_HREF,
          };

        case "near-miss":
        default:
          return {
            key,
            title: `${percent(score)} — needs ${percent(WIN_ACCURACY)}`,
            body: `No obvious dirt left and the sample isn't skewed, so the remaining gap is the guesswork itself: every imputed value is an estimate, and estimates carry less information than measurements. Try capping where you imputed, or the reverse, and watch which way the meter moves.`,
            tone: "warn",
            conceptHref: CLEANING_HREF,
          };
      }
    }
  }
}
