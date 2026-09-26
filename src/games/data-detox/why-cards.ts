import type { WhyCardContent } from "@/components";
import {
  BATCH_SIZE,
  FEATURES,
  MAX_BALANCE_DRIFT,
  MIN_ROWS,
  SENSOR_MAX,
  SENSOR_MIN,
  WIN_ACCURACY,
  isOutOfRange,
  type CleaningAction,
  type ColumnStats,
  type Evaluation,
  type FeatureName,
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
 *
 * The model is a cross-entropy classifier and the dirt is in its INPUTS, so the
 * copy never blames outliers on "squared error". What a spike does to this net
 * is swamp the weighted sum inside every neuron it feeds, and send back a
 * gradient that scales with it.
 */

export type DataDetoxEvent =
  | { kind: "reset" }
  | {
      kind: "sorted";
      action: CleaningAction;
      row: Row;
      pipeline: PipelineResult;
      /** The column statistics Impute and Cap actually used. */
      stats: ColumnStats;
    }
  | {
      kind: "retrained";
      accuracy: number;
      previous: number | null;
      /** The pipeline this model was trained on. */
      pipeline: PipelineResult;
      /**
       * Rows whose decision differs from the one the previous model trained
       * on — new sorts and re-decisions alike. Zero means the same pipeline.
       */
      decisions: number;
    }
  | { kind: "checked"; evaluation: Evaluation };

const CLEANING_HREF = "/concepts/data-cleaning";
const MISSING_HREF = "/concepts/missing-data";

const percent = (value: number) => `${Math.round(value * 100)}%`;
const signed = (value: number) =>
  `${value >= 0 ? "+" : "−"}${Math.abs(Math.round(value * 100))}%`;

/** The first reading on this row that is physically impossible, if any. */
function spikeOn(row: Row): { feature: FeatureName; value: number } | null {
  for (const feature of FEATURES) {
    const value = row.features[feature];
    if (value !== null && isOutOfRange(value)) return { feature, value };
  }
  return null;
}

export function whyCardFor(event: DataDetoxEvent): WhyCardContent {
  switch (event.kind) {
    case "reset":
      return {
        key: "reset",
        title: "Sort the rows, then watch the model",
        body: `Blank cells are outlined in red, impossible readings sit oversized. Impute fills blanks with the column median, Cap clamps extremes into the column's normal range, Keep passes the row through untouched, Drop throws it away. Every ${BATCH_SIZE} decisions the model retrains and the accuracy meter moves.`,
        tone: "info",
        conceptHref: CLEANING_HREF,
        conceptLabel: "What data cleaning does",
      };

    case "sorted": {
      const { action, row, pipeline, stats } = event;
      const key = `sorted-${row.id}-${action}`;
      const spike = spikeOn(row);

      // Name the tradeoff this specific choice just made on this specific row.
      if (action === "drop") {
        // Rows that could still reach the model: kept so far, plus the ones
        // not yet decided. Kept-so-far alone reads "0 rows left" on the first
        // drop, with a starvation tone, while 71 rows are still on the belt.
        const available = pipeline.kept + pipeline.undecided;
        return {
          key,
          title: `Dropped — ${available} rows can still train`,
          body: row.isNull
            ? `That row's label is gone from the training set along with its problem. Dropping is the only action that costs you data, and blank readings aren't spread evenly across the two classes — so dropping them quietly reshapes what the model thinks the world looks like.`
            : row.isOutlier
              ? // Spikes hit both classes about equally, but a row only gets
                // here without a blank, and blanks are MNAR — so spike-only
                // rows are mostly blighted, and the rail's drift shows it.
                `That row's label is gone along with its impossible reading, and dropping is the only action that costs you data. Spikes land on both classes at about the same rate, but blanks cluster on healthy plots — so rows whose only problem is a spike lean blighted, and dropping many of them tips the balance toward healthy. Class balance drift is ${signed(
                  pipeline.balanceDrift,
                )} now.`
              : `That row was clean, and dropping is the only action that costs you data: its label is gone from the training set, and nothing was wrong with it.`,
          tone: available < MIN_ROWS ? "bad" : "warn",
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
          title: "Kept a row with an impossible reading",
          body: `${spike ? `${spike.feature} reads ${spike.value.toFixed(2)}` : "One reading is far out of range"} on a sensor that reads ${SENSOR_MIN} to ${SENSOR_MAX}. Every neuron sums weight × input, so that one number swamps the rest of the row — and the gradient it sends back scales with it too. ${pipeline.keptWithOutlier} of your ${pipeline.kept} rows still carry one.`,
          tone: "warn",
          conceptHref: CLEANING_HREF,
          conceptLabel: "Outliers are not automatically errors",
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
                  ? " Note this row also has an impossible reading, and imputing doesn't touch that."
                  : ""
              }`
            : `Impute only affects blank cells, and this row had none, so nothing changed. Not harmful — just not useful.`,
          tone: row.isNull ? "good" : "info",
          conceptHref: MISSING_HREF,
        };
      }

      // action === "cap"
      const fence = spike
        ? spike.value > stats.high[spike.feature]
          ? stats.high[spike.feature]
          : stats.low[spike.feature]
        : null;
      return {
        key,
        title: row.isOutlier
          ? "Capped — impossible reading clamped into range"
          : "Capped a row with nothing extreme",
        body:
          row.isOutlier && spike && fence !== null
            ? `${spike.feature} went from ${spike.value.toFixed(2)} to ${fence.toFixed(2)}, the edge of the column's normal range: still high, no longer absurd. The true reading is lost either way — a glitch carries no trace of it — but one bad sensor no longer dominates the row.${
                row.isNull
                  ? " This row also has a blank, and capping can't fill a blank — it became 0."
                  : ""
              }`
            : `Cap clamps values into the column's normal range, and everything here was already inside it. Harmless, but it did nothing.${
                row.isNull
                  ? " This row has a blank, though, and capping can't fill a blank — it became 0."
                  : ""
              }`,
        tone: row.isOutlier && !row.isNull ? "good" : row.isNull ? "warn" : "info",
      };
    }

    case "retrained": {
      const { accuracy, previous, pipeline, decisions } = event;
      const delta = previous === null ? null : accuracy - previous;
      const lately =
        decisions === 1 ? "Your last decision" : `Your last ${decisions} decisions`;

      return {
        key: `retrained-${pipeline.kept}-${accuracy.toFixed(4)}`,
        title:
          delta === null
            ? `Model trained — ${percent(accuracy)} on held-out data`
            : `Retrained — ${percent(accuracy)} (${signed(delta)})`,
        body:
          delta === null
            ? `That's a real network, trained on your ${pipeline.kept} cleaned rows and scored on rows it has never seen. Keep sorting; it retrains every ${BATCH_SIZE} decisions.`
            : decisions === 0
              ? `Same pipeline, trained again from scratch on ${pipeline.kept} rows: ${percent(accuracy)} on held-out data.`
              : delta >= 0
                ? `${lately} moved held-out accuracy ${signed(delta)}, trained on ${pipeline.kept} rows. Whatever you did to those rows preserved more signal than it destroyed.`
                : `Held-out accuracy fell ${signed(delta)} on ${pipeline.kept} rows. Something in ${decisions === 1 ? "that decision" : "that batch"} cost the model more than it gave — check whether you dropped rows, kept blanks that became zeros, or kept an impossible reading.`,
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

      const key = `checked-${outcome}-${score === null ? "none" : score.toFixed(4)}`;
      const measured =
        accuracy === null ? "" : ` Held-out accuracy ${percent(accuracy)}.`;

      switch (outcome) {
        case "incomplete":
          return {
            key,
            title: "Still rows on the belt",
            body: `Sort every row before scoring — an unfinished pipeline isn't a pipeline. Drop is a valid answer if you can justify it.`,
            tone: "warn",
          };

        case "untrained":
          return {
            key,
            title: "Train the model on this pipeline first",
            body: `Every row is decided and nothing is structurally wrong, but no model has been trained on exactly these decisions yet, so there is no accuracy to score. Wait for the retrain to finish, press Retrain now, or await api.retrain() before api.check().`,
            tone: "info",
            conceptHref: CLEANING_HREF,
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
            body: `Your surviving rows are ${percent(keptPositiveShare)} healthy; the data you started with was ${percent(sourcePositiveShare)} healthy. ${
              keptPositiveShare < sourcePositiveShare
                ? `Blanks aren't spread evenly across the classes, so dropping ${dropped} of them removed one class in particular.`
                : `Your ${dropped} drops took blighted rows out of proportion, so healthy plots are now over-represented.`
            } The model is now well fitted to a world that doesn't exist.`,
            tone: "bad",
            conceptHref: MISSING_HREF,
            conceptLabel: "Missing not at random",
          };

        case "unhandled-nulls":
          return {
            key,
            title: "Unhandled missing values",
            body: `${keptWithNaiveFill} of ${kept} training rows still had a blank, and the model read every one as 0 — a specific wrong measurement, not "unknown".${measured} Impute would have filled them with the median instead.`,
            tone: "bad",
            conceptHref: MISSING_HREF,
          };

        case "outlier-contamination":
          return {
            key,
            title: "Outlier contamination",
            body: `${keptWithOutlier} of ${kept} training rows still carried a reading that is physically impossible for the sensor. Each one swamps the weighted sum inside every neuron it feeds, so a handful of glitches pulls the weights further than dozens of sensible rows.${measured} Cap would have clamped them to the column's normal range.`,
            tone: "bad",
            conceptHref: CLEANING_HREF,
            conceptLabel: "Outliers are not automatically errors",
          };

        case "win":
          return {
            key,
            title: `Clean enough — ${percent(score ?? 0)}`,
            body: `${percent(accuracy ?? 0)} on held-out data from ${kept} rows${
              timePenalty > 0 ? `, minus ${percent(timePenalty)} for time` : ""
            }. You imputed ${binCounts.impute}, capped ${binCounts.cap}, kept ${binCounts.keep} and dropped ${binCounts.drop} — and the class balance came through intact, within ${percent(MAX_BALANCE_DRIFT)}.`,
            tone: "good",
            conceptHref: CLEANING_HREF,
          };

        case "near-miss":
        default: {
          const clockCostIt =
            accuracy !== null && accuracy >= WIN_ACCURACY && timePenalty > 0;
          return {
            key,
            title: `${percent(score ?? 0)} — needs ${percent(WIN_ACCURACY)}`,
            body: clockCostIt
              ? `${percent(accuracy ?? 0)} on held-out data cleared the bar, and the ${percent(timePenalty)} time penalty took it back under. No dirt left and the sample isn't skewed, so a little more headroom from the cleaning itself is all it needs: try capping where you imputed, or the reverse, and watch which way the meter moves.`
              : `No obvious dirt left and the sample isn't skewed, so the remaining gap is the guesswork itself: every imputed value is an estimate, and estimates carry less information than measurements. Try capping where you imputed, or the reverse, and watch which way the meter moves.`,
            tone: "warn",
            conceptHref: CLEANING_HREF,
          };
        }
      }
    }
  }
}
