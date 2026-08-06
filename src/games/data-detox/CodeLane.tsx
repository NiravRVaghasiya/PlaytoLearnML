"use client";

import { useMemo } from "react";
import { Play, RotateCcw } from "lucide-react";
import { Button, CodeEditor } from "@/components";
import { useCodeLane } from "@/engine/useCodeLane";
import { CLEANING_ACTIONS, type CleaningAction } from "./ml";
import { useDataDetoxStore } from "./store";

/**
 * Data Detox — the code lane.
 *
 * The visual lane sorts rows one at a time; here you write the rule that sorts all
 * of them. Both call the same `setRowAction`, so a policy written in code and the
 * same policy clicked out by hand produce byte-identical training matrices
 * (CLAUDE.md two-lane rule).
 *
 * This is what a preprocessing pipeline actually looks like in practice — a
 * function from row to decision — which makes the "player action = algorithm"
 * claim literal rather than analogical.
 */

const STARTER_CODE = `// A preprocessing pipeline is a function from row to decision.
// Write the rule; it applies to every row on the belt.

api.forEachRow((row) => {
  // row.isNull     — a cell is blank
  // row.isOutlier  — a value is ~7x out of range
  if (row.isNull) return 'impute';   // fill with the column median
  if (row.isOutlier) return 'cap';   // clamp into the 5th-95th range
  return 'keep';
});

log('kept   ', api.kept());
log('dropped', api.dropped());
log('blanks left  ', api.blanksLeft());
log('extremes left', api.extremesLeft());
log('class balance drift', api.balanceDrift().toFixed(3));

// Retrain on the result and score it on held-out rows.
const accuracy = await api.retrain();
log('held-out accuracy', accuracy.toFixed(3));
log('verdict', api.check().outcome);

// Now try 'drop' instead of 'impute' for blanks. Accuracy falls AND
// the balance drift jumps — blanks are far more common on healthy
// plots, so dropping them deletes one class in particular.`;

export interface CodeLaneProps {
  /** Retrains the model and resolves with held-out accuracy. */
  retrain: () => Promise<number>;
}

export function CodeLane({ retrain }: CodeLaneProps) {
  const store = useDataDetoxStore;

  const api = useMemo(
    () => ({
      /** Apply a decision function to every row. The pipeline, as code. */
      forEachRow: (decide: (row: unknown) => CleaningAction) => {
        const rows = store.getState().rows;
        for (const row of rows) {
          const action = decide({
            id: row.id,
            features: { ...row.features },
            isNull: row.isNull,
            isOutlier: row.isOutlier,
          });
          if (!CLEANING_ACTIONS.includes(action)) {
            throw new Error(
              `Row ${row.id}: expected 'keep', 'impute', 'cap' or 'drop', got ${JSON.stringify(action)}`,
            );
          }
          store.getState().setRowAction(row.id, action);
        }
      },

      /** Decide one row by id. Same action the bins call. */
      setAction: (rowId: string, action: CleaningAction) =>
        store.getState().setRowAction(rowId, action),

      /** The rows, as plain data. */
      rows: () =>
        store.getState().rows.map((row) => ({
          id: row.id,
          features: { ...row.features },
          isNull: row.isNull,
          isOutlier: row.isOutlier,
          action: row.playerAction,
        })),

      kept: () => store.getState().pipeline.kept,
      dropped: () => store.getState().pipeline.dropped,
      blanksLeft: () => store.getState().pipeline.keptWithNaiveFill,
      extremesLeft: () => store.getState().pipeline.keptWithOutlier,
      balanceDrift: () => store.getState().pipeline.balanceDrift,
      binCounts: () => ({ ...store.getState().pipeline.binCounts }),

      /** Column medians and cap bounds, as the pipeline computes them. */
      stats: () => {
        const { stats } = store.getState();
        return {
          median: { ...stats.median },
          low: { ...stats.low },
          high: { ...stats.high },
        };
      },

      /** Retrain and return held-out accuracy. Real TF.js, awaited. */
      retrain,

      accuracy: () => store.getState().modelAccuracy,
      check: () => store.getState().check(),
      reset: () => store.getState().reset(),
    }),
    [store, retrain],
  );

  const lane = useCodeLane({ initialCode: STARTER_CODE, api, maxRunMs: 20000 });

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="primary"
          size="sm"
          onClick={() => void lane.run()}
          disabled={lane.running}
          icon={<Play className="size-4" />}
        >
          {lane.running ? "Running…" : "Run"}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={lane.reset}
          icon={<RotateCcw className="size-4" />}
        >
          Restore snippet
        </Button>
        {lane.dirty ? (
          <span className="font-mono text-xs text-text-muted">edited</span>
        ) : null}
      </div>

      <CodeEditor
        className="min-h-0 flex-1"
        label="Preprocessing pipeline"
        value={lane.code}
        onChange={lane.setCode}
        language="javascript"
        error={lane.error}
        rows={18}
        hint="api.forEachRow · api.setAction · api.rows · api.kept · api.dropped · api.blanksLeft · api.extremesLeft · api.balanceDrift · api.stats · api.retrain · api.check · api.reset"
      />

      <section aria-label="Script output" className="min-h-24">
        {/* h2: directly inside the lane, so it must not skip a level. */}
        <h2 className="mb-1 text-xs font-semibold tracking-wide text-text-muted uppercase">
          Output
        </h2>
        {lane.logs.length === 0 ? (
          <p className="font-mono text-xs text-text-muted">
            Run the script to see output here.
          </p>
        ) : (
          <ul className="max-h-40 overflow-auto rounded-md border border-border bg-bg p-2 font-mono text-xs">
            {lane.logs.map((entry, index) => (
              <li
                key={index}
                className={
                  entry.level === "error" ? "text-wrong" : "text-text-muted"
                }
              >
                {entry.message}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
