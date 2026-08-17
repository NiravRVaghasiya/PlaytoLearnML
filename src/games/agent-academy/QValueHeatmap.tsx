"use client";

import { ACTIONS, ACTION_ARROWS, ACTION_LABELS, type QTable } from "./ml";

export interface QValueHeatmapProps {
  qTable: QTable;
  /** Which cell to open up. Defaults to the start. */
  state: number;
  label: string;
  episodesUsed: number;
}

/**
 * One cell's four Q-values (spec: `<QValueHeatmap>`).
 *
 * The grid already carries the heat map across all 28 cells — a shade plus the
 * best value printed in the corner. What that view cannot show is the thing a
 * policy is actually made of: the FOUR numbers in a single cell, and how close the
 * decision between them was.
 *
 * That gap matters here more than usual. "The agent chose to go right" is not
 * interesting; "the agent rates right at +11.6 and up at 0.0, and the 0.0 is not a
 * judgement, it is an action it has never once taken" is the entire explanation for
 * why epsilon exists. So this component opens one cell and lists its row, marking
 * which entries have never been updated.
 */
export function QValueHeatmap({
  qTable,
  state,
  label,
  episodesUsed,
}: QValueHeatmapProps) {
  const row = qTable[state] ?? [0, 0, 0, 0];
  const best = Math.max(...row);
  const untouched = row.filter((value) => value === 0).length;
  const magnitude = Math.max(...row.map((value) => Math.abs(value)), 1e-6);

  return (
    <section aria-labelledby="qvalues-heading" className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-2">
        {/*
         * h2, not h3. This section is a direct child of the shell's <main>, whose
         * only heading ancestor is the page h1 — an h3 here skips a level and
         * Lighthouse flags it. The control rail's h3s are fine because they sit
         * under the rail's own "Controls" h2.
         */}
        <h2 id="qvalues-heading" className="text-sm font-semibold">
          Q-values at {label}
        </h2>
        <span className="text-xs text-text-muted">
          {episodesUsed === 0
            ? "all four still 0"
            : untouched === 0
              ? "all four visited"
              : `${untouched} never tried`}
        </span>
      </div>

      <table className="w-full text-xs">
        <caption className="sr-only-live">
          The four action values the agent has learned for {label}, and which
          actions it has never taken from there
        </caption>
        <thead>
          <tr className="text-text-muted">
            <th scope="col" className="py-1 text-left font-medium">
              Action
            </th>
            <th scope="col" className="py-1 text-right font-medium">
              Q(s,a)
            </th>
            <th scope="col" className="py-1 pl-3 text-left font-medium">
              <span className="sr-only-live">Relative size</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {ACTIONS.map((action) => {
            const value = row[action] ?? 0;
            const chosen = value === best;
            const never = value === 0;
            return (
              <tr key={action} className="border-t border-border">
                <th scope="row" className="py-1.5 text-left font-normal">
                  <span aria-hidden="true" className="mr-1.5">
                    {ACTION_ARROWS[action]}
                  </span>
                  {ACTION_LABELS[action]}
                  {chosen ? (
                    <span className="ml-1.5 rounded-sm bg-[var(--primary)] px-1 text-[10px] font-medium text-[var(--primary-ink)]">
                      chosen
                    </span>
                  ) : null}
                </th>
                <td className="py-1.5 text-right font-mono tabular-nums">
                  {value.toFixed(2)}
                </td>
                <td className="py-1.5 pl-3">
                  <span className="flex h-2 w-full items-center">
                    <span
                      aria-hidden="true"
                      className="block h-2 rounded-sm"
                      style={{
                        width: `${(Math.abs(value) / magnitude) * 100}%`,
                        background: never
                          ? "var(--border)"
                          : value >= 0
                            ? "var(--class-a)"
                            : "var(--class-b)",
                      }}
                    />
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {untouched > 0 ? (
        <p className="text-xs text-text-muted">
          A 0 here is not a low opinion, it is the absence of one — that action has
          never been taken from this square, so nothing has ever updated it. An
          agent at ε 0 picks the largest of these numbers, which means a 0 can only
          ever be beaten, never corrected.
        </p>
      ) : null}
    </section>
  );
}
