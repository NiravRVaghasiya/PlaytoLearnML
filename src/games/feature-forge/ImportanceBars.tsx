"use client";

import { useMemo } from "react";

export interface ImportanceBarsProps {
  columnNames: string[];
  importances: number[];
  /** Cap the list so a 24-column matrix does not fill the lane. */
  limit?: number;
}

/**
 * |weight| per matrix column, re-ranked after every retrain (spec:
 * `<ImportanceBars>`).
 *
 * A real `<table>` rather than a row of divs: it is a ranked list of names and
 * numbers, which is what a table is for, and it means a screen reader announces
 * "bin(age)[4], 1.23" instead of two unrelated fragments.
 *
 * What this shows is the weight the FIXED linear model put on each column, which is
 * only interpretable because the model is linear and every column is standardised
 * or one-hot. That is worth saying in the caption — "feature importance" from a
 * tree or a deep net does not mean the same thing, and the habit of reading these
 * bars as truth is worth complicating early.
 */
export function ImportanceBars({
  columnNames,
  importances,
  limit = 10,
}: ImportanceBarsProps) {
  const ranked = useMemo(() => {
    const rows = columnNames.map((name, index) => ({
      name,
      weight: importances[index] ?? 0,
    }));
    return rows.sort((a, b) => b.weight - a.weight).slice(0, limit);
  }, [columnNames, importances, limit]);

  if (ranked.length === 0) {
    return (
      <p className="text-xs text-text-muted">
        No model fitted yet. Weights appear here after the first training run.
      </p>
    );
  }

  const top = ranked[0]!.weight || 1;

  return (
    <table className="w-full border-collapse text-left text-xs">
      <caption className="mb-1 text-left text-[11px] text-text-muted">
        Absolute weight the fixed model gave each matrix column. Only readable
        because the model is linear and every column is on a comparable scale —
        &quot;importance&quot; from a tree or a deep net is a different quantity.
      </caption>
      <thead>
        <tr className="border-b border-border">
          <th scope="col" className="py-1 font-medium">
            Column
          </th>
          <th scope="col" className="py-1 text-right font-medium">
            |weight|
          </th>
        </tr>
      </thead>
      <tbody>
        {ranked.map((row) => (
          <tr key={row.name} className="border-b border-border/50">
            <th scope="row" className="w-1/2 py-1 font-mono font-normal">
              {row.name}
            </th>
            <td className="py-1">
              <span className="flex items-center gap-2">
                <span
                  aria-hidden="true"
                  className="inline-block h-2 rounded-sm bg-primary"
                  style={{ width: `${Math.max(2, (row.weight / top) * 100)}%` }}
                />
                <span className="ml-auto shrink-0 font-mono">
                  {row.weight.toFixed(3)}
                </span>
              </span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
