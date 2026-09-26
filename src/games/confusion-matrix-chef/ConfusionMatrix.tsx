"use client";

import type { ConfusionMatrix as Matrix, Scenario } from "./ml";

export interface ConfusionMatrixProps {
  matrix: Matrix;
  scenario: Scenario;
  total: number;
}

/**
 * The live 2×2 matrix (spec: `<ConfusionMatrix>`).
 *
 * A real `<table>` with header cells, not a grid of divs. The matrix IS tabular
 * data, and marking it up as such means a screen reader announces "Actually a
 * contaminated dish, flagged: 43" — the row and column context comes for free,
 * where a div grid would read out four bare numbers in a row.
 *
 * Each cell carries the scenario's own language as well as the abstract name. "38
 * missed contaminated dishes" is the thing the player has to care about; "false
 * negatives" is the thing they have to learn to call it. Showing both is how the
 * second becomes memorable.
 */
export function ConfusionMatrix({
  matrix,
  scenario,
  total,
}: ConfusionMatrixProps) {
  const { truePositives, falsePositives, trueNegatives, falseNegatives } = matrix;

  const cell = (
    count: number,
    abbreviation: string,
    name: string,
    story: string,
    good: boolean,
  ) => (
    <td
      data-testid={`cell-${abbreviation.toLowerCase()}`}
      className={`border p-2 align-top ${
        good
          ? "border-correct/40 bg-correct/10"
          : "border-wrong/40 bg-wrong/10"
      }`}
    >
      <span className="flex items-baseline gap-1.5">
        <span className="font-mono text-lg font-semibold">{count}</span>
        {/* Explicit space: without a text node between the spans the cell's
            textContent runs together as "43TP", which reads badly aloud. */}{" "}
        <span className="font-mono text-[10px] text-text-muted">
          {abbreviation}
        </span>
      </span>
      <span className="mt-0.5 block text-[11px] font-medium">{name}</span>
      <span className="mt-0.5 block text-[10px] text-text-muted">{story}</span>
    </td>
  );

  return (
    <figure className="m-0">
      <table className="w-full border-collapse text-left">
        <caption className="mb-1.5 text-left text-xs text-text-muted">
          {total} cases at this cutoff. Green cells are correct decisions, red are
          the two kinds of mistake — which one you can afford is what the brief
          decides.
        </caption>
        <thead>
          <tr>
            <td className="p-1" />
            <th
              scope="col"
              className="border border-border bg-surface-2 p-2 text-xs font-medium"
            >
              You flagged it
              <span className="mt-0.5 block text-[10px] font-normal text-text-muted">
                {scenario.flagAction}
              </span>
            </th>
            <th
              scope="col"
              className="border border-border bg-surface-2 p-2 text-xs font-medium"
            >
              You let it through
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th
              scope="row"
              className="border border-border bg-surface-2 p-2 text-xs font-medium"
            >
              Really a {scenario.positiveLabel}
            </th>
            {cell(
              truePositives,
              "TP",
              "Caught",
              `${scenario.positiveLabelPlural} stopped`,
              true,
            )}
            {cell(
              falseNegatives,
              "FN",
              "Missed",
              scenario.falseNegativeCost,
              false,
            )}
          </tr>
          <tr>
            <th
              scope="row"
              className="border border-border bg-surface-2 p-2 text-xs font-medium"
            >
              Actually fine
            </th>
            {cell(
              falsePositives,
              "FP",
              "False alarm",
              scenario.falsePositiveCost,
              false,
            )}
            {cell(
              trueNegatives,
              "TN",
              "Correctly passed",
              "no action needed",
              true,
            )}
          </tr>
        </tbody>
      </table>
    </figure>
  );
}
