"use client";

import { cx } from "@/lib/utils";
import { nodeById, type NodeVerdict } from "./ml";
import { meterCaption } from "./why-cards";

export interface CorrectnessMeterProps {
  fraction: number;
  matched: number;
  total: number;
  answered: number;
  coincidences: number;
  verdicts: NodeVerdict[];
  showTruth: boolean;
}

const signed = (value: number) =>
  `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;

/**
 * Running gradient correctness (spec: `<CorrectnessMeter>`).
 *
 * The bar is the headline, but the table under it is what makes the meter usable:
 * it separates the nodes the player got wrong from the nodes that are wrong because
 * something downstream was. Without that split, one mistake near the loss turns the
 * whole list red and the player re-checks a dozen correct decisions looking for it.
 * "Inherited" is the most useful word this panel can say.
 */
export function CorrectnessMeter({
  fraction,
  matched,
  total,
  answered,
  coincidences,
  verdicts,
  showTruth,
}: CorrectnessMeterProps) {
  const inherited = verdicts.filter((verdict) => verdict.inherited).length;
  const mine = verdicts.filter(
    (verdict) => !verdict.correct && verdict.mine !== null && !verdict.inherited,
  ).length;

  return (
    <section aria-labelledby="meter-heading" className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-2">
        <h2 id="meter-heading" className="text-sm font-semibold">
          Gradient correctness
        </h2>
        <span className="font-mono text-xs tabular-nums">
          {matched}/{total}
        </span>
      </div>

      <div
        role="progressbar"
        aria-valuenow={Math.round(fraction * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Node gradients matching the autograd trace"
        className="h-2 w-full overflow-hidden rounded-sm bg-surface-2"
      >
        <div
          className="h-full rounded-sm transition-[width]"
          style={{
            width: `${Math.round(fraction * 100)}%`,
            background:
              fraction >= 1
                ? "var(--correct)"
                : fraction >= 0.5
                  ? "var(--warn)"
                  : "var(--wrong)",
          }}
        />
      </div>

      <p className="text-xs text-text-muted">
        {meterCaption(answered, fraction, coincidences)}
      </p>

      {mine > 0 || inherited > 0 ? (
        <p className="text-xs">
          <span className="text-wrong">{mine} wrong here</span>
          {inherited > 0 ? (
            <>
              {" · "}
              <span className="text-warn">
                {inherited} only wrong because something downstream is
              </span>
            </>
          ) : null}
        </p>
      ) : null}

      <table className="w-full text-xs">
        <caption className="sr-only-live">
          Every node gradient, yours against the autograd trace
        </caption>
        <thead className="text-text-muted">
          <tr>
            <th scope="col" className="py-1 text-left font-medium">
              Node
            </th>
            <th scope="col" className="py-1 text-right font-medium">
              Yours
            </th>
            {showTruth ? (
              <th scope="col" className="py-1 text-right font-medium">
                Autograd
              </th>
            ) : null}
            <th scope="col" className="py-1 pl-2 text-left font-medium">
              State
            </th>
          </tr>
        </thead>
        <tbody>
          {verdicts.map((verdict) => (
            <tr key={verdict.id} className="border-t border-border">
              <th scope="row" className="py-1 text-left font-normal">
                {nodeById(verdict.id).label}
              </th>
              <td
                className={cx(
                  "py-1 text-right font-mono tabular-nums",
                  verdict.mine === null
                    ? "text-text-muted"
                    : verdict.correct
                      ? "text-correct"
                      : "text-wrong",
                )}
              >
                {verdict.mine === null ? "—" : signed(verdict.mine)}
              </td>
              {showTruth ? (
                <td className="py-1 text-right font-mono tabular-nums text-text-muted">
                  {signed(verdict.truth)}
                </td>
              ) : null}
              <td className="py-1 pl-2 text-text-muted">
                {verdict.mine === null
                  ? "not routed"
                  : verdict.correct
                    ? "matches"
                    : verdict.inherited
                      ? "inherited"
                      : "wrong rule"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
