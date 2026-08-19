"use client";

import { cx } from "@/lib/utils";
import {
  VARIANCE_TARGET,
  type Analysis,
  type Separation,
} from "./ml";
import { gaugeCaption, separationCaption } from "./why-cards";

export interface VarianceGaugeProps {
  retained: number;
  analysis: Analysis;
  separation: Separation;
  separationShare: number;
  separable: boolean;
  submitted: boolean;
}

/**
 * Variance retained, and what it is a proxy for (spec: `<VarianceGauge>`).
 *
 * The gauge is the headline. The eigenvalue table underneath is the part that turns
 * it from a score into an explanation: three numbers that say in advance what any
 * projection of this cloud can possibly be worth, and which one you are currently
 * throwing away.
 *
 * The separation reading sits beside it rather than below it on purpose. These two
 * numbers are the game's central argument — that maximising the first does not
 * reliably maximise the second — and an argument is easier to notice when both
 * halves are visible at once.
 */
export function VarianceGauge({
  retained,
  analysis,
  separation,
  separationShare,
  separable,
  submitted,
}: VarianceGaugeProps) {
  const share = analysis.best <= 1e-12 ? 1 : retained / analysis.best;
  const total =
    analysis.eigen.values[0] + analysis.eigen.values[1] + analysis.eigen.values[2];

  return (
    <section aria-labelledby="gauge-heading" className="flex flex-col gap-3">
      <div className="flex items-baseline justify-between gap-2">
        <h2 id="gauge-heading" className="text-sm font-semibold">
          Variance retained
        </h2>
        <span className="font-mono text-xs tabular-nums">
          ceiling {(analysis.best * 100).toFixed(1)}%
        </span>
      </div>

      <div
        role="progressbar"
        aria-valuenow={Math.round(retained * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Fraction of the cloud's spread kept by this projection"
        className="relative h-3 w-full overflow-hidden rounded-sm bg-surface-2"
      >
        <div
          className="h-full rounded-sm transition-[width]"
          style={{
            width: `${Math.round(retained * 100)}%`,
            background:
              share >= VARIANCE_TARGET
                ? "var(--correct)"
                : share >= 0.85
                  ? "var(--warn)"
                  : "var(--wrong)",
          }}
        />
        {/* The ceiling, marked, so the bar is read against what is possible rather
            than against 100% — which is unreachable for any cloud with three real
            dimensions. */}
        <span
          aria-hidden="true"
          className="absolute top-0 h-full w-0.5 bg-[var(--text)]"
          style={{ left: `${analysis.best * 100}%` }}
        />
      </div>

      <p className="text-xs text-text-muted">
        {gaugeCaption(retained, analysis.best, submitted)}
      </p>

      <div className="rounded-md border border-border bg-surface-2 p-3">
        <h3 className="text-xs font-semibold">
          Variance along the cloud&apos;s own axes
        </h3>
        <table className="mt-1.5 w-full text-xs">
          <caption className="sr-only-live">
            The three eigenvalues of the covariance matrix, largest first, with each
            one&apos;s share of the total spread
          </caption>
          <thead className="text-text-muted">
            <tr>
              <th scope="col" className="text-left font-medium">
                Axis
              </th>
              <th scope="col" className="text-right font-medium">
                Variance
              </th>
              <th scope="col" className="text-right font-medium">
                Share
              </th>
            </tr>
          </thead>
          <tbody>
            {analysis.eigen.values.map((value, index) => (
              <tr key={index} className="border-t border-border">
                <th scope="row" className="py-1 text-left font-normal">
                  {index + 1}
                  {index === 2 ? (
                    <span className="ml-1.5 text-text-muted">
                      smallest — discard this one
                    </span>
                  ) : null}
                </th>
                <td className="py-1 text-right font-mono tabular-nums">
                  {value.toFixed(2)}
                </td>
                <td className="py-1 text-right font-mono tabular-nums">
                  {((value / total) * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-2 border-t border-border pt-2 text-xs text-text-muted">
          Keep the top two and you keep {(analysis.best * 100).toFixed(1)}%. That is
          the whole of what PCA computes.
        </p>
      </div>

      <div className="border-t border-border pt-3">
        <div className="flex items-baseline justify-between gap-2">
          <h3 className="text-xs font-semibold">Group separation</h3>
          <span
            className={cx(
              "font-mono text-xs tabular-nums",
              !separable
                ? "text-text-muted"
                : separationShare >= 0.85
                  ? "text-correct"
                  : separationShare >= 0.4
                    ? "text-warn"
                    : "text-wrong",
            )}
          >
            {separation.ratio.toFixed(2)}
          </span>
        </div>
        <p className="mt-1 text-xs text-text-muted">
          {separationCaption(
            separation.ratio,
            analysis.reachableSeparation.ratio,
            separable,
          )}
        </p>
      </div>
    </section>
  );
}
