"use client";

import { useMemo } from "react";
import { scaleLinear } from "d3";
import { cx } from "@/lib/utils";
import type { ElbowPoint } from "./ml";

export interface ElbowMeterProps {
  curve: ElbowPoint[];
  /** The player's current k, highlighted on the chart. */
  currentK: number;
  className?: string;
}

/**
 * The elbow chart (spec: `<ElbowMeter>`).
 *
 * Plots the best achievable inertia at every k. This is the honest answer to
 * "how many clusters are there?" — and it is deliberately presented as a chart to
 * read rather than a verdict to obey. The kink is where extra flags stop buying
 * real structure; spotting it is the skill the game teaches, so the component
 * never labels it.
 *
 * The chart is decorative to assistive tech; the data is exposed as a real table
 * underneath, which is both more useful than any aria-label and lets a
 * screen-reader user do the same reasoning by comparing numbers.
 */
export function ElbowMeter({ curve, currentK, className }: ElbowMeterProps) {
  const width = 260;
  const height = 110;
  const pad = { top: 8, right: 8, bottom: 20, left: 30 };

  const { xScale, yScale, path } = useMemo(() => {
    if (curve.length === 0) {
      return { xScale: null, yScale: null, path: "" };
    }

    const ks = curve.map((point) => point.k);
    const inertias = curve.map((point) => point.inertia);

    const x = scaleLinear()
      .domain([Math.min(...ks), Math.max(...ks)])
      .range([pad.left, width - pad.right]);
    const y = scaleLinear()
      .domain([0, Math.max(...inertias)])
      .range([height - pad.bottom, pad.top]);

    return {
      xScale: x,
      yScale: y,
      path: curve
        .map(
          (point, index) =>
            `${index === 0 ? "M" : "L"}${x(point.k).toFixed(1)},${y(point.inertia).toFixed(1)}`,
        )
        .join(" "),
    };
  }, [curve, height, pad.bottom, pad.left, pad.right, pad.top, width]);

  if (!xScale || !yScale) return null;

  return (
    <div className={cx("w-full", className)}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        aria-hidden="true"
        focusable="false"
      >
        <line
          x1={pad.left}
          y1={height - pad.bottom}
          x2={width - pad.right}
          y2={height - pad.bottom}
          stroke="var(--border)"
          strokeWidth={1}
        />
        <line
          x1={pad.left}
          y1={pad.top}
          x2={pad.left}
          y2={height - pad.bottom}
          stroke="var(--border)"
          strokeWidth={1}
        />

        <path d={path} fill="none" stroke="var(--primary)" strokeWidth={1.6} />

        {curve.map((point) => {
          const isCurrent = point.k === currentK;
          return (
            <g key={point.k}>
              {isCurrent ? (
                <line
                  x1={xScale(point.k)}
                  y1={pad.top}
                  x2={xScale(point.k)}
                  y2={height - pad.bottom}
                  stroke="var(--xp)"
                  strokeWidth={1}
                  strokeDasharray="2 2"
                />
              ) : null}
              <circle
                cx={xScale(point.k)}
                cy={yScale(point.inertia)}
                r={isCurrent ? 3.4 : 2}
                fill={isCurrent ? "var(--xp)" : "var(--primary)"}
              />
              <text
                x={xScale(point.k)}
                y={height - pad.bottom + 13}
                textAnchor="middle"
                fontSize={9}
                fill={isCurrent ? "var(--xp)" : "var(--text-muted)"}
              >
                {point.k}
              </text>
            </g>
          );
        })}

        <text
          x={pad.left - 4}
          y={pad.top + 6}
          textAnchor="end"
          fontSize={8}
          fill="var(--text-muted)"
        >
          high
        </text>
        <text
          x={pad.left - 4}
          y={height - pad.bottom}
          textAnchor="end"
          fontSize={8}
          fill="var(--text-muted)"
        >
          0
        </text>
      </svg>

      {/* The same data as a table: better than an aria-label, and it lets a
          screen-reader user find the elbow by comparing the numbers. */}
      <details className="mt-1">
        <summary className="cursor-pointer text-xs text-text-muted">
          Inertia at each k (table)
        </summary>
        <table className="mt-2 w-full font-mono text-xs">
          <caption className="sr-only">
            Best achievable inertia for each number of flags. The elbow is where
            the drop between rows becomes small.
          </caption>
          <thead>
            <tr className="text-text-muted">
              <th scope="col" className="text-left font-medium">
                Flags
              </th>
              <th scope="col" className="text-right font-medium">
                Inertia
              </th>
              <th scope="col" className="text-right font-medium">
                Drop
              </th>
            </tr>
          </thead>
          <tbody>
            {curve.map((point, index) => {
              const previous = curve[index - 1];
              const drop = previous
                ? `${(((previous.inertia - point.inertia) / previous.inertia) * 100).toFixed(0)}%`
                : "—";
              return (
                <tr
                  key={point.k}
                  className={point.k === currentK ? "text-xp" : undefined}
                >
                  <th scope="row" className="text-left font-normal">
                    {point.k}
                    {point.k === currentK ? " (yours)" : ""}
                  </th>
                  <td className="text-right">{point.inertia.toFixed(2)}</td>
                  <td className="text-right">{drop}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </details>
    </div>
  );
}
