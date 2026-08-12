"use client";

import { useMemo } from "react";
import { line, scaleLinear } from "d3";
import type { RocPoint } from "./ml";

export interface RocCurveProps {
  curve: RocPoint[];
  point: RocPoint;
  auc: number;
}

const WIDTH = 260;
const HEIGHT = 260;
const PAD = 34;

/**
 * The ROC curve with the player's operating point (spec: `<ROCCurve>`).
 *
 * Built with d3's scale and line generators, which is the spec's stated stack for
 * this game and genuinely the right tool: the curve is a path through a few
 * hundred points and d3 owns that translation cleanly.
 *
 * The reason this view earns its space next to the matrix: the curve is a property
 * of the MODEL and does not move when the slider does. Only the dot moves. Once
 * that lands, "tune the threshold" stops sounding like "improve the model" — you
 * are choosing a point on a fixed frontier, and every point on it costs something.
 */
export function RocCurve({ curve, point, auc }: RocCurveProps) {
  const { path, x, y } = useMemo(() => {
    const xScale = scaleLinear().domain([0, 1]).range([PAD, WIDTH - 10]);
    const yScale = scaleLinear().domain([0, 1]).range([HEIGHT - PAD, 10]);
    const generator = line<RocPoint>()
      .x((entry) => xScale(entry.falsePositiveRate))
      .y((entry) => yScale(entry.truePositiveRate));
    return {
      path: generator(curve) ?? "",
      x: xScale,
      y: yScale,
    };
  }, [curve]);

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="w-full rounded-md border border-border bg-surface-2"
        role="img"
        aria-label={`ROC curve, area under curve ${auc.toFixed(
          3,
        )}. Your cutoff of ${point.threshold.toFixed(2)} sits at ${Math.round(
          point.falsePositiveRate * 100,
        )} percent false positive rate and ${Math.round(
          point.truePositiveRate * 100,
        )} percent recall. The curve is fixed by the model; moving the threshold only moves the point along it.`}
      >
        {/* Chance line. Anything below it is worse than guessing. */}
        <line
          x1={x(0)}
          y1={y(0)}
          x2={x(1)}
          y2={y(1)}
          stroke="var(--border)"
          strokeWidth={1}
          strokeDasharray="4 4"
        />

        <path d={path} fill="none" stroke="var(--primary)" strokeWidth={1.75} />

        {/* The operating point: the only thing the slider moves. */}
        <circle
          cx={x(point.falsePositiveRate)}
          cy={y(point.truePositiveRate)}
          r={5}
          fill="var(--class-b)"
          stroke="var(--bg)"
          strokeWidth={1.5}
        />

        {/* Axes */}
        <line
          x1={PAD}
          y1={HEIGHT - PAD}
          x2={WIDTH - 10}
          y2={HEIGHT - PAD}
          stroke="var(--border)"
        />
        <line x1={PAD} y1={10} x2={PAD} y2={HEIGHT - PAD} stroke="var(--border)" />

        <text
          x={(WIDTH + PAD) / 2}
          y={HEIGHT - 6}
          textAnchor="middle"
          className="fill-[var(--text-muted)] text-[9px]"
        >
          false positive rate
        </text>
        <text
          x={10}
          y={HEIGHT / 2}
          textAnchor="middle"
          transform={`rotate(-90 10 ${HEIGHT / 2})`}
          className="fill-[var(--text-muted)] text-[9px]"
        >
          recall (true positive rate)
        </text>
        <text
          x={WIDTH - 14}
          y={HEIGHT - PAD - 8}
          textAnchor="end"
          className="fill-[var(--text-muted)] text-[9px]"
        >
          AUC {auc.toFixed(3)}
        </text>
      </svg>

      <figcaption className="mt-1.5 text-xs text-text-muted">
        The curve belongs to the model and never moves. The orange dot is your
        cutoff — drag the slider and only the dot travels.
      </figcaption>
    </figure>
  );
}
