"use client";

import type { ClassCounts } from "./ml";

export interface PurityPieProps {
  counts: ClassCounts;
  gini: number;
  size?: number;
  /** Rendered inside a diagram that already has a description. */
  decorative?: boolean;
}

/**
 * Class mix at a node, as a pie (spec: `<PurityPie>`).
 *
 * Colour is never the only cue. The wedge sizes carry the mix, the ring thickens
 * as impurity rises, and the counts are printed underneath — so the node is
 * readable in greyscale and by anyone who cannot see the pie at all.
 *
 * The ring is the part worth having: a pure node gets a thin quiet outline and a
 * 50/50 node gets a thick warning one, which makes "how much work is left here"
 * scannable across a whole tree without reading any numbers.
 */
export function PurityPie({
  counts,
  gini,
  size = 34,
  decorative = false,
}: PurityPieProps) {
  const radius = size / 2;
  const safeFraction = counts.total === 0 ? 0 : counts.positive / counts.total;

  // Single wedge for the positive class, drawn as a stroked arc so a 0% or 100%
  // node degenerates cleanly instead of producing an invalid path.
  const circumference = 2 * Math.PI * (radius - 4);

  const label =
    counts.total === 0
      ? "Empty node, no training plots reach it."
      : `${counts.positive} safe and ${counts.negative} unsafe of ${
          counts.total
        } plots, impurity ${gini.toFixed(2)}.`;

  return (
    <span className="inline-flex flex-col items-center gap-0.5">
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        {...(decorative
          ? { "aria-hidden": true as const }
          : { role: "img" as const, "aria-label": label })}
      >
        <circle
          cx={radius}
          cy={radius}
          r={radius - 4}
          fill="none"
          stroke="var(--class-a)"
          strokeWidth={6}
        />
        {safeFraction > 0 ? (
          <circle
            cx={radius}
            cy={radius}
            r={radius - 4}
            fill="none"
            stroke="var(--class-b)"
            strokeWidth={6}
            strokeDasharray={`${circumference * safeFraction} ${circumference}`}
            transform={`rotate(-90 ${radius} ${radius})`}
          />
        ) : null}
        {/* Impurity ring: thin when pure, thick when evenly mixed. */}
        <circle
          cx={radius}
          cy={radius}
          r={radius - 1}
          fill="none"
          stroke={gini > 0.35 ? "var(--warn)" : "var(--border)"}
          strokeWidth={Math.max(0.5, gini * 3)}
        />
      </svg>
      <span className="font-mono text-[9px] leading-none text-text-muted">
        {gini.toFixed(2)}
      </span>
    </span>
  );
}
