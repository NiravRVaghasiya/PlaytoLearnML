"use client";

import { clamp } from "@/lib/utils";

export interface BudgetBarProps {
  used: number;
  budget: number;
}

/**
 * The compute budget (spec: "a compute budget caps total neurons, forcing
 * efficient designs").
 *
 * A real `<progress>`-style bar rather than a bare number, because the thing the
 * player needs to feel is how much room is left — and it fills toward a cap that
 * the controls physically will not let them cross.
 */
export function BudgetBar({ used, budget }: BudgetBarProps) {
  const fraction = clamp(used / Math.max(1, budget), 0, 1);
  const full = used >= budget;

  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-sm font-medium">Compute budget</span>
        <span className="font-mono text-sm">
          {used}
          <span className="text-text-muted"> / {budget}</span>
        </span>
      </div>

      <div
        role="progressbar"
        aria-valuenow={used}
        aria-valuemin={0}
        aria-valuemax={budget}
        aria-label={`Compute budget: ${used} of ${budget} neurons used`}
        className="mt-1.5 h-2.5 w-full overflow-hidden rounded-full bg-surface-2"
      >
        <div
          className={`h-full rounded-full transition-[width] duration-standard ${
            full ? "bg-warn" : "bg-primary"
          }`}
          style={{ width: `${fraction * 100}%` }}
        />
      </div>

      <p className="mt-1.5 text-xs text-text-muted">
        {full
          ? "Budget full. From here it is rearranging, not adding — depth and activation are still free to change."
          : `${budget - used} neuron${budget - used === 1 ? "" : "s"} left to spend.`}
      </p>
    </div>
  );
}
