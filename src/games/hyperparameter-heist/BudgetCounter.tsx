"use client";

export interface BudgetCounterProps {
  used: number;
  budget: number;
  bestObjective: number;
  crackThreshold: number;
}

/**
 * Tries spent against tries left (spec: `<BudgetCounter>`).
 *
 * The compute budget is the entire constraint of the game — without it, every
 * strategy eventually finds the optimum and there is nothing to teach. So it is
 * shown as a depleting resource rather than a running total.
 */
export function BudgetCounter({
  used,
  budget,
  bestObjective,
  crackThreshold,
}: BudgetCounterProps) {
  const left = Math.max(0, budget - used);
  const fraction = Math.min(1, used / Math.max(1, budget));
  const cracked = bestObjective >= crackThreshold;

  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-sm font-medium">Compute budget</span>
        <span className="font-mono text-sm">
          {left}
          <span className="text-text-muted"> / {budget} left</span>
        </span>
      </div>

      <div
        role="progressbar"
        aria-valuenow={used}
        aria-valuemin={0}
        aria-valuemax={budget}
        aria-label={`Compute budget: ${used} of ${budget} tries spent, ${left} remaining`}
        className="mt-1.5 h-2.5 w-full overflow-hidden rounded-full bg-surface-2"
      >
        <div
          className={`h-full rounded-full transition-[width] duration-standard ${
            cracked ? "bg-correct" : left <= 3 ? "bg-wrong" : "bg-primary"
          }`}
          style={{ width: `${fraction * 100}%` }}
        />
      </div>

      <p className="mt-1.5 text-xs text-text-muted">
        {cracked
          ? `Open on try ${used}. Every try you did not need is score.`
          : left === 0
            ? "No tries left."
            : `Each try buys one reading of the objective. Nothing else.`}
      </p>
    </div>
  );
}
