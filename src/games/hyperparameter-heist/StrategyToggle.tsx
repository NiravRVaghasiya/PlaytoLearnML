"use client";

import { DIAL_COUNT, STRATEGY_LABELS, gridLevels, type Strategy } from "./ml";

export interface StrategyToggleProps {
  strategy: Strategy | "manual";
  budget: number;
  disabled: boolean;
  onChange: (strategy: Strategy | "manual") => void;
}

const OPTIONS: Array<{ value: Strategy | "manual"; label: string; hint: string }> = [
  { value: "manual", label: "By hand", hint: "you choose every dial" },
  { value: "grid", label: STRATEGY_LABELS.grid, hint: "every combination" },
  { value: "random", label: STRATEGY_LABELS.random, hint: "independent draws" },
  {
    value: "bayesian",
    label: STRATEGY_LABELS.bayesian,
    hint: "model, then pick",
  },
];

/**
 * Choosing a search strategy (spec: `<StrategyToggle>`).
 *
 * A radio group, because these are mutually exclusive modes and a native radio
 * group gives arrow-key navigation and a single tab stop for free — which is
 * exactly the right behaviour for four related choices.
 *
 * Each option states what it does in three words. The grid option also states its
 * cost, because k^d is the number that makes the decision and nobody computes it
 * in their head.
 *
 * Every option is at least 44px tall — DESIGN.md's touch-target minimum. These
 * are the game's primary mode controls, and the whole label is the target.
 */
export function StrategyToggle({
  strategy,
  budget,
  disabled,
  onChange,
}: StrategyToggleProps) {
  const levels = gridLevels(budget);

  return (
    <fieldset>
      <legend className="text-sm font-medium">Search strategy</legend>
      <div className="mt-2 flex flex-col gap-1">
        {OPTIONS.map((option) => (
          <label
            key={option.value}
            className={`flex min-h-11 cursor-pointer items-center gap-2 rounded border px-2 py-1.5 text-xs ${
              option.value === strategy
                ? "border-primary bg-primary/10"
                : "border-border bg-surface-2 hover:bg-surface"
            } ${disabled ? "cursor-not-allowed opacity-60" : ""}`}
          >
            <input
              type="radio"
              name="search-strategy"
              value={option.value}
              checked={option.value === strategy}
              disabled={disabled}
              onChange={() => onChange(option.value)}
              className="accent-[var(--primary)]"
            />
            <span className="font-medium">{option.label}</span>
            <span className="text-text-muted">— {option.hint}</span>
            {option.value === "grid" ? (
              <span className="ml-auto font-mono text-[10px] text-text-muted">
                {levels}
                <sup>{DIAL_COUNT}</sup> = {levels ** DIAL_COUNT}
              </span>
            ) : null}
          </label>
        ))}
      </div>
    </fieldset>
  );
}
