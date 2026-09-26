"use client";

import { useRef } from "react";
import { Code2, MousePointerClick } from "lucide-react";
import { cx } from "@/lib/utils";
import type { Lane } from "@/engine/types";

export interface LaneToggleProps {
  value: Lane;
  onChange: (lane: Lane) => void;
  /** Set when a game's code lane isn't wired yet; keeps the control honest. */
  codeDisabled?: boolean;
  codeDisabledReason?: string;
  className?: string;
}

const OPTIONS: ReadonlyArray<{
  lane: Lane;
  label: string;
  Icon: typeof Code2;
}> = [
  { lane: "visual", label: "Visual", Icon: MousePointerClick },
  { lane: "code", label: "Code", Icon: Code2 },
];

/**
 * The Visual ⇄ Code segmented control (DESIGN.md §5, §6).
 *
 * Implemented as a real radiogroup so arrow keys move between lanes and screen
 * readers announce "Visual, radio button, 1 of 2, selected". Roving tabindex
 * keeps the group a single tab stop, and — per the APG radio pattern — an arrow
 * key moves focus AND selection together. Moving only the selection left focus
 * on the radio that had just become unchecked, so a screen reader said
 * "Visual, not checked" and never announced the lane that was chosen.
 *
 * Game state persists across the toggle because both lanes read the same
 * Zustand store — the shell swaps which view is mounted, it never owns the
 * data (two-lane rule, CLAUDE.md).
 */
export function LaneToggle({
  value,
  onChange,
  codeDisabled = false,
  codeDisabledReason,
  className,
}: LaneToggleProps) {
  const isDisabled = (lane: Lane) => lane === "code" && codeDisabled;
  const buttonRefs = useRef<Array<HTMLButtonElement | null>>([]);

  const move = (direction: 1 | -1) => {
    const index = OPTIONS.findIndex((o) => o.lane === value);
    const nextIndex = (index + direction + OPTIONS.length) % OPTIONS.length;
    const next = OPTIONS[nextIndex];
    if (!next || isDisabled(next.lane)) return;
    onChange(next.lane);
    buttonRefs.current[nextIndex]?.focus();
  };

  return (
    <div
      role="radiogroup"
      aria-label="Lane"
      className={cx(
        "inline-flex rounded-full border border-border bg-surface-2 p-1",
        className,
      )}
    >
      {OPTIONS.map(({ lane, label, Icon }, optionIndex) => {
        const selected = value === lane;
        const disabled = isDisabled(lane);

        return (
          <button
            key={lane}
            ref={(node) => {
              buttonRefs.current[optionIndex] = node;
            }}
            type="button"
            role="radio"
            aria-checked={selected}
            aria-disabled={disabled || undefined}
            title={disabled ? codeDisabledReason : undefined}
            // Roving tabindex: the group is one tab stop.
            tabIndex={selected ? 0 : -1}
            onClick={() => {
              if (!disabled) onChange(lane);
            }}
            // Arrow keys live on the radios, not the group: the group is a
            // container and must not be a focus stop (ARIA APG radiogroup
            // pattern), so it can't sensibly own key handling.
            onKeyDown={(event) => {
              if (event.key === "ArrowRight" || event.key === "ArrowDown") {
                event.preventDefault();
                move(1);
              } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
                event.preventDefault();
                move(-1);
              }
            }}
            className={cx(
              "inline-flex min-h-11 items-center gap-2 rounded-full px-4 text-sm font-medium",
              "transition-colors dur-micro",
              selected
                ? "bg-primary text-primary-ink"
                : "text-text-muted hover:text-text",
              disabled ? "cursor-not-allowed opacity-40" : null,
            )}
          >
            <Icon aria-hidden="true" className="size-4" />
            {label}
          </button>
        );
      })}
    </div>
  );
}
