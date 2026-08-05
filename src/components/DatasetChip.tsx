"use client";

import { Check, Lock } from "lucide-react";
import { cx } from "@/lib/utils";

export interface DatasetChipProps {
  name: string;
  /** e.g. "150 rows · 4 features". */
  detail?: string;
  selected?: boolean;
  /** Premium/unlockable datasets (spec §4, §8). */
  locked?: boolean;
  lockedReason?: string;
  onSelect?: () => void;
  className?: string;
}

/**
 * Selectable dataset chip (DESIGN.md §6), with a locked state for datasets
 * gated behind progression or a paid tier.
 *
 * Selection and lock state are both carried by an icon and the accessible name
 * as well as by colour, so the control still reads correctly in greyscale
 * (DESIGN.md §9).
 */
export function DatasetChip({
  name,
  detail,
  selected = false,
  locked = false,
  lockedReason,
  onSelect,
  className,
}: DatasetChipProps) {
  const label = locked
    ? `${name}, locked${lockedReason ? `: ${lockedReason}` : ""}`
    : name;

  return (
    <button
      type="button"
      aria-pressed={locked ? undefined : selected}
      aria-disabled={locked || undefined}
      aria-label={label}
      title={locked ? lockedReason : undefined}
      onClick={() => {
        if (!locked) onSelect?.();
      }}
      className={cx(
        "inline-flex min-h-11 items-center gap-2 rounded-full border px-3 text-sm",
        "transition-colors dur-micro",
        selected && !locked
          ? "border-primary bg-primary/15 text-text"
          : "border-border bg-surface-2 text-text-muted hover:text-text",
        locked ? "cursor-not-allowed opacity-60" : null,
        className,
      )}
    >
      {locked ? (
        <Lock aria-hidden="true" className="size-4 shrink-0" />
      ) : selected ? (
        <Check aria-hidden="true" className="size-4 shrink-0 text-primary" />
      ) : null}
      <span className="font-medium">{name}</span>
      {detail ? (
        <span className="font-mono text-xs text-text-muted">{detail}</span>
      ) : null}
    </button>
  );
}
