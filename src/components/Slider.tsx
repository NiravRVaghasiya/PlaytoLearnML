"use client";

import { useId } from "react";
import { clamp, cx } from "@/lib/utils";

export type ControlScale = "linear" | "log";

export interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  /** Linear step. Ignored when `scale` is "log". */
  step?: number;
  onChange: (value: number) => void;
  /**
   * Log scale for knobs that span orders of magnitude — learning rate is the
   * canonical case (0.001 → 2). Requires `min > 0`.
   */
  scale?: ControlScale;
  /** Render the live value. Defaults to 3 significant decimals. */
  format?: (value: number) => string;
  unit?: string;
  /** One line on what this control does to the algorithm, not the UI. */
  hint?: string;
  disabled?: boolean;
  className?: string;
}

/** Normalised position (0–1) → real value. */
export function fromPosition(
  position: number,
  min: number,
  max: number,
  scale: ControlScale,
): number {
  const t = clamp(position, 0, 1);
  if (scale === "log" && min > 0) {
    return min * Math.pow(max / min, t);
  }
  return min + (max - min) * t;
}

/** Real value → normalised position (0–1). */
export function toPosition(
  value: number,
  min: number,
  max: number,
  scale: ControlScale,
): number {
  if (scale === "log" && min > 0 && value > 0) {
    return clamp(Math.log(value / min) / Math.log(max / min), 0, 1);
  }
  if (max === min) return 0;
  return clamp((value - min) / (max - min), 0, 1);
}

const POSITION_STEPS = 1000;

/**
 * Continuous slider (DESIGN.md §6): shows its live value, 44px minimum target.
 *
 * Backed by a native `<input type="range">`, so keyboard support, screen-reader
 * semantics, and touch behaviour come from the platform rather than from
 * hand-rolled key handlers. On a log scale the input drives a normalised
 * position and `aria-valuetext` reports the real value, which is what the
 * player and the algorithm both care about.
 */
export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  scale = "linear",
  format,
  unit,
  hint,
  disabled = false,
  className,
}: SliderProps) {
  const id = useId();
  const hintId = `${id}-hint`;

  const render = format ?? ((v: number) => v.toPrecision(3));
  const display = `${render(value)}${unit ? ` ${unit}` : ""}`;

  const isLog = scale === "log" && min > 0;

  return (
    <div className={cx("w-full", className)}>
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="text-sm font-medium">
          {label}
        </label>
        {/* `<output>` is an implicit role="status" live region, so leaving it
            exposed made every change speak twice: once from the input's
            aria-valuetext, once from here — and code-lane changes were read out
            of context. The input's valuetext is the one channel. */}
        <output
          htmlFor={id}
          aria-hidden="true"
          className="font-mono text-sm tabular-nums"
        >
          {display}
        </output>
      </div>

      <input
        id={id}
        type="range"
        disabled={disabled}
        aria-describedby={hint ? hintId : undefined}
        aria-valuetext={display}
        // Log sliders operate on a normalised 0..1000 track.
        min={isLog ? 0 : min}
        max={isLog ? POSITION_STEPS : max}
        step={isLog ? 1 : (step ?? "any")}
        value={
          isLog ? Math.round(toPosition(value, min, max, scale) * POSITION_STEPS) : value
        }
        onChange={(event) => {
          const raw = Number(event.target.value);
          onChange(
            isLog ? fromPosition(raw / POSITION_STEPS, min, max, scale) : raw,
          );
        }}
        className={cx(
          "mt-2 h-11 w-full cursor-pointer accent-[var(--primary)]",
          disabled ? "cursor-not-allowed opacity-50" : null,
        )}
      />

      {hint ? (
        <p id={hintId} className="mt-1 text-xs text-text-muted">
          {hint}
        </p>
      ) : null}
    </div>
  );
}
