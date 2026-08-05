"use client";

import { useCallback, useId, useRef } from "react";
import { clamp, cx } from "@/lib/utils";
import { fromPosition, toPosition, type ControlScale } from "./Slider";

export interface DialProps {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
  scale?: ControlScale;
  format?: (value: number) => string;
  unit?: string;
  hint?: string;
  disabled?: boolean;
  /** Fraction of the range moved by an arrow key. Default 2%. */
  keyStep?: number;
  className?: string;
}

/** Sweep angle of the dial track, in degrees (270° gap at the bottom). */
const SWEEP = 270;
const START_ANGLE = -135;

/**
 * Rotary dial (DESIGN.md §6). Used where a knob reads better than a slider —
 * the Skier's learning-rate dial, the Heist's safe dials.
 *
 * Accessibility is the hard part of a custom control, so this exposes
 * `role="slider"` with full keyboard support (arrows, Page Up/Down, Home/End)
 * and an `aria-valuetext` carrying the real value. Vertical pointer drag maps to
 * value, which is more predictable than absolute angle tracking and avoids the
 * jump-on-grab problem. The hit area is 44×44 minimum.
 */
export function Dial({
  label,
  value,
  min,
  max,
  onChange,
  scale = "linear",
  format,
  unit,
  hint,
  disabled = false,
  keyStep = 0.02,
  className,
}: DialProps) {
  const id = useId();
  const hintId = `${id}-hint`;
  const dragRef = useRef<{ startY: number; startPosition: number } | null>(null);

  const position = toPosition(value, min, max, scale);
  const render = format ?? ((v: number) => v.toPrecision(3));
  const display = `${render(value)}${unit ? ` ${unit}` : ""}`;

  const setPosition = useCallback(
    (next: number) => {
      onChange(fromPosition(clamp(next, 0, 1), min, max, scale));
    },
    [onChange, min, max, scale],
  );

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (disabled) return;

    const big = keyStep * 5;
    const handlers: Record<string, () => void> = {
      ArrowUp: () => setPosition(position + keyStep),
      ArrowRight: () => setPosition(position + keyStep),
      ArrowDown: () => setPosition(position - keyStep),
      ArrowLeft: () => setPosition(position - keyStep),
      PageUp: () => setPosition(position + big),
      PageDown: () => setPosition(position - big),
      Home: () => setPosition(0),
      End: () => setPosition(1),
    };

    const handler = handlers[event.key];
    if (handler) {
      event.preventDefault();
      handler();
    }
  };

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (disabled) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { startY: event.clientY, startPosition: position };
  };

  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || disabled) return;
    // 160px of vertical travel covers the full range; up increases.
    const delta = (drag.startY - event.clientY) / 160;
    setPosition(drag.startPosition + delta);
  };

  const endDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const angle = START_ANGLE + position * SWEEP;

  return (
    <div className={cx("flex flex-col items-center gap-2", className)}>
      <span id={`${id}-label`} className="text-sm font-medium">
        {label}
      </span>

      <div
        role="slider"
        tabIndex={disabled ? -1 : 0}
        aria-labelledby={`${id}-label`}
        aria-describedby={hint ? hintId : undefined}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={Number(value.toPrecision(4))}
        aria-valuetext={display}
        aria-disabled={disabled || undefined}
        aria-orientation="vertical"
        onKeyDown={onKeyDown}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        className={cx(
          "relative size-16 touch-none rounded-full border-2 border-border bg-surface-2",
          disabled ? "cursor-not-allowed opacity-50" : "cursor-ns-resize",
        )}
        style={{
          background: `conic-gradient(from ${START_ANGLE + 90}deg, var(--primary) 0deg ${position * SWEEP}deg, var(--surface-2) ${position * SWEEP}deg ${SWEEP}deg, transparent ${SWEEP}deg)`,
        }}
      >
        {/* Pointer needle. Decorative — the value is in aria-valuetext and the
            visible readout below. */}
        <span
          aria-hidden="true"
          className="absolute top-1/2 left-1/2 h-6 w-0.5 origin-bottom rounded-full bg-text"
          style={{
            transform: `translate(-50%, -100%) rotate(${angle}deg)`,
            transformOrigin: "bottom center",
          }}
        />
      </div>

      <output className="font-mono text-sm tabular-nums">{display}</output>

      {hint ? (
        <p id={hintId} className="max-w-40 text-center text-xs text-text-muted">
          {hint}
        </p>
      ) : null}
    </div>
  );
}
