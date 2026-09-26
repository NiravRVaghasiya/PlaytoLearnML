"use client";

import { useCallback, useId, useRef, useState } from "react";
import { Minus, Plus } from "lucide-react";
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
 *
 * Dragging is never the only pointer path: − / + buttons either side step the
 * value one arrow-key step per tap (WCAG 2.2 SC 2.5.7), so a player who can't
 * drag — or is on a touch screen with no keyboard — can still set it.
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
  /**
   * Where the last − / + tap sent the value (see the live region), or null
   * once the dial itself took over. Compared with the current value rather
   * than kept as an on/off flag, so a later change the tap didn't make (the
   * code lane's `setLearningRate`, a reset) is not read out of context.
   */
  const [steppedTo, setSteppedTo] = useState<number | null>(null);

  const position = toPosition(value, min, max, scale);
  // Within half a step, because a game may round what it stores.
  const showStep =
    steppedTo !== null &&
    Math.abs(toPosition(steppedTo, min, max, scale) - position) < keyStep / 2;
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
      setSteppedTo(null);
      handler();
    }
  };

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (disabled) return;
    setSteppedTo(null);
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

  const step = (direction: 1 | -1) => {
    if (disabled) return;
    const next = fromPosition(
      clamp(position + direction * keyStep, 0, 1),
      min,
      max,
      scale,
    );
    setSteppedTo(next);
    onChange(next);
  };

  const stepButton = (direction: 1 | -1) => {
    const Icon = direction > 0 ? Plus : Minus;
    return (
      <button
        type="button"
        onClick={() => step(direction)}
        disabled={disabled}
        aria-label={`${direction > 0 ? "Increase" : "Decrease"} ${label}`}
        className={cx(
          "inline-flex size-11 shrink-0 items-center justify-center rounded-full",
          "border border-border bg-surface-2 text-text transition-colors dur-micro",
          "hover:border-primary/60 disabled:cursor-not-allowed disabled:opacity-50",
        )}
      >
        <Icon aria-hidden="true" className="size-4" />
      </button>
    );
  };

  return (
    <div className={cx("flex flex-col items-center gap-2", className)}>
      <span id={`${id}-label`} className="text-sm font-medium">
        {label}
      </span>

      {/* − / + are the single-pointer alternative to dragging (WCAG 2.2 SC
          2.5.7): a tap moves the value exactly one arrow-key step. Keyboard
          users keep the arrows, Page keys and Home/End on the dial itself. */}
      <div className="flex items-center gap-2">
        {stepButton(-1)}

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

        {stepButton(1)}
      </div>

      {/* Hidden from AT: `<output>` is an implicit live region, and the dial's
          aria-valuetext already carries this value, so exposing both spoke
          every change twice. */}
      <output aria-hidden="true" className="font-mono text-sm tabular-nums">
        {display}
      </output>

      {/* Focus stays on a step button after a tap, so the dial's own
          valuetext isn't re-read. This says the new value — only while the
          value is the one a tap produced; arrow keys and drags on the dial
          are announced by the dial, and nobody else's changes by this. */}
      <span className="sr-only-live" aria-live="polite" aria-atomic="true">
        {showStep ? display : ""}
      </span>

      {hint ? (
        <p id={hintId} className="max-w-40 text-center text-xs text-text-muted">
          {hint}
        </p>
      ) : null}
    </div>
  );
}
