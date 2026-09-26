"use client";

import { useEffect, useRef, useState } from "react";
import { cx } from "@/lib/utils";
import { useAnimatedNumber } from "./useAnimatedNumber";

export type MetricFormat = "percent" | "decimal" | "integer";

/** Which direction counts as improvement. Accuracy is "up", loss is "down". */
export type MetricDirection = "up" | "down";

export type MetricState = "neutral" | "good" | "bad" | "warn";

export interface MetricSpec {
  /** e.g. "Accuracy", "Loss", "Inertia". */
  label: string;
  /** The real, computed value. Never a hardcoded number (CLAUDE.md guardrail). */
  value: number;
  format?: MetricFormat;
  /** Decimal places. Defaults: percent 0, decimal 3, integer 0. */
  precision?: number;
  goodDirection?: MetricDirection;
  /**
   * Force the colour state. Games set `"bad"` when a named failure mode fires
   * so the metric goes red and stays red until retry (DESIGN.md §8).
   */
  state?: MetricState;
  /** Small caption under the number, e.g. "on held-out set". */
  caption?: string;
}

export interface MetricReadoutProps extends MetricSpec {
  size?: "md" | "lg";
  /** Set false only for secondary metrics; the primary metric must announce. */
  announce?: boolean;
  className?: string;
}

const STATE_COLOR: Record<MetricState, string> = {
  neutral: "text-text",
  good: "text-correct",
  bad: "text-wrong",
  warn: "text-warn",
};

export function formatMetric(
  value: number,
  format: MetricFormat = "decimal",
  precision?: number,
): string {
  if (!Number.isFinite(value)) return "—";
  switch (format) {
    case "percent":
      return `${(value * 100).toFixed(precision ?? 0)}%`;
    case "integer":
      return Math.round(value).toLocaleString("en-US");
    case "decimal":
    default:
      return value.toFixed(precision ?? 3);
  }
}

/** Screen-reader sentence, e.g. "Accuracy increased to 84 percent". */
export function metricAnnouncement(
  label: string,
  value: number,
  previous: number | null,
  format: MetricFormat = "decimal",
  precision?: number,
): string {
  // Games use a non-finite value to mean "nothing measured yet" — Overfit Tower
  // Defense starts with gap = NaN and captions it "deploy to measure" — and
  // `formatMetric` renders that as an em dash. The percent branch below formats
  // the number itself rather than going through `formatMetric`, so without this
  // guard the screen shows "—" while a screen reader hears "NaN percent".
  if (!Number.isFinite(value)) return `${label} not measured yet`;

  const spoken =
    format === "percent"
      ? `${(value * 100).toFixed(precision ?? 0)} percent`
      : formatMetric(value, format, precision);

  if (previous === null || !Number.isFinite(previous) || previous === value) {
    return `${label} ${spoken}`;
  }
  return `${label} ${value > previous ? "increased" : "decreased"} to ${spoken}`;
}

/**
 * The always-visible live metric (DESIGN.md §6, §8).
 *
 * This is the single most important component in the product: it is the thing
 * that moves when the player acts, which is what turns a mechanic into a
 * lesson. Rules baked in here:
 *
 * - Mono font so a changing value never shifts layout (DESIGN.md §3).
 * - Value tweens rather than snapping; colour pulses green/red on change
 *   according to `goodDirection`, and a `state` override pins it red on failure.
 * - Direction is conveyed by an arrow glyph AND the spoken sentence, never by
 *   colour alone (DESIGN.md §9).
 * - An `aria-live="polite"` region announces the *settled* value. It is
 *   debounced so a 50-epoch training run doesn't fire 50 announcements.
 */
export function MetricReadout({
  label,
  value,
  format = "decimal",
  precision,
  goodDirection = "up",
  state,
  caption,
  size = "lg",
  announce = true,
  className,
}: MetricReadoutProps) {
  const displayed = useAnimatedNumber(value);

  const previousRef = useRef<number | null>(null);
  const [trend, setTrend] = useState<"up" | "down" | "flat">("flat");
  const [pulse, setPulse] = useState<"good" | "bad" | null>(null);
  const [announcement, setAnnouncement] = useState("");

  /**
   * One effect owns everything derived from a value change.
   *
   * `trend` and `pulse` are state rather than values derived from `previousRef`
   * during render: the ref is mutated here, so a render that reads it would show
   * the right glyph once and then flicker back to neutral on the very next
   * render. State keeps the direction glyph stable until the value moves again.
   *
   * The announcement is debounced and built from the `previous` captured in this
   * closure. Reading the ref inside the timeout would compare the value against
   * itself and lose the "increased"/"decreased" wording entirely.
   */
  useEffect(() => {
    const previous = previousRef.current;
    previousRef.current = value;

    // A change needs a real number on BOTH sides. Games use NaN for "not
    // measured yet", and every comparison with NaN is false, so the first real
    // measurement used to read as "down" — a red ▼ and a bad pulse on Data
    // Detox's very first accuracy, whichever way was good.
    const comparable =
      previous !== null && Number.isFinite(previous) && Number.isFinite(value);
    const changed = comparable && previous !== value;

    if (changed) {
      setTrend(value > previous ? "up" : "down");
      const improving =
        goodDirection === "up" ? value > previous : value < previous;
      setPulse(improving ? "good" : "bad");
    } else if (!comparable) {
      // Into or out of "not measured": there is no direction to show, and a
      // stale arrow beside "—" would claim one.
      setTrend("flat");
      setPulse(null);
    }

    const announceTimer = announce
      ? setTimeout(() => {
          setAnnouncement(
            metricAnnouncement(label, value, previous, format, precision),
          );
        }, 600)
      : null;

    const pulseTimer = changed ? setTimeout(() => setPulse(null), 900) : null;

    return () => {
      if (announceTimer) clearTimeout(announceTimer);
      if (pulseTimer) clearTimeout(pulseTimer);
    };
  }, [value, goodDirection, announce, label, format, precision]);

  const effectiveState: MetricState = state ?? pulse ?? "neutral";
  const arrow = trend === "up" ? "▲" : trend === "down" ? "▼" : "–";

  return (
    <div
      className={cx(
        "rounded-md border border-border bg-surface-2 p-4",
        className,
      )}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-sm font-medium text-text-muted">{label}</span>
        {/* Redundant, non-colour cue for the direction of travel. */}
        <span
          aria-hidden="true"
          data-testid="metric-trend"
          className={cx(
            "font-mono text-sm transition-colors dur-standard",
            STATE_COLOR[effectiveState],
          )}
        >
          {arrow}
        </span>
      </div>

      {/* `<output>` carries an implicit role="status", which makes it a live
          region. Since the number tweens, leaving it exposed would fire an
          announcement on every animation frame — a 50-epoch training run would
          bury a screen-reader user. So the visible number is hidden from AT and
          the debounced sentence below is the single announcement channel. */}
      <output
        aria-hidden="true"
        data-testid="metric-value"
        className={cx(
          "mt-1 block font-mono tabular-nums transition-colors dur-standard",
          size === "lg" ? "text-2xl" : "text-xl",
          STATE_COLOR[effectiveState],
          // The sparkle is part of the pulse, so an explicit state overrides it
          // the same way it overrides the pulse colour. A count the game marks
          // `neutral` (K-Means' number of flags) is not "better" for rising, and
          // a green sparkle on a red, failing metric would contradict it.
          pulse === "good" && (state === undefined || state === "good")
            ? "metric-sparkle"
            : null,
        )}
      >
        {formatMetric(displayed, format, precision)}
      </output>

      {caption ? (
        <p className="mt-1 text-xs text-text-muted">{caption}</p>
      ) : null}

      {/* DESIGN.md §9 — screen readers hear the metric move. */}
      {announce ? (
        <p className="sr-only-live" aria-live="polite" aria-atomic="true">
          {announcement}
        </p>
      ) : null}
    </div>
  );
}
