"use client";

import { useEffect, useRef, useState } from "react";
import { useReducedMotion } from "@/lib/useReducedMotion";

export interface UseAnimatedNumberOptions {
  /** Tween duration in ms. DESIGN.md §7 standard = 240ms. */
  durationMs?: number;
  /**
   * When true, skip the tween and snap. Callers normally leave this alone —
   * `prefers-reduced-motion` is honoured automatically.
   */
  disabled?: boolean;
}

/**
 * Tween a number toward `target` so live metrics count up/down instead of
 * snapping (DESIGN.md §7: "the metric readout: value counts up/down (never
 * snaps)").
 *
 * Under `prefers-reduced-motion` — or when there's no animation clock, as during
 * SSR — the value is returned unchanged. The number still updates, so the
 * functional feedback survives; only the decoration goes.
 *
 * Implementation note: state is only ever written from inside the
 * requestAnimationFrame callback, never synchronously in the effect body. When
 * snapping we return `target` directly rather than pushing it through state,
 * which avoids a cascading render on every value change.
 */
export function useAnimatedNumber(
  target: number,
  { durationMs = 240, disabled = false }: UseAnimatedNumberOptions = {},
): number {
  const reducedMotion = useReducedMotion();

  const snap =
    disabled ||
    reducedMotion ||
    durationMs <= 0 ||
    !Number.isFinite(target) ||
    typeof requestAnimationFrame !== "function";

  const [tweened, setTweened] = useState(target);
  const fromRef = useRef(target);
  const frameRef = useRef<number | null>(null);

  useEffect(() => {
    if (snap) {
      // Keep the origin current so a later tween starts from the right place.
      fromRef.current = target;
      return;
    }

    const from = fromRef.current;
    const start = performance.now();

    // Always schedule at least one frame, even when `from === target`: that
    // frame is what reconciles `tweened` without a synchronous setState.
    const tick = (now: number) => {
      const t = durationMs > 0 ? Math.min(1, (now - start) / durationMs) : 1;
      const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
      const value = from + (target - from) * eased;

      fromRef.current = value;
      setTweened(value);

      if (t < 1) {
        frameRef.current = requestAnimationFrame(tick);
      } else {
        fromRef.current = target;
        frameRef.current = null;
      }
    };

    frameRef.current = requestAnimationFrame(tick);

    return () => {
      if (frameRef.current !== null) {
        cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }
    };
  }, [target, durationMs, snap]);

  return snap ? target : tweened;
}
