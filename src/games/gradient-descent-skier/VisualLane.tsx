"use client";

import { useMemo, useState } from "react";
import { Box, Map as MapIcon } from "lucide-react";
import { Button } from "@/components";
import { cx } from "@/lib/utils";
import { GLOBAL_MINIMUM, STEP_BUDGET, gradientNorm } from "./ml";
import { useGradientSkierStore } from "./store";
import { ContourView } from "./ContourView";
import { LossTerrain, isWebGLAvailable } from "./LossTerrain";

/**
 * Gradient Descent Skier — the no-code lane.
 *
 * Two views of the same surface: the 3-D mountain and a top-down contour map. The
 * contour view is not a downgrade — DESIGN.md §10 requires a non-3D path through
 * any 3D game, and it is also what gets shown when WebGL is unavailable rather
 * than a dead canvas.
 *
 * No interaction happens in either view. The dials and the step buttons live in
 * the control rail as real form widgets, which keeps the whole game playable by
 * keyboard regardless of which view is on screen.
 */

export function VisualLane() {
  const trail = useGradientSkierStore((s) => s.trail);
  const skier = useGradientSkierStore((s) => s.skier);
  const currentLoss = useGradientSkierStore((s) => s.currentLoss);
  const gradient = useGradientSkierStore((s) => s.gradient);
  const diverged = useGradientSkierStore((s) => s.diverged);
  const settled = useGradientSkierStore((s) => s.settled);
  const stepsTaken = useGradientSkierStore((s) => s.stepsTaken);
  const lossHistory = useGradientSkierStore((s) => s.lossHistory);
  const lastStepOvershot = useGradientSkierStore((s) => s.lastStepOvershot);

  /**
   * Probed during render, not in an effect. This component only ever mounts on
   * the client (`dynamic(ssr: false)`), so touching the DOM here is safe — and it
   * means the first paint already knows which view to show, instead of rendering
   * the contour map and then swapping.
   */
  const webgl = useMemo(() => isWebGLAvailable(), []);
  const [view, setView] = useState<"terrain" | "contour">(
    webgl ? "terrain" : "contour",
  );

  /** Loss sparkline — the spec's "loss drops as you descend, spikes red". */
  const sparkline = useMemo(() => {
    if (lossHistory.length < 2) return "";
    const finite = lossHistory.filter((value) => Number.isFinite(value));
    const low = Math.min(...finite);
    const high = Math.max(...finite);
    const range = Math.max(1e-9, high - low);

    return lossHistory
      .map((value, index) => {
        const x = (index / Math.max(1, lossHistory.length - 1)) * 100;
        const safe = Number.isFinite(value) ? value : high;
        const y = 24 - ((safe - low) / range) * 22;
        return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");
  }, [lossHistory]);

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm text-text-muted">
          {diverged
            ? "The skier left the mountain."
            : settled
              ? "Settled — the slope here is flat."
              : `Step ${stepsTaken} of ${STEP_BUDGET}. Use the controls to step.`}
        </p>

        <div
          role="group"
          aria-label="Surface view"
          className="inline-flex gap-1 rounded-full border border-border bg-surface-2 p-1"
        >
          <Button
            variant={view === "terrain" ? "primary" : "ghost"}
            size="sm"
            onClick={() => setView("terrain")}
            disabled={!webgl}
            aria-pressed={view === "terrain"}
            icon={<Box className="size-4" />}
          >
            3D
          </Button>
          <Button
            variant={view === "contour" ? "primary" : "ghost"}
            size="sm"
            onClick={() => setView("contour")}
            aria-pressed={view === "contour"}
            icon={<MapIcon className="size-4" />}
          >
            Contour
          </Button>
        </div>
      </div>

      {!webgl ? (
        <p className="rounded-md border border-warn/40 bg-warn/10 p-2 text-xs text-warn">
          This browser has no WebGL, so the 3-D mountain is unavailable. The
          contour map below shows the same surface, the same path, and the same
          numbers.
        </p>
      ) : null}

      <div className="min-h-0 flex-1">
        {view === "terrain" && webgl ? (
          <LossTerrain trail={trail} position={skier.pos} diverged={diverged} />
        ) : (
          <ContourView
            trail={trail}
            position={skier.pos}
            diverged={diverged}
          />
        )}
      </div>

      {/* Loss trace. Decorative — the numbers below carry the same information. */}
      <div>
        <div className="flex items-baseline justify-between gap-2">
          {/* h2, not h3: this sits directly inside the lane, which is a
              top-level region under the page's h1. Skipping a level trips
              heading-order for screen-reader users navigating by structure. */}
          <h2 className="text-xs font-semibold tracking-wide text-text-muted uppercase">
            Loss so far
          </h2>
          {lastStepOvershot ? (
            <span className="flash-wrong font-mono text-xs text-wrong">
              overshoot
            </span>
          ) : null}
        </div>
        <svg
          viewBox="0 0 100 26"
          preserveAspectRatio="none"
          className="mt-1 h-10 w-full rounded-md bg-bg"
          aria-hidden="true"
        >
          <path
            d={sparkline}
            fill="none"
            stroke={
              diverged
                ? "var(--wrong)"
                : lastStepOvershot
                  ? "var(--warn)"
                  : "var(--primary)"
            }
            strokeWidth={1}
            vectorEffect="non-scaling-stroke"
          />
        </svg>
      </div>

      <p
        className={cx(
          "font-mono text-xs",
          diverged ? "text-wrong" : "text-text-muted",
        )}
      >
        pos ({skier.pos.x.toFixed(2)}, {skier.pos.y.toFixed(2)}) · loss{" "}
        {Number.isFinite(currentLoss) ? currentLoss.toFixed(3) : "diverged"} ·
        slope {gradientNorm(gradient).toFixed(3)} · deepest{" "}
        {GLOBAL_MINIMUM.loss.toFixed(3)} at x {GLOBAL_MINIMUM.x.toFixed(2)}
      </p>
    </div>
  );
}
