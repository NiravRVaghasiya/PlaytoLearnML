"use client";

import { useMemo } from "react";

export interface LossCurveProps {
  history: number[];
  totalEpochs: number;
  training: boolean;
}

const WIDTH = 320;
const HEIGHT = 96;
const PAD = 4;

/**
 * Training loss per epoch (spec: `<LossCurve>`, live "while training animates").
 *
 * The shape carries a diagnosis the single loss number cannot: a curve that drops
 * then flattens high has converged to the best this architecture can do, which
 * means the fix is the architecture and not more training. The caption says so
 * once the curve has actually flattened, because that is the moment the player is
 * deciding whether to wait or to go change something.
 */
export function LossCurve({ history, totalEpochs, training }: LossCurveProps) {
  const view = useMemo(() => {
    const finite = history.filter((value) => Number.isFinite(value));
    if (finite.length === 0) return null;

    const max = Math.max(...finite);
    const min = Math.min(...finite);
    // Guard a flat curve: a zero-height range would divide by zero.
    const span = Math.max(max - min, 1e-6);
    const innerW = WIDTH - PAD * 2;
    const innerH = HEIGHT - PAD * 2;

    const points = finite.map((value, index) => {
      const x =
        PAD + (index / Math.max(1, totalEpochs - 1)) * innerW;
      const y = PAD + ((max - value) / span) * innerH;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });

    // "Flattened" = the last fifth of the curve moved less than 2% of the total
    // drop so far. Only meaningful once there is a curve to judge.
    const tail = finite.slice(-Math.max(4, Math.floor(finite.length / 5)));
    const tailSpan = Math.max(...tail) - Math.min(...tail);
    const flattened =
      finite.length >= 20 && tailSpan < Math.max(span * 0.02, 1e-4);

    return {
      path: points.join(" "),
      first: finite[0]!,
      last: finite[finite.length - 1]!,
      min,
      max,
      flattened,
    };
  }, [history, totalEpochs]);

  return (
    <figure className="m-0">
      <figcaption className="mb-1.5 flex items-baseline justify-between gap-2">
        <span className="text-sm font-medium">Training loss</span>
        <span className="font-mono text-xs text-text-muted">
          {view === null
            ? "not started"
            : `${view.last.toFixed(4)} at epoch ${history.length}`}
        </span>
      </figcaption>

      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="w-full rounded-md border border-border bg-surface-2"
        role="img"
        aria-label={
          view === null
            ? "Loss curve, empty. Training has not run yet."
            : `Loss curve over ${history.length} of ${totalEpochs} epochs. Started at ${view.first.toFixed(
                3,
              )}, now ${view.last.toFixed(3)}.${
                view.flattened
                  ? " The curve has flattened, so this architecture has converged."
                  : ""
              }`
        }
      >
        {view === null ? (
          <text
            x={WIDTH / 2}
            y={HEIGHT / 2}
            textAnchor="middle"
            dominantBaseline="middle"
            className="fill-[var(--text-muted)] text-[11px]"
          >
            Train the network to draw the curve
          </text>
        ) : (
          <polyline
            points={view.path}
            fill="none"
            stroke="var(--primary)"
            strokeWidth={2}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        )}
      </svg>

      <p className="mt-1.5 text-xs text-text-muted">
        {view === null
          ? "Binary cross-entropy on the training points, one value per epoch."
          : view.flattened && !training
            ? "Flat and level: more epochs will not move this curve — only a different architecture will."
            : training
              ? "Falling means the weights are still improving."
              : `Fell from ${view.first.toFixed(3)} to ${view.last.toFixed(3)} over ${history.length} epochs.`}
      </p>
    </figure>
  );
}
