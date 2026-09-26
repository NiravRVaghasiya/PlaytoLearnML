"use client";

import { useMemo } from "react";
import { SURFACE_RESOLUTION, type Point2D } from "./ml";

export interface DecisionSurfaceProps {
  /** P(class 1) per grid cell, row-major, or null before training. */
  surface: Float32Array | null;
  points: Point2D[];
  /** Held-out accuracy, for the caption. */
  accuracy: number | null;
}

const VIEW = 100;
const CELL = VIEW / SURFACE_RESOLUTION;

/**
 * Okabe-Ito blue ↔ orange, through a neutral mid.
 *
 * Diverging on purpose: the interesting structure is the *boundary*, which is
 * where the network is unsure, and a diverging ramp puts its most neutral values
 * exactly there. Hue alone is never the only cue — the two classes also differ in
 * marker shape below, so the plot survives being read in greyscale.
 *
 * The neutral is the page background (`--bg`), not a raised surface. Class-A
 * blue is 3.65:1 against `--bg` but only 2.88:1 against `--surface-2`, so points
 * sitting on the boundary — the ones the player most needs to see — would drop
 * below the 3:1 non-text contrast floor on the lighter neutral.
 */
function surfaceColor(probability: number): string {
  const classA = [0x00, 0x72, 0xb2];
  const classB = [0xe6, 0x9f, 0x00];
  const mid = [0x0e, 0x11, 0x16];

  const t = Math.min(Math.max(probability, 0), 1);
  const [from, to, local] =
    t < 0.5 ? [classA, mid, t * 2] : [mid, classB, (t - 0.5) * 2];

  const channel = (index: number) =>
    Math.round(from[index]! + (to[index]! - from[index]!) * local);

  return `rgb(${channel(0)} ${channel(1)} ${channel(2)})`;
}

/** Data space [-1,1] → SVG space [0,100], with y flipped so +y points up. */
const toSvgX = (x: number) => ((x + 1) / 2) * VIEW;
const toSvgY = (y: number) => ((1 - y) / 2) * VIEW;

/**
 * The live decision-surface heatmap (spec: `<DecisionSurface>`).
 *
 * This is the game's always-visible feedback in its most literal form — it is a
 * picture of what the architecture can represent. A straight colour split means
 * the model is linear no matter how many neurons are in it, which is the whole
 * "No non-linearity" failure visible at a glance and before any number is read.
 */
export function DecisionSurface({
  surface,
  points,
  accuracy,
}: DecisionSurfaceProps) {
  const cells = useMemo(() => {
    if (surface === null) return null;
    const out: Array<{ key: string; x: number; y: number; fill: string }> = [];
    for (let row = 0; row < SURFACE_RESOLUTION; row += 1) {
      for (let column = 0; column < SURFACE_RESOLUTION; column += 1) {
        const probability = surface[row * SURFACE_RESOLUTION + column];
        if (probability === undefined) continue;
        out.push({
          key: `${row}-${column}`,
          x: column * CELL,
          // Row 0 of the grid is y = -1, which is the BOTTOM of the plot.
          y: VIEW - (row + 1) * CELL,
          fill: surfaceColor(probability),
        });
      }
    }
    return out;
  }, [surface]);

  const label =
    surface === null
      ? `Scatter plot of ${points.length} training points in two classes. Train the network to see the decision surface it learns.`
      : `Decision surface with ${points.length} training points overlaid. Blue regions are predicted class A, orange regions class B.${
          accuracy === null
            ? ""
            : ` Held-out accuracy ${Math.round(accuracy * 100)} percent.`
        }`;

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${VIEW} ${VIEW}`}
        // --bg rather than --surface-2, for the reason in surfaceColor. Before
        // training the points sit on this directly, and their outline is --bg
        // too, so the fill alone has to clear 3:1 against it.
        className="w-full rounded-md border border-border bg-bg"
        role="img"
        aria-label={label}
      >
        {cells === null ? (
          <rect width={VIEW} height={VIEW} fill="var(--bg)" />
        ) : (
          // shapeRendering avoids hairline seams between adjacent cells.
          <g shapeRendering="crispEdges">
            {cells.map((cell) => (
              <rect
                key={cell.key}
                x={cell.x}
                y={cell.y}
                width={CELL + 0.05}
                height={CELL + 0.05}
                fill={cell.fill}
              />
            ))}
          </g>
        )}

        {/* Axes through the origin, so XOR's quadrants are readable. */}
        <line
          x1={0}
          y1={VIEW / 2}
          x2={VIEW}
          y2={VIEW / 2}
          stroke="var(--border)"
          strokeWidth={0.3}
        />
        <line
          x1={VIEW / 2}
          y1={0}
          x2={VIEW / 2}
          y2={VIEW}
          stroke="var(--border)"
          strokeWidth={0.3}
        />

        {points.map((point, index) =>
          point.label === 1 ? (
            <rect
              key={index}
              x={toSvgX(point.x) - 1.15}
              y={toSvgY(point.y) - 1.15}
              width={2.3}
              height={2.3}
              fill="var(--class-b)"
              stroke="var(--bg)"
              strokeWidth={0.35}
            />
          ) : (
            <circle
              key={index}
              cx={toSvgX(point.x)}
              cy={toSvgY(point.y)}
              r={1.3}
              fill="var(--class-a)"
              stroke="var(--bg)"
              strokeWidth={0.35}
            />
          ),
        )}
      </svg>

      <figcaption className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-text-muted">
        <span className="inline-flex items-center gap-1.5">
          <span
            aria-hidden="true"
            className="inline-block size-2.5 rounded-full"
            style={{ background: "var(--class-a)" }}
          />
          Class A (circles)
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span
            aria-hidden="true"
            className="inline-block size-2.5"
            style={{ background: "var(--class-b)" }}
          />
          Class B (squares)
        </span>
        <span>
          {surface === null
            ? "Background shows the learned surface once trained."
            : "Background is the network's prediction at every point."}
        </span>
      </figcaption>
    </figure>
  );
}
