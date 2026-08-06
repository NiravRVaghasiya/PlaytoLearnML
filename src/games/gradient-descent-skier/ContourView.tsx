"use client";

import { useMemo } from "react";
import { scaleLinear } from "d3";
import { cx } from "@/lib/utils";
import {
  DOMAIN,
  GLOBAL_MINIMUM,
  LOCAL_MINIMA,
  START,
  lossAt,
  type Vec2,
} from "./ml";

export interface ContourViewProps {
  trail: readonly Vec2[];
  position: Vec2;
  diverged: boolean;
  className?: string;
}

/**
 * Top-down contour view of the loss surface.
 *
 * This is not a consolation prize for the 3D terrain — it is a first-class view,
 * and it exists for two reasons that both matter:
 *
 *   1. DESIGN.md §10 requires a non-3D path through any 3D game. A rotatable
 *      camera is unusable with a screen reader or switch access, so the whole
 *      surface, the path taken, and both minima are rendered here as flat SVG with
 *      a text description underneath.
 *   2. WebGL is not always available. Rather than showing a broken canvas, the
 *      game falls back to this automatically.
 *
 * Contour plots are also simply how practitioners look at loss surfaces, so this
 * teaches the same thing the mountain does with less spectacle.
 */

const VIEW_W = 200;
const VIEW_H = 150;
const PAD = 6;

/** Loss levels to draw bands for, low to high. */
const LEVELS = [-0.3, -0.1, 0.15, 0.4, 0.7, 1.1, 1.7, 2.5, 3.5, 5];

export function ContourView({
  trail,
  position,
  diverged,
  className,
}: ContourViewProps) {
  const { xScale, yScale, cells, minLoss, maxLoss } = useMemo(() => {
    const x = scaleLinear()
      .domain([DOMAIN.minX, DOMAIN.maxX])
      .range([PAD, VIEW_W - PAD]);
    const y = scaleLinear()
      .domain([DOMAIN.minY, DOMAIN.maxY])
      .range([VIEW_H - PAD, PAD]);

    // A coarse grid of cells shaded by loss. Cheap, and it reads as a heat map
    // without needing marching squares.
    const columns = 48;
    const rows = 36;
    const cellW = (VIEW_W - PAD * 2) / columns;
    const cellH = (VIEW_H - PAD * 2) / rows;

    const built: Array<{ x: number; y: number; loss: number }> = [];
    let low = Infinity;
    let high = -Infinity;

    for (let column = 0; column < columns; column += 1) {
      for (let row = 0; row < rows; row += 1) {
        const dataX =
          DOMAIN.minX +
          ((column + 0.5) / columns) * (DOMAIN.maxX - DOMAIN.minX);
        const dataY =
          DOMAIN.minY + ((row + 0.5) / rows) * (DOMAIN.maxY - DOMAIN.minY);
        const loss = lossAt(dataX, dataY);
        low = Math.min(low, loss);
        high = Math.max(high, loss);
        built.push({
          x: PAD + column * cellW,
          y: PAD + (rows - 1 - row) * cellH,
          loss,
        });
      }
    }

    return {
      xScale: x,
      yScale: y,
      cells: built.map((cell) => ({ ...cell, w: cellW + 0.4, h: cellH + 0.4 })),
      minLoss: low,
      maxLoss: high,
    };
  }, []);

  const shade = (loss: number) => {
    // Deep valleys dark, high ground bright. Lightness only, so the map stays
    // readable in greyscale and for any colour vision.
    const t = (loss - minLoss) / Math.max(1e-9, maxLoss - minLoss);
    const lightness = 8 + Math.pow(t, 0.55) * 46;
    return `hsl(215 30% ${lightness.toFixed(1)}%)`;
  };

  const trailPath = trail
    .map(
      (point, index) =>
        `${index === 0 ? "M" : "L"}${xScale(point.x).toFixed(2)},${yScale(point.y).toFixed(2)}`,
    )
    .join(" ");

  const inView =
    position.x >= DOMAIN.minX &&
    position.x <= DOMAIN.maxX &&
    position.y >= DOMAIN.minY &&
    position.y <= DOMAIN.maxY;

  return (
    <div className={cx("flex flex-col gap-2", className)}>
      <svg
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        role="img"
        aria-label={`Contour map of the loss surface. The deepest valley is at x ${GLOBAL_MINIMUM.x.toFixed(
          2,
        )}, the shallow one at x ${(LOCAL_MINIMA[0]?.x ?? 0).toFixed(
          2,
        )}. You are at x ${position.x.toFixed(2)}, y ${position.y.toFixed(2)}.`}
        className="w-full rounded-md bg-bg"
      >
        {cells.map((cell, index) => (
          <rect
            key={index}
            x={cell.x}
            y={cell.y}
            width={cell.w}
            height={cell.h}
            fill={shade(cell.loss)}
          />
        ))}

        {/* Contour lines at fixed loss levels, so the shape is readable as
            structure and not just as brightness. */}
        {LEVELS.map((level) => (
          <ContourLine key={level} level={level} xScale={xScale} yScale={yScale} />
        ))}

        {/* Both minima, distinguished by shape as well as colour. */}
        <g>
          <circle
            cx={xScale(GLOBAL_MINIMUM.x)}
            cy={yScale(GLOBAL_MINIMUM.y)}
            r={3.4}
            fill="none"
            stroke="var(--correct)"
            strokeWidth={1.4}
          />
          <text
            x={xScale(GLOBAL_MINIMUM.x)}
            y={yScale(GLOBAL_MINIMUM.y) - 6}
            textAnchor="middle"
            fontSize={6}
            fill="var(--correct)"
          >
            deepest
          </text>
        </g>
        {LOCAL_MINIMA.map((minimum) => (
          <g key={minimum.x}>
            <rect
              x={xScale(minimum.x) - 2.8}
              y={yScale(minimum.y) - 2.8}
              width={5.6}
              height={5.6}
              fill="none"
              stroke="var(--warn)"
              strokeWidth={1.2}
            />
            <text
              x={xScale(minimum.x)}
              y={yScale(minimum.y) - 6}
              textAnchor="middle"
              fontSize={6}
              fill="var(--warn)"
            >
              shallow
            </text>
          </g>
        ))}

        {/* Start marker. */}
        <path
          d={`M${xScale(START.x) - 3},${yScale(START.y)} l3,-3 l3,3 l-3,3 z`}
          fill="var(--text-muted)"
        />

        {/* The optimization path (spec: PathTrail). */}
        <path
          d={trailPath}
          fill="none"
          stroke="var(--primary)"
          strokeWidth={1.1}
          strokeLinejoin="round"
          opacity={0.9}
        />

        {/* The skier. */}
        {inView ? (
          <circle
            cx={xScale(position.x)}
            cy={yScale(position.y)}
            r={2.8}
            fill={diverged ? "var(--wrong)" : "var(--xp)"}
            stroke="var(--bg)"
            strokeWidth={0.8}
          />
        ) : null}
      </svg>

      <p className="text-xs text-text-muted">
        Darker is deeper. The circled valley at x{" "}
        {GLOBAL_MINIMUM.x.toFixed(2)} is the deepest point (loss{" "}
        {GLOBAL_MINIMUM.loss.toFixed(2)}); the squared one at x{" "}
        {(LOCAL_MINIMA[0]?.x ?? 0).toFixed(2)} is the shallow trap (loss{" "}
        {(LOCAL_MINIMA[0]?.loss ?? 0).toFixed(2)}).{" "}
        {diverged
          ? "The skier has left the surface."
          : `You are at x ${position.x.toFixed(2)}, y ${position.y.toFixed(2)}.`}
      </p>
    </div>
  );
}

/**
 * One iso-loss line, traced by walking a coarse grid and drawing short segments
 * where the surface crosses the level. Not marching squares — good enough to read
 * the shape, and cheap enough to redraw freely.
 */
function ContourLine({
  level,
  xScale,
  yScale,
}: {
  level: number;
  xScale: (value: number) => number;
  yScale: (value: number) => number;
}) {
  const path = useMemo(() => {
    const columns = 90;
    const rows = 68;
    const segments: string[] = [];

    for (let column = 0; column < columns; column += 1) {
      for (let row = 0; row < rows; row += 1) {
        const x0 =
          DOMAIN.minX + (column / columns) * (DOMAIN.maxX - DOMAIN.minX);
        const x1 =
          DOMAIN.minX + ((column + 1) / columns) * (DOMAIN.maxX - DOMAIN.minX);
        const y0 = DOMAIN.minY + (row / rows) * (DOMAIN.maxY - DOMAIN.minY);
        const y1 =
          DOMAIN.minY + ((row + 1) / rows) * (DOMAIN.maxY - DOMAIN.minY);

        const a = lossAt(x0, y0);
        const b = lossAt(x1, y0);
        const c = lossAt(x0, y1);

        // Horizontal crossing.
        if ((a - level) * (b - level) < 0) {
          const t = (level - a) / (b - a);
          const cx = x0 + t * (x1 - x0);
          segments.push(
            `M${xScale(cx).toFixed(2)},${yScale(y0).toFixed(2)} l0.6,0`,
          );
        }
        // Vertical crossing.
        if ((a - level) * (c - level) < 0) {
          const t = (level - a) / (c - a);
          const cy = y0 + t * (y1 - y0);
          segments.push(
            `M${xScale(x0).toFixed(2)},${yScale(cy).toFixed(2)} l0,0.6`,
          );
        }
      }
    }

    return segments.join(" ");
  }, [level, xScale, yScale]);

  return (
    <path
      d={path}
      stroke="var(--border)"
      strokeWidth={0.35}
      fill="none"
      opacity={0.7}
    />
  );
}
