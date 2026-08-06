"use client";

import { useCallback, useMemo, useRef } from "react";
import { scaleLinear } from "d3";
import { useKMeansStore } from "./store";

/**
 * K-Means Territory Wars — the no-code lane (spec: `<ClusterMap>`,
 * `<CentroidFlag>`).
 *
 * Villages recolour by assignment; flags are dragged and stepped. Dropping and
 * moving a flag *is* choosing k and the initialization, and pressing
 * Assign/Update *is* running the algorithm — the pedagogy contract's second
 * point, with no simulation layer in between.
 *
 * ── Accessibility ───────────────────────────────────────────────────────────
 * A flag is a 2-D position, and ARIA has no 2-D equivalent of `role="slider"`.
 * So there are two complete paths to move one:
 *
 *   - here on the map: each flag is focusable, arrow keys move it, and its
 *     accessible name reports its coordinates and territory size;
 *   - in the controls rail: a labelled X and Y `<Slider>` pair for the selected
 *     flag, which gives proper `aria-valuetext` announcements on every change.
 *
 * Pointer drag is a third affordance, never the only one. Cluster identity is
 * carried by shape as well as colour (DESIGN.md §9), and the whole board is
 * summarised in text.
 */

const VIEW = 100;
const PAD = 5;

/** Cluster colours, in the DESIGN.md §2 semantic order. */
const CLUSTER_COLORS = [
  "var(--class-a)",
  "var(--class-b)",
  "var(--class-c)",
  "var(--class-d)",
  "var(--primary)",
  "var(--warn)",
  "var(--correct)",
  "var(--star)",
] as const;

/**
 * A distinct glyph per cluster so membership survives greyscale and
 * colourblindness. Eight clusters, eight shapes.
 */
function villageGlyph(cluster: number | null, cx: number, cy: number) {
  const r = 1.35;
  switch (cluster === null ? -1 : cluster % 8) {
    case 0: // circle
      return <circle cx={cx} cy={cy} r={r} />;
    case 1: // triangle up
      return (
        <polygon
          points={`${cx},${cy - r * 1.2} ${cx - r * 1.1},${cy + r} ${cx + r * 1.1},${cy + r}`}
        />
      );
    case 2: // square
      return <rect x={cx - r} y={cy - r} width={r * 2} height={r * 2} />;
    case 3: // diamond
      return (
        <polygon
          points={`${cx},${cy - r * 1.3} ${cx + r * 1.3},${cy} ${cx},${cy + r * 1.3} ${cx - r * 1.3},${cy}`}
        />
      );
    case 4: // triangle down
      return (
        <polygon
          points={`${cx},${cy + r * 1.2} ${cx - r * 1.1},${cy - r} ${cx + r * 1.1},${cy - r}`}
        />
      );
    case 5: // plus
      return (
        <path
          d={`M${cx - r * 1.3},${cy} H${cx + r * 1.3} M${cx},${cy - r * 1.3} V${cy + r * 1.3}`}
          strokeWidth={0.9}
          strokeLinecap="round"
        />
      );
    case 6: // cross
      return (
        <path
          d={`M${cx - r},${cy - r} L${cx + r},${cy + r} M${cx + r},${cy - r} L${cx - r},${cy + r}`}
          strokeWidth={0.9}
          strokeLinecap="round"
        />
      );
    case 7: // hexagon-ish
      return (
        <polygon
          points={`${cx - r},${cy - r * 0.6} ${cx},${cy - r * 1.3} ${cx + r},${cy - r * 0.6} ${cx + r},${cy + r * 0.6} ${cx},${cy + r * 1.3} ${cx - r},${cy + r * 0.6}`}
        />
      );
    default: // unassigned
      return <circle cx={cx} cy={cy} r={r * 0.8} />;
  }
}

export function VisualLane() {
  const points = useKMeansStore((s) => s.points);
  const centroids = useKMeansStore((s) => s.centroids);
  const sizes = useKMeansStore((s) => s.sizes);
  const selectedFlag = useKMeansStore((s) => s.selectedFlag);
  const assignmentStale = useKMeansStore((s) => s.assignmentStale);
  const converged = useKMeansStore((s) => s.converged);
  const moveFlag = useKMeansStore((s) => s.moveFlag);
  const nudgeFlag = useKMeansStore((s) => s.nudgeFlag);
  const selectFlag = useKMeansStore((s) => s.selectFlag);
  const removeFlag = useKMeansStore((s) => s.removeFlag);

  const svgRef = useRef<SVGSVGElement>(null);
  const draggingRef = useRef<number | null>(null);

  const { xScale, yScale } = useMemo(
    () => ({
      xScale: scaleLinear().domain([0, 1]).range([PAD, VIEW - PAD]),
      yScale: scaleLinear().domain([0, 1]).range([VIEW - PAD, PAD]),
    }),
    [],
  );

  const pointerToField = useCallback(
    (clientX: number, clientY: number) => {
      const svg = svgRef.current;
      if (!svg) return null;
      const rect = svg.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return null;
      return {
        x: xScale.invert(((clientX - rect.left) / rect.width) * VIEW),
        y: yScale.invert(((clientY - rect.top) / rect.height) * VIEW),
      };
    },
    [xScale, yScale],
  );

  const onPointerMove = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      const index = draggingRef.current;
      if (index === null) return;
      const field = pointerToField(event.clientX, event.clientY);
      if (field) moveFlag(index, field.x, field.y);
    },
    [pointerToField, moveFlag],
  );

  const endDrag = useCallback((event: React.PointerEvent<SVGSVGElement>) => {
    draggingRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }, []);

  const onFlagKeyDown = useCallback(
    (event: React.KeyboardEvent, index: number) => {
      const step = event.shiftKey ? 0.01 : 0.03;
      const moves: Record<string, [number, number]> = {
        ArrowUp: [0, step],
        ArrowDown: [0, -step],
        ArrowLeft: [-step, 0],
        ArrowRight: [step, 0],
      };

      const move = moves[event.key];
      if (move) {
        event.preventDefault();
        nudgeFlag(index, move[0], move[1]);
        return;
      }
      if (event.key === "Delete" || event.key === "Backspace") {
        event.preventDefault();
        removeFlag(index);
      }
    },
    [nudgeFlag, removeFlag],
  );

  const assignedCount = points.filter((p) => p.clusterId !== null).length;

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <p className="text-sm text-text-muted">
        Drag a flag, or focus one and use the arrow keys. Delete removes it.
        {assignmentStale
          ? " Colours are out of date — press Assign."
          : converged
            ? " Converged: no flag has anywhere better to go."
            : ""}
      </p>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${VIEW} ${VIEW}`}
        role="group"
        aria-label="Village map"
        className="min-h-0 w-full flex-1 touch-none rounded-md bg-bg"
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onPointerLeave={endDrag}
      >
        <title>
          Map of {points.length} villages clustered around {centroids.length}{" "}
          flags
        </title>

        {/* Villages, coloured and shaped by which flag they joined. */}
        {points.map((point) => {
          const colour =
            point.clusterId === null
              ? "var(--text-muted)"
              : CLUSTER_COLORS[point.clusterId % CLUSTER_COLORS.length]!;
          return (
            <g
              key={point.id}
              fill={colour}
              stroke={colour}
              opacity={assignmentStale ? 0.55 : 0.9}
            >
              {villageGlyph(point.clusterId, xScale(point.x), yScale(point.y))}
            </g>
          );
        })}

        {/* Spokes from each village to its flag: makes "within-cluster
            distance" — which is literally what inertia sums — visible. */}
        {points.map((point) => {
          if (point.clusterId === null) return null;
          const centroid = centroids[point.clusterId];
          if (!centroid) return null;
          return (
            <line
              key={`spoke-${point.id}`}
              x1={xScale(point.x)}
              y1={yScale(point.y)}
              x2={xScale(centroid.x)}
              y2={yScale(centroid.y)}
              stroke={CLUSTER_COLORS[point.clusterId % CLUSTER_COLORS.length]!}
              strokeWidth={0.18}
              opacity={0.35}
            />
          );
        })}

        {/* Flags. */}
        {centroids.map((centroid, index) => {
          const colour = CLUSTER_COLORS[index % CLUSTER_COLORS.length]!;
          const size = sizes[index] ?? 0;
          const cx = xScale(centroid.x);
          const cy = yScale(centroid.y);
          const isSelected = selectedFlag === index;

          return (
            <g
              key={centroid.id}
              role="button"
              tabIndex={0}
              aria-label={`Flag ${index + 1} of ${centroids.length}, at x ${Math.round(
                centroid.x * 100,
              )} y ${Math.round(centroid.y * 100)}, ${size} village${size === 1 ? "" : "s"}${
                size === 0 ? " — empty, it cannot move" : ""
              }. Arrow keys move it, Delete removes it.`}
              aria-pressed={isSelected}
              onFocus={() => selectFlag(index)}
              onKeyDown={(event) => onFlagKeyDown(event, index)}
              onPointerDown={(event) => {
                draggingRef.current = index;
                selectFlag(index);
                svgRef.current?.setPointerCapture(event.pointerId);
              }}
              className="cursor-grab focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--primary)]"
            >
              {/* Generous invisible hit area. */}
              <circle cx={cx} cy={cy} r={6} fill="transparent" />
              {/* An empty flag is drawn hollow with a warning ring, so the
                  failure is visible before the verdict explains it. */}
              <circle
                cx={cx}
                cy={cy}
                r={3.2}
                fill={size === 0 ? "none" : colour}
                stroke={size === 0 ? "var(--wrong)" : "var(--bg)"}
                strokeWidth={size === 0 ? 1 : 0.9}
                strokeDasharray={size === 0 ? "1.5 1.5" : undefined}
              />
              {isSelected ? (
                <circle
                  cx={cx}
                  cy={cy}
                  r={5.2}
                  fill="none"
                  stroke={colour}
                  strokeWidth={0.7}
                />
              ) : null}
              <text
                x={cx}
                y={cy + 1.4}
                textAnchor="middle"
                fontSize={4}
                fill={size === 0 ? "var(--wrong)" : "var(--bg)"}
                style={{ pointerEvents: "none" }}
              >
                {index + 1}
              </text>
            </g>
          );
        })}
      </svg>

      <p className="text-xs text-text-muted">
        {points.length} villages, {centroids.length} flags. Territory sizes:{" "}
        {sizes.map((size, index) => `flag ${index + 1}: ${size}`).join(", ")}.
        {assignedCount < points.length
          ? ` ${points.length - assignedCount} villages not yet assigned.`
          : ""}
      </p>
    </div>
  );
}
