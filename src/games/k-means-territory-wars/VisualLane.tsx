"use client";

import { useCallback, useMemo, useRef } from "react";
import { scaleLinear } from "d3";
import { useKMeansStore } from "./store";
import {
  GRAB_RADIUS_PX,
  MIN_GRAB_RADIUS_UNITS,
  clientToUser,
  flagUnderPointer,
} from "./field";

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
  /** Flag elements by index, so a pointer press can focus the one it grabs. */
  const flagRefs = useRef<Array<SVGGElement | null>>([]);

  const { xScale, yScale } = useMemo(
    () => ({
      xScale: scaleLinear().domain([0, 1]).range([PAD, VIEW - PAD]),
      yScale: scaleLinear().domain([0, 1]).range([VIEW - PAD, PAD]),
    }),
    [],
  );

  /**
   * A pointer position on the 0–1 map, through the SVG's own transform so the
   * letterboxing on a tall desktop column can't displace it (see `field.ts`).
   */
  const pointerToField = useCallback(
    (clientX: number, clientY: number) => {
      const view = clientToUser(
        clientX,
        clientY,
        svgRef.current?.getScreenCTM() ?? null,
      );
      if (!view) return null;
      return { x: xScale.invert(view.x), y: yScale.invert(view.y) };
    },
    [xScale, yScale],
  );

  /**
   * One press handler for the whole map: it grabs the flag NEAREST the press
   * within a 44 px target, instead of whichever hit circle was painted last.
   * A press on empty ground does nothing — the rail's sliders are the
   * single-pointer way to place a flag.
   */
  const onPointerDown = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      if (!event.isPrimary || event.button !== 0) return;
      const ctm = event.currentTarget.getScreenCTM();
      const view = clientToUser(event.clientX, event.clientY, ctm);
      if (!view) return;

      // CSS px per viewBox unit, so the radius is a real 22 px on any screen.
      const scale = ctm?.a || 1;
      const radius = Math.max(MIN_GRAB_RADIUS_UNITS, GRAB_RADIUS_PX / scale);
      const index = flagUnderPointer(
        view,
        centroids.map((centroid) => ({
          x: xScale(centroid.x),
          y: yScale(centroid.y),
        })),
        radius,
      );
      if (index === null) return;

      draggingRef.current = index;
      selectFlag(index);
      event.currentTarget.setPointerCapture(event.pointerId);
      flagRefs.current[index]?.focus({ preventScroll: true });
    },
    [centroids, selectFlag, xScale, yScale],
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
      <div className="text-sm text-text-muted">
        <p>
          Drag a flag, or focus one and use the arrow keys. Delete removes it.
        </p>
        {/* Status on its own line, with that line reserved even when empty.
            It flips the moment a drag starts ("out of date"), and when it used
            to append to the sentence above, the reflow shifted the map about
            one text line under the player's finger mid-drag. Two lines are
            reserved on narrow screens, where either message can wrap. */}
        <p className="min-h-10 sm:min-h-5">
          {assignmentStale
            ? "Colours are out of date — press Assign."
            : converged
              ? "Converged: no flag has anywhere better to go."
              : ""}
        </p>
      </div>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${VIEW} ${VIEW}`}
        role="group"
        aria-label="Village map"
        className="min-h-0 w-full flex-1 touch-none rounded-md bg-bg"
        onPointerDown={onPointerDown}
        // The press handler focuses the flag it grabbed. Without this, the
        // browser's own mousedown focusing would then hand focus to whatever
        // was under the pointer — a neighbouring flag, or the canvas itself.
        onMouseDown={(event) => event.preventDefault()}
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
          // Sizes are from the last assign. While that's out of date a new
          // flag always reads 0 — it hasn't been offered any villages yet —
          // so "empty" is only claimed on a fresh assignment.
          const empty = size === 0 && !assignmentStale;
          const cx = xScale(centroid.x);
          const cy = yScale(centroid.y);
          const isSelected = selectedFlag === index;

          return (
            <g
              key={centroid.id}
              ref={(element) => {
                flagRefs.current[index] = element;
              }}
              role="button"
              tabIndex={0}
              aria-label={`Flag ${index + 1} of ${centroids.length}, at x ${Math.round(
                centroid.x * 100,
              )} y ${Math.round(centroid.y * 100)}, ${
                assignmentStale
                  ? `${size} village${size === 1 ? "" : "s"} as of the last assign`
                  : `${size} village${size === 1 ? "" : "s"}`
              }${
                empty ? " — empty, it cannot move" : ""
              }. Arrow keys move it, Delete removes it.`}
              aria-pressed={isSelected}
              onFocus={() => selectFlag(index)}
              onKeyDown={(event) => onFlagKeyDown(event, index)}
              className="cursor-grab focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--primary)]"
            >
              {/* Hover cursor and focus-ring size. Grabbing is decided at the
                  SVG level (onPointerDown) by distance, not by this circle. */}
              <circle
                cx={cx}
                cy={cy}
                r={MIN_GRAB_RADIUS_UNITS}
                fill="transparent"
              />
              {/* An empty flag is drawn hollow with a warning ring, so the
                  failure is visible before the verdict explains it. */}
              <circle
                cx={cx}
                cy={cy}
                r={3.2}
                fill={empty ? "none" : colour}
                stroke={empty ? "var(--wrong)" : "var(--bg)"}
                strokeWidth={empty ? 1 : 0.9}
                strokeDasharray={empty ? "1.5 1.5" : undefined}
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
                fill={empty ? "var(--wrong)" : "var(--bg)"}
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
