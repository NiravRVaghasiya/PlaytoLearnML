"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import { scaleLinear } from "d3";
import { useSortItStore } from "./store";
import { boundaryAt, trueBoundary } from "./ml";

/**
 * Sort-It Arcade — the no-code lane.
 *
 * The player drags knots on a boundary; every drag recomputes the real
 * classification in `ml.ts`. That drag *is* the fitting step, which is the
 * pedagogy contract's second point: the action and the algorithm are the same
 * operation.
 *
 * ── Accessibility, which is the hard part here ───────────────────────────────
 * A drag-only boundary would be unplayable without a mouse and would fail the
 * contract outright. So each knot is a real `role="slider"` with arrow / Page /
 * Home / End keys, its own accessible name, and an `aria-valuetext` in plain
 * language. Pointer drag is an *additional* affordance, not the only one.
 *
 * Class identity never rests on colour alone (DESIGN.md §9): class A is a blue
 * circle, class B an orange triangle, and misclassified points carry a cross.
 * The whole field is summarised in text for anyone who can't see it at all.
 */

/** Field padding inside the viewBox, in viewBox units. */
const PAD = 6;
const VIEW = 100;

export function VisualLane() {
  const trainPoints = useSortItStore((s) => s.trainPoints);
  const params = useSortItStore((s) => s.boundary.params);
  const boundaryType = useSortItStore((s) => s.boundary.type);
  const misclassifiedIds = useSortItStore((s) => s.misclassifiedIds);
  const setKnot = useSortItStore((s) => s.setKnot);
  const nudgeKnot = useSortItStore((s) => s.nudgeKnot);
  const accuracy = useSortItStore((s) => s.accuracy);
  const checked = useSortItStore((s) => s.testAccuracy !== null);

  const [showTruth, setShowTruth] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);
  const draggingRef = useRef<number | null>(null);

  // D3 scales map field coordinates (0–1) into viewBox units. y is inverted so
  // "up" on screen means a larger y value, which is what players expect.
  const { xScale, yScale } = useMemo(
    () => ({
      xScale: scaleLinear().domain([0, 1]).range([PAD, VIEW - PAD]),
      yScale: scaleLinear().domain([0, 1]).range([VIEW - PAD, PAD]),
    }),
    [],
  );

  const misclassified = useMemo(
    () => new Set(misclassifiedIds),
    [misclassifiedIds],
  );

  const knots = useMemo(
    () =>
      params.map((height, index) => ({
        index,
        x: params.length <= 1 ? 0.5 : index / (params.length - 1),
        y: height,
      })),
    [params],
  );

  /** Boundary path, sampled densely enough to read as a smooth polyline. */
  const boundaryPath = useMemo(() => {
    const steps = 200;
    const commands: string[] = [];
    for (let step = 0; step <= steps; step += 1) {
      const x = step / steps;
      const y = boundaryAt(params, x);
      commands.push(
        `${step === 0 ? "M" : "L"}${xScale(x).toFixed(2)},${yScale(y).toFixed(2)}`,
      );
    }
    return commands.join(" ");
  }, [params, xScale, yScale]);

  const truthPath = useMemo(() => {
    const steps = 200;
    const commands: string[] = [];
    for (let step = 0; step <= steps; step += 1) {
      const x = step / steps;
      commands.push(
        `${step === 0 ? "M" : "L"}${xScale(x).toFixed(2)},${yScale(
          trueBoundary(x),
        ).toFixed(2)}`,
      );
    }
    return commands.join(" ");
  }, [xScale, yScale]);

  /** Convert a pointer event to a field-space y value. */
  const pointerToFieldY = useCallback(
    (clientY: number): number | null => {
      const svg = svgRef.current;
      if (!svg) return null;
      const rect = svg.getBoundingClientRect();
      if (rect.height === 0) return null;
      const viewY = ((clientY - rect.top) / rect.height) * VIEW;
      return yScale.invert(viewY);
    },
    [yScale],
  );

  const handlePointerMove = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      const index = draggingRef.current;
      if (index === null) return;
      const y = pointerToFieldY(event.clientY);
      if (y !== null) setKnot(index, y);
    },
    [pointerToFieldY, setKnot],
  );

  const endDrag = useCallback((event: React.PointerEvent<SVGSVGElement>) => {
    draggingRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }, []);

  const onKnotKeyDown = useCallback(
    (event: React.KeyboardEvent, index: number) => {
      const step = event.shiftKey ? 0.01 : 0.04;
      const handlers: Record<string, () => void> = {
        ArrowUp: () => nudgeKnot(index, step),
        ArrowDown: () => nudgeKnot(index, -step),
        PageUp: () => nudgeKnot(index, step * 4),
        PageDown: () => nudgeKnot(index, -step * 4),
        Home: () => setKnot(index, 1),
        End: () => setKnot(index, 0),
      };
      const handler = handlers[event.key];
      if (handler) {
        event.preventDefault();
        handler();
      }
    },
    [nudgeKnot, setKnot],
  );

  const classACount = trainPoints.filter((p) => p.label === 0).length;
  const classBCount = trainPoints.length - classACount;

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm text-text-muted">
          Drag a handle, or focus one and use the arrow keys.
        </p>
        <label className="inline-flex min-h-11 cursor-pointer items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={showTruth}
            onChange={(event) => setShowTruth(event.target.checked)}
            className="size-4 accent-[var(--primary)]"
          />
          Show the real boundary
        </label>
      </div>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${VIEW} ${VIEW}`}
        role="group"
        aria-label="Classification field"
        className="min-h-0 w-full flex-1 touch-none rounded-md bg-bg"
        onPointerMove={handlePointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onPointerLeave={endDrag}
      >
        <title>
          Two-dimensional field of {trainPoints.length} labelled points with an
          adjustable decision boundary
        </title>

        {/* Region tint: which side the boundary predicts. Reinforced by the
            point shapes, never the only cue. */}
        <path
          d={`${boundaryPath} L${xScale(1)},${yScale(1)} L${xScale(0)},${yScale(1)} Z`}
          fill="var(--class-b)"
          opacity={0.07}
        />
        <path
          d={`${boundaryPath} L${xScale(1)},${yScale(0)} L${xScale(0)},${yScale(0)} Z`}
          fill="var(--class-a)"
          opacity={0.07}
        />

        {showTruth ? (
          <path
            d={truthPath}
            fill="none"
            stroke="var(--text-muted)"
            strokeWidth={0.6}
            strokeDasharray="2 2"
          />
        ) : null}

        {/* Points. Shape carries the class; a cross marks a mistake. */}
        {trainPoints.map((point) => {
          const cx = xScale(point.x);
          const cy = yScale(point.y);
          const wrong = misclassified.has(point.id);
          const colour =
            point.label === 0 ? "var(--class-a)" : "var(--class-b)";

          return (
            <g key={point.id}>
              {point.label === 0 ? (
                <circle cx={cx} cy={cy} r={1.5} fill={colour} />
              ) : (
                <polygon
                  points={`${cx},${cy - 1.7} ${cx - 1.6},${cy + 1.3} ${cx + 1.6},${cy + 1.3}`}
                  fill={colour}
                />
              )}
              {wrong ? (
                <g
                  className="flash-wrong"
                  stroke="var(--wrong)"
                  strokeWidth={0.7}
                  strokeLinecap="round"
                >
                  <line x1={cx - 2.4} y1={cy - 2.4} x2={cx + 2.4} y2={cy + 2.4} />
                  <line x1={cx + 2.4} y1={cy - 2.4} x2={cx - 2.4} y2={cy + 2.4} />
                </g>
              ) : null}
            </g>
          );
        })}

        {/* The boundary itself. */}
        <path
          d={boundaryPath}
          fill="none"
          stroke="var(--primary)"
          strokeWidth={1.1}
          strokeLinejoin="round"
        />

        {/* Draggable + keyboard-operable knots. */}
        {knots.map((knot) => (
          <g
            key={knot.index}
            role="slider"
            tabIndex={0}
            aria-label={`Boundary handle ${knot.index + 1} of ${knots.length}`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(knot.y * 100)}
            aria-valuetext={`Handle ${knot.index + 1} at height ${Math.round(
              knot.y * 100,
            )} of 100. Field accuracy ${Math.round(accuracy * 100)} percent.`}
            aria-orientation="vertical"
            onKeyDown={(event) => onKnotKeyDown(event, knot.index)}
            onPointerDown={(event) => {
              draggingRef.current = knot.index;
              svgRef.current?.setPointerCapture(event.pointerId);
            }}
            className="cursor-ns-resize focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--primary)]"
          >
            {/* Oversized transparent hit area: 44px-equivalent target. */}
            <circle
              cx={xScale(knot.x)}
              cy={yScale(knot.y)}
              r={5}
              fill="transparent"
            />
            <circle
              cx={xScale(knot.x)}
              cy={yScale(knot.y)}
              r={2.2}
              fill="var(--primary)"
              stroke="var(--bg)"
              strokeWidth={0.8}
            />
          </g>
        ))}
      </svg>

      {/* Text equivalent of the field, for anyone who can't see it.
          Not a live region: the metric already announces changes. */}
      <p className="text-xs text-text-muted">
        {boundaryType} boundary with {knots.length} handles.{" "}
        {trainPoints.length} training points: {classACount} blue circles (class
        A) and {classBCount} orange triangles (class B).{" "}
        {misclassifiedIds.length} on the wrong side, marked with a cross.
        {checked
          ? " Held-out result is in — see the metric panel."
          : " Press Check generalization when you're happy with the fit."}
      </p>
    </div>
  );
}
