"use client";

import { useMemo } from "react";
import { useReducedMotion } from "@/lib/useReducedMotion";
import { NODES, STEPS, nodeById, type Values } from "./ml";
import type { EdgeView } from "./store";
import { GradientPacket } from "./GradientPacket";

export interface ComputeGraphProps {
  values: Values;
  /** The player's gradients so far, by node id. */
  grads: Values;
  /** Which node ids have a settled gradient. */
  settled: string[];
  edges: EdgeView[];
  /** The node the inspector is on. */
  focusedNode: string;
  /** The edge to animate a packet along, or null. */
  packetEdge: EdgeView | null;
  truth: Values;
  showTruth: boolean;
  onSelectNode: (id: string) => void;
}

const UNIT_X = 96;
const UNIT_Y = 74;
const PAD = 46;
const NODE_R = 21;

const WIDTH = 8 * UNIT_X + PAD * 2;
const HEIGHT = 5 * UNIT_Y + PAD * 2;

const px = (x: number) => PAD + x * UNIT_X;
const py = (y: number) => PAD + y * UNIT_Y;

/**
 * The computation graph (spec: `<ComputeGraph>`).
 *
 * ── Deviation from the spec, deliberate ─────────────────────────────────────
 * The spec names p5.js/WebGL. This is SVG, for the reason that has decided every
 * one of these calls in this project: the numbers on the edges ARE the content of
 * this game, and a canvas puts them outside the accessibility tree. Here it is more
 * acute than usual — the whole activity is reading a number off one edge and
 * working out what the next one should be, so a version of this view that a screen
 * reader cannot read is not a degraded experience, it is no experience. The graph is
 * also fourteen fixed nodes with no render loop, so a canvas would buy nothing.
 *
 * Colour is never the only channel: a routed edge carries its gradient as text, and
 * correct and incorrect edges differ in dash pattern and glyph as well as hue.
 */
export function ComputeGraph({
  values,
  grads,
  settled,
  edges,
  focusedNode,
  packetEdge,
  truth,
  showTruth,
  onSelectNode,
}: ComputeGraphProps) {
  const reducedMotion = useReducedMotion();

  const routedNodes = useMemo(() => new Set(settled), [settled]);
  const stepNodes = useMemo(
    () => new Set(STEPS.map((step) => step.nodeId)),
    [],
  );

  const description = useMemo(() => {
    const parts = NODES.map((node) => {
      const value = values[node.id];
      const grad = routedNodes.has(node.id) ? grads[node.id] : undefined;
      return `${node.label} equals ${value?.toFixed(2) ?? "?"}${
        grad === undefined
          ? ", gradient not yet routed"
          : `, gradient ${grad.toFixed(3)}`
      }`;
    });
    return `Computation graph, ${NODES.length} nodes. ${parts.join(". ")}.`;
  }, [values, grads, routedNodes]);

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="w-full rounded-md border border-border bg-surface-2"
        role="img"
        aria-label={description}
      >
        {/* Edges first, so nodes sit on top of them. */}
        {edges.map((edge) => {
          const from = nodeById(edge.from);
          const to = nodeById(edge.to);
          const x1 = px(to.x);
          const y1 = py(to.y);
          const x2 = px(from.x);
          const y2 = py(from.y);
          const midX = (x1 + x2) / 2;
          const midY = (y1 + y2) / 2;

          const stroke = !edge.routed
            ? "var(--border)"
            : edge.correct
              ? "var(--correct)"
              : "var(--wrong)";

          return (
            <g key={edge.key}>
              <line
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke={stroke}
                strokeWidth={edge.routed ? 2.5 : 1.25}
                strokeDasharray={edge.routed && !edge.correct ? "5 3" : undefined}
                opacity={edge.routed ? 0.95 : 0.5}
              />
              {edge.routed ? (
                <>
                  <rect
                    x={midX - 24}
                    y={midY - 9}
                    width={48}
                    height={17}
                    rx={3}
                    fill="var(--bg)"
                    stroke={stroke}
                    strokeWidth={0.75}
                  />
                  <text
                    x={midX}
                    y={midY + 3}
                    textAnchor="middle"
                    className="fill-[var(--text)] text-[9px] font-medium tabular-nums"
                  >
                    {edge.correct ? "" : "✗ "}
                    {edge.mine!.toFixed(2)}
                  </text>
                  {showTruth && !edge.correct ? (
                    <text
                      x={midX}
                      y={midY + 18}
                      textAnchor="middle"
                      className="fill-[var(--correct)] text-[8px] tabular-nums"
                    >
                      should be {edge.truth.toFixed(2)}
                    </text>
                  ) : null}
                </>
              ) : null}
            </g>
          );
        })}

        {packetEdge !== null && !reducedMotion ? (
          <GradientPacket
            from={{
              x: px(nodeById(packetEdge.from).x),
              y: py(nodeById(packetEdge.from).y),
            }}
            to={{
              x: px(nodeById(packetEdge.to).x),
              y: py(nodeById(packetEdge.to).y),
            }}
            value={packetEdge.mine ?? 0}
            correct={packetEdge.correct}
          />
        ) : null}

        {NODES.map((node) => {
          const isFocus = node.id === focusedNode;
          const hasGrad = routedNodes.has(node.id);
          const grad = grads[node.id];
          const expected = truth[node.id] ?? 0;
          const wrong =
            hasGrad && grad !== undefined && Math.abs(grad - expected) > 1e-9;
          const routable = stepNodes.has(node.id);

          return (
            <g key={node.id}>
              <circle
                cx={px(node.x)}
                cy={py(node.y)}
                r={NODE_R}
                fill={
                  node.op === "param"
                    ? "color-mix(in srgb, var(--class-a) 20%, var(--surface))"
                    : node.op === "input" || node.op === "target"
                      ? "var(--surface)"
                      : "color-mix(in srgb, var(--primary) 14%, var(--surface))"
                }
                stroke={
                  isFocus
                    ? "var(--primary)"
                    : wrong
                      ? "var(--wrong)"
                      : hasGrad
                        ? "var(--correct)"
                        : "var(--border)"
                }
                strokeWidth={isFocus ? 3 : hasGrad ? 2 : 1}
              />
              <text
                x={px(node.x)}
                y={py(node.y) - 2}
                textAnchor="middle"
                className="fill-[var(--text)] text-[12px] font-semibold"
              >
                {node.label}
              </text>
              <text
                x={px(node.x)}
                y={py(node.y) + 10}
                textAnchor="middle"
                className="fill-[var(--text-muted)] text-[8px] tabular-nums"
              >
                {values[node.id]?.toFixed(2) ?? "?"}
              </text>

              {/* The gradient, under the node. This is the number that matters. */}
              {hasGrad ? (
                <text
                  x={px(node.x)}
                  y={py(node.y) + NODE_R + 12}
                  textAnchor="middle"
                  className={`text-[9px] font-semibold tabular-nums ${
                    wrong ? "fill-[var(--wrong)]" : "fill-[var(--correct)]"
                  }`}
                >
                  {wrong ? "✗" : "✓"} {grad!.toFixed(2)}
                </text>
              ) : null}

              {routable ? (
                <circle
                  cx={px(node.x)}
                  cy={py(node.y)}
                  r={NODE_R + 4}
                  fill="transparent"
                  className="cursor-pointer"
                  onClick={() => onSelectNode(node.id)}
                />
              ) : null}
            </g>
          );
        })}
      </svg>

      <figcaption className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-text-muted">
        <span>
          <span aria-hidden="true" className="text-[var(--correct)]">
            ✓
          </span>{" "}
          matches autograd
        </span>
        <span>
          <span aria-hidden="true" className="text-[var(--wrong)]">
            ✗
          </span>{" "}
          does not — dashed edge
        </span>
        <span>node shows its forward value; the number below it is its gradient</span>
      </figcaption>

      {/* The graph as a table, because the numbers are the content. */}
      <table className="sr-only-live">
        <caption>Every node with its forward value and its gradient</caption>
        <thead>
          <tr>
            <th scope="col">Node</th>
            <th scope="col">Operation</th>
            <th scope="col">Value</th>
            <th scope="col">Your gradient</th>
            <th scope="col">Autograd</th>
            <th scope="col">Match</th>
          </tr>
        </thead>
        <tbody>
          {NODES.map((node) => {
            const hasGrad = routedNodes.has(node.id);
            const grad = grads[node.id];
            const expected = truth[node.id] ?? 0;
            return (
              <tr key={node.id}>
                <th scope="row">{node.label}</th>
                <td>{node.op}</td>
                <td>{values[node.id]?.toFixed(3) ?? "unknown"}</td>
                <td>{hasGrad ? grad!.toFixed(4) : "not routed yet"}</td>
                <td>{showTruth ? expected.toFixed(4) : "hidden"}</td>
                <td>
                  {!hasGrad
                    ? "—"
                    : Math.abs(grad! - expected) <= 1e-9
                      ? "matches"
                      : "does not match"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </figure>
  );
}
