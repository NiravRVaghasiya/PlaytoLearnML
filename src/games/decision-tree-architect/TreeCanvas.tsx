"use client";

import { useMemo } from "react";
import {
  Handle,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { PurityPie } from "./PurityPie";
import {
  MIN_HONEST_LEAF,
  describeSplit,
  type NodeStat,
  type Tree,
} from "./ml";

const NODE_WIDTH = 116;
const LEVEL_HEIGHT = 92;

interface GateData extends Record<string, unknown> {
  label: string;
  counts: NodeStat["counts"];
  gini: number;
  gain: number | null;
  isLeaf: boolean;
  selected: boolean;
  starved: boolean;
  majority: 0 | 1;
}

/** One node of the tree (spec: `<SplitGate>` as it appears on the canvas). */
function GateNode({ data }: NodeProps) {
  const { label, counts, gini, gain, isLeaf, selected, starved, majority } =
    data as GateData;

  return (
    <div
      className={`flex w-[110px] flex-col items-center rounded-md border px-1.5 py-1 ${
        selected
          ? "border-primary bg-primary/10"
          : starved
            ? "border-warn/70 bg-warn/10"
            : isLeaf
              ? "border-border bg-surface"
              : "border-border bg-surface-2"
      }`}
    >
      <span className="font-mono text-[10px] leading-tight">{label}</span>
      <PurityPie counts={counts} gini={gini} size={28} decorative />
      <span className="text-[9px] leading-tight text-text-muted">
        {counts.total} plot{counts.total === 1 ? "" : "s"}
        {isLeaf ? ` · says ${majority === 1 ? "safe" : "unsafe"}` : ""}
      </span>
      {gain !== null ? (
        <span className="font-mono text-[8px] leading-tight text-text-muted">
          gain {gain.toFixed(3)}
        </span>
      ) : null}
      <Handle
        type="target"
        position={Position.Top}
        isConnectable={false}
        className="!size-1 !min-w-0 !border-0 !bg-transparent"
      />
      <Handle
        type="source"
        position={Position.Bottom}
        isConnectable={false}
        className="!size-1 !min-w-0 !border-0 !bg-transparent"
      />
    </div>
  );
}

/** Stable identity: a fresh object makes React Flow warn every render. */
const NODE_TYPES = { gate: GateNode };

/**
 * Lay the tree out by walking it, assigning each leaf the next horizontal slot.
 *
 * A parent sits centred over its children, which is what makes a decision tree
 * legible — the alternative, fixed slots per depth, leaves long gaps once one
 * branch grows deeper than another.
 */
function layout(
  tree: Tree,
  stats: Record<string, NodeStat>,
  selectedNodeId: string,
): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  let nextSlot = 0;

  const place = (id: string): number => {
    const node = tree[id];
    if (!node) return 0;
    const stat = stats[id];
    const counts = stat?.counts ?? { negative: 0, positive: 0, total: 0 };

    let x: number;
    if (node.split === null) {
      x = nextSlot * NODE_WIDTH;
      nextSlot += 1;
    } else {
      const leftX = node.left ? place(node.left) : 0;
      const rightX = node.right ? place(node.right) : 0;
      x = (leftX + rightX) / 2;

      for (const [childId, branch] of [
        [node.left, "yes"],
        [node.right, "no"],
      ] as const) {
        if (!childId) continue;
        edges.push({
          id: `${id}-${childId}`,
          source: id,
          target: childId,
          label: branch,
          labelStyle: { fill: "var(--text-muted)", fontSize: 9 },
          labelBgStyle: { fill: "var(--bg)" },
          style: { stroke: "var(--border)", strokeWidth: 1.25 },
        });
      }
    }

    nodes.push({
      id,
      type: "gate",
      position: { x, y: node.depth * LEVEL_HEIGHT },
      data: {
        label: node.split === null ? "leaf" : describeSplit(node.split),
        counts,
        gini: stat?.gini ?? 0,
        gain: stat?.gain ?? null,
        isLeaf: node.split === null,
        selected: id === selectedNodeId,
        starved:
          node.split === null &&
          counts.total > 0 &&
          counts.total < MIN_HONEST_LEAF,
        majority: stat?.majority ?? 0,
      } satisfies GateData,
      draggable: false,
      selectable: false,
      connectable: false,
    });

    return x;
  };

  place(Object.keys(tree)[0] === undefined ? "" : "n0");
  return { nodes, edges };
}

export interface TreeCanvasProps {
  tree: Tree;
  stats: Record<string, NodeStat>;
  selectedNodeId: string;
}

/**
 * The tree diagram (spec: `<TreeCanvas>`, React Flow).
 *
 * Read-only, like Neuron Forge's editor and for the same reason. The spec
 * describes dragging gates onto the canvas, but what a gate actually is is a
 * feature and a threshold, and dragging is the one way of choosing those that a
 * keyboard cannot do. Selection and editing live in the split picker beside this;
 * React Flow does what it is genuinely good at, which is drawing the graph.
 *
 * All interaction is off including React Flow's own keyboard handling, so the
 * canvas adds no focus stops. The figcaption states the tree in words.
 */
export function TreeCanvas({ tree, stats, selectedNodeId }: TreeCanvasProps) {
  const { nodes, edges } = useMemo(
    () => layout(tree, stats, selectedNodeId),
    [tree, stats, selectedNodeId],
  );

  const leafCount = nodes.filter((node) => (node.data as GateData).isLeaf).length;
  const gateCount = nodes.length - leafCount;

  return (
    <figure className="m-0">
      <div className="h-[300px] w-full overflow-hidden rounded-md border border-border bg-surface-2">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={NODE_TYPES}
          fitView
          fitViewOptions={{ padding: 0.15 }}
          nodesDraggable={false}
          nodesConnectable={false}
          nodesFocusable={false}
          edgesFocusable={false}
          elementsSelectable={false}
          panOnDrag={false}
          panOnScroll={false}
          zoomOnScroll={false}
          zoomOnPinch={false}
          zoomOnDoubleClick={false}
          preventScrolling={false}
          disableKeyboardA11y
          proOptions={{ hideAttribution: false }}
        />
      </div>

      <figcaption className="mt-1.5 text-xs text-text-muted">
        {gateCount === 0
          ? "One leaf, no gates: every plot gets the same verdict."
          : `${gateCount} gate${gateCount === 1 ? "" : "s"} and ${leafCount} leaves. Each node shows its class mix, how many plots reach it, and the gain its gate achieved. Amber outlines mark leaves with too few plots to decide from.`}
      </figcaption>
    </figure>
  );
}
