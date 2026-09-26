"use client";

import { useEffect, useMemo, type ReactNode } from "react";
import {
  Handle,
  Position,
  ReactFlow,
  useReactFlow,
  useStore,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { PurityPie } from "./PurityPie";
import {
  MIN_HONEST_LEAF,
  ROOT_ID,
  describeSplit,
  type NodeStat,
  type Tree,
} from "./ml";

const NODE_WIDTH = 116;
const LEVEL_HEIGHT = 92;

/** Breathing room round the fitted tree, as a fraction of the pane. */
const FIT_PADDING = 0.15;
/**
 * Zoom bounds for the fit. The ceiling stops a lone root being blown up to
 * twice size (React Flow's default maxZoom); the floor is low enough for the
 * widest tree the plots grow to fit a phone-width pane. Measured: greedy grown
 * out on the hillside has 70 leaves, about 8,100 px at 116 px a slot, which
 * needs a zoom near 0.03 in a 330 px pane — React Flow's default floor of 0.5
 * would clip anything past about 1,750 px. At that zoom the labels are
 * unreadable, which is why the outline list below states the tree in words.
 */
const MIN_ZOOM = 0.02;
const MAX_ZOOM = 1.25;

/**
 * Re-fit the viewport whenever the set of nodes, or the pane's size, changes.
 *
 * `fitView` on <ReactFlow> is a mount-time option in v12: it fits the first
 * render — a single root — and never again, and with pan and zoom switched off
 * every gate built after that grew off the bottom of a fixed 300px frame with
 * no way to reach it. Keyed on the node ids, so moving the selection (which
 * rebuilds every node object) does not re-fit, but growing or pruning does.
 * `fitView()` queues itself until React Flow has measured the new nodes.
 *
 * Keyed on the pane's measured size as well. React Flow's resize observer only
 * records the new size and keeps the old transform, so a fitted tree in a pane
 * that then narrowed (a phone turned upright, a window made smaller) stayed
 * clipped past the right edge until the next gate. Exported for its tests.
 */
export function RefitOnChange({ signature }: { signature: string }) {
  const { fitView } = useReactFlow();
  const width = useStore((state) => state.width);
  const height = useStore((state) => state.height);
  useEffect(() => {
    void fitView({
      padding: FIT_PADDING,
      minZoom: MIN_ZOOM,
      maxZoom: MAX_ZOOM,
      duration: 0,
    });
  }, [signature, width, height, fitView]);
  return null;
}

/**
 * React Flow stamps role="application" on its wrapper, after spreading our
 * props, so no prop can override it. On a read-only diagram with nothing
 * focusable that role only switches screen readers into focus mode for no
 * reason. React never rewrites an attribute whose value has not changed, so
 * removing it once on mount holds for the component's life.
 */
function dropApplicationRole(wrapper: HTMLDivElement | null) {
  wrapper?.removeAttribute("role");
}

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
/** Stable for the same reason; the mount-time fit, before RefitOnChange runs. */
const FIT_OPTIONS = { padding: FIT_PADDING, minZoom: MIN_ZOOM, maxZoom: MAX_ZOOM };

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
          // Otherwise each edge is an image announced as "Edge from n0 to
          // n0L". The outline list below carries the structure in words.
          domAttributes: { "aria-hidden": true },
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
      // Hidden for the same reason as the edges: the outline list says it all.
      domAttributes: { "aria-hidden": true },
    });

    return x;
  };

  place(Object.keys(tree)[0] === undefined ? "" : "n0");
  return { nodes, edges };
}

/**
 * The tree in words: a nested list, root first, each gate followed by its "yes"
 * and "no" branches. This is the diagram's text alternative — which gate sits
 * where, how many plots reach it and what each leaf decides — for anyone who
 * cannot see the canvas, and for trees too wide to read once fitted.
 */
function outline(
  tree: Tree,
  stats: Record<string, NodeStat>,
  id: string,
  branch: string | null,
): ReactNode {
  const node = tree[id];
  if (!node) return null;
  const stat = stats[id];
  const total = stat?.counts.total ?? 0;
  const plots = `${total} plot${total === 1 ? "" : "s"}`;
  const impurity = `impurity ${(stat?.gini ?? 0).toFixed(2)}`;
  const prefix = branch === null ? "" : `${branch}: `;

  if (node.split === null) {
    const starved = total > 0 && total < MIN_HONEST_LEAF;
    return (
      <li key={id}>
        {`${prefix}leaf ${id}, ${plots}, ${impurity}, says ${
          (stat?.majority ?? 0) === 1 ? "safe" : "unsafe"
        }${starved ? ", too few plots to decide from" : ""}`}
      </li>
    );
  }

  return (
    <li key={id}>
      {`${prefix}gate ${id}, ${describeSplit(node.split)}, ${plots}, ${impurity}${
        stat?.gain === null || stat?.gain === undefined
          ? ""
          : `, gain ${stat.gain.toFixed(3)}`
      }`}
      <ul>
        {node.left ? outline(tree, stats, node.left, "yes") : null}
        {node.right ? outline(tree, stats, node.right, "no") : null}
      </ul>
    </li>
  );
}

/** The outline as a list. Exported for its tests; React Flow needs a real layout. */
export function TreeOutline({
  tree,
  stats,
}: {
  tree: Tree;
  stats: Record<string, NodeStat>;
}) {
  return (
    // A list, not a table, so Tailwind's sr-only is enough: it cannot widen
    // the page the way a hidden <table> does on a phone.
    <ul className="sr-only" aria-label="The tree, gate by gate">
      {outline(tree, stats, ROOT_ID, null)}
    </ul>
  );
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
 * canvas adds no focus stops, and the view re-fits itself as the tree grows (see
 * RefitOnChange) — pan and zoom stay off, so the fit is the only way to see a
 * gate, and it has to include every one. The drawing is hidden from assistive
 * tech in favour of a nested outline list that states the tree in words.
 */
export function TreeCanvas({ tree, stats, selectedNodeId }: TreeCanvasProps) {
  const { nodes, edges } = useMemo(
    () => layout(tree, stats, selectedNodeId),
    [tree, stats, selectedNodeId],
  );
  const signature = useMemo(
    () => nodes.map((node) => node.id).sort().join(","),
    [nodes],
  );

  const leafCount = nodes.filter((node) => (node.data as GateData).isLeaf).length;
  const gateCount = nodes.length - leafCount;

  return (
    <figure className="m-0">
      <div className="h-[300px] w-full overflow-hidden rounded-md border border-border bg-surface-2">
        <ReactFlow
          ref={dropApplicationRole}
          nodes={nodes}
          edges={edges}
          nodeTypes={NODE_TYPES}
          fitView
          fitViewOptions={FIT_OPTIONS}
          minZoom={MIN_ZOOM}
          maxZoom={MAX_ZOOM}
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
        >
          <RefitOnChange signature={signature} />
        </ReactFlow>
      </div>

      <TreeOutline tree={tree} stats={stats} />

      <figcaption className="mt-1.5 text-xs text-text-muted">
        {gateCount === 0
          ? "One leaf, no gates: every plot gets the same verdict."
          : `${gateCount} gate${gateCount === 1 ? "" : "s"} and ${leafCount} leaves. Each node shows its class mix, how many plots reach it, and the gain its gate achieved. Amber outlines mark leaves with too few plots to decide from.`}
      </figcaption>
    </figure>
  );
}
