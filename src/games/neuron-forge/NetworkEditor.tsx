"use client";

import { useMemo } from "react";
import {
  Background,
  Handle,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { Architecture, Layer } from "./ml";

const COLUMN_WIDTH = 132;
const ROW_HEIGHT = 38;

type NeuronKind = "input" | "hidden" | "output";

interface NeuronData extends Record<string, unknown> {
  label: string;
  sublabel?: string;
  kind: NeuronKind;
}

/**
 * A single unit (spec: `<NeuronNode>`).
 *
 * Handles are hidden but still present — React Flow needs them as anchor points
 * for the edges, and showing 150 connection dots would only add noise.
 */
function NeuronNode({ data }: NodeProps) {
  const { label, sublabel, kind } = data as NeuronData;

  const tone =
    kind === "input"
      ? "border-border bg-surface text-text-muted"
      : kind === "output"
        ? "border-primary/60 bg-primary/10 text-text"
        : "border-border bg-surface-2 text-text";

  return (
    <div
      className={`flex min-w-[68px] flex-col items-center rounded-md border px-2 py-1 ${tone}`}
    >
      <span className="font-mono text-[11px] leading-tight">{label}</span>
      {sublabel ? (
        <span className="text-[9px] leading-tight text-text-muted">
          {sublabel}
        </span>
      ) : null}
      <Handle
        type="target"
        position={Position.Left}
        isConnectable={false}
        className="!size-1 !min-w-0 !border-0 !bg-transparent"
      />
      <Handle
        type="source"
        position={Position.Right}
        isConnectable={false}
        className="!size-1 !min-w-0 !border-0 !bg-transparent"
      />
    </div>
  );
}

/** Stable identity: a fresh object here makes React Flow warn every render. */
const NODE_TYPES = { neuron: NeuronNode };

function buildGraph(layers: Layer[]): { nodes: Node[]; edges: Edge[] } {
  // Columns of ids, so edges can be produced by zipping neighbours.
  const columns: string[][] = [];
  const nodes: Node[] = [];

  const tallest = Math.max(2, ...layers.map((layer) => layer.neurons), 1);
  const columnCount = layers.length + 2;

  const place = (
    column: number,
    ids: string[],
    build: (index: number) => NeuronData,
  ) => {
    const offset = ((tallest - ids.length) * ROW_HEIGHT) / 2;
    ids.forEach((id, index) => {
      nodes.push({
        id,
        type: "neuron",
        position: {
          x: column * COLUMN_WIDTH,
          y: offset + index * ROW_HEIGHT,
        },
        data: build(index),
        draggable: false,
        selectable: false,
        connectable: false,
      });
    });
    columns.push(ids);
  };

  place(0, ["in-0", "in-1"], (index) => ({
    label: index === 0 ? "x₁" : "x₂",
    sublabel: "input",
    kind: "input",
  }));

  layers.forEach((layer, layerIndex) => {
    place(
      layerIndex + 1,
      Array.from({ length: layer.neurons }, (_, index) => `h${layerIndex}-${index}`),
      () => ({
        label: layer.activation,
        sublabel: `layer ${layerIndex + 1}`,
        kind: "hidden",
      }),
    );
  });

  place(columnCount - 1, ["out"], () => ({
    label: "ŷ",
    sublabel: "sigmoid",
    kind: "output",
  }));

  // Dense layers are fully connected, so the diagram is too — that visual weight
  // is the point. It is why 8 neurons costs so much more than 2.
  const edges: Edge[] = [];
  for (let column = 0; column < columns.length - 1; column += 1) {
    for (const source of columns[column]!) {
      for (const target of columns[column + 1]!) {
        edges.push({
          id: `${source}->${target}`,
          source,
          target,
          style: { stroke: "var(--border)", strokeWidth: 1 },
        });
      }
    }
  }

  return { nodes, edges };
}

export interface NetworkEditorProps {
  architecture: Architecture;
}

/**
 * The network diagram (spec: `<NetworkEditor>`, React Flow).
 *
 * Deliberately read-only. The spec describes dragging neurons around, but the
 * thing being edited is really two numbers and an enum per layer, and dragging is
 * the one way of expressing that which a keyboard or a screen reader cannot use.
 * So editing lives in the controls as steppers and selects, and React Flow does
 * what it is genuinely best at: drawing the graph that results.
 *
 * All interaction is off, including React Flow's own keyboard handling, so the
 * canvas adds no focus stops to the page. The `figcaption` states the
 * architecture in words for anyone not reading the picture.
 */
export function NetworkEditor({ architecture }: NetworkEditorProps) {
  const { nodes, edges } = useMemo(
    () => buildGraph(architecture.layers),
    [architecture.layers],
  );

  const described =
    architecture.layers.length === 0
      ? "two inputs wired straight to one sigmoid output, with no hidden layer"
      : `two inputs, then ${architecture.layers
          .map(
            (layer, index) =>
              `hidden layer ${index + 1} with ${layer.neurons} ${layer.activation} neuron${
                layer.neurons === 1 ? "" : "s"
              }`,
          )
          .join(", then ")}, then one sigmoid output`;

  return (
    <figure className="m-0">
      <div className="h-[248px] w-full overflow-hidden rounded-md border border-border bg-surface-2">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={NODE_TYPES}
          fitView
          fitViewOptions={{ padding: 0.18 }}
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
          <Background gap={18} size={1} color="var(--border)" />
        </ReactFlow>
      </div>

      <figcaption className="mt-1.5 text-xs text-text-muted">
        {architecture.totalNeurons === 0
          ? "No hidden neurons: "
          : `${architecture.totalNeurons} hidden neuron${
              architecture.totalNeurons === 1 ? "" : "s"
            }: `}
        {described}. Every line is a weight the optimizer has to learn.
      </figcaption>
    </figure>
  );
}
