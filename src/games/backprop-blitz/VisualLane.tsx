"use client";

import { useMemo } from "react";
import {
  STEPS,
  autograd,
  consumersOf,
  edgeKey,
  fanOut,
  nodeById,
  replay,
} from "./ml";
import {
  edgeViews,
  scenario,
  useBlitzStore,
} from "./store";
import { ComputeGraph } from "./ComputeGraph";
import { NodeInspector } from "./NodeInspector";

/**
 * The visual lane.
 *
 * Selectors take stable slices only — never a freshly built object, which breaks
 * `useSyncExternalStore`'s cached-snapshot check. Both traces are derived here in
 * `useMemo` from the scenario and the routing, because a stored gradient that has
 * not caught up with the player's latest choice would be the one bug in this game
 * that actively teaches the wrong thing.
 */
export function VisualLane() {
  const routing = useBlitzStore((s) => s.routing);
  const focused = useBlitzStore((s) => s.focused);
  const showTruth = useBlitzStore((s) => s.showTruth);
  const scenarioIndex = useBlitzStore((s) => s.scenarioIndex);
  const choose = useBlitzStore((s) => s.choose);
  const clearStep = useBlitzStore((s) => s.clearStep);
  const focusStep = useBlitzStore((s) => s.focusStep);
  const current = useBlitzStore(scenario);

  const leaves = current.leaves;

  const mine = useMemo(() => replay(leaves, routing), [leaves, routing]);
  const truth = useMemo(() => autograd(leaves), [leaves]);
  const edges = useMemo(
    () => edgeViews(mine.edges, truth.edges),
    [mine.edges, truth.edges],
  );

  const step = STEPS[focused]!;
  const chosen = routing[focused] ?? null;

  /** Contributions waiting at a fan-out node, in consumer order. */
  const pending = useMemo(() => {
    if (step.kind !== "accumulate") return [];
    return consumersOf(step.nodeId).map(
      (consumer) => mine.edges[edgeKey(consumer, step.nodeId)] ?? 0,
    );
  }, [step, mine.edges]);

  const incoming = useMemo(() => {
    if (step.kind === "accumulate") return null;
    return mine.settled.includes(step.nodeId)
      ? (mine.grads[step.nodeId] ?? 0)
      : null;
  }, [step, mine]);

  const outgoing = useMemo(() => {
    if (chosen === null) return [];
    if (step.kind === "accumulate") {
      // An accumulate step produces the node's own gradient, not an edge.
      return [
        {
          to: step.nodeId,
          mine: mine.grads[step.nodeId] ?? null,
          truth: truth.grads[step.nodeId] ?? 0,
          correct:
            Math.abs(
              (mine.grads[step.nodeId] ?? 0) - (truth.grads[step.nodeId] ?? 0),
            ) < 1e-9,
        },
      ];
    }
    return nodeById(step.nodeId).inputs.map((input) => {
      const key = edgeKey(step.nodeId, input);
      const value = mine.edges[key];
      const expected = truth.edges[key] ?? 0;
      return {
        to: input,
        mine: value ?? null,
        truth: expected,
        correct: value !== undefined && Math.abs(value - expected) < 1e-9,
      };
    });
  }, [chosen, step, mine, truth]);

  /** Animate a packet along the edge the last routing produced. */
  const packetEdge = useMemo(() => {
    if (chosen === null || step.kind !== "route") return null;
    const first = nodeById(step.nodeId).inputs[0];
    if (first === undefined) return null;
    return edges.find((edge) => edge.key === edgeKey(step.nodeId, first)) ?? null;
  }, [chosen, step, edges]);

  return (
    <div className="flex flex-col gap-5">
      <div>
        <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-sm font-semibold">
            {current.title}
            <span className="ml-2 font-normal text-text-muted">
              scenario {scenarioIndex + 1} of 3
            </span>
          </h2>
          <span className="text-xs text-text-muted">
            forward pass done · loss {truth.values.L!.toFixed(3)}
          </span>
        </div>
        <ComputeGraph
          values={mine.values}
          grads={mine.grads}
          settled={mine.settled}
          edges={edges}
          focusedNode={step.nodeId}
          packetEdge={packetEdge}
          truth={truth.grads}
          showTruth={showTruth}
          onSelectNode={(id) => {
            // Jump to the step that routes this node, preferring the accumulate
            // step when there is one, since it comes first.
            const index = STEPS.findIndex(
              (candidate) =>
                candidate.nodeId === id &&
                (fanOut(id) > 1 ? candidate.kind === "accumulate" : true),
            );
            if (index >= 0) focusStep(index);
          }}
        />
      </div>

      <div className="border-t border-border pt-4">
        <NodeInspector
          step={step}
          stepIndex={focused}
          chosen={chosen}
          values={mine.values}
          incoming={incoming}
          pending={pending}
          outgoing={outgoing}
          answeredCount={Object.keys(routing).length}
          onChoose={(ruleId) => choose(focused, ruleId)}
          onClear={() => clearStep(focused)}
          onStep={focusStep}
        />
      </div>
    </div>
  );
}
