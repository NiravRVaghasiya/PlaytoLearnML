"use client";

import { useMemo } from "react";
import { DecisionSurface } from "./DecisionSurface";
import { LossCurve } from "./LossCurve";
import { NetworkEditor } from "./NetworkEditor";
import { architectureOf } from "./ml";
import { pattern, useNeuronForgeStore } from "./store";

/**
 * Neuron Forge — the visual lane.
 *
 * Reads the same store the code lane writes to. The decision surface is the hero:
 * it is the only view that shows *what the architecture can represent*, which is
 * the thing being taught. The network diagram and loss curve explain how it got
 * there.
 */
export function VisualLane() {
  const layers = useNeuronForgeStore((s) => s.layers);
  const arch = useMemo(() => architectureOf(layers), [layers]);
  const spec = useNeuronForgeStore(pattern);
  const dataset = useNeuronForgeStore((s) => s.dataset);
  const surface = useNeuronForgeStore((s) => s.surface);
  const accuracy = useNeuronForgeStore((s) => s.accuracy);
  const lossHistory = useNeuronForgeStore((s) => s.lossHistory);
  const training = useNeuronForgeStore((s) => s.training);

  return (
    <div className="flex flex-col gap-4">
      <section aria-labelledby="surface-heading">
        {/* h2: the lane sits directly under the page h1. */}
        <h2
          id="surface-heading"
          className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1"
        >
          <span className="font-display text-lg font-semibold">
            {spec.name}
          </span>
          <span className="text-xs font-normal text-text-muted">
            {spec.hint}
          </span>
        </h2>
        <DecisionSurface
          surface={surface}
          points={dataset.train}
          accuracy={accuracy}
        />
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <section aria-labelledby="network-heading">
          <h2
            id="network-heading"
            className="mb-1.5 font-display text-sm font-semibold"
          >
            Your network
          </h2>
          <NetworkEditor architecture={arch} />
        </section>

        <section aria-labelledby="loss-heading">
          <h2
            id="loss-heading"
            className="mb-1.5 font-display text-sm font-semibold"
          >
            Loss
          </h2>
          <LossCurve
            history={lossHistory}
            totalEpochs={Math.max(lossHistory.length, 1)}
            training={training}
          />
        </section>
      </div>
    </div>
  );
}
