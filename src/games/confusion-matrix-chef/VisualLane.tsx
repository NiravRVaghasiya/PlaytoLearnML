"use client";

import { useMemo } from "react";
import { ConfusionMatrix } from "./ConfusionMatrix";
import { RocCurve } from "./RocCurve";
import { auc, confusionAt, rocCurve, rocPointAt } from "./ml";
import { currentScenario, useChefStore } from "./store";

/**
 * Confusion Matrix Chef — the visual lane.
 *
 * Reads the same store the code lane writes. Both views are derived from the
 * samples and the threshold on every render rather than cached, which is the
 * cheapest way to guarantee the spec's promise that "the confusion matrix
 * recomputes exactly as it would in production" — there is no second copy of the
 * counts anywhere to fall out of step.
 */
export function VisualLane() {
  const scenario = useChefStore(currentScenario);
  const samples = useChefStore((s) => s.samples);
  const threshold = useChefStore((s) => s.threshold);

  const matrix = useMemo(
    () => confusionAt(samples, threshold),
    [samples, threshold],
  );
  // The curve depends only on the samples, so it is memoised apart from the
  // threshold — sliding must not recompute a few hundred points every frame.
  const curve = useMemo(() => rocCurve(samples), [samples]);
  const area = useMemo(() => auc(samples), [samples]);
  const point = useMemo(
    () => rocPointAt(samples, threshold),
    [samples, threshold],
  );

  return (
    <div className="flex flex-col gap-4">
      <section aria-labelledby="matrix-heading">
        {/* h2: the lane sits directly under the page h1. */}
        <h2
          id="matrix-heading"
          className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1"
        >
          <span className="font-display text-lg font-semibold">
            Confusion matrix
          </span>
          <span className="font-mono text-xs font-normal text-text-muted">
            cutoff {threshold.toFixed(2)}
          </span>
        </h2>
        <ConfusionMatrix
          matrix={matrix}
          scenario={scenario}
          total={samples.length}
        />
      </section>

      <section aria-labelledby="roc-heading">
        <h2
          id="roc-heading"
          className="mb-1.5 font-display text-sm font-semibold"
        >
          Every cutoff at once
        </h2>
        <div className="max-w-[320px]">
          <RocCurve curve={curve} point={point} auc={area} />
        </div>
      </section>
    </div>
  );
}
