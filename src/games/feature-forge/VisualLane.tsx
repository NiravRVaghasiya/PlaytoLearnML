"use client";

import { ImportanceBars } from "./ImportanceBars";
import { MetricGauge } from "./MetricGauge";
import { TARGET_LIFT } from "./ml";
import { hasLeak, useForgeStore } from "./store";

/**
 * Feature Forge — the visual lane.
 *
 * Reads the same store the Python lane writes. The gauge is the hero because the
 * whole game is one number moving while the model stays still; the importance bars
 * underneath answer the follow-up question of WHICH column the model started
 * leaning on.
 */
export function VisualLane() {
  const baselineScore = useForgeStore((s) => s.baselineScore);
  const currentScore = useForgeStore((s) => s.currentScore);
  const trainScore = useForgeStore((s) => s.trainScore);
  const ready = useForgeStore((s) => s.ready);
  const training = useForgeStore((s) => s.training);
  const columnNames = useForgeStore((s) => s.columnNames);
  const importances = useForgeStore((s) => s.importances);
  const leaked = useForgeStore(hasLeak);

  return (
    <div className="flex flex-col gap-4">
      <section aria-labelledby="gauge-heading">
        {/* h2: the lane sits directly under the page h1. */}
        <h2
          id="gauge-heading"
          className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1"
        >
          <span className="font-display text-lg font-semibold">
            Validation accuracy
          </span>
          <span className="text-xs font-normal text-text-muted">
            {training
              ? "retraining the fixed model…"
              : `beat baseline by ${Math.round(TARGET_LIFT * 100)} points`}
          </span>
        </h2>
        <MetricGauge
          baselineScore={baselineScore}
          currentScore={currentScore}
          trainScore={trainScore}
          ready={ready}
          leaked={leaked}
        />
      </section>

      <section aria-labelledby="importance-heading">
        <h2
          id="importance-heading"
          className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1"
        >
          <span className="font-display text-sm font-semibold">
            What the model leaned on
          </span>
          <span className="text-xs font-normal text-text-muted">
            {columnNames.length} matrix column
            {columnNames.length === 1 ? "" : "s"}
          </span>
        </h2>
        <ImportanceBars columnNames={columnNames} importances={importances} />
      </section>
    </div>
  );
}
