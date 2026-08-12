"use client";

import { useMemo } from "react";
import { ObjectiveSurface } from "./ObjectiveSurface";
import { TrajectoryTrail } from "./TrajectoryTrail";
import { CRACK_THRESHOLD, suggestNext } from "./ml";
import { bestObjective, useHeistStore } from "./store";

/**
 * Hyperparameter Heist — the visual lane.
 *
 * Reads the same store the code lane writes. The surface is the hero, but note
 * what it is showing: the surrogate's belief while the run is live, the truth only
 * once it is over. The trail beside it is the complete record in all four dials,
 * which the surface cannot be.
 */
export function VisualLane() {
  const trials = useHeistStore((s) => s.trials);
  const dials = useHeistStore((s) => s.dials);
  const revealed = useHeistStore((s) => s.revealed);
  const strategy = useHeistStore((s) => s.strategy);
  const budget = useHeistStore((s) => s.budget);
  const best = useHeistStore(bestObjective);

  // Memoised on the trial list: suggestNext fits a GP and scans 512 candidates,
  // which must not run on every unrelated store read.
  const suggestion = useMemo(
    () =>
      strategy === "bayesian" && !revealed ? suggestNext(trials) : null,
    [strategy, revealed, trials],
  );

  return (
    <div className="flex flex-col gap-4">
      <section aria-labelledby="surface-heading">
        {/* h2: the lane sits directly under the page h1. */}
        <h2
          id="surface-heading"
          className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1"
        >
          <span className="font-display text-lg font-semibold">
            {revealed ? "The safe, revealed" : "What you know so far"}
          </span>
          <span className="text-xs font-normal text-text-muted">
            opens at {Math.round(CRACK_THRESHOLD * 100)}%
            {Number.isFinite(best)
              ? ` · best ${(best * 100).toFixed(1)}%`
              : " · nothing tried"}
          </span>
        </h2>
        <div className="max-w-[340px]">
          <ObjectiveSurface
            trials={trials}
            dials={dials}
            revealed={revealed}
            suggestion={suggestion}
          />
        </div>
      </section>

      <section aria-labelledby="trail-heading">
        <h2
          id="trail-heading"
          className="mb-1.5 font-display text-sm font-semibold"
        >
          Every try
        </h2>
        <TrajectoryTrail trials={trials} budget={budget} />
      </section>
    </div>
  );
}
