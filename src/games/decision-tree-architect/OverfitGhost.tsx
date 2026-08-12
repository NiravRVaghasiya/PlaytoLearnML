"use client";

import { Ghost } from "lucide-react";
import { MIN_HONEST_LEAF } from "./ml";

export interface OverfitGhostProps {
  /** How far below the player's best validation accuracy they are now. */
  shortfall: number;
  peakValidation: number;
  peakDepth: number;
  starved: number;
}

/**
 * The overfit ghost (spec: "adding too many splits summons an overfit ghost as
 * validation accuracy dips").
 *
 * It appears on a measured condition — validation below the best this player
 * already reached — and it says which depth they had it at, so the response is
 * obvious rather than ominous. A warning that cannot be acted on is just noise.
 *
 * Not a live region: it appears while the player is building, and announcing it
 * on every gate would talk over the metric readout, which is the one channel that
 * should be interrupting. The accuracy pair already carries the same fact in its
 * accessible description.
 */
export function OverfitGhost({
  shortfall,
  peakValidation,
  peakDepth,
  starved,
}: OverfitGhostProps) {
  return (
    <div
      data-testid="overfit-ghost"
      className="flex items-start gap-2.5 rounded-md border border-warn/50 bg-warn/10 p-2.5"
    >
      <Ghost aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-warn" />
      <div>
        <p className="text-xs font-semibold">
          Overfitting — {(shortfall * 100).toFixed(1)} points below your best
        </p>
        <p className="mt-0.5 text-[11px] text-text-muted">
          You reached {Math.round(peakValidation * 100)}% validation at depth{" "}
          {peakDepth}.{" "}
          {starved > 0
            ? `${starved} leaf${starved === 1 ? "" : "s"} now hold${
                starved === 1 ? "s" : ""
              } fewer than ${MIN_HONEST_LEAF} plots, which is not enough to decide anything from.`
            : "The extra gates are fitting verdicts that are wrong in the data."}{" "}
          Pruning back will raise validation accuracy and lower training accuracy —
          that trade is the whole point.
        </p>
      </div>
    </div>
  );
}
