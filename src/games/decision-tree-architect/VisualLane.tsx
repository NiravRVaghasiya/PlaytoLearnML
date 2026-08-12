"use client";

import { useMemo } from "react";
import { AccuracyPair } from "./AccuracyPair";
import { OverfitGhost } from "./OverfitGhost";
import { TreeCanvas } from "./TreeCanvas";
import { nodeStats } from "./ml";
import {
  currentRound,
  isGivingGroundBack,
  useArchitectStore,
} from "./store";

/**
 * Decision Tree Architect — the visual lane.
 *
 * Reads the same store the code lane writes. The accuracy pair sits above the
 * tree on purpose: the tree shows what you built, and the pair shows what it cost,
 * and the second is the thing players stop looking at once the first gets
 * interesting.
 */
export function VisualLane() {
  const round = useArchitectStore(currentRound);
  const tree = useArchitectStore((s) => s.tree);
  const train = useArchitectStore((s) => s.dataset.train);
  const achievable = useArchitectStore((s) => s.dataset.achievable);
  const selectedNodeId = useArchitectStore((s) => s.selectedNodeId);
  const trainAccuracy = useArchitectStore((s) => s.trainAccuracy);
  const validationAccuracy = useArchitectStore((s) => s.validationAccuracy);
  const peakValidation = useArchitectStore((s) => s.peakValidation);
  const peakDepth = useArchitectStore((s) => s.peakDepth);
  const starved = useArchitectStore((s) => s.starved);
  const givingBack = useArchitectStore(isGivingGroundBack);

  // Derived per render rather than stored: the per-node figures must always match
  // the tree that produced them.
  const stats = useMemo(() => nodeStats(tree, train), [tree, train]);

  return (
    <div className="flex flex-col gap-4">
      <section aria-labelledby="accuracy-heading">
        {/* h2: the lane sits directly under the page h1. */}
        <h2
          id="accuracy-heading"
          className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1"
        >
          <span className="font-display text-lg font-semibold">
            Train against validation
          </span>
          <span className="text-xs font-normal text-text-muted">
            {train.length} surveyed plots · ceiling{" "}
            {Math.round(achievable * 100)}%
          </span>
        </h2>
        <AccuracyPair
          trainAccuracy={trainAccuracy}
          validationAccuracy={validationAccuracy}
          achievable={achievable}
          target={round.target}
          peakValidation={peakValidation}
          peakDepth={peakDepth}
        />
      </section>

      {givingBack ? (
        <OverfitGhost
          shortfall={peakValidation - validationAccuracy}
          peakValidation={peakValidation}
          peakDepth={peakDepth}
          starved={starved}
        />
      ) : null}

      <section aria-labelledby="tree-heading">
        <h2
          id="tree-heading"
          className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1"
        >
          <span className="font-display text-sm font-semibold">Your tree</span>
          <span className="text-xs font-normal text-text-muted">
            {round.name} · depth limit {round.maxDepth}
          </span>
        </h2>
        <TreeCanvas
          tree={tree}
          stats={stats}
          selectedNodeId={selectedNodeId}
        />
      </section>
    </div>
  );
}
