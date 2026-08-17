"use client";

import { useMemo } from "react";
import { Button } from "@/components";
import { PELLET, START, stateOf } from "./ml";
import { cellValues, optimalTrail, useAcademyStore } from "./store";
import { GridWorld, HeatLegend } from "./GridWorld";
import { QValueHeatmap } from "./QValueHeatmap";
import { RewardCurve } from "./RewardCurve";
import { trailCaption } from "./why-cards";

/**
 * The visual lane.
 *
 * Selectors take stable slices only — never a freshly built object. Returning
 * `{ ...something }` from a zustand selector breaks `useSyncExternalStore`'s
 * cached-snapshot check and re-renders forever; the derived shapes below are all
 * built in `useMemo` from primitives and stable references instead.
 */
export function VisualLane() {
  const qTable = useAcademyStore((s) => s.qTable);
  const rewards = useAcademyStore((s) => s.rewards);
  const optimal = useAcademyStore((s) => s.optimal);
  const history = useAcademyStore((s) => s.history);
  const trail = useAcademyStore((s) => s.trail);
  const report = useAcademyStore((s) => s.report);
  const episodesUsed = useAcademyStore((s) => s.episodesUsed);
  const showHeatmap = useAcademyStore((s) => s.showHeatmap);
  const showOptimal = useAcademyStore((s) => s.showOptimal);
  const toggleHeatmap = useAcademyStore((s) => s.toggleHeatmap);
  const toggleOptimal = useAcademyStore((s) => s.toggleOptimal);

  const values = useMemo(
    () => (showHeatmap ? cellValues(qTable) : null),
    [showHeatmap, qTable],
  );

  const wanted = useMemo(
    () => (showOptimal ? optimalTrail(optimal, rewards) : null),
    [showOptimal, optimal, rewards],
  );

  /**
   * Which cell to open up in the Q-value table.
   *
   * The start when the agent is still stuck at home, the cheese once it has
   * learned to sit there — because that is where the interesting four numbers are.
   * Following the evidence rather than pinning it to one square means the panel is
   * showing the decision that explains what is on screen.
   */
  const focus = useMemo(() => {
    if (report.meanPellets >= 2) {
      return { state: stateOf(PELLET), label: "the cheese" };
    }
    return { state: stateOf(START), label: "the start" };
  }, [report.meanPellets]);

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start">
        <div className="flex-1">
          <GridWorld
            trail={trail}
            optimalTrail={wanted}
            values={values}
            episodesUsed={episodesUsed}
          />
        </div>

        <div className="flex flex-col gap-3 lg:w-64">
          <fieldset className="flex flex-col gap-2">
            <legend className="text-xs font-semibold">Overlays</legend>
            <Button
              variant={showHeatmap ? "primary" : "ghost"}
              size="sm"
              aria-pressed={showHeatmap}
              onClick={toggleHeatmap}
            >
              Learned values
            </Button>
            <Button
              variant={showOptimal ? "primary" : "ghost"}
              size="sm"
              aria-pressed={showOptimal}
              onClick={toggleOptimal}
            >
              What your rewards ask for
            </Button>
          </fieldset>

          {showHeatmap ? <HeatLegend /> : null}

          <p className="text-xs text-text-muted">
            {trailCaption(episodesUsed, report)}
          </p>
        </div>
      </div>

      <RewardCurve history={history} optimalReward={optimal.episodeReward} />

      <div className="border-t border-border pt-4">
        <QValueHeatmap
          qTable={qTable}
          state={focus.state}
          label={focus.label}
          episodesUsed={episodesUsed}
        />
      </div>
    </div>
  );
}
