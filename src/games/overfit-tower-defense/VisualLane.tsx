"use client";

import { Battlefield } from "./Battlefield";
import { GapMeter } from "./GapMeter";
import {
  biasOf,
  currentWave,
  gapOf,
  useTowerDefenseStore,
} from "./store";

/**
 * Overfit Tower Defense — the visual lane.
 *
 * Reads the same store the code lane writes. The gap meter is the hero: the
 * battlefield dramatises the two errors, but the meter is where you can see that
 * they move in opposite directions.
 */
export function VisualLane() {
  const wave = useTowerDefenseStore(currentWave);
  const gap = useTowerDefenseStore(gapOf);
  const bias = useTowerDefenseStore(biasOf);
  const coreHp = useTowerDefenseStore((s) => s.coreHp);
  const phase = useTowerDefenseStore((s) => s.phase);
  const result = useTowerDefenseStore((s) => s.lastResult);
  const trainAccuracy = useTowerDefenseStore((s) => s.trainAccuracy);
  const validationAccuracy = useTowerDefenseStore((s) => s.validationAccuracy);
  const achievable = useTowerDefenseStore((s) => s.dataset.achievable);
  const trainPoints = useTowerDefenseStore((s) => s.dataset.train.length);

  return (
    <div className="flex flex-col gap-4">
      <section aria-labelledby="gap-heading">
        {/* h2: the lane sits directly under the page h1. */}
        <h2
          id="gap-heading"
          className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1"
        >
          <span className="font-display text-lg font-semibold">
            Train vs validation
          </span>
          <span className="text-xs font-normal text-text-muted">
            {trainPoints} training points · ceiling{" "}
            {Math.round(achievable * 100)}%
          </span>
        </h2>
        <GapMeter
          trainAccuracy={trainAccuracy}
          validationAccuracy={validationAccuracy}
          achievable={achievable}
        />
      </section>

      <section aria-labelledby="battlefield-heading">
        <h2
          id="battlefield-heading"
          className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1"
        >
          <span className="font-display text-sm font-semibold">
            Wave {wave.index}: {wave.name}
          </span>
          <span className="text-xs font-normal text-text-muted">
            {wave.overfitEnemies} overfit · {wave.underfitEnemies} underfit
          </span>
        </h2>
        <Battlefield
          wave={wave}
          gap={gap}
          bias={bias}
          coreHp={coreHp}
          result={result}
          advancing={phase === "resolved" || phase === "lost" || phase === "won"}
        />
        <p className="mt-1.5 text-xs text-text-muted">{wave.briefing}</p>
      </section>
    </div>
  );
}
