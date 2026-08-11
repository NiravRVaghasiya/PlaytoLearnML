"use client";

import { CORE_MAX_HP, WAVES, type Wave } from "./ml";

export interface WaveTrackerProps {
  wave: number;
  wavesCleared: number;
  coreHp: number;
}

/**
 * Wave progress and core health (spec: `<WaveTracker>`).
 *
 * Each wave shows its data budget, because that is the reason the right answer
 * moves: the pips are not difficulty levels, they are different amounts of
 * training data, and the player can see the famine coming.
 */
export function WaveTracker({ wave, wavesCleared, coreHp }: WaveTrackerProps) {
  const hpFraction = Math.max(0, Math.min(1, coreHp / CORE_MAX_HP));
  const critical = coreHp <= CORE_MAX_HP * 0.3;

  const stateOf = (entry: Wave) =>
    entry.index <= wavesCleared
      ? "cleared"
      : entry.index === wave
        ? "current"
        : "upcoming";

  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-sm font-medium">Generalization core</span>
        <span className="font-mono text-sm">
          {Math.round(coreHp)}
          <span className="text-text-muted"> / {CORE_MAX_HP} HP</span>
        </span>
      </div>

      <div
        role="progressbar"
        aria-valuenow={Math.round(coreHp)}
        aria-valuemin={0}
        aria-valuemax={CORE_MAX_HP}
        aria-label={`Core health: ${Math.round(coreHp)} of ${CORE_MAX_HP} hit points`}
        className="mt-1.5 h-2.5 w-full overflow-hidden rounded-full bg-surface-2"
      >
        <div
          className={`h-full rounded-full transition-[width] duration-standard ${
            critical ? "bg-wrong" : coreHp <= CORE_MAX_HP * 0.6 ? "bg-warn" : "bg-correct"
          }`}
          style={{ width: `${hpFraction * 100}%` }}
        />
      </div>

      <ol className="mt-3 flex flex-col gap-1" aria-label="Wave progress">
        {WAVES.map((entry) => {
          const state = stateOf(entry);
          return (
            <li
              key={entry.index}
              className={`flex items-center gap-2 rounded px-1.5 py-1 text-xs ${
                state === "current"
                  ? "bg-surface-2 text-text"
                  : state === "cleared"
                    ? "text-text-muted"
                    : "text-text-muted"
              }`}
            >
              <span
                aria-hidden="true"
                className={`inline-block size-2 shrink-0 rounded-full ${
                  state === "cleared"
                    ? "bg-correct"
                    : state === "current"
                      ? "bg-primary"
                      : "bg-border"
                }`}
              />
              <span className="font-medium">{entry.index}.</span>
              <span className="truncate">{entry.name}</span>
              <span className="ml-auto shrink-0 font-mono">
                {entry.trainPoints}pt
              </span>
              <span className="shrink-0 font-mono text-text-muted">
                {Math.round(entry.noiseRate * 100)}%
              </span>
              {/* Screen readers get the state, which colour alone would hide. */}
              <span className="sr-only">
                {state === "cleared"
                  ? "cleared"
                  : state === "current"
                    ? "current wave"
                    : "upcoming"}
                , {entry.trainPoints} training points,{" "}
                {Math.round(entry.noiseRate * 100)} percent label noise
              </span>
            </li>
          );
        })}
      </ol>

      <p className="mt-1.5 text-xs text-text-muted">
        Columns are training points and label noise. Both change every wave, which
        is why the right amount of defence changes with them.
      </p>
    </div>
  );
}
