"use client";

import { CRACK_THRESHOLD, DIALS, temperatureOf, toRealValue, type Trial } from "./ml";

export interface TrajectoryTrailProps {
  trials: Trial[];
  budget: number;
}

/**
 * Every try, in order (spec: `<TrajectoryTrail>`).
 *
 * A real `<table>`, not a decorative trail. The surface plot beside this shows
 * WHERE the tries went in two of the four dimensions; this shows all four, in
 * order, with what each one scored — which is the only complete record and the only
 * one a screen reader can read.
 *
 * Sorted by try order rather than by score on purpose. The sequence is the thing
 * being taught: a Bayesian run and a random run with the same best score look
 * completely different read top to bottom.
 */
export function TrajectoryTrail({ trials, budget }: TrajectoryTrailProps) {
  if (trials.length === 0) {
    return (
      <p className="text-xs text-text-muted">
        No tries yet. Every try you spend appears here with its reading, so you can
        see what you have covered rather than only what you scored.
      </p>
    );
  }

  const best = trials.reduce((top, trial) =>
    trial.objectiveValue > top.objectiveValue ? trial : top,
  );

  return (
    <div className="max-h-[240px] overflow-auto">
      <table className="w-full border-collapse text-left text-xs">
        <caption className="mb-1 text-left text-[11px] text-text-muted">
          {trials.length} of {budget} tries spent. Best {(
            best.objectiveValue * 100
          ).toFixed(1)}% on try {best.index}.
        </caption>
        <thead className="sticky top-0 bg-surface">
          <tr className="border-b border-border">
            <th scope="col" className="py-1 font-medium">
              #
            </th>
            {DIALS.map((dial) => (
              <th key={dial.name} scope="col" className="py-1 text-right font-medium">
                {dial.short}
              </th>
            ))}
            <th scope="col" className="py-1 text-right font-medium">
              score
            </th>
          </tr>
        </thead>
        <tbody>
          {trials.map((trial) => {
            const isBest = trial.index === best.index;
            const cracked = trial.objectiveValue >= CRACK_THRESHOLD;
            return (
              <tr
                key={trial.index}
                className={`border-b border-border/50 ${
                  cracked
                    ? "bg-correct/10"
                    : isBest
                      ? "bg-primary/10"
                      : ""
                }`}
              >
                <th scope="row" className="py-1 font-mono font-normal">
                  {trial.index}
                </th>
                {DIALS.map((dial, index) => (
                  <td key={dial.name} className="py-1 text-right font-mono">
                    {dial.format(toRealValue(dial, trial.params[index] ?? 0.5))}
                  </td>
                ))}
                <td className="py-1 text-right font-mono">
                  {(trial.objectiveValue * 100).toFixed(1)}%
                  {/* Temperature in words: colour alone would not carry it. */}
                  <span className="sr-only">
                    {" "}
                    — {temperatureOf(trial.objectiveValue)}
                    {isBest ? ", best so far" : ""}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
