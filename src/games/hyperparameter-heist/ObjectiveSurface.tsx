"use client";

import { useMemo } from "react";
import { scaleLinear } from "d3";
import {
  BAYESIAN_SEED_TRIALS,
  CRACK_THRESHOLD,
  DIALS,
  LEARNING_RATE,
  MOMENTUM,
  fitSurrogate,
  objectiveAt,
  type Suggestion,
  type Trial,
  type UnitPoint,
} from "./ml";

export interface ObjectiveSurfaceProps {
  trials: Trial[];
  /** Current dial positions — the other two dials fix which slice this is. */
  dials: UnitPoint;
  /** Show the true surface. Only ever true once the run is over. */
  revealed: boolean;
  /** The acquisition function's pick, when Bayesian mode is on. */
  suggestion: Suggestion | null;
}

const SIZE = 300;
const PAD = 34;
const RESOLUTION = 36;

/** Okabe-Ito blue → orange through a dark neutral. Colourblind-safe. */
function ramp(value: number): string {
  const cold = [0x00, 0x72, 0xb2];
  const mid = [0x21, 0x27, 0x34];
  const hot = [0xe6, 0x9f, 0x00];

  const t = Math.min(Math.max(value, 0), 1);
  const [from, to, local] =
    t < 0.5 ? [cold, mid, t * 2] : [mid, hot, (t - 0.5) * 2];
  const channel = (index: number) =>
    Math.round(from[index]! + (to[index]! - from[index]!) * local);
  return `rgb(${channel(0)} ${channel(1)} ${channel(2)})`;
}

/**
 * The objective surface over the two dials that matter (spec: `<ObjectiveSurface>`,
 * D3).
 *
 * ── What it shows, and when ─────────────────────────────────────────────────
 * While the run is live it draws the SURROGATE'S belief, not the truth — because
 * nobody tuning a real model can see the truth, and drawing it would hand over the
 * answer on turn one. Before there are enough readings to fit a surrogate it shows
 * nothing but the tries themselves, which is an honest picture of knowing nothing.
 *
 * When the run ends the true surface is revealed. That is the moment this view
 * earns its place: you get to see what your sixteen readings were covering, and a
 * grid's sixteen dots sitting in two vertical lines is a more convincing argument
 * than any number.
 *
 * ── Why this slice ──────────────────────────────────────────────────────────
 * Four dials cannot be drawn, so this is learning rate against momentum with the
 * other two held at wherever the player has them. Those are the two that carry the
 * surface, and the caption says so — a player who does not realise they are looking
 * at a slice will misread a marker that sits in a hot region and scored badly.
 */
export function ObjectiveSurface({
  trials,
  dials,
  revealed,
  suggestion,
}: ObjectiveSurfaceProps) {
  const surrogate = useMemo(
    () => (trials.length >= BAYESIAN_SEED_TRIALS ? fitSurrogate(trials) : null),
    [trials],
  );

  const x = useMemo(
    () => scaleLinear().domain([0, 1]).range([PAD, SIZE - 8]),
    [],
  );
  const y = useMemo(
    () => scaleLinear().domain([0, 1]).range([SIZE - PAD, 8]),
    [],
  );

  const cellW = (SIZE - 8 - PAD) / RESOLUTION;
  const cellH = (SIZE - PAD - 8) / RESOLUTION;

  const cells = useMemo(() => {
    if (!revealed && surrogate === null) return null;

    const batch = dials[2] ?? 0.5;
    const decay = dials[3] ?? 0.5;
    const out: Array<{ key: string; x: number; y: number; fill: string }> = [];

    for (let row = 0; row < RESOLUTION; row += 1) {
      for (let column = 0; column < RESOLUTION; column += 1) {
        const lr = (column + 0.5) / RESOLUTION;
        const momentum = (row + 0.5) / RESOLUTION;
        const point = [lr, momentum, batch, decay];
        const value = revealed
          ? objectiveAt(point)
          : (surrogate?.predict(point).mean ?? 0);

        out.push({
          key: `${row}-${column}`,
          x: PAD + column * cellW,
          y: SIZE - PAD - (row + 1) * cellH,
          // Stretch the interesting band: everything below a coin flip looks the
          // same and the crack threshold should sit near the top of the ramp.
          fill: ramp((value - 0.45) / (CRACK_THRESHOLD - 0.45 + 0.06)),
        });
      }
    }
    return out;
  }, [revealed, surrogate, dials, cellW, cellH]);

  const lrDial = DIALS[LEARNING_RATE]!;
  const momentumDial = DIALS[MOMENTUM]!;

  const label = revealed
    ? `The true objective surface over ${lrDial.name} and ${momentumDial.name}, with your ${trials.length} tries marked. The bright band is where the safe opens.`
    : surrogate === null
      ? `Objective surface, unknown. ${trials.length} ${
          trials.length === 1 ? "try" : "tries"
        } marked; at least ${BAYESIAN_SEED_TRIALS} are needed before a surrogate can be fitted.`
      : `The surrogate's current belief about the objective over ${lrDial.name} and ${momentumDial.name}, fitted to ${trials.length} tries. This is a model of the safe, not the safe.`;

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        className="w-full rounded-md border border-border bg-surface-2"
        role="img"
        aria-label={label}
      >
        {cells === null ? (
          <>
            <rect
              x={PAD}
              y={8}
              width={SIZE - 8 - PAD}
              height={SIZE - PAD - 8}
              fill="var(--surface-2)"
            />
            <text
              x={(SIZE + PAD) / 2}
              y={SIZE / 2 - 6}
              textAnchor="middle"
              className="fill-[var(--text-muted)] text-[10px]"
            >
              Nothing known yet
            </text>
            <text
              x={(SIZE + PAD) / 2}
              y={SIZE / 2 + 8}
              textAnchor="middle"
              className="fill-[var(--text-muted)] text-[9px]"
            >
              {BAYESIAN_SEED_TRIALS - trials.length > 0
                ? `${BAYESIAN_SEED_TRIALS - trials.length} more tries to fit a model`
                : ""}
            </text>
          </>
        ) : (
          <g shapeRendering="crispEdges">
            {cells.map((cell) => (
              <rect
                key={cell.key}
                x={cell.x}
                y={cell.y}
                width={cellW + 0.4}
                height={cellH + 0.4}
                fill={cell.fill}
              />
            ))}
          </g>
        )}

        {/* Tries, numbered so the trail can be followed. */}
        {trials.map((trial) => {
          const cracked = trial.objectiveValue >= CRACK_THRESHOLD;
          return (
            <g key={trial.index}>
              <circle
                cx={x(trial.params[LEARNING_RATE] ?? 0.5)}
                cy={y(trial.params[MOMENTUM] ?? 0.5)}
                r={cracked ? 5.5 : 4}
                fill={cracked ? "var(--correct)" : "var(--surface)"}
                stroke="var(--text)"
                strokeWidth={1}
              />
              <text
                x={x(trial.params[LEARNING_RATE] ?? 0.5)}
                y={y(trial.params[MOMENTUM] ?? 0.5) + 2.4}
                textAnchor="middle"
                className="fill-[var(--text)] text-[6px] font-semibold"
              >
                {trial.index}
              </text>
            </g>
          );
        })}

        {/* Where the acquisition function wants to go next. */}
        {suggestion ? (
          <g>
            <circle
              cx={x(suggestion.point[LEARNING_RATE] ?? 0.5)}
              cy={y(suggestion.point[MOMENTUM] ?? 0.5)}
              r={7}
              fill="none"
              stroke="var(--warn)"
              strokeWidth={2}
              strokeDasharray="3 2"
            />
          </g>
        ) : null}

        {/* Current dial position. */}
        <circle
          cx={x(dials[LEARNING_RATE] ?? 0.5)}
          cy={y(dials[MOMENTUM] ?? 0.5)}
          r={3}
          fill="var(--primary)"
          stroke="var(--bg)"
          strokeWidth={1}
        />

        <line
          x1={PAD}
          y1={SIZE - PAD}
          x2={SIZE - 8}
          y2={SIZE - PAD}
          stroke="var(--border)"
        />
        <line x1={PAD} y1={8} x2={PAD} y2={SIZE - PAD} stroke="var(--border)" />
        <text
          x={(SIZE + PAD) / 2}
          y={SIZE - 6}
          textAnchor="middle"
          className="fill-[var(--text-muted)] text-[9px]"
        >
          {lrDial.name} ({lrDial.min} → {lrDial.max}, log)
        </text>
        <text
          x={11}
          y={SIZE / 2}
          textAnchor="middle"
          transform={`rotate(-90 11 ${SIZE / 2})`}
          className="fill-[var(--text-muted)] text-[9px]"
        >
          {momentumDial.name}
        </text>
      </svg>

      <figcaption className="mt-1.5 text-xs text-text-muted">
        {revealed
          ? "The true surface. Look at how your tries are distributed across it — coverage, not luck, is what decided this run."
          : surrogate === null
            ? `A slice through ${lrDial.name} and ${momentumDial.name} at your current ${
                DIALS[2]!.name
              } and ${DIALS[3]!.name}. Nothing is drawn because nothing is known.`
            : `The surrogate's belief, not the truth — and only a slice of it, at your current ${
                DIALS[2]!.name
              } and ${DIALS[3]!.name}. A try can look well placed here and still score badly on a dial this view does not show.`}
      </figcaption>
    </figure>
  );
}
