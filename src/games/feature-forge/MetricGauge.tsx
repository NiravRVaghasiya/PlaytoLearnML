"use client";

import { TARGET_LIFT } from "./ml";

export interface MetricGaugeProps {
  baselineScore: number;
  currentScore: number;
  trainScore: number;
  ready: boolean;
  leaked: boolean;
  /** Why the baseline fit failed, when it did. */
  error?: string | null;
}

const WIDTH = 320;
const HEIGHT = 96;
const LEFT = 8;
const RIGHT = WIDTH - 8;

/** The gauge spans a band around the baseline, not 0–100%. */
const LOW = 0.5;
const HIGH = 1;
const at = (value: number) =>
  LEFT + ((Math.min(Math.max(value, LOW), HIGH) - LOW) / (HIGH - LOW)) * (RIGHT - LEFT);

/**
 * Validation accuracy against the baseline (spec: `<MetricGauge>`).
 *
 * Scaled 50–100% rather than 0–100%: the interesting range is a few points either
 * side of a baseline near 63%, and a full-width axis would render every lift as an
 * invisible twitch.
 *
 * The target marker sits at baseline + the required lift, so "how much further" is
 * a distance on screen rather than a subtraction the player has to do. When a leaky
 * feature is in the forge the whole gauge goes red — a high number is the WORST
 * outcome in that state, and a gauge that stayed green would be lying.
 */
export function MetricGauge({
  baselineScore,
  currentScore,
  trainScore,
  ready,
  leaked,
  error = null,
}: MetricGaugeProps) {
  const target = baselineScore + TARGET_LIFT;
  const lift = currentScore - baselineScore;
  const cleared = lift >= TARGET_LIFT;

  const label = !ready
    ? error !== null
      ? `The baseline could not be trained: ${error}. Press Retry to fit it again.`
      : "Establishing the baseline — training the fixed model on the raw columns."
    : `Validation accuracy ${(currentScore * 100).toFixed(
        1,
      )} percent against a baseline of ${(baselineScore * 100).toFixed(
        1,
      )} percent, a lift of ${(lift * 100).toFixed(
        1,
      )} points. Target is ${(target * 100).toFixed(1)} percent.${
        leaked ? " A leaky feature is in the forge, so this score is not real." : ""
      }`;

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="w-full rounded-md border border-border bg-surface-2"
        role="img"
        aria-label={label}
      >
        {/* Baseline: the line to beat. */}
        <line
          x1={at(baselineScore)}
          y1={10}
          x2={at(baselineScore)}
          y2={HEIGHT - 26}
          stroke="var(--text-muted)"
          strokeWidth={1}
          strokeDasharray="3 3"
        />
        <text
          x={at(baselineScore)}
          y={HEIGHT - 14}
          textAnchor="middle"
          className="fill-[var(--text-muted)] text-[8px]"
        >
          baseline
        </text>

        {/* Target: baseline plus the required lift. */}
        <line
          x1={at(target)}
          y1={10}
          x2={at(target)}
          y2={HEIGHT - 26}
          stroke="var(--primary)"
          strokeWidth={1}
        />
        <text
          x={at(target)}
          y={HEIGHT - 14}
          textAnchor="middle"
          className="fill-[var(--primary)] text-[8px]"
        >
          target
        </text>

        {/* The lift, shaded, so it reads as a distance travelled. */}
        {ready && Math.abs(lift) > 0.002 ? (
          <rect
            x={Math.min(at(baselineScore), at(currentScore))}
            y={26}
            width={Math.abs(at(currentScore) - at(baselineScore))}
            height={22}
            fill={
              leaked ? "var(--wrong)" : lift > 0 ? "var(--correct)" : "var(--wrong)"
            }
            opacity={0.2}
          />
        ) : null}

        <rect
          x={LEFT}
          y={28}
          width={Math.max(0, at(currentScore) - LEFT)}
          height={18}
          rx={3}
          fill={
            leaked
              ? "var(--wrong)"
              : cleared
                ? "var(--correct)"
                : "var(--class-a)"
          }
        />

        {/* Train accuracy, thinner, to expose memorisation. */}
        <rect
          x={LEFT}
          y={50}
          width={Math.max(0, at(trainScore) - LEFT)}
          height={8}
          rx={2}
          fill="var(--class-b)"
          opacity={0.75}
        />

        <text
          x={Math.min(at(currentScore) + 4, WIDTH - 60)}
          y={41}
          className="fill-[var(--text)] text-[10px] font-semibold"
        >
          {ready ? `${(currentScore * 100).toFixed(1)}%` : "…"}
        </text>
        <text
          x={Math.min(at(trainScore) + 4, WIDTH - 46)}
          y={58}
          className="fill-[var(--text-muted)] text-[8px]"
        >
          train {ready ? `${(trainScore * 100).toFixed(1)}%` : "…"}
        </text>
      </svg>

      <figcaption className="mt-1.5 text-xs text-text-muted">
        {leaked
          ? "This score is not real — a leaky column is in the forge."
          : !ready
            ? error !== null
              ? "The baseline could not be trained. Press Retry."
              : "Training the baseline…"
            : cleared
              ? `Cleared the target with ${(lift * 100).toFixed(1)} points of lift, from the same model.`
              : `${(lift * 100).toFixed(1)} points of lift. The thin bar is training accuracy — when it pulls away, the model is memorising.`}
      </figcaption>
    </figure>
  );
}
