"use client";

import { useMemo } from "react";
import { EPISODE_BATCH, type EpisodeRecord } from "./ml";

export interface RewardCurveProps {
  history: readonly EpisodeRecord[];
  /** Reward per episode the optimal policy would collect, as a target line. */
  optimalReward: number;
}

const WIDTH = 480;
const HEIGHT = 150;
const PAD_L = 34;
const PAD_R = 8;
const PAD_T = 10;
const PAD_B = 20;

/** Mean of every `size` consecutive episodes. */
function smooth(values: readonly number[], size: number): number[] {
  const out: number[] = [];
  for (let index = 0; index < values.length; index += 1) {
    const from = Math.max(0, index - size + 1);
    let total = 0;
    for (let k = from; k <= index; k += 1) total += values[k]!;
    out.push(total / (index - from + 1));
  }
  return out;
}

/**
 * Episode-by-episode reward (spec: `<RewardCurve>`).
 *
 * Two series, and the second one is the one that teaches. The faint line is raw
 * per-episode reward, which under epsilon-greedy exploration is genuinely noisy —
 * a run that took three random turns into a wall scores worse than the same policy
 * on a luckier day. The solid line is a 25-episode running mean, which is what
 * "is it learning?" actually looks like.
 *
 * The dashed target is what the player's OWN reward function pays for the best
 * possible policy. That matters more than a fixed target would: when someone
 * cranks the cheese up, this line rises with it, and the curve then climbs
 * beautifully toward a target that has nothing to do with finishing the maze.
 * A reward curve going up is not evidence that the right thing is happening, and
 * this chart is arranged so that lesson is visible rather than asserted.
 */
export function RewardCurve({ history, optimalReward }: RewardCurveProps) {
  const { rawPath, meanPath, low, high, span } = useMemo(() => {
    if (history.length === 0) {
      return { rawPath: "", meanPath: "", low: 0, high: 1, span: 1 };
    }

    const rewards = history.map((record) => record.reward);
    const means = smooth(rewards, 25);

    const lowest = Math.min(...rewards, optimalReward, 0);
    const highest = Math.max(...rewards, optimalReward, 0);
    const range = Math.max(highest - lowest, 1e-6);

    const px = (index: number) =>
      PAD_L +
      (history.length <= 1
        ? 0
        : (index / (history.length - 1)) * (WIDTH - PAD_L - PAD_R));
    const py = (value: number) =>
      PAD_T + (1 - (value - lowest) / range) * (HEIGHT - PAD_T - PAD_B);

    const toPath = (series: number[]) =>
      series
        .map((value, index) => `${index === 0 ? "M" : "L"}${px(index)},${py(value)}`)
        .join(" ");

    return {
      rawPath: toPath(rewards),
      meanPath: toPath(means),
      low: lowest,
      high: highest,
      span: range,
    };
  }, [history, optimalReward]);

  const py = (value: number) =>
    PAD_T + (1 - (value - low) / span) * (HEIGHT - PAD_T - PAD_B);

  const last = history.at(-1);
  const recent = history.slice(-EPISODE_BATCH);
  const recentMean =
    recent.length > 0
      ? recent.reduce((total, record) => total + record.reward, 0) / recent.length
      : Number.NaN;

  const description =
    history.length === 0
      ? "No episodes yet, so there is no reward curve."
      : `Reward per episode over ${history.length} episodes. Last episode ${last!.reward.toFixed(
          1,
        )}. Mean of the last ${recent.length}: ${recentMean.toFixed(
          1,
        )}. The best possible policy for these rewards would score ${optimalReward.toFixed(
          1,
        )} per episode.`;

  return (
    <figure className="m-0">
      <figcaption className="mb-1 flex flex-wrap items-baseline justify-between gap-2 text-xs">
        <span className="font-medium">Episode reward</span>
        <span className="text-text-muted">
          {history.length === 0
            ? "nothing trained yet"
            : `last ${recent.length}: ${recentMean.toFixed(
                1,
              )} · best possible ${optimalReward.toFixed(1)}`}
        </span>
      </figcaption>

      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="w-full rounded-md border border-border bg-surface-2"
        role="img"
        aria-label={description}
      >
        <line
          x1={PAD_L}
          y1={PAD_T}
          x2={PAD_L}
          y2={HEIGHT - PAD_B}
          stroke="var(--border)"
        />
        <line
          x1={PAD_L}
          y1={HEIGHT - PAD_B}
          x2={WIDTH - PAD_R}
          y2={HEIGHT - PAD_B}
          stroke="var(--border)"
        />

        {history.length > 0 ? (
          <>
            {/* Zero, so "is it losing money?" is answerable at a glance. */}
            {low < 0 && high > 0 ? (
              <line
                x1={PAD_L}
                y1={py(0)}
                x2={WIDTH - PAD_R}
                y2={py(0)}
                stroke="var(--border)"
                strokeDasharray="2 3"
              />
            ) : null}

            <line
              x1={PAD_L}
              y1={py(optimalReward)}
              x2={WIDTH - PAD_R}
              y2={py(optimalReward)}
              stroke="var(--warn)"
              strokeWidth={1.5}
              strokeDasharray="5 4"
            />

            <path
              d={rawPath}
              fill="none"
              stroke="var(--text-muted)"
              strokeWidth={0.75}
              opacity={0.45}
            />
            <path
              d={meanPath}
              fill="none"
              stroke="var(--primary)"
              strokeWidth={2}
              strokeLinejoin="round"
            />

            <text
              x={PAD_L - 4}
              y={py(high) + 3}
              textAnchor="end"
              className="fill-[var(--text-muted)] text-[8px] tabular-nums"
            >
              {high.toFixed(0)}
            </text>
            <text
              x={PAD_L - 4}
              y={py(low) + 3}
              textAnchor="end"
              className="fill-[var(--text-muted)] text-[8px] tabular-nums"
            >
              {low.toFixed(0)}
            </text>
            <text
              x={WIDTH - PAD_R}
              y={HEIGHT - 6}
              textAnchor="end"
              className="fill-[var(--text-muted)] text-[8px] tabular-nums"
            >
              episode {history.length}
            </text>
          </>
        ) : (
          <text
            x={WIDTH / 2}
            y={HEIGHT / 2}
            textAnchor="middle"
            className="fill-[var(--text-muted)] text-[10px]"
          >
            train the agent to see its reward per episode
          </text>
        )}
      </svg>

      {history.length > 0 ? (
        <p className="mt-1 text-xs text-text-muted">
          <span
            aria-hidden="true"
            className="mr-1 inline-block h-0.5 w-4 bg-[var(--primary)] align-middle"
          />
          25-episode mean ·
          <span
            aria-hidden="true"
            className="mx-1 inline-block h-0.5 w-4 border-t-2 border-dashed border-[var(--warn)] align-middle"
          />
          best possible under YOUR rewards — note that this line moves when you
          change them
        </p>
      ) : null}
    </figure>
  );
}
