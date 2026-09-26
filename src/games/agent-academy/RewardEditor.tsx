"use client";

import { Slider } from "@/components";
import { cx } from "@/lib/utils";
import {
  DEFAULT_REWARDS,
  REWARD_KNOBS,
  farmValue,
  roundTripValue,
  straightExitValue,
  type OptimalPolicy,
  type RewardConfig,
} from "./ml";

export interface RewardEditorProps {
  rewards: RewardConfig;
  optimal: OptimalPolicy;
  disabled: boolean;
  onChange: (key: keyof RewardConfig, value: number) => void;
}

const signed = (value: number) =>
  `${value >= 0 ? "+" : ""}${value.toFixed(1)}`;

/**
 * The reward editor (spec: `<RewardEditor>`).
 *
 * Five numbers. That is the player's entire vocabulary for describing the task,
 * and the point of the game is that describing a task is harder than it looks.
 *
 * The readout underneath is the part that earns its space. It shows the round-trip
 * arithmetic on the cheese — `pellet + 2 x step` — because that single expression
 * is what decides whether the agent will ever leave, and it is not obvious from
 * looking at two sliders. At the defaults it reads exactly +0.0, which is not a
 * coincidence: the game is balanced on that zero.
 *
 * The three values below it are all measured from the start, so they can be read
 * against each other: pacing by the cheese forever (the farming loop is worth the
 * same from the start as from the cheese, because the start is two moves away —
 * one off, one on, exactly like the loop), walking straight out, and the best
 * policy there is. The last one is V*(start) from value iteration: the value of
 * whatever the best policy does — the exit, the cheese loop, the pit — which is
 * exactly why it cannot stand in for the exit. It used to be labelled "Walking
 * out", and once farming won it printed the farming figure twice.
 */
export function RewardEditor({
  rewards,
  optimal,
  disabled,
  onChange,
}: RewardEditorProps) {
  const roundTrip = roundTripValue(rewards);
  const farm = farmValue(rewards);
  const walkOut = straightExitValue(rewards);
  const changed = (Object.keys(DEFAULT_REWARDS) as Array<keyof RewardConfig>)
    .filter((key) => rewards[key] !== DEFAULT_REWARDS[key])
    .length;

  return (
    <section aria-labelledby="reward-editor-heading" className="flex flex-col gap-3">
      <div className="flex items-baseline justify-between gap-2">
        <h3 id="reward-editor-heading" className="text-sm font-semibold">
          Reward function
        </h3>
        <span className="text-xs text-text-muted">
          {changed === 0
            ? "starting values"
            : `${changed} change${changed === 1 ? "" : "s"}`}
        </span>
      </div>

      {REWARD_KNOBS.map((knob) => (
        <Slider
          key={knob.key}
          label={knob.label}
          value={rewards[knob.key]}
          min={knob.min}
          max={knob.max}
          step={knob.step}
          disabled={disabled}
          format={(value) => signed(value)}
          hint={knob.blurb}
          onChange={(value) =>
            onChange(knob.key, Number(value.toFixed(2)))
          }
        />
      ))}

      <div className="rounded-md border border-border bg-surface-2 p-3">
        <h4 className="text-xs font-semibold">What that adds up to</h4>
        <dl className="mt-2 space-y-1.5 text-xs">
          <div className="flex items-baseline justify-between gap-2">
            <dt className="text-text-muted">Round trip to the cheese</dt>
            <dd
              className={cx(
                "font-mono tabular-nums",
                roundTrip > 0 ? "text-[var(--wrong)]" : "text-text",
              )}
            >
              {signed(roundTrip)}
            </dd>
          </div>
          <div className="flex items-baseline justify-between gap-2">
            <dt className="text-text-muted">Pacing there forever is worth</dt>
            <dd className="font-mono tabular-nums">{signed(farm)}</dd>
          </div>
          <div className="flex items-baseline justify-between gap-2">
            <dt className="text-text-muted">Walking straight out is worth</dt>
            <dd className="font-mono tabular-nums">{signed(walkOut)}</dd>
          </div>
          <div className="flex items-baseline justify-between gap-2">
            <dt className="text-text-muted">The best policy is worth</dt>
            <dd className="font-mono tabular-nums">
              {signed(optimal.valueAtStart)}
            </dd>
          </div>
        </dl>
        <p
          className={cx(
            "mt-2 border-t border-border pt-2 text-xs",
            optimal.reachesGoal ? "text-text-muted" : "text-[var(--wrong)]",
          )}
        >
          {optimal.reachesGoal
            ? `Best possible policy: out in ${optimal.steps} steps for ${signed(
                optimal.episodeReward,
              )}. These rewards do ask for the exit.`
            : `Best possible policy: ${
                optimal.behaviour === "farm"
                  ? `eat the cheese ${optimal.pelletsEaten} times and never leave`
                  : optimal.behaviour === "quit"
                    ? "walk into the pit"
                    : "wander until time runs out"
              }, for ${signed(
                optimal.episodeReward,
              )}. These rewards do not ask for the exit.`}
        </p>
      </div>
    </section>
  );
}
