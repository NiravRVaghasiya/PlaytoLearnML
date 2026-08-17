"use client";

import { Slider } from "@/components";
import { LOW_EPSILON } from "./ml";

export interface EpsilonSliderProps {
  epsilon: number;
  disabled: boolean;
  episodesUsed: number;
  onChange: (epsilon: number) => void;
}

/**
 * The exploration control (spec: `<EpsilonSlider>`).
 *
 * Split out from the reward editor on purpose, and it is not just layout. The
 * reward function is a statement about what you WANT; epsilon is a statement about
 * how the agent should SEARCH. Those are different kinds of decision with
 * different failure modes, and the game's whole diagnostic distinguishes them, so
 * putting them in one undifferentiated stack of six sliders would work against
 * the lesson.
 *
 * The hint spells out what the number means operationally rather than
 * mathematically — "one move in five is a coin flip" is the fact that explains the
 * behaviour on screen.
 */
export function EpsilonSlider({
  epsilon,
  disabled,
  episodesUsed,
  onChange,
}: EpsilonSliderProps) {
  const randomMoves = Math.round(epsilon * 100);

  return (
    <section aria-labelledby="epsilon-heading" className="flex flex-col gap-2">
      <h3 id="epsilon-heading" className="text-sm font-semibold">
        Exploration
      </h3>

      <Slider
        label="ε — chance of a random move"
        value={epsilon}
        min={0}
        max={1}
        step={0.05}
        disabled={disabled}
        format={(value) => value.toFixed(2)}
        hint={
          epsilon === 0
            ? "Pure exploitation. It will only ever take the action it already rates highest, so a value it has not visited can never change."
            : `${randomMoves} moves in every 100 are chosen at random; the other ${
                100 - randomMoves
              } follow what the table currently believes.`
        }
        onChange={(value) => onChange(Number(value.toFixed(2)))}
      />

      {epsilon <= LOW_EPSILON && episodesUsed > 0 ? (
        <p className="text-xs text-[var(--warn)]">
          The exit is nine steps from the start. At ε {epsilon.toFixed(2)} the
          agent is unlikely to string nine unfamiliar moves together before
          something nearer distracts it.
        </p>
      ) : null}
    </section>
  );
}
