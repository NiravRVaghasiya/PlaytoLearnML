import type { WhyCardContent } from "@/components";
import {
  GLOBAL_MINIMUM,
  LOCAL_MINIMA,
  STEP_BUDGET,
  WIN_SCORE,
  gradientNorm,
  nearestMinimum,
  type Evaluation,
  type Vec2,
} from "./ml";

/**
 * "Why did that happen?" copy for Gradient Descent Skier.
 *
 * Keyed to the player's last action (spec §4). Every number comes from the real
 * gradient computation in `ml.ts`.
 *
 * The one thing this copy must never do is describe the learning rate as "speed".
 * It is a multiplier on the slope, which is why the same dial is safe on a gentle
 * incline and explosive on a steep one — and that distinction is the lesson.
 */

export type SkierEvent =
  | { kind: "reset" }
  | {
      kind: "rate-changed";
      learningRate: number;
      previous: number;
      gradient: Vec2;
    }
  | { kind: "momentum-changed"; momentum: number; previous: number }
  | {
      kind: "stepped";
      loss: number;
      previousLoss: number;
      gradient: Vec2;
      overshot: boolean;
      diverged: boolean;
      settled: boolean;
      pos: Vec2;
      learningRate: number;
      momentum: number;
    }
  | { kind: "checked"; evaluation: Evaluation };

const GD_HREF = "/concepts/gradient-descent";
const LR_HREF = "/concepts/learning-rate";

const n2 = (value: number) => value.toFixed(2);
const n3 = (value: number) => value.toPrecision(3);
const percent = (value: number) => `${Math.round(value * 100)}%`;

export function whyCardFor(event: SkierEvent): WhyCardContent {
  switch (event.kind) {
    case "reset":
      return {
        key: "reset",
        title: "One dial, one decision",
        body: `Each step moves you downhill by the slope times your learning rate. Too small and you run out of steps; too big and you overshoot the valley and fly off. The valley you can see from here is not the deepest one on the mountain.`,
        tone: "info",
        conceptHref: GD_HREF,
        conceptLabel: "How gradient descent works",
      };

    case "rate-changed": {
      const { learningRate, previous, gradient } = event;
      const slope = gradientNorm(gradient);
      const stepSize = slope * learningRate;

      return {
        key: `rate-${learningRate.toFixed(5)}`,
        title: `Learning rate ${n3(learningRate)}`,
        body: `The slope here is ${n2(
          slope,
        )}, so your next step will move about ${n2(
          stepSize,
        )} across the surface — the rate is a multiplier on the slope, not a speed. ${
          learningRate > previous
            ? "Bigger steps cover ground faster, and overshoot harder where the surface is steep."
            : "Smaller steps are safer everywhere and slower everywhere."
        }`,
        tone: "info",
        conceptHref: LR_HREF,
        conceptLabel: "Why learning rate dominates",
      };
    }

    case "momentum-changed": {
      const { momentum, previous } = event;
      if (momentum === 0) {
        return {
          key: "momentum-0",
          title: "Momentum off",
          body: `Each step now depends only on the slope where you stand. Where the slope is zero you stop — even if a deeper valley is just over the ridge.`,
          tone: "info",
          conceptHref: GD_HREF,
        };
      }
      return {
        key: `momentum-${momentum.toFixed(3)}`,
        title: `Momentum ${momentum.toFixed(2)}`,
        body: `You now keep ${percent(
          momentum,
        )} of your previous step and add the new one to it. Speed accumulates downhill, which is what carries you across a ridge the gradient alone would stop at${
          momentum > previous ? " — and past the valley floor if you overdo it." : "."
        }`,
        tone: momentum > 0.9 ? "warn" : "info",
        conceptHref: GD_HREF,
        conceptLabel: "What momentum does",
      };
    }

    case "stepped": {
      const {
        loss,
        previousLoss,
        gradient,
        overshot,
        diverged,
        settled,
        pos,
        momentum,
      } = event;
      const delta = loss - previousLoss;
      const key = `step-${loss.toFixed(5)}-${pos.x.toFixed(4)}`;

      if (diverged) {
        return {
          key,
          title: "Off the mountain",
          body: `Loss jumped from ${n2(previousLoss)} to ${
            Number.isFinite(loss) ? n2(loss) : "infinity"
          }. The step was longer than the slope it was measured on, so it landed somewhere steeper — where the next step is longer still. That runaway is divergence.`,
          tone: "bad",
          conceptHref: LR_HREF,
        };
      }

      if (settled) {
        const nearest = nearestMinimum(pos);
        const isGlobal = nearest.minimum.kind === "global";
        return {
          key,
          title: isGlobal ? "Bottom of the mountain" : "Stopped in a dip",
          body: isGlobal
            ? `The slope is ${n2(
                gradientNorm(gradient),
              )} — flat — and you've stopped moving. Loss ${n2(
                loss,
              )}, which is the deepest point on this surface.`
            : `The slope is ${n2(
                gradientNorm(gradient),
              )} here, so gradient descent has nothing left to follow. But loss ${n2(
                loss,
              )} against ${n2(
                GLOBAL_MINIMUM.loss,
              )} at the real bottom means this is the shallow valley. ${
                momentum > 0
                  ? "More momentum would carry you over the ridge."
                  : "Momentum would carry you over the ridge."
              }`,
          tone: isGlobal ? "good" : "warn",
          conceptHref: GD_HREF,
        };
      }

      if (overshot) {
        return {
          key,
          title: `Overshot — loss up ${n2(delta)}`,
          body: `You stepped past the valley floor and landed on the far slope, so the loss went UP. One overshoot is recoverable; a pattern of them means the step size is too big for how sharply this surface curves.`,
          tone: "warn",
          conceptHref: LR_HREF,
        };
      }

      return {
        key,
        title: `Loss ${n2(loss)} (${delta <= 0 ? "−" : "+"}${n2(Math.abs(delta))})`,
        body: `Slope ${n2(
          gradientNorm(gradient),
        )} at your feet, and you moved downhill along it. That is the entire algorithm: measure the slope, step against it, repeat.`,
        tone: "good",
      };
    }

    case "checked": {
      const {
        outcome,
        finalLoss,
        globalLoss,
        steps,
        score,
        progress,
        distanceToGlobal,
        overshoots,
      } = event.evaluation;
      const key = `checked-${outcome}-${score.toFixed(4)}`;

      switch (outcome) {
        case "diverged":
          return {
            key,
            title: "Divergence",
            body: `Gone in ${steps} steps. Divergence isn't the surface being unfair — it's the step size exceeding what the curvature tolerates. Below roughly 0.4 here the same descent is stable; above it, every step lands somewhere steeper than the last.`,
            tone: "bad",
            conceptHref: LR_HREF,
            conceptLabel: "Why too-large rates explode",
          };

        case "local-minimum":
          return {
            key,
            title: "Local minimum",
            body: `Settled at loss ${n2(finalLoss)} in the shallow valley; the real bottom is ${n2(
              globalLoss,
            )}, ${n2(
              distanceToGlobal,
            )} away over a ridge. The gradient here is genuinely zero, so plain descent is finished — this is exactly the trap momentum exists for.`,
            tone: "bad",
            conceptHref: GD_HREF,
            conceptLabel: "Local versus global minima",
          };

        case "oscillating":
          return {
            key,
            title: "Oscillation",
            body: `${steps} steps, ${overshoots} of them uphill, and never came to rest at loss ${n2(
              finalLoss,
            )}. This is the band just below divergence: big enough to cross the valley every step, not big enough to escape it. Halving the rate usually settles it.`,
            tone: "bad",
            conceptHref: LR_HREF,
          };

        case "slow-convergence":
          return {
            key,
            title: "Slow convergence",
            body: `Every step went downhill and you still only covered ${percent(
              progress,
            )} of the descent in ${steps} steps, ${n2(
              distanceToGlobal,
            )} short. Nothing is broken — the steps are just too small. This is the failure people mistake for a bad model.`,
            tone: "bad",
            conceptHref: LR_HREF,
          };

        case "near-miss":
          return {
            key,
            title: `Reached the bottom — ${percent(score)}`,
            body: `Loss ${n2(finalLoss)} at the deepest point, but it took ${steps} of ${STEP_BUDGET} steps. Getting there matters more than getting there fast, so this counts — but a better rate, or some momentum, arrives in a fraction of the steps. Needs ${percent(
              WIN_SCORE,
            )}.`,
            tone: "warn",
            conceptHref: LR_HREF,
          };

        case "win":
          return {
            key,
            title: `Global minimum — ${percent(score)}`,
            body: `Loss ${n2(finalLoss)} in ${steps} steps, ${
              LOCAL_MINIMA.length > 0
                ? "past the shallow valley and "
                : ""
            }all the way to the deepest point on the surface. Big enough steps to make progress, small enough not to overshoot, and enough momentum to clear the ridge.`,
            tone: "good",
            conceptHref: GD_HREF,
          };

        case "running":
        default:
          return {
            key,
            title: "Still descending",
            body: `Loss ${n2(finalLoss)} after ${steps} steps, ${n2(
              distanceToGlobal,
            )} from the bottom. Keep stepping — or run it out and see where this rate lands.`,
            tone: "info",
          };
      }
    }
  }
}
