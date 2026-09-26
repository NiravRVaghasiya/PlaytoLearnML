import type { WhyCardContent } from "@/components";
import {
  GLOBAL_MINIMUM,
  LOCAL_MINIMA,
  RIDGE_X,
  STABLE_RATE,
  STEP_BUDGET,
  WIN_SCORE,
  gradientNorm,
  nearestMinimum,
  nextStepLength,
  type Evaluation,
  type Skier,
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
 *
 * The other is to talk as if momentum were always off. Momentum changes what
 * a step is (it carries the last one forward), so every claim about step length,
 * ridges and overshoot here checks β and the path before it is made.
 */

export type SkierEvent =
  | { kind: "reset" }
  | {
      kind: "rate-changed";
      learningRate: number;
      previous: number;
      /** Gradient where the skier stands now. */
      gradient: Vec2;
      /** The skier with the new rate, velocity and momentum included. */
      skier: Skier;
      /** The run has ended, so there is no next step to preview. */
      finished: boolean;
    }
  | { kind: "momentum-changed"; momentum: number; previous: number }
  | {
      kind: "stepped";
      loss: number;
      previousLoss: number;
      /** Gradient where the step landed — "at your feet". */
      gradient: Vec2;
      overshot: boolean;
      diverged: boolean;
      settled: boolean;
      pos: Vec2;
      learningRate: number;
      momentum: number;
      /** Furthest left the path has been this run. */
      minX: number;
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
      const { learningRate, previous, gradient, skier, finished } = event;
      const slope = gradientNorm(gradient);
      // The real next step, momentum included — not slope × rate, which is
      // only the step when the skier has no velocity to carry.
      const next = nextStepLength(skier);
      const carrying =
        skier.momentum > 0 && Math.hypot(skier.velocity.x, skier.velocity.y) > 0;

      return {
        key: `rate-${learningRate.toFixed(5)}${finished ? "-after" : ""}`,
        title: `Learning rate ${n3(learningRate)}`,
        body: finished
          ? `This run is over, so there is no next step — the new rate applies from the top of the mountain. Press Back to the top to try it. The rate is a multiplier on the slope, not a speed.`
          : `The slope here is ${n2(slope)}, so your next step will move about ${n2(
              next,
            )} across the surface${
              carrying
                ? ` — rate × slope, plus ${percent(skier.momentum)} of your last step carried forward`
                : ""
            }. The rate is a multiplier on the slope, not a speed. ${
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
        minX,
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
        const why =
          momentum === 0
            ? minX < RIDGE_X
              ? `Your plain step did hop the ridge (as far as x ${n2(
                  minX,
                )}) and bounced back: that step was too big to stop in the deep valley, not too small to reach it.`
              : "Momentum — or a plain step big enough to hop the shallow basin — can carry you over the ridge."
            : minX < RIDGE_X
              ? `You did cross the ridge (as far as x ${n2(
                  minX,
                )}) and rolled back: that's too much momentum, not too little.`
              : `Momentum ${momentum.toFixed(2)} didn't build enough speed to clear the ridge at x ${n2(
                  RIDGE_X,
                )}.`;
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
              )} at the real bottom means this is the shallow valley. ${why}`,
          tone: isGlobal ? "good" : "warn",
          conceptHref: GD_HREF,
        };
      }

      if (overshot) {
        return {
          key,
          title: `Overshot — loss up ${n2(delta)}`,
          body: `You stepped past the valley floor and landed on the far slope, so the loss went UP. One overshoot is recoverable; a pattern of them means the step${
            momentum > 0 ? ", momentum included," : ""
          } is too big for how sharply this surface curves.`,
          tone: "warn",
          conceptHref: LR_HREF,
        };
      }

      return {
        key,
        title: `Loss ${n2(loss)} (${delta <= 0 ? "−" : "+"}${n2(Math.abs(delta))})`,
        body: `You moved downhill, and the slope at your feet is now ${n2(
          gradientNorm(gradient),
        )}. That is the entire algorithm: measure the slope, step against it, repeat.`,
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
        reachedGlobal,
        rises,
        learningRate,
        momentum,
        minX,
        crossedRidge,
        passedDeepValley,
        advice,
        withoutMomentum,
      } = event.evaluation;
      const key = `checked-${outcome}-${score.toFixed(4)}`;
      const then = advice ? ` ${advice}` : "";

      switch (outcome) {
        case "diverged":
          return {
            key,
            title: "Divergence",
            body:
              momentum > 0
                ? `Gone in ${steps} step${steps === 1 ? "" : "s"}. With momentum ${momentum.toFixed(
                    2,
                  )} the push accumulates, so each step heads toward α/(1−β) ≈ ${n3(
                    learningRate / (1 - momentum),
                  )} times the slope — that is the step size the curvature had to tolerate, not ${n3(
                    learningRate,
                  )} alone, and it didn't.${then}`
                : `Gone in ${steps} step${steps === 1 ? "" : "s"}. Divergence isn't the surface being unfair — it's the step size exceeding what the curvature tolerates. With no momentum, even the deep valley only holds a rate below 2 ÷ its curvature ≈ ${n2(
                    STABLE_RATE,
                  )}; at ${n3(learningRate)} every step landed somewhere steeper than the last.${then}`,
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
            )}, ${n2(distanceToGlobal)} away over a ridge. ${
              momentum === 0
                ? crossedRidge
                  ? `The gradient here is genuinely zero, but this plain step was big enough to hop the ridge — as far as x ${n2(
                      minX,
                    )}${passedDeepValley ? ", past the deep valley's floor" : ""} — and too big to stop there, so it bounced back. Too big a step, not too small.`
                  : "The gradient here is genuinely zero, so plain descent is finished — this is the trap momentum exists for, though a plain step big enough to hop the basin gets out too."
                : crossedRidge
                  ? `Momentum ${momentum.toFixed(2)} did carry you over the ridge — as far as x ${n2(
                      minX,
                    )}${passedDeepValley ? ", past the deep valley's floor" : ""} — and then back again. Too much momentum, not too little.`
                  : `Momentum ${momentum.toFixed(2)} wasn't enough to carry you over the ridge.`
            }${then}`,
            tone: "bad",
            conceptHref: GD_HREF,
            conceptLabel: "Local versus global minima",
          };

        case "oscillating":
          return {
            key,
            title: "Oscillation",
            body: `${steps} steps, ${rises} of them uphill, and never came to rest at loss ${n2(
              finalLoss,
            )}. ${
              momentum > 0
                ? `Rate and momentum ${momentum.toFixed(2)} together carry each step past ${
                    !reachedGlobal && !crossedRidge
                      ? "the shallow valley's floor"
                      : "the floor"
                  } and back — too much speed is its own kind of overshoot.`
                : "Each step is big enough to cross the valley, not big enough to escape it."
            }${then}`,
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
            body: `Loss ${n2(finalLoss)} at the deepest point, but it took ${steps} of ${STEP_BUDGET} steps. Getting there matters more than getting there fast, so this counts as a clear — but a better rate, or some momentum, arrives in far fewer steps. Needs ${percent(
              WIN_SCORE,
            )} for the win.`,
            tone: "warn",
            conceptHref: LR_HREF,
          };

        case "win":
          return {
            key,
            title: `Global minimum — ${percent(score)}`,
            body: `Loss ${n2(finalLoss)} in ${steps} steps, ${
              LOCAL_MINIMA.length > 0 ? "past the shallow valley and " : ""
            }all the way to the deepest point on the surface. ${
              momentum === 0
                ? "No momentum at all: steps big enough to hop the shallow valley, and small enough not to overshoot the deep one."
                : withoutMomentum === "local-minimum"
                  ? `Big enough steps to make progress, small enough not to overshoot, and momentum ${momentum.toFixed(
                      2,
                    )} to clear the ridge — with none, this same rate stops in the shallow valley.`
                  : withoutMomentum === "win" || withoutMomentum === "near-miss"
                    ? `This rate clears the ridge even with no momentum; momentum ${momentum.toFixed(
                        2,
                      )} changed the route, not whether you got here.`
                    : `Big enough steps to make progress, small enough not to overshoot, with momentum ${momentum.toFixed(
                        2,
                      )} in the mix.`
            }`,
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
