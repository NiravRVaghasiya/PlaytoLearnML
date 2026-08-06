"use client";

import { create } from "zustand";
import { useProgression } from "@/engine/progression";
import type { Lane, NamedFailure } from "@/engine/types";
import type { WhyCardContent } from "@/components";
import { clamp } from "@/lib/utils";
import {
  DEFAULT_LEARNING_RATE,
  MAX_LEARNING_RATE,
  MAX_MOMENTUM,
  MIN_LEARNING_RATE,
  OVERSHOOT_DELTA,
  SETTLED_GRADIENT,
  START,
  START_LOSS,
  STEP_BUDGET,
  evaluate,
  gradientAt,
  gradientNorm,
  lossAt,
  makeSkier,
  step as gradientStep,
  type Evaluation,
  type Skier,
  type Vec2,
} from "./ml";
import { whyCardFor } from "./why-cards";

export const SLUG = "gradient-descent-skier";

/**
 * Gradient Descent Skier state.
 *
 * The spec's data model is:
 *
 *   Surface   { fn(x,y) → loss, minima[] }
 *   Skier     { pos, velocity, learningRate, momentum }
 *   GameState { surface, skier, currentLoss, timer }
 *
 * The surface is a module constant in `ml.ts` rather than state — it never
 * changes, and `MINIMA` is computed once from it. `timer` is a STEP BUDGET rather
 * than a clock: see the docblock on `stepsRemaining`.
 */

export interface GradientSkierState {
  skier: Skier;
  currentLoss: number;
  /** Last computed gradient, for the readout. */
  gradient: Vec2;
  /** Every position visited (spec: `<PathTrail>`). */
  trail: Vec2[];
  /** Loss at each visited position, for the sparkline and overshoot detection. */
  lossHistory: number[];

  stepsTaken: number;
  /**
   * Steps left, not seconds left.
   *
   * The spec asks for a timer. An iteration budget is used instead: it is what a
   * real optimizer is actually limited by, "too small a learning rate never
   * reaches the bottom in time" survives the change intact, and a wall clock would
   * turn a lesson about step size into a reaction test — locking out anyone using
   * a screen reader or switch access, and making the whole thing untestable.
   */
  stepsRemaining: number;

  /** Steps that went uphill by more than OVERSHOOT_DELTA. */
  overshoots: number;
  /** True when the last step went uphill — drives the red spike. */
  lastStepOvershot: boolean;

  diverged: boolean;
  settled: boolean;
  /** True while the descent is auto-running. */
  running: boolean;

  failure: NamedFailure | null;
  whyCard: WhyCardContent | null;
  lastEvaluation: Evaluation | null;
  won: boolean;
  lane: Lane;

  // ── actions ──────────────────────────────────────────────────────────
  setLane: (lane: Lane) => void;
  /** THE dial. Setting it is choosing the step size, and nothing else. */
  setLearningRate: (rate: number) => void;
  setMomentum: (momentum: number) => void;
  setRunning: (running: boolean) => void;

  /** One gradient step. The whole algorithm, once. */
  step: () => void;
  /** Step until settled, diverged, or out of budget. */
  runToEnd: () => void;

  check: () => Evaluation;
  /** Back to the top of the mountain, keeping the dials. */
  reset: () => void;
  /** Back to the top with default dials. */
  fullReset: () => void;
}

function freshRun(learningRate: number, momentum: number) {
  const skier = makeSkier(learningRate, momentum);
  return {
    skier,
    currentLoss: START_LOSS,
    gradient: gradientAt(START.x, START.y),
    trail: [{ ...START }],
    lossHistory: [START_LOSS],
    stepsTaken: 0,
    stepsRemaining: STEP_BUDGET,
    overshoots: 0,
    lastStepOvershot: false,
    diverged: false,
    settled: false,
    running: false,
    failure: null,
    lastEvaluation: null,
    won: false,
  };
}

export const useGradientSkierStore = create<GradientSkierState>((set, get) => ({
  ...freshRun(DEFAULT_LEARNING_RATE, 0),
  whyCard: whyCardFor({ kind: "reset" }),
  lane: "visual" as Lane,

  setLane: (lane) => set({ lane }),

  setLearningRate: (rate) => {
    const learningRate = clamp(rate, MIN_LEARNING_RATE, MAX_LEARNING_RATE);
    const { skier } = get();
    set({
      skier: { ...skier, learningRate },
      whyCard: whyCardFor({
        kind: "rate-changed",
        learningRate,
        previous: skier.learningRate,
        gradient: get().gradient,
      }),
    });
  },

  setMomentum: (value) => {
    const momentum = clamp(value, 0, MAX_MOMENTUM);
    const { skier } = get();
    set({
      skier: { ...skier, momentum },
      whyCard: whyCardFor({
        kind: "momentum-changed",
        momentum,
        previous: skier.momentum,
      }),
    });
  },

  setRunning: (running) => set({ running }),

  step: () => {
    const state = get();
    if (state.diverged || state.settled || state.stepsRemaining <= 0) return;

    const previousLoss = state.currentLoss;
    const result = gradientStep(state.skier);

    const overshot = result.loss - previousLoss > OVERSHOOT_DELTA;
    const settled =
      !result.diverged &&
      gradientNorm(result.gradient) < SETTLED_GRADIENT &&
      result.stepLength < SETTLED_GRADIENT;

    // Trails are capped so a long run can't grow the array without bound.
    const trail = [...state.trail, { ...result.skier.pos }].slice(-STEP_BUDGET - 1);
    const lossHistory = [...state.lossHistory, result.loss].slice(
      -STEP_BUDGET - 1,
    );

    set({
      skier: result.skier,
      currentLoss: result.loss,
      gradient: result.gradient,
      trail,
      lossHistory,
      stepsTaken: state.stepsTaken + 1,
      stepsRemaining: state.stepsRemaining - 1,
      overshoots: state.overshoots + (overshot ? 1 : 0),
      lastStepOvershot: overshot,
      diverged: result.diverged,
      settled,
      running: result.diverged || settled ? false : state.running,
      whyCard: whyCardFor({
        kind: "stepped",
        loss: result.loss,
        previousLoss,
        gradient: result.gradient,
        overshot,
        diverged: result.diverged,
        settled,
        pos: result.skier.pos,
        learningRate: state.skier.learningRate,
        momentum: state.skier.momentum,
      }),
    });
  },

  runToEnd: () => {
    // Bounded by the budget, so this always terminates.
    for (let guard = 0; guard <= STEP_BUDGET; guard += 1) {
      const state = get();
      if (state.diverged || state.settled || state.stepsRemaining <= 0) break;
      state.step();
    }
    set({ running: false });
  },

  check: () => {
    const state = get();
    const evaluation = evaluate({
      pos: state.skier.pos,
      finalLoss: state.currentLoss,
      steps: state.stepsTaken,
      diverged: state.diverged,
      settled: state.settled,
      learningRate: state.skier.learningRate,
      momentum: state.skier.momentum,
      overshoots: state.overshoots,
    });

    set({
      failure: evaluation.failure,
      won: evaluation.outcome === "win",
      lastEvaluation: evaluation,
      whyCard: whyCardFor({ kind: "checked", evaluation }),
    });

    if (evaluation.outcome === "win") {
      useProgression.getState().recordResult({
        slug: SLUG,
        score: evaluation.score,
        lane: state.lane,
        completed: true,
        codeLaneCleared: state.lane === "code",
      });
    }

    return evaluation;
  },

  reset: () => {
    const { skier } = get();
    set({
      ...freshRun(skier.learningRate, skier.momentum),
      whyCard: whyCardFor({ kind: "reset" }),
    });
  },

  fullReset: () => {
    set({
      ...freshRun(DEFAULT_LEARNING_RATE, 0),
      whyCard: whyCardFor({ kind: "reset" }),
    });
  },
}));

/** Current loss at an arbitrary point — used by the terrain renderers. */
export { lossAt };
export type { Evaluation, Skier, Vec2 };
