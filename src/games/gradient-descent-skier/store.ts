"use client";

import { create } from "zustand";
import { useProgression } from "@/engine/progression";
import type { Lane, NamedFailure } from "@/engine/types";
import type { WhyCardContent } from "@/components";
import { clamp } from "@/lib/utils";
import {
  DEFAULT_LEARNING_RATE,
  GLOBAL_MINIMUM,
  MAX_LEARNING_RATE,
  MAX_MOMENTUM,
  MIN_LEARNING_RATE,
  OVERSHOOT_DELTA,
  SETTLED_GRADIENT,
  START,
  START_LOSS,
  STEP_BUDGET,
  UPHILL_DELTA,
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
  /**
   * The gradient where the skier stands NOW — what the next step will use.
   *
   * It used to be the gradient the last step had used, i.e. at the previous
   * position, so "slope at your feet", the next-step preview and api.gradient()
   * were all one step behind: after one step at α = 0.05 the readout said 4.83
   * while the true slope underfoot was 2.51.
   */
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
  /** Steps that went uphill by more than UPHILL_DELTA — the verdict's count. */
  rises: number;
  /** True when the last step went uphill — drives the red spike. */
  lastStepOvershot: boolean;
  /** Furthest left the path has been: whether, and how far, it crossed the ridge. */
  minX: number;

  /**
   * The dials the most recent step actually used. The verdict quotes these,
   * not whatever the dial reads when Score is pressed — turning the dial after
   * diverging at 1.0 used to produce "a learning rate of 0.0500 made each step
   * longer than the slope…".
   */
  runLearningRate: number;
  runMomentum: number;
  /** A dial moved between steps of this run, so no single α/β describes it. */
  dialsChangedMidRun: boolean;

  diverged: boolean;
  settled: boolean;
  /** True while the descent is auto-running. */
  running: boolean;

  failure: NamedFailure | null;
  whyCard: WhyCardContent | null;
  lastEvaluation: Evaluation | null;
  won: boolean;
  /** Lanes this run has already been recorded for, so a second Score can't double-count. */
  recordedFor: Lane[];
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

  /**
   * Score the run. `source` is where the request came from — "code" only for
   * `api.check()` — and that, not the visible tab, earns the code-lane star.
   */
  check: (source?: Lane) => Evaluation;
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
    rises: 0,
    lastStepOvershot: false,
    minX: START.x,
    runLearningRate: learningRate,
    runMomentum: momentum,
    dialsChangedMidRun: false,
    diverged: false,
    settled: false,
    running: false,
    failure: null,
    lastEvaluation: null,
    won: false,
    recordedFor: [] as Lane[],
  };
}

/** Has this run ended, one way or another? */
function isFinished(state: {
  diverged: boolean;
  settled: boolean;
  stepsRemaining: number;
}): boolean {
  return state.diverged || state.settled || state.stepsRemaining <= 0;
}

export const useGradientSkierStore = create<GradientSkierState>((set, get) => ({
  ...freshRun(DEFAULT_LEARNING_RATE, 0),
  whyCard: whyCardFor({ kind: "reset" }),
  lane: "visual" as Lane,

  setLane: (lane) => set({ lane }),

  setLearningRate: (rate) => {
    const learningRate = clamp(rate, MIN_LEARNING_RATE, MAX_LEARNING_RATE);
    const state = get();
    const skier = { ...state.skier, learningRate };
    const finished = isFinished(state);
    set({
      skier,
      dialsChangedMidRun:
        state.dialsChangedMidRun ||
        (state.stepsTaken > 0 &&
          !finished &&
          learningRate !== state.runLearningRate),
      whyCard: whyCardFor({
        kind: "rate-changed",
        learningRate,
        previous: state.skier.learningRate,
        gradient: state.gradient,
        skier,
        finished,
      }),
    });
  },

  setMomentum: (value) => {
    const momentum = clamp(value, 0, MAX_MOMENTUM);
    const state = get();
    set({
      skier: { ...state.skier, momentum },
      dialsChangedMidRun:
        state.dialsChangedMidRun ||
        (state.stepsTaken > 0 &&
          !isFinished(state) &&
          momentum !== state.runMomentum),
      whyCard: whyCardFor({
        kind: "momentum-changed",
        momentum,
        previous: state.skier.momentum,
      }),
    });
  },

  setRunning: (running) => set({ running }),

  step: () => {
    const state = get();
    if (isFinished(state)) return;

    const previousLoss = state.currentLoss;
    const result = gradientStep(state.skier);

    const overshot = result.loss - previousLoss > OVERSHOOT_DELTA;
    // `result.gradient` is the one this step USED, at the old position, which
    // is also what `descend` settles on — keep the two in lockstep so a run
    // clicked out by hand ends exactly where the code lane's does.
    const settled =
      !result.diverged &&
      gradientNorm(result.gradient) < SETTLED_GRADIENT &&
      result.stepLength < SETTLED_GRADIENT;
    // …but the readout shows the slope where the skier now stands.
    const here = gradientAt(result.skier.pos.x, result.skier.pos.y);

    // Trails are capped so a long run can't grow the array without bound.
    const trail = [...state.trail, { ...result.skier.pos }].slice(-STEP_BUDGET - 1);
    const lossHistory = [...state.lossHistory, result.loss].slice(
      -STEP_BUDGET - 1,
    );
    const minX = Math.min(state.minX, result.skier.pos.x);

    set({
      skier: result.skier,
      currentLoss: result.loss,
      gradient: here,
      trail,
      lossHistory,
      stepsTaken: state.stepsTaken + 1,
      stepsRemaining: state.stepsRemaining - 1,
      overshoots: state.overshoots + (overshot ? 1 : 0),
      rises: state.rises + (result.loss - previousLoss > UPHILL_DELTA ? 1 : 0),
      lastStepOvershot: overshot,
      minX,
      runLearningRate: state.skier.learningRate,
      runMomentum: state.skier.momentum,
      diverged: result.diverged,
      settled,
      running: result.diverged || settled ? false : state.running,
      whyCard: whyCardFor({
        kind: "stepped",
        loss: result.loss,
        previousLoss,
        gradient: here,
        overshot,
        diverged: result.diverged,
        settled,
        pos: result.skier.pos,
        learningRate: state.skier.learningRate,
        momentum: state.skier.momentum,
        minX,
      }),
    });
  },

  runToEnd: () => {
    // Bounded by the budget, so this always terminates.
    for (let guard = 0; guard <= STEP_BUDGET; guard += 1) {
      const state = get();
      if (isFinished(state)) break;
      state.step();
    }
    set({ running: false });
  },

  check: (source = "visual") => {
    const state = get();
    const ran = state.stepsTaken > 0;
    const evaluation = evaluate({
      pos: state.skier.pos,
      finalLoss: state.currentLoss,
      steps: state.stepsTaken,
      diverged: state.diverged,
      settled: state.settled,
      learningRate: ran ? state.runLearningRate : state.skier.learningRate,
      momentum: ran ? state.runMomentum : state.skier.momentum,
      overshoots: state.overshoots,
      rises: state.rises,
      minX: state.minX,
      constantDials: !state.dialsChangedMidRun,
    });

    // Reaching the deepest valley is what completes the game (spec: "Reach the
    // global minimum before the timer"), however many steps it took — a slow
    // arrival is a near-miss on SCORE, and its one star is real. The second
    // star is the score threshold, which only a quick arrival clears.
    const reached =
      evaluation.outcome === "win" || evaluation.outcome === "near-miss";
    const alreadyRecorded = state.recordedFor.includes(source);

    set({
      failure: evaluation.failure,
      won: evaluation.outcome === "win",
      lastEvaluation: evaluation,
      whyCard: whyCardFor({ kind: "checked", evaluation }),
      recordedFor:
        reached && !alreadyRecorded
          ? [...state.recordedFor, source]
          : state.recordedFor,
    });

    if (reached && !alreadyRecorded) {
      useProgression.getState().recordResult({
        slug: SLUG,
        score: evaluation.score,
        lane: source,
        completed: true,
        codeLaneCleared: source === "code",
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

// ── The code lane's API ─────────────────────────────────────────────────────

/** A finite number, or a named TypeError saying which argument wasn't. */
function finite(verb: string, name: string, value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(
      `${verb}: ${name} must be a finite number, got ${
        typeof value === "number" ? String(value) : typeof value
      }`,
    );
  }
  return value;
}

/** A finite number inside [min, max], or a named RangeError. */
function inRange(
  verb: string,
  name: string,
  value: unknown,
  min: number,
  max: number,
): number {
  const number = finite(verb, name, value);
  if (number < min || number > max) {
    throw new RangeError(
      `${verb}: ${name} ${number} is outside the dial's range, ${min} to ${max}`,
    );
  }
  return number;
}

/**
 * The `api` object the code lane's snippet receives. Built here, next to the
 * store, so its argument checking is unit-testable without rendering anything.
 *
 * `api.step()` is the same store action the Step button calls, so the loop
 * written in code and the loop clicked out by hand are the same computation on
 * the same state (CLAUDE.md two-lane rule).
 */
export function createCodeApi() {
  const store = useGradientSkierStore;

  return {
    /** One gradient step. Same action as the Step button. */
    step: () => store.getState().step(),
    /** Step until settled, diverged, or out of budget. */
    runToEnd: () => store.getState().runToEnd(),

    /**
     * THE dial: α in the update rule. Out-of-range values throw rather than
     * being silently clamped, so a script never runs with a rate it didn't ask for.
     */
    setLearningRate: (value: number) =>
      store
        .getState()
        .setLearningRate(
          inRange(
            "setLearningRate",
            "the rate",
            value,
            MIN_LEARNING_RATE,
            MAX_LEARNING_RATE,
          ),
        ),
    /** β in the update rule. */
    setMomentum: (value: number) =>
      store
        .getState()
        .setMomentum(inRange("setMomentum", "momentum", value, 0, MAX_MOMENTUM)),

    learningRate: () => store.getState().skier.learningRate,
    momentum: () => store.getState().skier.momentum,

    position: () => ({ ...store.getState().skier.pos }),
    velocity: () => ({ ...store.getState().skier.velocity }),
    /** Live loss — the same number the metric shows. */
    loss: () => store.getState().currentLoss,
    /** The exact gradient at the skier's feet — where the next step starts. */
    gradient: () => ({ ...store.getState().gradient }),
    slope: () => gradientNorm(store.getState().gradient),

    /** Loss anywhere on the surface, without moving. */
    lossAt: (x: number, y: number) =>
      lossAt(finite("lossAt", "x", x), finite("lossAt", "y", y)),
    /** Gradient anywhere on the surface, without moving. */
    gradientAt: (x: number, y: number) =>
      gradientAt(finite("gradientAt", "x", x), finite("gradientAt", "y", y)),
    globalMinimum: () => ({ ...GLOBAL_MINIMUM }),

    steps: () => store.getState().stepsTaken,
    stepsRemaining: () => store.getState().stepsRemaining,
    overshoots: () => store.getState().overshoots,
    settled: () => store.getState().settled,
    diverged: () => store.getState().diverged,
    trail: () => store.getState().trail.map((point) => ({ ...point })),

    check: () => store.getState().check("code"),
    /** Back to the top of the mountain, keeping the dials. */
    reset: () => store.getState().reset(),
  };
}

/** Current loss at an arbitrary point — used by the terrain renderers. */
export { lossAt };
export type { Evaluation, Skier, Vec2 };
