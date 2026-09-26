"use client";

import { create } from "zustand";
import { useProgression } from "@/engine/progression";
import type { Lane, NamedFailure } from "@/engine/types";
import type { WhyCardContent } from "@/components";
import { clamp } from "@/lib/utils";
import {
  DEFAULT_THRESHOLD,
  SCENARIOS,
  auc,
  confusionAt,
  generateSamples,
  judgeRound,
  majorityBaseline,
  metricsOf,
  rocCurve,
  scenarioAt,
  type RoundResult,
  type Sample,
  type Scenario,
} from "./ml";
import { whyCardFor } from "./why-cards";

export const SLUG = "confusion-matrix-chef";

/** Same seed the tests measure the shifts against. */
export const DATA_SEED = 9310;

export type Phase = "tuning" | "cleared" | "complete";

/** Anything but an explicit "code" is a visual-lane action (a click passes an event). */
const laneOf = (source: unknown): Lane => (source === "code" ? "code" : "visual");

/**
 * Confusion Matrix Chef state.
 *
 * The spec's data model is:
 *
 *   Sample    { score, trueLabel }
 *   GameState { threshold, TP, FP, TN, FN, scenarioTarget }
 *
 * TP/FP/TN/FN are deliberately NOT stored. They are a pure function of the
 * samples and the threshold, and keeping a copy would create four numbers that
 * can disagree with the slider — in a game whose entire subject is that the
 * matrix recomputes exactly as it would in production. Components derive them.
 *
 * `servedResult` is stored, because that one IS a distinct fact: it is the
 * judgement of the cutoff the player committed to, which must not follow the
 * slider around afterwards.
 */
export interface ChefState {
  scenarioIndex: number;
  threshold: number;
  samples: Sample[];

  /** The judgement of the committed cutoff. Null while still tuning. */
  servedResult: RoundResult | null;
  /** Attempts on the current shift. */
  attempts: number;
  /**
   * Best score per cleared shift, indexed by shift (shift 1 at index 0).
   *
   * Keyed rather than appended, so retrying a cleared shift replaces its score
   * instead of counting the shift twice. Shifts are only reachable in order, so
   * the array never has holes and its length is the number signed off.
   */
  clearedScores: number[];
  /**
   * Has any shift been cleared by a serve that came through the code lane?
   * The third star's criterion, tracked from the action rather than from which
   * tab happened to be open when the last shift was signed off.
   */
  codeLaneWin: boolean;

  phase: Phase;
  failure: NamedFailure | null;
  whyCard: WhyCardContent | null;
  lane: Lane;

  setLane: (lane: Lane) => void;
  /** The decision cutoff (spec: `<ThresholdSlider>`). */
  setThreshold: (threshold: number) => void;
  /**
   * Commit the current cutoff and let the critic judge it.
   *
   * `source` is the lane the serve came from — the button, or `api.serve` — so
   * XP and the code-lane star follow the action.
   *
   * Only a shift still being tuned can be served. Once it is signed off, serving
   * again (running the starter snippet twice does it) returns a fresh judgement
   * of the current cutoff WITHOUT touching the state: a cleared shift cannot be
   * double-counted, and cannot be un-cleared by a worse cutoff.
   */
  serve: (source?: Lane) => RoundResult;
  nextShift: () => void;
  /**
   * Start the current shift over — what the shell's Retry does. Shifts already
   * signed off stay signed off; that is what "Back to shift one" is for.
   */
  retryShift: () => void;
  restart: () => void;
}

// ── selectors: primitives and stable references only ─────────────────────

export function currentScenario(state: ChefState): Scenario {
  return scenarioAt(state.scenarioIndex);
}

export function clearedCount(state: ChefState): number {
  return state.clearedScores.length;
}

function samplesFor(scenario: Scenario): Sample[] {
  return generateSamples(
    scenario.prevalence,
    scenario.separability,
    DATA_SEED + scenario.index * 37,
  );
}

function freshShift(scenarioIndex: number) {
  const scenario = scenarioAt(scenarioIndex);
  return {
    scenarioIndex: scenario.index,
    threshold: DEFAULT_THRESHOLD,
    samples: samplesFor(scenario),
    servedResult: null,
    attempts: 0,
    phase: "tuning" as Phase,
    failure: null,
  };
}

export const useChefStore = create<ChefState>((set, get) => ({
  ...freshShift(1),
  clearedScores: [],
  codeLaneWin: false,
  whyCard: whyCardFor({ kind: "shift-briefing", scenario: scenarioAt(1) }),
  lane: "visual" as Lane,

  setLane: (lane) => set({ lane }),

  setThreshold: (threshold) => {
    const state = get();
    // clamp() passes NaN straight through, and a NaN cutoff flags nothing while
    // the slider shows garbage. The code lane rejects it with a message; this
    // is the backstop for any other caller.
    if (!Number.isFinite(threshold)) return;
    const next = clamp(threshold, 0, 1);
    if (next === state.threshold) return;

    set({
      threshold: next,
      // Moving the slider retracts the served judgement: the critic scored a
      // cutoff, and this is no longer that cutoff.
      servedResult: state.phase === "tuning" ? null : state.servedResult,
      failure: state.phase === "tuning" ? null : state.failure,
      whyCard: whyCardFor({
        kind: "threshold-moved",
        scenario: scenarioAt(state.scenarioIndex),
        samples: state.samples,
        threshold: next,
        previous: state.threshold,
      }),
    });
  },

  serve: (source) => {
    const state = get();
    const scenario = scenarioAt(state.scenarioIndex);
    const result = judgeRound({
      scenario,
      samples: state.samples,
      threshold: state.threshold,
    });

    // Re-entrancy guard. See the interface comment.
    if (state.phase !== "tuning") return result;

    const lane = laneOf(source);
    const won = result.outcome === "win";
    const lastShift = state.scenarioIndex >= SCENARIOS.length;
    const clearedScores = [...state.clearedScores];
    if (won) {
      const slot = state.scenarioIndex - 1;
      clearedScores[slot] = Math.max(clearedScores[slot] ?? 0, result.score);
    }
    const codeLaneWin = state.codeLaneWin || (won && lane === "code");
    const complete = won && lastShift;

    set({
      servedResult: result,
      attempts: state.attempts + 1,
      clearedScores,
      codeLaneWin,
      phase: complete ? "complete" : won ? "cleared" : "tuning",
      failure: result.failure,
      whyCard: whyCardFor({
        kind: "served",
        scenario,
        result,
        attempts: state.attempts + 1,
        complete,
      }),
    });

    // Only reachable on the transition INTO "complete": the guard above keeps a
    // finished week from recording itself again.
    if (complete) {
      const mean =
        clearedScores.reduce((total, value) => total + value, 0) /
        Math.max(1, clearedScores.length);
      useProgression.getState().recordResult({
        slug: SLUG,
        score: mean,
        lane,
        completed: true,
        codeLaneCleared: codeLaneWin,
      });
    }

    return result;
  },

  nextShift: () => {
    const state = get();
    if (state.phase !== "cleared") return;
    const scenario = scenarioAt(state.scenarioIndex + 1);
    set({
      ...freshShift(scenario.index),
      whyCard: whyCardFor({ kind: "shift-briefing", scenario }),
    });
  },

  retryShift: () => {
    const state = get();
    const scenario = scenarioAt(state.scenarioIndex);
    set({
      ...freshShift(scenario.index),
      whyCard: whyCardFor({ kind: "shift-briefing", scenario }),
    });
  },

  restart: () => {
    set({
      ...freshShift(1),
      clearedScores: [],
      codeLaneWin: false,
      whyCard: whyCardFor({ kind: "shift-briefing", scenario: scenarioAt(1) }),
    });
  },
}));

// ── The code lane's api ───────────────────────────────────────────────────

/** A cutoff argument from a script: a finite number in [0, 1], or a named error. */
function checkedThreshold(verb: string, threshold: unknown): number {
  if (typeof threshold !== "number" || !Number.isFinite(threshold)) {
    throw new Error(
      `${verb} needs a finite number between 0 and 1, got ${String(threshold)}`,
    );
  }
  if (threshold < 0 || threshold > 1) {
    throw new Error(`${verb}: threshold must be between 0 and 1, got ${threshold}`);
  }
  return threshold;
}

/**
 * What the code lane can do, and nothing more.
 *
 * `api.setThreshold` writes the same value the slider writes and `api.serve`
 * calls the same judgement the Serve button calls (CLAUDE.md two-lane rule) —
 * tagged as a code-lane serve, which is what the third star counts.
 */
export function createCodeApi() {
  const store = useChefStore;

  return {
    /** The decision cutoff. Same action as the slider. */
    setThreshold: (threshold: number) => {
      store.getState().setThreshold(checkedThreshold("setThreshold", threshold));
    },
    /** Commit the current cutoff for judgement. Same as the Serve button. */
    serve: () => store.getState().serve("code"),
    nextShift: () => store.getState().nextShift(),
    retryShift: () => store.getState().retryShift(),
    restart: () => store.getState().restart(),

    threshold: () => store.getState().threshold,

    /** The matrix at any cutoff, without committing to it. */
    matrixAt: (threshold: number) =>
      confusionAt(
        store.getState().samples,
        checkedThreshold("matrixAt", threshold),
      ),
    /** Every metric at any cutoff. */
    metricsAt: (threshold: number) =>
      metricsOf(
        confusionAt(
          store.getState().samples,
          checkedThreshold("metricsAt", threshold),
        ),
      ),

    /** The critic's brief, as data. */
    scenario: () => {
      const scenario = scenarioAt(store.getState().scenarioIndex);
      return {
        index: scenario.index,
        id: scenario.id,
        name: scenario.name,
        prevalence: scenario.prevalence,
        positiveLabel: scenario.positiveLabel,
        primary: scenario.primary,
        falsePositiveCost: scenario.falsePositiveCost,
        falseNegativeCost: scenario.falseNegativeCost,
      };
    },
    constraints: () =>
      scenarioAt(store.getState().scenarioIndex).constraints.map(
        (constraint) => ({ ...constraint }),
      ),
    /** Accuracy obtainable by ignoring the scores entirely. */
    majorityBaseline: () =>
      majorityBaseline(scenarioAt(store.getState().scenarioIndex).prevalence),

    /** The full ROC curve, and its area. */
    roc: () => rocCurve(store.getState().samples),
    auc: () => auc(store.getState().samples),

    /** The scores themselves, read-only. */
    samples: () => store.getState().samples.map((sample) => ({ ...sample })),
    sampleCount: () => store.getState().samples.length,

    shift: () => store.getState().scenarioIndex,
    cleared: () => clearedCount(store.getState()),
    lastResult: () => store.getState().servedResult,
    phase: () => store.getState().phase,
  };
}

export type ChefCodeApi = ReturnType<typeof createCodeApi>;
