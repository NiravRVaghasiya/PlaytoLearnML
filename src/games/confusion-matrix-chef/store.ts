"use client";

import { create } from "zustand";
import { useProgression } from "@/engine/progression";
import type { Lane, NamedFailure } from "@/engine/types";
import type { WhyCardContent } from "@/components";
import { clamp } from "@/lib/utils";
import {
  DEFAULT_THRESHOLD,
  SCENARIOS,
  generateSamples,
  judgeRound,
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
  /** Score per cleared shift, for the final progression score. */
  clearedScores: number[];

  phase: Phase;
  failure: NamedFailure | null;
  whyCard: WhyCardContent | null;
  lane: Lane;

  setLane: (lane: Lane) => void;
  /** The decision cutoff (spec: `<ThresholdSlider>`). */
  setThreshold: (threshold: number) => void;
  /** Commit the current cutoff and let the critic judge it. */
  serve: () => RoundResult;
  nextShift: () => void;
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
  whyCard: whyCardFor({ kind: "shift-briefing", scenario: scenarioAt(1) }),
  lane: "visual" as Lane,

  setLane: (lane) => set({ lane }),

  setThreshold: (threshold) => {
    const state = get();
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

  serve: () => {
    const state = get();
    const scenario = scenarioAt(state.scenarioIndex);
    const result = judgeRound({
      scenario,
      samples: state.samples,
      threshold: state.threshold,
    });

    const won = result.outcome === "win";
    const lastShift = state.scenarioIndex >= SCENARIOS.length;
    const clearedScores = won
      ? [...state.clearedScores, result.score]
      : state.clearedScores;
    const complete = won && lastShift;

    set({
      servedResult: result,
      attempts: state.attempts + 1,
      clearedScores,
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

    if (complete) {
      const mean =
        clearedScores.reduce((total, value) => total + value, 0) /
        Math.max(1, clearedScores.length);
      useProgression.getState().recordResult({
        slug: SLUG,
        score: mean,
        lane: state.lane,
        completed: true,
        codeLaneCleared: state.lane === "code",
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

  restart: () => {
    set({
      ...freshShift(1),
      clearedScores: [],
      whyCard: whyCardFor({ kind: "shift-briefing", scenario: scenarioAt(1) }),
    });
  },
}));
