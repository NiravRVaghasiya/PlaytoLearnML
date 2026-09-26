"use client";

import { create } from "zustand";
import { useProgression } from "@/engine/progression";
import type { Lane, NamedFailure } from "@/engine/types";
import type { WhyCardContent } from "@/components";
import { clamp } from "@/lib/utils";
import {
  GRADED_IDS,
  SCENARIOS,
  STEPS,
  STEP_COUNT,
  autograd,
  edgeKey,
  evaluate,
  nodeById,
  replay,
  roundScore,
  ruleById,
  train,
  type EdgeGrads,
  type Evaluation,
  type Routing,
  type Trace,
  type Values,
} from "./ml";
import { whyCardFor } from "./why-cards";

export const SLUG = "backprop-blitz";

export type Phase = "routing" | "cleared" | "broken";

/**
 * Backprop Blitz state.
 *
 * The spec's data model is:
 *
 *   Node      { id, value, localGrad, incomingGrad, outgoingGrad }
 *   Edge      { from, to, weight }
 *   GameState { graph, playerRouting[], trueGrads[], correctness }
 *
 * `graph` is not in state — it is the same fourteen nodes every round, so it lives
 * as a constant in ml.ts. Keeping a per-store copy would be the same graph again
 * with a chance of drifting from the one the autograd walks.
 *
 * The per-node gradient fields are not stored either, and that is the important
 * one. `value`, `localGrad`, `incomingGrad` and `outgoingGrad` are all pure
 * functions of the scenario and `routing`, and a stale gradient on a node is the
 * single most misleading thing this game could display — the entire premise is that
 * these numbers follow from the player's rules. So `routing` is the state, and every
 * gradient on screen is derived from it on read.
 *
 * `trueGrads` and `correctness` are likewise derived, from `autograd` and from the
 * diff against it.
 */
export interface BlitzState {
  scenarioIndex: number;
  /** Rule id per step index. The only real state in the game. */
  routing: Routing;
  /** Which step the inspector is showing. */
  focused: number;
  /**
   * Wrong turns: steps the player walked on from with a wrong rule still in
   * place, this round. Charged by `roundScore`.
   *
   * Counted when the inspector LEAVES a step (Next, Back, or clicking another
   * node), not when a radio is checked. The first version counted every wrong
   * radio ever checked, which sounds stricter and is wrong twice over: a native
   * radio group checks each option as the arrow keys pass through it, so it
   * would have charged keyboard players for moving through the list, and it
   * charged the thing the game asks for — try a rule, read the tick or cross,
   * change it. Walking on with a mistake is the wrong turn.
   */
  fumbled: number[];
  /** Scenario ids cleared this session. */
  clearedIds: string[];
  /**
   * Scenario ids whose real trace was revealed before they were cleared. Session
   * wide, like `clearedIds`, so "Start this scenario again" cannot launder a peek
   * — the same rule Dimension Diver applies to its PCA hint.
   */
  peekedIds: string[];
  /**
   * What this round has already sent to progression, so clearing the same round
   * again (re-choosing a rule, or running the starter snippet, which walks
   * cleared → broken → cleared a dozen times) records nothing new unless it is
   * genuinely better, or is the round's first code-lane clear.
   */
  recorded: { score: number; code: boolean } | null;

  evaluation: Evaluation;
  phase: Phase;
  failure: NamedFailure | null;
  whyCard: WhyCardContent | null;
  lane: Lane;
  /** Show the real trace alongside the player's. Off until they ask. */
  showTruth: boolean;

  setLane: (lane: Lane) => void;
  focusStep: (index: number) => void;
  /**
   * `source` is where the choice came from — the radio buttons, or `api.choose`
   * — so XP and the code-lane star follow the action rather than whichever tab
   * happens to be visible.
   */
  choose: (stepIndex: number, ruleId: string, source?: Lane) => void;
  clearStep: (stepIndex: number) => void;
  toggleTruth: () => void;
  nextScenario: () => void;
  reset: () => void;
}

// ── selectors: primitives and stable references only ─────────────────────

export const scenario = (state: BlitzState) => SCENARIOS[state.scenarioIndex]!;
export const correctnessFraction = (state: BlitzState): number =>
  state.evaluation.correctness.fraction;
export const answeredCount = (state: BlitzState): number =>
  Object.keys(state.routing).length;

/** The player's trace: values, gradients, and which nodes are settled. */
export function playerTraceFor(state: BlitzState) {
  return replay(scenario(state).leaves, state.routing);
}

/** The real trace, for the reveal and for the diff. */
export function truthTraceFor(state: BlitzState): Trace {
  return autograd(scenario(state).leaves);
}

export interface EdgeView {
  from: string;
  to: string;
  key: string;
  mine: number | null;
  truth: number;
  correct: boolean;
  /** This edge has been routed by the player. */
  routed: boolean;
}

/** Every edge with the gradient flowing along it, for `<ComputeGraph>`. */
export function edgeViews(
  mine: EdgeGrads,
  truth: EdgeGrads,
): EdgeView[] {
  const views: EdgeView[] = [];
  for (const node of STEPS.map((step) => step.nodeId)) {
    for (const producer of nodeById(node).inputs) {
      const key = edgeKey(node, producer);
      if (views.some((view) => view.key === key)) continue;
      const value = mine[key];
      const expected = truth[key] ?? 0;
      views.push({
        from: node,
        to: producer,
        key,
        mine: value ?? null,
        truth: expected,
        correct: value !== undefined && Math.abs(value - expected) < 1e-9,
        routed: value !== undefined,
      });
    }
  }
  return views;
}

function freshRound(scenarioIndex: number) {
  const leaves = SCENARIOS[scenarioIndex]!.leaves;
  return {
    scenarioIndex,
    routing: {} as Routing,
    focused: 0,
    fumbled: [] as number[],
    evaluation: evaluate({ leaves, routing: {} }),
    phase: "routing" as Phase,
    failure: null,
    showTruth: false,
    recorded: null,
  };
}

// The recorded score lives in ml.ts, beside `evaluate`, so the why-cards can
// quote its real price without importing the store that imports them.
export { PEEK_FACTOR, WRONG_TURN_COST, roundScore } from "./ml";

/** `fumbled` plus `stepIndex`, if the rule standing there is a wrong one. */
function withWrongTurn(state: BlitzState, stepIndex: number): number[] {
  const chosen = state.routing[stepIndex];
  const rule = chosen === undefined ? undefined : ruleById(chosen);
  if (rule === undefined || rule.correct || state.fumbled.includes(stepIndex)) {
    return state.fumbled;
  }
  return [...state.fumbled, stepIndex];
}

export const useBlitzStore = create<BlitzState>((set, get) => ({
  ...freshRound(0),
  clearedIds: [],
  peekedIds: [],
  whyCard: whyCardFor({ kind: "briefing", scenario: SCENARIOS[0]! }),
  lane: "visual" as Lane,

  setLane: (lane) => set({ lane }),

  focusStep: (index) => {
    if (!Number.isFinite(index)) return;
    const state = get();
    const next = clamp(Math.round(index), 0, STEP_COUNT - 1);
    if (next === state.focused) return;
    // Leaving a step is committing to it: a wrong rule still standing is a
    // wrong turn. See `fumbled`.
    set({ focused: next, fumbled: withWrongTurn(state, state.focused) });
  },

  toggleTruth: () => {
    const state = get();
    const next = !state.showTruth;
    const current = scenario(state);
    set({
      showTruth: next,
      // Revealing after the clear is reading the answer, not using it.
      peekedIds:
        next && state.phase !== "cleared" && !state.peekedIds.includes(current.id)
          ? [...state.peekedIds, current.id]
          : state.peekedIds,
      whyCard: next
        ? whyCardFor({ kind: "revealed" })
        : state.whyCard,
    });
  },

  choose: (stepIndex, ruleId, source = "visual") => {
    const state = get();
    const step = STEPS[stepIndex];
    const rule = ruleById(ruleId);
    if (step === undefined || rule === undefined) return;
    if (!step.options.some((option) => option.id === ruleId)) return;
    // Choosing what is already chosen changes nothing, so it records nothing —
    // with one exception. A snippet that re-asserts a routing the buttons already
    // cleared has cleared it from the code lane, and returning here would drop the
    // round's first code-lane clear (the third star) without a word. It falls
    // through once, re-evaluates to the same score, and is recorded as code.
    const firstCodeClear =
      source === "code" &&
      state.phase === "cleared" &&
      state.recorded !== null &&
      !state.recorded.code;
    if (state.routing[stepIndex] === ruleId && !firstCodeClear) return;

    const routing: Routing = { ...state.routing, [stepIndex]: ruleId };
    const leaves = scenario(state).leaves;
    const evaluation = evaluate({ leaves, routing });

    const cleared = evaluation.outcome === "cleared";
    const broken = evaluation.outcome === "broken";
    const current = scenario(state);

    // Deliberately NOT auto-advancing.
    //
    // The first version jumped to the next unanswered step the moment a rule was
    // picked, which felt brisk and was wrong: the whole activity is choosing a rule
    // and then LOOKING at the gradient it produced, tick or cross, against the
    // reference. Moving the panel away is moving the answer away. It also meant the
    // radio never settled into a checked state, which is confusing with a keyboard
    // and outright broken for a screen reader.
    const clearedIds =
      cleared && !state.clearedIds.includes(current.id)
        ? [...state.clearedIds, current.id]
        : state.clearedIds;

    // What this clear is worth, and whether progression has already heard it.
    const isCode = source === "code";
    const score = cleared
      ? roundScore({
          routingScore: evaluation.score,
          wrongTurns: state.fumbled.length,
          peeked: state.peekedIds.includes(current.id),
          clearedCount: clearedIds.length,
        })
      : 0;
    const previous = state.recorded;
    const worthRecording =
      cleared &&
      (previous === null || score > previous.score || (isCode && !previous.code));

    set({
      routing,
      evaluation,
      phase: cleared ? "cleared" : broken ? "broken" : "routing",
      failure: evaluation.failure,
      clearedIds,
      recorded: worthRecording
        ? {
            score: Math.max(score, previous?.score ?? 0),
            code: (previous?.code ?? false) || isCode,
          }
        : previous,
      whyCard: whyCardFor({
        kind: "chose",
        step,
        stepIndex,
        rule,
        evaluation,
        scenario: current,
        leaves,
        routing,
      }),
    });

    if (worthRecording) {
      useProgression.getState().recordResult({
        slug: SLUG,
        score,
        lane: source,
        completed: true,
        codeLaneCleared: isCode,
      });
    }
  },

  clearStep: (stepIndex) => {
    const state = get();
    if (state.routing[stepIndex] === undefined) return;
    const routing = { ...state.routing };
    delete routing[stepIndex];
    const leaves = scenario(state).leaves;
    const evaluation = evaluate({ leaves, routing });
    set({
      routing,
      evaluation,
      focused: stepIndex,
      phase: "routing",
      failure: null,
    });
  },

  nextScenario: () => {
    const state = get();
    const next = (state.scenarioIndex + 1) % SCENARIOS.length;
    set({
      ...freshRound(next),
      clearedIds: state.clearedIds,
      whyCard: whyCardFor({ kind: "briefing", scenario: SCENARIOS[next]! }),
    });
  },

  reset: () => {
    const state = get();
    set({
      ...freshRound(state.scenarioIndex),
      clearedIds: state.clearedIds,
      whyCard: whyCardFor({
        kind: "briefing",
        scenario: SCENARIOS[state.scenarioIndex]!,
      }),
    });
  },
}));

/**
 * What the code lane can do, and nothing more.
 *
 * Every verb writes the same store the buttons write (CLAUDE.md two-lane rule).
 * `api.autograd` is the interesting one: it hands over the real trace, which the
 * visual lane deliberately keeps hidden until asked. In the code lane that is not
 * cheating, it is the point — writing the loop that walks the graph backward and
 * checking it against a reference is exactly how you would verify a real autograd
 * implementation, and it is a better exercise than clicking the right options.
 */
export interface BlitzCodeApi {
  /** Choose a rule at a step. Same state the buttons write. */
  choose: (stepIndex: number, ruleId: string) => void;
  /** Choose the rule with a given id at every step that offers it. */
  chooseAll: (ruleId: string) => number;
  reset: () => void;
  nextScenario: () => void;

  steps: () => Array<{
    index: number;
    kind: string;
    node: string;
    op: string;
    prompt: string;
    options: Array<{ id: string; label: string; correct?: boolean }>;
    chosen: string | null;
  }>;
  scenario: () => { id: string; title: string; leaves: Values };
  /** Forward values and the player's gradients so far. */
  mine: () => { values: Values; grads: Values; settled: string[] };
  /** The reference trace. Real reverse-mode autograd. */
  autograd: () => { values: Values; grads: Values };
  correctness: () => {
    fraction: number;
    matched: number;
    total: number;
    angle: number;
    ratio: number;
    coincidences: number[];
    wrong: string[];
  };
  outcome: () => string;
  /** Descend for n steps with the player's rules, or with the real ones. */
  train: (options?: { useTruth?: boolean; steps?: number }) => {
    losses: number[];
    final: number;
    diverged: boolean;
  };
}

export function createCodeApi(): BlitzCodeApi {
  const store = useBlitzStore;

  return {
    choose: (stepIndex, ruleId) => {
      // `STEPS["2"]` and `STEPS[1.5]` are both things JavaScript will happily
      // index, so the step has to be an actual integer before it is looked up.
      const step = Number.isInteger(stepIndex) ? STEPS[stepIndex] : undefined;
      if (step === undefined) {
        throw new Error(
          `No step ${String(stepIndex)}. There are ${STEP_COUNT}, numbered 0 to ${
            STEP_COUNT - 1
          }.`,
        );
      }
      if (!step.options.some((option) => option.id === ruleId)) {
        throw new Error(
          `Step ${stepIndex} (${
            nodeById(step.nodeId).label
          }) does not offer "${String(ruleId)}". Options: ${step.options
            .map((option) => option.id)
            .join(", ")}.`,
        );
      }
      store.getState().choose(stepIndex, ruleId, "code");
    },

    chooseAll: (ruleId) => {
      if (ruleById(ruleId) === undefined) {
        throw new Error(
          `No rule "${String(ruleId)}". Rules: ${[
            ...new Set(STEPS.flatMap((step) => step.options.map((option) => option.id))),
          ].join(", ")}.`,
        );
      }
      let applied = 0;
      STEPS.forEach((step, index) => {
        if (step.options.some((option) => option.id === ruleId)) {
          store.getState().choose(index, ruleId, "code");
          applied += 1;
        }
      });
      return applied;
    },

    reset: () => store.getState().reset(),
    nextScenario: () => store.getState().nextScenario(),

    steps: () =>
      STEPS.map((step, index) => ({
        index,
        kind: step.kind,
        node: nodeById(step.nodeId).label,
        op: nodeById(step.nodeId).op,
        prompt: step.prompt,
        options: step.options.map((option) => ({
          id: option.id,
          label: option.label,
        })),
        chosen: store.getState().routing[index] ?? null,
      })),

    scenario: () => {
      const current = scenario(store.getState());
      return {
        id: current.id,
        title: current.title,
        leaves: { ...current.leaves },
      };
    },

    mine: () => {
      const state = store.getState();
      const trace = replay(scenario(state).leaves, state.routing);
      return {
        values: { ...trace.values },
        grads: { ...trace.grads },
        settled: [...trace.settled],
      };
    },

    autograd: () => {
      const trace = autograd(scenario(store.getState()).leaves);
      return { values: { ...trace.values }, grads: { ...trace.grads } };
    },

    correctness: () => {
      const { evaluation } = store.getState();
      const { correctness } = evaluation;
      return {
        fraction: correctness.fraction,
        matched: correctness.matched,
        total: correctness.total,
        angle: evaluation.angle,
        ratio: evaluation.ratio,
        coincidences: [...correctness.coincidences],
        wrong: correctness.verdicts
          .filter((verdict) => !verdict.correct)
          .map((verdict) => verdict.id),
      };
    },

    outcome: () => store.getState().evaluation.outcome,

    train: ({ useTruth = false, steps = 40 } = {}) => {
      if (!Number.isFinite(steps) || steps < 1 || steps > 5000) {
        throw new Error("steps must be between 1 and 5000.");
      }
      const state = store.getState();
      const run = train(
        scenario(state).leaves,
        useTruth ? null : state.routing,
        Math.floor(steps),
      );
      return { losses: [...run.losses], final: run.final, diverged: run.diverged };
    },
  };
}

/** Node gradient verdicts in graph order, for the inspector and the tests. */
export function verdictOrder(): string[] {
  return GRADED_IDS;
}
