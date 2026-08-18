import type { WhyCardContent } from "@/components";
import {
  PARAM_IDS,
  SCENARIOS,
  STEP_COUNT,
  TRAINING_STEPS,
  nodeById,
  type Evaluation,
  type Rule,
  type Scenario,
  type Step,
  type Values,
} from "./ml";

/**
 * "Why did that happen?" copy for Backprop Blitz.
 *
 * The rule this copy follows: when the player gets a node right, explain what made
 * it right, not that it was right. A green tick is already on screen. "The gradient
 * arrives as a rate, and a + node changes one-for-one with both inputs, so both get
 * all of it" is the sentence that transfers to the next graph they meet.
 *
 * And when they get one wrong, say which misconception it was. There are only about
 * ten ways to break a backward pass and every one of them has a name, so there is
 * no excuse for "incorrect".
 */
export type BlitzEvent =
  | { kind: "briefing"; scenario: Scenario }
  | {
      kind: "chose";
      step: Step;
      stepIndex: number;
      rule: Rule;
      evaluation: Evaluation;
      scenario: Scenario;
      leaves: Values;
    }
  | { kind: "revealed" };

const percent = (value: number) => `${Math.round(value * 100)}%`;
const signed = (value: number) =>
  `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;

export function whyCardFor(event: BlitzEvent): WhyCardContent {
  switch (event.kind) {
    case "briefing": {
      const { scenario } = event;
      const isFirst = scenario.id === SCENARIOS[0]!.id;
      return {
        key: `briefing-${scenario.id}`,
        title: isFirst
          ? "The error is at the output. Push it back."
          : scenario.title,
        body: `${scenario.lesson} ${
          isFirst
            ? `${STEP_COUNT} decisions, one per node, walking backwards. At each one you choose how the arriving gradient transforms — and the numbers you produce are checked against a real reverse-mode autograd trace, not against a stored answer key. Wrong choices are allowed to keep flowing, because that is the thing worth seeing: a broken link does not make one node wrong, it poisons everything behind it.`
            : `Same graph, different numbers. A rule that got away with it last time may not this time.`
        }`,
        tone: "info",
      };
    }

    case "revealed":
      return {
        key: "revealed",
        title: "The real trace, alongside yours",
        body: `Every node now shows what autograd computed next to what you produced. Worth reading backwards from the loss: each number is the one above it times one local derivative, and nothing else. There is no global calculation anywhere in a backward pass — just that, repeated.`,
        tone: "info",
      };

    case "chose": {
      const { step, rule, evaluation, leaves } = event;
      const node = nodeById(step.nodeId);
      const { correctness, outcome } = evaluation;

      if (outcome === "cleared") {
        const lucky = correctness.coincidences.length;
        return {
          key: `cleared-${event.scenario.id}-${correctness.rulesRight}`,
          title:
            lucky > 0
              ? `Every gradient matches — but ${lucky} of your rules is wrong`
              : `Every gradient matches the autograd trace`,
          body:
            lucky > 0
              ? `The arithmetic is right and one of your rules is not. On these particular numbers it made no difference: with the gate open, "always pass the gradient" and "pass it only if the input was positive" give identical answers, so the mistake is invisible here. It will not be invisible in the scenario where z comes out negative. This is the most valuable thing that can happen to you in this game — a rule that works for the wrong reason is one you will keep.`
              : `${correctness.rulesRight} of ${STEP_COUNT} rules right, and the gradients agree with a real autograd trace to the last decimal. Look at the parameter gradients: ${PARAM_IDS.map(
                  (id) =>
                    `${nodeById(id).label} ${signed(
                      evaluation.correctness.verdicts.find(
                        (verdict) => verdict.id === id,
                      )?.truth ?? 0,
                    )}`,
                ).join(", ")}. Each of those came from the loss by nothing more than multiplying by one local derivative at a time, and they are what a step of gradient descent actually consumes. ${
                  evaluation.truthRun === null
                    ? ""
                    : `Descend along them for ${TRAINING_STEPS} steps and the loss goes from ${leaves.t !== undefined ? evaluation.truth!.before.toFixed(3) : "?"} to ${evaluation.truthRun.final.toFixed(
                        4,
                      )}.`
                }`,
          tone: "good",
        };
      }

      if (outcome === "broken") {
        return {
          key: `broken-${correctness.matched}-${rule.id}`,
          title: `${correctness.matched} of ${correctness.total} gradients match`,
          body: `The walk is finished and the trace does not agree. ${
            evaluation.angle > 90
              ? `Your gradient points uphill — ${evaluation.angle.toFixed(
                  0,
                )}° from the truth — so subtracting it would make the loss worse.`
              : evaluation.angle < 0.5
                ? `Notice that the DIRECTION is right and only the length is off, at ${evaluation.ratio.toFixed(
                    2,
                  )} times the real gradient. That is the most dangerous kind of backward-pass bug, because it behaves like a learning rate nobody set.`
                : `It points ${evaluation.angle.toFixed(
                    0,
                  )}° away from the real gradient and is ${evaluation.ratio.toFixed(
                    2,
                  )} times its length.`
          } The red node nearest the loss is the one to fix — everything behind it is only wrong because of it.`,
          tone: "bad",
        };
      }

      // Mid-walk. Explain the rule just applied.
      const settled = correctness.verdicts.find(
        (verdict) => verdict.id === step.nodeId,
      );

      if (!rule.correct) {
        return {
          key: `wrong-${event.stepIndex}-${rule.id}`,
          title: `That is not the rule at ${node.label}`,
          body: `${rule.detail} It is still flowing, though — nothing here corrects your arithmetic for you, and that is deliberate. Watch what happens to the nodes behind this one as you carry on, then come back and change it.`,
          tone: "warn",
        };
      }

      return {
        key: `right-${event.stepIndex}-${rule.id}`,
        title: `${node.label}: ${rule.label}`,
        body: `${rule.detail}${
          settled?.mine !== null && settled !== undefined
            ? ` ${node.label}'s gradient settles at ${signed(
                settled.mine ?? 0,
              )}.`
            : ""
        } ${percent(correctness.fraction)} of the graph's gradients now match.`,
        tone: "good",
      };
    }
  }
}

/** Caption for the correctness meter. */
export function meterCaption(
  answered: number,
  fraction: number,
  coincidences: number,
): string {
  if (answered === 0) {
    return `Nothing routed yet. Only the loss knows its own gradient, and it is 1 — the derivative of anything with respect to itself.`;
  }
  if (fraction >= 1) {
    return coincidences > 0
      ? `All gradients match, but ${coincidences} rule${
          coincidences === 1 ? " is" : "s are"
        } wrong and got away with it here.`
      : `Every gradient in the graph matches the reference trace.`;
  }
  return `${answered} of ${STEP_COUNT} nodes routed. Gradients are compared against a real autograd trace, so a red node behind a red node is usually only the first one's fault.`;
}
