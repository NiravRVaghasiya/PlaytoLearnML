import type { WhyCardContent } from "@/components";
import { HIGH_SCORE_THRESHOLD } from "@/engine/progression";
import {
  PARAM_IDS,
  SCENARIOS,
  STEPS,
  STEP_COUNT,
  TRAINING_STEPS,
  WRONG_TURN_COST,
  compareRuns,
  forward,
  lossText,
  nodeById,
  replay,
  roundScore,
  ruleById,
  type Evaluation,
  type Routing,
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
      /** The whole routing, so a coincidence can be explained step by step. */
      routing: Routing;
    }
  | { kind: "revealed" };

/**
 * The one concept page this game genuinely feeds into. A cleared walk ends on the
 * parameter gradients "a step of gradient descent actually consumes", and the
 * library's gradient-descent entry opens on exactly that loop (and names
 * backpropagation as how the gradient is computed). Nothing else in the library
 * covers backprop, so nothing else is linked — a link to a page that does not
 * answer the card's question is worse than no link.
 */
const GD_HREF = "/concepts/gradient-descent";

const percent = (value: number) => `${Math.round(value * 100)}%`;
const signed = (value: number) =>
  `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;

/**
 * Why one wrong rule produced the right numbers anyway, in this scenario.
 *
 * Built from the step and the forward pass rather than written once, because the
 * first version was: it told the "open gate" story — "with the gate open, 'always
 * pass' and 'pass if positive' agree" — to every coincidence, including the three
 * the shut scenario produces, where the gate is shut and the rule involved was
 * "always block". Measured, there are three mechanisms and each is named for
 * what actually happened:
 *
 *   gate open  — ignoring the ReLU costs nothing when z > 0.
 *   gate shut  — "always block" agrees with the real rule when z ≤ 0.
 *   zero input — behind a shut gate nothing arrives, and every rule sends 0.
 */
export type CoincidenceMechanism =
  | "gate-open"
  | "gate-shut"
  | "zero-input"
  | "numbers";

export interface CoincidenceReason {
  mechanism: CoincidenceMechanism;
  /** One step's story: which rule, at which node, and why it got away with it. */
  sentence: string;
}

export function coincidenceReason(
  stepIndex: number,
  routing: Routing,
  leaves: Values,
): CoincidenceReason | null {
  const step = STEPS[stepIndex];
  const rule = ruleById(routing[stepIndex] ?? "");
  if (step === undefined || rule === undefined) return null;
  const node = nodeById(step.nodeId);
  const quoted = `At ${node.label} you chose "${rule.label}"`;
  const z = forward(leaves).z ?? 0;
  const gateOpen = z > 0;

  if (rule.id === "relu-always" && gateOpen) {
    return {
      mechanism: "gate-open",
      sentence: `${quoted}. The gate is open here (z = ${z.toFixed(
        2,
      )}), so ignoring it cost nothing: "always pass the gradient" and "pass it only if the input was positive" give identical answers.`,
    };
  }

  if (rule.id === "relu-never" && !gateOpen) {
    return {
      mechanism: "gate-shut",
      sentence: `${quoted}. The gate is shut here (z = ${z.toFixed(
        2,
      )}), so blocking always and blocking only below zero agree.`,
    };
  }

  const incoming =
    step.kind === "route" ? (replay(leaves, routing).grads[step.nodeId] ?? 0) : null;
  if (incoming !== null && Math.abs(incoming) < 1e-12) {
    return {
      mechanism: "zero-input",
      sentence: `${quoted}. Nothing arrives at ${node.label}: ${
        gateOpen
          ? "its incoming gradient is exactly zero"
          : "the gate at h is shut, so everything behind it receives exactly zero"
      }, and any rule applied to zero gives zero.`,
    };
  }

  return {
    mechanism: "numbers",
    sentence: `${quoted}, and on these particular numbers it happens to produce the same gradients as the correct rule.`,
  };
}

/**
 * What walking on from a wrong rule costs, in the scoring's own numbers: the
 * share one wrong turn takes, and whether one alone is enough to pull a
 * flawless first clear under the two-star line. Computed, so the card that
 * states the price cannot drift from the price.
 */
export function wrongTurnPrice(): string {
  const firstClear = (wrongTurns: number) =>
    roundScore({ routingScore: 1, wrongTurns, peeked: false, clearedCount: 1 });
  const costsTheStar =
    firstClear(0) >= HIGH_SCORE_THRESHOLD && firstClear(1) < HIGH_SCORE_THRESHOLD;
  return `each one takes ${percent(WRONG_TURN_COST)} off this round's score${
    costsTheStar ? " — on a first clear, one is enough to miss the two-star line" : ""
  }`;
}

/** What each mechanism will do once the numbers change. Said once per card. */
const COINCIDENCE_OUTLOOK: Record<CoincidenceMechanism, string> = {
  "gate-open": "It will not be invisible in a scenario where z comes out negative.",
  "gate-shut":
    'In a scenario where z is positive, "always block" stops every gradient behind h.',
  "zero-input":
    "A rule that only ever saw zero has not been tested yet — it will not hide once a real gradient reaches it.",
  numbers: "Different numbers will expose it.",
};

/** The coincidence card's middle: every step's story, then each outlook once. */
function coincidenceStory(
  indices: readonly number[],
  routing: Routing,
  leaves: Values,
): string {
  const reasons = indices
    .map((index) => coincidenceReason(index, routing, leaves))
    .filter((reason): reason is CoincidenceReason => reason !== null);
  const outlooks = [...new Set(reasons.map((reason) => reason.mechanism))].map(
    (mechanism) => COINCIDENCE_OUTLOOK[mechanism],
  );
  return [...reasons.map((reason) => reason.sentence), ...outlooks].join(" ");
}

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
      const { step, rule, evaluation, leaves, routing } = event;
      const node = nodeById(step.nodeId);
      const { correctness, outcome } = evaluation;

      if (outcome === "cleared") {
        const lucky = correctness.coincidences.length;
        if (lucky > 0) {
          const reasons = coincidenceStory(
            correctness.coincidences,
            routing,
            leaves,
          );
          return {
            key: `cleared-${event.scenario.id}-${correctness.rulesRight}-${correctness.coincidences.join(".")}`,
            title: `Every gradient matches — but ${lucky} of your rules ${
              lucky === 1 ? "is" : "are"
            } wrong`,
            body: `The arithmetic is right and ${
              lucky === 1 ? "one of your rules is" : `${lucky} of your rules are`
            } not. ${reasons} This is the most valuable thing that can happen to you in this game — a rule that works for the wrong reason is one you will keep.`,
            tone: "good",
          };
        }

        return {
          key: `cleared-${event.scenario.id}-${correctness.rulesRight}`,
          title: `Every gradient matches the autograd trace`,
          body: `${correctness.rulesRight} of ${STEP_COUNT} rules right, and the gradients agree with a real autograd trace to the last decimal. Look at the parameter gradients: ${PARAM_IDS.map(
            (id) =>
              `${nodeById(id).label} ${signed(
                evaluation.correctness.verdicts.find(
                  (verdict) => verdict.id === id,
                )?.truth ?? 0,
              )}`,
          ).join(
            ", ",
          )}. Each of those came from the loss by nothing more than multiplying by one local derivative at a time, and they are what a step of gradient descent actually consumes. ${
            evaluation.truthRun === null || evaluation.truth === null
              ? ""
              : `Descend along them for ${TRAINING_STEPS} steps and the loss goes from ${evaluation.truth.before.toFixed(
                  3,
                )} to ${lossText(evaluation.truthRun.final)}.`
          }`,
          tone: "good",
          conceptHref: GD_HREF,
          conceptLabel: "How gradient descent works",
        };
      }

      if (outcome === "broken") {
        const rose =
          evaluation.mine !== null && evaluation.mine.after > evaluation.mine.before;
        const lowerAfterTraining =
          evaluation.mineRun !== null &&
          evaluation.truthRun !== null &&
          compareRuns(evaluation.mineRun, evaluation.truthRun) === "lower";
        return {
          key: `broken-${correctness.matched}-${rule.id}`,
          title: `${correctness.matched} of ${correctness.total} gradients match`,
          body: `The walk is finished and the trace does not agree. ${
            evaluation.angle > 90
              ? `Your gradient points uphill — ${evaluation.angle.toFixed(
                  0,
                )}° from the truth — so ${
                  rose
                    ? "the first step along it makes the loss worse."
                    : "it runs up the slope, even where one step happens to land lower."
                }`
              : evaluation.angle < 0.5 && Math.abs(evaluation.ratio - 1) > 0.02
                ? `Notice that the DIRECTION is right and only the length is off, at ${evaluation.ratio.toFixed(
                    2,
                  )} times the real gradient. That is the most dangerous kind of backward-pass bug, because it behaves like a learning rate nobody set.`
                : evaluation.angle < 0.5
                  ? `Oddly, your parameter gradients come out identical to the real ones on these numbers — the wrong values are all in intermediate nodes and never reach a weight here.`
                  : `It points ${evaluation.angle.toFixed(
                      0,
                    )}° away from the real gradient and is ${evaluation.ratio.toFixed(
                      2,
                    )} times its length.`
          }${
            lowerAfterTraining
              ? ` And yes, it trains to a lower loss than the real gradient here — on one example that is luck, not a sign the rule is right.`
              : ""
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
          // The cost is stated here, before the player acts on it: a penalty
          // discovered afterwards would be a trap. And carrying on is offered as
          // a choice with its real price, not recommended: the first version said
          // "carry on and watch, then come back" — the exact walk that is charged
          // — and called a lost star "a little score".
          body: `${rule.detail} It is still flowing, though — nothing here corrects your arithmetic for you, and that is deliberate. You can carry on and watch what it does to the nodes behind this one, but that has a price: moving on with a wrong rule counts as a wrong turn, and ${wrongTurnPrice()}. Changing it before you move on costs nothing, and "Start this scenario again" wipes the wrong turns once you have seen the damage.`,
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
