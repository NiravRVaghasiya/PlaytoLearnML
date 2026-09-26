import * as tf from "@tensorflow/tfjs";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import {
  ACCUMULATE_RULES,
  GRADED_IDS,
  LEARNING_RATE,
  NODES,
  NO_GRADIENT_IDS,
  PARAM_IDS,
  ROUTE_RULES,
  SCENARIOS,
  STEPS,
  STEP_COUNT,
  TRAINING_STEPS,
  MATH_NOTES,
  applyStep,
  autograd,
  compareRuns,
  consumersOf,
  displayOrder,
  edgeKey,
  evaluate,
  fanOut,
  forward,
  gradientAngle,
  isLeaf,
  magnitudeRatio,
  nodeById,
  replay,
  ruleById,
  rulesForOp,
  scoreRouting,
  topologicalOrder,
  train,
  type Routing,
  type Values,
} from "./ml";
import {
  PEEK_FACTOR,
  SLUG,
  WRONG_TURN_COST,
  createCodeApi,
  edgeViews,
  roundScore,
  useBlitzStore,
} from "./store";
import {
  coincidenceReason,
  meterCaption,
  whyCardFor,
  wrongTurnPrice,
} from "./why-cards";
import { STARTER_CODE } from "./CodeLane";
import {
  HIGH_SCORE_THRESHOLD,
  createMemoryAdapter,
  useProgression,
} from "@/engine/progression";
import { createJsExecutor } from "@/engine/useCodeLane";

/**
 * The claim this game makes about itself is that the ground truth is a real
 * reverse-mode autograd trace. That is checked here against `tf.grads` on the same
 * expression — if the two ever disagree, every number the game shows is suspect and
 * these tests should be the thing that says so.
 *
 * The rest was measured by a throwaway probe: which misconceptions are
 * distinguishable, what each costs in angle and in length, and which wrong rules
 * coincidentally produce the right answer. The probe is gone; the findings are here.
 */

beforeAll(async () => {
  await tf.ready();
}, 60000);

/** The routing that picks the correct rule at every step. */
function perfectRouting(): Routing {
  const routing: Routing = {};
  STEPS.forEach((step, index) => {
    const right = step.options.find((option) => option.correct);
    if (right) routing[index] = right.id;
  });
  return routing;
}

/**
 * The routing that picks the correct rule everywhere except one step.
 *
 * `atNode` matters: several rules are offered at more than one node (there are
 * three adds and two multiplies), and which one you break changes the blast radius
 * completely — a wrong rule at `u` corrupts two leaves, the same rule at `v`
 * corrupts half the graph.
 */
function routingWith(
  ruleId: string,
  atNode?: string,
): { routing: Routing; index: number } {
  const routing = perfectRouting();
  const index = STEPS.findIndex(
    (step) =>
      step.options.some((option) => option.id === ruleId) &&
      (atNode === undefined || step.nodeId === atNode),
  );
  if (index < 0) {
    throw new Error(`no step${atNode ? ` at ${atNode}` : ""} offers "${ruleId}"`);
  }
  routing[index] = ruleId;
  return { routing, index };
}

const scenarioById = (id: string) =>
  SCENARIOS.find((candidate) => candidate.id === id)!;

describe("the graph", () => {
  it("is fourteen nodes with the loss at the end", () => {
    expect(NODES).toHaveLength(14);
    expect(nodeById("L").op).toBe("square");
    expect(consumersOf("L")).toHaveLength(0);
  });

  it("orders producers before consumers", () => {
    const order = topologicalOrder();
    expect(order).toHaveLength(NODES.length);
    for (const node of NODES) {
      for (const input of node.inputs) {
        expect(order.indexOf(input)).toBeLessThan(order.indexOf(node.id));
      }
    }
  });

  it("has exactly one value used twice, and it is the activation", () => {
    // The whole reason this graph is a residual block rather than a chain: without
    // a fan-out there is nowhere for the multivariable chain rule to happen.
    const branched = NODES.filter((node) => fanOut(node.id) > 1).map(
      (node) => node.id,
    );
    expect(branched).toEqual(["h"]);
    expect(consumersOf("h").sort()).toEqual(["s", "v"]);
  });

  it("gives the target no gradient, because nothing optimises it", () => {
    expect(NO_GRADIENT_IDS).toEqual(["t"]);
    expect(GRADED_IDS).not.toContain("t");
    expect(GRADED_IDS).not.toContain("L");
  });

  it("has four learnable parameters", () => {
    expect(PARAM_IDS).toEqual(["w1", "b1", "w2", "b2"]);
  });
});

describe("the forward pass", () => {
  it("computes the residual block by hand-checkable arithmetic", () => {
    const values = forward(scenarioById("open").leaves);
    // x=2, w1=1.5, b1=-1, w2=0.5, b2=0.25, t=1
    expect(values.u).toBe(3); // 1.5 * 2
    expect(values.z).toBe(2); // 3 - 1
    expect(values.h).toBe(2); // relu(2)
    expect(values.v).toBe(1); // 0.5 * 2
    expect(values.s).toBe(3); // 1 + 2, the skip connection
    expect(values.y).toBe(3.25); // 3 + 0.25
    expect(values.d).toBe(2.25); // 3.25 - 1
    expect(values.L).toBeCloseTo(5.0625, 10); // 2.25^2
  });

  it("shuts the ReLU when its input is negative", () => {
    const values = forward(scenarioById("shut").leaves);
    expect(values.z).toBe(-2);
    expect(values.h).toBe(0);
    expect(values.v).toBe(0);
    expect(values.s).toBe(0);
  });
});

describe("the autograd trace is real", () => {
  it("matches tf.grads on every scenario, exactly", () => {
    // The load-bearing test in this file. "Real autograd as ground truth" is a
    // claim the game makes on screen, and this is what makes it true.
    for (const scenario of SCENARIOS) {
      const { leaves } = scenario;
      const mine = autograd(leaves);

      // No tf.tidy in the loss: tidy disposes the intermediate tensors the
      // backward pass still needs, and tf.grads spreads its arguments.
      const loss = (
        w1: tf.Tensor,
        b1: tf.Tensor,
        w2: tf.Tensor,
        b2: tf.Tensor,
      ): tf.Scalar => {
        const x = tf.scalar(leaves.x!);
        const t = tf.scalar(leaves.t!);
        const h = tf.relu(tf.add(tf.mul(w1, x), b1));
        const y = tf.add(tf.add(tf.mul(w2, h), h), b2);
        return tf.square(tf.sub(y, t)) as tf.Scalar;
      };

      const inputs = PARAM_IDS.map((id) => tf.scalar(leaves[id]!));
      const grads = tf.grads(loss)(inputs);
      const theirs = grads.map((g) => g.dataSync()[0]!);
      grads.forEach((g) => g.dispose());
      inputs.forEach((v) => v.dispose());

      PARAM_IDS.forEach((id, index) => {
        expect(
          mine.grads[id],
          `${scenario.id}: d${id} disagrees with tf.grads`,
        ).toBeCloseTo(theirs[index]!, 9);
      });
    }
  }, 60000);

  it("starts the loss off at a gradient of one", () => {
    // The derivative of anything with respect to itself.
    expect(autograd(scenarioById("open").leaves).grads.L).toBe(1);
  });

  it("sums the two contributions at the branch", () => {
    const trace = autograd(scenarioById("open").leaves);
    // s sends 4.5 to h, v sends 4.5 * w2 = 2.25 to h, and they ADD.
    expect(trace.edges[edgeKey("s", "h")]).toBeCloseTo(4.5, 10);
    expect(trace.edges[edgeKey("v", "h")]).toBeCloseTo(2.25, 10);
    expect(trace.grads.h).toBeCloseTo(6.75, 10);
  });

  it("crosses the operands at a multiply", () => {
    const trace = autograd(scenarioById("open").leaves);
    // dv/dw2 = h = 2, so w2 gets 4.5 * 2 = 9. dv/dh = w2 = 0.5.
    expect(trace.grads.w2).toBeCloseTo(9, 10);
    expect(trace.edges[edgeKey("v", "h")]).toBeCloseTo(4.5 * 0.5, 10);
  });

  it("copies rather than divides at an add", () => {
    const trace = autograd(scenarioById("open").leaves);
    expect(trace.edges[edgeKey("y", "s")]).toBeCloseTo(4.5, 10);
    expect(trace.edges[edgeKey("y", "b2")]).toBeCloseTo(4.5, 10);
  });

  it("passes exactly zero through a shut gate", () => {
    const trace = autograd(scenarioById("shut").leaves);
    // h itself has a gradient; z and everything behind it gets nothing.
    expect(trace.grads.h).toBeCloseTo(-2.25, 10);
    expect(trace.grads.z).toBe(0);
    expect(trace.grads.w1).toBe(0);
    expect(trace.grads.b1).toBe(0);
    // b2 is downstream of the gate, so it still learns.
    expect(trace.grads.b2).toBeCloseTo(-1.5, 10);
  });

  it("lowers the loss when you follow it", () => {
    for (const scenario of SCENARIOS) {
      const trace = autograd(scenario.leaves);
      const update = applyStep(scenario.leaves, trace.grads);
      expect(update.after, `${scenario.id}`).toBeLessThan(update.before);
    }
  });
});

describe("the steps", () => {
  it("walk backwards, with an accumulate before the branch is routed", () => {
    const order = STEPS.map(
      (step) => `${step.kind === "accumulate" ? "acc:" : ""}${step.nodeId}`,
    );
    expect(order).toEqual([
      "L",
      "d",
      "y",
      "s",
      "v",
      "acc:h",
      "h",
      "z",
      "u",
    ]);
    expect(STEP_COUNT).toBe(9);
  });

  it("offer exactly one correct option each", () => {
    for (const step of STEPS) {
      const right = step.options.filter((option) => option.correct);
      expect(right, `${step.nodeId} (${step.kind})`).toHaveLength(1);
      expect(step.options.length).toBeGreaterThan(1);
    }
  });

  it("give every wrong option a named misconception", () => {
    for (const rule of [...ROUTE_RULES, ...ACCUMULATE_RULES]) {
      if (rule.correct) continue;
      expect(rule.misconception, rule.id).toBeDefined();
      expect(rule.detail.length).toBeGreaterThan(20);
    }
  });

  it("offer rules for every operation in the graph", () => {
    for (const node of NODES) {
      if (isLeaf(node)) continue;
      expect(rulesForOp(node.op).length, node.op).toBeGreaterThan(0);
    }
  });
});

describe("replaying the player's choices", () => {
  it("reproduces the autograd trace exactly when every rule is right", () => {
    // If this ever fails, the game is marking against a different algorithm from
    // the one it teaches.
    for (const scenario of SCENARIOS) {
      const mine = replay(scenario.leaves, perfectRouting());
      const truth = autograd(scenario.leaves);
      for (const id of GRADED_IDS) {
        expect(mine.grads[id], `${scenario.id}: ${id}`).toBeCloseTo(
          truth.grads[id]!,
          12,
        );
      }
    }
  });

  it("scores a perfect routing at 100%", () => {
    for (const scenario of SCENARIOS) {
      const score = scoreRouting(scenario.leaves, perfectRouting());
      expect(score.fraction, scenario.id).toBe(1);
      expect(score.rulesRight).toBe(STEP_COUNT);
      expect(score.coincidences).toHaveLength(0);
    }
  });

  it("settles nothing before it is routed", () => {
    const mine = replay(scenarioById("open").leaves, {});
    // Only the loss knows its own gradient without being told.
    expect(mine.settled).toEqual(["L"]);
    const score = scoreRouting(scenarioById("open").leaves, {});
    expect(score.fraction).toBe(0);
  });

  it("trains identically to real autograd when the routing is right", () => {
    for (const scenario of SCENARIOS) {
      const mine = train(scenario.leaves, perfectRouting());
      const truth = train(scenario.leaves, null);
      expect(mine.final, scenario.id).toBeCloseTo(truth.final, 12);
      expect(mine.diverged).toBe(false);
    }
  });
});

describe("what each misconception costs", () => {
  const open = scenarioById("open").leaves;

  it("halving at an add keeps the direction and halves the length", () => {
    // Measured, and the most interesting result the probe turned up: this is not a
    // wrong direction, it is a wrong learning rate wearing a disguise.
    const { routing } = routingWith("add-split");
    const evaluation = evaluate({ leaves: open, routing });
    expect(evaluation.outcome).toBe("broken");
    expect(evaluation.angle).toBeCloseTo(0, 6);
    expect(evaluation.ratio).toBeCloseTo(0.5, 6);
    expect(evaluation.failure?.name).toBe("Add nodes copy, they do not split");
    expect(evaluation.failure?.detail).toMatch(/learning-rate bug in disguise/);
  });

  it("dropping the factor of two at the square does the same thing", () => {
    const { routing } = routingWith("square-d");
    const evaluation = evaluate({ leaves: open, routing });
    expect(evaluation.angle).toBeCloseTo(0, 6);
    expect(evaluation.ratio).toBeCloseTo(0.5, 6);
    expect(evaluation.failure?.name).toBe(
      "Reused the function instead of its derivative",
    );
  });

  it("flipping the sign points uphill and diverges", () => {
    const { routing } = routingWith("sub-negate");
    const evaluation = evaluate({ leaves: open, routing });
    expect(evaluation.angle).toBeCloseTo(180, 6);
    expect(evaluation.mineRun?.diverged).toBe(true);
    expect(evaluation.truthRun?.diverged).toBe(false);
    expect(evaluation.failure?.name).toBe(
      "Sent the gradient backward with the wrong sign",
    );
    expect(evaluation.failure?.detail).toMatch(/UPHILL/);
  });

  it("swapping a multiply's operands points somewhere else entirely", () => {
    const { routing } = routingWith("mul-self");
    const evaluation = evaluate({ leaves: open, routing });
    expect(evaluation.angle).toBeGreaterThan(5);
    expect(evaluation.angle).toBeLessThan(90);
    expect(evaluation.failure?.name).toBe("Swapped the multiply's partners");
  });

  it("averaging a branch is distinguishable from dropping one", () => {
    const mean = evaluate({ leaves: open, routing: routingWith("acc-mean").routing });
    const first = evaluate({ leaves: open, routing: routingWith("acc-first").routing });
    expect(mean.failure?.name).toBe("Averaged a branch instead of adding it");
    expect(first.failure?.name).toBe("Forgot one of the branch's paths");
    expect(mean.angle).not.toBeCloseTo(first.angle, 3);
  });

  it("never claims the loss went up when it did not", () => {
    // The first version of this copy asserted a rising loss. Measured, that is
    // usually false at this learning rate — a swapped multiply once produced a
    // LOWER loss after one step than the correct gradient. The copy must not
    // promise it.
    for (const ruleId of ["add-split", "mul-self", "acc-mean", "square-d"]) {
      const { routing } = routingWith(ruleId);
      const evaluation = evaluate({ leaves: open, routing });
      expect(evaluation.mine!.after).toBeLessThan(evaluation.mine!.before);
      expect(evaluation.failure?.detail).not.toMatch(/loss.{0,12}UP/i);
    }
  });

  it("reports every wrong rule with its own name", () => {
    const names = new Set<string>();
    for (const rule of [...ROUTE_RULES, ...ACCUMULATE_RULES]) {
      if (rule.correct) continue;
      const { routing } = routingWith(rule.id);
      const evaluation = evaluate({ leaves: open, routing });
      if (evaluation.outcome !== "broken") continue;
      expect(evaluation.failure).not.toBeNull();
      expect(evaluation.failure!.detail).toMatch(/\d/);
      expect(evaluation.failure!.detail.length).toBeGreaterThan(150);
      names.add(evaluation.failure!.name);
    }
    // Not one generic message reused: distinct misconceptions, distinct names.
    expect(names.size).toBeGreaterThanOrEqual(7);
    expect(names).not.toContain("Broken chain rule");
  });

  it("blames the earliest wrong step, because a chain fails from the front", () => {
    const routing = perfectRouting();
    const early = STEPS.findIndex((step) =>
      step.options.some((option) => option.id === "square-d"),
    );
    const late = STEPS.findIndex((step) =>
      step.options.some((option) => option.id === "mul-self"),
    );
    expect(early).toBeLessThan(late);
    routing[early] = "square-d";
    routing[late] = "mul-self";
    const evaluation = evaluate({ leaves: open, routing });
    // Two broken links cannot honestly be pinned on one misconception, so this is
    // the case the catalog calls a broken chain rule — but the detail still quotes
    // the EARLIEST wrong step, because that is the one to fix first.
    expect(evaluation.failure?.name).toBe("Broken chain rule");
    expect(evaluation.failure?.detail).toMatch(/g × d/);
    expect(evaluation.failure?.detail).toMatch(/2 of your rules are wrong/);
  });

  it("names the single misconception when only one link is broken", () => {
    const { routing } = routingWith("square-d");
    const evaluation = evaluate({ leaves: open, routing });
    expect(evaluation.failure?.name).toBe(
      "Reused the function instead of its derivative",
    );
  });

  it("separates a wrong rule from a gradient that merely inherited one", () => {
    // Without this split, one mistake near the loss turns the whole list red and
    // the player re-checks a dozen correct decisions hunting for it.
    const { routing } = routingWith("mul-self", "v");
    const score = scoreRouting(open, routing);
    const wrongHere = score.verdicts.filter(
      (verdict) => !verdict.correct && !verdict.inherited,
    );
    const inherited = score.verdicts.filter((verdict) => verdict.inherited);

    // The victims of a bad rule at `v` are its INPUTS — w2 and h — not `v` itself,
    // whose own gradient arrived from `s` and is untouched by what `v` does with it.
    expect(wrongHere.map((verdict) => verdict.id).sort()).toEqual(["h", "w2"]);
    expect(score.verdicts.find((verdict) => verdict.id === "v")?.correct).toBe(
      true,
    );
    // Everything behind h inherits the damage without being anybody's own mistake.
    expect(inherited.map((verdict) => verdict.id)).toContain("w1");
    expect(inherited.length).toBeGreaterThan(0);
  });
});

describe("the coincidence detector", () => {
  it("catches a wrong rule that happens to give the right answer", () => {
    // Measured: with the gate open, "always pass" and "pass if positive" agree, so
    // a player holding the wrong rule scores 100%. Letting that pass silently
    // would certify the misconception.
    const { routing, index } = routingWith("relu-always");
    const evaluation = evaluate({
      leaves: scenarioById("open").leaves,
      routing,
    });
    expect(evaluation.outcome).toBe("cleared");
    expect(evaluation.correctness.fraction).toBe(1);
    expect(evaluation.correctness.coincidences).toContain(index);
    expect(evaluation.correctness.rulesRight).toBe(STEP_COUNT - 1);
  });

  it("and the same rule collapses once the gate is shut", () => {
    const { routing } = routingWith("relu-always");
    const evaluation = evaluate({
      leaves: scenarioById("shut").leaves,
      routing,
    });
    expect(evaluation.outcome).toBe("broken");
    expect(evaluation.failure?.name).toBe(
      "Invented gradient through a shut gate",
    );
  });

  it("reports nothing when every rule is right", () => {
    for (const scenario of SCENARIOS) {
      expect(
        scoreRouting(scenario.leaves, perfectRouting()).coincidences,
      ).toHaveLength(0);
    }
  });

  it("notices that nothing behind a shut gate can be got wrong", () => {
    // A consequence worth knowing rather than a bug: with the gate closed the
    // gradient behind it is zero, so any rule applied to it produces zero. Note
    // this is the multiply at `u`, which sits BEHIND the gate — the same mistake at
    // `v`, in front of it, is caught normally.
    const { routing, index } = routingWith("mul-self", "u");
    const evaluation = evaluate({
      leaves: scenarioById("shut").leaves,
      routing,
    });
    expect(evaluation.correctness.coincidences).toContain(index);
    expect(evaluation.outcome).toBe("cleared");
  });
});

describe("angle and length", () => {
  it("read zero degrees for identical gradients", () => {
    const trace = autograd(scenarioById("open").leaves);
    expect(gradientAngle(trace.grads, trace.grads)).toBeCloseTo(0, 9);
    expect(magnitudeRatio(trace.grads, trace.grads)).toBeCloseTo(1, 9);
  });

  it("read 180 degrees for a negated gradient", () => {
    const trace = autograd(scenarioById("open").leaves);
    const flipped: Values = {};
    for (const id of PARAM_IDS) flipped[id] = -(trace.grads[id] ?? 0);
    expect(gradientAngle(flipped, trace.grads)).toBeCloseTo(180, 6);
    expect(magnitudeRatio(flipped, trace.grads)).toBeCloseTo(1, 9);
  });

  it("call a zero gradient against a real one a right angle, not a match", () => {
    const trace = autograd(scenarioById("open").leaves);
    const zeros: Values = {};
    for (const id of PARAM_IDS) zeros[id] = 0;
    expect(gradientAngle(zeros, trace.grads)).toBe(90);
    expect(gradientAngle(zeros, zeros)).toBe(0);
  });
});

describe("training", () => {
  it("uses the learning rate it advertises", () => {
    const leaves = scenarioById("open").leaves;
    const trace = autograd(leaves);
    const update = applyStep(leaves, trace.grads);
    expect(update.params.w1).toBeCloseTo(
      leaves.w1! - LEARNING_RATE * trace.grads.w1!,
      12,
    );
    // The input and the target are not parameters and must not move.
    expect(update.params.x).toBe(leaves.x);
    expect(update.params.t).toBe(leaves.t);
  });

  it("records one loss per step plus the starting point", () => {
    const run = train(scenarioById("open").leaves, null, 10);
    expect(run.losses).toHaveLength(11);
    expect(run.losses[0]).toBeCloseTo(5.0625, 6);
    expect(run.final).toBeLessThan(run.losses[0]!);
  });

  it("stops rather than reporting NaN when a gradient points uphill", () => {
    const { routing } = routingWith("sub-negate");
    const run = train(scenarioById("open").leaves, routing, TRAINING_STEPS);
    expect(run.diverged).toBe(true);
    expect(Number.isFinite(run.final)).toBe(true);
  });
});

describe("the store", () => {
  beforeEach(() => {
    // reset() deliberately keeps the current scenario, so a block that ran
    // nextScenario() would otherwise leak scenario 2 into the next block.
    useBlitzStore.setState({ clearedIds: [], scenarioIndex: 0 });
    useBlitzStore.getState().reset();
  });

  it("starts on the first scenario with nothing routed", () => {
    const state = useBlitzStore.getState();
    expect(state.scenarioIndex).toBe(0);
    expect(state.routing).toEqual({});
    expect(state.focused).toBe(0);
    expect(state.phase).toBe("routing");
    expect(state.evaluation.correctness.fraction).toBe(0);
  });

  it("records a choice and stays on the step so the result can be read", () => {
    // Not auto-advancing is deliberate: the point of the game is to choose a rule
    // and then look at the gradient it produced against the reference. Moving the
    // panel on would move the answer out of sight, and it also left the radio
    // never settling into a checked state.
    useBlitzStore.getState().choose(0, "square-2d");
    const state = useBlitzStore.getState();
    expect(state.routing[0]).toBe("square-2d");
    expect(state.focused).toBe(0);
    expect(state.evaluation.correctness.fraction).toBeGreaterThan(0);
  });

  it("ignores a rule the step does not offer", () => {
    useBlitzStore.getState().choose(0, "mul-cross");
    expect(useBlitzStore.getState().routing[0]).toBeUndefined();
  });

  it("ignores a step that does not exist", () => {
    useBlitzStore.getState().choose(999, "square-2d");
    expect(Object.keys(useBlitzStore.getState().routing)).toHaveLength(0);
  });

  it("remembers a wrong turn even after the player fixes it", () => {
    // The score is about the walk taken, not the state arrived at. A wrong turn is
    // walking ON with a wrong rule in place — here, Next to step 1 — and coming
    // back to fix it does not un-take it.
    useBlitzStore.getState().choose(0, "square-d");
    useBlitzStore.getState().focusStep(1);
    expect(useBlitzStore.getState().fumbled).toEqual([0]);
    useBlitzStore.getState().focusStep(0);
    useBlitzStore.getState().choose(0, "square-2d");
    expect(useBlitzStore.getState().fumbled).toEqual([0]);
    expect(useBlitzStore.getState().routing[0]).toBe("square-2d");
  });

  it("does not charge a wrong rule that is changed before moving on", () => {
    // A native radio group checks every option the arrow keys pass through, so
    // charging each checked radio would charge keyboard players for moving
    // through the list — and would charge the try-read-change loop the game is
    // built around.
    useBlitzStore.getState().choose(0, "square-d");
    useBlitzStore.getState().choose(0, "square-d2");
    useBlitzStore.getState().choose(0, "square-2d");
    useBlitzStore.getState().focusStep(1);
    expect(useBlitzStore.getState().fumbled).toEqual([]);
  });

  it("does not charge a code-lane search, which tries every rule by design", () => {
    const api = createCodeApi();
    for (const option of STEPS[0]!.options) api.choose(0, option.id);
    expect(useBlitzStore.getState().fumbled).toEqual([]);
  });

  it("lets a wrong gradient keep flowing", () => {
    // Deliberate: the point is that a broken link poisons the subgraph behind it.
    useBlitzStore.getState().choose(0, "square-d");
    useBlitzStore.getState().choose(1, "sub-pass");
    const state = useBlitzStore.getState();
    const verdicts = state.evaluation.correctness.verdicts;
    expect(verdicts.find((verdict) => verdict.id === "d")?.correct).toBe(false);
    expect(verdicts.find((verdict) => verdict.id === "y")?.correct).toBe(false);
    expect(verdicts.find((verdict) => verdict.id === "y")?.inherited).toBe(true);
  });

  it("un-routes a step and reopens the walk", () => {
    for (const [index, id] of Object.entries(perfectRouting())) {
      useBlitzStore.getState().choose(Number(index), id);
    }
    expect(useBlitzStore.getState().phase).toBe("cleared");
    useBlitzStore.getState().clearStep(4);
    const state = useBlitzStore.getState();
    expect(state.routing[4]).toBeUndefined();
    expect(state.phase).toBe("routing");
    expect(state.failure).toBeNull();
    expect(state.focused).toBe(4);
  });

  it("clears a scenario when every gradient matches", () => {
    for (const [index, id] of Object.entries(perfectRouting())) {
      useBlitzStore.getState().choose(Number(index), id);
    }
    const state = useBlitzStore.getState();
    expect(state.phase).toBe("cleared");
    expect(state.failure).toBeNull();
    expect(state.evaluation.correctness.fraction).toBe(1);
    expect(state.clearedIds).toEqual([SCENARIOS[0]!.id]);
    expect(state.evaluation.score).toBeGreaterThan(0.9);
  });

  it("names the failure when the walk finishes wrong", () => {
    const { routing } = routingWith("acc-mean");
    for (const [index, id] of Object.entries(routing)) {
      useBlitzStore.getState().choose(Number(index), id);
    }
    const state = useBlitzStore.getState();
    expect(state.phase).toBe("broken");
    expect(state.failure?.name).toBe("Averaged a branch instead of adding it");
  });

  it("moves to the next scenario and keeps the cleared record", () => {
    for (const [index, id] of Object.entries(perfectRouting())) {
      useBlitzStore.getState().choose(Number(index), id);
    }
    useBlitzStore.getState().nextScenario();
    const state = useBlitzStore.getState();
    expect(state.scenarioIndex).toBe(1);
    expect(state.routing).toEqual({});
    expect(state.clearedIds).toEqual([SCENARIOS[0]!.id]);
  });

  it("clamps the focused step to the walk", () => {
    useBlitzStore.getState().focusStep(-5);
    expect(useBlitzStore.getState().focused).toBe(0);
    useBlitzStore.getState().focusStep(9999);
    expect(useBlitzStore.getState().focused).toBe(STEP_COUNT - 1);
  });

  it("keeps the truth hidden until asked", () => {
    expect(useBlitzStore.getState().showTruth).toBe(false);
    useBlitzStore.getState().toggleTruth();
    expect(useBlitzStore.getState().showTruth).toBe(true);
    expect(useBlitzStore.getState().whyCard?.title).toMatch(/real trace/i);
  });
});

describe("derived views", () => {
  it("reports one edge view per graph edge, unrouted at first", () => {
    const mine = replay(scenarioById("open").leaves, {});
    const truth = autograd(scenarioById("open").leaves);
    const views = edgeViews(mine.edges, truth.edges);
    const edgeCount = NODES.reduce((total, node) => total + node.inputs.length, 0);
    expect(views).toHaveLength(edgeCount);
    expect(views.every((view) => !view.routed)).toBe(true);
  });

  it("marks edges correct once they carry the right gradient", () => {
    const leaves = scenarioById("open").leaves;
    const mine = replay(leaves, perfectRouting());
    const truth = autograd(leaves);
    const views = edgeViews(mine.edges, truth.edges);
    // Every edge except the one into the target, which is never routed for use.
    expect(views.filter((view) => view.routed).every((view) => view.correct)).toBe(
      true,
    );
  });
});

describe("the code lane api", () => {
  beforeEach(() => {
    // reset() deliberately keeps the current scenario, so a block that ran
    // nextScenario() would otherwise leak scenario 2 into the next block.
    useBlitzStore.setState({ clearedIds: [], scenarioIndex: 0 });
    useBlitzStore.getState().reset();
  });

  it("writes the same state the radio buttons write", () => {
    const api = createCodeApi();
    api.choose(0, "square-2d");
    expect(useBlitzStore.getState().routing[0]).toBe("square-2d");
    expect(api.steps()[0]!.chosen).toBe("square-2d");
  });

  it("rejects an unknown step or rule, by name", () => {
    const api = createCodeApi();
    expect(() => api.choose(99, "square-2d")).toThrow(/No step 99/);
    expect(() => api.choose(0, "mul-cross")).toThrow(/does not offer/);
  });

  it("hands over the reference trace", () => {
    const api = createCodeApi();
    const truth = api.autograd();
    expect(truth.grads.w1).toBeCloseTo(13.5, 9);
    expect(truth.values.L).toBeCloseTo(5.0625, 9);
  });

  it("reports correctness, angle and coincidences", () => {
    const api = createCodeApi();
    for (const [index, id] of Object.entries(routingWith("relu-always").routing)) {
      api.choose(Number(index), id);
    }
    const report = api.correctness();
    expect(report.fraction).toBe(1);
    expect(report.coincidences).toHaveLength(1);
    expect(report.angle).toBeCloseTo(0, 6);
    expect(report.wrong).toHaveLength(0);
  });

  it("trains with the player's rules or with the real ones", () => {
    const api = createCodeApi();
    for (const [index, id] of Object.entries(routingWith("add-split").routing)) {
      api.choose(Number(index), id);
    }
    const mine = api.train({ steps: 20 });
    const truth = api.train({ useTruth: true, steps: 20 });
    expect(mine.losses).toHaveLength(21);
    expect(mine.final).not.toBeCloseTo(truth.final, 6);
  });

  it("guards the training budget", () => {
    const api = createCodeApi();
    expect(() => api.train({ steps: 0 })).toThrow(/between 1 and 5000/);
    expect(() => api.train({ steps: 99999 })).toThrow(/between 1 and 5000/);
  });

  it("applies one rule everywhere it is offered", () => {
    const api = createCodeApi();
    const applied = api.chooseAll("add-copy");
    // Three add nodes in the graph: z, s and y.
    expect(applied).toBe(3);
  });

  it("hands back copies, not live state", () => {
    const api = createCodeApi();
    const scenarioCopy = api.scenario();
    scenarioCopy.leaves.w1 = 9999;
    expect(useBlitzStore.getState().evaluation).toBeDefined();
    expect(SCENARIOS[0]!.leaves.w1).not.toBe(9999);
  });
});

describe("why-cards", () => {
  it("opens by saying wrong answers are allowed to flow", () => {
    const card = whyCardFor({ kind: "briefing", scenario: SCENARIOS[0]! });
    expect(card.body).toMatch(/poisons everything behind it/);
  });

  it("explains a correct rule rather than just approving it", () => {
    const step = STEPS[2]!;
    const rule = step.options.find((option) => option.correct)!;
    const routing = { 0: "square-2d", 1: "sub-pass", 2: rule.id };
    const card = whyCardFor({
      kind: "chose",
      step,
      stepIndex: 2,
      rule,
      evaluation: evaluate({ leaves: SCENARIOS[0]!.leaves, routing }),
      scenario: SCENARIOS[0]!,
      leaves: SCENARIOS[0]!.leaves,
      routing,
    });
    expect(card.tone).toBe("good");
    expect(card.body).toMatch(/copied, never divided|local derivatives/);
  });

  it("tells a player who cleared by coincidence that they did", () => {
    const { routing, index } = routingWith("relu-always");
    const evaluation = evaluate({ leaves: SCENARIOS[0]!.leaves, routing });
    const card = whyCardFor({
      kind: "chose",
      step: STEPS[index]!,
      stepIndex: index,
      rule: ruleById("relu-always")!,
      evaluation,
      scenario: SCENARIOS[0]!,
      leaves: SCENARIOS[0]!.leaves,
      routing,
    });
    expect(card.title).toMatch(/1 of your rules is wrong/);
    expect(card.body).toMatch(/works for the wrong reason/);
    // And it is the open-gate story, because here the gate IS open.
    expect(card.body).toMatch(/gate is open here \(z = 2\.00\)/);
  });

  it("explains a shut-gate coincidence as a shut gate, not an open one", () => {
    // The card used to tell the open-gate story to every coincidence. In the shut
    // scenario three wrong rules clear together — "always block" at h, and two
    // rules behind it that only ever see a zero — and none of that is an open gate.
    const shut = scenarioById("shut");
    const routing = perfectRouting();
    const at = (ruleId: string, node: string) =>
      STEPS.findIndex(
        (step) =>
          step.nodeId === node && step.options.some((option) => option.id === ruleId),
      );
    routing[at("relu-never", "h")] = "relu-never";
    routing[at("add-split", "z")] = "add-split";
    routing[at("mul-self", "u")] = "mul-self";

    const evaluation = evaluate({ leaves: shut.leaves, routing });
    expect(evaluation.outcome).toBe("cleared");
    expect(evaluation.correctness.coincidences).toHaveLength(3);

    const index = at("mul-self", "u");
    const card = whyCardFor({
      kind: "chose",
      step: STEPS[index]!,
      stepIndex: index,
      rule: ruleById("mul-self")!,
      evaluation,
      scenario: shut,
      leaves: shut.leaves,
      routing,
    });
    expect(card.title).toMatch(/3 of your rules are wrong/);
    expect(card.body).not.toMatch(/gate (is )?open/i);
    expect(card.body).toMatch(/gate is shut here \(z = -2\.00\)/);
    expect(card.body).toMatch(/Nothing arrives at z/);
    expect(card.body).toMatch(/Nothing arrives at u/);
    expect(card.body).toMatch(/any rule applied to zero gives zero/);
    // Every coincident step is named by the rule the player actually holds.
    for (const ruleId of ["relu-never", "add-split", "mul-self"]) {
      expect(card.body).toContain(`"${ruleById(ruleId)!.label}"`);
    }
  });

  it("links a clean clear to the gradient-descent concept it feeds", () => {
    const evaluation = evaluate({
      leaves: SCENARIOS[0]!.leaves,
      routing: perfectRouting(),
    });
    const index = STEP_COUNT - 1;
    const card = whyCardFor({
      kind: "chose",
      step: STEPS[index]!,
      stepIndex: index,
      rule: STEPS[index]!.options.find((option) => option.correct)!,
      evaluation,
      scenario: SCENARIOS[0]!,
      leaves: SCENARIOS[0]!.leaves,
      routing: perfectRouting(),
    });
    expect(card.conceptHref).toBe("/concepts/gradient-descent");
    expect(card.conceptLabel).toBe("How gradient descent works");
  });

  it("states the cost of a wrong turn before the player takes one", () => {
    const { index } = routingWith("square-d");
    const early: Routing = { [index]: "square-d" };
    const card = whyCardFor({
      kind: "chose",
      step: STEPS[index]!,
      stepIndex: index,
      rule: ruleById("square-d")!,
      evaluation: evaluate({ leaves: SCENARIOS[0]!.leaves, routing: early }),
      scenario: SCENARIOS[0]!,
      leaves: SCENARIOS[0]!.leaves,
      routing: early,
    });
    expect(card.body).toMatch(/counts as a wrong turn/);
    expect(card.body).toMatch(/changing it before you move on costs nothing/i);
  });

  it("prices carrying on at what it really costs, and offers the way back", () => {
    // The card used to recommend "carry on and watch, then come back" — the walk
    // that is charged — and call the cost "a little score" when it was a star.
    const { index } = routingWith("square-d");
    const early: Routing = { [index]: "square-d" };
    const { body } = whyCardFor({
      kind: "chose",
      step: STEPS[index]!,
      stepIndex: index,
      rule: ruleById("square-d")!,
      evaluation: evaluate({ leaves: SCENARIOS[0]!.leaves, routing: early }),
      scenario: SCENARIOS[0]!,
      leaves: SCENARIOS[0]!.leaves,
      routing: early,
    });
    expect(body).toContain(`${Math.round(WRONG_TURN_COST * 100)}% off this round's score`);
    expect(body).not.toMatch(/a little score/);
    expect(body).not.toMatch(/then come back and change it/);
    expect(body).toMatch(/Start this scenario again/);

    // The star claim is made only because the scoring makes it true.
    const firstClear = (wrongTurns: number) =>
      roundScore({ routingScore: 1, wrongTurns, peeked: false, clearedCount: 1 });
    expect(firstClear(0)).toBeGreaterThanOrEqual(HIGH_SCORE_THRESHOLD);
    expect(firstClear(1)).toBeLessThan(HIGH_SCORE_THRESHOLD);
    expect(body).toMatch(/on a first clear, one is enough to miss the two-star line/);
    expect(wrongTurnPrice()).toBe(body.match(/each one takes [^.]*/)![0]);
  });

  it("means it when it says a restart wipes the wrong turns", () => {
    useBlitzStore.setState({ clearedIds: [], peekedIds: [], scenarioIndex: 0 });
    useBlitzStore.getState().reset();
    useBlitzStore.getState().choose(0, "square-d");
    useBlitzStore.getState().focusStep(1);
    expect(useBlitzStore.getState().fumbled).toEqual([0]);
    useBlitzStore.getState().reset();
    expect(useBlitzStore.getState().fumbled).toEqual([]);
  });

  it("describes the meter honestly at each stage", () => {
    expect(meterCaption(0, 0, 0)).toMatch(/derivative of anything with respect to itself/);
    expect(meterCaption(9, 1, 0)).toMatch(/Every gradient/);
    expect(meterCaption(9, 1, 1)).toMatch(/got away with it/);
    expect(meterCaption(4, 0.4, 0)).toMatch(/only the first one's fault/);
  });
});

// ── The copy has to agree with the numbers it sits next to ────────────────

describe("scenario copy", () => {
  it("calls the third scenario an undershoot, because d is negative", () => {
    // It used to be "Overshooting the target… d is positive", beside a graph
    // showing y = 0, t = 3 and d = -3.
    const scenario = scenarioById("negative");
    const values = forward(scenario.leaves);
    expect(values.y).toBeLessThan(values.t!);
    expect(values.d).toBeLessThan(0);
    expect(scenario.title).toMatch(/Undershooting/);
    expect(scenario.lesson).toMatch(/below the target, so d is negative/);
    expect(scenario.lesson).not.toMatch(/positive|above the target/);
  });

  it("agrees with sign(d) in every scenario that names one", () => {
    for (const scenario of SCENARIOS) {
      const d = forward(scenario.leaves).d!;
      if (/below the target|d is negative/.test(scenario.lesson)) {
        expect(d, scenario.id).toBeLessThan(0);
      }
      if (/above the target|d is positive/.test(scenario.lesson)) {
        expect(d, scenario.id).toBeGreaterThan(0);
      }
    }
  });

  it("is right that the gradients near the output flip sign against scenario 1", () => {
    const open = autograd(scenarioById("open").leaves).grads;
    const under = autograd(scenarioById("negative").leaves).grads;
    for (const id of ["d", "y", "s", "v", "b2"]) {
      expect(Math.sign(open[id]!), id).toBe(1);
      expect(Math.sign(under[id]!), id).toBe(-1);
    }
  });

  it("is right that the two gradients at h have opposite signs and partly cancel", () => {
    const leaves = scenarioById("negative").leaves;
    const trace = autograd(leaves);
    expect(leaves.w2).toBeLessThan(0);
    const viaV = trace.edges[edgeKey("v", "h")]!;
    const viaS = trace.edges[edgeKey("s", "h")]!;
    expect(Math.sign(viaV)).toBe(-Math.sign(viaS));
    const total = trace.grads.h!;
    expect(total).toBeCloseTo(viaV + viaS, 12);
    expect(Math.abs(total)).toBeLessThan(
      Math.min(Math.abs(viaV), Math.abs(viaS)),
    );
  });

  it("keeps the first branch gradient to arrive from s, as the rule's text says", () => {
    // "keep the first to arrive" keeps the skip connection's contribution.
    const leaves = scenarioById("open").leaves;
    const { routing } = routingWith("acc-first");
    const mine = replay(leaves, routing);
    expect(mine.grads.h).toBeCloseTo(mine.edges[edgeKey("s", "h")]!, 12);
    expect(ruleById("acc-first")!.detail).toMatch(/skip connection, from s/);
  });
});

describe("the options are not an answer key by position", () => {
  const pickAt = (scenarioId: string, position: number): Routing => {
    const routing: Routing = {};
    STEPS.forEach((step, index) => {
      const shown = displayOrder(step.options, scenarioId, index);
      routing[index] = shown[Math.min(position, shown.length - 1)]!.id;
    });
    return routing;
  };

  it("never clears a scenario by picking the same position everywhere", () => {
    // Before the shuffle the correct rule was always listed first, and "top radio
    // every time" cleared every scenario at 100%.
    for (const scenario of SCENARIOS) {
      for (let position = 0; position < 4; position += 1) {
        const evaluation = evaluate({
          leaves: scenario.leaves,
          routing: pickAt(scenario.id, position),
        });
        expect(evaluation.outcome, `${scenario.id} @${position}`).not.toBe(
          "cleared",
        );
      }
    }
  });

  it("moves the correct rule around within every scenario", () => {
    for (const scenario of SCENARIOS) {
      const positions = STEPS.map((step, index) =>
        displayOrder(step.options, scenario.id, index).findIndex(
          (rule) => rule.correct,
        ),
      );
      expect(new Set(positions).size, scenario.id).toBeGreaterThan(1);
      expect(
        positions.some((position) => position !== 0),
        scenario.id,
      ).toBe(true);
    }
  });

  it("is a stable permutation, so nothing moves under a keyboard user", () => {
    for (const scenario of SCENARIOS) {
      STEPS.forEach((step, index) => {
        const once = displayOrder(step.options, scenario.id, index).map(
          (rule) => rule.id,
        );
        const again = displayOrder(step.options, scenario.id, index).map(
          (rule) => rule.id,
        );
        expect(again).toEqual(once);
        expect([...once].sort()).toEqual(
          step.options.map((rule) => rule.id).sort(),
        );
      });
    }
  });

  it("leaves the code lane's order alone, so greedy ties still favour the real rule", () => {
    const api = createCodeApi();
    api.steps().forEach((step, index) => {
      expect(step.options.map((option) => option.id)).toEqual(
        STEPS[index]!.options.map((option) => option.id),
      );
    });
  });
});

describe("skipping a step does not settle it for the player", () => {
  it("leaves h unsettled when the accumulate step is unanswered", () => {
    // replay() used to sum the two branch gradients on the player's behalf when
    // they routed h without answering the accumulate step, so the meter read
    // 100% without the one decision this graph exists to ask about.
    const leaves = scenarioById("open").leaves;
    const routing = perfectRouting();
    const accumulate = STEPS.findIndex((step) => step.kind === "accumulate");
    delete routing[accumulate];

    const evaluation = evaluate({ leaves, routing });
    expect(evaluation.outcome).toBe("routing");
    expect(evaluation.nextStep).toBe(accumulate);
    expect(evaluation.correctness.fraction).toBeLessThan(1);
    const h = evaluation.correctness.verdicts.find(
      (verdict) => verdict.id === "h",
    )!;
    expect(h.mine).toBeNull();
    // And nothing behind h has a number the player did not produce.
    const settled = replay(leaves, routing).settled;
    for (const id of ["z", "u", "b1", "w1"]) {
      expect(settled, id).not.toContain(id);
    }
  });

  it("does not invent a zero for a node whose consumer is unrouted", () => {
    const leaves = scenarioById("open").leaves;
    // Route y (step 2) without d (step 1).
    const mine = replay(leaves, { 0: "square-2d", 2: "add-copy" });
    expect(mine.settled).not.toContain("y");
    expect(mine.edges[edgeKey("y", "s")]).toBeUndefined();
  });
});

describe("what 40 steps of training is allowed to claim", () => {
  const sweep = () =>
    SCENARIOS.flatMap((scenario) =>
      STEPS.flatMap((step, index) =>
        step.options
          .filter((option) => !option.correct)
          .map((option) => {
            const routing = { ...perfectRouting(), [index]: option.id };
            return {
              label: `${scenario.id}/${step.nodeId}/${option.id}`,
              evaluation: evaluate({ leaves: scenario.leaves, routing }),
            };
          }),
      ),
    ).filter(({ evaluation }) => evaluation.outcome === "broken");

  it("never presents a lower final loss as the cost of a wrong gradient", () => {
    // Measured: in nine single-wrong-rule routings the wrong gradient ends BELOW
    // the true one after 40 steps, and the copy used to print that under "the two
    // paths separate". Wherever that happens it now says so, and why.
    let lower = 0;
    for (const { label, evaluation } of sweep()) {
      const detail = evaluation.failure!.detail;
      const comparison = compareRuns(evaluation.mineRun!, evaluation.truthRun!);
      if (comparison === "lower") {
        lower += 1;
        expect(detail, label).toMatch(/ends LOWER/);
        expect(detail, label).toMatch(/not a sign the rule is right/);
      }
      if (comparison !== "higher") {
        expect(detail, label).not.toMatch(/the two paths separate/);
      }
    }
    expect(lower).toBeGreaterThan(0);
  });

  it("only says the first step raised the loss when it did", () => {
    for (const { label, evaluation } of sweep()) {
      const rose = evaluation.mine!.after > evaluation.mine!.before;
      const detail = evaluation.failure!.detail;
      if (/raises the loss/.test(detail)) expect(rose, label).toBe(true);
      if (/one step lowers the loss anyway/.test(detail)) {
        expect(rose, label).toBe(false);
      }
    }
  });

  it("names the dead gate when that is why the real run is slower", () => {
    // Shut scenario, "always pass" at h: the true derivative through a shut gate is
    // zero, so the real run can never move w1 or b1, and the invented gradient can.
    const { routing } = routingWith("relu-always");
    const evaluation = evaluate({
      leaves: scenarioById("shut").leaves,
      routing,
    });
    expect(compareRuns(evaluation.mineRun!, evaluation.truthRun!)).toBe("lower");
    expect(evaluation.failure!.detail).toMatch(/a dead unit learns nothing/);
  });

  it("backs the notes' claim that the '=' bug can end below the real gradient", () => {
    // MATH_NOTES: "the final loss can even come out lower than the real
    // gradient's". Keeping only the first branch gradient is the '=' bug.
    expect(MATH_NOTES).toMatch(/can even come out lower/);
    expect(MATH_NOTES).not.toMatch(/trains worse than it should/);
    const { routing } = routingWith("acc-first");
    const evaluation = evaluate({
      leaves: scenarioById("negative").leaves,
      routing,
    });
    expect(evaluation.mineRun!.final).toBeLessThan(evaluation.truthRun!.final);
  });
});

// ── What a clear is worth ────────────────────────────────────────────────

/** Drive the store through a routing, in walk order, as the visual lane would. */
function walk(routing: Routing) {
  STEPS.forEach((_, index) => {
    const id = routing[index];
    if (id === undefined) return;
    useBlitzStore.getState().focusStep(index);
    useBlitzStore.getState().choose(index, id);
  });
}

function resetProgression() {
  useProgression.getState().setAdapter(createMemoryAdapter());
  useProgression.setState({ xp: 0, games: {}, badges: [], lastGain: null });
  useBlitzStore.setState({ clearedIds: [], peekedIds: [], scenarioIndex: 0 });
  useBlitzStore.getState().reset();
}

describe("scoring and progression", () => {
  beforeEach(resetProgression);

  const progress = () => useProgression.getState().games[SLUG];

  it("gives a flawless first clear two stars, not one", () => {
    // The campaign multiplier used to land a perfect first clear on
    // 0.7999999999999999 — one ulp under the 0.8 line.
    walk(perfectRouting());
    expect(useBlitzStore.getState().phase).toBe("cleared");
    expect(progress()?.bestScore).toBeGreaterThanOrEqual(HIGH_SCORE_THRESHOLD);
    expect(progress()?.stars).toBe(2);
    expect(progress()?.codeLaneCleared).toBe(false);
  });

  it("gives a code-lane clear the third star", () => {
    const api = createCodeApi();
    for (const [index, id] of Object.entries(perfectRouting())) {
      api.choose(Number(index), id);
    }
    expect(progress()?.stars).toBe(3);
    expect(progress()?.codeLaneCleared).toBe(true);
  });

  it("does not give the code-lane star for a clear made with the buttons", () => {
    // Attribution follows the action, not whichever tab is on screen.
    useBlitzStore.getState().setLane("code");
    walk(perfectRouting());
    expect(progress()?.codeLaneCleared).toBe(false);
    expect(progress()?.stars).toBe(2);
  });

  it("charges a wrong turn, and one on a first clear costs the second star", () => {
    const clean = roundScore({
      routingScore: 1,
      wrongTurns: 0,
      peeked: false,
      clearedCount: 1,
    });

    // Walk on from a wrong rule at L, come back, fix it, finish cleanly.
    useBlitzStore.getState().choose(0, "square-d");
    useBlitzStore.getState().focusStep(1);
    useBlitzStore.getState().focusStep(0);
    walk(perfectRouting());
    expect(useBlitzStore.getState().phase).toBe("cleared");
    expect(useBlitzStore.getState().fumbled).toEqual([0]);

    const recorded = progress()!.bestScore;
    expect(recorded).toBeLessThan(clean);
    expect(recorded).toBeLessThan(HIGH_SCORE_THRESHOLD);
    expect(progress()?.stars).toBe(1);
  });

  it("charges revealing the trace before the clear, and a restart does not launder it", () => {
    useBlitzStore.getState().toggleTruth();
    useBlitzStore.getState().reset();
    expect(useBlitzStore.getState().showTruth).toBe(false);
    walk(perfectRouting());
    const peeked = roundScore({
      routingScore: 1,
      wrongTurns: 0,
      peeked: true,
      clearedCount: 1,
    });
    const clean = roundScore({
      routingScore: 1,
      wrongTurns: 0,
      peeked: false,
      clearedCount: 1,
    });
    expect(progress()?.bestScore).toBeCloseTo(peeked, 9);
    expect(peeked).toBeCloseTo(clean * PEEK_FACTOR, 5);
    expect(progress()?.stars).toBe(1);
  });

  it("does not charge reading the trace after the clear", () => {
    walk(perfectRouting());
    useBlitzStore.getState().toggleTruth();
    expect(useBlitzStore.getState().peekedIds).toEqual([]);
  });

  it("records a round once, however many times it is re-cleared", () => {
    walk(perfectRouting());
    expect(progress()?.playCount).toBe(1);
    // Break it and put it back: the same round, cleared again at the same score.
    useBlitzStore.getState().choose(0, "square-d");
    useBlitzStore.getState().choose(0, "square-2d");
    expect(useBlitzStore.getState().phase).toBe("cleared");
    expect(progress()?.playCount).toBe(1);
    // Choosing what is already chosen is not a new clear either.
    useBlitzStore.getState().choose(0, "square-2d");
    expect(progress()?.playCount).toBe(1);
  });

  it("gives the third star to code that re-asserts a routing the buttons cleared", () => {
    // Choosing what is already chosen used to return before the recording
    // decision, so the code lane's first clear of an already-cleared round was
    // dropped without a word and the third star never came.
    walk(perfectRouting());
    expect(progress()?.stars).toBe(2);
    expect(progress()?.playCount).toBe(1);

    const api = createCodeApi();
    const reassert = () => {
      for (const [index, id] of Object.entries(perfectRouting())) {
        api.choose(Number(index), id);
      }
    };
    reassert();
    expect(progress()?.codeLaneCleared).toBe(true);
    expect(progress()?.stars).toBe(3);
    expect(progress()?.playCount).toBe(2);
    expect(useBlitzStore.getState().recorded).toEqual({
      score: progress()!.bestScore,
      code: true,
    });

    // Once is enough: running the snippet again is the same code-lane clear.
    reassert();
    expect(progress()?.playCount).toBe(2);
  });

  it("does not re-record a visual re-choose of the cleared routing", () => {
    walk(perfectRouting());
    walk(perfectRouting());
    expect(progress()?.playCount).toBe(1);
    expect(progress()?.codeLaneCleared).toBe(false);
  });

  it("does record a better clear of the same round", () => {
    // A coincidence clear first (a wrong rule standing), then the fix.
    const { routing, index } = routingWith("relu-always");
    expect(
      evaluate({ leaves: scenarioById("open").leaves, routing }).outcome,
    ).toBe("cleared");
    STEPS.forEach((_, step) =>
      useBlitzStore.getState().choose(step, routing[step]!),
    );
    const first = progress()!.bestScore;
    useBlitzStore.getState().choose(index, "relu-gate");
    expect(progress()!.bestScore).toBeGreaterThan(first);
    expect(progress()?.playCount).toBe(2);
  });

  it("keeps an exact 0.8 an exact 0.8", () => {
    // Two wrong turns on a full campaign is 1 × 0.8 × 1 — which must not become
    // 0.7999999999999999 on the way to progression.
    expect(
      roundScore({ routingScore: 1, wrongTurns: 2, peeked: false, clearedCount: 3 }),
    ).toBe(0.8);
  });
});

describe("the starter snippet", () => {
  beforeEach(resetProgression);

  it("runs against the real api and records its clear once per run", async () => {
    const logs: string[] = [];
    await createJsExecutor<ReturnType<typeof createCodeApi>>()(STARTER_CODE, {
      api: createCodeApi(),
      log: (...args: unknown[]) => logs.push(args.map(String).join(" ")),
      checkBudget: () => {},
    });
    expect(logs.join("\n")).toMatch(/brute-forced routing: 100%/);
    expect(useBlitzStore.getState().evaluation.outcome).toBe("cleared");
    // It walks cleared → broken → cleared a dozen times pricing each wrong rule;
    // progression hears about the round once.
    expect(useProgression.getState().games[SLUG]?.playCount).toBe(1);
    expect(useProgression.getState().games[SLUG]?.codeLaneCleared).toBe(true);
  }, 60000);
});

describe("the code lane rejects bad input by name", () => {
  beforeEach(() => {
    useBlitzStore.setState({ clearedIds: [], scenarioIndex: 0 });
    useBlitzStore.getState().reset();
  });

  it("rejects a step that is not an integer", () => {
    const api = createCodeApi();
    expect(() => api.choose(1.5, "sub-pass")).toThrow(/No step 1\.5/);
    expect(() => api.choose(Number.NaN, "sub-pass")).toThrow(/No step NaN/);
    expect(() => api.choose("1" as unknown as number, "sub-pass")).toThrow(
      /No step 1\./,
    );
  });

  it("rejects an unknown rule in chooseAll instead of applying it nowhere", () => {
    const api = createCodeApi();
    expect(() => api.chooseAll("add-copyy")).toThrow(/No rule "add-copyy"/);
  });
});

describe("coincidence reasons", () => {
  it("say nothing about a step that does not exist", () => {
    expect(coincidenceReason(999, {}, scenarioById("open").leaves)).toBeNull();
  });

  it("name the mechanism that actually applies in each scenario", () => {
    const at = (ruleId: string) => routingWith(ruleId);
    const open = at("relu-always");
    expect(
      coincidenceReason(open.index, open.routing, scenarioById("open").leaves)
        ?.mechanism,
    ).toBe("gate-open");
    const shut = at("relu-never");
    expect(
      coincidenceReason(shut.index, shut.routing, scenarioById("shut").leaves)
        ?.mechanism,
    ).toBe("gate-shut");
    const behind = routingWith("mul-self", "u");
    expect(
      coincidenceReason(behind.index, behind.routing, scenarioById("shut").leaves)
        ?.mechanism,
    ).toBe("zero-input");
  });
});
