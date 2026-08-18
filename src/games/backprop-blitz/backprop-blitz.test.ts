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
  applyStep,
  autograd,
  consumersOf,
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
import { createCodeApi, edgeViews, useBlitzStore } from "./store";
import { meterCaption, whyCardFor } from "./why-cards";

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

  it("remembers a fumble even after the player fixes it", () => {
    // The score is about the walk taken, not the state arrived at.
    useBlitzStore.getState().choose(0, "square-d");
    expect(useBlitzStore.getState().fumbled).toEqual([0]);
    useBlitzStore.getState().choose(0, "square-2d");
    expect(useBlitzStore.getState().fumbled).toEqual([0]);
    expect(useBlitzStore.getState().routing[0]).toBe("square-2d");
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
    });
    expect(card.title).toMatch(/rules is wrong/);
    expect(card.body).toMatch(/works for the wrong reason/);
  });

  it("describes the meter honestly at each stage", () => {
    expect(meterCaption(0, 0, 0)).toMatch(/derivative of anything with respect to itself/);
    expect(meterCaption(9, 1, 0)).toMatch(/Every gradient/);
    expect(meterCaption(9, 1, 1)).toMatch(/got away with it/);
    expect(meterCaption(4, 0.4, 0)).toMatch(/only the first one's fault/);
  });
});
