import type { NamedFailure } from "@/engine/types";
import { clamp } from "@/lib/utils";

/**
 * Backprop Blitz — the chain rule, executed by hand.
 *
 * The player is handed a real computation graph with a real forward pass, and
 * pushes the error backward through it one node at a time. At each node they
 * choose how the incoming gradient transforms; at each place a value was used
 * twice they choose how the two arriving gradients combine. Their choices produce
 * actual numbers, which are diffed against a real reverse-mode autograd trace.
 *
 * ── Why the wrong answer is allowed to propagate ────────────────────────────
 * A quiz would mark each choice and move on. This does not: a wrong routing rule
 * keeps flowing, and every node upstream of it inherits the corruption. That is
 * the single most useful thing to know about backpropagation — a broken link does
 * not produce a locally wrong number, it poisons the entire subgraph behind it —
 * and it is only teachable if the game lets it happen.
 *
 * ── Why not TF.js ───────────────────────────────────────────────────────────
 * The ground truth this game needs is the gradient at every INTERMEDIATE node, and
 * that is exactly what a tensor library hides: `tf.grads` hands back gradients with
 * respect to the inputs you asked about and throws the trace away. So the autograd
 * here is written out — about forty lines of reverse-mode accumulation — which also
 * makes it inspectable, which a game about how autograd works probably owes the
 * reader. The tests check its parameter gradients against `tf.grads` on the same
 * expression, so "real autograd" is a verified claim rather than a description.
 *
 * ── Deviations from the spec, flagged ───────────────────────────────────────
 * 1. `<ComputeGraph>` is SVG, not p5.js/WebGL. The graph is fourteen nodes with
 *    numbers on every edge, and those numbers ARE the content — a canvas would put
 *    them outside the accessibility tree and require building the whole view twice.
 *    Same decision as Overfit Tower Defense and Agent Academy, same reason.
 * 2. The spec's data model has `Edge { from, to, weight }`. Weights here are
 *    nodes, not edge attributes, because the entire lesson is that a weight
 *    RECEIVES a gradient. An attribute on an edge cannot be a differentiation
 *    target in the same uniform way, and making weights first-class nodes is also
 *    how real autograd engines represent them.
 */

// ── The graph ─────────────────────────────────────────────────────────────

export type Op =
  | "input"
  | "param"
  | "target"
  | "mul"
  | "add"
  | "sub"
  | "relu"
  | "square";

export interface NodeSpec {
  id: string;
  op: Op;
  /** Producer node ids, in order. Empty for leaves. */
  inputs: string[];
  /** Display label, e.g. "w₁" or "h". */
  label: string;
  /** What this node is, in words. */
  blurb: string;
  /** Layout, in graph units. */
  x: number;
  y: number;
}

/**
 * A tiny residual block and a squared error.
 *
 *   u = w1 · x        z = u + b1       h = relu(z)
 *   v = w2 · h        s = v + h        y = s + b2
 *   d = y − t         L = d²
 *
 * The `s = v + h` line is the reason this graph exists rather than a plain chain.
 * It makes `h` feed two consumers, so its gradient is the SUM of two arriving
 * contributions — the multivariable chain rule, which is the step everyone
 * either forgets or averages. It is also a residual connection, so the shape is
 * one every modern network actually contains.
 */
const GRAPH: readonly NodeSpec[] = [
  { id: "x", op: "input", inputs: [], label: "x", blurb: "The input.", x: 0, y: 0 },
  { id: "w1", op: "param", inputs: [], label: "w₁", blurb: "First weight.", x: 0, y: 1 },
  { id: "b1", op: "param", inputs: [], label: "b₁", blurb: "First bias.", x: 0, y: 2 },
  { id: "w2", op: "param", inputs: [], label: "w₂", blurb: "Second weight.", x: 0, y: 3 },
  { id: "b2", op: "param", inputs: [], label: "b₂", blurb: "Output bias.", x: 0, y: 4 },
  { id: "t", op: "target", inputs: [], label: "t", blurb: "The target. A constant.", x: 0, y: 5 },

  { id: "u", op: "mul", inputs: ["w1", "x"], label: "u", blurb: "w₁ times x.", x: 1, y: 0.5 },
  { id: "z", op: "add", inputs: ["u", "b1"], label: "z", blurb: "u plus the bias.", x: 2, y: 1 },
  { id: "h", op: "relu", inputs: ["z"], label: "h", blurb: "The activation. Used twice.", x: 3, y: 1 },
  { id: "v", op: "mul", inputs: ["w2", "h"], label: "v", blurb: "w₂ times h.", x: 4, y: 2 },
  { id: "s", op: "add", inputs: ["v", "h"], label: "s", blurb: "v plus h — the skip connection.", x: 5, y: 1.5 },
  { id: "y", op: "add", inputs: ["s", "b2"], label: "y", blurb: "The prediction.", x: 6, y: 2.5 },
  { id: "d", op: "sub", inputs: ["y", "t"], label: "d", blurb: "Prediction minus target.", x: 7, y: 3 },
  { id: "L", op: "square", inputs: ["d"], label: "L", blurb: "Squared error. The loss.", x: 8, y: 3 },
] as const;

export const NODES = GRAPH;
export const nodeById = (id: string): NodeSpec =>
  GRAPH.find((node) => node.id === id) ??
  (() => {
    throw new Error(`no node "${id}"`);
  })();

export const isLeaf = (node: NodeSpec): boolean => node.inputs.length === 0;
/** Leaves that learn. `x` and `t` are given, not learned. */
export const PARAM_IDS = GRAPH.filter((node) => node.op === "param").map(
  (node) => node.id,
);
/** Nodes that never receive a gradient, because nothing is optimising them. */
export const NO_GRADIENT_IDS = ["t"];

/** Which nodes consume this one. */
export function consumersOf(id: string): string[] {
  return GRAPH.filter((node) => node.inputs.includes(id)).map((node) => node.id);
}

/** Fan-out: how many consumers read this node's value. */
export const fanOut = (id: string): number => consumersOf(id).length;

/** Producers before consumers. */
export function topologicalOrder(): string[] {
  const order: string[] = [];
  const seen = new Set<string>();
  const visit = (id: string) => {
    if (seen.has(id)) return;
    seen.add(id);
    for (const input of nodeById(id).inputs) visit(input);
    order.push(id);
  };
  for (const node of GRAPH) visit(node.id);
  return order;
}

// ── Scenarios ─────────────────────────────────────────────────────────────

export interface Scenario {
  id: string;
  title: string;
  /** Values for every leaf. */
  leaves: Record<string, number>;
  /** What this arrangement is here to teach. */
  lesson: string;
}

/**
 * Three arrangements of the same graph.
 *
 * The numbers are chosen so that every wrong routing rule produces a VISIBLY
 * different answer. That matters more than it sounds: with `w₂ = 1` a swapped
 * multiply is invisible, and with `b₁ = 0` a halved add looks plausible. Each
 * scenario was checked to make sure the canonical mistakes are distinguishable
 * from the truth by eye.
 */
export const SCENARIOS: readonly Scenario[] = [
  {
    id: "open",
    title: "The gate is open",
    leaves: { x: 2, w1: 1.5, b1: -1, w2: 0.5, b2: 0.25, t: 1 },
    lesson:
      "Every rule in the game is exercised here except a shut gate: a square, two multiplies, three adds, and one value used twice.",
  },
  {
    id: "shut",
    title: "The gate is shut",
    leaves: { x: 2, w1: -1.5, b1: 1, w2: 0.5, b2: 0.25, t: 1 },
    lesson:
      "z comes out negative, so the ReLU outputs zero — and it passes zero gradient backward. Everything behind it learns nothing at all this step.",
  },
  {
    id: "negative",
    title: "Overshooting the target",
    leaves: { x: -1.5, w1: 2, b1: 4, w2: -1.5, b2: 0.5, t: 3 },
    lesson:
      "The prediction lands above the target, so d is positive and the gradients change sign. Watch which way the weights move.",
  },
] as const;

// ── Forward pass ──────────────────────────────────────────────────────────

export type Values = Record<string, number>;

export function forward(leaves: Values): Values {
  const values: Values = { ...leaves };
  for (const id of topologicalOrder()) {
    const node = nodeById(id);
    if (isLeaf(node)) continue;
    const [a, b] = node.inputs.map((input) => values[input] ?? 0);
    switch (node.op) {
      case "mul":
        values[id] = a! * b!;
        break;
      case "add":
        values[id] = a! + b!;
        break;
      case "sub":
        values[id] = a! - b!;
        break;
      case "relu":
        values[id] = Math.max(0, a!);
        break;
      case "square":
        values[id] = a! * a!;
        break;
      default:
        values[id] = 0;
    }
  }
  return values;
}

// ── Reverse-mode autograd: the ground truth ───────────────────────────────

/** Gradient on each edge, keyed `consumer->producer`. */
export type EdgeGrads = Record<string, number>;
export const edgeKey = (consumer: string, producer: string): string =>
  `${consumer}->${producer}`;

export interface Trace {
  values: Values;
  /** dL/dnode for every node. */
  grads: Values;
  /** The gradient each consumer sends to each of its producers. */
  edges: EdgeGrads;
}

/**
 * Reverse-mode automatic differentiation, written out.
 *
 * Walk the topological order backwards. Each node takes the gradient that has
 * accumulated on it — accumulated, note, by summing over every consumer, which is
 * the whole multivariable chain rule in one `+=` — multiplies by each local
 * derivative, and adds the result onto its producers.
 */
export function autograd(leaves: Values): Trace {
  const values = forward(leaves);
  const grads: Values = {};
  const edges: EdgeGrads = {};
  for (const node of GRAPH) grads[node.id] = 0;

  grads.L = 1;

  for (const id of [...topologicalOrder()].reverse()) {
    const node = nodeById(id);
    if (isLeaf(node)) continue;

    const incoming = grads[id] ?? 0;
    const inputs = node.inputs;
    const [a, b] = inputs.map((input) => values[input] ?? 0);

    const send = (producer: string, amount: number) => {
      edges[edgeKey(id, producer)] = amount;
      grads[producer] = (grads[producer] ?? 0) + amount;
    };

    switch (node.op) {
      case "mul":
        // Cross-over: each input's local derivative is the OTHER input.
        send(inputs[0]!, incoming * b!);
        send(inputs[1]!, incoming * a!);
        break;
      case "add":
        // Both local derivatives are 1, so the gradient is copied, not divided.
        send(inputs[0]!, incoming);
        send(inputs[1]!, incoming);
        break;
      case "sub":
        send(inputs[0]!, incoming);
        send(inputs[1]!, -incoming);
        break;
      case "relu":
        // The gate: closed at or below zero, and closed means zero, not small.
        send(inputs[0]!, values[inputs[0]!]! > 0 ? incoming : 0);
        break;
      case "square":
        send(inputs[0]!, incoming * 2 * a!);
        break;
      default:
        break;
    }
  }

  return { values, grads, edges };
}

// ── The routing rules the player chooses between ──────────────────────────

export type RuleKind = "route" | "accumulate";

export interface Rule {
  id: string;
  /** Which op this rule is offered for. */
  op: Op | "accumulate";
  label: string;
  /** The maths, in words. */
  detail: string;
  correct: boolean;
  /**
   * The misconception this option represents, when it is wrong. Used to name the
   * failure precisely rather than saying "wrong" four different ways.
   */
  misconception?: MisconceptionId;
  /** Outgoing gradients, in input order. `values` are the forward values. */
  apply: (incoming: number, inputValues: number[]) => number[];
  /** For accumulate rules: combine the arriving contributions. */
  combine?: (contributions: number[]) => number;
}

export type MisconceptionId =
  | "add-splits"
  | "mul-swapped"
  | "mul-passthrough"
  | "gate-ignored"
  | "gate-always-shut"
  | "branch-averaged"
  | "branch-dropped"
  | "branch-maxed"
  | "square-wrong-power"
  | "sub-sign";

export const ROUTE_RULES: readonly Rule[] = [
  // ── square ──
  {
    id: "square-2d",
    op: "square",
    label: "g × 2d",
    detail: "The derivative of d² is 2d, so the gradient scales by twice d.",
    correct: true,
    apply: (g, [d]) => [g * 2 * d!],
  },
  {
    id: "square-d2",
    op: "square",
    label: "g × d²",
    detail: "Reusing the function instead of differentiating it.",
    correct: false,
    misconception: "square-wrong-power",
    apply: (g, [d]) => [g * d! * d!],
  },
  {
    id: "square-d",
    op: "square",
    label: "g × d",
    detail: "Forgetting the factor of two.",
    correct: false,
    misconception: "square-wrong-power",
    apply: (g, [d]) => [g * d!],
  },

  // ── sub ──
  {
    id: "sub-pass",
    op: "sub",
    label: "pass g through unchanged",
    detail:
      "d = y − t, so ∂d/∂y = 1. The gradient reaches y untouched. (t is a constant, so its gradient goes nowhere.)",
    correct: true,
    apply: (g) => [g, -g],
  },
  {
    id: "sub-negate",
    op: "sub",
    label: "negate g",
    detail: "Using ∂d/∂t = −1 for the wrong input.",
    correct: false,
    misconception: "sub-sign",
    apply: (g) => [-g, g],
  },

  // ── add ──
  {
    id: "add-copy",
    op: "add",
    label: "copy g to both inputs",
    detail:
      "Both local derivatives of a + b are 1, so each input receives the whole gradient. Nothing is divided.",
    correct: true,
    apply: (g) => [g, g],
  },
  {
    id: "add-split",
    op: "add",
    label: "split g in half",
    detail:
      "The intuition that a gradient is a quantity to be shared out. It is not — it is a rate, and both inputs affect the sum equally.",
    correct: false,
    misconception: "add-splits",
    apply: (g) => [g / 2, g / 2],
  },
  {
    id: "add-first",
    op: "add",
    label: "send g to the first input only",
    detail: "Treating an add as a pipe rather than a junction.",
    correct: false,
    misconception: "add-splits",
    apply: (g) => [g, 0],
  },

  // ── mul ──
  {
    id: "mul-cross",
    op: "mul",
    label: "each input gets g × the OTHER input",
    detail:
      "For c = a·b, ∂c/∂a = b and ∂c/∂b = a. The values cross over, which is why a multiply node has to remember its inputs.",
    correct: true,
    apply: (g, [a, b]) => [g * b!, g * a!],
  },
  {
    id: "mul-self",
    op: "mul",
    label: "each input gets g × itself",
    detail:
      "The most common slip at a multiply, and the reason to say the rule out loud: the partner's value, not your own.",
    correct: false,
    misconception: "mul-swapped",
    apply: (g, [a, b]) => [g * a!, g * b!],
  },
  {
    id: "mul-pass",
    op: "mul",
    label: "pass g through unchanged",
    detail: "Treating a multiply like an add.",
    correct: false,
    misconception: "mul-passthrough",
    apply: (g) => [g, g],
  },

  // ── relu ──
  {
    id: "relu-gate",
    op: "relu",
    label: "pass g only if the input was positive",
    detail:
      "∂/∂z max(0, z) is 1 above zero and 0 below it. The forward pass decides which, so the backward pass has to look at what happened.",
    correct: true,
    apply: (g, [z]) => [z! > 0 ? g : 0],
  },
  {
    id: "relu-always",
    op: "relu",
    label: "always pass g through",
    detail:
      "Forgetting that a ReLU is a switch. Costs you nothing when the gate happened to be open, and silently invents gradient when it was not.",
    correct: false,
    misconception: "gate-ignored",
    apply: (g) => [g],
  },
  {
    id: "relu-never",
    op: "relu",
    label: "always block g",
    detail: "Nothing would ever learn.",
    correct: false,
    misconception: "gate-always-shut",
    apply: () => [0],
  },
] as const;

export const ACCUMULATE_RULES: readonly Rule[] = [
  {
    id: "acc-sum",
    op: "accumulate",
    label: "add the arriving gradients",
    detail:
      "A value used in two places affects the loss through both, and the effects add. This is the multivariable chain rule, and it is a plus sign.",
    correct: true,
    apply: (g) => [g],
    combine: (parts) => parts.reduce((total, part) => total + part, 0),
  },
  {
    id: "acc-mean",
    op: "accumulate",
    label: "average them",
    detail:
      "The instinct to keep the magnitude 'reasonable'. It halves every gradient behind the branch, so everything upstream learns at half speed for no reason.",
    correct: false,
    misconception: "branch-averaged",
    apply: (g) => [g],
    combine: (parts) =>
      parts.length === 0
        ? 0
        : parts.reduce((total, part) => total + part, 0) / parts.length,
  },
  {
    id: "acc-max",
    op: "accumulate",
    label: "keep the largest",
    detail: "Borrowing max-pooling's rule, which does not apply here.",
    correct: false,
    misconception: "branch-maxed",
    apply: (g) => [g],
    combine: (parts) =>
      parts.length === 0 ? 0 : parts.reduce((best, part) => (Math.abs(part) > Math.abs(best) ? part : best), parts[0]!),
  },
  {
    id: "acc-first",
    op: "accumulate",
    label: "keep the first to arrive",
    detail: "Forgetting the second path exists at all — the skip connection's.",
    correct: false,
    misconception: "branch-dropped",
    apply: (g) => [g],
    combine: (parts) => parts[0] ?? 0,
  },
] as const;

export const rulesForOp = (op: Op): Rule[] =>
  ROUTE_RULES.filter((rule) => rule.op === op);
export const ruleById = (id: string): Rule | undefined =>
  [...ROUTE_RULES, ...ACCUMULATE_RULES].find((rule) => rule.id === id);

// ── The steps the player takes ────────────────────────────────────────────

export interface Step {
  kind: RuleKind;
  /** The node being routed, or the fan-out node being accumulated. */
  nodeId: string;
  options: Rule[];
  prompt: string;
}

/**
 * The backward walk, as a list of decisions.
 *
 * Reverse topological order, and for every node with more than one consumer an
 * extra `accumulate` step is inserted BEFORE it is routed — because you cannot
 * push a gradient onward until you have decided what arrived.
 */
export function buildSteps(): Step[] {
  const steps: Step[] = [];

  for (const id of [...topologicalOrder()].reverse()) {
    const node = nodeById(id);

    if (fanOut(id) > 1) {
      steps.push({
        kind: "accumulate",
        nodeId: id,
        options: [...ACCUMULATE_RULES],
        prompt: `${node.label} was used by ${consumersOf(id)
          .map((consumer) => nodeById(consumer).label)
          .join(" and ")}. Two gradients have arrived. What is ${
          node.label
        }'s gradient?`,
      });
    }

    if (isLeaf(node)) continue;

    steps.push({
      kind: "route",
      nodeId: id,
      options: rulesForOp(node.op),
      prompt: `Push the gradient back through ${node.label}. How does it transform?`,
    });
  }

  return steps;
}

export const STEPS = buildSteps();
export const STEP_COUNT = STEPS.length;

// ── Replaying the player's choices ────────────────────────────────────────

/** Rule id chosen per step index. */
export type Routing = Record<number, string>;

export interface PlayerTrace {
  values: Values;
  grads: Values;
  edges: EdgeGrads;
  /** Nodes whose gradient is settled, in the order they were settled. */
  settled: string[];
}

/**
 * Run the backward pass using the player's rules instead of the real ones.
 *
 * Deliberately the same walk as `autograd`, with the local derivatives swapped for
 * whatever the player picked — so a correct set of choices reproduces the autograd
 * trace exactly, and that equivalence is asserted in the tests. Contributions are
 * held per edge and only merged when the player says how, which is what makes the
 * branch decision a real decision rather than a quiz question about one.
 */
export function replay(leaves: Values, routing: Routing): PlayerTrace {
  const values = forward(leaves);
  const grads: Values = {};
  const edges: EdgeGrads = {};
  const pending: Record<string, number[]> = {};
  const settled: string[] = [];

  for (const node of GRAPH) pending[node.id] = [];
  grads.L = 1;
  settled.push("L");

  STEPS.forEach((step, index) => {
    const chosen = routing[index];
    if (chosen === undefined) return;
    const rule = ruleById(chosen);
    if (rule === undefined) return;

    const node = nodeById(step.nodeId);

    if (step.kind === "accumulate") {
      const parts = pending[step.nodeId] ?? [];
      grads[step.nodeId] = rule.combine ? rule.combine(parts) : 0;
      settled.push(step.nodeId);
      return;
    }

    // Single-consumer nodes settle as soon as their one contribution exists.
    if (grads[step.nodeId] === undefined) {
      const parts = pending[step.nodeId] ?? [];
      grads[step.nodeId] = parts.reduce((total, part) => total + part, 0);
      if (!settled.includes(step.nodeId)) settled.push(step.nodeId);
    }

    const incoming = grads[step.nodeId] ?? 0;
    const inputValues = node.inputs.map((input) => values[input] ?? 0);
    const outgoing = rule.apply(incoming, inputValues);

    node.inputs.forEach((producer, position) => {
      const amount = outgoing[position] ?? 0;
      edges[edgeKey(step.nodeId, producer)] = amount;
      pending[producer] = [...(pending[producer] ?? []), amount];
      // A producer read by exactly one consumer needs no accumulate decision.
      if (fanOut(producer) === 1) {
        grads[producer] = amount;
        if (!settled.includes(producer)) settled.push(producer);
      }
    });
  });

  return { values, grads, edges, settled };
}

// ── Scoring ───────────────────────────────────────────────────────────────

/** Two gradients count as equal within this. */
export const TOLERANCE = 1e-9;
export const gradsMatch = (a: number, b: number): boolean =>
  Math.abs(a - b) <= TOLERANCE + 1e-9 * Math.max(Math.abs(a), Math.abs(b));

/** Nodes that are supposed to end up with a gradient. */
export const GRADED_IDS = GRAPH.filter(
  (node) => !NO_GRADIENT_IDS.includes(node.id) && node.id !== "L",
).map((node) => node.id);

export interface NodeVerdict {
  id: string;
  label: string;
  truth: number;
  mine: number | null;
  correct: boolean;
  /** True when a wrong rule further downstream caused this. */
  inherited: boolean;
}

export interface Correctness {
  /** Fraction of graded nodes matching the autograd trace. */
  fraction: number;
  matched: number;
  total: number;
  verdicts: NodeVerdict[];
  /** Steps answered so far. */
  answered: number;
  /** Steps answered with the correct rule. */
  rulesRight: number;
  /**
   * Steps where a WRONG rule produced the right numbers anyway.
   *
   * Measured, and the reason this field exists: with the gate open, "always pass
   * the gradient" and "pass it only if the input was positive" give identical
   * answers, so a player holding the wrong rule scores 100%. Their arithmetic is
   * right and their understanding is not, and a game that reported only the
   * arithmetic would be certifying the misconception. The next scenario shuts the
   * gate and the same rule collapses — so the honest thing is to let it pass, say
   * plainly that it passed by coincidence, and then let the data make the point.
   */
  coincidences: number[];
}

export function scoreRouting(
  leaves: Values,
  routing: Routing,
): Correctness {
  const truth = autograd(leaves);
  const mine = replay(leaves, routing);

  const verdicts: NodeVerdict[] = GRADED_IDS.map((id) => {
    const settled = mine.settled.includes(id);
    const value = settled ? (mine.grads[id] ?? 0) : null;
    const expected = truth.grads[id] ?? 0;
    return {
      id,
      label: nodeById(id).label,
      truth: expected,
      mine: value,
      correct: value !== null && gradsMatch(value, expected),
      inherited: false,
    };
  });

  /**
   * Separate "your rule produced this wrong number" from "this is wrong because
   * something nearer the loss was". Conflating them turns the whole list red after
   * one mistake and sends the player re-checking a dozen correct decisions.
   *
   * The subtlety: a bad rule at a node corrupts that node's INPUTS, not the node
   * itself — `v`'s own gradient arrives from `s` and is unaffected by whatever rule
   * `v` applies. So the direct victims are the producers a wrong step sent to, and
   * for an accumulate step it is the branch node itself, since that step is what
   * sets its gradient.
   */
  const wrongSteps = new Set<string>();
  STEPS.forEach((step, index) => {
    const chosen = routing[index];
    const rule = chosen === undefined ? undefined : ruleById(chosen);
    if (rule === undefined || rule.correct) return;
    if (step.kind === "accumulate") {
      wrongSteps.add(step.nodeId);
      return;
    }
    for (const producer of nodeById(step.nodeId).inputs) {
      wrongSteps.add(producer);
    }
  });
  for (const verdict of verdicts) {
    if (!verdict.correct && verdict.mine !== null && !wrongSteps.has(verdict.id)) {
      verdict.inherited = true;
    }
  }

  const answered = Object.keys(routing).length;
  const rulesRight = Object.entries(routing).filter(([, id]) => {
    const rule = ruleById(id);
    return rule !== undefined && rule.correct;
  }).length;

  const matched = verdicts.filter((verdict) => verdict.correct).length;

  // A wrong rule whose numbers happen to agree with the truth on this scenario.
  // Detected by swapping it for the correct one and seeing if anything moves.
  const coincidences: number[] = [];
  STEPS.forEach((step, index) => {
    const chosen = routing[index];
    if (chosen === undefined) return;
    const rule = ruleById(chosen);
    if (rule === undefined || rule.correct) return;

    const corrected: Routing = { ...routing };
    const right = step.options.find((option) => option.correct);
    if (right === undefined) return;
    corrected[index] = right.id;

    const withRule = replay(leaves, routing).grads;
    const withTruth = replay(leaves, corrected).grads;
    const identical = GRADED_IDS.every((id) =>
      gradsMatch(withRule[id] ?? 0, withTruth[id] ?? 0),
    );
    if (identical) coincidences.push(index);
  });

  return {
    fraction: matched / verdicts.length,
    matched,
    total: verdicts.length,
    verdicts,
    answered,
    rulesRight,
    coincidences,
  };
}

// ── What the gradients are FOR ────────────────────────────────────────────

export const LEARNING_RATE = 0.02;
export const TRAINING_STEPS = 40;

export interface UpdateResult {
  /** Loss before the step. */
  before: number;
  /** Loss after stepping with these gradients. */
  after: number;
  /** New parameter values. */
  params: Values;
}

/** One gradient-descent step. */
export function applyStep(leaves: Values, grads: Values): UpdateResult {
  const before = forward(leaves).L ?? 0;
  const params: Values = { ...leaves };
  for (const id of PARAM_IDS) {
    params[id] = (leaves[id] ?? 0) - LEARNING_RATE * (grads[id] ?? 0);
  }
  return { before, after: forward(params).L ?? 0, params };
}

/**
 * The angle between the player's parameter gradient and the true one.
 *
 * This is the honest single-number summary of a wrong backward pass, and it took a
 * measurement to work out that it had to be. The obvious check — take one step and
 * see whether the loss went up — turns out to be nearly useless here: at a small
 * learning rate any direction with a positive dot product against the truth still
 * goes downhill, and measured on this graph a swapped multiply produced a LOWER
 * loss after one step than the correct gradient did. So "your loss went up" would
 * have been a claim the game could not support. The angle is what is actually
 * wrong, and it is wrong whether or not one lucky step happened to land well.
 */
export function gradientAngle(mine: Values, truth: Values): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (const id of PARAM_IDS) {
    const a = mine[id] ?? 0;
    const b = truth[id] ?? 0;
    dot += a * b;
    normA += a * a;
    normB += b * b;
  }
  if (normA < 1e-18 || normB < 1e-18) {
    // One of them is the zero vector: there is no angle, but "identical" and
    // "you produced nothing" are different answers.
    return normA < 1e-18 && normB < 1e-18 ? 0 : 90;
  }
  const cosine = clamp(dot / Math.sqrt(normA * normB), -1, 1);
  return (Math.acos(cosine) * 180) / Math.PI;
}

/**
 * How long the player's gradient is, relative to the real one.
 *
 * Needed alongside the angle, and finding out why was the most interesting thing
 * the tuning turned up. Several genuinely wrong rules — halving at an add, dropping
 * the factor of two at the square — produce a gradient that is a positive SCALAR
 * MULTIPLE of the truth. The direction is perfect; only the length is wrong. Those
 * bugs are nearly undetectable in practice because they are absorbed into the
 * learning rate, and a game reporting only the angle would call them harmless.
 */
export function magnitudeRatio(mine: Values, truth: Values): number {
  let normA = 0;
  let normB = 0;
  for (const id of PARAM_IDS) {
    normA += (mine[id] ?? 0) ** 2;
    normB += (truth[id] ?? 0) ** 2;
  }
  if (normB < 1e-18) return normA < 1e-18 ? 1 : Number.POSITIVE_INFINITY;
  return Math.sqrt(normA / normB);
}

export interface TrainingRun {
  losses: number[];
  final: number;
  params: Values;
  /** True when the loss ran away rather than settling. */
  diverged: boolean;
}

/**
 * Train for many steps, recomputing the gradients each time with a given routing.
 *
 * This is the answer to "so what?", and it needs more than one step to be an
 * answer at all. A wrong rule is not a one-off error, it is a wrong rule applied
 * again at every step, so the two trajectories separate — which is exactly how a
 * broken autograd behaves in practice: the network still trains, and it trains
 * worse, for reasons that never show up in a single number.
 *
 * Pass `routing` as null to train with the real gradients.
 */
export function train(
  leaves: Values,
  routing: Routing | null,
  steps = TRAINING_STEPS,
): TrainingRun {
  let params: Values = { ...leaves };
  const losses: number[] = [forward(params).L ?? 0];

  let diverged = false;

  for (let step = 0; step < steps; step += 1) {
    const grads =
      routing === null
        ? autograd(params).grads
        : replay(params, routing).grads;
    for (const id of PARAM_IDS) {
      params = {
        ...params,
        [id]: (params[id] ?? 0) - LEARNING_RATE * (grads[id] ?? 0),
      };
    }
    const loss = forward(params).L ?? 0;
    // A gradient pointing uphill sends this to infinity within a few dozen steps,
    // and an unguarded loop then reports NaN, which is not a teaching moment.
    if (!Number.isFinite(loss) || loss > 1e12) {
      diverged = true;
      break;
    }
    losses.push(loss);
  }

  return { losses, final: losses.at(-1) ?? 0, params, diverged };
}

// ── Judging ───────────────────────────────────────────────────────────────

/** Every graded node has to match to clear a scenario. */
export const TARGET_FRACTION = 1;

export type Outcome = "routing" | "cleared" | "broken";

export interface Evaluation {
  outcome: Outcome;
  correctness: Correctness;
  /** The step the player should look at next, or null when finished. */
  nextStep: number | null;
  mine: UpdateResult | null;
  truth: UpdateResult | null;
  /** Degrees between the player's parameter gradient and the real one. */
  angle: number;
  /** Length of the player's gradient relative to the real one. */
  ratio: number;
  /** 40 steps of descent with the player's rules, and with the real ones. */
  mineRun: TrainingRun | null;
  truthRun: TrainingRun | null;
  score: number;
  failure: NamedFailure | null;
}

const signed = (value: number) =>
  `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;

/**
 * What a wrong gradient costs, stated only as strongly as the numbers allow.
 *
 * The first draft of this said "and now the loss goes up". Measured, that is
 * usually false: at this learning rate anything within 90° of the truth still
 * descends, and on this graph a swapped multiply once produced a LOWER loss after
 * one step than the correct gradient did. So the copy reports what is actually
 * true, and the three cases turn out to be genuinely different lessons:
 *
 *   uphill (angle > 90°)  — descent becomes ascent and the run blows up.
 *   rescaled (angle ~ 0)  — the direction is perfect and only the length is wrong,
 *                           which is a learning-rate bug wearing a disguise and the
 *                           hardest kind to ever notice.
 *   off-axis (in between) — still downhill, still wrong, and the two trajectories
 *                           separate over many steps.
 */
function describeCost(
  angle: number,
  ratio: number,
  mineRun: TrainingRun,
  truthRun: TrainingRun,
): string {
  if (angle > 90) {
    return ` And it points UPHILL: ${angle.toFixed(
      0,
    )}° away from the true gradient, which is more than a right angle, so subtracting it increases the loss. ${
      mineRun.diverged
        ? `Left to run for ${TRAINING_STEPS} steps it does not converge at all — the loss runs away to infinity while the real gradients settle at ${truthRun.final.toFixed(
            4,
          )}.`
        : `Over ${TRAINING_STEPS} steps it reaches ${mineRun.final.toFixed(
            4,
          )} against ${truthRun.final.toFixed(4)} for the real ones.`
    }`;
  }

  if (angle < 0.5 && Number.isFinite(ratio) && Math.abs(ratio - 1) > 0.02) {
    return ` Here is the interesting part: your gradient points in exactly the right DIRECTION — 0° from the truth — and is ${ratio.toFixed(
      2,
    )} times its length. A rule that rescales every gradient uniformly is a learning-rate bug in disguise, which is why this class of mistake survives in real codebases for years: the model still trains, at an effective learning rate nobody chose and nobody can find in the config.`;
  }

  if (angle < 0.5) {
    return ` Oddly, the errors cancel on these particular numbers and your gradient comes out identical to the real one. That will not survive a different scenario.`;
  }

  return ` Your parameter gradient points ${angle.toFixed(
    0,
  )}° away from the real one and is ${
    Number.isFinite(ratio) ? `${ratio.toFixed(2)} times` : "wildly different in"
  } its length. Still downhill, note — one step lowers the loss anyway, which is exactly why a broken backward pass is so hard to catch. Over ${TRAINING_STEPS} steps the two paths separate: yours reaches ${mineRun.final.toFixed(
    4,
  )}, the real gradients reach ${truthRun.final.toFixed(4)}.`;
}

const MISCONCEPTION_NAMES: Record<MisconceptionId, string> = {
  "add-splits": "Add nodes copy, they do not split",
  "mul-swapped": "Swapped the multiply's partners",
  "mul-passthrough": "Treated a multiply like an add",
  "gate-ignored": "Invented gradient through a shut gate",
  "gate-always-shut": "Closed a gate that was open",
  "branch-averaged": "Averaged a branch instead of adding it",
  "branch-dropped": "Forgot one of the branch's paths",
  "branch-maxed": "Took the largest at a branch instead of the sum",
  "square-wrong-power": "Reused the function instead of its derivative",
  "sub-sign": "Sent the gradient backward with the wrong sign",
};

/**
 * Judge the backward pass.
 *
 * The diagnosis is precise because it can afford to be: the player's wrong choice
 * is a known misconception with a name, so there is no reason to say "incorrect"
 * when the game can say which rule was misremembered and what it cost. The first
 * wrong step is the one reported, because a chain rule fails from the front — the
 * earliest mistake in the walk is the one that corrupted everything after it.
 */
export function evaluate({
  leaves,
  routing,
}: {
  leaves: Values;
  routing: Routing;
}): Evaluation {
  const correctness = scoreRouting(leaves, routing);
  const truth = autograd(leaves);

  const firstUnanswered = STEPS.findIndex(
    (_, index) => routing[index] === undefined,
  );
  const nextStep = firstUnanswered === -1 ? null : firstUnanswered;

  if (nextStep !== null) {
    return {
      outcome: "routing",
      correctness,
      nextStep,
      mine: null,
      truth: null,
      angle: 0,
      ratio: 1,
      mineRun: null,
      truthRun: null,
      score: 0,
      failure: null,
    };
  }

  const mineGrads = replay(leaves, routing).grads;
  const mineUpdate = applyStep(leaves, mineGrads);
  const truthUpdate = applyStep(leaves, truth.grads);
  const angle = gradientAngle(mineGrads, truth.grads);
  const ratio = magnitudeRatio(mineGrads, truth.grads);
  const mineRun = train(leaves, routing);
  const truthRun = train(leaves, null);

  if (correctness.fraction >= TARGET_FRACTION) {
    // Fewer wrong turns is a better score; the walk itself is not a race.
    const cleanliness = clamp(correctness.rulesRight / STEP_COUNT, 0, 1);
    return {
      outcome: "cleared",
      correctness,
      nextStep: null,
      mine: mineUpdate,
      truth: truthUpdate,
      angle,
      ratio,
      mineRun,
      truthRun,
      score: clamp(0.6 + 0.4 * cleanliness, 0, 1),
      failure: null,
    };
  }

  // The earliest wrong step: a chain fails from the front.
  let culprit: { index: number; rule: Rule; step: Step } | null = null;
  for (let index = 0; index < STEPS.length; index += 1) {
    const chosen = routing[index];
    if (chosen === undefined) continue;
    const rule = ruleById(chosen);
    if (rule === undefined || rule.correct) continue;
    culprit = { index, rule, step: STEPS[index]! };
    break;
  }

  const poisoned = correctness.verdicts.filter(
    (verdict) => !verdict.correct,
  ).length;

  // How many links are broken. One misconception gets named precisely; several at
  // once cannot honestly be blamed on any single one of them, and that is the case
  // the catalog calls a broken chain rule.
  const wrongRuleCount = Object.entries(routing).filter(([, id]) => {
    const rule = ruleById(id);
    return rule !== undefined && !rule.correct;
  }).length;

  const name =
    wrongRuleCount > 1
      ? "Broken chain rule"
      : culprit?.rule.misconception !== undefined
        ? MISCONCEPTION_NAMES[culprit.rule.misconception]
        : "Broken chain rule";

  const worstParam = correctness.verdicts
    .filter((verdict) => PARAM_IDS.includes(verdict.id) && !verdict.correct)
    .sort(
      (a, b) =>
        Math.abs(b.truth - (b.mine ?? 0)) - Math.abs(a.truth - (a.mine ?? 0)),
    )[0];

  /**
   * What the wrong gradient actually costs.
   *
   * Deliberately NOT "your loss went up after one step" — measured, that is
   * usually false here, because at this learning rate anything pointing vaguely
   * downhill still descends. The angle is the real damage, and 40 steps of descent
   * is where it becomes visible as a number the player cares about.
   */
  const costLine = describeCost(angle, ratio, mineRun, truthRun);

  return {
    outcome: "broken",
    correctness,
    nextStep: null,
    mine: mineUpdate,
    truth: truthUpdate,
    angle,
    ratio,
    mineRun,
    truthRun,
    score: clamp(correctness.fraction * 0.5, 0, 1),
    failure: {
      name,
      detail: `${
        culprit === null
          ? `${correctness.matched} of ${correctness.total} node gradients match the autograd trace.`
          : `At ${nodeById(culprit.step.nodeId).label} you chose "${
              culprit.rule.label
            }". ${culprit.rule.detail}`
      } ${
        culprit === null
          ? ""
          : `That one choice is why ${poisoned} of ${
              correctness.total
            } node gradients are now wrong: a backward pass is a chain, so everything behind the break inherits it. ${
              worstParam === undefined
                ? ""
                : `${worstParam.label}'s gradient should be ${signed(
                    worstParam.truth,
                  )} and you have ${signed(worstParam.mine ?? 0)}.`
            }`
      }${
        wrongRuleCount > 1
          ? ` ${wrongRuleCount} of your rules are wrong, so no single misconception explains this one — the earliest is the one quoted above, and it is the one to fix first.`
          : ""
      }${costLine} Fix the earliest red node and watch the rest go green on their own.`,
    },
  };
}

// ── Reveal the math ───────────────────────────────────────────────────────

export const MATH_EQUATION = String.raw`\frac{\partial L}{\partial \theta} = \underbrace{\frac{\partial L}{\partial n}}_{\text{arrived here}} \cdot \underbrace{\frac{\partial n}{\partial \theta}}_{\text{local}}
\\[1.2em]
\text{a value used twice:}\quad \frac{\partial L}{\partial h} = \sum_{c \,\in\, \text{consumers}(h)} \frac{\partial L}{\partial c}\cdot\frac{\partial c}{\partial h}
\\[1.2em]
\frac{\partial}{\partial a}(ab) = b \qquad \frac{\partial}{\partial a}(a+b) = 1 \qquad \frac{\partial}{\partial z}\max(0,z) = [z>0]`;

export const MATH_CODE = `// Reverse-mode autograd. This is the whole algorithm.
grads[loss] = 1;

for (const node of topologicalOrder().reverse()) {
  const g = grads[node.id];              // what arrived from downstream

  for (const [input, localDerivative] of node.locals()) {
    grads[input] += g * localDerivative;  // chain rule, then ACCUMULATE
  }
}

// The '+=' is the part people skip. A value used in three places gets
// three contributions, and they add. Write '=' instead and every branch
// in your network silently loses gradient.

// The local derivatives are the only per-operation knowledge required:
//   mul(a, b) -> [b, a]        the partner's value, not your own
//   add(a, b) -> [1, 1]        copied, never divided
//   relu(z)   -> [z > 0]       a switch the forward pass already threw
//   square(d) -> [2 * d]`;

export const MATH_NOTES = `A gradient is not a quantity flowing through a pipe, and almost every mistake in this game comes from imagining that it is. It is a rate: how much the loss would change per unit change in this value. That single reframing settles most of the rules by itself.

Take the add node. If a gradient were a quantity, splitting it between two inputs would be the careful thing to do. But a + b changes one-for-one with a AND one-for-one with b — nudge either by 0.01 and the sum moves by 0.01 — so both local derivatives are 1 and each input receives the whole gradient. Nothing is shared out because nothing was ever a fixed amount.

The multiply is where the chain rule earns its keep. For c = a·b the derivative with respect to a is b, the OTHER input. That is why every autograd implementation stores its operands during the forward pass: the backward pass needs values it has no other way to know. It is also why a multiply node is where people slip, and the fix is to say the rule out loud rather than to concentrate harder.

The ReLU is a switch that the forward pass already threw. Above zero it passes gradient untouched; at or below zero it passes exactly zero — not a little, zero. Which means a unit that was silent on this example contributes nothing to learning on this example, and everything behind it learns nothing through that path. That is the same fact that Convolution Kitchen calls a dead filter, seen from the other direction.

And the branch. A value used in two places affects the loss through both routes, and the two effects add. Not average, not maximum, not whichever you noticed first — add. This is the multivariable chain rule and in code it is the difference between '+=' and '='. It is also the single most consequential character in a hand-written autograd engine: write '=' and every skip connection, every shared embedding, every reused feature map quietly drops half its gradient, the network still trains, and it trains worse than it should for reasons nobody can find.

One more thing worth noticing while you play. Nothing in this backward pass knows what the network is FOR. Each node applies a local rule to the number that arrived and passes the result on. Blame gets assigned across an arbitrarily deep graph by nothing more than that, repeated — which is why the same fourteen lines of code differentiate a two-weight toy and a two-billion-parameter model.`;
