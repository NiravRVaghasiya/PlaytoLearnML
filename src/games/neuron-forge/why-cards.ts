import type { WhyCardContent } from "@/components";
import {
  MAX_LAYERS,
  MAX_NEURONS_PER_LAYER,
  TRAIN_EPOCHS,
  isEffectivelyLinear,
  isSingleCut,
  shortOf,
  type Architecture,
  type Collapse,
  type Evaluation,
  type PatternSpec,
} from "./ml";

/**
 * "Why did that happen?" copy for Neuron Forge.
 *
 * Keyed to the player's last edit. The rule this copy follows: never describe
 * capacity as a single number. "More neurons" and "another layer" and "a
 * different activation" do different things, and collapsing them into one idea
 * of "bigger" is the misconception the game exists to break.
 *
 * So the linear-activation cards deliberately talk about *shape* rather than
 * *size*, and they fire before training as a prediction prompt — noticing that a
 * stack of linear layers must fail is worth more than being told afterwards.
 *
 * Result cards lead with the outcome and its number in the title. The title is
 * what gets announced to screen readers, and "Solved: 91% held-out" is the one
 * sentence a player who cannot see the surface needs to hear.
 */
export type ForgeEvent =
  | { kind: "reset" }
  | { kind: "retry"; architecture: Architecture; pattern: PatternSpec }
  | { kind: "pattern-changed"; pattern: PatternSpec }
  | {
      kind: "architecture-changed";
      architecture: Architecture;
      pattern: PatternSpec;
      /** What the player just did, for the opening clause. */
      change: "layer-added" | "layer-removed" | "neurons" | "activation";
    }
  | { kind: "training-started"; architecture: Architecture; pattern: PatternSpec }
  | {
      kind: "trained";
      evaluation: Evaluation;
      pattern: PatternSpec;
      architecture: Architecture;
      collapse: Collapse | null;
    };

/**
 * Concept Library entries these cards link to. Each label is a section heading
 * on that page, verbatim — `concepts.test.ts` holds the link to that promise.
 */
const BOUNDARY_HREF = "/concepts/decision-boundaries";
const DESCENT_HREF = "/concepts/gradient-descent";

const percent = (value: number) => `${Math.round(value * 100)}%`;

const plural = (count: number, word: string) =>
  `${count} ${word}${count === 1 ? "" : "s"}`;

/** "one hidden layer of 4…" opens a sentence in several cards below. */
const capitalise = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);

function describe(architecture: Architecture): string {
  if (architecture.layers.length === 0) return "no hidden layers";
  return architecture.layers
    .map((layer) => `${layer.neurons}×${layer.activation}`)
    .join(" → ");
}

export function whyCardFor(event: ForgeEvent): WhyCardContent {
  switch (event.kind) {
    case "reset":
      return {
        key: "reset",
        title: "An empty forge",
        body: `With no hidden layers, the network is one straight cut through the plane — logistic regression, nothing more. That is genuinely the right answer for some of these puzzles and hopeless for others. Start by training it and finding out which kind you are looking at.`,
        tone: "info",
      };

    case "retry":
      return {
        key: `retry-${describe(event.architecture)}`,
        title: `Result cleared — ${describe(event.architecture)} kept`,
        body: `The weights start from the same place every time, so training this exact network again gives this exact result — to change the outcome, change the architecture. ${capitalise(
          event.pattern.minimalSolution,
        )} is known to clear ${percent(event.pattern.target)} here. "Clear the forge" starts from nothing.`,
        tone: "info",
      };

    case "pattern-changed":
      return {
        key: `pattern-${event.pattern.id}`,
        title: event.pattern.name,
        body: `${event.pattern.hint} Budget is ${plural(
          event.pattern.budget,
          "neuron",
        )}, and you need ${percent(
          event.pattern.target,
        )} on held-out points to clear it. The architecture resets with the puzzle, because a design that fits one shape rarely fits the next.`,
        tone: "info",
      };

    case "architecture-changed": {
      const { architecture, pattern, change } = event;
      const spent = architecture.totalNeurons;
      const linear = isEffectivelyLinear(architecture);

      // The prediction prompt. This is the highest-value card in the game, so it
      // takes priority over the ordinary "you changed a number" copy.
      if (linear && architecture.layers.length > 0) {
        return {
          key: `all-linear-${spent}`,
          title: "Every layer is linear — predict what happens",
          body: `You have ${plural(spent, "neuron")} across ${plural(
            architecture.layers.length,
            "layer",
          )}, and every one uses the linear activation. Before you train: a linear function of a linear function is still linear, so multiply the weight matrices together and the whole stack is one matrix — a single straight cut. ${
            pattern.needsNonlinearity
              ? `This shape cannot be cut straight, so the neurons will buy you nothing. Train it and watch.`
              : `This shape *can* be cut straight, so this will work — it is just a roundabout way of writing a much smaller model.`
          }`,
          tone: pattern.needsNonlinearity ? "warn" : "info",
        };
      }

      const last = architecture.layers[architecture.layers.length - 1];
      const opening =
        change === "layer-added"
          ? architecture.layers.length === 1
            ? `A hidden layer. Each of its neurons is one straight cut through the plane, and its activation is what lets the output combine those cuts into a bend.`
            : `Another layer. Depth composes bends: layer two bends what layer one already bent, which is how a network builds shapes a single layer cannot.`
          : change === "layer-removed"
            ? `One layer fewer. You just gave up the ability to bend a bend.`
            : change === "activation"
              ? `New activation. This is the part that decides whether the boundary can curve at all.`
              : `Width changed. More neurons in a layer means more separate cuts available at that depth, and the next layer gets to combine them.`;

      // Two predictions worth making before the player spends a training run.
      // Both are properties of the architecture, not guesses about the result.
      const caveat = isSingleCut(architecture)
        ? ` With one neuron in front of the first bend there is only one cut, so the boundary is still a straight line — add neurons to layer 1 before expecting a curve.`
        : architecture.layers.length > 1 &&
            last?.neurons === 1 &&
            last.activation === "relu"
          ? ` A one-neuron relu layer is a bottleneck: everything after it sees a single number, and if that one unit dies during training it takes the whole network with it.`
          : "";

      return {
        key: `arch-${change}-${describe(architecture)}`,
        title: describe(architecture),
        body: `${opening}${caveat} You are spending ${spent} of ${plural(
          pattern.budget,
          "neuron",
        )}${
          spent >= pattern.budget
            ? " — the budget is full, so from here it is rearranging, not adding."
            : "."
        } Nothing is measured until you train it.`,
        tone: "info",
      };
    }

    case "training-started":
      return {
        key: "training",
        title: "Training",
        body: `Adam is adjusting every weight in ${describe(
          event.architecture,
        )} to reduce binary cross-entropy on the training points. Watch the loss curve: if it flattens early and high, no amount of extra training will change that — either the architecture cannot represent this shape, or training has stalled in it. The result will say which.`,
        tone: "info",
        conceptHref: DESCENT_HREF,
        conceptLabel: "How gradient descent works",
      };

    case "trained":
      return trainedCard(event);
  }
}

function trainedCard({
  evaluation,
  pattern,
  architecture,
  collapse,
}: Extract<ForgeEvent, { kind: "trained" }>): WhyCardContent {
  if (evaluation.outcome === "untrained") {
    return {
      key: "not-scored",
      title: "Training did not finish — not scored",
      body: `The run ended before the network could be measured, so there is no result to judge. Train again; nothing about the architecture has changed.`,
      tone: "warn",
    };
  }

  if (evaluation.outcome === "stopped") {
    return {
      key: `stopped-${evaluation.epochsRun}`,
      title: `Stopped at epoch ${evaluation.epochsRun} of ${TRAIN_EPOCHS} — not scored`,
      body: `${percent(
        evaluation.accuracy,
      )} on held-out points so far, from a network that had not finished learning. A half-trained network says nothing about what the architecture can represent, so it is not judged either way. Train it to the end to find out.`,
      tone: "info",
    };
  }

  if (evaluation.outcome === "win") {
    const leftover = evaluation.budget - evaluation.totalNeurons;
    return {
      key: `win-${pattern.id}-${evaluation.totalNeurons}`,
      title: `Solved: ${percent(evaluation.accuracy)} held-out with ${plural(
        evaluation.totalNeurons,
        "neuron",
      )}`,
      body: `${percent(
        evaluation.accuracy,
      )} on points the network never trained on, past the ${percent(
        pattern.target,
      )} bar. ${
        leftover > 0
          ? `You left ${leftover} of the budget unspent, which is where the efficiency bonus comes from — a smaller network that fits is worth more than a large one that also fits, because it has less room to memorise.`
          : `You used the whole budget. Worth trying again leaner: see how much you can take away before it breaks.`
      }`,
      tone: "good",
      conceptHref: BOUNDARY_HREF,
      conceptLabel: "Bias and capacity",
    };
  }

  if (evaluation.outcome === "no-nonlinearity") {
    return {
      key: `no-nonlinearity-${evaluation.totalNeurons}`,
      title: `No non-linearity — stuck at ${percent(evaluation.accuracy)}`,
      body: `${percent(
        evaluation.accuracy,
      )} — and it would be the same with ten times as many neurons, because every layer here is linear and the whole stack collapses to a single straight cut. The neurons were never the problem: this is the difference between the *size* and the *shape* of a model. Change one activation to relu and train again; nothing else needs to move.`,
      tone: "bad",
      conceptHref: BOUNDARY_HREF,
      conceptLabel: "What is a decision boundary?",
    };
  }

  if (evaluation.outcome === "dead-network" && collapse) {
    const name = evaluation.failure?.name ?? "Training collapsed";
    const where =
      collapse.layer === null ? "one layer" : `layer ${collapse.layer + 1}`;
    const mechanism =
      collapse.kind === "dead-relu"
        ? `Every relu unit in ${where} went below zero for every training point during training. A relu at zero has zero slope, so the gradient that would have pulled it back is zero too — once a layer dies, nothing can revive it.`
        : collapse.kind === "saturated"
          ? `Every ${collapse.activation} unit in ${where} is pinned at the flat end of its curve for every training point, where the slope is almost zero, so the gradient that should train the layers below all but vanishes.`
          : `The layers stopped depending on the input, so the gradient has nothing to push against.`;
    return {
      key: `collapse-${collapse.kind}-${describe(architecture)}`,
      title: `${name} — one answer for every data point, ${percent(
        evaluation.accuracy,
      )} held-out`,
      // Worded as measured: the check covers the training and held-out points,
      // not every cell of the heatmap, which can still vary where no data lies.
      body: `The network answers P(class B) = ${collapse.output.toFixed(
        2,
      )} for every training and held-out point. ${mechanism} This architecture could represent more than it shows; the optimiser never got it there. Fewer, wider layers train where this stalled — ${pattern.minimalSolution} solves this puzzle.`,
      tone: "bad",
    };
  }

  if (evaluation.outcome === "optimisation-failure") {
    const extra = architecture.layers.length - pattern.minimalLayers.length;
    const isMinimal =
      extra === 0 &&
      architecture.layers.every(
        (layer, index) => layer.neurons === pattern.minimalLayers[index]?.neurons,
      );
    const holds = isMinimal
      ? `it is the arrangement measured to clear this puzzle (${pattern.minimalSolution})`
      : pattern.minimalLayers.length === 0
        ? `it can draw anything a model with ${pattern.minimalSolution} can, which clears this puzzle`
        : `it holds ${pattern.minimalSolution}, which clears this puzzle`;
    return {
      key: `optimisation-${describe(architecture)}`,
      title: `Optimisation failure — ${shortOf(
        evaluation.accuracy,
        pattern.target,
      )} of the ${percent(pattern.target)} needed`,
      body: `Its answers still vary from point to point, so this network is not dead — and ${holds}, so more neurons would not help. Training ended short of what the network can represent: gradient descent only follows the local slope, and from these starting weights it ended somewhere worse. ${
        extra > 0
          ? `Deep, narrow stacks are harder to optimise — each extra layer is another place for the gradient to shrink — so drop back to ${pattern.minimalSolution}.`
          : isMinimal
            ? `The same weights train the same way every run, so change the arrangement to change the outcome.`
            : `Drop back to ${pattern.minimalSolution}.`
      }`,
      tone: "bad",
      conceptHref: DESCENT_HREF,
      conceptLabel: "Local versus global minima",
    };
  }

  // Insufficient capacity: it trained, it is not linear, it is not dead, and it
  // does not hold the minimal solution.
  const remaining = evaluation.budget - evaluation.totalNeurons;
  const shape = isSingleCut(architecture)
    ? `There is one neuron in front of the first bend, so the surface can only split along straight lines, all parallel to that one cut. Curves come from combining several cuts, so widen layer 1 first.`
    : `The boundary bends, but not in enough places at once.`;
  return {
    key: `insufficient-${describe(architecture)}`,
    title: `Insufficient capacity — ${shortOf(
      evaluation.accuracy,
      pattern.target,
    )} of the ${percent(pattern.target)} needed`,
    body: `${shape} ${
      remaining >= 2
        ? `${capitalise(pattern.minimalSolution)} is known to clear this, and you have ${plural(
            remaining,
            "neuron",
          )} spare.`
        : `You are at the cap, so this is now about arrangement: ${MAX_LAYERS} layers of up to ${MAX_NEURONS_PER_LAYER} are available, and depth buys different things than width.`
    }`,
    tone: "bad",
    conceptHref: BOUNDARY_HREF,
    conceptLabel: "Bias and capacity",
  };
}
