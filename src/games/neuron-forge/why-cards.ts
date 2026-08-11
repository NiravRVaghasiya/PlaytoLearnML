import type { WhyCardContent } from "@/components";
import {
  MAX_LAYERS,
  MAX_NEURONS_PER_LAYER,
  isEffectivelyLinear,
  type Architecture,
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
 */
export type ForgeEvent =
  | { kind: "reset" }
  | { kind: "pattern-changed"; pattern: PatternSpec }
  | {
      kind: "architecture-changed";
      architecture: Architecture;
      pattern: PatternSpec;
      /** What the player just did, for the opening clause. */
      change: "layer-added" | "layer-removed" | "neurons" | "activation";
    }
  | { kind: "training-started"; architecture: Architecture; pattern: PatternSpec }
  | { kind: "trained"; evaluation: Evaluation; pattern: PatternSpec };

const percent = (value: number) => `${Math.round(value * 100)}%`;

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

    case "pattern-changed":
      return {
        key: `pattern-${event.pattern.id}`,
        title: event.pattern.name,
        body: `${event.pattern.hint} Budget is ${event.pattern.budget} neuron${
          event.pattern.budget === 1 ? "" : "s"
        }, and you need ${percent(
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
          body: `You have ${spent} neuron${
            spent === 1 ? "" : "s"
          } across ${architecture.layers.length} layer${
            architecture.layers.length === 1 ? "" : "s"
          }, and every one uses the linear activation. Before you train: a linear function of a linear function is still linear, so multiply the weight matrices together and the whole stack is one matrix — a single straight cut. ${
            pattern.needsNonlinearity
              ? `This shape cannot be cut straight, so the neurons will buy you nothing. Train it and watch.`
              : `This shape *can* be cut straight, so this will work — it is just a roundabout way of writing a much smaller model.`
          }`,
          tone: pattern.needsNonlinearity ? "warn" : "info",
        };
      }

      const opening =
        change === "layer-added"
          ? `Another layer. Depth composes bends: layer two bends what layer one already bent, which is how a network builds shapes a single layer cannot.`
          : change === "layer-removed"
            ? `One layer fewer. You just gave up the ability to bend a bend.`
            : change === "activation"
              ? `New activation. This is the part that decides whether the boundary can curve at all.`
              : `Width changed. More neurons in a layer means more separate cuts available at that depth, and the next layer gets to combine them.`;

      return {
        key: `arch-${change}-${describe(architecture)}`,
        title: describe(architecture),
        body: `${opening} You are spending ${spent} of ${pattern.budget} neuron${
          pattern.budget === 1 ? "" : "s"
        }${
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
        )} to reduce binary cross-entropy on the training points. Watch the loss curve: if it flattens early and high, the architecture cannot represent this shape, and no amount of extra training will change that.`,
        tone: "info",
      };

    case "trained": {
      const { evaluation, pattern } = event;

      if (evaluation.outcome === "win") {
        const leftover = evaluation.budget - evaluation.totalNeurons;
        return {
          key: `win-${pattern.id}-${evaluation.totalNeurons}`,
          title: `Solved with ${evaluation.totalNeurons} neuron${
            evaluation.totalNeurons === 1 ? "" : "s"
          }`,
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
        };
      }

      if (evaluation.outcome === "no-nonlinearity") {
        return {
          key: `no-nonlinearity-${evaluation.totalNeurons}`,
          title: "The neurons were never the problem",
          body: `${percent(
            evaluation.accuracy,
          )} — and it would be the same with ten times as many neurons, because every layer here is linear and the whole stack collapses to a single straight cut. This is the difference between *size* and *shape* of a model. Change one activation to relu and train again; nothing else needs to move.`,
          tone: "bad",
        };
      }

      if (evaluation.outcome === "insufficient-capacity") {
        const remaining = evaluation.budget - evaluation.totalNeurons;
        return {
          key: `insufficient-${evaluation.totalNeurons}`,
          title: "Right shape, not enough of it",
          body: `${percent(
            evaluation.accuracy,
          )} against a ${percent(
            pattern.target,
          )} bar. The boundary is bending — you can see it curve in the surface — but it cannot bend in enough places at once. ${
            remaining >= 2
              ? `${pattern.minimalSolution} is known to clear this, and you have ${remaining} neuron${
                  remaining === 1 ? "" : "s"
                } spare.`
              : `You are at the cap, so this is now about arrangement: ${MAX_LAYERS} layers of up to ${MAX_NEURONS_PER_LAYER} are available, and depth buys different things than width.`
          }`,
          tone: "bad",
        };
      }

      return {
        key: "near-miss",
        title: "Close",
        body: `${percent(evaluation.accuracy)} on held-out points, just short of ${percent(
          pattern.target,
        )}. Train again — the weights start from the same place every time, so if you want a different result you have to change the architecture, not re-roll it.`,
        tone: "warn",
      };
    }
  }
}
