import type { WhyCardContent } from "@/components";
import {
  FEATURES,
  MIN_HONEST_LEAF,
  ROUNDS,
  bestSplit,
  countsOf,
  describeSplit,
  gainOf,
  giniOf,
  lookaheadGainOf,
  partition,
  type Evaluation,
  type Round,
  type Sample,
  type Split,
} from "./ml";

/**
 * "Why did that happen?" copy for Decision Tree Architect.
 *
 * The rule this copy follows: every claim about a gate is stated in samples, not
 * in adjectives. "That split was weak" is an opinion; "that split moved 6 of 140
 * plots and left both sides as mixed as before" is the thing that transfers.
 *
 * It also never calls depth bad. Depth is what lets a tree describe a diagonal at
 * all, and three of these plots need more than two gates. What is bad is depth
 * spent after validation stopped improving — a different statement, and the only
 * one the numbers actually support.
 */
export type ArchitectEvent =
  | { kind: "round-briefing"; round: Round }
  | {
      kind: "gate-built";
      round: Round;
      nodeId: string;
      split: Split;
      /** Training samples that were at the node before the split. */
      bucket: Sample[];
      trainAccuracy: number;
      validationAccuracy: number;
      previousValidation: number;
      peakValidation: number;
      peakDepth: number;
      starved: number;
      depth: number;
    }
  | {
      kind: "pruned";
      round: Round;
      validationAccuracy: number;
      previousValidation: number;
      peakValidation: number;
      depth: number;
    }
  | {
      kind: "signed-off";
      round: Round;
      evaluation: Evaluation;
      complete: boolean;
      attempts: number;
    };

const percent = (value: number) => `${Math.round(value * 100)}%`;
const points = (value: number) => `${(value * 100).toFixed(1)}%`;

export function whyCardFor(event: ArchitectEvent): WhyCardContent {
  switch (event.kind) {
    case "round-briefing": {
      const { round } = event;
      return {
        key: `brief-${round.index}`,
        title: `Plot ${round.index} of ${ROUNDS.length}: ${round.name}`,
        body: `${round.brief} You have ${round.trainPoints} surveyed plots to build from and ${percent(
          round.noiseRate,
        )} of their verdicts are simply wrong, so ${percent(
          1 - round.noiseRate,
        )} is the most any tree can honestly reach. Sign off at ${percent(
          round.target,
        )} validation accuracy, depth ${round.maxDepth} or less. ${round.lesson}`,
        tone: "info",
      };
    }

    case "gate-built": {
      const {
        round,
        split,
        bucket,
        trainAccuracy,
        validationAccuracy,
        previousValidation,
        peakValidation,
        peakDepth,
        starved,
        depth,
      } = event;

      const gain = gainOf(bucket, split);
      const { left, right } = partition(bucket, split);
      const parentGini = giniOf(countsOf(bucket));
      const lookahead = lookaheadGainOf(bucket, split);
      const validationDelta = validationAccuracy - previousValidation;

      // Gave ground back. The most important card in the game.
      if (peakValidation > 0 && peakValidation - validationAccuracy >= 0.01) {
        return {
          key: `giving-back-${depth}-${validationAccuracy.toFixed(3)}`,
          title: "Training up, validation down",
          body: `${describeSplit(split)} took training accuracy to ${points(
            trainAccuracy,
          )} and validation to ${points(
            validationAccuracy,
          )} — you had ${points(
            peakValidation,
          )} at depth ${peakDepth}. Every gate can only improve training accuracy, because every gate is chosen to reduce impurity on the samples in front of it. Validation has no such guarantee. ${
            starved > 0
              ? `${starved} of your leaves now hold fewer than ${MIN_HONEST_LEAF} plots, which is few enough that they are describing individual surveys rather than ground.`
              : `This is the point where the gates start fitting the ${percent(
                  round.noiseRate,
                )} of verdicts that are wrong.`
          }`,
          tone: "warn",
        };
      }

      // A gate that barely moved anything.
      if (gain < 0.01 && lookahead < 0.05) {
        return {
          key: `weak-gate-${describeSplit(split)}`,
          title: `${describeSplit(split)} bought almost nothing`,
          body: `Gain ${gain.toFixed(
            4,
          )} on ${bucket.length} plots: impurity went from ${parentGini.toFixed(
            3,
          )} to a weighted ${(parentGini - gain).toFixed(
            3,
          )}, and the two sides came out ${left.length} and ${
            right.length
          }. ${
            FEATURES[split.feature]?.irrelevant
              ? `${FEATURES[split.feature]!.name} is a reading that has nothing to do with whether the ground holds — no threshold on it will ever help, though on a small sample one will always look slightly better than the rest.`
              : `The lookahead column is ${lookahead.toFixed(
                  3,
                )} too, so there is no hidden structure waiting below this one either.`
          } Prune it and try another feature.`,
          tone: "warn",
        };
      }

      // A gate with little immediate gain but a lot underneath — the lesson of
      // the ridge plot.
      if (gain < 0.02 && lookahead >= 0.15) {
        return {
          key: `lookahead-gate-${describeSplit(split)}`,
          title: "Almost no gain, and exactly the right gate",
          body: `${describeSplit(split)} gained only ${gain.toFixed(
            4,
          )} — a greedy learner would have rejected it. But it split ${
            bucket.length
          } plots into ${left.length} and ${
            right.length
          }, and the best gate available inside those two is now worth ${lookahead.toFixed(
            3,
          )}. That is what the lookahead column was showing you. Real CART only ever looks one gate ahead, which is exactly why it cannot solve this shape and you can.`,
          tone: "good",
        };
      }

      return {
        key: `gate-${depth}-${describeSplit(split)}`,
        title: `${describeSplit(split)} — gain ${gain.toFixed(3)}`,
        body: `${bucket.length} plots split ${left.length} / ${
          right.length
        }, impurity ${parentGini.toFixed(3)} down to a weighted ${(
          parentGini - gain
        ).toFixed(3)}. Training accuracy ${points(
          trainAccuracy,
        )}, validation ${points(validationAccuracy)}${
          validationDelta >= 0.001
            ? ` — up ${points(validationDelta)}, so this gate found something real about the ground and not just about these ${bucket.length} surveys.`
            : validationDelta <= -0.001
              ? ` — down ${points(
                  Math.abs(validationDelta),
                )}, which is worth watching.`
              : ", unchanged."
        }`,
        tone: "info",
      };
    }

    case "pruned": {
      const { validationAccuracy, previousValidation, peakValidation, depth } =
        event;
      const recovered = validationAccuracy - previousValidation;
      return {
        key: `pruned-${depth}-${validationAccuracy.toFixed(3)}`,
        title:
          recovered > 0.001
            ? `Pruning recovered ${points(recovered)}`
            : `Pruned back to depth ${depth}`,
        body:
          recovered > 0.001
            ? `Validation is back to ${points(
                validationAccuracy,
              )} with a smaller tree. Removing gates raised accuracy on plots the tree had never seen, which is only possible if those gates were describing this particular sample rather than the ground. That is what overfitting is, stated as an experiment you just ran.`
            : `Validation is ${points(
                validationAccuracy,
              )}${
                peakValidation > validationAccuracy
                  ? `, still short of the ${points(peakValidation)} you reached earlier.`
                  : "."
              } A smaller tree that scores the same is the better piece of engineering — fewer gates means fewer places for the next survey to surprise you.`,
        tone: "info",
      };
    }

    case "signed-off": {
      const { round, evaluation, complete, attempts } = event;

      if (complete) {
        return {
          key: "all-signed",
          title: "Every plot signed off",
          body: `Three boundaries, three different trees. The rectangle took two gates and no more; the diagonal needed a staircase and still could not follow the line exactly; the ridge could not be touched by any single gate and fell apart into two after the right one. None of that is about tree size. It is about the shape a tree can describe at all — axis-aligned cuts, one feature at a time — and choosing depth to match the shape in front of you rather than the accuracy you want.`,
          tone: "good",
        };
      }

      if (evaluation.outcome === "win") {
        return {
          key: `cleared-${round.index}-${evaluation.depth}`,
          title: `${round.name} signed off at depth ${evaluation.depth}`,
          body: `Validation ${points(
            evaluation.validationAccuracy,
          )} against the ${percent(round.target)} needed, from ${
            evaluation.splits
          } gate${evaluation.splits === 1 ? "" : "s"}. Training is ${points(
            evaluation.trainAccuracy,
          )}, so the gap is ${points(
            Math.max(0, evaluation.trainAccuracy - evaluation.validationAccuracy),
          )} — ${
            evaluation.trainAccuracy - evaluation.validationAccuracy < 0.06
              ? "narrow, which means what the tree learned holds outside the sample it learned from."
              : "wide enough that some of this tree is about these particular surveys."
          } ${
            attempts === 1
              ? "First tree you signed off."
              : `Took ${attempts} attempts.`
          } ${round.lesson}`,
          tone: "good",
        };
      }

      if (evaluation.outcome === "overfit-depth") {
        return {
          key: `overfit-${evaluation.depth}`,
          title: "You had it, and built past it",
          body: `Depth ${evaluation.depth}, ${
            evaluation.splits
          } gates, training ${points(
            evaluation.trainAccuracy,
          )} and validation ${points(
            evaluation.validationAccuracy,
          )}. At depth ${evaluation.peakDepth} you were at ${points(
            evaluation.peakValidation,
          )}. The two curves separating is not a defect in the tree — it is the definition of overfitting, and a tree shows it to you one gate at a time. ${
            evaluation.starved > 0
              ? `${evaluation.starved} leaves are down to fewer than ${MIN_HONEST_LEAF} plots each.`
              : ""
          } Prune back.`,
          tone: "bad",
        };
      }

      if (evaluation.outcome === "underfit-stump") {
        return {
          key: `underfit-${evaluation.splits}`,
          title: "Not enough tree",
          body: `Training accuracy ${points(
            evaluation.trainAccuracy,
          )} with ${evaluation.splits} gate${
            evaluation.splits === 1 ? "" : "s"
          } — the tree cannot describe this ground even where it has been given the answers, so nothing here is about generalisation yet. The gap to validation is only ${points(
            Math.max(0, evaluation.trainAccuracy - evaluation.validationAccuracy),
          )}, which is the signature of a model that is too simple rather than too complex. Add gates: check the gain table on your largest leaf, and look at the lookahead column as well as the gain.`,
          tone: "bad",
        };
      }

      return {
        key: `missed-${evaluation.depth}-${evaluation.validationAccuracy.toFixed(3)}`,
        title: `Validation ${points(evaluation.validationAccuracy)}, needed ${percent(round.target)}`,
        body: `Depth ${evaluation.depth} of ${round.maxDepth} allowed, ${
          evaluation.splits
        } gates, training ${points(evaluation.trainAccuracy)}. ${
          evaluation.trainAccuracy - evaluation.validationAccuracy > 0.1
            ? `The ${points(
                evaluation.trainAccuracy - evaluation.validationAccuracy,
              )} gap says part of this tree is specific to these ${round.trainPoints} surveys. Pruning may raise validation even though it lowers training.`
            : `Training and validation are close, so the tree is generalising what it knows — there is just not enough of it. Look for the leaf with the most plots in it and check what a gate there would buy.`
        }`,
        tone: "warn",
      };
    }
  }
}

/** The best gate available at a node, for the "what would CART do" hint. */
export function greedyHintFor(bucket: Sample[]): string {
  const candidate = bestSplit(bucket);
  if (candidate === null || candidate.gain <= 0) {
    return "No gate here improves impurity at all.";
  }
  return `${describeSplit(candidate)} — gain ${candidate.gain.toFixed(3)}`;
}
