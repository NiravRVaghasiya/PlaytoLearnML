import type { WhyCardContent } from "@/components";
import {
  FEATURES,
  GIVING_BACK_DROP,
  MIN_HONEST_LEAF,
  NARROW_GAP,
  ROUNDS,
  bestSplit,
  countsOf,
  describeSplit,
  gainOf,
  giniOf,
  givesGroundBack,
  lookaheadGainOf,
  partition,
  unlockedGainOf,
  type Evaluation,
  type Round,
  type Sample,
  type Split,
} from "./ml";

/**
 * Concept Library links. Only to sections that genuinely cover the card — the
 * label must equal a section heading exactly (concepts.test.ts enforces it), and
 * this game is listed on both pages, so each link funnels back here.
 */
const OVERFIT_HREF = "/concepts/overfitting";
const BOUNDARY_HREF = "/concepts/decision-boundaries";

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
      /** Training accuracy of the peak tree: overfitting needs it matched. */
      peakTrainAccuracy: number;
      starved: number;
      depth: number;
    }
  | {
      /** A whole greedy tree applied in one step (the code lane's growGreedy). */
      kind: "greedy-grown";
      round: Round;
      depth: number;
      splits: number;
      trainAccuracy: number;
      validationAccuracy: number;
      peakValidation: number;
      peakDepth: number;
      peakTrainAccuracy: number;
      starved: number;
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
        peakTrainAccuracy,
        starved,
        depth,
      } = event;

      const gain = gainOf(bucket, split);
      const { left, right } = partition(bucket, split);
      const parentGini = giniOf(countsOf(bucket));
      const lookahead = lookaheadGainOf(bucket, split);
      // What one gate could already buy here, and how much lookahead this gate
      // adds on top of it. Raw lookahead cannot tell a gate that CREATES
      // structure from a sliver that passes the node's structure straight down
      // (terrace root, soil pH < 0.97: lookahead 0.150 against 0.153 already
      // available). Only the difference can.
      const parentBest = Math.max(0, bestSplit(bucket)?.gain ?? 0);
      const unlocked = unlockedGainOf(bucket, split, parentBest);
      const irrelevant = FEATURES[split.feature]?.irrelevant ?? false;
      const validationDelta = validationAccuracy - previousValidation;

      // Gave ground back. The most important card in the game — and only
      // when training is at least where it was at the peak, the ghost's own
      // test. A smaller or worse tree is below the peak on both, and calling
      // that overfitting sent players to prune a tree that needed gates.
      if (
        givesGroundBack({
          peakValidation,
          validationAccuracy,
          trainAccuracy,
          peakTrainAccuracy,
        })
      ) {
        return {
          key: `giving-back-${depth}-${validationAccuracy.toFixed(3)}`,
          title:
            trainAccuracy > peakTrainAccuracy
              ? "Training up, validation down"
              : "Training no higher, validation down",
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
          conceptHref: OVERFIT_HREF,
          conceptLabel: "What is overfitting?",
        };
      }

      // A gate that barely moved anything, and uncovered nothing either.
      if (gain < 0.01 && unlocked < 0.05) {
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
            irrelevant
              ? `${FEATURES[split.feature]!.name} is a reading that has nothing to do with whether the ground holds — no threshold on it will ever help, though on a small sample one will always look slightly better than the rest.`
              : `The best gate underneath it is worth ${lookahead.toFixed(
                  3,
                )}, and a single gate here could already buy ${parentBest.toFixed(
                  3,
                )}, so it has not uncovered any hidden structure either — only passed the same problem down a level.`
          } Prune it and try another feature.`,
          tone: "warn",
        };
      }

      // A gate with little immediate gain but real structure underneath — the
      // lesson of the ridge plot. Judged on what it UNLOCKED over the node's own
      // best gate, and never for a reading that carries no signal: an
      // irrelevant sliver inherits a big lookahead from the node it barely
      // touched, and used to be praised as "exactly the right gate".
      if (gain < 0.02 && unlocked >= 0.1 && !irrelevant) {
        return {
          key: `lookahead-gate-${describeSplit(split)}`,
          title: "Almost no gain now, a lot underneath",
          body: `${describeSplit(split)} gained only ${gain.toFixed(
            4,
          )} — a greedy learner would have rejected it. But it split ${
            bucket.length
          } plots into ${left.length} and ${
            right.length
          }, and the best gate available inside those two is now worth ${lookahead.toFixed(
            3,
          )}, against the ${parentBest.toFixed(
            3,
          )} the best single gate here could buy. That difference is what lookahead measures and gain cannot see. Real CART only scores the gate it is about to build, which is exactly why it cannot solve this shape and you can.`,
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

    case "greedy-grown": {
      const {
        round,
        depth,
        splits,
        trainAccuracy,
        validationAccuracy,
        peakValidation,
        peakDepth,
        peakTrainAccuracy,
        starved,
      } = event;
      const givingBack = givesGroundBack(event);
      // Below the peak on validation AND training: a tree with less shape
      // than the peak's, not one that has memorised more.
      const smaller =
        !givingBack &&
        peakValidation - validationAccuracy >= GIVING_BACK_DROP &&
        trainAccuracy < peakTrainAccuracy;
      return {
        key: `greedy-${depth}-${validationAccuracy.toFixed(3)}`,
        title: `Greedy CART, grown to depth ${depth}`,
        body: `${splits} gate${
          splits === 1 ? "" : "s"
        }, each the best immediate gain at its leaf — the choice "Take the greedy gate" makes, applied to every leaf at once. Training ${points(
          trainAccuracy,
        )}, validation ${points(validationAccuracy)}${
          givingBack
            ? ` — below the ${points(
                peakValidation,
              )} you reached at depth ${peakDepth}. ${
                starved > 0
                  ? `${starved} leaf${starved === 1 ? " holds" : "s hold"} fewer than ${MIN_HONEST_LEAF} plots. `
                  : ""
              }The gates past that depth are fitting these ${
                round.trainPoints
              } surveys, not the ground under them.`
            : smaller
              ? ` — short of the ${points(
                  peakValidation,
                )} you reached at depth ${peakDepth}, but training is short of that tree's ${points(
                  peakTrainAccuracy,
                )} as well, so this tree fits less rather than memorising more.`
              : "."
        }`,
        tone: givingBack ? "warn" : "info",
        ...(givingBack
          ? { conceptHref: OVERFIT_HREF, conceptLabel: "What is overfitting?" }
          : {}),
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
        ...(recovered > 0.001
          ? { conceptHref: OVERFIT_HREF, conceptLabel: "Why simpler generalises" }
          : {}),
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
            // The ghost can still be up on a win: a drop smaller than the
            // named failure's. Then the gap is not the story; the peak is.
            evaluation.givingGroundBack
              ? `but validation is ${points(
                  evaluation.peakValidation - evaluation.validationAccuracy,
                )} below the ${points(
                  evaluation.peakValidation,
                )} you had at depth ${evaluation.peakDepth}, so the gates since then bought nothing that holds outside this sample. The smaller tree was the better one.`
              : evaluation.trainAccuracy - evaluation.validationAccuracy <
                  NARROW_GAP
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
              ? `${evaluation.starved} leaf${
                  evaluation.starved === 1 ? " is" : "s are"
                } down to fewer than ${MIN_HONEST_LEAF} plots${
                  evaluation.starved === 1 ? "" : " each"
                }.`
              : ""
          } Prune back.`,
          tone: "bad",
          conceptHref: OVERFIT_HREF,
          conceptLabel: "Why complexity is a cost",
        };
      }

      if (evaluation.outcome === "underfit-stump") {
        const underfitGap = Math.max(
          0,
          evaluation.trainAccuracy - evaluation.validationAccuracy,
        );
        return {
          key: `underfit-${evaluation.splits}`,
          title: "Not enough tree",
          body: `Training accuracy ${points(
            evaluation.trainAccuracy,
          )} with ${evaluation.splits} gate${
            evaluation.splits === 1 ? "" : "s"
          } — the tree cannot describe this ground even where it has been given the answers. ${
            // Said only when the numbers say it: beside the ghost, "nothing is
            // about generalisation" contradicts the screen.
            evaluation.givingGroundBack
              ? `Validation has also slipped below the ${points(
                  evaluation.peakValidation,
                )} you had at depth ${
                  evaluation.peakDepth
                } while training has not fallen, so the gates since then describe these surveys rather than the ground — overfitting inside a tree that is still too simple. Add gates that follow the boundary, not more like those: look at the lookahead column as well as the gain.`
              : `${
                  underfitGap < NARROW_GAP
                    ? `Nothing here is about generalisation yet: the gap to validation is only ${points(
                        underfitGap,
                      )}, the signature of a model that is too simple rather than too complex.`
                    : `The gap to validation is ${points(
                        underfitGap,
                      )}, so some of these gates describe these surveys rather than the ground — but mostly the tree has too little of the right shape.`
                } Add gates: check the gain table on your largest leaf, and look at the lookahead column as well as the gain.`
          }`,
          tone: "bad",
          conceptHref: BOUNDARY_HREF,
          conceptLabel: "Bias and capacity",
        };
      }

      return {
        key: `missed-${evaluation.depth}-${evaluation.validationAccuracy.toFixed(3)}`,
        title: `Validation ${points(evaluation.validationAccuracy)}, needed ${percent(round.target)}`,
        body: `Depth ${evaluation.depth} of ${round.maxDepth} allowed, ${
          evaluation.splits
        } gates, training ${points(evaluation.trainAccuracy)}. ${
          evaluation.givingGroundBack
            ? `Validation is below the ${points(
                evaluation.peakValidation,
              )} you had at depth ${
                evaluation.peakDepth
              } while training has not fallen, so the gates since then are specific to these ${round.trainPoints} surveys. Pruning back toward depth ${
                evaluation.peakDepth
              } may raise validation even though it lowers training.`
            : evaluation.trainAccuracy - evaluation.validationAccuracy > 0.1
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
