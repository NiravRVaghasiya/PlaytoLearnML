import type { WhyCardContent } from "@/components";
import {
  CORE_MAX_HP,
  parameterCount,
  type Regularization,
  type TowerType,
  type Wave,
  type WaveResult,
} from "./ml";

/**
 * "Why did that happen?" copy for Overfit Tower Defense.
 *
 * The rule this copy follows: never say "the model got worse" without saying
 * WHICH way it got worse. Every sentence about damage names either the gap or the
 * bias, because those two have opposite fixes and a player who reads "worse" and
 * guesses will be wrong half the time.
 *
 * It also avoids calling regularization "making the model weaker". A regularised
 * model is usually the STRONGER one — it just scores lower on the data it was
 * shown, which is exactly the trade the game is about.
 */
export type DefenseEvent =
  | { kind: "run-start"; wave: Wave }
  | { kind: "wave-briefing"; wave: Wave }
  | {
      kind: "complexity-changed";
      complexity: number;
      previous: number;
      wave: Wave;
    }
  | {
      kind: "tower-added";
      type: TowerType;
      count: number;
      regularization: Regularization;
      wave: Wave;
    }
  | {
      kind: "tower-removed";
      type: TowerType;
      count: number;
      regularization: Regularization;
      wave: Wave;
    }
  | { kind: "towers-cleared"; wave: Wave }
  | {
      kind: "trial";
      wave: Wave;
      trainAccuracy: number;
      validationAccuracy: number;
      achievable: number;
    }
  | {
      kind: "training";
      wave: Wave;
      complexity: number;
      regularization: Regularization;
    }
  | {
      kind: "wave-resolved";
      wave: Wave;
      result: WaveResult;
      coreHp: number;
      destroyed: boolean;
      finished: boolean;
      regularization: Regularization;
      complexity: number;
    };

const percent = (value: number) => `${Math.round(value * 100)}%`;

const TOWER_NAMES: Record<TowerType, string> = {
  l1: "L1",
  l2: "L2",
  dropout: "Dropout",
};

const TOWER_MECHANISM: Record<TowerType, string> = {
  l1: "L1 adds a penalty proportional to the size of each weight, pushing with the same force no matter how small the weight already is — so weights that aren't earning their keep get driven to exactly zero. That is what lets it switch off the four distractor features entirely.",
  l2: "L2 penalises the square of each weight, so the push weakens as a weight shrinks. Nothing quite reaches zero; everything just gets quieter and no single feature can dominate.",
  dropout:
    "Dropout switches off a random fraction of hidden units on every training step, so no unit can become load-bearing. The model is forced to spread the work, which is much harder to do by memorising individual points.",
};

export function whyCardFor(event: DefenseEvent): WhyCardContent {
  switch (event.kind) {
    case "run-start":
      return {
        key: "run-start",
        title: "The core is your model's ability to generalise",
        body: `Two things attack it, and they are not really monsters — they are the two ways a model can be wrong. Overfit attackers are powered by the gap between your training and validation accuracy. Underfit attackers are powered by how far your model falls short even on data it has seen. Close one by overshooting and you feed the other, so watch both meters. ${event.wave.briefing}`,
        tone: "info",
      };

    case "wave-briefing":
      return {
        key: `wave-${event.wave.index}`,
        title: `Wave ${event.wave.index}: ${event.wave.name}`,
        body: `${event.wave.briefing} You get ${event.wave.trainPoints} training points with ${percent(
          event.wave.noiseRate,
        )} of the labels wrong, so ${percent(
          1 - event.wave.noiseRate,
        )} is the most any honest model can score. Beating that number on your training data is not skill — it is memorisation, and the gap will show it.`,
        tone: "info",
      };

    case "complexity-changed": {
      const { complexity, previous, wave } = event;
      const grew = complexity > previous;
      return {
        key: `complexity-${complexity}`,
        title: `${complexity} hidden units — ${parameterCount(complexity)} parameters`,
        body: grew
          ? `More capacity. With ${wave.trainPoints} training points that is now ${(
              parameterCount(complexity) / wave.trainPoints
            ).toFixed(
              2,
            )} parameters per point — the higher that ratio, the more room the model has to describe individual points instead of the pattern behind them. Capacity is not bad in itself; unrestrained capacity on thin data is.`
          : `Less capacity. Fewer parameters means less room to memorise, but a model that is too small cannot represent the boundary at all — and that shows up as bias, not as a gap. There is a floor below which you are simply making it worse.`,
        tone: "info",
      };
    }

    case "tower-added":
      return {
        key: `tower-add-${event.type}-${event.count}`,
        title: `${TOWER_NAMES[event.type]} tower ${event.count} deployed`,
        body: `${TOWER_MECHANISM[event.type]} Now at L1 ${event.regularization.l1.toFixed(
          3,
        )}, L2 ${event.regularization.l2.toFixed(
          3,
        )}, dropout ${event.regularization.dropout.toFixed(
          2,
        )}. Expect training accuracy to fall. That is not the model getting worse — it is the model being stopped from memorising, and validation accuracy is the number that tells you whether the trade paid off.`,
        tone: "info",
      };

    case "tower-removed":
      return {
        key: `tower-remove-${event.type}-${event.count}`,
        title: `${TOWER_NAMES[event.type]} tower withdrawn`,
        body: `${
          event.count === 0
            ? `No ${TOWER_NAMES[event.type]} left.`
            : `${event.count} ${TOWER_NAMES[event.type]} tower${event.count === 1 ? "" : "s"} still up.`
        } Now at L1 ${event.regularization.l1.toFixed(3)}, L2 ${event.regularization.l2.toFixed(
          3,
        )}, dropout ${event.regularization.dropout.toFixed(
          2,
        )}. Giving capacity back is the right move when the data can support it — ${event.wave.trainPoints} points this wave. Over-defending is a real failure mode, not a safe default.`,
        tone: "info",
      };

    case "towers-cleared":
      return {
        key: "towers-cleared",
        title: "All towers down",
        body: `The model is now free to fit ${event.wave.trainPoints} points however it likes, including the ${percent(
          event.wave.noiseRate,
        )} of them that are mislabelled. Worth doing deliberately once, to see the gap open up.`,
        tone: "warn",
      };

    case "trial": {
      const gap = Math.max(0, event.trainAccuracy - event.validationAccuracy);
      const bias = Math.max(0, event.achievable - event.trainAccuracy);
      return {
        key: `trial-${event.trainAccuracy}-${event.validationAccuracy}`,
        title: `Trial: ${percent(event.validationAccuracy)} on held-out data`,
        body: `Train ${percent(event.trainAccuracy)}, validation ${percent(
          event.validationAccuracy,
        )} — gap ${percent(gap)}, bias ${percent(
          bias,
        )}. No damage taken: this was a trial, not a deployment. Comparing candidates on data the model has not trained on, and only then committing one, is how regularization actually gets chosen — the gap is the thing you are selecting on.`,
        tone: "info",
      };
    }

    case "training":
      return {
        key: "training",
        title: "Training under fire",
        body: `${parameterCount(event.complexity)} parameters, ${
          event.regularization.l1 + event.regularization.l2 === 0 &&
          event.regularization.dropout === 0
            ? "no penalty on the weights at all"
            : `penalised by L1 ${event.regularization.l1.toFixed(3)}, L2 ${event.regularization.l2.toFixed(
                3,
              )}, dropout ${event.regularization.dropout.toFixed(2)}`
        }. The two lines being drawn are accuracy on the data it is fitting and accuracy on ${
          event.wave.trainPoints
        } points held back from it. Watch the moment they separate — that is overfitting happening, live.`,
        tone: "info",
      };

    case "wave-resolved": {
      const { wave, result, coreHp, destroyed, finished, regularization } = event;

      if (finished) {
        return {
          key: "run-won",
          title: "The core held",
          body: `Five waves, and it survived with ${Math.round(coreHp)} of ${CORE_MAX_HP} hit points. What you actually did was walk a moving optimum: the right amount of regularization changed every wave because the amount of data changed, and there was never one setting that was correct throughout. That is the whole reason this is tuned per problem rather than set once in a library default.`,
          tone: "good",
        };
      }

      if (destroyed) {
        return {
          key: `core-lost-${result.dominant}`,
          title:
            result.dominant === "underfit"
              ? "The core fell to the underfit wave"
              : "The core fell to the overfit wave",
          body:
            result.dominant === "underfit"
              ? `${percent(result.trainAccuracy)} on its own training data, short of the ${percent(
                  result.achievable,
                )} available — and a gap of only ${percent(
                  result.gap,
                )}, so nothing was being memorised. The model was held too tightly to learn. More regularization here would have been exactly the wrong move.`
              : `${percent(result.trainAccuracy)} on training data against ${percent(
                  result.validationAccuracy,
                )} on data it had never seen — a ${percent(
                  result.gap,
                )} gap, worth ${Math.round(
                  result.overfitDamage,
                )} damage this wave. The model learned things about these ${wave.trainPoints} points that were not true of anything else.`,
          tone: "bad",
        };
      }

      const clean = result.damage < 4;
      return {
        key: `wave-${wave.index}-cleared`,
        title: clean
          ? `Wave ${wave.index} held for ${Math.round(result.damage)} damage`
          : `Wave ${wave.index} survived, ${Math.round(result.damage)} damage taken`,
        body: `Train ${percent(result.trainAccuracy)}, validation ${percent(
          result.validationAccuracy,
        )} — a gap of ${percent(result.gap)} and a bias of ${percent(
          result.bias,
        )}. ${
          result.dominant === "overfit"
            ? `The gap did most of the harm (${Math.round(
                result.overfitDamage,
              )} against ${Math.round(
                result.underfitDamage,
              )}), so there is still memorisation to squeeze out${
                regularization.l1 + regularization.l2 + regularization.dropout === 0
                  ? " — and no towers up yet."
                  : "."
              }`
            : result.dominant === "underfit"
              ? `Bias did most of the harm (${Math.round(
                  result.underfitDamage,
                )} against ${Math.round(
                  result.overfitDamage,
                )}), which means the model is being held back rather than running loose. Easing off is the move.`
              : `The two are close to balanced, which is roughly where you want to be.`
        } ${Math.round(coreHp)} of ${CORE_MAX_HP} hit points left.`,
        tone: clean ? "good" : "warn",
      };
    }
  }
}
