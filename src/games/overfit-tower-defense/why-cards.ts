import type { WhyCardContent } from "@/components";
import {
  CORE_MAX_HP,
  SAMPLE_EVERY,
  TRAIN_EPOCHS,
  VALIDATION_POINTS,
  WAVES,
  failureSide,
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
 *
 * Result cards lead with the outcome and its numbers in the title. The title is
 * what screen readers are told, and "Wave 3 held: 21 damage, core 73" is the
 * sentence a player who cannot see the battlefield needs.
 */

/** One resolved wave, as the run remembers it. */
export interface WaveRecord {
  wave: number;
  damage: number;
  overfitDamage: number;
  underfitDamage: number;
  complexity: number;
  towers: Record<TowerType, number>;
}

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
  | {
      /** Several tower types changed at once — the code lane's `setTowers`. */
      kind: "towers-set";
      counts: Record<TowerType, number>;
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
      /** The player pressed Stop: nothing was scored. */
      kind: "stopped";
      wave: Wave;
      epochsRun: number;
      /** True when it was a Deploy, false for a trial. */
      deploy: boolean;
    }
  | {
      /** The fit threw before it could be measured: nothing was scored. */
      kind: "fit-failed";
      wave: Wave;
      deploy: boolean;
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
      /** Every wave of this run, this one included. */
      history: readonly WaveRecord[];
    };

/**
 * The Concept Library page these cards link to. Each label is one of its section
 * headings, verbatim — `concepts.test.ts` holds the link to that promise.
 */
const OVERFIT_HREF = "/concepts/overfitting";

const percent = (value: number) => `${Math.round(value * 100)}%`;

const TOWER_NAMES: Record<TowerType, string> = {
  l1: "L1",
  l2: "L2",
  dropout: "Dropout",
};

const TOWER_MECHANISM: Record<TowerType, string> = {
  l1: "L1 adds a penalty proportional to the size of each weight, pushing with the same force no matter how small the weight already is — so weights that aren't earning their keep are driven close to zero, far closer than L2 manages. That is what quiets the four distractor features. Close, not exactly: Adam keeps stepping back and forth across zero rather than landing on it.",
  l2: "L2 penalises the square of each weight, so the push weakens as a weight shrinks. Nothing quite reaches zero; everything just gets quieter and no single feature can dominate.",
  dropout:
    "Dropout switches off a random fraction of hidden units on every training step, so no unit can become load-bearing. The model is forced to spread the work, which is much harder to do by memorising individual points.",
};

function lambdas(regularization: Regularization): string {
  return `L1 ${regularization.l1.toFixed(3)}, L2 ${regularization.l2.toFixed(
    3,
  )}, dropout ${regularization.dropout.toFixed(2)}`;
}

function totalOf(towers: Record<TowerType, number>): number {
  return towers.l1 + towers.l2 + towers.dropout;
}

function sameSetup(a: WaveRecord, b: WaveRecord): boolean {
  return (
    a.complexity === b.complexity &&
    a.towers.l1 === b.towers.l1 &&
    a.towers.l2 === b.towers.l2 &&
    a.towers.dropout === b.towers.dropout
  );
}

/**
 * The victory card, written from what the player actually did.
 *
 * It used to congratulate everyone on "walking a moving optimum" — including a
 * run that never raised a tower. That run can survive at the default complexity,
 * and telling it that it tuned regularization would be teaching a lie.
 */
function wonCard(coreHp: number, history: readonly WaveRecord[]): WhyCardContent {
  const hp = `${Math.round(coreHp)} of ${CORE_MAX_HP}`;
  const worst = history.reduce<WaveRecord | null>(
    (most, record) => (most === null || record.damage > most.damage ? record : most),
    null,
  );
  const worstLine = worst
    ? `Wave ${worst.wave} cost the most, ${Math.round(worst.damage)} damage, ${
        worst.overfitDamage >= worst.underfitDamage
          ? `${Math.round(worst.overfitDamage)} of it from the train/validation gap`
          : `${Math.round(worst.underfitDamage)} of it from bias`
      }.`
    : "";

  if (history.length > 0 && history.every((record) => totalOf(record.towers) === 0)) {
    return {
      key: "run-won-undefended",
      title: `The core held — ${hp} HP, and not one tower raised`,
      body: `Five waves without any regularization. ${worstLine} That gap is the model memorising its training points, and it is exactly the damage regularization exists to buy back. Run it again with towers up where the data is thin and compare what is left of the core.`,
      tone: "good",
      conceptHref: OVERFIT_HREF,
      conceptLabel: "What is overfitting?",
    };
  }

  if (history.length > 1 && history.every((record) => sameSetup(record, history[0]!))) {
    return {
      key: "run-won-fixed",
      title: `The core held — ${hp} HP on one fixed setting`,
      body: `The same complexity and the same towers for all five waves. The data did not stay the same — ${WAVES.map(
        (wave) => wave.trainPoints,
      ).join(
        ", ",
      )} training points — and the right amount of regularization moves with it, so a single setting pays for too much defence on some waves and too little on others. ${worstLine}`,
      tone: "good",
      conceptHref: OVERFIT_HREF,
      conceptLabel: "Why simpler generalises",
    };
  }

  return {
    key: "run-won",
    title: `The core held — ${hp} HP`,
    body: `Five waves, and you changed the defence as the data changed. That is the real skill here: the right amount of regularization moved every wave because the amount of data moved, and there was never one setting that was correct throughout. ${worstLine} It is also why regularization is tuned per problem rather than set once in a library default.`,
    tone: "good",
    conceptHref: OVERFIT_HREF,
    conceptLabel: "Why simpler generalises",
  };
}

export function whyCardFor(event: DefenseEvent): WhyCardContent {
  switch (event.kind) {
    case "run-start":
      return {
        key: "run-start",
        title: "The core is your model's ability to generalise",
        body: `Two things attack it, and they are not really monsters — they are the two ways a model can be wrong. Overfit attackers are powered by the gap between your training and validation accuracy. Underfit attackers are powered by how far your model falls short even on data it has seen. Close one by overshooting and you feed the other, so watch both meters. ${event.wave.briefing}`,
        tone: "info",
        conceptHref: OVERFIT_HREF,
        conceptLabel: "What is overfitting?",
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
        conceptHref: OVERFIT_HREF,
        conceptLabel: "Why complexity is a cost",
      };
    }

    case "tower-added":
      return {
        key: `tower-add-${event.type}-${event.count}`,
        title: `${TOWER_NAMES[event.type]} tower ${event.count} deployed`,
        body: `${TOWER_MECHANISM[event.type]} Now at ${lambdas(
          event.regularization,
        )}. Expect training accuracy to fall. That is not the model getting worse — it is the model being stopped from memorising. The overfit attackers feed on the gap itself, so closing it cuts their damage even on a wave where validation accuracy barely moves; the underfit attackers feed on bias, so a tower that drags training accuracy below the ceiling hands the damage to them instead.`,
        tone: "info",
        conceptHref: OVERFIT_HREF,
        conceptLabel: "Why simpler generalises",
      };

    case "tower-removed":
      return {
        key: `tower-remove-${event.type}-${event.count}`,
        title: `${TOWER_NAMES[event.type]} tower withdrawn`,
        body: `${
          event.count === 0
            ? `No ${TOWER_NAMES[event.type]} left.`
            : `${event.count} ${TOWER_NAMES[event.type]} tower${event.count === 1 ? "" : "s"} still up.`
        } Now at ${lambdas(
          event.regularization,
        )}. Giving capacity back is the right move when the data can support it — ${event.wave.trainPoints} points this wave. Over-defending is a real failure mode, not a safe default.`,
        tone: "info",
      };

    case "towers-set":
      return {
        key: `towers-set-${event.counts.l1}-${event.counts.l2}-${event.counts.dropout}`,
        title: `Towers set: L1 ×${event.counts.l1}, L2 ×${event.counts.l2}, Dropout ×${event.counts.dropout}`,
        body: `Now at ${lambdas(
          event.regularization,
        )}. More towers pull training accuracy down; whether that buys back validation accuracy depends on how much data this wave has — ${event.wave.trainPoints} points. Train it to find out which way the trade went.`,
        tone: "info",
        conceptHref: OVERFIT_HREF,
        conceptLabel: "Why simpler generalises",
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
        title: `Trial: ${percent(event.validationAccuracy)} validation, gap ${percent(
          gap,
        )}`,
        body: `Train ${percent(event.trainAccuracy)}, validation ${percent(
          event.validationAccuracy,
        )} — gap ${percent(gap)}, bias ${percent(
          bias,
        )}. No damage taken: this was a trial, not a deployment. Comparing candidates on data the model has not trained on, and only then committing one, is how regularization actually gets chosen: you select on validation accuracy. The wave then charges the gap and the bias separately, so read both before you deploy.`,
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
            : `penalised by ${lambdas(event.regularization)}`
        }. The gap meter's two bars are accuracy on the ${
          event.wave.trainPoints
        } points it is fitting and on ${VALIDATION_POINTS} points held back from it, re-measured every ${SAMPLE_EVERY} epochs. Watch the moment they separate — that is overfitting happening, live.`,
        tone: "info",
      };

    case "stopped":
      return {
        key: `stopped-${event.deploy ? "deploy" : "trial"}-${event.epochsRun}`,
        title: `Stopped at epoch ${event.epochsRun} of ${TRAIN_EPOCHS} — not scored`,
        body: event.deploy
          ? `Wave ${event.wave.index} did not attack and the core took nothing. A half-trained model says nothing about your configuration, so its numbers were cleared rather than shown as a result. Deploy again to fight the wave.`
          : `A half-trained model says nothing about your configuration, so its numbers were cleared rather than shown as a trial result. Run the trial to the end to measure it.`,
        tone: "info",
      };

    case "fit-failed":
      // Not "stopped at epoch 0": the player did not press Stop, and saying so
      // would send them looking for something they did.
      return {
        key: `fit-failed-${event.deploy ? "deploy" : "trial"}`,
        title: "Training did not finish — not scored",
        body: event.deploy
          ? `The fit ended with an error before the model could be measured; the error is shown with the controls. Wave ${event.wave.index} did not attack and the core took nothing. Deploy again to fight the wave.`
          : `The fit ended with an error before the model could be measured, so there is no trial result; the error is shown with the controls. Run the trial again.`,
        tone: "warn",
      };

    case "wave-resolved": {
      const { wave, result, coreHp, destroyed, finished, regularization } = event;

      if (finished) return wonCard(coreHp, event.history);

      if (destroyed) {
        const side = failureSide(result);
        return {
          key: `core-lost-${side}`,
          title:
            side === "underfit"
              ? `Underfitting: the core fell on wave ${wave.index}`
              : `Overfitting: the core fell on wave ${wave.index}`,
          body:
            side === "underfit"
              ? `${percent(result.trainAccuracy)} on its own training data, short of the ${percent(
                  result.achievable,
                )} available — and a gap of only ${percent(
                  result.gap,
                )}, so nothing was being memorised. Bias did ${Math.round(
                  result.underfitDamage,
                )} damage this wave against ${Math.round(
                  result.overfitDamage,
                )} from the gap. The model was held too tightly to learn; more regularization here would have been exactly the wrong move.`
              : `${percent(result.trainAccuracy)} on training data against ${percent(
                  result.validationAccuracy,
                )} on data it had never seen — a ${percent(
                  result.gap,
                )} gap, worth ${Math.round(
                  result.overfitDamage,
                )} damage this wave against ${Math.round(
                  result.underfitDamage,
                )} from bias. The model learned things about these ${wave.trainPoints} points that were not true of anything else.`,
          tone: "bad",
          conceptHref: OVERFIT_HREF,
          conceptLabel: "What is overfitting?",
        };
      }

      const clean = result.damage < 4;
      const hp = Math.round(coreHp);
      return {
        key: `wave-${wave.index}-cleared`,
        title: clean
          ? `Wave ${wave.index} held: ${Math.round(result.damage)} damage, core ${hp}`
          : `Wave ${wave.index} survived: ${Math.round(result.damage)} damage, core ${hp}`,
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
              : clean
                ? `The two are close to balanced and both small, which is roughly where you want to be.`
                : `Gap and bias did about equal harm (${Math.round(
                    result.overfitDamage,
                  )} and ${Math.round(
                    result.underfitDamage,
                  )}): the model is memorising and falling short at once, so the fix is a setting that trades one for the other more cheaply, not simply more or fewer towers.`
        } ${hp} of ${CORE_MAX_HP} hit points left.`,
        tone: clean ? "good" : "warn",
        conceptHref: OVERFIT_HREF,
        conceptLabel: "What is overfitting?",
      };
    }
  }
}
