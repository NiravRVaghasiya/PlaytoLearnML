"use client";

import { create } from "zustand";
import { useProgression } from "@/engine/progression";
import type { Lane, NamedFailure } from "@/engine/types";
import type { WhyCardContent } from "@/components";
import { clamp } from "@/lib/utils";
import {
  CORE_MAX_HP,
  DEFAULT_COMPLEXITY,
  MAX_COMPLEXITY,
  MAX_TOWERS_PER_TYPE,
  MIN_COMPLEXITY,
  TOWER_TYPES,
  WAVES,
  evaluateRun,
  generateDataset,
  regularizationOf,
  scoreWave,
  totalTowers,
  waveAt,
  type Dataset,
  type Evaluation,
  type Tower,
  type TowerType,
  type Wave,
  type WaveResult,
} from "./ml";
import { whyCardFor, type DefenseEvent, type WaveRecord } from "./why-cards";

export const SLUG = "overfit-tower-defense";

/** Same seed the tests measure against, so the game behaves as tuned. */
export const DATA_SEED = 5150;

/** Where the run is. Drives which controls are live. */
export type Phase = "tuning" | "training" | "resolved" | "won" | "lost";

export interface ProgressSample {
  epoch: number;
  trainAccuracy: number;
  validationAccuracy: number;
}

/**
 * Overfit Tower Defense state.
 *
 * The spec's data model is:
 *
 *   Enemy     { type:'underfit'|'overfit', strength }
 *   Tower     { type:'L1'|'L2'|'dropout', strength }
 *   GameState { modelComplexity, towers[], trainAcc, valAcc, wave, coreHP }
 *
 * `Enemy` is not stored. Enemy strength IS the measured error — an overfit
 * enemy's strength is the train/validation gap and an underfit enemy's is the
 * bias — so storing it separately would create a second copy of a number that
 * must never disagree with the model. `WaveResult` carries both, derived.
 */
export interface TowerDefenseState {
  wave: number;
  phase: Phase;
  coreHp: number;
  wavesCleared: number;

  modelComplexity: number;
  towers: Tower[];

  dataset: Dataset;

  /** Null until the wave's model has been trained. */
  trainAccuracy: number | null;
  validationAccuracy: number | null;
  epoch: number;
  /** Train/validation accuracy sampled during training, for the live meter. */
  progress: ProgressSample[];

  lastResult: WaveResult | null;
  lastEvaluation: Evaluation | null;
  failure: NamedFailure | null;
  whyCard: WhyCardContent | null;
  lane: Lane;

  /**
   * Which training run is current. Bumped by `beginTraining` and by `restart`,
   * and handed back by the trainer when a fit resolves, so a fit that was
   * running when the player pressed Retry can never resolve the fresh run's
   * wave. Before this, it did: the stale fit finished, `resolveWave` applied its
   * numbers to the new run, and wave 1 cleared itself.
   */
  attempt: number;
  /**
   * The phase a training run started from, restored when it ends. A trial
   * started after a wave resolved must leave the wave resolved — it used to drop
   * the run back to "tuning", which re-opened the wave to a second Deploy and
   * counted it twice.
   */
  phaseBeforeTraining: Phase;
  /** Every wave resolved in this run, for the victory card. */
  history: WaveRecord[];
  /**
   * True while every wave of this run was deployed by `api.deploy()`. Mastery's
   * third star is "clear a run from the code lane", so it is earned by the path
   * that did the work, not by which tab was open when the rail's button was hit.
   */
  runFromCode: boolean;

  // ── actions ──────────────────────────────────────────────────────────
  setLane: (lane: Lane) => void;
  /** The bias-variance knob (spec: `<ComplexitySlider>`). */
  setComplexity: (complexity: number) => void;
  addTower: (type: TowerType) => void;
  removeTower: (type: TowerType) => void;
  clearTowers: () => void;

  /** Set every tower count at once — the code lane's entry point. */
  setTowers: (counts: Partial<Record<TowerType, number>>) => void;

  /** Returns the run's `attempt` token, for the calls that end it. */
  beginTraining: () => number;
  /** A live sample. Dropped unless `attempt` is the run that is training. */
  recordProgress: (sample: ProgressSample, attempt?: number) => void;
  /**
   * Record a measurement WITHOUT resolving the wave.
   *
   * Trying a configuration and scoring a wave with it are different acts. A
   * practitioner compares candidates on validation data before committing one,
   * and if every trial cost the core its health the game would punish exactly the
   * habit it wants to teach.
   */
  finishTrial: (measured: Measured, attempt?: number) => void;
  /**
   * Resolve the wave against the trained model.
   *
   * Only an open wave resolves: null — and no change — when the wave has already
   * resolved, the run is over, or `attempt` names a superseded run. That makes a
   * second Deploy, or the starter snippet run twice, unable to count a wave twice.
   */
  resolveWave: (
    measured: Measured,
    attempt?: number,
    options?: { fromCode?: boolean },
  ) => WaveResult | null;
  /**
   * The fit did not finish — the player stopped it, or it failed (`failed`).
   * Nothing is scored and the phase is restored either way; the card says which.
   */
  abortTraining: (
    attempt: number,
    ended: { epochsRun: number; deploy: boolean; failed?: boolean },
  ) => void;
  nextWave: () => void;
  /** Start the run over, keeping the current tuning. */
  restart: () => void;
}

export interface Measured {
  trainAccuracy: number;
  validationAccuracy: number;
}

// ── selectors: primitives and stable references only ─────────────────────

export function currentWave(state: TowerDefenseState): Wave {
  return waveAt(state.wave);
}

export function gapOf(state: TowerDefenseState): number {
  if (state.trainAccuracy === null || state.validationAccuracy === null) {
    return Number.NaN;
  }
  return Math.max(0, state.trainAccuracy - state.validationAccuracy);
}

export function biasOf(state: TowerDefenseState): number {
  if (state.trainAccuracy === null) return Number.NaN;
  return Math.max(0, state.dataset.achievable - state.trainAccuracy);
}

export function towerCount(state: TowerDefenseState, type: TowerType): number {
  return (
    state.towers.find((tower) => tower.type === type)?.strength ?? 0
  );
}

/**
 * NOT exported as a selector on purpose.
 *
 * `regularizationOf` builds a new object, and a zustand selector that returns a
 * fresh reference each call violates React's requirement that
 * `useSyncExternalStore` snapshots be cached — it either re-renders on every
 * store write or throws outright. Components select `towers`, which is a stable
 * array reference, and memoise this themselves.
 */

function countsOf(towers: Tower[]): Record<TowerType, number> {
  const count = (type: TowerType) =>
    towers.find((tower) => tower.type === type)?.strength ?? 0;
  return { l1: count("l1"), l2: count("l2"), dropout: count("dropout") };
}

/**
 * The why-card for a bulk tower change, from what actually changed.
 *
 * `setTowers({ l1: 3 })` used to announce "L2 tower 0 deployed" with L2's
 * mechanism, whatever the script had set.
 */
function towerChangeEvent(
  before: Tower[],
  after: Tower[],
  wave: Wave,
): DefenseEvent {
  const from = countsOf(before);
  const to = countsOf(after);
  const regularization = regularizationOf(after);
  if (totalTowers(after) === 0) return { kind: "towers-cleared", wave };

  const changed = TOWER_TYPES.filter((type) => from[type] !== to[type]);
  const [only] = changed;
  if (changed.length === 1 && only !== undefined) {
    return {
      kind: to[only] > from[only] ? "tower-added" : "tower-removed",
      type: only,
      count: to[only],
      regularization,
      wave,
    };
  }
  return { kind: "towers-set", counts: to, regularization, wave };
}

function withTowerCount(
  towers: Tower[],
  type: TowerType,
  strength: number,
): Tower[] {
  const next = towers.filter((tower) => tower.type !== type);
  if (strength > 0) next.push({ type, strength });
  // Keep a stable order so the palette never reshuffles under the player.
  return next.sort((a, b) => a.type.localeCompare(b.type));
}

/** Results cleared: no stale accuracy may sit on screen after a retune. */
function untrained() {
  return {
    trainAccuracy: null,
    validationAccuracy: null,
    epoch: 0,
    progress: [] as ProgressSample[],
    lastResult: null,
  };
}

function freshRun(complexity: number, towers: Tower[]) {
  const wave = waveAt(1);
  return {
    wave: 1,
    phase: "tuning" as Phase,
    phaseBeforeTraining: "tuning" as Phase,
    coreHp: CORE_MAX_HP,
    wavesCleared: 0,
    modelComplexity: complexity,
    towers,
    dataset: generateDataset(wave.trainPoints, wave.noiseRate, DATA_SEED),
    ...untrained(),
    lastEvaluation: null,
    failure: null,
    history: [] as WaveRecord[],
    runFromCode: true,
  };
}

export const useTowerDefenseStore = create<TowerDefenseState>((set, get) => ({
  ...freshRun(DEFAULT_COMPLEXITY, []),
  whyCard: whyCardFor({ kind: "run-start", wave: waveAt(1) }),
  lane: "visual" as Lane,
  attempt: 0,

  setLane: (lane) => set({ lane }),

  setComplexity: (complexity) => {
    const state = get();
    if (state.phase === "training") return;
    const next = clamp(Math.round(complexity), MIN_COMPLEXITY, MAX_COMPLEXITY);
    if (next === state.modelComplexity) return;

    set({
      modelComplexity: next,
      ...untrained(),
      // The phase is left alone. Retuning after a wave resolves, or after the
      // run ends, must not re-open a wave that has already been fought: moving
      // the slider on the victory screen used to hand back a Deploy button for
      // wave 5, and a sixth "cleared" wave.
      whyCard: whyCardFor({
        kind: "complexity-changed",
        complexity: next,
        previous: state.modelComplexity,
        wave: waveAt(state.wave),
      }),
    });
  },

  addTower: (type) => {
    const state = get();
    if (state.phase === "training") return;
    const current = towerCount(state, type);
    if (current >= MAX_TOWERS_PER_TYPE) return;

    const towers = withTowerCount(state.towers, type, current + 1);
    set({
      towers,
      ...untrained(),
      whyCard: whyCardFor({
        kind: "tower-added",
        type,
        count: current + 1,
        regularization: regularizationOf(towers),
        wave: waveAt(state.wave),
      }),
    });
  },

  removeTower: (type) => {
    const state = get();
    if (state.phase === "training") return;
    const current = towerCount(state, type);
    if (current <= 0) return;

    const towers = withTowerCount(state.towers, type, current - 1);
    set({
      towers,
      ...untrained(),
      whyCard: whyCardFor({
        kind: "tower-removed",
        type,
        count: current - 1,
        regularization: regularizationOf(towers),
        wave: waveAt(state.wave),
      }),
    });
  },

  clearTowers: () => {
    const state = get();
    if (state.phase === "training") return;
    set({
      towers: [],
      ...untrained(),
      whyCard: whyCardFor({
        kind: "towers-cleared",
        wave: waveAt(state.wave),
      }),
    });
  },

  setTowers: (counts) => {
    const state = get();
    if (state.phase === "training") return;

    let towers = state.towers;
    for (const type of ["l1", "l2", "dropout"] as TowerType[]) {
      const requested = counts[type];
      if (requested === undefined) continue;
      towers = withTowerCount(
        towers,
        type,
        clamp(Math.round(requested), 0, MAX_TOWERS_PER_TYPE),
      );
    }

    set({
      towers,
      ...untrained(),
      whyCard: whyCardFor(
        towerChangeEvent(state.towers, towers, waveAt(state.wave)),
      ),
    });
  },

  finishTrial: ({ trainAccuracy, validationAccuracy }, attempt) => {
    const state = get();
    if (attempt !== undefined && attempt !== state.attempt) return;
    set({
      phase: state.phase === "training" ? state.phaseBeforeTraining : state.phase,
      trainAccuracy,
      validationAccuracy,
      whyCard: whyCardFor({
        kind: "trial",
        wave: waveAt(state.wave),
        trainAccuracy,
        validationAccuracy,
        achievable: state.dataset.achievable,
      }),
    });
  },

  beginTraining: () => {
    const state = get();
    const attempt = state.attempt + 1;
    set({
      attempt,
      // A second beginTraining while one is in flight keeps the original phase.
      phaseBeforeTraining:
        state.phase === "training" ? state.phaseBeforeTraining : state.phase,
      phase: "training",
      ...untrained(),
      whyCard: whyCardFor({
        kind: "training",
        wave: waveAt(state.wave),
        complexity: state.modelComplexity,
        regularization: regularizationOf(state.towers),
      }),
    });
    return attempt;
  },

  recordProgress: (sample, attempt) => {
    const current = get();
    if (current.phase !== "training") return;
    if (attempt !== undefined && attempt !== current.attempt) return;
    set((state) => ({
      epoch: sample.epoch,
      progress: [...state.progress, sample],
      // Show the live numbers as they arrive: this is the always-visible metric.
      trainAccuracy: sample.trainAccuracy,
      validationAccuracy: sample.validationAccuracy,
    }));
  },

  abortTraining: (attempt, { epochsRun, deploy, failed = false }) => {
    const state = get();
    if (attempt !== state.attempt || state.phase !== "training") return;
    const wave = waveAt(state.wave);
    set({
      phase: state.phaseBeforeTraining,
      // The live samples were from a half-trained model. Leaving them on the
      // meters would present them as a measurement of this configuration.
      ...untrained(),
      whyCard: whyCardFor(
        failed
          ? { kind: "fit-failed", wave, deploy }
          : { kind: "stopped", wave, epochsRun, deploy },
      ),
    });
  },

  resolveWave: ({ trainAccuracy, validationAccuracy }, attempt, options) => {
    const state = get();
    if (attempt !== undefined && attempt !== state.attempt) return null;
    const open =
      state.phase === "training"
        ? state.phaseBeforeTraining === "tuning"
        : state.phase === "tuning";
    if (!open) return null;

    const wave = waveAt(state.wave);
    const fromCode = options?.fromCode === true;

    const result = scoreWave({
      wave,
      trainAccuracy,
      validationAccuracy,
      achievable: state.dataset.achievable,
      coreHp: state.coreHp,
    });

    const coreHp = Math.max(0, state.coreHp - result.damage);
    const destroyed = coreHp <= 0;
    const lastWave = state.wave >= WAVES.length;
    const wavesCleared = destroyed
      ? state.wavesCleared
      : Math.min(WAVES.length, state.wavesCleared + 1);
    const finished = !destroyed && lastWave;
    const runFromCode = state.runFromCode && fromCode;
    const history: WaveRecord[] = [
      ...state.history,
      {
        wave: wave.index,
        damage: result.damage,
        overfitDamage: result.overfitDamage,
        underfitDamage: result.underfitDamage,
        complexity: state.modelComplexity,
        towers: countsOf(state.towers),
      },
    ];

    const evaluation = evaluateRun({
      result,
      coreHp,
      wavesCleared,
      complexity: state.modelComplexity,
      towers: state.towers,
      finished,
    });

    const phase: Phase = destroyed ? "lost" : finished ? "won" : "resolved";

    set({
      trainAccuracy,
      validationAccuracy,
      coreHp,
      wavesCleared,
      phase,
      phaseBeforeTraining: phase,
      history,
      runFromCode,
      lastResult: result,
      lastEvaluation: evaluation,
      failure: evaluation.failure,
      whyCard: whyCardFor({
        kind: "wave-resolved",
        wave,
        result,
        coreHp,
        destroyed,
        finished,
        regularization: regularizationOf(state.towers),
        complexity: state.modelComplexity,
        history,
      }),
    });

    if (finished) {
      useProgression.getState().recordResult({
        slug: SLUG,
        score: evaluation.score,
        lane: runFromCode ? "code" : "visual",
        completed: true,
        codeLaneCleared: runFromCode,
      });
    }

    return result;
  },

  nextWave: () => {
    const state = get();
    if (state.phase !== "resolved") return;
    const wave = waveAt(state.wave + 1);
    set({
      wave: wave.index,
      phase: "tuning",
      phaseBeforeTraining: "tuning",
      dataset: generateDataset(wave.trainPoints, wave.noiseRate, DATA_SEED),
      ...untrained(),
      failure: null,
      whyCard: whyCardFor({ kind: "wave-briefing", wave }),
    });
  },

  restart: () => {
    const state = get();
    set({
      ...freshRun(state.modelComplexity, state.towers),
      // Retires any fit still in flight: its result can no longer land here.
      attempt: state.attempt + 1,
      whyCard: whyCardFor({ kind: "run-start", wave: waveAt(1) }),
    });
  },
}));

export { CORE_MAX_HP, totalTowers };
export type { Evaluation, Tower, TowerType, Wave, WaveResult };
