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
import { whyCardFor } from "./why-cards";

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

  // ── actions ──────────────────────────────────────────────────────────
  setLane: (lane: Lane) => void;
  /** The bias-variance knob (spec: `<ComplexitySlider>`). */
  setComplexity: (complexity: number) => void;
  addTower: (type: TowerType) => void;
  removeTower: (type: TowerType) => void;
  clearTowers: () => void;

  /** Set every tower count at once — the code lane's entry point. */
  setTowers: (counts: Partial<Record<TowerType, number>>) => void;

  beginTraining: () => void;
  recordProgress: (sample: ProgressSample) => void;
  /**
   * Record a measurement WITHOUT resolving the wave.
   *
   * Trying a configuration and scoring a wave with it are different acts. A
   * practitioner compares candidates on validation data before committing one,
   * and if every trial cost the core its health the game would punish exactly the
   * habit it wants to teach.
   */
  finishTrial: (measured: {
    trainAccuracy: number;
    validationAccuracy: number;
  }) => void;
  /** Resolve the wave against the trained model. */
  resolveWave: (measured: {
    trainAccuracy: number;
    validationAccuracy: number;
  }) => WaveResult;
  nextWave: () => void;
  /** Start the run over, keeping the current tuning. */
  restart: () => void;
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
    coreHp: CORE_MAX_HP,
    wavesCleared: 0,
    modelComplexity: complexity,
    towers,
    dataset: generateDataset(wave.trainPoints, wave.noiseRate, DATA_SEED),
    ...untrained(),
    lastEvaluation: null,
    failure: null,
  };
}

export const useTowerDefenseStore = create<TowerDefenseState>((set, get) => ({
  ...freshRun(DEFAULT_COMPLEXITY, []),
  whyCard: whyCardFor({ kind: "run-start", wave: waveAt(1) }),
  lane: "visual" as Lane,

  setLane: (lane) => set({ lane }),

  setComplexity: (complexity) => {
    const state = get();
    if (state.phase === "training") return;
    const next = clamp(Math.round(complexity), MIN_COMPLEXITY, MAX_COMPLEXITY);
    if (next === state.modelComplexity) return;

    set({
      modelComplexity: next,
      ...untrained(),
      phase: state.phase === "resolved" ? "resolved" : "tuning",
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
      whyCard: whyCardFor({
        kind: totalTowers(towers) === 0 ? "towers-cleared" : "tower-added",
        type: "l2",
        count: towerCount({ ...state, towers }, "l2"),
        regularization: regularizationOf(towers),
        wave: waveAt(state.wave),
      }),
    });
  },

  finishTrial: ({ trainAccuracy, validationAccuracy }) => {
    const state = get();
    set({
      phase: "tuning",
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
    set({
      phase: "training",
      ...untrained(),
      whyCard: whyCardFor({
        kind: "training",
        wave: waveAt(state.wave),
        complexity: state.modelComplexity,
        regularization: regularizationOf(state.towers),
      }),
    });
  },

  recordProgress: (sample) => {
    set((state) => ({
      epoch: sample.epoch,
      progress: [...state.progress, sample],
      // Show the live numbers as they arrive: this is the always-visible metric.
      trainAccuracy: sample.trainAccuracy,
      validationAccuracy: sample.validationAccuracy,
    }));
  },

  resolveWave: ({ trainAccuracy, validationAccuracy }) => {
    const state = get();
    const wave = waveAt(state.wave);

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
    const wavesCleared = destroyed ? state.wavesCleared : state.wavesCleared + 1;
    const finished = !destroyed && lastWave;

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
      }),
    });

    if (finished) {
      useProgression.getState().recordResult({
        slug: SLUG,
        score: evaluation.score,
        lane: state.lane,
        completed: true,
        codeLaneCleared: state.lane === "code",
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
      whyCard: whyCardFor({ kind: "run-start", wave: waveAt(1) }),
    });
  },
}));

export { CORE_MAX_HP, totalTowers };
export type { Evaluation, Tower, TowerType, Wave, WaveResult };
