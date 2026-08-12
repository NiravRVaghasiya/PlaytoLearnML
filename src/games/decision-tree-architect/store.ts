"use client";

import { create } from "zustand";
import { useProgression } from "@/engine/progression";
import type { Lane, NamedFailure } from "@/engine/types";
import type { WhyCardContent } from "@/components";
import {
  ROOT_ID,
  ROUNDS,
  accuracyOf,
  applySplit,
  bestSplit,
  depthOf,
  emptyTree,
  evaluateTree,
  generateDataset,
  nodeStats,
  pruneAt,
  roundAt,
  routeSample,
  splitCountOf,
  starvedLeaves,
  type Dataset,
  type Evaluation,
  type Round,
  type Sample,
  type Split,
  type Tree,
  type TreeNode,
} from "./ml";
import { whyCardFor } from "./why-cards";

export const SLUG = "decision-tree-architect";

/** Same seed the tests measure the plots against. */
export const DATA_SEED = 4417;

export type Phase = "building" | "cleared" | "complete";

/**
 * Decision Tree Architect state.
 *
 * The spec's data model is:
 *
 *   TreeNode  { id, feature, threshold, gini, left, right, samples[] }
 *   GameState { tree, trainAcc, valAcc, depth }
 *
 * `gini` and `samples[]` are not stored on nodes — both are exact functions of
 * the tree and the dataset, and a stored copy is a second version of a number
 * that must never disagree with the gate above it. `nodeStats` derives them.
 *
 * `trainAccuracy`, `validationAccuracy` and `depth` ARE stored, unlike the
 * per-node figures. They are needed by four components at once and by the peak
 * tracking below, and they are written in exactly one place — `withTree` — so
 * there is no path that can update the tree without updating them.
 */
export interface ArchitectState {
  roundIndex: number;
  dataset: Dataset;
  tree: Tree;

  /** Which node the split picker is aimed at. */
  selectedNodeId: string;

  trainAccuracy: number;
  validationAccuracy: number;
  depth: number;
  splits: number;
  starved: number;

  /**
   * The best validation accuracy this player has reached on this plot, and the
   * tree shape that got there. Tracked live while they build, because "you
   * already had 92.7% at depth 3" is a far harder thing to argue with than a
   * comparison against an optimum they never saw.
   */
  peakValidation: number;
  peakDepth: number;
  peakTrainAccuracy: number;

  evaluation: Evaluation | null;
  attempts: number;
  clearedScores: number[];

  phase: Phase;
  failure: NamedFailure | null;
  whyCard: WhyCardContent | null;
  lane: Lane;

  setLane: (lane: Lane) => void;
  selectNode: (nodeId: string) => void;
  /** Build a gate at a leaf (spec: `<SplitGate>`). */
  splitAt: (nodeId: string, split: Split) => void;
  /** Take whatever greedy CART would take here. */
  takeGreedySplit: (nodeId: string) => void;
  /** Remove everything below a node. */
  prune: (nodeId: string) => void;
  clearTree: () => void;
  /** Commit the tree for inspection. */
  signOff: () => Evaluation;
  nextRound: () => void;
  restart: () => void;
}

// ── selectors: primitives and stable references only ─────────────────────

export function currentRound(state: ArchitectState): Round {
  return roundAt(state.roundIndex);
}

export function clearedCount(state: ArchitectState): number {
  return state.clearedScores.length;
}

/** Is validation currently below the best this player reached? */
export function isGivingGroundBack(state: ArchitectState): boolean {
  return (
    state.peakValidation > 0 &&
    state.peakValidation - state.validationAccuracy >= 0.01 &&
    state.trainAccuracy >= state.peakTrainAccuracy
  );
}

/**
 * The training samples reaching one node. Derived, never stored.
 *
 * NOT usable as a zustand selector: it builds a new array, and a selector
 * returning a fresh reference each call breaks React's requirement that
 * `useSyncExternalStore` snapshots be cached. Components select `tree` and
 * `dataset.train` and memoise this themselves.
 */
export function samplesInNode(
  tree: Tree,
  train: Sample[],
  nodeId: string,
): Sample[] {
  return train.filter((sample) => routeSampleThrough(tree, sample, nodeId));
}

/** Convenience for non-render callers, which have no snapshot contract. */
export function samplesAt(state: ArchitectState, nodeId: string): Sample[] {
  return samplesInNode(state.tree, state.dataset.train, nodeId);
}

function routeSampleThrough(tree: Tree, sample: Sample, nodeId: string): boolean {
  let id: string | null = ROOT_ID;
  while (id !== null && tree[id]) {
    if (id === nodeId) return true;
    // Annotated: `id` is reassigned from this node's children, so inference
    // would be circular without it.
    const node: TreeNode | undefined = tree[id];
    if (!node || node.split === null) return false;
    const value: number = sample.features[node.split.feature] ?? 0;
    id = value < node.split.threshold ? node.left : node.right;
  }
  return false;
}

/**
 * Recompute everything a tree change affects, in one place.
 *
 * Every mutating action routes through this, so accuracies, depth and the peak
 * cannot fall out of step with the tree that produced them.
 */
function withTree(state: ArchitectState, tree: Tree): Partial<ArchitectState> {
  const stats = nodeStats(tree, state.dataset.train);
  const trainAccuracy = accuracyOf(tree, stats, state.dataset.train);
  const validationAccuracy = accuracyOf(tree, stats, state.dataset.validation);
  const improved = validationAccuracy > state.peakValidation;

  return {
    tree,
    trainAccuracy,
    validationAccuracy,
    depth: depthOf(tree),
    splits: splitCountOf(tree),
    starved: starvedLeaves(tree, stats).length,
    peakValidation: improved ? validationAccuracy : state.peakValidation,
    peakDepth: improved ? depthOf(tree) : state.peakDepth,
    peakTrainAccuracy: improved ? trainAccuracy : state.peakTrainAccuracy,
    // A tree change invalidates the previous sign-off.
    evaluation: null,
    failure: null,
  };
}

function freshRound(roundIndex: number) {
  const round = roundAt(roundIndex);
  const dataset = generateDataset(round, DATA_SEED + round.index * 53);
  const tree = emptyTree();
  const stats = nodeStats(tree, dataset.train);

  return {
    roundIndex: round.index,
    dataset,
    tree,
    selectedNodeId: ROOT_ID,
    trainAccuracy: accuracyOf(tree, stats, dataset.train),
    validationAccuracy: accuracyOf(tree, stats, dataset.validation),
    depth: 0,
    splits: 0,
    starved: 0,
    peakValidation: 0,
    peakDepth: 0,
    peakTrainAccuracy: 0,
    evaluation: null,
    attempts: 0,
    phase: "building" as Phase,
    failure: null,
  };
}

export const useArchitectStore = create<ArchitectState>((set, get) => ({
  ...freshRound(1),
  clearedScores: [],
  whyCard: whyCardFor({ kind: "round-briefing", round: roundAt(1) }),
  lane: "visual" as Lane,

  setLane: (lane) => set({ lane }),

  selectNode: (nodeId) => {
    if (!get().tree[nodeId]) return;
    set({ selectedNodeId: nodeId });
  },

  splitAt: (nodeId, split) => {
    const state = get();
    const node = state.tree[nodeId];
    if (!node || node.split !== null) return;

    const bucket = samplesAt(state, nodeId);
    const tree = applySplit(state.tree, nodeId, split);
    const next = withTree(state, tree);

    set({
      ...next,
      // Follow the player down: the left child is where the next decision is.
      selectedNodeId: `${nodeId}L`,
      whyCard: whyCardFor({
        kind: "gate-built",
        round: roundAt(state.roundIndex),
        nodeId,
        split,
        bucket,
        trainAccuracy: next.trainAccuracy ?? state.trainAccuracy,
        validationAccuracy: next.validationAccuracy ?? state.validationAccuracy,
        previousValidation: state.validationAccuracy,
        peakValidation: state.peakValidation,
        peakDepth: state.peakDepth,
        starved: next.starved ?? 0,
        depth: next.depth ?? 0,
      }),
    });
  },

  takeGreedySplit: (nodeId) => {
    const state = get();
    const bucket = samplesAt(state, nodeId);
    const candidate = bestSplit(bucket);
    if (candidate === null || candidate.gain <= 0) return;
    get().splitAt(nodeId, {
      feature: candidate.feature,
      threshold: candidate.threshold,
    });
  },

  prune: (nodeId) => {
    const state = get();
    const node = state.tree[nodeId];
    if (!node || node.split === null) return;

    const tree = pruneAt(state.tree, nodeId);
    const next = withTree(state, tree);

    set({
      ...next,
      selectedNodeId: nodeId,
      whyCard: whyCardFor({
        kind: "pruned",
        round: roundAt(state.roundIndex),
        validationAccuracy: next.validationAccuracy ?? state.validationAccuracy,
        previousValidation: state.validationAccuracy,
        peakValidation: state.peakValidation,
        depth: next.depth ?? 0,
      }),
    });
  },

  clearTree: () => {
    const state = get();
    const next = withTree(state, emptyTree());
    set({
      ...next,
      selectedNodeId: ROOT_ID,
      whyCard: whyCardFor({
        kind: "round-briefing",
        round: roundAt(state.roundIndex),
      }),
    });
  },

  signOff: () => {
    const state = get();
    const round = roundAt(state.roundIndex);

    const evaluation = evaluateTree({
      round,
      tree: state.tree,
      dataset: state.dataset,
      peakValidation: state.peakValidation,
      peakDepth: state.peakDepth,
      peakTrainAccuracy: state.peakTrainAccuracy,
    });

    const won = evaluation.outcome === "win";
    const lastRound = state.roundIndex >= ROUNDS.length;
    const clearedScores = won
      ? [...state.clearedScores, evaluation.score]
      : state.clearedScores;
    const complete = won && lastRound;

    set({
      evaluation,
      attempts: state.attempts + 1,
      clearedScores,
      phase: complete ? "complete" : won ? "cleared" : "building",
      failure: evaluation.failure,
      whyCard: whyCardFor({
        kind: "signed-off",
        round,
        evaluation,
        complete,
        attempts: state.attempts + 1,
      }),
    });

    if (complete) {
      const mean =
        clearedScores.reduce((total, value) => total + value, 0) /
        Math.max(1, clearedScores.length);
      useProgression.getState().recordResult({
        slug: SLUG,
        score: mean,
        lane: state.lane,
        completed: true,
        codeLaneCleared: state.lane === "code",
      });
    }

    return evaluation;
  },

  nextRound: () => {
    const state = get();
    if (state.phase !== "cleared") return;
    const round = roundAt(state.roundIndex + 1);
    set({
      ...freshRound(round.index),
      whyCard: whyCardFor({ kind: "round-briefing", round }),
    });
  },

  restart: () => {
    set({
      ...freshRound(1),
      clearedScores: [],
      whyCard: whyCardFor({ kind: "round-briefing", round: roundAt(1) }),
    });
  },
}));

export { ROOT_ID, routeSample };
