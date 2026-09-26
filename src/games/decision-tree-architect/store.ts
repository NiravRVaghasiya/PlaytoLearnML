"use client";

import { create } from "zustand";
import { useProgression } from "@/engine/progression";
import type { Lane, NamedFailure } from "@/engine/types";
import type { MetricState, WhyCardContent } from "@/components";
import {
  DEPTH_ARG_LIMIT,
  FEATURES,
  ROOT_ID,
  ROUNDS,
  accuracyOf,
  applySplit,
  bestSplit,
  buildGreedyTree,
  candidateSplitsWithLookahead,
  countsOf,
  depthCurve,
  depthOf,
  describeSplit,
  emptyTree,
  evaluateTree,
  gainOf,
  generateDataset,
  giniOf,
  givesGroundBack,
  leavesOf,
  lookaheadGainOf,
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

/** Anything but an explicit "code" is a visual-lane action (a click passes an event). */
const laneOf = (source: unknown): Lane => (source === "code" ? "code" : "visual");

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
  /**
   * Best score per signed-off plot, indexed by plot (plot 1 at index 0).
   *
   * Keyed rather than appended, so signing a retried plot off again replaces its
   * score instead of counting the plot twice. Plots are only reachable in order,
   * so the array never has holes and its length is the number signed off.
   */
  clearedScores: number[];
  /**
   * Has any plot been signed off by a sign-off that came through the code lane?
   * The third star's criterion, tracked from the action rather than from which
   * tab happened to be open when the last plot was signed off.
   */
  codeLaneWin: boolean;

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
  /**
   * Replace the tree with greedy CART grown to `maxDepth`, in ONE update.
   *
   * The peak is a high-water mark over the trees the player has had. Growing
   * gate by gate through `splitAt` recorded every half-built level in between —
   * on the terrace, the root plus ONE of its two second-level gates scored
   * 93.8%, above the finished depth-3 tree's 92.7% — so growing straight to
   * greedy's own peak then reported the player as overfitting against a tree
   * the script never chose.
   *
   * What it does count is each COMPLETE level greedy grows through on the way
   * down, because those are real trees and each is a prefix of the next:
   * growing to depth 6 does pass through the depth-3 tree, so "you had 92.7% at
   * depth 3" stays true, and it is the same trajectory as taking the greedy
   * gate at every leaf, level by level, in the visual lane.
   */
  growGreedy: (maxDepth: number) => void;
  /** Remove everything below a node. */
  prune: (nodeId: string) => void;
  clearTree: () => void;
  /**
   * Commit the tree for inspection.
   *
   * `source` is the lane the sign-off came from — the button, or `api.signOff`
   * — so XP and the code-lane star follow the action.
   *
   * Only a plot still being built can be signed off. Once it is, signing off
   * again (running the starter snippet twice does it) returns a fresh
   * evaluation of the current tree WITHOUT touching the state: a signed-off
   * plot cannot be counted twice, and cannot be un-signed by a worse tree.
   */
  signOff: (source?: Lane) => Evaluation;
  nextRound: () => void;
  /**
   * Start the current plot over — what the shell's Retry does. Plots already
   * signed off stay signed off; starting the survey over is "Back to plot one".
   */
  retryRound: () => void;
  restart: () => void;
}

// ── selectors: primitives and stable references only ─────────────────────

export function currentRound(state: ArchitectState): Round {
  return roundAt(state.roundIndex);
}

export function clearedCount(state: ArchitectState): number {
  return state.clearedScores.length;
}

/**
 * Is validation below the best this player reached, with training no lower?
 * The same test the WhyCards use (`givesGroundBack`), so the ghost and the card
 * beside it cannot disagree.
 */
export function isGivingGroundBack(state: ArchitectState): boolean {
  return givesGroundBack(state);
}

/**
 * The colour of the live metric, validation accuracy.
 *
 * Its warning is the ghost's own test, and it outranks a signed-off plot: a tree
 * grown past its peak after the sign-off (the code lane can do it in one line)
 * must not read green while the ghost and the WhyCard beside it say
 * overfitting. Undefined leaves the readout free to pulse on a change.
 */
export function validationMetricState(
  state: ArchitectState,
): MetricState | undefined {
  if (state.failure !== null) return "bad";
  if (isGivingGroundBack(state)) return "warn";
  if (state.phase !== "building") return "good";
  return state.validationAccuracy >= currentRound(state).target
    ? "good"
    : undefined;
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
  codeLaneWin: false,
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
        peakTrainAccuracy: state.peakTrainAccuracy,
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

  growGreedy: (maxDepth) => {
    const state = get();
    const depth = Number.isFinite(maxDepth)
      ? Math.max(0, Math.min(DEPTH_ARG_LIMIT, Math.floor(maxDepth)))
      : 0;

    // The complete levels on the way down (see the interface comment). The
    // earliest depth wins a tie, as it does in `withTree`.
    let passedThrough: ArchitectState = state;
    for (const level of depthCurve(state.dataset, depth)) {
      if (level.validationAccuracy > passedThrough.peakValidation) {
        passedThrough = {
          ...passedThrough,
          peakValidation: level.validationAccuracy,
          peakDepth: level.depth,
          peakTrainAccuracy: level.trainAccuracy,
        };
      }
    }

    // buildGreedyTree takes the best gain at every leaf, level by level, with
    // the same impurity code the picker's gates use — the tree the old
    // breadth-first walk through `splitAt` produced, gate for gate.
    const tree = buildGreedyTree(state.dataset.train, depth);
    const next = withTree(passedThrough, tree);

    // Aim the picker at the biggest leaf, the NodePicker's own ordering.
    const stats = nodeStats(tree, state.dataset.train);
    const biggest = leavesOf(tree).reduce<TreeNode | null>(
      (best, leaf) =>
        best === null ||
        (stats[leaf.id]?.counts.total ?? 0) > (stats[best.id]?.counts.total ?? 0)
          ? leaf
          : best,
      null,
    );

    set({
      ...next,
      selectedNodeId: biggest?.id ?? ROOT_ID,
      whyCard: whyCardFor({
        kind: "greedy-grown",
        round: roundAt(state.roundIndex),
        depth: next.depth ?? 0,
        splits: next.splits ?? 0,
        trainAccuracy: next.trainAccuracy ?? state.trainAccuracy,
        validationAccuracy: next.validationAccuracy ?? state.validationAccuracy,
        peakValidation: next.peakValidation ?? state.peakValidation,
        peakDepth: next.peakDepth ?? state.peakDepth,
        peakTrainAccuracy: next.peakTrainAccuracy ?? state.peakTrainAccuracy,
        starved: next.starved ?? 0,
      }),
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

  signOff: (source) => {
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

    // Re-entrancy guard. See the interface comment.
    if (state.phase !== "building") return evaluation;

    const lane = laneOf(source);
    const won = evaluation.outcome === "win";
    const lastRound = state.roundIndex >= ROUNDS.length;
    const clearedScores = [...state.clearedScores];
    if (won) {
      const slot = state.roundIndex - 1;
      clearedScores[slot] = Math.max(clearedScores[slot] ?? 0, evaluation.score);
    }
    const codeLaneWin = state.codeLaneWin || (won && lane === "code");
    const complete = won && lastRound;

    set({
      evaluation,
      attempts: state.attempts + 1,
      clearedScores,
      codeLaneWin,
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

    // Only reachable on the transition INTO "complete": the guard above keeps a
    // finished survey from recording itself again.
    if (complete) {
      const mean =
        clearedScores.reduce((total, value) => total + value, 0) /
        Math.max(1, clearedScores.length);
      useProgression.getState().recordResult({
        slug: SLUG,
        score: mean,
        lane,
        completed: true,
        codeLaneCleared: codeLaneWin,
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

  retryRound: () => {
    const round = roundAt(get().roundIndex);
    set({
      ...freshRound(round.index),
      whyCard: whyCardFor({ kind: "round-briefing", round }),
    });
  },

  restart: () => {
    set({
      ...freshRound(1),
      clearedScores: [],
      codeLaneWin: false,
      whyCard: whyCardFor({ kind: "round-briefing", round: roundAt(1) }),
    });
  },
}));

export { ROOT_ID, routeSample };

// ── The code lane's api ───────────────────────────────────────────────────

/** A node id from a script: must name a node in the current tree. */
function checkedNode(verb: string, nodeId: unknown): string {
  if (typeof nodeId !== "string" || !useArchitectStore.getState().tree[nodeId]) {
    throw new Error(
      `${verb}: no node "${String(nodeId)}" in the tree. Leaves right now: ${leavesOf(
        useArchitectStore.getState().tree,
      )
        .map((leaf) => leaf.id)
        .join(", ")}`,
    );
  }
  return nodeId;
}

/** A node that can take a gate: it exists and is still a leaf. */
function checkedLeaf(verb: string, nodeId: unknown): string {
  const id = checkedNode(verb, nodeId);
  const split = useArchitectStore.getState().tree[id]?.split ?? null;
  if (split !== null) {
    throw new Error(
      `${verb}: ${id} already has a gate (${describeSplit(
        split,
      )}). Prune it first, or build on one of its leaves.`,
    );
  }
  return id;
}

function checkedFeature(verb: string, feature: unknown): number {
  if (
    typeof feature !== "number" ||
    !Number.isInteger(feature) ||
    feature < 0 ||
    feature >= FEATURES.length
  ) {
    throw new Error(
      `${verb}: feature must be an integer 0..${FEATURES.length - 1} (${FEATURES.map(
        (meta, index) => `${index} = ${meta.name}`,
      ).join(", ")}), got ${String(feature)}`,
    );
  }
  return feature;
}

function checkedThreshold(verb: string, threshold: unknown): number {
  if (typeof threshold !== "number" || !Number.isFinite(threshold)) {
    throw new Error(`${verb}: threshold must be a finite number, got ${String(threshold)}`);
  }
  return threshold;
}

/** A depth from a script: a whole number, bounded so no loop can run away. */
function checkedDepth(verb: string, depth: unknown): number {
  if (
    typeof depth !== "number" ||
    !Number.isInteger(depth) ||
    depth < 0 ||
    depth > DEPTH_ARG_LIMIT
  ) {
    throw new Error(
      `${verb}: depth must be a whole number from 0 to ${DEPTH_ARG_LIMIT}, got ${String(depth)}`,
    );
  }
  return depth;
}

/**
 * What the code lane can do, and nothing more.
 *
 * `api.split` and `api.prune` are the same store actions the picker calls, and
 * `api.signOff` is the same judgement the button calls (CLAUDE.md two-lane rule)
 * — tagged as a code-lane sign-off, which is what the third star counts. Every
 * verb checks its arguments and throws a named error, so a typo in a script
 * reads as a message rather than a silently empty result or a hung tab.
 */
export function createCodeApi() {
  const store = useArchitectStore;
  const bucketAt = (verb: string, nodeId: unknown) =>
    samplesAt(store.getState(), checkedNode(verb, nodeId));

  return {
    /** Build a gate at a node. Same action as the picker. */
    split: (nodeId: string, feature: number, threshold: number) => {
      store
        .getState()
        .splitAt(checkedLeaf("split", nodeId), {
          feature: checkedFeature("split", feature),
          threshold: checkedThreshold("split", threshold),
        });
    },
    /** Take whatever greedy CART would take at a node. */
    greedyAt: (nodeId: string) =>
      store.getState().takeGreedySplit(checkedLeaf("greedyAt", nodeId)),
    prune: (nodeId: string) => {
      const id = checkedNode("prune", nodeId);
      if (store.getState().tree[id]?.split === null) {
        throw new Error(`prune: ${id} is a leaf, so there is nothing below it to remove`);
      }
      store.getState().prune(id);
    },
    clear: () => store.getState().clearTree(),

    /** Grow the whole tree greedily to a depth, replacing what is there. */
    growGreedy: (maxDepth: number) =>
      store.getState().growGreedy(checkedDepth("growGreedy", maxDepth)),

    signOff: () => store.getState().signOff("code"),
    nextRound: () => store.getState().nextRound(),
    retryRound: () => store.getState().retryRound(),
    restart: () => store.getState().restart(),

    /** The gain table at a node, lookahead column included. */
    gainTable: (nodeId = ROOT_ID) => {
      const bucket = bucketAt("gainTable", nodeId);
      return candidateSplitsWithLookahead(bucket).map((candidate, index) => ({
        feature: index,
        name: FEATURES[index]!.name,
        threshold: candidate?.threshold ?? null,
        gain: candidate?.gain ?? 0,
        lookaheadGain: candidate?.lookaheadGain ?? 0,
      }));
    },
    /** Gain of any gate you like, without building it. */
    gainOf: (nodeId: string, feature: number, threshold: number) =>
      gainOf(bucketAt("gainOf", nodeId), {
        feature: checkedFeature("gainOf", feature),
        threshold: checkedThreshold("gainOf", threshold),
      }),
    /** Lookahead of any gate you like — the slider readout, as a function. */
    lookaheadOf: (nodeId: string, feature: number, threshold: number) =>
      lookaheadGainOf(bucketAt("lookaheadOf", nodeId), {
        feature: checkedFeature("lookaheadOf", feature),
        threshold: checkedThreshold("lookaheadOf", threshold),
      }),
    impurityAt: (nodeId: string) =>
      giniOf(countsOf(bucketAt("impurityAt", nodeId))),
    countsAt: (nodeId: string) => countsOf(bucketAt("countsAt", nodeId)),

    leaves: () => leavesOf(store.getState().tree).map((leaf) => leaf.id),
    gate: (nodeId: string) => {
      const node = store.getState().tree[checkedNode("gate", nodeId)];
      return node?.split === null || !node
        ? null
        : { ...node.split, describe: describeSplit(node.split) };
    },

    /** Greedy CART's accuracies at each depth, without touching your tree. */
    depthCurve: (maxDepth = 9) =>
      depthCurve(store.getState().dataset, checkedDepth("depthCurve", maxDepth)),
    /** How many starved leaves greedy CART would have at a depth. */
    starvedAtDepth: (maxDepth: number) => {
      const { dataset } = store.getState();
      const tree = buildGreedyTree(
        dataset.train,
        checkedDepth("starvedAtDepth", maxDepth),
      );
      return starvedLeaves(tree, nodeStats(tree, dataset.train)).length;
    },
    /** Score any depth of greedy CART without committing to it. */
    tryGreedy: (maxDepth: number) => {
      const { dataset } = store.getState();
      const tree = buildGreedyTree(dataset.train, checkedDepth("tryGreedy", maxDepth));
      const stats = nodeStats(tree, dataset.train);
      return {
        trainAccuracy: accuracyOf(tree, stats, dataset.train),
        validationAccuracy: accuracyOf(tree, stats, dataset.validation),
      };
    },

    trainAccuracy: () => store.getState().trainAccuracy,
    validationAccuracy: () => store.getState().validationAccuracy,
    depth: () => store.getState().depth,
    splits: () => store.getState().splits,
    starved: () => store.getState().starved,
    peak: () => ({
      validation: store.getState().peakValidation,
      depth: store.getState().peakDepth,
    }),

    round: () => {
      const round = roundAt(store.getState().roundIndex);
      return {
        index: round.index,
        name: round.name,
        boundary: round.boundary,
        target: round.target,
        maxDepth: round.maxDepth,
        trainPoints: round.trainPoints,
      };
    },
    trainPoints: () => store.getState().dataset.train.length,
    achievable: () => store.getState().dataset.achievable,
    features: () => FEATURES.map((meta) => ({ ...meta })),
    phase: () => store.getState().phase,
    cleared: () => clearedCount(store.getState()),
    lastEvaluation: () => store.getState().evaluation,
  };
}

export type ArchitectCodeApi = ReturnType<typeof createCodeApi>;
