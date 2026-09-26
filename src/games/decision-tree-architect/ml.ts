import type { NamedFailure } from "@/engine/types";
import { clamp, gaussian, seededRandom } from "@/lib/utils";

/**
 * Decision Tree Architect — the real tree learner.
 *
 * ── Deviation from the spec, deliberate ─────────────────────────────────────
 * The spec asks for "TF.js decision-forest for reference scoring". No such thing
 * exists in the browser: TensorFlow Decision Forests is Python-only, and
 * `tfjs-tfdf` can only run inference on a model already trained elsewhere. So the
 * reference learner here is a real greedy CART implementation, written out below.
 *
 * That turns out to be the better answer rather than a compromise. The player's
 * hand-built tree and the reference tree call the *same* `giniOf` and
 * `bestSplitOn` functions, so the spec's promise — "the game computes the
 * resulting impurity just like a real tree learner" — is not an approximation.
 * It is the same code path. A tree is also not a tensor computation, so there is
 * no TF.js in this game at all; adding it would be decoration.
 *
 * ── The one subtle thing ────────────────────────────────────────────────────
 * A leaf's prediction is the majority class of the TRAINING samples that land in
 * it. Validation samples are routed through the same tree but never get a vote.
 * Letting them vote would make validation accuracy climb with depth forever and
 * destroy the entire lesson, so `predictWith` takes the training statistics
 * explicitly rather than deriving them from whatever data it is scoring.
 */

// ── Features ──────────────────────────────────────────────────────────────

export interface FeatureMeta {
  name: string;
  /** How it reads in a split condition, e.g. "slope < 0.42". */
  unit: string;
  /** True for the two features that carry no signal at all. */
  irrelevant: boolean;
  description: string;
}

/**
 * Four survey readings, two of which are pure noise.
 *
 * The irrelevant pair is named as plausibly as the useful pair on purpose. There
 * is no way to tell them apart by reading the labels — only by looking at what
 * splitting on them actually buys, which is the habit the gain table is there to
 * build.
 */
export const FEATURES: readonly FeatureMeta[] = [
  {
    name: "slope",
    unit: "grade",
    irrelevant: false,
    description: "Ground gradient across the plot.",
  },
  {
    name: "bedrock",
    unit: "depth",
    irrelevant: false,
    description: "How far down the load-bearing rock sits.",
  },
  {
    name: "soil pH",
    unit: "pH",
    irrelevant: true,
    description: "Acidity of the topsoil.",
  },
  {
    name: "wind",
    unit: "index",
    irrelevant: true,
    description: "Average exposure at roof height.",
  },
] as const;

export const FEATURE_COUNT = FEATURES.length;

export interface Sample {
  features: number[];
  /** 1 = safe to build. */
  label: 0 | 1;
}

export interface Dataset {
  train: Sample[];
  validation: Sample[];
  /** Ceiling set by the label noise. */
  achievable: number;
}

export const VALIDATION_POINTS = 600;

// ── Rounds ────────────────────────────────────────────────────────────────

export type BoundaryKind = "box" | "diagonal" | "quadrants";

export interface Round {
  index: number;
  name: string;
  boundary: BoundaryKind;
  /**
   * Surveyed plots available to build the tree from.
   *
   * Per round, not global, because how much data you have decides how much depth
   * is worth having. Measured: a diagonal boundary at 140 plots peaks at depth 2
   * and gets worse after, so depth never appeared to help and the staircase
   * lesson was unreachable. Each step of a staircase needs samples to stand on.
   */
  trainPoints: number;
  noiseRate: number;
  /** Validation accuracy needed to sign the plot off. */
  target: number;
  /** Depth the player must not exceed. */
  maxDepth: number;
  brief: string;
  /** What this shape teaches about axis-aligned splitting. */
  lesson: string;
}

/**
 * Three plots, chosen to make one point each about axis-aligned splits.
 *
 * The order is not difficulty for its own sake. A box is what trees are *built*
 * for and two splits solve it outright; a diagonal is the case they can only
 * approximate with a staircase, so depth genuinely buys accuracy for a while;
 * quadrants cannot be done at depth 1 at all, no matter which feature you pick,
 * which is the cleanest demonstration that a single split is not a weak version
 * of a tree but a different thing entirely.
 */
export const ROUNDS: readonly Round[] = [
  {
    index: 1,
    name: "Terrace plot",
    boundary: "box",
    trainPoints: 140,
    noiseRate: 0.08,
    target: 0.85,
    maxDepth: 6,
    brief:
      "Safe ground here is a rectangle: gentle slope AND shallow bedrock. Two gates should be enough — find them and stop.",
    lesson:
      "Axis-aligned rectangles are exactly what a split on one feature at a time can describe, so this is the shape trees are best at.",
  },
  {
    index: 2,
    name: "Hillside plot",
    boundary: "diagonal",
    trainPoints: 400,
    noiseRate: 0.1,
    target: 0.82,
    maxDepth: 8,
    brief:
      "The safe region runs diagonally: slope and bedrock trade off against each other. No single gate can follow a diagonal, so you will be building a staircase.",
    lesson:
      "A diagonal boundary needs many axis-aligned steps to approximate. Depth genuinely helps here — right up until the steps start tracing individual points.",
  },
  {
    index: 3,
    name: "Split ridge",
    boundary: "quadrants",
    trainPoints: 200,
    noiseRate: 0.08,
    target: 0.8,
    maxDepth: 8,
    brief:
      "Safe ground sits in two opposite corners. One gate cannot separate this at all — whichever feature you pick, both sides come out equally mixed.",
    lesson:
      "Zero information gain from the best first split does not mean the features are useless. It means the structure only appears after a second split.",
  },
] as const;

export function roundAt(index: number): Round {
  return ROUNDS[clamp(index - 1, 0, ROUNDS.length - 1)] ?? ROUNDS[0]!;
}

function cleanLabel(boundary: BoundaryKind, slope: number, bedrock: number): 0 | 1 {
  switch (boundary) {
    case "box":
      return slope < 0.45 && bedrock > 0.4 ? 1 : 0;
    case "diagonal":
      return slope + (1 - bedrock) < 0.95 ? 1 : 0;
    case "quadrants":
      return (slope < 0.5) !== (bedrock < 0.5) ? 1 : 0;
  }
}

function sampleOf(random: () => number, round: Round): Sample {
  const slope = random();
  const bedrock = random();
  const clean = cleanLabel(round.boundary, slope, bedrock);

  return {
    features: [
      slope,
      bedrock,
      // Irrelevant readings, drawn without reference to the label.
      clamp(0.5 + gaussian(random) * 0.18, 0, 1),
      clamp(0.5 + gaussian(random) * 0.18, 0, 1),
    ],
    label: random() < round.noiseRate ? ((1 - clean) as 0 | 1) : clean,
  };
}

export function generateDataset(round: Round, seed: number): Dataset {
  const trainRandom = seededRandom(seed);
  const validationRandom = seededRandom(seed + 7717);

  const train: Sample[] = [];
  for (let index = 0; index < round.trainPoints; index += 1) {
    train.push(sampleOf(trainRandom, round));
  }

  const validation: Sample[] = [];
  for (let index = 0; index < VALIDATION_POINTS; index += 1) {
    validation.push(sampleOf(validationRandom, round));
  }

  return { train, validation, achievable: 1 - round.noiseRate };
}

// ── Impurity ──────────────────────────────────────────────────────────────

export interface ClassCounts {
  negative: number;
  positive: number;
  total: number;
}

export function countsOf(samples: Sample[]): ClassCounts {
  let positive = 0;
  for (const sample of samples) if (sample.label === 1) positive += 1;
  return {
    negative: samples.length - positive,
    positive,
    total: samples.length,
  };
}

/** Gini impurity: 1 − Σp². Zero for a pure node, 0.5 for an even split. */
export function giniOf(counts: ClassCounts): number {
  return giniFrom(counts.positive, counts.total);
}

function giniFrom(positive: number, total: number): number {
  if (total === 0) return 0;
  const p = positive / total;
  return 1 - (p * p + (1 - p) * (1 - p));
}

/** Shannon entropy in bits. Offered alongside Gini per the spec. */
export function entropyOf(counts: ClassCounts): number {
  if (counts.total === 0) return 0;
  const p = counts.positive / counts.total;
  if (p === 0 || p === 1) return 0;
  return -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
}

export interface Split {
  feature: number;
  threshold: number;
}

export interface SplitCandidate extends Split {
  /** Weighted Gini decrease. This is the information gain the learner maximises. */
  gain: number;
  leftCount: number;
  rightCount: number;
  leftGini: number;
  rightGini: number;
}

export function partition(
  samples: Sample[],
  split: Split,
): { left: Sample[]; right: Sample[] } {
  const left: Sample[] = [];
  const right: Sample[] = [];
  for (const sample of samples) {
    // "< threshold" goes left, matching the label shown on the gate.
    if ((sample.features[split.feature] ?? 0) < split.threshold) left.push(sample);
    else right.push(sample);
  }
  return { left, right };
}

export function gainOf(samples: Sample[], split: Split): number {
  if (samples.length === 0) return 0;
  const parent = giniOf(countsOf(samples));
  const { left, right } = partition(samples, split);
  if (left.length === 0 || right.length === 0) return 0;
  const weighted =
    (left.length / samples.length) * giniOf(countsOf(left)) +
    (right.length / samples.length) * giniOf(countsOf(right));
  return parent - weighted;
}

export function describeSplit(split: Split): string {
  return `${FEATURES[split.feature]?.name ?? "?"} < ${split.threshold.toFixed(2)}`;
}

/**
 * The best threshold on one feature, by exhaustive search over midpoints.
 *
 * Midpoints between consecutive distinct values, which is what CART does: any
 * threshold between two observed values partitions identically, so only the gaps
 * are worth testing. Ties keep the earlier threshold so the result is
 * deterministic.
 */
export function bestSplitOn(
  samples: Sample[],
  feature: number,
): SplitCandidate | null {
  const total = samples.length;
  if (total < 2) return null;

  // Sort once and sweep, accumulating class counts. O(n log n) rather than the
  // O(n²) of re-partitioning at every candidate threshold — which matters because
  // the lookahead column below needs this run several times per node.
  const sorted = samples
    .map((sample) => ({
      value: sample.features[feature] ?? 0,
      label: sample.label as number,
    }))
    .sort((a, b) => a.value - b.value);

  let totalPositive = 0;
  for (const entry of sorted) totalPositive += entry.label;
  const parentGini = giniFrom(totalPositive, total);

  let best: SplitCandidate | null = null;
  let leftPositive = 0;

  for (let index = 0; index < total - 1; index += 1) {
    leftPositive += sorted[index]!.label;
    // Identical values cannot be separated, so there is no threshold here.
    if (sorted[index]!.value === sorted[index + 1]!.value) continue;

    const leftCount = index + 1;
    const rightCount = total - leftCount;
    const rightPositive = totalPositive - leftPositive;

    const leftGini = giniFrom(leftPositive, leftCount);
    const rightGini = giniFrom(rightPositive, rightCount);
    const gain =
      parentGini -
      ((leftCount / total) * leftGini + (rightCount / total) * rightGini);

    if (best === null || gain > best.gain) {
      best = {
        feature,
        threshold: (sorted[index]!.value + sorted[index + 1]!.value) / 2,
        gain,
        leftCount,
        rightCount,
        leftGini,
        rightGini,
      };
    }
  }
  return best;
}

/**
 * How much impurity the NEXT level could remove, if this split were taken.
 *
 * This is one-step lookahead, and real CART does not do it — which is precisely
 * why it is here. On an XOR-shaped boundary the best first split has almost no
 * immediate gain, so a greedy learner picks a worthless feature and never
 * recovers. Measured on the ridge plot: slope < 0.50 gains 0.0011 and has a
 * lookahead of 0.393, while the useless wind reading's best gate gains 0.0177
 * and has a lookahead of 0.019.
 *
 * Two things this number is NOT, both measured, both easy to get wrong:
 *
 *   - It is not shown for slope < 0.50 in the gain table. The table tries each
 *     reading at its best IMMEDIATE threshold, which on XOR data is a sliver at
 *     the edge (slope < 0.04, lookahead 0.017). The right gate is found with the
 *     threshold slider, whose readout shows this number live.
 *   - A high value alone is not "hidden structure". A gate that peels off a
 *     sliver leaves the big side almost exactly as it was, so its lookahead
 *     is roughly the best gain the node ALREADY had (terrace root: soil pH <
 *     0.97 gains 0.0125, lookahead 0.150, against 0.153 available anyway). Use
 *     `unlockedGainOf` to ask whether a gate created structure.
 */
export function lookaheadGainOf(samples: Sample[], split: Split): number {
  const { left, right } = partition(samples, split);
  if (left.length === 0 || right.length === 0) return 0;

  const leftBest = bestSplit(left)?.gain ?? 0;
  const rightBest = bestSplit(right)?.gain ?? 0;

  return (
    (left.length / samples.length) * Math.max(0, leftBest) +
    (right.length / samples.length) * Math.max(0, rightBest)
  );
}

/**
 * Lookahead beyond what the node could already buy in one gate.
 *
 * Lookahead minus the best immediate gain at the node. Positive means the gate
 * uncovered structure a one-step score could not see — the ridge root's
 * slope < 0.50 unlocks 0.375. Near zero or negative means it only passed the
 * node's existing problem down a level, however large its lookahead looks.
 * This is the number the "almost no gain, a lot underneath" reading needs.
 */
export function unlockedGainOf(
  samples: Sample[],
  split: Split,
  parentBest: number = bestSplit(samples)?.gain ?? 0,
): number {
  return lookaheadGainOf(samples, split) - Math.max(0, parentBest);
}

export interface SplitCandidateWithLookahead extends SplitCandidate {
  lookaheadGain: number;
}

/** The gain table with the lookahead column. Used by the split picker. */
export function candidateSplitsWithLookahead(
  samples: Sample[],
): (SplitCandidateWithLookahead | null)[] {
  return candidateSplits(samples).map((candidate) =>
    candidate === null
      ? null
      : {
          ...candidate,
          lookaheadGain: lookaheadGainOf(samples, candidate),
        },
  );
}

/** The best split on every feature — the table the player chooses from. */
export function candidateSplits(samples: Sample[]): (SplitCandidate | null)[] {
  return FEATURES.map((_, feature) => bestSplitOn(samples, feature));
}

/** The single best split across all features. What greedy CART would take. */
export function bestSplit(samples: Sample[]): SplitCandidate | null {
  return candidateSplits(samples).reduce<SplitCandidate | null>(
    (best, candidate) =>
      candidate !== null && (best === null || candidate.gain > best.gain)
        ? candidate
        : best,
    null,
  );
}

// ── The tree ──────────────────────────────────────────────────────────────

export interface TreeNode {
  id: string;
  split: Split | null;
  left: string | null;
  right: string | null;
  /** Depth from the root, root being 0. */
  depth: number;
}

/**
 * The tree, as a flat map from id to node.
 *
 * The spec's `TreeNode` carries `gini` and `samples[]`. Neither is stored here:
 * both are exact functions of the tree and the dataset, and a stored copy is a
 * second version of a number that must never disagree with the split above it.
 * `nodeStats` derives them for every node in one pass.
 */
export type Tree = Record<string, TreeNode>;

export const ROOT_ID = "n0";

/**
 * The deepest a script may ask for in one call (`growGreedy`, `depthCurve`, …).
 *
 * Every plot's greedy tree is pure throughout by depth 12 (terrace 6, hillside
 * 12, ridge 10, measured), so past this a depth argument can only be a typo —
 * or a loop bound like 1e9 that would hang the tab building trees.
 */
export const DEPTH_ARG_LIMIT = 16;

export function emptyTree(): Tree {
  return { [ROOT_ID]: { id: ROOT_ID, split: null, left: null, right: null, depth: 0 } };
}

export function leavesOf(tree: Tree): TreeNode[] {
  return Object.values(tree).filter((node) => node.split === null);
}

export function internalNodesOf(tree: Tree): TreeNode[] {
  return Object.values(tree).filter((node) => node.split !== null);
}

export function depthOf(tree: Tree): number {
  return Object.values(tree).reduce(
    (deepest, node) => Math.max(deepest, node.depth),
    0,
  );
}

export function splitCountOf(tree: Tree): number {
  return internalNodesOf(tree).length;
}

/** Add a split at a leaf, returning a new tree. */
export function applySplit(tree: Tree, nodeId: string, split: Split): Tree {
  const node = tree[nodeId];
  if (!node || node.split !== null) return tree;

  const leftId = `${nodeId}L`;
  const rightId = `${nodeId}R`;

  return {
    ...tree,
    [nodeId]: { ...node, split, left: leftId, right: rightId },
    [leftId]: {
      id: leftId,
      split: null,
      left: null,
      right: null,
      depth: node.depth + 1,
    },
    [rightId]: {
      id: rightId,
      split: null,
      left: null,
      right: null,
      depth: node.depth + 1,
    },
  };
}

/** Remove everything below a node, turning it back into a leaf. */
export function pruneAt(tree: Tree, nodeId: string): Tree {
  const node = tree[nodeId];
  if (!node || node.split === null) return tree;

  const doomed = new Set<string>();
  const walk = (id: string | null) => {
    if (id === null) return;
    const current = tree[id];
    if (!current) return;
    doomed.add(id);
    walk(current.left);
    walk(current.right);
  };
  walk(node.left);
  walk(node.right);

  const next: Tree = {};
  for (const [id, entry] of Object.entries(tree)) {
    if (!doomed.has(id)) next[id] = entry;
  }
  next[nodeId] = { ...node, split: null, left: null, right: null };
  return next;
}

/** Which leaf a sample lands in. */
export function routeSample(tree: Tree, sample: Sample): string {
  let id = ROOT_ID;
  for (;;) {
    const node = tree[id];
    if (!node || node.split === null) return id;
    const value = sample.features[node.split.feature] ?? 0;
    const next = value < node.split.threshold ? node.left : node.right;
    if (next === null || !tree[next]) return id;
    id = next;
  }
}

export interface NodeStat {
  counts: ClassCounts;
  gini: number;
  entropy: number;
  /** Majority class of the TRAINING samples here. */
  majority: 0 | 1;
  /** Gain the split at this node achieved, or null for a leaf. */
  gain: number | null;
}

/** Per-node statistics over a given dataset. */
export function nodeStats(
  tree: Tree,
  samples: Sample[],
): Record<string, NodeStat> {
  const buckets: Record<string, Sample[]> = {};
  for (const id of Object.keys(tree)) buckets[id] = [];

  // Walk each sample down the tree, recording it at every node it passes.
  for (const sample of samples) {
    let id: string | null = ROOT_ID;
    while (id !== null && tree[id]) {
      buckets[id]!.push(sample);
      // Annotated: `id` is reassigned from this node's children, so inference
      // would be circular without it.
      const node: TreeNode | undefined = tree[id];
      if (!node || node.split === null) break;
      const value: number = sample.features[node.split.feature] ?? 0;
      id = value < node.split.threshold ? node.left : node.right;
    }
  }

  const stats: Record<string, NodeStat> = {};
  for (const [id, node] of Object.entries(tree)) {
    const bucket = buckets[id] ?? [];
    const counts = countsOf(bucket);
    stats[id] = {
      counts,
      gini: giniOf(counts),
      entropy: entropyOf(counts),
      majority: counts.positive >= counts.negative ? 1 : 0,
      gain: node.split === null ? null : gainOf(bucket, node.split),
    };
  }
  return stats;
}

/**
 * Predict with a tree, using training statistics for the leaf votes.
 *
 * `trainStats` is required rather than optional. Scoring validation data against
 * leaves labelled by that same validation data would make accuracy rise with
 * depth without limit — every leaf would be right about itself — and the overfit
 * lesson would be unreachable.
 */
export function predictWith(
  tree: Tree,
  trainStats: Record<string, NodeStat>,
  sample: Sample,
): 0 | 1 {
  const leafId = routeSample(tree, sample);
  const stat = trainStats[leafId];
  // An empty leaf has no training evidence; fall back to the root's majority.
  if (!stat || stat.counts.total === 0) {
    return trainStats[ROOT_ID]?.majority ?? 0;
  }
  return stat.majority;
}

export function accuracyOf(
  tree: Tree,
  trainStats: Record<string, NodeStat>,
  samples: Sample[],
): number {
  if (samples.length === 0) return 0;
  let correct = 0;
  for (const sample of samples) {
    if (predictWith(tree, trainStats, sample) === sample.label) correct += 1;
  }
  return correct / samples.length;
}

/**
 * A leaf holding fewer training samples than this is deciding on almost no
 * evidence. Not a rule the game enforces — a diagnosis it reports, because this
 * is the mechanism behind overfit depth made local and countable.
 */
export const MIN_HONEST_LEAF = 5;

export function starvedLeaves(
  tree: Tree,
  trainStats: Record<string, NodeStat>,
): TreeNode[] {
  return leavesOf(tree).filter((leaf) => {
    const total = trainStats[leaf.id]?.counts.total ?? 0;
    return total > 0 && total < MIN_HONEST_LEAF;
  });
}

// ── The reference learner ─────────────────────────────────────────────────

/**
 * Greedy CART, depth-limited. The same impurity code the player's gates use.
 *
 * This is what "the computer would have done" — used to tell the player how their
 * tree compares, and to find the depth at which validation accuracy peaks.
 */
export function buildGreedyTree(samples: Sample[], maxDepth: number): Tree {
  let tree = emptyTree();

  const expand = (nodeId: string, bucket: Sample[], depth: number) => {
    if (depth >= maxDepth || bucket.length < 2) return;
    const counts = countsOf(bucket);
    if (counts.positive === 0 || counts.negative === 0) return;

    const split = bestSplit(bucket);
    if (split === null || split.gain <= 0) return;

    tree = applySplit(tree, nodeId, { feature: split.feature, threshold: split.threshold });
    const { left, right } = partition(bucket, split);
    expand(`${nodeId}L`, left, depth + 1);
    expand(`${nodeId}R`, right, depth + 1);
  };

  expand(ROOT_ID, samples, 0);
  return tree;
}

export interface DepthCurvePoint {
  depth: number;
  trainAccuracy: number;
  validationAccuracy: number;
  splits: number;
}

/** Greedy CART's train and validation accuracy at each depth. */
export function depthCurve(dataset: Dataset, maxDepth = 10): DepthCurvePoint[] {
  const curve: DepthCurvePoint[] = [];
  for (let depth = 0; depth <= maxDepth; depth += 1) {
    const tree = buildGreedyTree(dataset.train, depth);
    const stats = nodeStats(tree, dataset.train);
    curve.push({
      depth,
      trainAccuracy: accuracyOf(tree, stats, dataset.train),
      validationAccuracy: accuracyOf(tree, stats, dataset.validation),
      splits: splitCountOf(tree),
    });
  }
  return curve;
}

/** The depth at which greedy CART's validation accuracy peaks. */
export function bestReferenceDepth(dataset: Dataset, maxDepth = 10): DepthCurvePoint {
  return depthCurve(dataset, maxDepth).reduce((best, point) =>
    point.validationAccuracy > best.validationAccuracy ? point : best,
  );
}

// ── Judging the plot ──────────────────────────────────────────────────────

export type Outcome = "win" | "overfit-depth" | "underfit-stump" | "missed";

export interface Evaluation {
  outcome: Outcome;
  trainAccuracy: number;
  validationAccuracy: number;
  /** Best validation accuracy seen on this plot, at whatever depth. */
  peakValidation: number;
  peakDepth: number;
  depth: number;
  splits: number;
  starved: number;
  score: number;
  /**
   * The ghost's test (`givesGroundBack`) on this tree. The copy of every other
   * verdict reads it, so a sign-off never says "nothing is being memorised"
   * beside a ghost that says overfitting.
   */
  givingGroundBack: boolean;
  failure: NamedFailure | null;
}

const percent = (value: number) => `${Math.round(value * 100)}%`;
const points = (value: number) => `${(value * 100).toFixed(1)}%`;

/** How far validation must fall below its own peak to count as overfitting. */
export const OVERFIT_DROP = 0.025;

/** How far below the peak the ghost and the WhyCards start warning. */
export const GIVING_BACK_DROP = 0.01;

/** A train-validation gap under this is narrow: the tree is generalising. */
export const NARROW_GAP = 0.06;

/**
 * Is the tree giving back validation accuracy the player already had?
 *
 * The one test behind the ghost, the live metric's warning and the WhyCards
 * that link to overfitting, so no two of them can disagree about a tree.
 * Validation below the peak is not enough on its own: a tree SMALLER than the
 * peak scores lower too. Grow the terrace to depth 3 and then back to depth 1,
 * and validation is 15 points down with training 17 points down beside it —
 * too little tree, which sign-off rightly calls an Underfit stump. Overfitting
 * is validation down while training is at least where it was at the peak.
 *
 * No depth test, deliberately: the peak can be a half-built level at the same
 * depth (see `growGreedy` in the store), and finishing that level lowers
 * validation at equal depth — overfitting all the same.
 */
export function givesGroundBack({
  peakValidation,
  validationAccuracy,
  trainAccuracy,
  peakTrainAccuracy,
}: {
  peakValidation: number;
  validationAccuracy: number;
  trainAccuracy: number;
  peakTrainAccuracy: number;
}): boolean {
  return (
    peakValidation > 0 &&
    peakValidation - validationAccuracy >= GIVING_BACK_DROP &&
    trainAccuracy >= peakTrainAccuracy
  );
}

/**
 * Score the tree the player signed off.
 *
 * Two named failures with opposite fixes:
 *
 *   "Overfit depth"   — validation fell below the best this player already
 *                       reached, while training accuracy went up. Fix: prune.
 *   "Underfit stump"  — the tree cannot fit even its own training data. Fix:
 *                       split more.
 *
 * The peak is the player's OWN best on this plot rather than the reference
 * learner's, because the claim being made is "you had this and gave it away",
 * which is a much harder thing to argue with than a comparison to an optimum
 * they never saw.
 */
export function evaluateTree({
  round,
  tree,
  dataset,
  peakValidation,
  peakDepth,
  peakTrainAccuracy,
}: {
  round: Round;
  tree: Tree;
  dataset: Dataset;
  peakValidation: number;
  peakDepth: number;
  peakTrainAccuracy: number;
}): Evaluation {
  const stats = nodeStats(tree, dataset.train);
  const trainAccuracy = accuracyOf(tree, stats, dataset.train);
  const validationAccuracy = accuracyOf(tree, stats, dataset.validation);
  const depth = depthOf(tree);
  const splits = splitCountOf(tree);
  const starved = starvedLeaves(tree, stats).length;

  // Reward accuracy at low depth, per the spec. A tree that ties on validation
  // with fewer gates is the better piece of engineering.
  const depthPenalty = round.maxDepth === 0 ? 0 : (depth / round.maxDepth) * 0.2;
  const score = clamp(validationAccuracy * (1 - depthPenalty), 0, 1);
  const gap = Math.max(0, trainAccuracy - validationAccuracy);
  const givingGroundBack = givesGroundBack({
    peakValidation,
    validationAccuracy,
    trainAccuracy,
    peakTrainAccuracy,
  });
  // Said in the two verdicts below whenever the ghost is up — a drop smaller
  // than OVERFIT_DROP, or one the target still clears — so they explain it
  // rather than contradict it.
  const sincePeak = `validation is ${points(
    validationAccuracy,
  )}, below the ${points(
    peakValidation,
  )} you had at depth ${peakDepth} while training has not fallen, so the gates since then are fitting these ${
    dataset.train.length
  } plots rather than the ground`;

  const base = {
    trainAccuracy,
    validationAccuracy,
    peakValidation,
    peakDepth,
    depth,
    splits,
    starved,
    score,
    givingGroundBack,
  };

  // 1. Had it, gave it away.
  //
  // Checked BEFORE the win, deliberately. The spec's lose condition is
  // "validation accuracy drops while training climbs", with no exemption for
  // clearing the bar — and a tree that meets the target having thrown away four
  // points of validation is exactly the mistake this game exists to make visible.
  // Letting it pass because it scraped over the line would make the lesson
  // optional. Nothing is lost by saying so: the failure strip offers a retry, and
  // pruning back to the peak wins immediately.
  //
  // It cannot fire on a first attempt, because there is no peak to fall from yet.
  if (
    peakValidation - validationAccuracy >= OVERFIT_DROP &&
    trainAccuracy > peakTrainAccuracy
  ) {
    return {
      ...base,
      outcome: "overfit-depth",
      failure: {
        name: "Overfit depth",
        detail: `Training accuracy is up to ${points(
          trainAccuracy,
        )} and validation has fallen to ${points(
          validationAccuracy,
        )} — you already had ${points(peakValidation)} at depth ${peakDepth}. ${
          starved > 0
            ? `${starved} leaf${starved === 1 ? "" : "s"} now hold${
                starved === 1 ? "s" : ""
              } fewer than ${MIN_HONEST_LEAF} training samples, so ${
                starved === 1 ? "it is" : "they are"
              } deciding on almost no evidence.`
            : `The extra gates are separating the ${percent(
                round.noiseRate,
              )} of labels that are simply wrong.`
        } Every split past the peak fitted this particular sample of ${
          dataset.train.length
        } plots rather than the ground underneath it. Prune back to depth ${peakDepth}.`,
      },
    };
  }

  if (validationAccuracy >= round.target && depth <= round.maxDepth) {
    return { ...base, outcome: "win", failure: null };
  }

  // 2. Cannot fit even its own training data.
  if (trainAccuracy < dataset.achievable - 0.12) {
    return {
      ...base,
      outcome: "underfit-stump",
      failure: {
        name: "Underfit stump",
        detail: `${points(
          trainAccuracy,
        )} on the plots you surveyed, when ${percent(
          dataset.achievable,
        )} is available — the tree cannot describe this ground even where it has the answers. ${
          splits === 0
            ? "There are no gates at all, so every plot gets the same verdict."
            : `${splits} gate${splits === 1 ? "" : "s"} at depth ${depth} is not enough shape for this boundary.`
        } ${
          // "Nothing is being memorised" only when the numbers say so: the
          // ridge's greedy gates fit noise long before they fit the ground.
          givingGroundBack
            ? `And ${sincePeak}. Add gates that follow the boundary, not more like those: look at the lookahead column as well as the gain.`
            : gap < NARROW_GAP
              ? `Validation is ${points(
                  validationAccuracy,
                )} and the gap to training is only ${points(
                  gap,
                )}, so nothing is being memorised. Add gates.`
              : `Validation is ${points(validationAccuracy)}, ${points(
                  gap,
                )} below training, so some of these gates describe the plots rather than the ground — but mostly the tree has too little of the right shape. Add gates that follow the boundary: look at the lookahead column as well as the gain.`
        }`,
      },
    };
  }

  // 3. In the right region, short of the bar.
  return {
    ...base,
    outcome: "missed",
    failure: {
      name: "Target not met",
      detail: `Validation accuracy ${points(
        validationAccuracy,
      )} against the ${percent(round.target)} this plot needs${
        depth > round.maxDepth
          ? `, and depth ${depth} is over the ${round.maxDepth} allowed`
          : ""
      }. Training is ${points(trainAccuracy)}, ${
        givingGroundBack
          ? `and ${sincePeak}. Pruning back toward depth ${peakDepth} may raise validation even though it lowers training.`
          : `so ${
              trainAccuracy - validationAccuracy > 0.1
                ? `the ${points(
                    trainAccuracy - validationAccuracy,
                  )} gap says some of what you have built is specific to these plots.`
                : "the tree is generalising what it has learned — there is just not enough of it yet."
            }`
      } Check the gain table on your largest leaf.`,
    },
  };
}

// ── Reveal the math ───────────────────────────────────────────────────────

export const MATH_EQUATION = String.raw`G(S) = 1 - \sum_{k} p_k^2
\qquad
H(S) = -\sum_k p_k \log_2 p_k
\\[1.2em]
\text{gain}(S, f, t) = G(S) \;-\; \frac{|S_{<t}|}{|S|}G(S_{<t}) \;-\; \frac{|S_{\ge t}|}{|S|}G(S_{\ge t})`;

export const MATH_CODE = `// The gain your gate achieves. This is the whole learner.
export function gainOf(samples, split) {
  const parent = giniOf(countsOf(samples));
  const { left, right } = partition(samples, split);
  if (left.length === 0 || right.length === 0) return 0;

  // Weighted by how many samples went each way: a split that peels off
  // two pure samples out of a hundred has barely improved anything.
  const weighted =
    (left.length  / samples.length) * giniOf(countsOf(left)) +
    (right.length / samples.length) * giniOf(countsOf(right));

  return parent - weighted;
}

// Greedy CART is this, applied recursively, always taking the best gain.
// Choosing a gate by hand is the same decision, made by you instead.
export function bestSplitOn(samples, feature) {
  // Numbers need a comparator: sort() without one compares them as strings.
  const values = [...new Set(samples.map(s => s.features[feature]))]
    .sort((a, b) => a - b);
  let best = null;
  // Only midpoints between observed values matter: any threshold between
  // two neighbours partitions the data identically.
  for (let i = 0; i < values.length - 1; i++) {
    const threshold = (values[i] + values[i + 1]) / 2;
    const gain = gainOf(samples, { feature, threshold });
    if (!best || gain > best.gain) best = { feature, threshold, gain };
  }
  return best;
}`;

export const MATH_NOTES = `Gini impurity is the chance you would mislabel a sample if you guessed according to the class proportions in the node. Zero means pure, 0.5 means an even mix. Entropy measures the same thing in bits and picks nearly the same splits; the two rarely disagree about which gate to build.

The weighting is the part worth staring at. A split that isolates three pure samples out of a hundred looks fantastic on one side and changes the weighted average by almost nothing — which is exactly right, because it has told you about three plots and nothing about the ninety-seven. Unweighted "purity" would rank that split above a gate that genuinely halves the problem.

Depth is where the bias-variance trade becomes tangible. Every gate you add can only raise training accuracy, because each split is chosen to reduce impurity on the training samples in front of it. Validation accuracy has no such guarantee, and once leaves get small enough to enclose individual mislabelled plots, it starts falling while training keeps climbing. The two curves separating is not a bug in the tree; it is the definition of overfitting, and a tree makes it visible one gate at a time.

Trees only cut along the axes. A rectangle takes two gates. A diagonal takes a staircase and never quite arrives. That is not a flaw to be fixed by more depth — it is the shape of the hypothesis space you chose.`;
