import { createElement } from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import {
  EMPTY_PROGRESSION,
  createMemoryAdapter,
  useProgression,
} from "@/engine/progression";
import { createJsExecutor } from "@/engine/useCodeLane";
import {
  DEPTH_ARG_LIMIT,
  FEATURES,
  GIVING_BACK_DROP,
  MATH_CODE,
  MIN_HONEST_LEAF,
  OVERFIT_DROP,
  ROOT_ID,
  ROUNDS,
  VALIDATION_POINTS,
  accuracyOf,
  applySplit,
  bestSplit,
  bestSplitOn,
  buildGreedyTree,
  candidateSplits,
  candidateSplitsWithLookahead,
  countsOf,
  depthCurve,
  depthOf,
  describeSplit,
  emptyTree,
  entropyOf,
  evaluateTree,
  gainOf,
  generateDataset,
  giniOf,
  givesGroundBack,
  leavesOf,
  lookaheadGainOf,
  nodeStats,
  partition,
  predictWith,
  pruneAt,
  roundAt,
  routeSample,
  splitCountOf,
  starvedLeaves,
  unlockedGainOf,
  type Dataset,
  type Round,
  type Sample,
  type Tree,
} from "./ml";
import {
  SLUG,
  createCodeApi,
  isGivingGroundBack,
  samplesAt,
  useArchitectStore,
  validationMetricState,
} from "./store";
import { whyCardFor } from "./why-cards";
import { STARTER_CODE } from "./CodeLane";
import { SplitGate } from "./SplitGate";
import { TreeOutline } from "./TreeCanvas";

const DATA_SEED = 4417;

const datasetFor = (round: Round): Dataset =>
  generateDataset(round, DATA_SEED + round.index * 53);

/** Score a tree the way the game does. */
function score(tree: Tree, dataset: Dataset) {
  const stats = nodeStats(tree, dataset.train);
  return {
    train: accuracyOf(tree, stats, dataset.train),
    validation: accuracyOf(tree, stats, dataset.validation),
    starved: starvedLeaves(tree, stats).length,
  };
}

/** The tree a human builds for XOR: both features at the middle. */
function handBuiltXor(): Tree {
  let tree = emptyTree();
  tree = applySplit(tree, ROOT_ID, { feature: 0, threshold: 0.5 });
  tree = applySplit(tree, `${ROOT_ID}L`, { feature: 1, threshold: 0.5 });
  tree = applySplit(tree, `${ROOT_ID}R`, { feature: 1, threshold: 0.5 });
  return tree;
}

/** Keep taking the best available gain below an existing tree. */
function greedyExtend(tree: Tree, dataset: Dataset, targetDepth: number): Tree {
  let current = tree;
  for (let pass = 0; pass < targetDepth; pass += 1) {
    const stats = nodeStats(current, dataset.train);
    const buckets = bucketsOf(current, dataset.train);
    for (const leaf of leavesOf(current)) {
      if (leaf.depth >= targetDepth) continue;
      const bucket = buckets[leaf.id] ?? [];
      const counts = stats[leaf.id]!.counts;
      if (bucket.length < 2 || counts.positive === 0 || counts.negative === 0) {
        continue;
      }
      const split = bestSplit(bucket);
      if (split === null || split.gain <= 0) continue;
      current = applySplit(current, leaf.id, {
        feature: split.feature,
        threshold: split.threshold,
      });
    }
  }
  return current;
}

/** Training samples reaching each node. */
function bucketsOf(tree: Tree, samples: Sample[]): Record<string, Sample[]> {
  const buckets: Record<string, Sample[]> = {};
  for (const id of Object.keys(tree)) buckets[id] = [];
  for (const sample of samples) {
    buckets[routeSample(tree, sample)]!.push(sample);
  }
  return buckets;
}

/** The box: slope at the boundary, then bedrock on the gentle side. */
function handBuiltBox(): Tree {
  let tree = emptyTree();
  tree = applySplit(tree, ROOT_ID, { feature: 0, threshold: 0.45 });
  tree = applySplit(tree, `${ROOT_ID}L`, { feature: 1, threshold: 0.4 });
  return tree;
}

describe("impurity", () => {
  it("reads zero for a pure node and a half for an even one", () => {
    expect(giniOf({ negative: 10, positive: 0, total: 10 })).toBe(0);
    expect(giniOf({ negative: 0, positive: 10, total: 10 })).toBe(0);
    expect(giniOf({ negative: 5, positive: 5, total: 10 })).toBeCloseTo(0.5, 6);
    expect(giniOf({ negative: 0, positive: 0, total: 0 })).toBe(0);
  });

  it("reads entropy in bits, one bit at an even split", () => {
    expect(entropyOf({ negative: 5, positive: 5, total: 10 })).toBeCloseTo(1, 6);
    expect(entropyOf({ negative: 10, positive: 0, total: 10 })).toBe(0);
  });

  it("weights gain by how many samples went each way", () => {
    // The load-bearing property. A split that peels off three pure samples from
    // a hundred looks perfect on one side and has told you nearly nothing.
    const samples: Sample[] = [];
    for (let index = 0; index < 97; index += 1) {
      samples.push({ features: [0.5, 0, 0, 0], label: index % 2 === 0 ? 1 : 0 });
    }
    for (let index = 0; index < 3; index += 1) {
      samples.push({ features: [0.9, 0, 0, 0], label: 1 });
    }

    const slice = gainOf(samples, { feature: 0, threshold: 0.7 });
    expect(slice).toBeGreaterThan(0);
    expect(slice).toBeLessThan(0.02);
  });

  it("reports zero gain for a split that separates nothing", () => {
    const samples: Sample[] = [
      { features: [0.2, 0, 0, 0], label: 1 },
      { features: [0.4, 0, 0, 0], label: 0 },
    ];
    expect(gainOf(samples, { feature: 0, threshold: 0.9 })).toBe(0);
    expect(gainOf([], { feature: 0, threshold: 0.5 })).toBe(0);
  });
});

describe("the split search", () => {
  const dataset = datasetFor(roundAt(1));

  it("agrees with an exhaustive search", () => {
    // bestSplitOn sweeps sorted values in O(n log n) rather than re-partitioning
    // at every threshold. This checks the fast version against the obvious one.
    const sample = dataset.train.slice(0, 60);

    for (let feature = 0; feature < FEATURES.length; feature += 1) {
      const values = [
        ...new Set(sample.map((entry) => entry.features[feature] ?? 0)),
      ].sort((a, b) => a - b);

      let brute = { gain: -1, threshold: 0 };
      for (let index = 0; index < values.length - 1; index += 1) {
        const threshold = (values[index]! + values[index + 1]!) / 2;
        const gain = gainOf(sample, { feature, threshold });
        if (gain > brute.gain) brute = { gain, threshold };
      }

      const fast = bestSplitOn(sample, feature);
      expect(fast, `feature ${feature}`).not.toBeNull();
      expect(fast!.gain, `feature ${feature} gain`).toBeCloseTo(brute.gain, 10);
      expect(fast!.threshold, `feature ${feature} threshold`).toBeCloseTo(
        brute.threshold,
        10,
      );
    }
  });

  it("declines to split what cannot be split", () => {
    expect(bestSplitOn([], 0)).toBeNull();
    expect(bestSplitOn([{ features: [0.5, 0, 0, 0], label: 1 }], 0)).toBeNull();
    // All identical values: no threshold separates them.
    expect(
      bestSplitOn(
        [
          { features: [0.5, 0, 0, 0], label: 1 },
          { features: [0.5, 0, 0, 0], label: 0 },
        ],
        0,
      ),
    ).toBeNull();
  });

  it("puts thresholds between observed values, not on them", () => {
    const best = bestSplitOn(dataset.train, 0)!;
    const values = dataset.train.map((entry) => entry.features[0] ?? 0);
    expect(values).not.toContain(best.threshold);
    expect(best.threshold).toBeGreaterThan(Math.min(...values));
    expect(best.threshold).toBeLessThan(Math.max(...values));
  });

  it("is deterministic", () => {
    expect(bestSplit(dataset.train)).toEqual(bestSplit(dataset.train));
  });

  it("describes a gate the way the node label reads", () => {
    expect(describeSplit({ feature: 0, threshold: 0.456 })).toBe("slope < 0.46");
  });
});

describe("the tree", () => {
  it("starts as a single leaf", () => {
    const tree = emptyTree();
    expect(Object.keys(tree)).toEqual([ROOT_ID]);
    expect(leavesOf(tree)).toHaveLength(1);
    expect(depthOf(tree)).toBe(0);
    expect(splitCountOf(tree)).toBe(0);
  });

  it("grows two leaves per split and tracks depth", () => {
    let tree = applySplit(emptyTree(), ROOT_ID, { feature: 0, threshold: 0.5 });
    expect(leavesOf(tree)).toHaveLength(2);
    expect(depthOf(tree)).toBe(1);

    tree = applySplit(tree, `${ROOT_ID}L`, { feature: 1, threshold: 0.5 });
    expect(leavesOf(tree)).toHaveLength(3);
    expect(depthOf(tree)).toBe(2);
    expect(splitCountOf(tree)).toBe(2);
  });

  it("refuses to split a node that already has a gate", () => {
    const once = applySplit(emptyTree(), ROOT_ID, { feature: 0, threshold: 0.5 });
    const twice = applySplit(once, ROOT_ID, { feature: 1, threshold: 0.2 });
    expect(twice).toBe(once);
  });

  it("prunes a whole subtree, leaving no orphans", () => {
    let tree = applySplit(emptyTree(), ROOT_ID, { feature: 0, threshold: 0.5 });
    tree = applySplit(tree, `${ROOT_ID}L`, { feature: 1, threshold: 0.5 });
    tree = applySplit(tree, `${ROOT_ID}LL`, { feature: 2, threshold: 0.5 });
    expect(Object.keys(tree)).toHaveLength(7);

    const pruned = pruneAt(tree, `${ROOT_ID}L`);
    expect(Object.keys(pruned)).toHaveLength(3);
    expect(pruned[`${ROOT_ID}LL`]).toBeUndefined();
    expect(pruned[`${ROOT_ID}L`]!.split).toBeNull();
    expect(depthOf(pruned)).toBe(1);
  });

  it("routes a sample to exactly one leaf", () => {
    let tree = applySplit(emptyTree(), ROOT_ID, { feature: 0, threshold: 0.5 });
    tree = applySplit(tree, `${ROOT_ID}L`, { feature: 1, threshold: 0.5 });

    const leafIds = new Set(leavesOf(tree).map((leaf) => leaf.id));
    for (const sample of datasetFor(roundAt(1)).train) {
      expect(leafIds.has(routeSample(tree, sample))).toBe(true);
    }
  });

  it("accounts for every sample at the root and splits them without loss", () => {
    const dataset = datasetFor(roundAt(1));
    const tree = applySplit(emptyTree(), ROOT_ID, { feature: 0, threshold: 0.5 });
    const stats = nodeStats(tree, dataset.train);
    expect(stats[ROOT_ID]!.counts.total).toBe(dataset.train.length);
    expect(
      stats[`${ROOT_ID}L`]!.counts.total + stats[`${ROOT_ID}R`]!.counts.total,
    ).toBe(dataset.train.length);
  });

  it("partitions on '< threshold goes left', matching the gate label", () => {
    const samples: Sample[] = [
      { features: [0.1, 0, 0, 0], label: 1 },
      { features: [0.5, 0, 0, 0], label: 0 },
      { features: [0.9, 0, 0, 0], label: 0 },
    ];
    const { left, right } = partition(samples, { feature: 0, threshold: 0.5 });
    expect(left).toHaveLength(1);
    expect(right).toHaveLength(2);
  });
});

describe("leaf votes come from training data only", () => {
  // If validation samples voted in the leaves they land in, every leaf would be
  // right about itself and validation accuracy would climb with depth forever.
  // The overfit lesson depends entirely on this being false.

  it("never reaches perfect validation accuracy on a deep tree", () => {
    const dataset = datasetFor(roundAt(1));
    const deep = buildGreedyTree(dataset.train, 12);
    const trainStats = nodeStats(deep, dataset.train);

    expect(accuracyOf(deep, trainStats, dataset.train)).toBeCloseTo(1, 1);
    expect(accuracyOf(deep, trainStats, dataset.validation)).toBeLessThan(0.95);
  });

  it("gives the same prediction whichever dataset is being scored", () => {
    const dataset = datasetFor(roundAt(2));
    const tree = buildGreedyTree(dataset.train, 3);
    const trainStats = nodeStats(tree, dataset.train);

    // Routing a validation sample must not consult validation labels.
    const sample = dataset.validation[0]!;
    const flipped: Sample = { ...sample, label: (1 - sample.label) as 0 | 1 };
    expect(predictWith(tree, trainStats, flipped)).toBe(
      predictWith(tree, trainStats, sample),
    );
  });

  it("falls back to the root majority for a leaf with no training samples", () => {
    const dataset = datasetFor(roundAt(1));
    // A gate no training sample can satisfy leaves one child empty.
    const tree = applySplit(emptyTree(), ROOT_ID, {
      feature: 0,
      threshold: -1,
    });
    const stats = nodeStats(tree, dataset.train);
    expect(stats[`${ROOT_ID}L`]!.counts.total).toBe(0);
    expect(predictWith(tree, stats, dataset.train[0]!)).toBe(
      stats[ROOT_ID]!.majority,
    );
  });
});

describe("the data", () => {
  it("is deterministic for a seed", () => {
    expect(datasetFor(roundAt(1))).toEqual(datasetFor(roundAt(1)));
  });

  it("holds validation apart from training, at the stated sizes", () => {
    for (const round of ROUNDS) {
      const dataset = datasetFor(round);
      expect(dataset.train).toHaveLength(round.trainPoints);
      expect(dataset.validation).toHaveLength(VALIDATION_POINTS);
      const seen = new Set(
        dataset.train.map((entry) => entry.features.join(",")),
      );
      expect(
        dataset.validation.filter((entry) => seen.has(entry.features.join(","))),
      ).toHaveLength(0);
    }
  });

  it("keeps the classes usable on every plot", () => {
    for (const round of ROUNDS) {
      const counts = countsOf(datasetFor(round).validation);
      const positive = counts.positive / counts.total;
      expect(positive, round.name).toBeGreaterThan(0.3);
      expect(positive, round.name).toBeLessThan(0.7);
    }
  });

  it("makes the two irrelevant readings genuinely uninformative", () => {
    // On the box and diagonal plots the useless features must be visibly worse in
    // the gain table, or the player has no way to learn to distrust them.
    for (const round of [roundAt(1), roundAt(2)]) {
      const gains = candidateSplits(datasetFor(round).train).map(
        (candidate) => candidate?.gain ?? 0,
      );
      const informative = Math.min(gains[0]!, gains[1]!);
      const noise = Math.max(gains[2]!, gains[3]!);
      expect(informative, round.name).toBeGreaterThan(noise * 2);
    }
  });
});

describe("depth is a tradeoff, measured", () => {
  it("never lets training accuracy fall as depth rises", () => {
    // Each split is chosen to reduce impurity on the samples in front of it, so
    // this direction is guaranteed. Validation has no such guarantee, and that
    // asymmetry is the whole lesson.
    for (const round of ROUNDS) {
      const curve = depthCurve(datasetFor(round), 9);
      for (let index = 1; index < curve.length; index += 1) {
        expect(
          curve[index]!.trainAccuracy,
          `${round.name} depth ${curve[index]!.depth}`,
        ).toBeGreaterThanOrEqual(curve[index - 1]!.trainAccuracy - 1e-9);
      }
    }
  });

  it("peaks validation partway down and then loses ground", () => {
    // Rounds 1 and 2 only. On the ridge plot greedy CART never learns anything at
    // all, so its curve has no peak to fall from — that failure is its own test
    // below, and overfitting there is checked by extending a tree that works.
    for (const round of [roundAt(1), roundAt(2)]) {
      const curve = depthCurve(datasetFor(round), 10);
      const peak = curve.reduce((best, point) =>
        point.validationAccuracy > best.validationAccuracy ? point : best,
      );
      const deepest = curve[curve.length - 1]!;

      expect(peak.depth, `${round.name} peaks at the very bottom`).toBeLessThan(
        deepest.depth,
      );
      expect(
        peak.validationAccuracy - deepest.validationAccuracy,
        `${round.name} never gives anything back`,
      ).toBeGreaterThan(OVERFIT_DROP);
    }
  });

  it("overfits the ridge plot too, once the tree that works is extended", () => {
    // The lesson has to hold on the plot where greedy is useless: a correct
    // two-gate tree gives ground back as soon as gates are piled below it.
    const dataset = datasetFor(roundAt(3));
    const correct = handBuiltXor();
    const extended = greedyExtend(correct, dataset, 6);

    const before = score(correct, dataset);
    const after = score(extended, dataset);

    expect(depthOf(extended)).toBeGreaterThan(depthOf(correct));
    expect(after.train).toBeGreaterThan(before.train);
    expect(after.validation).toBeLessThan(before.validation - OVERFIT_DROP);
    expect(after.starved).toBeGreaterThan(before.starved);
  });

  it("starves more leaves the deeper it goes", () => {
    const dataset = datasetFor(roundAt(2));
    const shallow = buildGreedyTree(dataset.train, 3);
    const deep = buildGreedyTree(dataset.train, 9);
    expect(
      starvedLeaves(deep, nodeStats(deep, dataset.train)).length,
    ).toBeGreaterThan(
      starvedLeaves(shallow, nodeStats(shallow, dataset.train)).length,
    );
  });
});

describe("every plot can be signed off", () => {
  it("clears the terrace plot with two gates", () => {
    const round = roundAt(1);
    const result = score(handBuiltBox(), datasetFor(round));
    expect(result.validation).toBeGreaterThanOrEqual(round.target);
    expect(result.starved).toBe(0);
  });

  it("clears the hillside plot with a staircase", () => {
    const round = roundAt(2);
    const dataset = datasetFor(round);
    const reachable = depthCurve(dataset, round.maxDepth).some(
      (point) => point.validationAccuracy >= round.target,
    );
    expect(reachable).toBe(true);
  });

  it("clears the ridge plot with two gates on the right features", () => {
    const round = roundAt(3);
    const result = score(handBuiltXor(), datasetFor(round));
    expect(result.validation).toBeGreaterThanOrEqual(round.target);
    expect(result.starved).toBe(0);
  });
});

describe("the greedy learner's blind spot", () => {
  // The ridge plot exists for this. Greedy CART maximises immediate gain, and on
  // an XOR boundary the immediate gain of the RIGHT split is almost zero, so it
  // takes a worthless feature instead and never recovers. A player who
  // understands that beats the algorithm outright.

  const round = roundAt(3);
  const dataset = datasetFor(round);

  it("leaves greedy CART near chance at every depth it is allowed", () => {
    for (const depth of [2, 3, 4, 5]) {
      const result = score(buildGreedyTree(dataset.train, depth), dataset);
      expect(result.validation, `greedy depth ${depth}`).toBeLessThan(0.65);
    }
  });

  it("lets a hand-built tree beat it by a wide margin", () => {
    const hand = score(handBuiltXor(), dataset);
    const greedy = score(buildGreedyTree(dataset.train, 5), dataset);
    expect(hand.validation).toBeGreaterThan(greedy.validation + 0.2);
  });

  it("gives the right first gate almost no immediate gain", () => {
    const gain = gainOf(dataset.train, { feature: 0, threshold: 0.5 });
    expect(gain).toBeLessThan(0.01);
  });

  it("reveals that gate through lookahead, where immediate gain cannot", () => {
    // This is why the split picker shows two columns.
    const right = { feature: 0, threshold: 0.5 };
    const lookahead = lookaheadGainOf(dataset.train, right);
    expect(lookahead).toBeGreaterThan(0.3);

    const table = candidateSplitsWithLookahead(dataset.train);
    const bestByGain = table.reduce((best, candidate) =>
      candidate !== null && (best === null || candidate.gain > best.gain)
        ? candidate
        : best,
    );
    const bestByLookahead = table.reduce((best, candidate) =>
      candidate !== null &&
      (best === null || candidate.lookaheadGain > best.lookaheadGain)
        ? candidate
        : best,
    );

    // They disagree, and the lookahead one is an informative feature.
    expect(bestByLookahead!.feature).not.toBe(bestByGain!.feature);
    expect(FEATURES[bestByLookahead!.feature]!.irrelevant).toBe(false);
  });
});

describe("naming the failure", () => {
  const round = roundAt(1);
  const dataset = datasetFor(round);

  it("wins when validation clears the bar inside the depth limit", () => {
    const evaluation = evaluateTree({
      round,
      tree: handBuiltBox(),
      dataset,
      peakValidation: 0.9,
      peakDepth: 2,
      peakTrainAccuracy: 0.94,
    });
    expect(evaluation.outcome).toBe("win");
    expect(evaluation.failure).toBeNull();
    expect(evaluation.score).toBeGreaterThan(0.7);
  });

  it("names Overfit depth when validation falls back from the player's own peak", () => {
    const deep = buildGreedyTree(dataset.train, 6);
    const evaluation = evaluateTree({
      round,
      tree: deep,
      dataset,
      // What they already had at depth 3, per the measured curve.
      peakValidation: 0.927,
      peakDepth: 3,
      peakTrainAccuracy: 0.95,
    });
    expect(evaluation.outcome).toBe("overfit-depth");
    expect(evaluation.failure?.name).toBe("Overfit depth");
    expect(evaluation.failure?.detail).toMatch(/prune back to depth 3/i);
    expect(evaluation.failure?.detail).toMatch(/\d+\.\d%/);
    // Training went UP while validation went down. Both must be reported.
    expect(evaluation.trainAccuracy).toBeGreaterThan(evaluation.peakValidation);
    expect(evaluation.validationAccuracy).toBeLessThan(0.927);
  });

  it("names the starved leaves inside the overfit diagnosis when there are any", () => {
    const deep = buildGreedyTree(dataset.train, 6);
    const evaluation = evaluateTree({
      round,
      tree: deep,
      dataset,
      peakValidation: 0.927,
      peakDepth: 3,
      peakTrainAccuracy: 0.95,
    });
    expect(evaluation.starved).toBeGreaterThan(0);
    expect(evaluation.failure?.detail).toMatch(
      new RegExp(`fewer than ${MIN_HONEST_LEAF} training samples`, "i"),
    );
  });

  it("names Underfit stump for a tree that cannot fit its own training data", () => {
    const evaluation = evaluateTree({
      round,
      tree: emptyTree(),
      dataset,
      peakValidation: 0,
      peakDepth: 0,
      peakTrainAccuracy: 0,
    });
    expect(evaluation.outcome).toBe("underfit-stump");
    expect(evaluation.failure?.name).toBe("Underfit stump");
    // The advice must be the opposite of the overfit advice.
    expect(evaluation.failure?.detail).toMatch(/add gates/i);
    expect(evaluation.failure?.detail).not.toMatch(/prune/i);
  });

  it("prefers the overfit diagnosis over the underfit one when both could apply", () => {
    // A deep tree fits training well, so underfitting cannot be the story.
    const deep = buildGreedyTree(dataset.train, 8);
    const evaluation = evaluateTree({
      round,
      tree: deep,
      dataset,
      peakValidation: 0.927,
      peakDepth: 3,
      peakTrainAccuracy: 0.95,
    });
    expect(evaluation.outcome).not.toBe("underfit-stump");
  });

  it("reports a plain miss without inventing a diagnosis", () => {
    const ridge = roundAt(3);
    const ridgeData = datasetFor(ridge);
    const tree = buildGreedyTree(ridgeData.train, 2);
    const evaluation = evaluateTree({
      round: ridge,
      tree,
      dataset: ridgeData,
      // No peak to have fallen from, and training is not hopeless.
      peakValidation: 0.5,
      peakDepth: 2,
      peakTrainAccuracy: 0.9,
    });
    expect(["missed", "underfit-stump"]).toContain(evaluation.outcome);
    expect(evaluation.failure).not.toBeNull();
  });

  it("rewards the same accuracy at lower depth", () => {
    const shallow = evaluateTree({
      round,
      tree: handBuiltBox(),
      dataset,
      peakValidation: 0.94,
      peakDepth: 2,
      peakTrainAccuracy: 0.94,
    });
    const deep = buildGreedyTree(dataset.train, 3);
    const deeper = evaluateTree({
      round,
      tree: deep,
      dataset,
      peakValidation: 0.94,
      peakDepth: 2,
      peakTrainAccuracy: 0.94,
    });
    // The two-gate tree scores at least as well despite fewer splits.
    expect(shallow.score).toBeGreaterThan(deeper.score);
    expect(shallow.depth).toBeLessThan(deeper.depth);
  });

  it("keeps the score inside the unit interval", () => {
    for (const entry of ROUNDS) {
      const data = datasetFor(entry);
      for (const depth of [0, 1, 3, 6, 9]) {
        const evaluation = evaluateTree({
          round: entry,
          tree: buildGreedyTree(data.train, depth),
          dataset: data,
          peakValidation: 0.9,
          peakDepth: 3,
          peakTrainAccuracy: 0.9,
        });
        expect(evaluation.score).toBeGreaterThanOrEqual(0);
        expect(evaluation.score).toBeLessThanOrEqual(1);
      }
    }
  });
});

describe("the rounds as a sequence", () => {
  it("teaches a different thing about axis-aligned splits in each", () => {
    expect(new Set(ROUNDS.map((round) => round.boundary)).size).toBe(
      ROUNDS.length,
    );
    for (const round of ROUNDS) {
      expect(round.lesson.length, round.name).toBeGreaterThan(40);
      expect(round.brief.length, round.name).toBeGreaterThan(40);
    }
  });

  it("solves the box in fewer gates than the diagonal", () => {
    const box = depthCurve(datasetFor(roundAt(1)), 8);
    const diagonal = depthCurve(datasetFor(roundAt(2)), 8);
    const firstClearing = (
      curve: typeof box,
      target: number,
    ): number =>
      curve.find((point) => point.validationAccuracy >= target)?.depth ?? 99;

    expect(firstClearing(box, roundAt(1).target)).toBeLessThan(
      firstClearing(diagonal, roundAt(2).target),
    );
  });

  it("clamps roundAt to the real range", () => {
    expect(roundAt(0).index).toBe(1);
    expect(roundAt(99).index).toBe(ROUNDS.length);
  });
});

describe("the store's build flow", () => {
  const store = useArchitectStore;

  beforeEach(() => {
    store.getState().restart();
  });

  it("starts on plot one with a single leaf and no peak", () => {
    expect(store.getState().roundIndex).toBe(1);
    expect(store.getState().splits).toBe(0);
    expect(store.getState().depth).toBe(0);
    expect(store.getState().peakValidation).toBe(0);
    expect(store.getState().selectedNodeId).toBe(ROOT_ID);
    expect(store.getState().phase).toBe("building");
  });

  it("hands the same data the tests measure against", () => {
    expect(store.getState().dataset).toEqual(datasetFor(roundAt(1)));
  });

  it("keeps accuracies in step with the tree on every change", () => {
    // withTree is the only writer, so these can never drift. Check it holds.
    store.getState().splitAt(ROOT_ID, { feature: 0, threshold: 0.45 });
    const state = store.getState();
    const stats = nodeStats(state.tree, state.dataset.train);
    expect(state.trainAccuracy).toBeCloseTo(
      accuracyOf(state.tree, stats, state.dataset.train),
      10,
    );
    expect(state.validationAccuracy).toBeCloseTo(
      accuracyOf(state.tree, stats, state.dataset.validation),
      10,
    );
    expect(state.depth).toBe(depthOf(state.tree));
    expect(state.splits).toBe(splitCountOf(state.tree));
  });

  it("tracks the player's own peak as they build", () => {
    store.getState().splitAt(ROOT_ID, { feature: 0, threshold: 0.45 });
    const afterOne = store.getState().validationAccuracy;
    expect(store.getState().peakValidation).toBeCloseTo(afterOne, 10);
    expect(store.getState().peakDepth).toBe(1);

    store.getState().splitAt(`${ROOT_ID}L`, { feature: 1, threshold: 0.4 });
    const afterTwo = store.getState().validationAccuracy;
    expect(afterTwo).toBeGreaterThan(afterOne);
    expect(store.getState().peakValidation).toBeCloseTo(afterTwo, 10);
    expect(store.getState().peakDepth).toBe(2);
  });

  it("does not lower the peak when the tree gets worse", () => {
    // The peak is a high-water mark. Overwriting it on a bad gate would erase the
    // evidence the overfit diagnosis depends on.
    store.getState().splitAt(ROOT_ID, { feature: 0, threshold: 0.45 });
    store.getState().splitAt(`${ROOT_ID}L`, { feature: 1, threshold: 0.4 });
    const peak = store.getState().peakValidation;
    const peakDepth = store.getState().peakDepth;

    // Pile on gates until validation drops.
    for (let pass = 0; pass < 4; pass += 1) {
      for (const leaf of leavesOf(store.getState().tree)) {
        const bucket = samplesAt(store.getState(), leaf.id);
        const candidate = bestSplit(bucket);
        if (candidate === null || candidate.gain <= 0) continue;
        store.getState().splitAt(leaf.id, {
          feature: candidate.feature,
          threshold: candidate.threshold,
        });
      }
    }

    expect(store.getState().validationAccuracy).toBeLessThan(peak);
    expect(store.getState().peakValidation).toBeCloseTo(peak, 10);
    expect(store.getState().peakDepth).toBe(peakDepth);
    expect(isGivingGroundBack(store.getState())).toBe(true);
  });

  it("refuses to split a node that already has a gate", () => {
    store.getState().splitAt(ROOT_ID, { feature: 0, threshold: 0.45 });
    const before = store.getState().tree;
    store.getState().splitAt(ROOT_ID, { feature: 1, threshold: 0.2 });
    expect(store.getState().tree).toBe(before);
  });

  it("recovers validation accuracy by pruning", () => {
    // The experiment that proves overfitting to the player: remove gates, and
    // accuracy on unseen plots goes UP.
    store.getState().splitAt(ROOT_ID, { feature: 0, threshold: 0.45 });
    store.getState().splitAt(`${ROOT_ID}L`, { feature: 1, threshold: 0.4 });
    const peak = store.getState().validationAccuracy;

    for (let pass = 0; pass < 4; pass += 1) {
      for (const leaf of leavesOf(store.getState().tree)) {
        const bucket = samplesAt(store.getState(), leaf.id);
        const candidate = bestSplit(bucket);
        if (candidate === null || candidate.gain <= 0) continue;
        store.getState().splitAt(leaf.id, {
          feature: candidate.feature,
          threshold: candidate.threshold,
        });
      }
    }
    const overfit = store.getState().validationAccuracy;
    expect(overfit).toBeLessThan(peak);

    // Prune at depth 2, which cuts everything below the two gates that worked.
    // Pruning the root's children instead would delete those gates too and leave
    // a depth-1 stump, which is a different mistake rather than a recovery.
    for (const node of Object.values(store.getState().tree)) {
      if (node.split !== null && node.depth >= 2) {
        store.getState().prune(node.id);
      }
    }

    expect(depthOf(store.getState().tree)).toBe(2);
    // The claim being tested: removing gates RAISED accuracy on unseen plots,
    // which is only possible if those gates described the sample and not the
    // ground. Not equality with the old peak — the root's other branch kept a
    // gate it did not have then, so this is a different depth-2 tree.
    expect(store.getState().validationAccuracy).toBeGreaterThan(overfit);
    expect(peak - store.getState().validationAccuracy).toBeLessThan(0.05);
  });

  it("clears a stale sign-off when the tree changes", () => {
    store.getState().signOff();
    expect(store.getState().evaluation).not.toBeNull();
    store.getState().splitAt(ROOT_ID, { feature: 0, threshold: 0.45 });
    expect(store.getState().evaluation).toBeNull();
    expect(store.getState().failure).toBeNull();
  });

  it("advances only after a plot is signed off", () => {
    store.getState().nextRound();
    expect(store.getState().roundIndex, "must not skip while building").toBe(1);

    store.getState().splitAt(ROOT_ID, { feature: 0, threshold: 0.45 });
    store.getState().splitAt(`${ROOT_ID}L`, { feature: 1, threshold: 0.4 });
    expect(store.getState().signOff().outcome).toBe("win");
    expect(store.getState().phase).toBe("cleared");

    store.getState().nextRound();
    expect(store.getState().roundIndex).toBe(2);
    expect(store.getState().splits).toBe(0);
    expect(store.getState().peakValidation).toBe(0);
    expect(store.getState().dataset).toEqual(datasetFor(roundAt(2)));
  });

  it("beats greedy CART on the ridge plot from the store", () => {
    // The whole point of plot 3, exercised through the real actions.
    store.getState().restart();
    store.getState().signOff();
    store.getState().splitAt(ROOT_ID, { feature: 0, threshold: 0.45 });
    store.getState().splitAt(`${ROOT_ID}L`, { feature: 1, threshold: 0.4 });
    store.getState().signOff();
    store.getState().nextRound();

    // Plot 2: grow greedily to the peak depth of 4.
    for (let pass = 0; pass < 4; pass += 1) {
      for (const leaf of leavesOf(store.getState().tree)) {
        if (leaf.depth !== pass) continue;
        const bucket = samplesAt(store.getState(), leaf.id);
        const candidate = bestSplit(bucket);
        if (candidate === null || candidate.gain <= 0) continue;
        store.getState().splitAt(leaf.id, {
          feature: candidate.feature,
          threshold: candidate.threshold,
        });
      }
    }
    expect(store.getState().signOff().outcome).toBe("win");
    store.getState().nextRound();
    expect(store.getState().roundIndex).toBe(3);

    // Plot 3: the hand-built XOR tree, which greedy cannot find.
    store.getState().splitAt(ROOT_ID, { feature: 0, threshold: 0.5 });
    store.getState().splitAt(`${ROOT_ID}L`, { feature: 1, threshold: 0.5 });
    store.getState().splitAt(`${ROOT_ID}R`, { feature: 1, threshold: 0.5 });

    const greedy = score(
      buildGreedyTree(store.getState().dataset.train, 5),
      store.getState().dataset,
    );
    expect(store.getState().validationAccuracy).toBeGreaterThan(
      greedy.validation + 0.2,
    );

    const evaluation = store.getState().signOff();
    expect(evaluation.outcome).toBe("win");
    expect(store.getState().phase).toBe("complete");
    expect(store.getState().clearedScores).toHaveLength(ROUNDS.length);
  });

  it("names Overfit depth through the real actions, not just the pure function", () => {
    // The peak is tracked live while building, so there is no need to sign the
    // good tree off first — and signing it off would close the plot (see the
    // re-entrancy tests below), which is not what this is about.
    store.getState().splitAt(ROOT_ID, { feature: 0, threshold: 0.45 });
    store.getState().splitAt(`${ROOT_ID}L`, { feature: 1, threshold: 0.4 });
    // The good tree IS the peak, and judging it now would be a win — the
    // contrast the deep tree below is judged against.
    const good = store.getState();
    expect(good.peakValidation).toBe(good.validationAccuracy);
    expect(isGivingGroundBack(good)).toBe(false);
    expect(
      evaluateTree({
        round: roundAt(good.roundIndex),
        tree: good.tree,
        dataset: good.dataset,
        peakValidation: good.peakValidation,
        peakDepth: good.peakDepth,
        peakTrainAccuracy: good.peakTrainAccuracy,
      }).outcome,
    ).toBe("win");

    for (let pass = 0; pass < 5; pass += 1) {
      for (const leaf of leavesOf(store.getState().tree)) {
        const bucket = samplesAt(store.getState(), leaf.id);
        const candidate = bestSplit(bucket);
        if (candidate === null || candidate.gain <= 0) continue;
        store.getState().splitAt(leaf.id, {
          feature: candidate.feature,
          threshold: candidate.threshold,
        });
      }
    }

    const evaluation = store.getState().signOff();
    expect(evaluation.outcome).toBe("overfit-depth");
    expect(store.getState().failure?.name).toBe("Overfit depth");
  });

  it("selects only nodes that exist", () => {
    store.getState().selectNode("nonsense");
    expect(store.getState().selectedNodeId).toBe(ROOT_ID);
  });
});

describe("what lookahead may and may not be credited with", () => {
  const terrace = datasetFor(roundAt(1));
  const ridge = datasetFor(roundAt(3));

  const gateCard = (round: Round, bucket: Sample[], split: { feature: number; threshold: number }) =>
    whyCardFor({
      kind: "gate-built",
      round,
      nodeId: ROOT_ID,
      split,
      bucket,
      trainAccuracy: 0.8,
      validationAccuracy: 0.8,
      previousValidation: 0.8,
      peakValidation: 0.8,
      peakDepth: 1,
      peakTrainAccuracy: 0.8,
      starved: 0,
      depth: 1,
    });

  it("measures what a gate unlocked against what the node already offered", () => {
    // The right ridge gate creates structure no single gate could see…
    expect(unlockedGainOf(ridge.train, { feature: 0, threshold: 0.5 })).toBeGreaterThan(0.3);
    // …while the terrace root's soil-pH sliver has a big lookahead only because
    // it left the node's own structure intact below it.
    const sliver = candidateSplits(terrace.train)[2]!;
    expect(FEATURES[sliver.feature]!.irrelevant).toBe(true);
    expect(lookaheadGainOf(terrace.train, sliver)).toBeGreaterThan(0.1);
    expect(unlockedGainOf(terrace.train, sliver)).toBeLessThan(0.01);
  });

  it("does not praise an irrelevant sliver as the right gate", () => {
    // Plot 1, first decision, one click on soil pH's Build button: the card
    // used to read "Almost no gain, and exactly the right gate".
    const sliver = candidateSplits(terrace.train)[2]!;
    const card = gateCard(roundAt(1), terrace.train, sliver);
    expect(card.tone).not.toBe("good");
    expect(card.title).not.toMatch(/right gate|a lot underneath/i);
  });

  it("still praises the right ridge gate, and says what CART actually does", () => {
    const card = gateCard(roundAt(3), ridge.train, { feature: 0, threshold: 0.5 });
    expect(card.tone).toBe("good");
    expect(card.title).toBe("Almost no gain now, a lot underneath");
    expect(card.body).toMatch(/Real CART only scores the gate it is about to build/);
    expect(card.body).not.toMatch(/looks one gate ahead/);
  });

  it("calls a sliver that passes the problem down a weak gate, in numbers", () => {
    // Ridge root, slope at the table's own threshold (< 0.04): no gain, and a
    // lookahead no bigger than what the node already had.
    const sliver = candidateSplits(ridge.train)[0]!;
    const card = gateCard(roundAt(3), ridge.train, sliver);
    expect(card.title).toMatch(/bought almost nothing/);
    expect(card.body).toMatch(/a single gate here could already buy/);
    expect(card.body).not.toMatch(/lookahead column is/);
  });

  it("shows the right gate's lookahead live in the manual readout", () => {
    // The table tries each reading at its best IMMEDIATE threshold, so on the
    // ridge it never shows slope < 0.50. The slider's readout must.
    render(
      createElement(SplitGate, {
        nodeId: ROOT_ID,
        bucket: ridge.train,
        isLeaf: true,
        atDepthLimit: false,
        onSplit: () => {},
        onPrune: () => {},
      }),
    );
    fireEvent.change(screen.getByRole("slider", { name: /Threshold on slope/ }), {
      target: { value: "0.5" },
    });
    const expected = lookaheadGainOf(ridge.train, { feature: 0, threshold: 0.5 });
    expect(expected).toBeGreaterThan(0.3);
    expect(screen.getByText(expected.toFixed(3))).toBeInTheDocument();
  });

  it("builds gates with the shared 44px Button, not a hand-rolled one", () => {
    render(
      createElement(SplitGate, {
        nodeId: ROOT_ID,
        bucket: terrace.train,
        isLeaf: true,
        atDepthLimit: false,
        onSplit: () => {},
        onPrune: () => {},
      }),
    );
    for (const meta of FEATURES) {
      const button = screen.getByRole("button", {
        name: new RegExp(`Build a gate on ${meta.name}`),
      });
      expect(button.className).toMatch(/\bmin-h-11\b/);
    }
  });

  it("reveals a split search that sorts numbers as numbers", () => {
    // Bare .sort() compares as strings: [10, 9, 2] becomes [10, 2, 9].
    expect(MATH_CODE).not.toMatch(/\.sort\(\)/);
    expect(MATH_CODE).toMatch(/\.sort\(\(a, b\) => a - b\)/);
  });
});

describe("the tree in words", () => {
  it("states every gate, branch and leaf verdict for assistive tech", () => {
    const dataset = datasetFor(roundAt(1));
    const tree = handBuiltBox();
    const stats = nodeStats(tree, dataset.train);
    render(createElement(TreeOutline, { tree, stats }));

    const list = screen.getByRole("list", { name: "The tree, gate by gate" });
    const text = list.textContent ?? "";
    expect(text).toContain(`gate n0, slope < 0.45, ${stats[ROOT_ID]!.counts.total} plots`);
    expect(text).toContain("yes: gate n0L, bedrock < 0.40");
    expect(text).toMatch(/no: leaf n0R, \d+ plots, impurity 0\.\d\d, says unsafe/);
    // One list item per node.
    expect(within(list).getAllByRole("listitem")).toHaveLength(Object.keys(tree).length);
  });
});

describe("growing greedily from the code lane", () => {
  const store = useArchitectStore;

  beforeEach(() => {
    store.getState().restart();
  });

  it("builds exactly the tree the breadth-first walk would, in one update", () => {
    const walked = (() => {
      let tree = emptyTree();
      const dataset = datasetFor(roundAt(1));
      for (let depth = 0; depth < 4; depth += 1) {
        for (const leaf of leavesOf(tree)) {
          if (leaf.depth !== depth) continue;
          const bucket = dataset.train.filter(
            (sample) => routeSample(tree, sample) === leaf.id,
          );
          const candidate = bestSplit(bucket);
          if (candidate === null || candidate.gain <= 0) continue;
          tree = applySplit(tree, leaf.id, candidate);
        }
      }
      return tree;
    })();

    let updates = 0;
    const unsubscribe = store.subscribe(() => {
      updates += 1;
    });
    store.getState().growGreedy(4);
    unsubscribe();

    expect(updates).toBe(1);
    expect(Object.keys(store.getState().tree).sort()).toEqual(Object.keys(walked).sort());
    for (const [id, node] of Object.entries(walked)) {
      expect(store.getState().tree[id]!.split).toEqual(
        node.split === null ? null : { feature: node.split.feature, threshold: node.split.threshold },
      );
    }
  });

  it("does not report overfitting after growing straight to greedy's peak", () => {
    // The starter snippet grows to the validation peak (terrace: depth 3,
    // 92.7%). Gate by gate, a half-built level scored 93.8% on the way, so the
    // ghost appeared the moment the snippet said "validation peaks here".
    store.getState().growGreedy(3);
    const state = store.getState();
    expect(state.depth).toBe(3);
    expect(state.peakDepth).toBe(3);
    expect(state.peakValidation).toBeCloseTo(state.validationAccuracy, 12);
    expect(isGivingGroundBack(state)).toBe(false);
    expect(state.whyCard?.title).toBe("Greedy CART, grown to depth 3");
    expect(state.signOff().outcome).toBe("win");
  });

  it("still remembers the complete levels it grew through on the way down", () => {
    // Growing to depth 6 passes through the depth-3 tree, so "you had 92.7% at
    // depth 3" is true of this run too — and signing it off is overfitting.
    store.getState().growGreedy(6);
    const state = store.getState();
    expect(state.peakDepth).toBe(3);
    expect(state.peakValidation).toBeGreaterThan(state.validationAccuracy + OVERFIT_DROP);
    expect(isGivingGroundBack(state)).toBe(true);
    expect(state.whyCard?.conceptHref).toBe("/concepts/overfitting");
    expect(state.signOff().outcome).toBe("overfit-depth");
    expect(store.getState().failure?.name).toBe("Overfit depth");
  });

  const runStarter = async () => {
    const logs: string[] = [];
    await createJsExecutor<ReturnType<typeof createCodeApi>>()(STARTER_CODE, {
      api: createCodeApi(),
      log: (...args: unknown[]) => logs.push(args.map(String).join(" ")),
      checkBudget: () => {},
    });
    return logs;
  };

  it("runs the starter snippet to a clean win, and can run it twice", async () => {
    const first = await runStarter();
    expect(first.some((line) => /verdict: win/.test(line))).toBe(true);
    expect(isGivingGroundBack(store.getState())).toBe(false);
    expect(store.getState().phase).toBe("cleared");

    await runStarter();
    expect(store.getState().clearedScores).toHaveLength(1);
    expect(store.getState().phase).toBe("cleared");
    expect(store.getState().codeLaneWin).toBe(true);
  });

  it("never builds past the plot's depth limit from the starter", async () => {
    // It used to sweep depthCurve(9) on every plot; the ridge allows 8.
    store.getState().growGreedy(2);
    store.getState().signOff();
    store.getState().nextRound();
    store.getState().growGreedy(4);
    store.getState().signOff();
    store.getState().nextRound();
    expect(store.getState().roundIndex).toBe(3);

    await runStarter();
    expect(store.getState().depth).toBeLessThanOrEqual(roundAt(3).maxDepth);
    expect(store.getState().failure?.detail ?? "").not.toMatch(/over the \d+ allowed/);
  });

  const runScript = async (code: string) => {
    const logs: string[] = [];
    await createJsExecutor<ReturnType<typeof createCodeApi>>()(code, {
      api: createCodeApi(),
      log: (...args: unknown[]) => logs.push(args.map(String).join(" ")),
      checkBudget: () => {},
    });
    return logs;
  };

  it("only says validation turns down past the peak when the sweep saw it", async () => {
    // The sweep stops at each plot's depth limit. On the ridge greedy is still
    // climbing there (depth 9 scores higher than 8), so "deeper than that,
    // validation does not" was a claim nothing had measured.
    for (const round of ROUNDS) {
      expect(store.getState().roundIndex).toBe(round.index);
      const logs = await runStarter();
      const curve = depthCurve(store.getState().dataset, round.maxDepth);
      const peak = curve.reduce((best, point) =>
        point.validationAccuracy > best.validationAccuracy ? point : best,
      );
      const said = (pattern: RegExp) => logs.some((line) => pattern.test(line));

      if (peak.depth < round.maxDepth) {
        expect(said(/deeper than that, training keeps rising and validation does not/)).toBe(true);
        // …and it is true of every deeper row the table printed.
        for (const point of curve.filter((entry) => entry.depth > peak.depth)) {
          expect(point.trainAccuracy, `depth ${point.depth}`).toBeGreaterThan(peak.trainAccuracy);
          expect(point.validationAccuracy, `depth ${point.depth}`).toBeLessThanOrEqual(
            peak.validationAccuracy,
          );
        }
      } else {
        expect(said(/deeper than that/), round.name).toBe(false);
        expect(said(/that is the depth limit/), round.name).toBe(true);
      }
      expect(said(/no depth up to the limit reaches/), round.name).toBe(
        peak.validationAccuracy < round.target,
      );

      if (round.index < ROUNDS.length) {
        // Walk on: the starter wins the first two plots itself.
        expect(store.getState().phase).toBe("cleared");
        store.getState().nextRound();
      }
    }
    // The ridge is the plot the limit line is for.
    expect(store.getState().roundIndex).toBe(3);
  });

  it("the experiment the starter's closing comment suggests names the failure", async () => {
    // "start it over first - api.retryRound() - then api.growGreedy(maxDepth)
    // and sign that off … on the first two plots the verdict changes". Without
    // the retry, the signed-off plot stays signed off and nothing is named.
    expect(STARTER_CODE).toContain("api.retryRound()");
    expect(STARTER_CODE).toContain("api.growGreedy(api.round().maxDepth)");
    for (const index of [1, 2]) {
      const starter = await runStarter();
      expect(starter.some((line) => /verdict: win/.test(line)), `plot ${index}`).toBe(true);

      const logs = await runScript(
        "api.retryRound();\napi.growGreedy(api.round().maxDepth);\nlog('verdict:', api.signOff().outcome);",
      );
      expect(logs.some((line) => /verdict: win/.test(line)), `plot ${index}`).toBe(false);
      expect(store.getState().failure?.name, `plot ${index}`).toBe("Overfit depth");
      expect(validationMetricState(store.getState())).toBe("bad");

      // Back to the peak to move on.
      store.getState().retryRound();
      await runStarter();
      store.getState().nextRound();
    }
  });
});

describe("the store cannot double-count, un-sign, or wipe a plot", () => {
  const store = useArchitectStore;

  beforeEach(() => {
    useProgression.setState({
      ...EMPTY_PROGRESSION,
      hydrated: true,
      syncError: null,
      lastGain: null,
    });
    useProgression.getState().setAdapter(createMemoryAdapter());
    store.getState().restart();
    store.getState().setLane("visual");
  });

  /** Sign off every plot with a tree that wins it. */
  const winEveryPlot = (source: "visual" | "code") => {
    store.getState().growGreedy(2);
    store.getState().signOff(source);
    store.getState().nextRound();
    store.getState().growGreedy(4);
    store.getState().signOff(source);
    store.getState().nextRound();
    store.getState().splitAt(ROOT_ID, { feature: 0, threshold: 0.5 });
    store.getState().splitAt(`${ROOT_ID}L`, { feature: 1, threshold: 0.5 });
    store.getState().splitAt(`${ROOT_ID}R`, { feature: 1, threshold: 0.5 });
    return store.getState().signOff(source);
  };

  it("banks a signed-off plot once, however many times it is signed off", () => {
    store.getState().growGreedy(2);
    expect(store.getState().signOff().outcome).toBe("win");
    store.getState().signOff();
    store.getState().signOff("code");
    expect(store.getState().clearedScores).toHaveLength(1);
    expect(store.getState().attempts).toBe(1);
    expect(store.getState().phase).toBe("cleared");
  });

  it("keeps a signed-off plot signed off when a worse tree is judged after it", () => {
    store.getState().growGreedy(3);
    const won = store.getState().signOff();
    store.getState().growGreedy(6);
    const judged = store.getState().signOff();

    // An honest verdict on the tree the caller asked about…
    expect(judged.outcome).toBe("overfit-depth");
    // …without un-signing the plot or re-scoring it.
    expect(store.getState().phase).toBe("cleared");
    expect(store.getState().clearedScores).toEqual([won.score]);
    expect(store.getState().failure).toBeNull();
  });

  it("records a finished survey with progression exactly once", () => {
    expect(winEveryPlot("visual").outcome).toBe("win");
    expect(store.getState().phase).toBe("complete");
    expect(useProgression.getState().games[SLUG]?.playCount).toBe(1);
    store.getState().signOff();
    store.getState().signOff("code");
    expect(useProgression.getState().games[SLUG]?.playCount).toBe(1);
  });

  it("retries the current plot and keeps the ones already signed off", () => {
    store.getState().growGreedy(2);
    store.getState().signOff();
    store.getState().nextRound();
    store.getState().signOff();
    expect(store.getState().failure?.name).toBe("Underfit stump");

    store.getState().retryRound();
    expect(store.getState().roundIndex).toBe(2);
    expect(store.getState().clearedScores).toHaveLength(1);
    expect(store.getState().failure).toBeNull();
    expect(store.getState().splits).toBe(0);
    expect(store.getState().peakValidation).toBe(0);
    expect(store.getState().phase).toBe("building");
  });

  it("counts the third star from the lane that signed off, not the tab that is open", () => {
    store.getState().setLane("code");
    winEveryPlot("visual");
    expect(useProgression.getState().games[SLUG]?.codeLaneCleared).toBe(false);
  });

  it("awards the code-lane clear when a plot is signed off through the api", () => {
    const api = createCodeApi();
    api.growGreedy(2);
    expect(api.signOff().outcome).toBe("win");
    store.getState().nextRound();
    store.getState().growGreedy(4);
    store.getState().signOff("visual");
    store.getState().nextRound();
    store.getState().splitAt(ROOT_ID, { feature: 0, threshold: 0.5 });
    store.getState().splitAt(`${ROOT_ID}L`, { feature: 1, threshold: 0.5 });
    store.getState().splitAt(`${ROOT_ID}R`, { feature: 1, threshold: 0.5 });
    store.getState().signOff("visual");
    expect(store.getState().phase).toBe("complete");
    expect(useProgression.getState().games[SLUG]?.codeLaneCleared).toBe(true);
  });

  it("rejects bad script input with a named error instead of corrupting state", () => {
    const api = createCodeApi();
    const before = store.getState().tree;
    expect(() => api.split("nowhere", 0, 0.5)).toThrow(/no node "nowhere"/);
    expect(() => api.split(ROOT_ID, 7, 0.5)).toThrow(/feature must be an integer/);
    expect(() => api.split(ROOT_ID, 0, Number.NaN)).toThrow(/finite number/);
    expect(() => api.gainOf(ROOT_ID, 1.5, 0.5)).toThrow(/feature/);
    expect(() => api.gainTable("n9")).toThrow(/no node/);
    expect(() => api.prune(ROOT_ID)).toThrow(/is a leaf/);
    // Depths are bounded, so a script cannot hang the tab building 1e9 trees.
    expect(() => api.growGreedy(1e9)).toThrow(/depth must be a whole number/);
    expect(() => api.depthCurve(-1)).toThrow(/depth/);
    expect(() => api.tryGreedy(2.5)).toThrow(/depth/);
    expect(() => api.starvedAtDepth(DEPTH_ARG_LIMIT + 1)).toThrow(/depth/);
    expect(store.getState().tree).toBe(before);

    api.split(ROOT_ID, 0, 0.45);
    expect(() => api.split(ROOT_ID, 1, 0.4)).toThrow(/already has a gate/);
    expect(() => api.greedyAt(ROOT_ID)).toThrow(/already has a gate/);
  });
});

describe("overfitting is only named on a tree that overfits", () => {
  const store = useArchitectStore;
  const OVERFIT = "/concepts/overfitting";

  /** Walk to a plot by winning the ones before it, as a player must. */
  const goToPlot = (index: number) => {
    store.getState().restart();
    if (index >= 2) {
      store.getState().growGreedy(2);
      store.getState().signOff();
      store.getState().nextRound();
    }
    if (index >= 3) {
      store.getState().growGreedy(4);
      store.getState().signOff();
      store.getState().nextRound();
    }
    expect(store.getState().roundIndex).toBe(index);
  };

  const judge = () => {
    const state = store.getState();
    return evaluateTree({
      round: roundAt(state.roundIndex),
      tree: state.tree,
      dataset: state.dataset,
      peakValidation: state.peakValidation,
      peakDepth: state.peakDepth,
      peakTrainAccuracy: state.peakTrainAccuracy,
    });
  };

  /** Copy that says the tree is NOT memorising anything. */
  const ALL_CLEAR =
    /nothing is being memorised|nothing here is about generalisation|generalising what it|holds outside the sample it learned/i;

  beforeEach(() => {
    store.getState().restart();
  });

  it("does not call a tree grown back to depth 1 overfit", () => {
    // The review's repro: depth 1 after depth 3 is 77.8% validation, 15 points
    // under the peak — with training 17 points under it too.
    store.getState().growGreedy(3);
    const peak = store.getState();
    store.getState().growGreedy(1);
    const state = store.getState();

    expect(state.trainAccuracy).toBeLessThan(peak.trainAccuracy);
    expect(isGivingGroundBack(state)).toBe(false);
    expect(state.whyCard?.tone).not.toBe("warn");
    expect(state.whyCard?.conceptHref).toBeUndefined();
    expect(state.whyCard?.body).not.toMatch(/fitting these/);
    // It says what did happen, in the numbers on screen.
    expect(state.whyCard?.body).toContain(
      `short of the ${(peak.validationAccuracy * 100).toFixed(1)}% you reached at depth 3`,
    );
    expect(state.whyCard?.body).toContain(
      `training is short of that tree's ${(peak.trainAccuracy * 100).toFixed(1)}%`,
    );
    expect(validationMetricState(state)).not.toBe("warn");
    // …and the sign-off one click later agrees: too little tree.
    expect(store.getState().signOff().outcome).toBe("underfit-stump");
  });

  it("does not call a different root gate overfit after pruning the first", () => {
    // Take the greedy root, prune it, try another reading: the basic visual-lane
    // loop. Training is under the peak's, so each gate is judged on what it
    // bought — and on soil pH that is nothing.
    store.getState().takeGreedySplit(ROOT_ID);
    store.getState().prune(ROOT_ID);
    for (const feature of [1, 2, 3]) {
      store.getState().splitAt(ROOT_ID, { feature, threshold: 0.5 });
      const state = store.getState();
      expect(state.trainAccuracy).toBeLessThan(state.peakTrainAccuracy);
      expect(state.whyCard?.title, `feature ${feature}`).not.toMatch(/validation down/);
      expect(state.whyCard?.conceptHref, `feature ${feature}`).toBeUndefined();
      store.getState().prune(ROOT_ID);
    }
    store.getState().splitAt(ROOT_ID, { feature: 2, threshold: 0.5 });
    expect(store.getState().whyCard?.title).toMatch(/bought almost nothing/);
  });

  it("does not call a gate that RAISED validation overfit after starting over", () => {
    // Greedy gates three levels deep, "Start the tree over", greedy root again:
    // validation goes from 68.0% to 77.8%. The peak is kept, and the card used to
    // read "Training up, validation down".
    for (let depth = 0; depth < 3; depth += 1) {
      for (const leaf of leavesOf(store.getState().tree)) {
        if (leaf.depth === depth) store.getState().takeGreedySplit(leaf.id);
      }
    }
    store.getState().clearTree();
    const before = store.getState().validationAccuracy;
    store.getState().takeGreedySplit(ROOT_ID);
    const state = store.getState();

    expect(state.validationAccuracy).toBeGreaterThan(before);
    expect(state.peakValidation - state.validationAccuracy).toBeGreaterThan(0.1);
    expect(isGivingGroundBack(state)).toBe(false);
    expect(state.whyCard?.tone).not.toBe("warn");
    expect(state.whyCard?.conceptHref).toBeUndefined();
    expect(state.whyCard?.body).toMatch(/— up \d+\.\d%/);
  });

  it("titles the card from the numbers: training up, or merely no higher", () => {
    const round = roundAt(1);
    const bucket = datasetFor(round).train;
    const card = (trainAccuracy: number) =>
      whyCardFor({
        kind: "gate-built",
        round,
        nodeId: ROOT_ID,
        split: { feature: 0, threshold: 0.45 },
        bucket,
        trainAccuracy,
        validationAccuracy: 0.9,
        previousValidation: 0.93,
        peakValidation: 0.94,
        peakDepth: 2,
        peakTrainAccuracy: 0.943,
        starved: 0,
        depth: 3,
      });
    expect(card(0.96).title).toBe("Training up, validation down");
    expect(card(0.943).title).toBe("Training no higher, validation down");
    expect(card(0.943).conceptHref).toBe(OVERFIT);
    // Training below the peak's: not overfitting, whatever validation did.
    expect(card(0.93).title).not.toMatch(/validation down/);
    expect(card(0.93).conceptHref).toBeUndefined();
  });

  it("ghost, WhyCard, live metric and sign-off agree on every plot, for every greedy depth pair", () => {
    // Grow greedy to depth a, then to depth b, on all three plots and every
    // pair of depths each allows. One predicate decides all four.
    const seen = { ghost: 0, clear: 0, smaller: 0 };
    for (const round of ROUNDS) {
      goToPlot(round.index);
      for (let a = 0; a <= round.maxDepth; a += 1) {
        for (let b = 0; b <= round.maxDepth; b += 1) {
          const label = `plot ${round.index}, depth ${a} then ${b}`;
          store.getState().retryRound();
          store.getState().growGreedy(a);
          store.getState().growGreedy(b);
          const state = store.getState();
          const ghost = isGivingGroundBack(state);
          const card = state.whyCard!;

          expect(ghost, label).toBe(givesGroundBack(state));
          expect(card.conceptHref === OVERFIT, label).toBe(ghost);
          expect(card.tone === "warn", label).toBe(ghost);
          expect(validationMetricState(state) === "warn", label).toBe(ghost);

          if (ghost) {
            seen.ghost += 1;
            expect(state.trainAccuracy, label).toBeGreaterThanOrEqual(
              state.peakTrainAccuracy,
            );
            expect(
              state.peakValidation - state.validationAccuracy,
              label,
            ).toBeGreaterThanOrEqual(GIVING_BACK_DROP);
          } else {
            seen.clear += 1;
          }
          // A greedy tree shallower than the peak is a prefix of it, so it has
          // less training accuracy: never overfitting.
          if (state.depth < state.peakDepth) {
            expect(ghost, label).toBe(false);
            if (state.peakValidation - state.validationAccuracy >= GIVING_BACK_DROP) {
              seen.smaller += 1;
              expect(card.body, label).toMatch(/fits less rather than memorising more/);
            }
          }

          // One click later, the verdict must not say the opposite.
          const evaluation = judge();
          expect(evaluation.givingGroundBack, label).toBe(ghost);
          const verdict = `${evaluation.failure?.detail ?? ""} ${
            whyCardFor({
              kind: "signed-off",
              round,
              evaluation,
              complete: false,
              attempts: 1,
            }).body
          }`;
          if (ghost) expect(verdict, label).not.toMatch(ALL_CLEAR);
          if (evaluation.outcome === "overfit-depth") expect(ghost, label).toBe(true);
        }
      }
    }
    // Not vacuous: every branch was reached.
    expect(seen.ghost).toBeGreaterThan(20);
    expect(seen.clear).toBeGreaterThan(20);
    expect(seen.smaller).toBeGreaterThan(20);
  });

  it("keeps the gate-built card and the ghost in step, gate by gate, on every plot", () => {
    // The visual lane's path: the greedy gate at every leaf, level by level, to
    // the plot's limit — then pruned back one level at a time and regrown.
    let gates = 0;
    let warnings = 0;
    for (const round of ROUNDS) {
      goToPlot(round.index);
      const check = (label: string) => {
        const state = store.getState();
        const ghost = isGivingGroundBack(state);
        expect(state.whyCard?.conceptHref === OVERFIT, label).toBe(ghost);
        expect(validationMetricState(state) === "warn", label).toBe(ghost);
        gates += 1;
        if (ghost) warnings += 1;
      };
      const grow = (to: number) => {
        for (let depth = 0; depth < to; depth += 1) {
          for (const leaf of leavesOf(store.getState().tree)) {
            if (leaf.depth !== depth) continue;
            const before = store.getState().tree;
            store.getState().takeGreedySplit(leaf.id);
            if (store.getState().tree !== before) {
              check(`plot ${round.index}, gate at ${leaf.id}`);
            }
          }
        }
      };
      grow(round.maxDepth);
      for (let depth = round.maxDepth - 1; depth >= 1; depth -= 1) {
        for (const node of Object.values(store.getState().tree)) {
          if (node.depth === depth && store.getState().tree[node.id]?.split) {
            store.getState().prune(node.id);
          }
        }
        // The prune card is never the overfitting warning.
        expect(store.getState().whyCard?.tone).toBe("info");
        grow(depth + 1);
      }
    }
    expect(gates).toBeGreaterThan(100);
    expect(warnings).toBeGreaterThan(0);
  });

  it("explains the ghost rather than denying it when the ridge's gates fit noise", () => {
    // Greedy's first gates on the ridge raise training and lower validation, so
    // the ghost is right — and the tree is still far too simple, so the sign-off
    // is Underfit stump. It must say both.
    goToPlot(3);
    store.getState().growGreedy(0);
    store.getState().growGreedy(2);
    expect(isGivingGroundBack(store.getState())).toBe(true);
    const evaluation = store.getState().signOff();
    expect(evaluation.outcome).toBe("underfit-stump");
    expect(evaluation.failure?.detail).toMatch(
      /fitting these 200 plots rather than the ground/,
    );
    expect(evaluation.failure?.detail).not.toMatch(/nothing is being memorised/);
    expect(evaluation.failure?.detail).toMatch(/add gates/i);
  });

  it("does not call a tree past its peak 'narrow' when it scrapes a win", () => {
    // Terrace, depth 3 then 4: 2.2 points under the peak — under the named
    // failure's 2.5, and over the target. It wins, with the ghost up.
    store.getState().growGreedy(3);
    store.getState().growGreedy(4);
    expect(isGivingGroundBack(store.getState())).toBe(true);
    const evaluation = store.getState().signOff();
    expect(evaluation.outcome).toBe("win");
    const body = store.getState().whyCard?.body ?? "";
    expect(body).toContain(
      `below the ${(evaluation.peakValidation * 100).toFixed(1)}% you had at depth 3`,
    );
    expect(body).not.toMatch(ALL_CLEAR);
  });
});

describe("the live metric's colour", () => {
  const store = useArchitectStore;

  beforeEach(() => {
    store.getState().restart();
  });

  it("is neutral below the target, good at it, and bad on a named failure", () => {
    expect(validationMetricState(store.getState())).toBeUndefined();
    store.getState().growGreedy(2);
    expect(validationMetricState(store.getState())).toBe("good");
    store.getState().restart();
    store.getState().signOff();
    expect(store.getState().failure?.name).toBe("Underfit stump");
    expect(validationMetricState(store.getState())).toBe("bad");
  });

  it("warns on a tree grown past its peak even after the plot is signed off", () => {
    // The starter clears the plot, and the experiment its comment suggests
    // grows past the peak afterwards: the metric used to stay green.
    store.getState().growGreedy(3);
    store.getState().signOff();
    expect(store.getState().phase).toBe("cleared");
    expect(validationMetricState(store.getState())).toBe("good");

    store.getState().growGreedy(roundAt(1).maxDepth);
    expect(store.getState().phase).toBe("cleared");
    expect(isGivingGroundBack(store.getState())).toBe(true);
    expect(validationMetricState(store.getState())).toBe("warn");
  });
});
