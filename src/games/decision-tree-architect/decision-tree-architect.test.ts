import { beforeEach, describe, expect, it } from "vitest";
import {
  FEATURES,
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
  type Dataset,
  type Round,
  type Sample,
  type Tree,
} from "./ml";
import {
  isGivingGroundBack,
  samplesAt,
  useArchitectStore,
} from "./store";

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
    store.getState().splitAt(ROOT_ID, { feature: 0, threshold: 0.45 });
    store.getState().splitAt(`${ROOT_ID}L`, { feature: 1, threshold: 0.4 });
    store.getState().signOff();
    expect(store.getState().failure).toBeNull();

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
