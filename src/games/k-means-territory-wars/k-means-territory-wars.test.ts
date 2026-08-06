import { beforeEach, describe, expect, it } from "vitest";
import {
  EMPTY_PROGRESSION,
  createMemoryAdapter,
  useProgression,
} from "@/engine/progression";
import {
  CONVERGENCE_EPSILON,
  K_FITNESS_DECAY,
  LOCAL_MINIMUM_RATIO,
  MAX_K,
  POINTS_PER_BLOB,
  ROUNDS,
  WIN_SCORE,
  assignPoints,
  clusterSizes,
  elbowCurve,
  elbowKFor,
  evaluate,
  generateVillages,
  inertiaOfAssignment,
  inertiaOfCentroids,
  kMeansPlusPlusInit,
  largestShift,
  nearestCentroid,
  runToConvergence,
  solveKMeans,
  updateCentroids,
  type Centroid,
  type Point,
} from "./ml";
import { useKMeansStore } from "./store";
import { whyCardFor } from "./why-cards";
import { seededRandom } from "@/lib/utils";

const flagsAt = (positions: Array<[number, number]>): Centroid[] =>
  positions.map(([x, y], id) => ({ id, x, y }));

/** True blob centres, from the generator's labels. Tests only — the game is
 *  unsupervised and never reads `trueCluster`. */
function blobCentre(points: Point[], blob: number) {
  const members = points.filter((point) => point.trueCluster === blob);
  return {
    x: members.reduce((sum, p) => sum + p.x, 0) / members.length,
    y: members.reduce((sum, p) => sum + p.y, 0) / members.length,
  };
}

/**
 * A starting layout that strands k-means in a local optimum at the correct k:
 * two flags split blob 0 while one flag straddles the last two blobs.
 *
 * This is a genuine local optimum, not a contrivance — no single flag improves
 * inertia by crossing the empty gap on its own, so the algorithm has nowhere to
 * go. Screening confirmed it traps every shipped round.
 *
 * Layouts that do NOT work, recorded so they aren't retried: leaving a blob
 * entirely unserved always escapes (a flag migrates over), and fanning every flag
 * inside one blob escapes on roughly half the maps.
 */
function straddleStart(points: Point[], k: number): Centroid[] {
  const centres = Array.from({ length: k }, (_, blob) =>
    blobCentre(points, blob),
  );
  const first = centres[0]!;
  const layout: Centroid[] = [
    { id: 0, x: first.x - 0.03, y: first.y - 0.03 },
    { id: 1, x: first.x + 0.03, y: first.y + 0.03 },
  ];
  for (let blob = 1; blob <= k - 3; blob += 1) {
    const centre = centres[blob]!;
    layout.push({ id: layout.length, x: centre.x, y: centre.y });
  }
  const penultimate = centres[k - 2]!;
  const last = centres[k - 1]!;
  layout.push({
    id: layout.length,
    x: (penultimate.x + last.x) / 2,
    y: (penultimate.y + last.y) / 2,
  });
  return layout.slice(0, k);
}

/** Every flag jammed into one corner — reliably strands at least one. */
function cornerStart(k: number): Centroid[] {
  return Array.from({ length: k }, (_, index) => ({
    id: index,
    x: 0.06 + index * 0.008,
    y: 0.06 + index * 0.008,
  }));
}

// ═══════════════════════════════════════════════════════════════════════════
// THE LESSON
//
// K-Means Territory Wars teaches three things: clustering finds structure
// without labels, k matters, and initialization matters. The trap is that
// inertia — the live metric — falls monotonically as k rises, so "minimise the
// metric" is a gameable objective. These assertions are what stop the game from
// rewarding the trap.
// ═══════════════════════════════════════════════════════════════════════════

describe("the lesson holds on every shipped round", () => {
  it.each([...ROUNDS])(
    "seed $seed with $trueK blobs teaches the intended lesson",
    ({ seed, trueK }) => {
      const { points } = generateVillages(seed, trueK);
      const curve = elbowCurve(points, MAX_K, seed);
      const elbow = elbowKFor(curve);

      // 1. The elbow must recover the true number of blobs. If it didn't, the
      //    game would mark a correct answer wrong.
      expect(elbow, `elbow for seed ${seed}`).toBe(trueK);

      const evalAt = (centroids: Centroid[], converged = true) =>
        evaluate({
          points: assignPoints(points, centroids),
          centroids,
          converged,
          elbowK: elbow,
          bestInertiaAtK: solveKMeans(points, centroids.length, { seed })
            .inertia,
        });

      // 2. Playing it right must win: k at the elbow, converged.
      const right = evalAt(
        runToConvergence(points, solveKMeans(points, elbow, { seed }).centroids)
          .centroids,
      );
      expect(right.outcome, `right-k on seed ${seed}`).toBe("win");
      expect(right.score).toBeGreaterThanOrEqual(WIN_SCORE);

      // 3. THE TRAP: more flags genuinely produce lower inertia...
      const greedyK = Math.min(MAX_K, elbow + 3);
      const greedy = evalAt(
        runToConvergence(points, solveKMeans(points, greedyK, { seed }).centroids)
          .centroids,
      );
      expect(greedy.inertia).toBeLessThan(right.inertia);

      // ...and must still be diagnosed as a bad k, not rewarded.
      expect(greedy.outcome, `greedy-k on seed ${seed}`).toBe("bad-k");
      expect(greedy.score).toBeLessThan(WIN_SCORE);

      // 4. Too few flags is also a bad k, from the other direction.
      const timid = evalAt(
        runToConvergence(
          points,
          solveKMeans(points, Math.max(1, elbow - 2), { seed }).centroids,
        ).centroids,
      );
      expect(timid.outcome, `timid-k on seed ${seed}`).toBe("bad-k");
      expect(timid.inertia).toBeGreaterThan(right.inertia);

      // 5. Initialization must be able to change the outcome at the SAME k, or
      //    "results depend heavily on initialization" is an unbacked claim.
      const trapped = evalAt(
        runToConvergence(points, straddleStart(points, elbow)).centroids,
      );
      expect(trapped.outcome, `local minimum on seed ${seed}`).toBe(
        "local-minimum",
      );
      // A local minimum by definition keeps every flag fed — a stranded flag is
      // a different failure with a different name.
      expect(trapped.emptyClusters).toEqual([]);
      expect(trapped.inertia).toBeGreaterThan(right.inertia);
      expect(trapped.convergenceQuality).toBeLessThan(1 / LOCAL_MINIMUM_RATIO);

      // 6. And a stranded flag must be diagnosable as its own failure.
      const stranded = evalAt(
        runToConvergence(points, cornerStart(elbow)).centroids,
      );
      expect(stranded.outcome, `empty cluster on seed ${seed}`).toBe(
        "empty-cluster",
      );
      expect(stranded.emptyClusters.length).toBeGreaterThan(0);
    },
  );

  it("never compares a number against itself in the too-few-flags message", () => {
    // A player who converges perfectly at k=2 has inertia equal to the best at
    // k=2, so the naive phrasing produced "stuck at 8.92 versus 8.92
    // achievable" — true, and useless. Found in a browser playthrough.
    const { seed, trueK } = ROUNDS[0]!;
    const { points } = generateVillages(seed, trueK);
    const elbow = elbowKFor(elbowCurve(points, MAX_K, seed));

    const twoFlags = runToConvergence(
      points,
      solveKMeans(points, 2, { seed }).centroids,
    ).centroids;

    const result = evaluate({
      points: assignPoints(points, twoFlags),
      centroids: twoFlags,
      converged: true,
      elbowK: elbow,
      bestInertiaAtK: solveKMeans(points, 2, { seed }).inertia,
      bestInertiaAtElbowK: solveKMeans(points, elbow, { seed }).inertia,
    });

    expect(result.outcome).toBe("bad-k");
    const detail = result.failure!.detail;

    // Pull every number out of the copy; no value may appear twice as a
    // "you got X versus X achievable" comparison.
    const numbers = detail.match(/\d+\.\d+/g) ?? [];
    expect(new Set(numbers).size).toBe(numbers.length);
    // And the comparison it does make must be the useful one.
    expect(numbers.length).toBeGreaterThanOrEqual(2);
  });

  it("names each failure with numbers that back it up", () => {
    const { seed, trueK } = ROUNDS[0]!;
    const { points } = generateVillages(seed, trueK);
    const elbow = elbowKFor(elbowCurve(points, MAX_K, seed));

    const greedyK = elbow + 3;
    const centroids = runToConvergence(
      points,
      solveKMeans(points, greedyK, { seed }).centroids,
    ).centroids;

    const result = evaluate({
      points: assignPoints(points, centroids),
      centroids,
      converged: true,
      elbowK: elbow,
      bestInertiaAtK: solveKMeans(points, greedyK, { seed }).inertia,
    });

    expect(result.failure).not.toBeNull();
    expect(result.failure!.name).toBe("Bad k");
    expect(result.failure!.detail).toContain(String(greedyK));
    expect(result.failure!.detail).toContain(String(elbow));
  });

  it("refuses to judge a board that has never been assigned", () => {
    // Without this guard, unassigned villages read as "every cluster is empty"
    // and inertia 0 — a confident verdict built on no data.
    const { points } = generateVillages(3001, 3);
    const result = evaluate({
      points,
      centroids: flagsAt([
        [0.3, 0.3],
        [0.7, 0.7],
      ]),
      converged: true,
      elbowK: 3,
      bestInertiaAtK: 1,
    });

    expect(result.outcome).toBe("not-converged");
    expect(result.failure).toBeNull();
    expect(result.emptyClusters).toEqual([]);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// The generative process
// ═══════════════════════════════════════════════════════════════════════════

describe("generateVillages", () => {
  it("is deterministic for a seed", () => {
    expect(generateVillages(11, 3)).toEqual(generateVillages(11, 3));
    expect(generateVillages(11, 3).points).not.toEqual(
      generateVillages(12, 3).points,
    );
  });

  it("makes trueK blobs of equal size", () => {
    for (const trueK of [3, 4, 5]) {
      const { points } = generateVillages(7, trueK);
      expect(points).toHaveLength(trueK * POINTS_PER_BLOB);
      for (let blob = 0; blob < trueK; blob += 1) {
        expect(points.filter((p) => p.trueCluster === blob)).toHaveLength(
          POINTS_PER_BLOB,
        );
      }
    }
  });

  it("keeps every village on the map", () => {
    for (const point of generateVillages(9, 5).points) {
      expect(point.x).toBeGreaterThanOrEqual(0);
      expect(point.x).toBeLessThanOrEqual(1);
      expect(point.y).toBeGreaterThanOrEqual(0);
      expect(point.y).toBeLessThanOrEqual(1);
    }
  });

  it("starts every village unassigned — this is unsupervised", () => {
    for (const point of generateVillages(9, 3).points) {
      expect(point.clusterId).toBeNull();
    }
  });

  it("separates the blobs well enough for the elbow to be crisp", () => {
    const { points } = generateVillages(3001, 4);
    const centres = [0, 1, 2, 3].map((blob) => blobCentre(points, blob));
    for (let i = 0; i < centres.length; i += 1) {
      for (let j = i + 1; j < centres.length; j += 1) {
        const distance = Math.hypot(
          centres[i]!.x - centres[j]!.x,
          centres[i]!.y - centres[j]!.y,
        );
        expect(distance).toBeGreaterThan(0.25);
      }
    }
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// The algorithm
// ═══════════════════════════════════════════════════════════════════════════

describe("nearestCentroid", () => {
  it("picks the closest flag", () => {
    const flags = flagsAt([
      [0, 0],
      [1, 1],
    ]);
    expect(nearestCentroid({ x: 0.1, y: 0.1 }, flags)).toBe(0);
    expect(nearestCentroid({ x: 0.9, y: 0.9 }, flags)).toBe(1);
  });

  it("returns null when there are no flags", () => {
    expect(nearestCentroid({ x: 0.5, y: 0.5 }, [])).toBeNull();
  });

  it("breaks exact ties toward the lower index, deterministically", () => {
    const flags = flagsAt([
      [0, 0],
      [1, 0],
    ]);
    expect(nearestCentroid({ x: 0.5, y: 0 }, flags)).toBe(0);
  });
});

describe("assignPoints", () => {
  it("gives every village the index of its nearest flag", () => {
    const points: Point[] = [
      { id: "a", x: 0.05, y: 0.05, clusterId: null, trueCluster: 0 },
      { id: "b", x: 0.95, y: 0.95, clusterId: null, trueCluster: 1 },
    ];
    const assigned = assignPoints(
      points,
      flagsAt([
        [0, 0],
        [1, 1],
      ]),
    );
    expect(assigned.map((p) => p.clusterId)).toEqual([0, 1]);
  });

  it("leaves villages unassigned when there are no flags", () => {
    const { points } = generateVillages(1, 3);
    expect(assignPoints(points, []).every((p) => p.clusterId === null)).toBe(
      true,
    );
  });

  it("does not mutate the input", () => {
    const { points } = generateVillages(1, 3);
    assignPoints(points, flagsAt([[0.5, 0.5]]));
    expect(points.every((p) => p.clusterId === null)).toBe(true);
  });
});

describe("updateCentroids", () => {
  it("moves each flag to the mean of its villages", () => {
    const points: Point[] = [
      { id: "a", x: 0, y: 0, clusterId: 0, trueCluster: 0 },
      { id: "b", x: 1, y: 1, clusterId: 0, trueCluster: 0 },
    ];
    const moved = updateCentroids(points, flagsAt([[0.9, 0.1]]));
    expect(moved[0]!.x).toBeCloseTo(0.5, 10);
    expect(moved[0]!.y).toBeCloseTo(0.5, 10);
  });

  it("leaves an empty flag exactly where it is", () => {
    // Deliberate: a stranded flag is a failure the player must see, not
    // something to silently re-seed.
    const points: Point[] = [
      { id: "a", x: 0.2, y: 0.2, clusterId: 0, trueCluster: 0 },
    ];
    const moved = updateCentroids(
      points,
      flagsAt([
        [0.2, 0.2],
        [0.8, 0.8],
      ]),
    );
    expect(moved[1]).toEqual({ id: 1, x: 0.8, y: 0.8 });
  });

  it("ignores unassigned villages", () => {
    const points: Point[] = [
      { id: "a", x: 0, y: 0, clusterId: 0, trueCluster: 0 },
      { id: "b", x: 1, y: 1, clusterId: null, trueCluster: 0 },
    ];
    const moved = updateCentroids(points, flagsAt([[0.5, 0.5]]));
    expect(moved[0]!.x).toBeCloseTo(0, 10);
  });
});

describe("inertia", () => {
  const points: Point[] = [
    { id: "a", x: 0, y: 0, clusterId: 0, trueCluster: 0 },
    { id: "b", x: 0, y: 2, clusterId: 0, trueCluster: 0 },
  ];

  it("sums squared distances to the assigned flag", () => {
    // Flag at (0,1): each village is 1 away, squared = 1, total 2.
    expect(inertiaOfAssignment(points, flagsAt([[0, 1]]))).toBeCloseTo(2, 10);
  });

  it("is zero when every flag sits on its single village", () => {
    const single: Point[] = [
      { id: "a", x: 0.3, y: 0.4, clusterId: 0, trueCluster: 0 },
    ];
    expect(inertiaOfAssignment(single, flagsAt([[0.3, 0.4]]))).toBeCloseTo(0, 12);
  });

  it("reflects the CURRENT assignment, so it goes stale when flags move", () => {
    // This is the mechanism behind the "colours are out of date" hint: the
    // assignment is fixed, but the flag moved, so inertia changes.
    const before = inertiaOfAssignment(points, flagsAt([[0, 1]]));
    const after = inertiaOfAssignment(points, flagsAt([[0, 5]]));
    expect(after).toBeGreaterThan(before);
  });

  it("inertiaOfCentroids reassigns first, so it can only be lower", () => {
    const { points: villages } = generateVillages(3001, 3);
    const flags = flagsAt([
      [0.2, 0.2],
      [0.8, 0.8],
    ]);
    const stale = assignPoints(villages, flagsAt([[0.5, 0.5]]));
    expect(inertiaOfCentroids(villages, flags)).toBeLessThanOrEqual(
      inertiaOfAssignment(stale, flags),
    );
  });

  it("falls monotonically as k rises — the trap, stated as a test", () => {
    const { points: villages } = generateVillages(3001, 3);
    const curve = elbowCurve(villages, MAX_K, 3001);
    for (let i = 1; i < curve.length; i += 1) {
      expect(curve[i]!.inertia).toBeLessThanOrEqual(curve[i - 1]!.inertia + 1e-9);
    }
  });
});

describe("the assign/update loop", () => {
  it("never increases inertia across a full step", () => {
    const { points } = generateVillages(3002, 4);
    let centroids = flagsAt([
      [0.2, 0.2],
      [0.3, 0.8],
      [0.8, 0.3],
      [0.7, 0.7],
    ]);
    let previous = Number.POSITIVE_INFINITY;

    for (let step = 0; step < 12; step += 1) {
      const assigned = assignPoints(points, centroids);
      centroids = updateCentroids(assigned, centroids);
      const inertia = inertiaOfCentroids(points, centroids);
      expect(inertia).toBeLessThanOrEqual(previous + 1e-9);
      previous = inertia;
    }
  });

  it("reaches a fixed point where nothing moves", () => {
    const { points } = generateVillages(3001, 3);
    const result = runToConvergence(
      points,
      solveKMeans(points, 3, { seed: 3001 }).centroids,
    );
    const assigned = assignPoints(points, result.centroids);
    const moved = updateCentroids(assigned, result.centroids);
    expect(largestShift(result.centroids, moved)).toBeLessThan(
      CONVERGENCE_EPSILON,
    );
  });

  it("largestShift reports the furthest a flag travelled", () => {
    expect(
      largestShift(flagsAt([[0, 0]]), flagsAt([[0.3, 0.4]])),
    ).toBeCloseTo(0.5, 10);
    expect(largestShift(flagsAt([[0, 0]]), flagsAt([[0, 0]]))).toBe(0);
  });

  it("clusterSizes counts villages per flag", () => {
    const { points } = generateVillages(3001, 3);
    const flags = solveKMeans(points, 3, { seed: 3001 }).centroids;
    const sizes = clusterSizes(assignPoints(points, flags), flags);
    expect(sizes).toHaveLength(3);
    expect(sizes.reduce((a, b) => a + b, 0)).toBe(points.length);
  });
});

describe("solver", () => {
  it("k-means++ returns k distinct starting flags", () => {
    const { points } = generateVillages(3001, 4);
    const init = kMeansPlusPlusInit(points, 4, seededRandom(5));
    expect(init).toHaveLength(4);
    const keys = new Set(init.map((c) => `${c.x},${c.y}`));
    expect(keys.size).toBe(4);
  });

  it("multi-restart is at least as good as a single run", () => {
    const { points } = generateVillages(3001, 4);
    const single = runToConvergence(
      points,
      kMeansPlusPlusInit(points, 4, seededRandom(99)),
    );
    const best = solveKMeans(points, 4, { seed: 3001 });
    expect(best.inertia).toBeLessThanOrEqual(single.inertia + 1e-9);
  });

  it("is deterministic for a seed", () => {
    const { points } = generateVillages(3001, 3);
    expect(solveKMeans(points, 3, { seed: 1 }).inertia).toBeCloseTo(
      solveKMeans(points, 3, { seed: 1 }).inertia,
      12,
    );
  });

  it("recovers the true blob centres at the right k", () => {
    const { points } = generateVillages(3001, 3);
    const solved = solveKMeans(points, 3, { seed: 3001 });
    const trueCentres = [0, 1, 2].map((blob) => blobCentre(points, blob));

    // Every true centre should have a flag essentially on top of it.
    for (const centre of trueCentres) {
      const closest = Math.min(
        ...solved.centroids.map((c) => Math.hypot(c.x - centre.x, c.y - centre.y)),
      );
      expect(closest).toBeLessThan(0.03);
    }
  });
});

describe("elbowKFor", () => {
  it("finds the k after the largest relative drop", () => {
    const curve = [
      { k: 1, inertia: 100 },
      { k: 2, inertia: 90 },
      { k: 3, inertia: 10 }, // 9x drop
      { k: 4, inertia: 9 },
    ];
    expect(elbowKFor(curve)).toBe(3);
  });

  it("handles the degenerate cases", () => {
    expect(elbowKFor([])).toBe(1);
    expect(elbowKFor([{ k: 1, inertia: 5 }])).toBe(1);
  });

  it("stops rather than dividing by a zero inertia", () => {
    const curve = [
      { k: 1, inertia: 10 },
      { k: 2, inertia: 2 },
      { k: 3, inertia: 0 },
    ];
    expect(() => elbowKFor(curve)).not.toThrow();
    expect(elbowKFor(curve)).toBe(2);
  });

  it("recovers the blob count for every trueK we ship", () => {
    for (const trueK of [3, 4, 5]) {
      for (let seed = 3001; seed <= 3010; seed += 1) {
        const { points } = generateVillages(seed, trueK);
        expect(
          elbowKFor(elbowCurve(points, MAX_K, seed)),
          `trueK=${trueK} seed=${seed}`,
        ).toBe(trueK);
      }
    }
  });
});

describe("scoring", () => {
  const { points } = generateVillages(3001, 3);

  it("caps the score below the win bar when k is off by one", () => {
    // Choosing k correctly is the lesson, so it is not optional: one flag out
    // costs K_FITNESS_DECAY, which must be enough to miss the bar.
    expect(1 - K_FITNESS_DECAY).toBeLessThan(WIN_SCORE);
  });

  it("scores a perfect solve at the elbow as 1", () => {
    const centroids = solveKMeans(points, 3, { seed: 3001 }).centroids;
    const result = evaluate({
      points: assignPoints(points, centroids),
      centroids,
      converged: true,
      elbowK: 3,
      bestInertiaAtK: solveKMeans(points, 3, { seed: 3001 }).inertia,
    });
    expect(result.convergenceQuality).toBeCloseTo(1, 6);
    expect(result.kFitness).toBe(1);
    expect(result.score).toBeCloseTo(1, 6);
  });

  it("reports not-converged before naming a k or quality problem", () => {
    const centroids = solveKMeans(points, 3, { seed: 3001 }).centroids;
    const result = evaluate({
      points: assignPoints(points, centroids),
      centroids,
      converged: false,
      elbowK: 3,
      bestInertiaAtK: solveKMeans(points, 3, { seed: 3001 }).inertia,
    });
    expect(result.outcome).toBe("not-converged");
    expect(result.failure).toBeNull();
  });

  it("puts empty clusters ahead of every other diagnosis", () => {
    const centroids = [
      ...solveKMeans(points, 3, { seed: 3001 }).centroids,
      { id: 3, x: 0.99, y: 0.01 },
      { id: 4, x: 0.01, y: 0.99 },
    ];
    const result = evaluate({
      points: assignPoints(points, centroids),
      centroids,
      converged: true,
      elbowK: 3,
      bestInertiaAtK: solveKMeans(points, 5, { seed: 3001 }).inertia,
    });
    // k is also wrong here, but an empty flag invalidates the other numbers.
    expect(result.emptyClusters.length).toBeGreaterThan(0);
    expect(result.outcome).toBe("empty-cluster");
  });

  it("only calls it a local minimum past the ratio threshold", () => {
    const best = solveKMeans(points, 3, { seed: 3001 }).inertia;
    const justInside = best * (LOCAL_MINIMUM_RATIO - 0.02);

    // Quality above 1/ratio must not be flagged.
    expect(best / justInside).toBeGreaterThan(1 / LOCAL_MINIMUM_RATIO);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Store: the live-feedback binding
// ═══════════════════════════════════════════════════════════════════════════

describe("store", () => {
  beforeEach(() => {
    useProgression.setState({
      ...EMPTY_PROGRESSION,
      hydrated: true,
      syncError: null,
      lastGain: null,
    });
    useProgression.getState().setAdapter(createMemoryAdapter());
    useKMeansStore.getState().startRound(1);
  });

  it("starts on round 1 with two flags, already assigned", () => {
    const state = useKMeansStore.getState();
    expect(state.round).toBe(1);
    expect(state.seed).toBe(ROUNDS[0]!.seed);
    expect(state.centroids).toHaveLength(2);
    expect(state.k).toBe(2);
    // Assigned on arrival so the metric means something immediately.
    expect(state.points.every((p) => p.clusterId !== null)).toBe(true);
    expect(state.inertia).toBeGreaterThan(0);
    expect(state.converged).toBe(false);
  });

  it("derives inertia from the real assignment, not a stored number", () => {
    const state = useKMeansStore.getState();
    expect(state.inertia).toBeCloseTo(
      inertiaOfAssignment(state.points, state.centroids),
      12,
    );
  });

  it("never reveals the true blob count in what the player can see", () => {
    const state = useKMeansStore.getState();
    // trueK exists for tests; nothing rendered derives from it.
    expect(state.trueK).toBe(ROUNDS[0]!.trueK);
    expect(state.elbowK).toBe(state.trueK);
  });

  it("moves the metric when a flag moves", () => {
    const before = useKMeansStore.getState().inertia;
    useKMeansStore.getState().moveFlag(0, 0.95, 0.05);
    expect(useKMeansStore.getState().inertia).not.toBe(before);
  });

  it("marks the assignment stale when a flag moves, and fresh after assign", () => {
    useKMeansStore.getState().moveFlag(0, 0.4, 0.4);
    expect(useKMeansStore.getState().assignmentStale).toBe(true);
    useKMeansStore.getState().assign();
    expect(useKMeansStore.getState().assignmentStale).toBe(false);
  });

  it("clamps flags to the map", () => {
    useKMeansStore.getState().moveFlag(0, 5, -5);
    const flag = useKMeansStore.getState().centroids[0]!;
    expect(flag.x).toBe(1);
    expect(flag.y).toBe(0);
  });

  it("adds and removes flags within bounds", () => {
    for (let i = 0; i < 20; i += 1) useKMeansStore.getState().addFlag();
    expect(useKMeansStore.getState().centroids).toHaveLength(MAX_K);

    for (let i = 0; i < 20; i += 1) useKMeansStore.getState().removeFlag(0);
    expect(useKMeansStore.getState().centroids).toHaveLength(1);
  });

  it("reassigns after removing a flag so no village points at a gone index", () => {
    useKMeansStore.getState().addFlag(0.2, 0.8);
    useKMeansStore.getState().assign();
    useKMeansStore.getState().removeFlag(0);

    const state = useKMeansStore.getState();
    for (const point of state.points) {
      expect(point.clusterId).not.toBeNull();
      expect(point.clusterId!).toBeLessThan(state.centroids.length);
    }
  });

  it("assign then update is the loop, and both change state", () => {
    useKMeansStore.getState().moveFlag(0, 0.1, 0.1);
    const stale = useKMeansStore.getState().inertia;

    useKMeansStore.getState().assign();
    const assigned = useKMeansStore.getState().inertia;
    // Reassignment can only reduce inertia for fixed flags.
    expect(assigned).toBeLessThanOrEqual(stale + 1e-9);

    const beforeUpdate = useKMeansStore.getState().centroids[0]!;
    useKMeansStore.getState().update();
    const afterUpdate = useKMeansStore.getState().centroids[0]!;
    expect(
      Math.hypot(afterUpdate.x - beforeUpdate.x, afterUpdate.y - beforeUpdate.y),
    ).toBeGreaterThan(0);
    expect(useKMeansStore.getState().iteration).toBe(1);
  });

  it("settles to convergence and reports it", () => {
    useKMeansStore.getState().settle();
    const state = useKMeansStore.getState();
    expect(state.converged).toBe(true);
    expect(state.iteration).toBeGreaterThan(0);
    expect(state.assignmentStale).toBe(false);
  });

  it("wins with the right k, settled, and awards XP once", () => {
    const target = useKMeansStore.getState().elbowK;
    while (useKMeansStore.getState().centroids.length < target) {
      useKMeansStore.getState().addFlag();
    }
    // Spread the flags so the fit doesn't start clumped in the middle.
    useKMeansStore.getState().centroids.forEach((_, index) => {
      const angle = (index / target) * Math.PI * 2;
      useKMeansStore
        .getState()
        .moveFlag(index, 0.5 + 0.3 * Math.cos(angle), 0.5 + 0.3 * Math.sin(angle));
    });
    useKMeansStore.getState().settle();

    const evaluation = useKMeansStore.getState().check();
    expect(evaluation.outcome).toBe("win");
    expect(useKMeansStore.getState().won).toBe(true);

    const xp = useProgression.getState().xp;
    expect(xp).toBeGreaterThan(0);
    useKMeansStore.getState().check();
    expect(useProgression.getState().xp).toBe(xp);
  });

  it("names Bad k when the player spams flags, and awards nothing", () => {
    const target = Math.min(MAX_K, useKMeansStore.getState().elbowK + 3);
    while (useKMeansStore.getState().centroids.length < target) {
      useKMeansStore.getState().addFlag();
    }
    useKMeansStore.getState().centroids.forEach((_, index) => {
      const angle = (index / target) * Math.PI * 2;
      useKMeansStore
        .getState()
        .moveFlag(index, 0.5 + 0.3 * Math.cos(angle), 0.5 + 0.3 * Math.sin(angle));
    });
    useKMeansStore.getState().settle();

    const evaluation = useKMeansStore.getState().check();
    expect(evaluation.outcome).toBe("bad-k");
    expect(useKMeansStore.getState().failure?.name).toBe("Bad k");
    expect(useKMeansStore.getState().won).toBe(false);
    expect(useProgression.getState().xp).toBe(0);
  });

  it("credits the code lane for the third star", () => {
    useKMeansStore.getState().setLane("code");
    const target = useKMeansStore.getState().elbowK;
    while (useKMeansStore.getState().centroids.length < target) {
      useKMeansStore.getState().addFlag();
    }
    useKMeansStore.getState().centroids.forEach((_, index) => {
      const angle = (index / target) * Math.PI * 2;
      useKMeansStore
        .getState()
        .moveFlag(index, 0.5 + 0.3 * Math.cos(angle), 0.5 + 0.3 * Math.sin(angle));
    });
    useKMeansStore.getState().settle();
    useKMeansStore.getState().check();

    expect(
      useProgression.getState().games["k-means-territory-wars"]?.codeLaneCleared,
    ).toBe(true);
    expect(
      useProgression.getState().games["k-means-territory-wars"]?.stars,
    ).toBe(3);
  });

  it("attaches a distinct why-card to every action", () => {
    const keys = new Set<string>();
    const record = () => {
      const card = useKMeansStore.getState().whyCard;
      expect(card).not.toBeNull();
      keys.add(card!.key);
    };

    record();
    useKMeansStore.getState().addFlag();
    record();
    useKMeansStore.getState().moveFlag(0, 0.25, 0.25);
    record();
    useKMeansStore.getState().assign();
    record();
    useKMeansStore.getState().update();
    record();
    useKMeansStore.getState().settle();
    record();
    useKMeansStore.getState().check();
    record();

    expect(keys.size).toBe(7);
  });

  it("resets to two flags but keeps the same villages", () => {
    const ids = useKMeansStore.getState().points.map((p) => p.id);
    useKMeansStore.getState().addFlag();
    useKMeansStore.getState().settle();
    useKMeansStore.getState().reset();

    const state = useKMeansStore.getState();
    expect(state.centroids).toHaveLength(2);
    expect(state.iteration).toBe(0);
    expect(state.converged).toBe(false);
    expect(state.points.map((p) => p.id)).toEqual(ids);
  });

  it("advances to the next curated map", () => {
    useKMeansStore.getState().newRound();
    const state = useKMeansStore.getState();
    expect(state.round).toBe(2);
    expect(state.seed).toBe(ROUNDS[1]!.seed);
    expect(state.elbowK).toBe(ROUNDS[1]!.trueK);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Why-cards
// ═══════════════════════════════════════════════════════════════════════════

describe("why-cards", () => {
  const { points } = generateVillages(3001, 3);
  const elbow = 3;

  const evalWith = (k: number, converged = true) => {
    const centroids = runToConvergence(
      points,
      solveKMeans(points, k, { seed: 3001 }).centroids,
    ).centroids;
    return evaluate({
      points: assignPoints(points, centroids),
      centroids,
      converged,
      elbowK: elbow,
      bestInertiaAtK: solveKMeans(points, k, { seed: 3001 }).inertia,
    });
  };

  it("explains Bad k by naming the trap directly", () => {
    const card = whyCardFor({ kind: "checked", evaluation: evalWith(6) });
    expect(card.title).toMatch(/bad k/i);
    expect(card.tone).toBe("bad");
    // The copy must say WHY the low number is untrustworthy.
    expect(card.body).toMatch(/always falls|can't tell you/i);
  });

  it("celebrates a settled map", () => {
    const card = whyCardFor({ kind: "checked", evaluation: evalWith(3) });
    expect(card.tone).toBe("good");
    expect(card.title).toMatch(/settled/i);
  });

  it("asks the player to keep stepping when not converged", () => {
    const card = whyCardFor({
      kind: "checked",
      evaluation: evalWith(3, false),
    });
    expect(card.title).toMatch(/not settled/i);
    expect(card.tone).toBe("warn");
  });

  it("distinguishes converged from still-moving on an update", () => {
    const moving = whyCardFor({
      kind: "updated",
      inertia: 4.2,
      shift: 0.08,
      converged: false,
    });
    const settled = whyCardFor({
      kind: "updated",
      inertia: 1.1,
      shift: 0,
      converged: true,
    });
    expect(moving.tone).toBe("info");
    expect(settled.tone).toBe("good");
    expect(settled.title).toMatch(/converged/i);
  });

  it("warns about empty territories on assign", () => {
    const card = whyCardFor({
      kind: "assigned",
      inertia: 3.3,
      sizes: [120, 0, 45],
    });
    expect(card.tone).toBe("warn");
    expect(card.body).toMatch(/no villages/i);
  });

  it("warns when adding flags past the elbow", () => {
    const card = whyCardFor({
      kind: "k-changed",
      k: 7,
      elbowK: 3,
      added: true,
    });
    expect(card.tone).toBe("warn");
    expect(card.body).toMatch(/zero|lowers inertia/i);
  });
});
