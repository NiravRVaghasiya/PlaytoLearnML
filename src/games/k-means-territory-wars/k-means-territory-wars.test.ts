import { beforeEach, describe, expect, it } from "vitest";
import { createElement } from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import {
  EMPTY_PROGRESSION,
  createMemoryAdapter,
  useProgression,
} from "@/engine/progression";
import {
  AT_BEST_QUALITY,
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
import { STARTER_CODE, createKMeansApi } from "./CodeLane";
import { VisualLane } from "./VisualLane";
import {
  GRAB_RADIUS_PX,
  MIN_GRAB_RADIUS_UNITS,
  clientToUser,
  flagUnderPointer,
  type ScreenMatrix,
} from "./field";

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

  it("only says too few flags are 'the best k can do' when they are", () => {
    // Round 3's default two flags settle 3% worse than the best two-flag
    // layout. The copy used to say "9.20 is the best 2 flags can do here" and
    // "not where you put them" regardless — true of the best layout, not of
    // this one.
    const { seed, trueK } = ROUNDS[2]!;
    const { points } = generateVillages(seed, trueK);
    const elbow = elbowKFor(elbowCurve(points, MAX_K, seed));
    const settled = runToConvergence(points, [
      { id: 0, x: 0.3, y: 0.3 },
      { id: 1, x: 0.7, y: 0.7 },
    ]).centroids;
    const best = solveKMeans(points, 2, { seed }).inertia;

    const result = evaluate({
      points: assignPoints(points, settled),
      centroids: settled,
      converged: true,
      elbowK: elbow,
      bestInertiaAtK: best,
      bestInertiaAtElbowK: solveKMeans(points, elbow, { seed }).inertia,
    });

    expect(result.outcome).toBe("bad-k");
    expect(result.convergenceQuality).toBeLessThan(AT_BEST_QUALITY);
    // It quotes the real best for two flags, and doesn't claim the player hit it.
    expect(result.failure!.detail).toContain(best.toFixed(2));
    expect(result.failure!.detail).not.toMatch(/is about the best/);
    const card = whyCardFor({ kind: "checked", evaluation: result });
    expect(card.body).toContain(best.toFixed(2));
    expect(card.body).not.toMatch(/really is/);

    // A layout that IS at the best keeps the stronger wording.
    const optimal = solveKMeans(points, 2, { seed }).centroids;
    const atBest = evaluate({
      points: assignPoints(points, optimal),
      centroids: optimal,
      converged: true,
      elbowK: elbow,
      bestInertiaAtK: best,
    });
    expect(atBest.convergenceQuality).toBeGreaterThanOrEqual(AT_BEST_QUALITY);
    expect(atBest.failure!.detail).toMatch(/is about the best 2 flags can do/);
  });

  it("judges nothing on a stale assignment", () => {
    // A flag dropped onto a blob owns 0 villages until Assign runs. Judged on
    // those memberships it used to be named "Empty cluster" — for a flag
    // sitting in the middle of 55 villages.
    const { seed, trueK } = ROUNDS[0]!;
    const { points } = generateVillages(seed, trueK);
    const two = runToConvergence(points, [
      { id: 0, x: 0.3, y: 0.3 },
      { id: 1, x: 0.7, y: 0.7 },
    ]).centroids;
    const stalePoints = assignPoints(points, two);
    const centre = blobCentre(points, 2);
    const three = [...two, { id: 2, x: centre.x, y: centre.y }];

    const input = {
      points: stalePoints,
      centroids: three,
      converged: false,
      elbowK: trueK,
      bestInertiaAtK: solveKMeans(points, 3, { seed }).inertia,
    };
    // The trap is real: without the stale flag, this reads as an empty cluster.
    expect(evaluate(input).outcome).toBe("empty-cluster");

    const result = evaluate({ ...input, assignmentStale: true });
    expect(result.outcome).toBe("not-converged");
    expect(result.failure).toBeNull();
    expect(result.emptyClusters).toEqual([]);
    const card = whyCardFor({ kind: "checked", evaluation: result });
    expect(card.title).toMatch(/out of date/i);
    expect(card.body).toMatch(/Assign/);
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

  it("does not call Update, Update convergence", () => {
    // The second Update re-averages the same stale memberships, so nothing
    // moves — yet 22 villages would switch flags on the next Assign. That used
    // to read "Nothing moved — converged", and Score then named a false
    // "Local minimum".
    const { addFlag, assign, update } = useKMeansStore.getState();
    addFlag();
    assign();
    update();
    expect(useKMeansStore.getState().assignmentStale).toBe(true);
    expect(useKMeansStore.getState().converged).toBe(false);

    update();
    const state = useKMeansStore.getState();
    expect(state.lastShift!).toBeLessThan(CONVERGENCE_EPSILON);
    expect(state.converged).toBe(false);
    expect(state.assignmentStale).toBe(true);
    expect(state.whyCard?.title).toMatch(/isn't convergence/);

    // The fresh assignment really does differ — this isn't a fixed point.
    const reassigned = assignPoints(state.points, state.centroids);
    const switched = reassigned.filter(
      (point, i) => point.clusterId !== state.points[i]!.clusterId,
    ).length;
    expect(switched).toBeGreaterThan(0);

    const evaluation = useKMeansStore.getState().check();
    expect(evaluation.outcome).toBe("not-converged");
    expect(evaluation.failure).toBeNull();
  });

  it("still converges on a genuine fixed point of assign then update", () => {
    const { settle, assign, update } = useKMeansStore.getState();
    settle();
    assign();
    update();
    const state = useKMeansStore.getState();
    expect(state.converged).toBe(true);
    expect(state.assignmentStale).toBe(false);
    expect(state.whyCard?.title).toMatch(/converged/i);
  });

  it("won't name an empty cluster for a flag placed on villages but not yet assigned", () => {
    const { settle, addFlag, moveFlag } = useKMeansStore.getState();
    settle();
    addFlag();
    const centre = blobCentre(useKMeansStore.getState().points, 2);
    moveFlag(2, centre.x, centre.y);

    const evaluation = useKMeansStore.getState().check();
    expect(evaluation.outcome).toBe("not-converged");
    expect(useKMeansStore.getState().failure).toBeNull();

    // And once villages are allowed to choose, it's the right answer.
    useKMeansStore.getState().assign();
    useKMeansStore.getState().settle();
    expect(useKMeansStore.getState().check().outcome).toBe("win");
  });

  it("clears the stale score whenever the flags change", () => {
    useKMeansStore.getState().settle();
    useKMeansStore.getState().check();
    expect(useKMeansStore.getState().lastEvaluation).not.toBeNull();

    useKMeansStore.getState().moveFlag(0, 0.2, 0.2);
    expect(useKMeansStore.getState().lastEvaluation).toBeNull();

    useKMeansStore.getState().check();
    useKMeansStore.getState().addFlag();
    expect(useKMeansStore.getState().lastEvaluation).toBeNull();

    useKMeansStore.getState().check();
    useKMeansStore.getState().removeFlag(0);
    expect(useKMeansStore.getState().lastEvaluation).toBeNull();
  });

  it("clears the score when the loop changes the board it judged", () => {
    const { addFlag, assign, settle, step, update, check } =
      useKMeansStore.getState();

    // Scored stale, then assigned: the tile must stop saying "assign first".
    addFlag();
    expect(check().assignmentStale).toBe(true);
    assign();
    expect(useKMeansStore.getState().lastEvaluation).toBeNull();

    // Scored mid-loop, then settled: the 9% verdict judged a board that is gone.
    expect(check().outcome).toBe("not-converged");
    settle();
    expect(useKMeansStore.getState().lastEvaluation).toBeNull();
    expect(useKMeansStore.getState().failure).toBeNull();

    // Scored after one step, then stepped again.
    useKMeansStore.getState().reset();
    addFlag();
    assign();
    update();
    assign();
    check();
    step();
    expect(useKMeansStore.getState().lastEvaluation).toBeNull();
  });

  it("keeps a win through loop presses that change nothing", () => {
    // Pressing Run to convergence again on a settled map must not take its
    // Next button away.
    const { addFlag, assign, settle, step, update, check } =
      useKMeansStore.getState();
    settle();
    addFlag();
    const centre = blobCentre(useKMeansStore.getState().points, 2);
    useKMeansStore.getState().moveFlag(2, centre.x, centre.y);
    assign();
    settle();
    expect(check().outcome).toBe("win");
    const verdict = useKMeansStore.getState().lastEvaluation;

    settle();
    assign();
    update();
    step();
    const state = useKMeansStore.getState();
    expect(state.won).toBe(true);
    expect(state.lastEvaluation).toBe(verdict);
  });

  it("keeps a named failure through loop presses that change nothing", () => {
    // Two flags on three blobs, settled: Bad k. Settling again changes
    // nothing, so the failure still describes the board on screen.
    const { settle, check } = useKMeansStore.getState();
    settle();
    expect(check().outcome).toBe("bad-k");
    settle();
    expect(useKMeansStore.getState().failure?.name).toBe("Bad k");
    expect(useKMeansStore.getState().lastEvaluation?.outcome).toBe("bad-k");
  });

  it("never stores a NaN flag, whatever it is handed", () => {
    const before = useKMeansStore.getState().centroids;
    const { moveFlag, nudgeFlag, addFlag, removeFlag } = useKMeansStore.getState();
    moveFlag(0, 0.5, Number.NaN);
    moveFlag(0, undefined as unknown as number, 0.5);
    moveFlag(0.5, 0.5, 0.5);
    nudgeFlag(0, Number.NaN, 0);
    addFlag(Number.NaN, 0.5);
    removeFlag(0.5);

    const state = useKMeansStore.getState();
    expect(state.centroids).toBe(before);
    expect(Number.isFinite(state.inertia)).toBe(true);
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
    // Nor counted twice.
    expect(
      useProgression.getState().games["k-means-territory-wars"]?.playCount,
    ).toBe(1);
    expect(useKMeansStore.getState().won).toBe(true);
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
    const api = createKMeansApi();
    useKMeansStore.getState().setLane("code");
    api.setK(useKMeansStore.getState().elbowK);
    api.scatterFlags();
    api.settle();
    api.check();

    expect(
      useProgression.getState().games["k-means-territory-wars"]?.codeLaneCleared,
    ).toBe(true);
    expect(
      useProgression.getState().games["k-means-territory-wars"]?.stars,
    ).toBe(3);
  });

  it("credits the lane the clearing call came from, not the visible tab", () => {
    // "Score this map" sits in the rail, on screen in the code tab too.
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
    expect(useKMeansStore.getState().check().outcome).toBe("win");

    const progress = useProgression.getState().games["k-means-territory-wars"];
    expect(progress?.completed).toBe(true);
    expect(progress?.codeLaneCleared).toBe(false);
    expect(progress?.stars).toBe(2);
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

// ═══════════════════════════════════════════════════════════════════════════
// Code lane: argument checks and the starter snippet
// ═══════════════════════════════════════════════════════════════════════════

describe("code lane api", () => {
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

  /** Run a snippet the way the JavaScript lane does. */
  const runSnippet = (code: string, api = createKMeansApi()) => {
    const logs: string[] = [];
    const fn = new Function("api", "log", "checkBudget", code) as (
      api: unknown,
      log: (...args: unknown[]) => void,
      checkBudget: () => void,
    ) => unknown;
    fn(api, (...args) => logs.push(args.join(" ")), () => {});
    return logs;
  };

  it("names a missing coordinate instead of storing a NaN flag", () => {
    // `api.setFlag(0, 0.5)` used to store y = NaN: the metric read "—", the
    // console logged an SVG attribute error, and the flag silently stopped
    // winning any village.
    const api = createKMeansApi();
    const before = useKMeansStore.getState().centroids;

    expect(() =>
      (api.setFlag as (index: number, x: number) => void)(0, 0.5),
    ).toThrow(/y must be a number from 0 to 1, got nothing/);
    expect(() => api.setFlag(0, Number.NaN, 0.5)).toThrow(TypeError);
    expect(() => api.setFlag(0, 1.2, 0.5)).toThrow(RangeError);
    expect(() => api.setFlag(5, 0.5, 0.5)).toThrow(/there are 2 flags/);
    expect(() => api.setFlag(-1, 0.5, 0.5)).toThrow(RangeError);

    expect(useKMeansStore.getState().centroids).toBe(before);
  });

  it("names a bad k", () => {
    const api = createKMeansApi();
    expect(() => api.setK(2.5)).toThrow(TypeError);
    expect(() => api.setK("3" as unknown as number)).toThrow(TypeError);
    expect(() => api.setK(Number.NaN)).toThrow(TypeError);
    expect(() => api.setK(0)).toThrow(/from 1 to 8/);
    expect(() => api.setK(9)).toThrow(RangeError);
    expect(useKMeansStore.getState().centroids).toHaveLength(2);
  });

  it("sets k exactly, both ways", () => {
    const api = createKMeansApi();
    api.setK(MAX_K);
    expect(api.k()).toBe(MAX_K);
    api.setK(1);
    expect(api.k()).toBe(1);
    api.setK(3);
    expect(api.k()).toBe(3);
  });

  it("credits the code lane for a board first scored from the rail", () => {
    const api = createKMeansApi();
    api.setK(useKMeansStore.getState().elbowK);
    api.scatterFlags();
    api.settle();
    // Scored from the rail first: two stars.
    useKMeansStore.getState().check();
    expect(
      useProgression.getState().games["k-means-territory-wars"]?.stars,
    ).toBe(2);

    // Then `api.check()` on the same board, from the code lane: ★3, once.
    api.check();
    api.check();
    useKMeansStore.getState().check();
    const progress = useProgression.getState().games["k-means-territory-wars"];
    expect(progress?.codeLaneCleared).toBe(true);
    expect(progress?.stars).toBe(3);
    expect(progress?.playCount).toBe(2);
  });

  it("clears round 1 with the starter snippet, and a second run counts nothing twice", () => {
    const first = runSnippet(STARTER_CODE);
    expect(first.join("\n")).toMatch(/verdict\s+win/);
    const progress = useProgression.getState().games["k-means-territory-wars"];
    expect(progress?.codeLaneCleared).toBe(true);
    expect(progress?.playCount).toBe(1);
    const xp = useProgression.getState().xp;

    // The snippet re-scatters the flags and converges to the same board —
    // which is the same clear, however many times it is reached.
    const second = runSnippet(STARTER_CODE);
    expect(second.join("\n")).toMatch(/verdict\s+win/);
    expect(useKMeansStore.getState().won).toBe(true);
    runSnippet(STARTER_CODE);
    expect(
      useProgression.getState().games["k-means-territory-wars"]?.playCount,
    ).toBe(1);
    expect(useProgression.getState().xp).toBe(xp);

    // Nor does reaching that clustering again by hand, from the rail.
    const { reset, addFlag, moveFlag, settle, check } = useKMeansStore.getState();
    reset();
    addFlag();
    useKMeansStore.getState().centroids.forEach((_, index) => {
      const angle = (index / 3) * Math.PI * 2 + 1;
      moveFlag(index, 0.5 + 0.3 * Math.cos(angle), 0.5 + 0.3 * Math.sin(angle));
    });
    settle();
    expect(check().outcome).toBe("win");
    expect(
      useProgression.getState().games["k-means-territory-wars"]?.playCount,
    ).toBe(1);
  });

  it("counts a different winning map as a new clear", () => {
    runSnippet(STARTER_CODE);
    useKMeansStore.getState().newRound();
    runSnippet(STARTER_CODE.replace("api.setK(3)", "api.setK(4)"));
    const progress = useProgression.getState().games["k-means-territory-wars"];
    expect(useKMeansStore.getState().round).toBe(2);
    expect(useKMeansStore.getState().won).toBe(true);
    expect(progress?.playCount).toBe(2);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Pointer geometry
// ═══════════════════════════════════════════════════════════════════════════

/**
 * The matrix a browser paints a `viewBox="0 0 100 100"` SVG with, for a box of
 * `width`×`height` CSS px at (left, top), under the default "xMidYMid meet".
 */
function meetMatrix(left: number, top: number, width: number, height: number) {
  const scale = Math.min(width, height) / 100;
  const e = left + (width - 100 * scale) / 2;
  const f = top + (height - 100 * scale) / 2;
  const matrix: ScreenMatrix = {
    a: scale,
    b: 0,
    c: 0,
    d: scale,
    e,
    f,
    inverse: () => ({ a: 1 / scale, b: 0, c: 0, d: 1 / scale, e: -e / scale, f: -f / scale }),
  };
  return {
    matrix,
    toClient: (x: number, y: number) => ({
      clientX: e + x * scale,
      clientY: f + y * scale,
    }),
  };
}

describe("pointer geometry", () => {
  it("maps a pointer through the painted transform, letterboxing included", () => {
    // 1440×900: the map renders 878 wide and 1329 tall. By bounding box, a
    // drag to y = 90 landed at y = 76.
    const { matrix, toClient } = meetMatrix(24, 90, 878, 1329);
    for (const [x, y] of [
      [5, 10],
      [50, 50],
      [95, 90],
    ] as const) {
      const { clientX, clientY } = toClient(x, y);
      const view = clientToUser(clientX, clientY, matrix);
      expect(view!.x).toBeCloseTo(x, 9);
      expect(view!.y).toBeCloseTo(y, 9);
    }
  });

  it("grabs the nearest flag, not the one painted last", () => {
    const flags = [
      { x: 50, y: 50 },
      { x: 53, y: 50 },
    ];
    expect(flagUnderPointer({ x: 50.4, y: 50 }, flags, 6)).toBe(0);
    expect(flagUnderPointer({ x: 52.6, y: 50 }, flags, 6)).toBe(1);
    // Exact tie: the lower index, as with villages.
    expect(flagUnderPointer({ x: 51.5, y: 50 }, flags, 6)).toBe(0);
    expect(flagUnderPointer({ x: 70, y: 50 }, flags, 6)).toBeNull();
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Visual lane: pointer wiring and honest flag labels
// ═══════════════════════════════════════════════════════════════════════════

describe("visual lane", () => {
  beforeEach(() => {
    useKMeansStore.getState().startRound(1);
  });

  function mountMap(box = { left: 24, top: 90, width: 878, height: 1329 }) {
    render(createElement(VisualLane));
    const svg = screen.getByRole("group", { name: "Village map" });
    const { matrix, toClient } = meetMatrix(box.left, box.top, box.width, box.height);
    Object.assign(svg, {
      getScreenCTM: () => matrix,
      setPointerCapture: () => {},
      hasPointerCapture: () => false,
      releasePointerCapture: () => {},
    });
    // Map (0–1) → viewBox, matching the lane's d3 scales (PAD = 5).
    const toView = (x: number, y: number) => ({ x: 5 + x * 90, y: 95 - y * 90 });
    const press = (type: "pointerDown" | "pointerMove" | "pointerUp", x: number, y: number) => {
      const v = toView(x, y);
      const { clientX, clientY } = toClient(v.x, v.y);
      act(() => {
        fireEvent[type](svg, { clientX, clientY, isPrimary: true, button: 0, pointerId: 1 });
      });
    };
    return { press };
  }

  it("drags a flag to exactly where the pointer is on a letterboxed map", () => {
    const { press } = mountMap();
    // Flag 1 starts at (0.3, 0.3).
    press("pointerDown", 0.3, 0.3);
    press("pointerMove", 0.5, 0.9);
    press("pointerUp", 0.5, 0.9);
    const flag = useKMeansStore.getState().centroids[0]!;
    expect(flag.x).toBeCloseTo(0.5, 6);
    expect(flag.y).toBeCloseTo(0.9, 6);
    expect(useKMeansStore.getState().selectedFlag).toBe(0);
  });

  it("grabs a flag from 22 CSS px away on a phone-sized map, and no further", () => {
    // 324 CSS px across 100 units: 3.24 px per unit, so 22 px is 6.8 units —
    // past the 6-unit floor, which alone came to a 39 px target here.
    const scale = 324 / 100;
    const inside = 6.5;
    const outside = 7.1;
    expect(inside).toBeGreaterThan(MIN_GRAB_RADIUS_UNITS);
    expect(inside * scale).toBeLessThanOrEqual(GRAB_RADIUS_PX);
    expect(outside * scale).toBeGreaterThan(GRAB_RADIUS_PX);

    const { press } = mountMap({ left: 18, top: 300, width: 324, height: 324 });
    // Flag 1 starts at (0.3, 0.3); map units are 90 viewBox units wide.
    const before = useKMeansStore.getState().centroids;
    press("pointerDown", 0.3 + outside / 90, 0.3);
    press("pointerMove", 0.5, 0.5);
    press("pointerUp", 0.5, 0.5);
    expect(useKMeansStore.getState().centroids).toBe(before);

    press("pointerDown", 0.3 + inside / 90, 0.3);
    press("pointerMove", 0.5, 0.5);
    press("pointerUp", 0.5, 0.5);
    const flag = useKMeansStore.getState().centroids[0]!;
    expect(flag.x).toBeCloseTo(0.5, 6);
    expect(flag.y).toBeCloseTo(0.5, 6);
  });

  it("ignores a press on empty ground", () => {
    const { press } = mountMap();
    const before = useKMeansStore.getState().centroids;
    press("pointerDown", 0.9, 0.1);
    press("pointerMove", 0.8, 0.2);
    press("pointerUp", 0.8, 0.2);
    expect(useKMeansStore.getState().centroids).toBe(before);
  });

  it("doesn't label a just-placed flag empty before villages have chosen", () => {
    render(createElement(VisualLane));
    act(() => {
      useKMeansStore.getState().addFlag(0.5, 0.5);
    });
    const flag = screen.getByRole("button", { name: /^Flag 3 of 3/ });
    expect(flag.getAttribute("aria-label")).not.toMatch(/empty/);
    expect(flag.getAttribute("aria-label")).toMatch(/as of the last assign/);
  });
});

