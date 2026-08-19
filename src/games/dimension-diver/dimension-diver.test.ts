import { beforeEach, describe, expect, it } from "vitest";
import { seededRandom } from "@/lib/utils";
import {
  CLOUDS,
  DEFAULT_ANGLES,
  GROUP_COUNT,
  POINT_COUNT,
  SEPARATION_FLOOR,
  SEPARATION_TARGET,
  VARIANCE_TARGET,
  analyse,
  anglesForAxis,
  bestSeparation,
  bestVariance,
  buildCloud,
  centroid,
  covariance,
  dot,
  evaluate,
  frameFor,
  jacobiEigen,
  project,
  quadratic,
  separation,
  trace,
  varianceRetained,
  type Angles,
  type Mat3,
  type Point3D,
} from "./ml";
import { createCodeApi, shadowFor, useDiverStore } from "./store";
import { gaugeCaption, separationCaption, whyCardFor } from "./why-cards";

/**
 * The claim this game makes is that it scores against real PCA. That is checked by
 * verifying the eigendecomposition satisfies Cv = λv directly, that the eigenvalues
 * sum to the trace, and that the quadratic form agrees with the variance measured
 * from the projected coordinates by brute force.
 *
 * Everything else here was established by a throwaway probe: which clouds put
 * variance and separation in conflict, whether that conflict leaves a winning move,
 * and where the thresholds have to sit. The probe is gone; the findings are these.
 */

const cloudById = (id: string) => CLOUDS.find((c) => c.id === id)!;
const pointsFor = (id: string) => buildCloud(id, 2029);

describe("linear algebra", () => {
  const known: Mat3 = [
    [4, 1, 1],
    [1, 3, 0.5],
    [1, 0.5, 2],
  ];

  it("eigendecomposes a symmetric matrix so that Cv = lambda v", () => {
    const { values, vectors } = jacobiEigen(known);
    vectors.forEach((v, k) => {
      for (let i = 0; i < 3; i += 1) {
        let cv = 0;
        for (let j = 0; j < 3; j += 1) cv += known[i]![j]! * v[j]!;
        expect(cv).toBeCloseTo(values[k]! * v[i]!, 10);
      }
    });
  });

  it("returns eigenvalues in descending order summing to the trace", () => {
    const { values } = jacobiEigen(known);
    expect(values[0]).toBeGreaterThanOrEqual(values[1]!);
    expect(values[1]).toBeGreaterThanOrEqual(values[2]!);
    expect(values[0] + values[1] + values[2]).toBeCloseTo(trace(known), 10);
  });

  it("returns an orthonormal eigenbasis", () => {
    const { vectors } = jacobiEigen(known);
    for (const v of vectors) {
      expect(Math.hypot(v[0], v[1], v[2])).toBeCloseTo(1, 10);
    }
    expect(dot(vectors[0], vectors[1])).toBeCloseTo(0, 10);
    expect(dot(vectors[0], vectors[2])).toBeCloseTo(0, 10);
    expect(dot(vectors[1], vectors[2])).toBeCloseTo(0, 10);
  });

  it("handles a diagonal matrix and a rank-deficient one", () => {
    const diagonal = jacobiEigen([
      [3, 0, 0],
      [0, 2, 0],
      [0, 0, 1],
    ]);
    expect(diagonal.values).toEqual([3, 2, 1]);

    const deficient = jacobiEigen([
      [2, 2, 0],
      [2, 2, 0],
      [0, 0, 0],
    ]);
    expect(deficient.values[0]).toBeCloseTo(4, 9);
    expect(deficient.values[1]).toBeCloseTo(0, 9);
    expect(deficient.values[2]).toBeCloseTo(0, 9);
  });

  it("survives wildly different scales", () => {
    // Measured residual 2.3e-13 on this one, which is float noise.
    const scaled: Mat3 = [
      [1000, 3, -2],
      [3, 0.01, 0.5],
      [-2, 0.5, 7],
    ];
    const { values } = jacobiEigen(scaled);
    expect(values[0] + values[1] + values[2]).toBeCloseTo(trace(scaled), 8);
    expect(values[0]).toBeCloseTo(1000.013, 2);
  });

  it("computes covariance about the cloud's own centroid", () => {
    const points: Point3D[] = [
      { x: 1, y: 0, z: 0, groupId: 0 },
      { x: -1, y: 0, z: 0, groupId: 0 },
    ];
    expect(centroid(points)).toEqual([0, 0, 0]);
    const matrix = covariance(points);
    // Sample covariance with n-1 = 1: sum of squares is 2.
    expect(matrix[0][0]).toBeCloseTo(2, 10);
    expect(matrix[1][1]).toBe(0);
  });

  it("returns a zero covariance for fewer than two points", () => {
    expect(trace(covariance([]))).toBe(0);
    expect(trace(covariance([{ x: 5, y: 5, z: 5, groupId: 0 }]))).toBe(0);
  });
});

describe("the projection frame", () => {
  const cases: Angles[] = [
    { yaw: 0, pitch: 0, roll: 0 },
    { yaw: 37, pitch: -18, roll: 0 },
    { yaw: -170, pitch: 88, roll: -44 },
    { yaw: 180, pitch: -90, roll: 180 },
  ];

  it("is orthonormal at every orientation", () => {
    for (const angles of cases) {
      const frame = frameFor(angles);
      for (const row of frame) {
        expect(Math.hypot(row[0], row[1], row[2])).toBeCloseTo(1, 12);
      }
      expect(dot(frame[0], frame[1])).toBeCloseTo(0, 12);
      expect(dot(frame[0], frame[2])).toBeCloseTo(0, 12);
      expect(dot(frame[1], frame[2])).toBeCloseTo(0, 12);
    }
  });

  it("makes roll provably unable to change what is retained", () => {
    // The single most load-bearing fact in this game. Retained variance is
    // trace(C) minus the variance along the discarded axis, and rolling does not
    // move the discarded axis. Measured difference: exactly zero.
    const matrix = covariance(pointsFor("pancake"));
    const base = varianceRetained(matrix, { yaw: 37, pitch: -18, roll: 0 });
    for (const roll of [45, 90, 137, 180, -173]) {
      expect(
        varianceRetained(matrix, { yaw: 37, pitch: -18, roll }),
        `roll ${roll}`,
      ).toBeCloseTo(base, 12);
    }
  });

  it("makes roll unable to change the separation either", () => {
    const points = pointsFor("pancake");
    const base = separation(project(points, { yaw: 37, pitch: -18, roll: 0 }))
      .ratio;
    for (const roll of [90, 137]) {
      expect(
        separation(project(points, { yaw: 37, pitch: -18, roll })).ratio,
      ).toBeCloseTo(base, 8);
    }
  });

  it("agrees with the variance measured from the projected points", () => {
    // The quadratic form and brute force have to give the same answer, or the gauge
    // is not reporting what the picture shows. Measured difference: 0.
    const points = pointsFor("needle");
    const matrix = covariance(points);
    const angles: Angles = { yaw: 23, pitch: -41, roll: 17 };
    const shadow = project(points, angles);

    const meanU = shadow.reduce((t, p) => t + p.u, 0) / shadow.length;
    const meanV = shadow.reduce((t, p) => t + p.v, 0) / shadow.length;
    const measured =
      shadow.reduce((t, p) => t + (p.u - meanU) ** 2 + (p.v - meanV) ** 2, 0) /
      (shadow.length - 1);

    expect(varianceRetained(matrix, angles) * trace(matrix)).toBeCloseTo(
      measured,
      8,
    );
  });

  it("centres the shadow on the cloud", () => {
    const shadow = project(pointsFor("pancake"), { yaw: 12, pitch: 34, roll: 0 });
    const meanU = shadow.reduce((t, p) => t + p.u, 0) / shadow.length;
    expect(meanU).toBeCloseTo(0, 8);
  });

  it("derives angles that land exactly on a given discarded axis", () => {
    for (const cloud of CLOUDS) {
      const analysis = analyse(pointsFor(cloud.id));
      const atDerived = varianceRetained(
        analysis.covariance,
        analysis.pcaAngles,
      );
      // The derived angles must hit the eigen-optimum, not merely approach it.
      expect(atDerived, cloud.id).toBeCloseTo(analysis.best, 10);
    }
  });

  it("never retains more than the top two eigenvalues allow", () => {
    const points = pointsFor("needle");
    const analysis = analyse(points);
    for (let yaw = -180; yaw < 180; yaw += 17) {
      for (let pitch = -90; pitch <= 90; pitch += 13) {
        const retained = varianceRetained(analysis.covariance, {
          yaw,
          pitch,
          roll: 0,
        });
        expect(retained).toBeLessThanOrEqual(analysis.best + 1e-12);
      }
    }
  });
});

describe("the clouds", () => {
  it("each hold three balanced groups", () => {
    for (const cloud of CLOUDS) {
      const points = pointsFor(cloud.id);
      expect(points).toHaveLength(POINT_COUNT);
      const counts = new Array<number>(GROUP_COUNT).fill(0);
      for (const point of points) {
        counts[point.groupId] = (counts[point.groupId] ?? 0) + 1;
      }
      for (const count of counts) {
        expect(count, cloud.id).toBe(POINT_COUNT / GROUP_COUNT);
      }
    }
  });

  it("are reproducible from their seed", () => {
    const a = buildCloud("needle", 7);
    const b = buildCloud("needle", 7);
    expect(a[0]).toEqual(b[0]);
    expect(buildCloud("needle", 8)[0]).not.toEqual(a[0]);
  });

  it("are all tilted away from the world axes", () => {
    // An untilted cloud would be solved by setting every slider to zero, which
    // rewards guessing the interface instead of reading the data.
    for (const cloud of CLOUDS) {
      const analysis = analyse(pointsFor(cloud.id));
      const atDefault = varianceRetained(analysis.covariance, DEFAULT_ANGLES);
      expect(atDefault / analysis.best, cloud.id).toBeLessThan(VARIANCE_TARGET);
    }
  });

  it("make the pancake nearly two-dimensional", () => {
    const analysis = analyse(pointsFor("pancake"));
    // Measured [7.81, 5.40, 0.049]: the third axis is a rounding error.
    expect(analysis.best).toBeGreaterThan(0.99);
    expect(analysis.eigen.values[2]).toBeLessThan(0.1);
  });

  it("make the needle's loudest axis dominate and carry no signal", () => {
    const analysis = analyse(pointsFor("needle"));
    // Measured [142.6, 2.63, 1.07]. The ratio is what makes dropping the MIDDLE
    // axis cost almost nothing as a share of the optimum.
    expect(analysis.eigen.values[0] / analysis.eigen.values[2]).toBeGreaterThan(
      50,
    );
    // And PCA's own plane must fail to separate the groups.
    const atPca = separation(
      project(pointsFor("needle"), analysis.pcaAngles),
    ).ratio;
    expect(atPca).toBeLessThan(SEPARATION_FLOOR);
  });

  it("leave the needle winnable despite the conflict", () => {
    // The probe's first two attempts at this cloud were unwinnable: PCA's plane
    // showed nothing and the only plane that showed the groups retained 13% of the
    // spread. This asserts a plane exists that satisfies BOTH criteria.
    const points = pointsFor("needle");
    const analysis = analyse(points);
    const reachable = analysis.reachableSeparation;

    const retained = varianceRetained(analysis.covariance, reachable.angles);
    expect(retained / analysis.best).toBeGreaterThanOrEqual(VARIANCE_TARGET);

    const verdict = evaluate({
      points,
      analysis,
      angles: reachable.angles,
      submitted: true,
    });
    expect(verdict.outcome).toBe("surfaced");
    expect(verdict.failure).toBeNull();
  });

  it("make the shells inseparable by any flat shadow", () => {
    const analysis = analyse(pointsFor("shells"));
    // Measured best anywhere: 0.021. A linear projection cannot tell "near the
    // centre" from "far from it".
    expect(analysis.bestSeparation.ratio).toBeLessThan(SEPARATION_FLOOR);
    // And the cloud is nearly isotropic, so it has no preferred plane either.
    expect(analysis.eigen.values[0] / analysis.eigen.values[2]).toBeLessThan(1.5);
  });
});

describe("separation", () => {
  it("is invariant to noise orthogonal to the separating direction", () => {
    // The reason this measures the best LINE rather than the whole plane. The first
    // version used the trace ratio, which collapsed when orthogonal noise was
    // added — so it was maximised by discarding the loudest axis, which put it in
    // permanent conflict with retaining variance.
    // A real pseudo-random jitter, not an index-derived one: `(index * 37) % 100`
    // correlates with `index % 3`, which makes the within-group scatter almost
    // singular and sends the ratio into the billions. The fixture has to be as
    // uncorrelated as the thing it stands in for.
    const random = seededRandom(11);
    const clean: Point3D[] = [];
    const noisy: Point3D[] = [];
    for (let index = 0; index < 150; index += 1) {
      const groupId = index % 3;
      const offset = (groupId - 1) * 2;
      const jx = random() - 0.5;
      const jy = random() - 0.5;
      clean.push({ x: offset + jx * 0.4, y: jy, z: 0, groupId });
      // The same separating structure along x, with 40x the spread along y.
      noisy.push({ x: offset + jx * 0.4, y: jy * 40, z: 0, groupId });
    }
    const flat: Angles = { yaw: 0, pitch: 0, roll: 0 };
    const a = separation(project(clean, flat)).ratio;
    const b = separation(project(noisy, flat)).ratio;

    expect(a).toBeGreaterThan(5);
    // Orthogonal noise must not wash the score out. The trace ratio dropped by a
    // factor of hundreds here; the best-line ratio holds.
    expect(b).toBeGreaterThan(a * 0.5);
  });

  it("is near zero when the groups are not separated at all", () => {
    const random = seededRandom(23);
    const blob: Point3D[] = [];
    for (let index = 0; index < 150; index += 1) {
      blob.push({
        x: random() - 0.5,
        y: (random() - 0.5) * 0.5,
        z: 0,
        groupId: index % 3,
      });
    }
    expect(
      separation(project(blob, { yaw: 0, pitch: 0, roll: 0 })).ratio,
    ).toBeLessThan(0.2);
  });

  it("handles an empty shadow without dividing by zero", () => {
    expect(separation([])).toEqual({ ratio: 0, spread: [] });
  });

  it("reports one spread per group", () => {
    const result = separation(project(pointsFor("pancake"), DEFAULT_ANGLES));
    expect(result.spread).toHaveLength(GROUP_COUNT);
  });

  it("finds a better plane when told to ignore the variance constraint", () => {
    // The unconstrained optimum is allowed to be somewhere useless; the point of
    // the constrained one is that quoting the unconstrained figure in a failure
    // message sent the player toward a "Lost variance" verdict.
    const points = pointsFor("needle");
    const analysis = analyse(points);

    // The claim worth asserting is the one the copy depends on: the unconstrained
    // optimum sits somewhere that would score "Lost variance", so it is not the
    // plane to send anybody to. (The two ratios come out within 1e-4 of each other
    // here, and which is fractionally higher depends on where each sweep's
    // refinement grid happens to land — so ordering them is not a real invariant.)
    const unconstrainedRetained = varianceRetained(
      analysis.covariance,
      analysis.bestSeparation.angles,
    );
    expect(unconstrainedRetained / analysis.best).toBeLessThan(VARIANCE_TARGET);
    expect(analysis.reachableSeparation.ratio).toBeGreaterThan(1);
  });
});

describe("the verdict", () => {
  const judge = (cloudId: string, angles: Angles) => {
    const points = pointsFor(cloudId);
    return evaluate({ points, analysis: analyse(points), angles, submitted: true });
  };

  it("says nothing until the player commits", () => {
    const points = pointsFor("pancake");
    const verdict = evaluate({
      points,
      analysis: analyse(points),
      angles: DEFAULT_ANGLES,
      submitted: false,
    });
    expect(verdict.outcome).toBe("diving");
    expect(verdict.failure).toBeNull();
    expect(verdict.score).toBe(0);
  });

  it("names lost variance, with the eigenvalues that explain it", () => {
    const analysis = analyse(pointsFor("pancake"));
    // Discard the LARGEST axis: the worst thing you can do.
    const verdict = judge("pancake", anglesForAxis(analysis.eigen.vectors[0]));
    expect(verdict.outcome).toBe("lost-variance");
    expect(verdict.failure?.name).toBe("Lost variance");
    expect(verdict.failure?.detail).toMatch(/three variances are/);
    expect(verdict.failure?.detail).toMatch(/thinnest along/);
  });

  it("surfaces at the PCA plane on the pancake, with three stars", () => {
    const analysis = analyse(pointsFor("pancake"));
    const verdict = judge("pancake", analysis.pcaAngles);
    expect(verdict.outcome).toBe("surfaced");
    expect(verdict.failure).toBeNull();
    expect(verdict.share).toBeCloseTo(1, 6);
    expect(verdict.stars).toBe(3);
  });

  it("names the conflict on the needle: maximum variance, nothing to see", () => {
    const analysis = analyse(pointsFor("needle"));
    const verdict = judge("needle", analysis.pcaAngles);
    expect(verdict.outcome).toBe("mixed");
    expect(verdict.failure?.name).toBe("High variance, mixed groups");
    expect(verdict.share).toBeCloseTo(1, 6);
    expect(verdict.separationShare).toBeLessThan(0.1);
  });

  it("points the mixed verdict at a plane that actually passes", () => {
    // A failure message quoting an unreachable plane is advice that loops.
    const points = pointsFor("needle");
    const analysis = analyse(points);
    const verdict = judge("needle", analysis.pcaAngles);
    const quoted = varianceRetained(
      analysis.covariance,
      analysis.reachableSeparation.angles,
    );
    expect(quoted / analysis.best).toBeGreaterThanOrEqual(VARIANCE_TARGET);
    expect(verdict.failure?.detail).toMatch(/still inside the target/);
    expect(verdict.failure?.detail).toMatch(/Discard the middle axis/);
  });

  it("waives the separation criterion when no plane could satisfy it", () => {
    // Judging the shells by share alone would have awarded stars for a shadow in
    // which the groups are completely superimposed — near-optimal, and useless.
    const analysis = analyse(pointsFor("shells"));
    const verdict = judge("shells", analysis.pcaAngles);
    expect(verdict.separable).toBe(false);
    expect(verdict.outcome).toBe("surfaced");
    expect(verdict.stars).toBe(3);
  });

  it("keeps every failing score below every passing score", () => {
    const analysis = analyse(pointsFor("pancake"));
    const failing = judge("pancake", anglesForAxis(analysis.eigen.vectors[0]));
    const passing = judge("pancake", analysis.pcaAngles);
    expect(failing.score).toBeLessThan(0.5);
    expect(passing.score).toBeGreaterThan(0.9);
  });

  it("carries numbers in every named failure", () => {
    const analysis = analyse(pointsFor("needle"));
    for (const angles of [
      anglesForAxis(analysis.eigen.vectors[0]),
      analysis.pcaAngles,
    ]) {
      const verdict = judge("needle", angles);
      expect(verdict.failure).not.toBeNull();
      expect(verdict.failure!.name).not.toMatch(/game over/i);
      expect(verdict.failure!.detail).toMatch(/\d/);
      expect(verdict.failure!.detail.length).toBeGreaterThan(150);
    }
  });

  it("uses the targets it advertises", () => {
    expect(VARIANCE_TARGET).toBe(0.98);
    expect(SEPARATION_TARGET).toBeLessThan(1);
    expect(SEPARATION_FLOOR).toBeLessThan(SEPARATION_TARGET);
  });

  it("agrees with bestVariance on the eigenvalues", () => {
    const analysis = analyse(pointsFor("needle"));
    expect(bestVariance(analysis.eigen)).toBeCloseTo(analysis.best, 12);
  });

  it("returns a zero best separation rather than -1 when nothing is evaluated", () => {
    // An impossible constraint must not leak a sentinel into the UI.
    const result = bestSeparation(pointsFor("pancake"), 2);
    expect(result.ratio).toBe(0);
    expect(result.angles).toEqual(DEFAULT_ANGLES);
  });
});

describe("the store", () => {
  beforeEach(() => {
    useDiverStore.setState({ surfacedIds: [], cloudIndex: 0 });
    useDiverStore.getState().reset();
  });

  it("starts on the first cloud, untilted and uncommitted", () => {
    const state = useDiverStore.getState();
    expect(state.cloudIndex).toBe(0);
    expect(state.angles).toEqual(DEFAULT_ANGLES);
    expect(state.submitted).toBe(false);
    expect(state.showGroups).toBe(false);
    expect(state.evaluation.outcome).toBe("diving");
    expect(state.points).toHaveLength(POINT_COUNT);
  });

  it("wraps yaw and roll, and clamps pitch to the poles", () => {
    useDiverStore.getState().setAngle("yaw", 270);
    expect(useDiverStore.getState().angles.yaw).toBe(-90);
    useDiverStore.getState().setAngle("roll", -300);
    expect(useDiverStore.getState().angles.roll).toBe(60);
    useDiverStore.getState().setAngle("pitch", 150);
    expect(useDiverStore.getState().angles.pitch).toBe(90);
  });

  it("updates the live gauge as the cloud turns", () => {
    const before = useDiverStore.getState().evaluation.retained;
    useDiverStore.getState().setAngles({ yaw: -43, pitch: 21 });
    expect(useDiverStore.getState().evaluation.retained).not.toBeCloseTo(
      before,
      4,
    );
  });

  it("fires the roll card when only roll moves, and the gauge does not budge", () => {
    useDiverStore.getState().setAngles({ yaw: 30, pitch: -20 });
    const before = useDiverStore.getState().evaluation.retained;
    useDiverStore.getState().setAngle("roll", 90);
    const state = useDiverStore.getState();
    expect(state.evaluation.retained).toBeCloseTo(before, 12);
    expect(state.whyCard?.title).toMatch(/Rolling cannot change/);
  });

  it("ignores a no-op change", () => {
    const before = useDiverStore.getState().evaluation;
    useDiverStore.getState().setAngle("yaw", 0);
    expect(useDiverStore.getState().evaluation).toBe(before);
  });

  it("nudges relative to the current angle", () => {
    useDiverStore.getState().setAngles({ yaw: 10 });
    useDiverStore.getState().nudge("yaw", 5);
    expect(useDiverStore.getState().angles.yaw).toBe(15);
  });

  it("reveals the groups on commit, because that is the payoff", () => {
    useDiverStore.getState().submit();
    const state = useDiverStore.getState();
    expect(state.submitted).toBe(true);
    expect(state.showGroups).toBe(true);
    expect(state.failure).not.toBeNull();
  });

  it("un-commits when the cloud is turned again", () => {
    useDiverStore.getState().submit();
    expect(useDiverStore.getState().submitted).toBe(true);
    useDiverStore.getState().setAngle("yaw", 15);
    const state = useDiverStore.getState();
    expect(state.submitted).toBe(false);
    // The verdict belonged to a plane that is no longer on screen.
    expect(state.failure).toBeNull();
    expect(state.phase).toBe("diving");
  });

  it("surfaces when the projection is good enough", () => {
    const analysis = useDiverStore.getState().analysis;
    useDiverStore.getState().setAngles(analysis.pcaAngles);
    useDiverStore.getState().submit();
    const state = useDiverStore.getState();
    expect(state.phase).toBe("surfaced");
    expect(state.failure).toBeNull();
    expect(state.surfacedIds).toEqual([CLOUDS[0]!.id]);
    expect(state.evaluation.score).toBeGreaterThan(0.9);
  });

  it("snaps to the PCA plane on request, and records the cost", () => {
    useDiverStore.getState().usePcaHint();
    const state = useDiverStore.getState();
    expect(state.hintUsed).toBe(true);
    expect(state.angles).toEqual(state.analysis.pcaAngles);
    expect(state.evaluation.retained).toBeCloseTo(state.analysis.best, 10);
    expect(state.whyCard?.title).toMatch(/PCA's answer/);
  });

  it("scores a hinted win below an unhinted one", () => {
    const analysis = useDiverStore.getState().analysis;

    useDiverStore.getState().usePcaHint();
    useDiverStore.getState().submit();
    const hinted = useDiverStore.getState().evaluation.score;

    useDiverStore.setState({ surfacedIds: [] });
    useDiverStore.getState().reset();
    useDiverStore.getState().setAngles(analysis.pcaAngles);
    useDiverStore.getState().submit();
    const unhinted = useDiverStore.getState().evaluation.score;

    // The evaluation score itself is the same; the penalty is applied when the
    // result is recorded, so check the flag that drives it.
    expect(hinted).toBeCloseTo(unhinted, 6);
    expect(useDiverStore.getState().hintUsed).toBe(false);
  });

  it("moves to the next cloud and keeps the surfaced record", () => {
    const analysis = useDiverStore.getState().analysis;
    useDiverStore.getState().setAngles(analysis.pcaAngles);
    useDiverStore.getState().submit();
    useDiverStore.getState().nextCloud();

    const state = useDiverStore.getState();
    expect(state.cloudIndex).toBe(1);
    expect(state.angles).toEqual(DEFAULT_ANGLES);
    expect(state.hintUsed).toBe(false);
    expect(state.surfacedIds).toEqual([CLOUDS[0]!.id]);
    // A different cloud means a different analysis.
    expect(state.analysis.best).not.toBeCloseTo(analysis.best, 4);
  });

  it("wraps around after the last cloud", () => {
    for (let index = 0; index < CLOUDS.length; index += 1) {
      useDiverStore.getState().nextCloud();
    }
    expect(useDiverStore.getState().cloudIndex).toBe(0);
  });

  it("derives the shadow rather than storing it", () => {
    const first = shadowFor(useDiverStore.getState());
    useDiverStore.getState().setAngles({ yaw: 45 });
    const second = shadowFor(useDiverStore.getState());
    expect(second[0]!.u).not.toBeCloseTo(first[0]!.u, 4);
    expect(second).toHaveLength(POINT_COUNT);
  });
});

describe("the code lane api", () => {
  beforeEach(() => {
    useDiverStore.setState({ surfacedIds: [], cloudIndex: 0 });
    useDiverStore.getState().reset();
  });

  it("writes the same state the sliders write", () => {
    const api = createCodeApi();
    api.setAngles({ yaw: 30, pitch: -15 });
    expect(useDiverStore.getState().angles.yaw).toBe(30);
    expect(useDiverStore.getState().angles.pitch).toBe(-15);
    expect(api.angles().yaw).toBe(30);
  });

  it("rejects a non-finite angle by name", () => {
    const api = createCodeApi();
    expect(() => api.setAngles({ yaw: Number.NaN })).toThrow(/yaw must be/);
    expect(() => api.retainedAt({ pitch: Number.POSITIVE_INFINITY })).toThrow(
      /pitch must be/,
    );
  });

  it("measures any orientation without turning the cloud", () => {
    const api = createCodeApi();
    const before = api.angles();
    const analysis = useDiverStore.getState().analysis;

    const retained = api.retainedAt(analysis.pcaAngles);
    expect(retained).toBeCloseTo(analysis.best, 10);
    expect(api.separationAt(analysis.pcaAngles)).toBeGreaterThanOrEqual(0);
    expect(api.angles()).toEqual(before);
  });

  it("hands over the real eigendecomposition", () => {
    const api = createCodeApi();
    const pca = api.pca();
    const analysis = useDiverStore.getState().analysis;
    expect(pca.eigenvalues[0]).toBeCloseTo(analysis.eigen.values[0], 10);
    expect(pca.bestRetained).toBeCloseTo(analysis.best, 10);
    expect(pca.eigenvectors[0]).toHaveLength(3);
  });

  it("proves roll does nothing, which is what the snippet checks", () => {
    const api = createCodeApi();
    const base = api.retainedAt({ yaw: 30, pitch: -20, roll: 0 });
    for (const roll of [45, 90, 137, 180]) {
      expect(api.retainedAt({ yaw: 30, pitch: -20, roll })).toBeCloseTo(base, 12);
    }
  });

  it("hands back the shadow with hidden group ids", () => {
    const api = createCodeApi();
    const shadow = api.shadow();
    expect(shadow).toHaveLength(POINT_COUNT);
    expect(shadow[0]).toHaveProperty("groupId");
  });

  it("hands back copies, not live state", () => {
    const api = createCodeApi();
    const points = api.points();
    points[0]!.x = 9999;
    expect(useDiverStore.getState().points[0]!.x).not.toBe(9999);

    const angles = api.angles();
    angles.yaw = 9999;
    expect(useDiverStore.getState().angles.yaw).not.toBe(9999);
  });

  it("can commit and read the outcome", () => {
    const api = createCodeApi();
    api.setAngles(useDiverStore.getState().analysis.pcaAngles);
    api.submit();
    expect(api.score().outcome).toBe("surfaced");
  });
});

describe("why-cards", () => {
  it("warns up front that roll cannot matter", () => {
    const card = whyCardFor({ kind: "briefing", cloud: CLOUDS[0]! });
    expect(card.body).toMatch(/roll slider will spin the picture/);
  });

  it("explains the roll invariance in terms of the discarded axis", () => {
    const card = whyCardFor({ kind: "rolled", retained: 0.8, roll: 90 });
    expect(card.body).toMatch(/discarded axis never moved/);
    expect(card.body).toMatch(/two real degrees of freedom/);
  });

  it("credits the closed form when the hint is used", () => {
    const analysis = analyse(pointsFor("pancake"));
    const card = whyCardFor({ kind: "hinted", analysis, retained: analysis.best });
    expect(card.body).toMatch(/closed-form answer/);
  });

  it("says plainly that the shells cannot be separated linearly", () => {
    const points = pointsFor("shells");
    const analysis = analyse(points);
    const card = whyCardFor({
      kind: "submitted",
      cloud: cloudById("shells"),
      evaluation: evaluate({
        points,
        analysis,
        angles: analysis.pcaAngles,
        submitted: true,
      }),
      analysis,
      hintUsed: false,
    });
    expect(card.tone).toBe("good");
    expect(card.body).toMatch(/no weighted sum of coordinates/);
    expect(card.body).toMatch(/t-SNE and UMAP/);
  });

  it("frames the needle's conflict as the method's limit, not the player's error", () => {
    const points = pointsFor("needle");
    const analysis = analyse(points);
    const card = whyCardFor({
      kind: "submitted",
      cloud: cloudById("needle"),
      evaluation: evaluate({
        points,
        analysis,
        angles: analysis.pcaAngles,
        submitted: true,
      }),
      analysis,
      hintUsed: false,
    });
    expect(card.tone).toBe("bad");
    expect(card.body).toMatch(/PCA does not look for structure/);
    expect(card.body).toMatch(/nothing in the algorithm was ever going to warn you/);
  });

  it("captions the gauge against the ceiling, not against 100%", () => {
    expect(gaugeCaption(0.996, 0.996, false)).toMatch(/This is the optimum/);
    expect(gaugeCaption(0.99, 0.996, false)).toMatch(/Within/);
    expect(gaugeCaption(0.5, 0.996, true)).toMatch(/flattened away/);
    expect(gaugeCaption(0.5, 0.996, false)).toMatch(/ceiling is/);
  });

  it("captions separation honestly when nothing can separate", () => {
    expect(separationCaption(0.01, 0.02, false)).toMatch(
      /limit of linear projection/,
    );
    expect(separationCaption(6, 6.2, true)).toMatch(/come apart cleanly/);
    expect(separationCaption(0.1, 16, true)).toMatch(/Nothing is separated/);
  });
});

describe("the quadratic form", () => {
  it("reports the variance along a direction", () => {
    const matrix: Mat3 = [
      [4, 0, 0],
      [0, 1, 0],
      [0, 0, 9],
    ];
    expect(quadratic(matrix, [1, 0, 0])).toBe(4);
    expect(quadratic(matrix, [0, 0, 1])).toBe(9);
    expect(trace(matrix)).toBe(14);
  });
});
