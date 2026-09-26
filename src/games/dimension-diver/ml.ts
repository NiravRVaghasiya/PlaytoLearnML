import type { NamedFailure } from "@/engine/types";
import { clamp, gaussian, seededRandom } from "@/lib/utils";

/**
 * Dimension Diver — finding a projection by eye, and being scored against PCA.
 *
 * A 3D cloud with hidden groups floats in space. The player turns it until the 2D
 * shadow it casts keeps as much of the spread as possible, and a real
 * eigendecomposition of the covariance matrix says how close they got.
 *
 * ── The one piece of geometry the whole game rests on ────────────────────────
 * Variance retained depends ONLY on the axis you throw away.
 *
 * Projecting onto the plane spanned by two orthonormal vectors keeps
 * v₁ᵀCv₁ + v₂ᵀCv₂ of the total spread, and because the three axes are orthonormal
 * that is exactly trace(C) − v₃ᵀCv₃. So the score is a function of the discarded
 * direction and nothing else. Two consequences worth knowing, and the game says
 * both out loud:
 *
 *   - Rolling the cloud — spinning the shadow in its own plane — cannot change the
 *     score. The picture rotates and the number does not move. That surprises
 *     people, and it is the clearest possible demonstration that a projection is a
 *     choice of PLANE rather than a choice of picture.
 *   - The player really has two degrees of freedom, not three: where to point the
 *     axis that gets discarded.
 *
 * ── Why there is a separation score as well ──────────────────────────────────
 * PCA maximises variance. Variance is not what anybody actually wants — they want
 * to see the structure — and those two things come apart. One of the three clouds
 * here is built so that the highest-variance directions carry no group information
 * at all, and the plane that shows the groups retains measurably LESS variance than
 * PCA's answer. A player who trusts the gauge will maximise it and see nothing.
 * That is the most important thing this game has to say, so it is not a footnote:
 * it is a scenario, with numbers.
 *
 * ── Deviations from the spec, flagged ───────────────────────────────────────
 * 1. Rotation is driven by sliders, not by dragging. Dragging a 3D cloud is
 *    mouse-only, and this project's accessibility floor does not allow the primary
 *    control of a game to be unreachable by keyboard — the same call as the
 *    read-only graphs in Neuron Forge and Decision Tree Architect.
 * 2. `Projection.basisVectors[2]` in the spec's data model is derived here rather
 *    than stored: the state is three angles, and the basis is computed from them.
 *    Storing both invites them to disagree, and the angles are what the controls
 *    actually write.
 * 3. No t-SNE or UMAP implementation. The spec asks for "PCA / t-SNE / UMAP
 *    intuition", and the intuition those two exist for is delivered by the third
 *    cloud, where no linear projection separates the groups and the game says why
 *    that is a limit of the method rather than of the player. Shipping a t-SNE the
 *    player could not relate to the rotation they are holding would teach less.
 */

// ── The cloud ─────────────────────────────────────────────────────────────

export interface Point3D {
  x: number;
  y: number;
  z: number;
  groupId: number;
}

export const GROUP_COUNT = 3;
export const GROUP_LABELS = ["Alpha", "Beta", "Gamma"] as const;
export const POINT_COUNT = 300;

export interface CloudSpec {
  id: string;
  title: string;
  /** What this arrangement is here to teach. */
  lesson: string;
  build: (random: () => number) => Point3D[];
}

/** Rotate a point by yaw, pitch, roll, in that order. Used to tilt the clouds. */
function tilt(point: Point3D, yaw: number, pitch: number): Point3D {
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const x1 = cy * point.x + sy * point.z;
  const z1 = -sy * point.x + cy * point.z;
  const y2 = cp * point.y - sp * z1;
  const z2 = sp * point.y + cp * z1;
  return { x: x1, y: y2, z: z2, groupId: point.groupId };
}

/**
 * Three clouds, each making a different point.
 *
 * All three are tilted away from the world axes on purpose. An untilted cloud would
 * be solvable by setting the sliders to zero, which would reward guessing the
 * interface rather than reading the data.
 */
export const CLOUDS: readonly CloudSpec[] = [
  {
    id: "pancake",
    title: "The pancake",
    lesson:
      "Three groups laid out on a plane, and the plane is tilted in space. Almost all the spread lives in two directions — find them and you lose next to nothing.",
    build: (random) => {
      const points: Point3D[] = [];
      // Group centres arranged in the plane the data actually occupies.
      const centres = [
        { u: -3.2, v: -1.6 },
        { u: 3.0, v: -1.2 },
        { u: 0.2, v: 3.1 },
      ];
      for (let index = 0; index < POINT_COUNT; index += 1) {
        const groupId = index % GROUP_COUNT;
        const centre = centres[groupId]!;
        points.push(
          tilt(
            {
              x: centre.u + gaussian(random) * 1.0,
              y: centre.v + gaussian(random) * 1.0,
              // A pancake, not a plane: some thickness, but little of it.
              z: gaussian(random) * 0.22,
              groupId,
            },
            0.7,
            -0.5,
          ),
        );
      }
      return points;
    },
  },
  {
    id: "needle",
    title: "The loud axis",
    // The lesson used to say the two quiet axes have "almost exactly the same
    // spread" and that "the gauge cannot tell those two apart" — left over from
    // the abandoned attempt two below. Measured, they are 2.63 and 1.07, and the
    // gauge reads 99.3% against 98.2% for discarding one or the other. Both
    // numbers this copy now relies on are asserted in the tests.
    lesson:
      "One direction is enormously loud and pure noise. Of the two quiet ones, the smaller holds the groups — and because the loud axis dominates the total, discarding the other quiet axis instead costs only about a point on the gauge. The gauge measures spread, not groups, so it cannot tell you which quiet axis holds them. You will have to look.",
    build: (random) => {
      const points: Point3D[] = [];
      // Offsets along the third axis are what tells the groups apart.
      //
      // The three variances here are the entire scenario, and getting them right
      // took two rounds of measurement.
      //
      // Attempt one made the signal axis the smallest by a wide margin. That gave a
      // perfect conflict and an unwinnable level: PCA's plane showed nothing, and
      // the only plane that showed the groups retained 13% of the spread.
      //
      // Attempt two set the two quiet axes nearly equal, which made it winnable and
      // destroyed the lesson — when two eigenvalues are close, their eigenvectors
      // are barely determined and mix, so PCA's plane picked up most of the signal
      // anyway and separated the groups fine.
      //
      // What works is a very loud first axis. Retained variance is 1 − λᵢ/Σλ, so
      // when λ₁ dominates the sum, dropping the middle axis instead of the smallest
      // costs almost nothing as a SHARE of the optimum — measured at 98.9% — while
      // λ₂ still sits three times λ₃, keeping the eigenvectors well separated and
      // the signal firmly in the axis PCA discards. So the plane that reveals the
      // groups is not PCA's answer, and it is still comfortably inside the variance
      // target. Both lessons, no contradiction.
      const offsets = [-1.2, 0, 1.2];
      for (let index = 0; index < POINT_COUNT; index += 1) {
        const groupId = index % GROUP_COUNT;
        points.push(
          tilt(
            {
              x: gaussian(random) * 12, // very loud, and pure noise
              y: gaussian(random) * 1.73, // quiet noise, ~3x the signal's variance
              z: offsets[groupId]! + gaussian(random) * 0.25, // quietest: the signal
              groupId,
            },
            -0.6,
            0.45,
          ),
        );
      }
      return points;
    },
  },
  {
    id: "shells",
    title: "Nested shells",
    lesson:
      "One group sits inside the other as a spherical shell. No flat shadow of this can pull them apart, however you turn it — and that is a fact about linear projection, not about you.",
    build: (random) => {
      const points: Point3D[] = [];
      const radii = [1.4, 3.2, 5.0];
      for (let index = 0; index < POINT_COUNT; index += 1) {
        const groupId = index % GROUP_COUNT;
        // A direction sampled uniformly on the sphere, then scaled to the shell.
        const a = gaussian(random);
        const b = gaussian(random);
        const c = gaussian(random);
        const length = Math.sqrt(a * a + b * b + c * c) || 1;
        const radius = radii[groupId]! + gaussian(random) * 0.22;
        points.push({
          x: (a / length) * radius,
          y: (b / length) * radius,
          z: (c / length) * radius,
          groupId,
        });
      }
      return points;
    },
  },
] as const;

export function buildCloud(cloudId: string, seed = 2029): Point3D[] {
  const spec = CLOUDS.find((candidate) => candidate.id === cloudId) ?? CLOUDS[0]!;
  return spec.build(seededRandom(seed));
}

// ── Linear algebra, only as much as is needed ────────────────────────────

export type Vec3 = [number, number, number];
export type Mat3 = [Vec3, Vec3, Vec3];

export const dot = (a: Vec3, b: Vec3): number =>
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export function centroid(points: readonly Point3D[]): Vec3 {
  if (points.length === 0) return [0, 0, 0];
  let x = 0;
  let y = 0;
  let z = 0;
  for (const point of points) {
    x += point.x;
    y += point.y;
    z += point.z;
  }
  return [x / points.length, y / points.length, z / points.length];
}

/** Covariance of the cloud, about its own centroid. */
export function covariance(points: readonly Point3D[]): Mat3 {
  const [mx, my, mz] = centroid(points);
  const matrix: Mat3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  if (points.length < 2) return matrix;

  // Accumulated flat, then folded into the tuple type: with
  // `noUncheckedIndexedAccess` a `matrix[i]![j] +=` reads as possibly undefined on
  // the left-hand side, and silencing that with more `!` would be noise.
  const sums = new Array<number>(9).fill(0);
  for (const point of points) {
    const d: Vec3 = [point.x - mx, point.y - my, point.z - mz];
    for (let i = 0; i < 3; i += 1) {
      for (let j = 0; j < 3; j += 1) {
        sums[i * 3 + j] = (sums[i * 3 + j] ?? 0) + d[i]! * d[j]!;
      }
    }
  }

  const n = points.length - 1;
  return [
    [sums[0]! / n, sums[1]! / n, sums[2]! / n],
    [sums[3]! / n, sums[4]! / n, sums[5]! / n],
    [sums[6]! / n, sums[7]! / n, sums[8]! / n],
  ];
}

export const trace = (matrix: Mat3): number =>
  matrix[0][0] + matrix[1][1] + matrix[2][2];

/** vᵀ M v — the variance of the cloud along direction v. */
export function quadratic(matrix: Mat3, v: Vec3): number {
  let total = 0;
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 3; j += 1) {
      total += v[i]! * matrix[i]![j]! * v[j]!;
    }
  }
  return total;
}

export interface Eigen {
  /** Descending. */
  values: [number, number, number];
  /** Unit eigenvectors, in the same order as `values`. */
  vectors: [Vec3, Vec3, Vec3];
}

/**
 * Eigendecomposition of a symmetric 3x3 matrix, by Jacobi rotations.
 *
 * This is the real thing rather than a closed-form shortcut: repeatedly zero the
 * largest off-diagonal entry with an orthogonal rotation, and the matrix converges
 * to its eigenvalues while the accumulated rotations become the eigenvectors. It is
 * about thirty lines and it is what makes "scored against real PCA" a true
 * statement — the tests check the result satisfies Cv = λv directly, and check the
 * eigenvalues sum to the trace.
 */
export function jacobiEigen(input: Mat3): Eigen {
  // Work on copies; Jacobi is destructive.
  const a: number[][] = input.map((row) => [...row]);
  let v: number[][] = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];

  for (let sweep = 0; sweep < 100; sweep += 1) {
    // Largest off-diagonal magnitude.
    let p = 0;
    let q = 1;
    let largest = Math.abs(a[0]![1]!);
    const candidates: Array<[number, number]> = [
      [0, 1],
      [0, 2],
      [1, 2],
    ];
    for (const [i, j] of candidates) {
      if (Math.abs(a[i]![j]!) > largest) {
        largest = Math.abs(a[i]![j]!);
        p = i;
        q = j;
      }
    }
    if (largest < 1e-14) break;

    const apq = a[p]![q]!;
    const app = a[p]![p]!;
    const aqq = a[q]![q]!;
    const theta = 0.5 * Math.atan2(2 * apq, aqq - app);
    const c = Math.cos(theta);
    const s = Math.sin(theta);

    // Rotate A by the Givens rotation in the (p,q) plane, both sides.
    const rotate = (m: number[][]) => {
      const next = m.map((row) => [...row]);
      for (let k = 0; k < 3; k += 1) {
        next[p]![k] = c * m[p]![k]! - s * m[q]![k]!;
        next[q]![k] = s * m[p]![k]! + c * m[q]![k]!;
      }
      const out = next.map((row) => [...row]);
      for (let k = 0; k < 3; k += 1) {
        out[k]![p] = c * next[k]![p]! - s * next[k]![q]!;
        out[k]![q] = s * next[k]![p]! + c * next[k]![q]!;
      }
      return out;
    };

    const rotated = rotate(a);
    for (let i = 0; i < 3; i += 1) {
      for (let j = 0; j < 3; j += 1) a[i]![j] = rotated[i]![j]!;
    }

    // Accumulate the same rotation into the eigenvector basis.
    const nextV = v.map((row) => [...row]);
    for (let k = 0; k < 3; k += 1) {
      nextV[k]![p] = c * v[k]![p]! - s * v[k]![q]!;
      nextV[k]![q] = s * v[k]![p]! + c * v[k]![q]!;
    }
    v = nextV;
  }

  const pairs: Array<{ value: number; vector: Vec3 }> = [0, 1, 2].map((k) => ({
    value: a[k]![k]!,
    vector: [v[0]![k]!, v[1]![k]!, v[2]![k]!] as Vec3,
  }));
  pairs.sort((left, right) => right.value - left.value);

  return {
    values: [pairs[0]!.value, pairs[1]!.value, pairs[2]!.value],
    vectors: [pairs[0]!.vector, pairs[1]!.vector, pairs[2]!.vector],
  };
}

// ── The player's projection ──────────────────────────────────────────────

export interface Angles {
  /** Degrees. */
  yaw: number;
  pitch: number;
  roll: number;
}

export const DEFAULT_ANGLES: Angles = { yaw: 0, pitch: 0, roll: 0 };

const radians = (degrees: number) => (degrees * Math.PI) / 180;

/**
 * The orthonormal frame the sliders describe.
 *
 * Rows 0 and 1 span the plane that gets kept — the shadow's horizontal and
 * vertical — and row 2 is the direction thrown away. Which is the one that decides
 * the score.
 */
export function frameFor({ yaw, pitch, roll }: Angles): Mat3 {
  const cy = Math.cos(radians(yaw));
  const sy = Math.sin(radians(yaw));
  const cp = Math.cos(radians(pitch));
  const sp = Math.sin(radians(pitch));
  const cr = Math.cos(radians(roll));
  const sr = Math.sin(radians(roll));

  // Rz(roll) · Rx(pitch) · Ry(yaw), written out.
  const ry: Mat3 = [
    [cy, 0, sy],
    [0, 1, 0],
    [-sy, 0, cy],
  ];
  const rx: Mat3 = [
    [1, 0, 0],
    [0, cp, -sp],
    [0, sp, cp],
  ];
  const rz: Mat3 = [
    [cr, -sr, 0],
    [sr, cr, 0],
    [0, 0, 1],
  ];

  const multiply = (a: Mat3, b: Mat3): Mat3 => {
    const out: Mat3 = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    for (let i = 0; i < 3; i += 1) {
      for (let j = 0; j < 3; j += 1) {
        let total = 0;
        for (let k = 0; k < 3; k += 1) total += a[i]![k]! * b[k]![j]!;
        out[i]![j] = total;
      }
    }
    return out;
  };

  return multiply(rz, multiply(rx, ry));
}

export interface Shadow {
  u: number;
  v: number;
  groupId: number;
  /** Distance along the discarded axis. Kept for the depth cue, not for scoring. */
  depth: number;
}

/** The 2D shadow of the cloud under a rotation. */
export function project(points: readonly Point3D[], angles: Angles): Shadow[] {
  const frame = frameFor(angles);
  const [mx, my, mz] = centroid(points);
  return points.map((point) => {
    const d: Vec3 = [point.x - mx, point.y - my, point.z - mz];
    return {
      u: dot(frame[0], d),
      v: dot(frame[1], d),
      depth: dot(frame[2], d),
      groupId: point.groupId,
    };
  });
}

/**
 * Fraction of the cloud's total variance that survives the projection.
 *
 * Computed from the discarded axis, because that is all it depends on — see the
 * header. Doing it this way rather than summing the two kept directions is not an
 * optimisation, it is the honest form of the expression, and it is why the roll
 * slider provably cannot move this number.
 */
export function varianceRetained(matrix: Mat3, angles: Angles): number {
  const total = trace(matrix);
  if (total <= 1e-12) return 1;
  const discarded = frameFor(angles)[2];
  return clamp(1 - quadratic(matrix, discarded) / total, 0, 1);
}

/** The best any projection can retain: the top two eigenvalues. */
export function bestVariance(eigen: Eigen): number {
  const total = eigen.values[0] + eigen.values[1] + eigen.values[2];
  if (total <= 1e-12) return 1;
  return (eigen.values[0] + eigen.values[1]) / total;
}

/**
 * Angles that land on the PCA plane.
 *
 * Derived by asking which yaw and pitch point the DISCARDED axis along the
 * smallest eigenvector — the third row of the frame, from the header's argument.
 * Roll is left at zero because it cannot matter.
 */
export function anglesForAxis(axis: Vec3): Angles {
  // Frame row 2 of Rx(pitch)·Ry(yaw) is [-sy·cp, sp, cy·cp] — invert that.
  const [x, y, z] = axis;
  const length = Math.sqrt(x * x + y * y + z * z) || 1;
  const nx = x / length;
  const ny = y / length;
  const nz = z / length;
  const pitch = Math.asin(clamp(ny, -1, 1));
  const yaw = Math.atan2(-nx, nz);
  return {
    yaw: (yaw * 180) / Math.PI,
    pitch: (pitch * 180) / Math.PI,
    roll: 0,
  };
}

// ── Separation: the thing variance is a proxy for ────────────────────────

export interface Separation {
  /** Between-group scatter over within-group scatter, in the shadow. */
  ratio: number;
  /** Per-group centroid distance from the overall centre, for the readout. */
  spread: number[];
}

/**
 * How well the shadow pulls the groups apart.
 *
 * The best Fisher ratio achievable by any LINE drawn in the shadow: the largest
 * eigenvalue of Sw⁻¹Sb, where Sb and Sw are the 2x2 between- and within-group
 * scatter matrices of the projected points. Solved in closed form, because the
 * whole thing is 2x2 and this gets called a few thousand times during the search.
 *
 * ── Why the best line and not the whole plane ───────────────────────────────
 * The first version used the plain trace ratio over both projected axes, and it was
 * measured to be actively wrong for this game. Adding a large amount of noise
 * ORTHOGONAL to the separating direction inflates the within-group scatter and
 * craters the trace ratio, even though a human looking at the picture would still
 * see three obvious bands. So the trace ratio was maximised by discarding the
 * loudest axis, which put it in permanent conflict with retaining variance and left
 * the middle cloud with no winning move at all.
 *
 * Asking whether any line in the shadow separates the groups is both the question a
 * person actually answers when they look at a scatter plot, and invariant to spread
 * along directions that do not matter. It is also supervised on purpose: the groups
 * are known to the game and hidden from the player, so "did your projection reveal
 * the structure that is there" is the honest question, and a label-free score like
 * silhouette would answer a different one.
 */
export function separation(shadow: readonly Shadow[]): Separation {
  if (shadow.length === 0) return { ratio: 0, spread: [] };

  const groups: Shadow[][] = Array.from({ length: GROUP_COUNT }, () => []);
  for (const point of shadow) {
    groups[point.groupId]?.push(point);
  }

  const meanU = shadow.reduce((total, p) => total + p.u, 0) / shadow.length;
  const meanV = shadow.reduce((total, p) => total + p.v, 0) / shadow.length;

  // 2x2 scatter matrices, [ [a, b], [b, c] ].
  let bA = 0;
  let bB = 0;
  let bC = 0;
  let wA = 0;
  let wB = 0;
  let wC = 0;
  const spread: number[] = [];

  for (const group of groups) {
    if (group.length === 0) {
      spread.push(0);
      continue;
    }
    const gu = group.reduce((total, p) => total + p.u, 0) / group.length;
    const gv = group.reduce((total, p) => total + p.v, 0) / group.length;
    const weight = group.length / shadow.length;
    const du = gu - meanU;
    const dv = gv - meanV;
    bA += weight * du * du;
    bB += weight * du * dv;
    bC += weight * dv * dv;

    for (const point of group) {
      const pu = point.u - gu;
      const pv = point.v - gv;
      wA += (pu * pu) / shadow.length;
      wB += (pu * pv) / shadow.length;
      wC += (pv * pv) / shadow.length;
    }
    spread.push(Math.hypot(du, dv));
  }

  return { ratio: fisherRatio(bA, bB, bC, wA, wB, wC), spread };
}

/**
 * The largest eigenvalue of Sw⁻¹Sb for 2x2 scatter matrices [[a, b], [b, c]].
 * Shared by `separation` and `separationProbe`, so the readout and the search
 * can never disagree about what "separation" means.
 */
function fisherRatio(
  bA: number,
  bB: number,
  bC: number,
  wA: number,
  wB: number,
  wC: number,
): number {
  // A ridge keeps Sw invertible when a projection collapses the cloud to a line.
  const ridge = 1e-9 * (wA + wC + 1);
  const sA = wA + ridge;
  const sC = wC + ridge;
  const det = sA * sC - wB * wB;
  if (Math.abs(det) < 1e-15) return 0;

  // M = Sw^-1 Sb, then its largest eigenvalue.
  const m11 = (sC * bA - wB * bB) / det;
  const m12 = (sC * bB - wB * bC) / det;
  const m21 = (sA * bB - wB * bA) / det;
  const m22 = (sA * bC - wB * bB) / det;

  const tr = m11 + m22;
  const dt = m11 * m22 - m12 * m21;
  const disc = Math.max(tr * tr - 4 * dt, 0);
  const ratio = (tr + Math.sqrt(disc)) / 2;

  return Number.isFinite(ratio) ? Math.max(ratio, 0) : 0;
}

/**
 * `separation(project(points, angles)).ratio`, without building the shadow.
 *
 * The plane search evaluates about eleven hundred orientations per sweep, and
 * `analyse` runs two sweeps. Through `project` + `separation` each evaluation
 * recomputed the centroid and allocated three hundred shadow objects and three
 * group arrays, which measured 50–100 ms per cloud on a desktop — and ran at
 * module load, before first paint. This centres the cloud once, then makes a
 * single allocation-free pass per orientation, accumulating each group's sums
 * of u, v, u², uv and v² and turning those into the same scatter matrices in
 * closed form (Σ(u − ū)² = Σu² − nū²). The tests hold it to the slow path to
 * nine digits on every cloud.
 */
export function separationProbe(
  points: readonly Point3D[],
): (angles: Angles) => number {
  const [mx, my, mz] = centroid(points);
  const n = points.length;
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  const zs = new Float64Array(n);
  const groupOf = new Int32Array(n);
  points.forEach((point, index) => {
    xs[index] = point.x - mx;
    ys[index] = point.y - my;
    zs[index] = point.z - mz;
    groupOf[index] =
      point.groupId >= 0 && point.groupId < GROUP_COUNT ? point.groupId : -1;
  });

  const count = new Float64Array(GROUP_COUNT);
  const su = new Float64Array(GROUP_COUNT);
  const sv = new Float64Array(GROUP_COUNT);
  const suu = new Float64Array(GROUP_COUNT);
  const suv = new Float64Array(GROUP_COUNT);
  const svv = new Float64Array(GROUP_COUNT);

  return (angles) => {
    if (n === 0) return 0;
    const frame = frameFor(angles);
    const [a0, a1, a2] = frame[0];
    const [b0, b1, b2] = frame[1];
    count.fill(0);
    su.fill(0);
    sv.fill(0);
    suu.fill(0);
    suv.fill(0);
    svv.fill(0);

    let totalU = 0;
    let totalV = 0;
    for (let index = 0; index < n; index += 1) {
      const x = xs[index]!;
      const y = ys[index]!;
      const z = zs[index]!;
      const u = a0 * x + a1 * y + a2 * z;
      const v = b0 * x + b1 * y + b2 * z;
      totalU += u;
      totalV += v;
      const group = groupOf[index]!;
      if (group < 0) continue;
      // Written out rather than `+=`: with `noUncheckedIndexedAccess` a typed
      // array element reads as possibly undefined on the left-hand side.
      count[group] = count[group]! + 1;
      su[group] = su[group]! + u;
      sv[group] = sv[group]! + v;
      suu[group] = suu[group]! + u * u;
      suv[group] = suv[group]! + u * v;
      svv[group] = svv[group]! + v * v;
    }

    const meanU = totalU / n;
    const meanV = totalV / n;
    let bA = 0;
    let bB = 0;
    let bC = 0;
    let wA = 0;
    let wB = 0;
    let wC = 0;
    for (let group = 0; group < GROUP_COUNT; group += 1) {
      const c = count[group]!;
      if (c === 0) continue;
      const gu = su[group]! / c;
      const gv = sv[group]! / c;
      const weight = c / n;
      const du = gu - meanU;
      const dv = gv - meanV;
      bA += weight * du * du;
      bB += weight * du * dv;
      bC += weight * dv * dv;
      wA += (suu[group]! - c * gu * gu) / n;
      wB += (suv[group]! - c * gu * gv) / n;
      wC += (svv[group]! - c * gv * gv) / n;
    }
    return fisherRatio(bA, bB, bC, wA, wB, wC);
  };
}

/**
 * The best separation any projection of this cloud achieves.
 *
 * Found by search rather than in closed form. Fisher's criterion does have an
 * analytic optimum for a single discriminant direction, but the quantity here is
 * over a PLANE and the ranking is not a plain eigenproblem — and a coarse-to-fine
 * sweep over the sphere of discarded axes is both exact enough for a target and
 * something the player can be told the shape of: a 6° grid over the hemisphere,
 * then 1° around the best cell — about eleven hundred evaluations.
 *
 * That was described here as "milliseconds", and through `project` + `separation`
 * it measured 45 ms per sweep on a desktop, two sweeps per cloud, at module load.
 * The sweep now goes through `separationProbe`, which gives the same numbers
 * without allocating, and the store caches each cloud's analysis so a retry
 * costs nothing at all.
 */
export function bestSeparation(
  points: readonly Point3D[],
  /**
   * When given, only planes retaining at least this fraction of the total variance
   * are considered.
   *
   * This exists because of a bug the probe caught in the copy rather than in the
   * code. The unconstrained best-separating plane for the loud-axis cloud retains
   * 49% of the variance, so a failure message quoting it was telling the player to
   * go somewhere that scores "Lost variance" — advice that loops. The number worth
   * quoting, and the honest denominator for a separation score, is the best that is
   * reachable WITHOUT giving up the variance target.
   */
  minimumRetained?: number,
): {
  ratio: number;
  angles: Angles;
} {
  let best = { ratio: -1, angles: DEFAULT_ANGLES };
  const matrix = minimumRetained === undefined ? null : covariance(points);
  const separationAt = separationProbe(points);

  const evaluate = (yaw: number, pitch: number) => {
    const angles: Angles = { yaw, pitch, roll: 0 };
    if (
      matrix !== null &&
      minimumRetained !== undefined &&
      varianceRetained(matrix, angles) < minimumRetained
    ) {
      return;
    }
    const ratio = separationAt(angles);
    if (ratio > best.ratio) best = { ratio, angles };
  };

  for (let yaw = -90; yaw < 90; yaw += 6) {
    for (let pitch = -90; pitch <= 90; pitch += 6) evaluate(yaw, pitch);
  }
  const coarse = best.angles;
  for (let yaw = coarse.yaw - 6; yaw <= coarse.yaw + 6; yaw += 1) {
    for (let pitch = coarse.pitch - 6; pitch <= coarse.pitch + 6; pitch += 1) {
      evaluate(yaw, pitch);
    }
  }

  return best.ratio < 0 ? { ratio: 0, angles: DEFAULT_ANGLES } : best;
}

// ── Judging the dive ──────────────────────────────────────────────────────

/** Fraction of the PCA optimum the player has to reach. */
export const VARIANCE_TARGET = 0.98;
/**
 * Largest over smallest eigenvalue below which a cloud is treated as round: its
 * principal axes exist, but barely. The same line the tests draw for the shells.
 */
export const NEAR_ISOTROPIC = 1.5;

export const isNearIsotropic = (eigen: Eigen): boolean =>
  eigen.values[2] > 1e-12 && eigen.values[0] / eigen.values[2] < NEAR_ISOTROPIC;

/**
 * The whole range the gauge can show for a cloud: discard the loudest axis and
 * you keep the least, discard the quietest and you keep the most.
 */
export function gaugeSpan(eigen: Eigen): [number, number] {
  const [l1, l2, l3] = eigen.values;
  const total = l1 + l2 + l3;
  if (total <= 1e-12) return [1, 1];
  return [(l2 + l3) / total, (l1 + l2) / total];
}
/** Below this share of the optimum, the projection has thrown away real structure. */
export const LOST_VARIANCE = 0.85;
/** A separation this far below the achievable best counts as groups still mixed. */
export const SEPARATION_TARGET = 0.7;
/** Below this ratio nothing is visibly separated at all. */
export const SEPARATION_FLOOR = 0.35;

export interface Analysis {
  covariance: Mat3;
  eigen: Eigen;
  best: number;
  /** The best separation ANY flat shadow achieves. Decides if the cloud is separable. */
  bestSeparation: { ratio: number; angles: Angles };
  /** The best separation reachable while still hitting the variance target. */
  reachableSeparation: { ratio: number; angles: Angles };
  pcaAngles: Angles;
}

/** Everything about a cloud that does not depend on the player. Compute once. */
export function analyse(points: readonly Point3D[]): Analysis {
  const matrix = covariance(points);
  const eigen = jacobiEigen(matrix);
  const best = bestVariance(eigen);
  return {
    covariance: matrix,
    eigen,
    best,
    bestSeparation: bestSeparation(points),
    reachableSeparation: bestSeparation(points, best * VARIANCE_TARGET),
    pcaAngles: anglesForAxis(eigen.vectors[2]),
  };
}

export type Outcome = "diving" | "surfaced" | "lost-variance" | "mixed";

export interface Evaluation {
  outcome: Outcome;
  retained: number;
  /** Retained as a share of the PCA optimum. */
  share: number;
  separation: Separation;
  /** Separation as a share of the best achievable. */
  separationShare: number;
  /** False when no flat shadow of this cloud separates the groups at all. */
  separable: boolean;
  score: number;
  /**
   * A 0–3 quality grade for this one dive: surfaced, on the optimum, and showing
   * the groups as well as a reachable plane can.
   *
   * NOT the mastery stars, and named so nobody mistakes it for them. Those come
   * from the progression service (a clear, a best score of 0.8, a code-lane
   * clear), and this used to be called `stars` while the star criteria on screen
   * described it — so the tooltip promised a third star the engine never gave.
   */
  grade: number;
  failure: NamedFailure | null;
}

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

/**
 * Judge the projection.
 *
 * `submitted` matters: the gauge is live and the player is meant to hunt with it,
 * so a named failure cannot fire while they are still turning the cloud — it would
 * be shouting at somebody mid-sentence. Nothing is judged until they commit.
 */
export function evaluate({
  points,
  analysis,
  angles,
  submitted,
}: {
  points: readonly Point3D[];
  analysis: Analysis;
  angles: Angles;
  submitted: boolean;
}): Evaluation {
  const retained = varianceRetained(analysis.covariance, angles);
  const share = analysis.best <= 1e-12 ? 1 : retained / analysis.best;
  const shadow = project(points, angles);
  const sep = separation(shadow);
  const separationShare =
    analysis.reachableSeparation.ratio <= 1e-9
      ? 1
      : clamp(sep.ratio / analysis.reachableSeparation.ratio, 0, 1);

  const base = {
    retained,
    share,
    separation: sep,
    separationShare,
    separable: analysis.bestSeparation.ratio >= SEPARATION_FLOOR,
  };

  if (!submitted) {
    return { ...base, outcome: "diving", score: 0, grade: 0, failure: null };
  }

  const hitVariance = share >= VARIANCE_TARGET;

  /**
   * Some clouds cannot be separated by ANY flat shadow, and the separation
   * criterion has to be waived for those rather than quietly passed.
   *
   * Measured on the nested shells: the best projection anywhere scores 0.012 on
   * between-over-within scatter, which is nothing, and the PCA plane scores 88% OF
   * that nothing. Judging by share alone would have awarded stars for a shadow in
   * which the groups are completely superimposed — technically near-optimal, and
   * useless. So an absolute floor decides whether the question is even askable.
   */
  const separable = analysis.bestSeparation.ratio >= SEPARATION_FLOOR;
  const hitSeparation = !separable || separationShare >= SEPARATION_TARGET;

  if (!hitVariance) {
    const wasted = analysis.best - retained;

    /**
     * A round cloud still has a best plane, but not much of one — and the copy
     * below this branch was written for clouds that do.
     *
     * Measured on the nested shells: the variances are 4.39, 4.27 and 3.86, the
     * sampling noise of 300 points on spheres, and EVERY possible shadow keeps
     * between 64.9% and 69.2%. The variance target still applies, so committing
     * at the default angles (65.4%) is still genuinely short of it — but "3.8% of
     * the structure has been flattened out of existence… a direction with real
     * spread in it" was describing structure this cloud does not have. So on a
     * near-isotropic spectrum the verdict says what is actually true: the whole
     * gauge spans four points, and which axis is thinnest is barely decided.
     */
    if (isNearIsotropic(analysis.eigen)) {
      const [floor, ceiling] = gaugeSpan(analysis.eigen);
      // How much of the whole gauge the shortfall is, stated as the number it
      // is. The first version said "most of the range there is" to every miss,
      // and about a quarter of them are short by less than half of it.
      const span = ceiling - floor;
      const gapShare = span > 1e-12 ? clamp(wasted / span, 0, 1) : 1;
      return {
        ...base,
        outcome: "lost-variance",
        score: clamp(share * 0.5, 0, 1),
        grade: 0,
        failure: {
          name: "Lost variance",
          detail: `This shadow keeps ${percent(
            retained,
          )} of the cloud's spread, and the best any flat shadow of it can keep is ${percent(
            analysis.best,
          )} — ${(wasted * 100).toFixed(
            1,
          )} points short, against a target of ${Math.round(
            VARIANCE_TARGET * 100,
          )}% of the best. On this cloud that gap is ${Math.round(
            gapShare * 100,
          )}% of the whole range there is: every possible shadow keeps between ${percent(
            floor,
          )} and ${percent(
            ceiling,
          )}, because the three variances are ${analysis.eigen.values
            .map((value) => value.toFixed(2))
            .join(
              ", ",
            )} — nearly equal. A round cloud has no real long or short axis, and differences this small are mostly the sampling noise of ${
            points.length
          } points rather than structure. PCA still has a best plane, barely, and it is the one that discards the direction the data is thinnest along, however slightly thinner that is. Turn the cloud a little at a time and watch the gauge, which is still an exact reading of it.`,
        },
      };
    }

    return {
      ...base,
      outcome: "lost-variance",
      score: clamp(share * 0.5, 0, 1),
      grade: 0,
      failure: {
        name: "Lost variance",
        detail: `This shadow keeps ${percent(
          retained,
        )} of the cloud's spread, and the best any flat shadow of it can keep is ${percent(
          analysis.best,
        )} — so ${percent(
          wasted,
        )} of the structure has been flattened out of existence. The axis you are discarding runs along a direction with real spread in it: the cloud's three variances are ${analysis.eigen.values
          .map((value) => value.toFixed(2))
          .join(", ")}, and throwing away anything but the smallest costs you. Turn the cloud until the direction pointing away from you is the one the data is thinnest along — that is the whole of what PCA computes, and it computes it by eigendecomposing exactly the ${percent(
          retained,
        )} you are looking at.`,
      },
    };
  }

  if (!hitSeparation) {
    return {
      ...base,
      outcome: "mixed",
      score: clamp(0.45 + 0.15 * share, 0, 1),
      grade: 1,
      failure: {
        name: "High variance, mixed groups",
        detail: `You found a variance-optimal plane — ${percent(
          retained,
        )} retained against a ceiling of ${percent(
          analysis.best,
        )} — and the groups are still sitting on top of each other. That is not your mistake, it is the limitation. PCA maximises VARIANCE, and variance is only ever a stand-in for the thing you actually wanted to see. This cloud's three variances are ${analysis.eigen.values
          .map((value) => value.toFixed(2))
          .join(
            ", ",
          )}: the loudest direction by a huge margin, and it carries no group information whatsoever. The groups live along the QUIETEST axis, which is exactly the one PCA throws away. Your shadow scores ${sep.ratio.toFixed(
          2,
        )} for the best line you could draw through it; there is another plane retaining ${percent(
          varianceRetained(
            analysis.covariance,
            analysis.reachableSeparation.angles,
          ),
        )} — still inside the target — that scores ${analysis.reachableSeparation.ratio.toFixed(
          2,
        )}. Discard the middle axis instead of the smallest. Because the loud axis dominates the total so completely, that costs you almost nothing on the gauge and it is the difference between seeing three groups and seeing one smear.`,
      },
    };
  }

  // Both. The extra grades are for landing on the optimum rather than near it,
  // and for a shadow that actually shows the structure.
  //
  // 0.85 was set against an earlier pancake, on which the best a variance-optimal
  // plane managed was a 90% separation share. Re-measured on the cloud that ships
  // (seed 2029), the bar has room to spare: the best-separating plane retains 83%
  // of the variance and the PCA plane itself reaches a 99.8% separation share, so
  // on the gentle cloud the top grade is simply "find the optimum". The tests hold
  // the PCA-plane figure.
  const grade =
    1 +
    (share >= 0.995 ? 1 : 0) +
    (!separable || separationShare >= 0.85 ? 1 : 0);
  return {
    ...base,
    outcome: "surfaced",
    separable,
    score: clamp(0.5 + 0.3 * share + 0.2 * (separable ? separationShare : 1), 0, 1),
    grade: Math.min(grade, 3),
    failure: null,
  };
}

// ── Reveal the math ───────────────────────────────────────────────────────

export const MATH_EQUATION = String.raw`C = \frac{1}{n-1}\sum_i (x_i - \bar{x})(x_i - \bar{x})^{\mathsf{T}}
\qquad C v_k = \lambda_k v_k
\\[1.2em]
\text{retained}(v_1, v_2) = \frac{v_1^{\mathsf{T}} C v_1 + v_2^{\mathsf{T}} C v_2}{\operatorname{tr} C}
= 1 - \frac{v_3^{\mathsf{T}} C v_3}{\operatorname{tr} C}
\\[1em]
\max_{v_1, v_2} \text{retained} = \frac{\lambda_1 + \lambda_2}{\lambda_1 + \lambda_2 + \lambda_3}`;

export const MATH_CODE = `// PCA, in full. There is no training loop.
const C = covariance(points);          // 3x3, symmetric
const { values, vectors } = jacobiEigen(C);   // descending

// The best plane is spanned by the top two eigenvectors, and the
// score is decided by the one you drop:
const retained = (values[0] + values[1]) / (values[0] + values[1] + values[2]);

// Which means the search space is not "all pairs of directions".
// It is a single direction on a sphere - the one to discard.
// Roll cannot change the answer, and neither can swapping the two
// kept axes. Fewer degrees of freedom than the sliders suggest.`;

export const MATH_NOTES = `Every projection onto a plane keeps v₁ᵀCv₁ + v₂ᵀCv₂ of the cloud's spread. Because the three axes of a rotation are orthonormal, that quantity equals trace(C) − v₃ᵀCv₃ — so the score depends only on the direction you throw away. This is why the roll slider does nothing to the number: rolling spins the picture inside the plane it already chose, and the discarded axis never moves. If that feels wrong, it is worth sitting with, because it is the difference between choosing a plane and choosing a picture.

PCA is then not a search at all. Once you accept that you are picking one direction to discard, you want the direction the data is thinnest along, and that is the eigenvector of the covariance matrix with the smallest eigenvalue. Eigendecompose a 3×3 symmetric matrix and you are done — no gradient descent, no learning rate, no epochs. It is one of the few things in this whole site with a closed-form answer, and the eigenvalues even tell you what the answer is worth before you look at it.

Which brings up what PCA does not do. It maximises variance, and nobody has ever actually wanted variance. What people want is to see the structure, and variance is a stand-in for that which happens to work most of the time. One of these clouds is built to break the correspondence: its loudest direction is pure noise with enormous spread, and the direction that separates the groups is the quietest of the three. PCA will confidently hand you a plane that retains over 99% of the spread and shows you nothing, and it will not warn you, because it was never looking at the labels. Nothing in the algorithm knows the groups exist.

And there is a harder limit than that. Take the nested shells: one group inside another. Turn it however you like — no flat shadow of that arrangement pulls them apart, because a linear projection can only ever take weighted sums of coordinates, and no weighted sum of coordinates distinguishes "near the centre" from "far from it". This is precisely the gap t-SNE and UMAP exist to fill: they abandon the requirement that the map be a linear projection, keep local neighbourhoods instead of global variance, and can therefore unroll structure that PCA has no vocabulary for. They also give up what PCA has — an exact answer, a meaning for each axis, and the ability to project a new point without refitting.`;
