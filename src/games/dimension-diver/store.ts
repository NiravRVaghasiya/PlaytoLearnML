"use client";

import { create } from "zustand";
import { useProgression } from "@/engine/progression";
import type { Lane, NamedFailure } from "@/engine/types";
import type { WhyCardContent } from "@/components";
import { clamp } from "@/lib/utils";
import {
  CLOUDS,
  DEFAULT_ANGLES,
  analyse,
  buildCloud,
  evaluate,
  project,
  separationProbe,
  varianceRetained,
  type Analysis,
  type Angles,
  type Evaluation,
  type Point3D,
  type Shadow,
} from "./ml";
import { whyCardFor } from "./why-cards";

export const SLUG = "dimension-diver";
export const CLOUD_SEED = 2029;

export type Phase = "diving" | "surfaced";
export type AxisName = "yaw" | "pitch" | "roll";

/**
 * Dimension Diver state.
 *
 * The spec's data model is:
 *
 *   Point3D    { x, y, z, groupId }
 *   Projection { basisVectors[2], varianceRetained }
 *   GameState  { cloud[], projection, bestVariance }
 *
 * `Projection` is not stored. Its basis vectors and its retained variance are both
 * functions of three angles, and the angles are what the sliders write — storing the
 * derived form as well would be storing the same decision twice and inviting the two
 * to disagree. `bestVariance` lives in `analysis`, alongside the eigendecomposition
 * it comes from, because it is a property of the cloud rather than of the game.
 *
 * `analysis` IS cached, and that is a deliberate exception to deriving everything.
 * It contains two sweeps over a thousand candidate planes to find the best
 * reachable separation — tens of milliseconds, measured, not the "milliseconds"
 * this comment used to claim — and it is completely independent of the player.
 * It is computed once per cloud per page (see `cloudData`), so turning the cloud,
 * retrying it and coming back to it all reuse the same answer.
 */
export interface DiverState {
  cloudIndex: number;
  points: Point3D[];
  analysis: Analysis;

  angles: Angles;
  /** True once the player has committed, which is when a verdict may be named. */
  submitted: boolean;
  /**
   * Whether the PCA hint has been used on this cloud. Multiplies the recorded
   * score by `HINT_FACTOR`, which costs the high-score star, not the third one
   * (that is the code-lane star, and the hint has nothing to do with it).
   */
  hintUsed: boolean;
  /**
   * Cloud ids the hint has been used on this session. Survives "Start this cloud
   * again" on purpose: the hint prints the exact angles on screen, so a restart
   * followed by typing them back in used to earn the unpenalised score.
   */
  hintedIds: string[];
  /** Cloud ids surfaced this session. */
  surfacedIds: string[];
  /**
   * This round has already sent progression a code-lane surfacing. Lets a code
   * commit of a projection the sliders already surfaced record once as code —
   * the third star — instead of being swallowed as "the same verdict again".
   */
  codeRecorded: boolean;

  evaluation: Evaluation;
  phase: Phase;
  failure: NamedFailure | null;
  whyCard: WhyCardContent | null;
  lane: Lane;
  /** Colour the shadow by hidden group. Off until submitted, or asked for. */
  showGroups: boolean;

  setLane: (lane: Lane) => void;
  setAngle: (axis: AxisName, degrees: number) => void;
  setAngles: (angles: Partial<Angles>) => void;
  nudge: (axis: AxisName, delta: number) => void;
  /**
   * Commit the current projection. `source` is where the commit came from, so XP
   * and the code-lane star follow the action rather than the visible tab.
   * A second commit of the same projection is a no-op: it is the same verdict —
   * unless it is the code lane's first commit of a surfaced one (`codeRecorded`).
   */
  submit: (source?: Lane) => void;
  /** Snap to the PCA plane and say what it is. */
  usePcaHint: () => void;
  toggleGroups: () => void;
  nextCloud: () => void;
  reset: () => void;
}

// ── selectors: primitives and stable references only ─────────────────────

export const cloudSpec = (state: DiverState) => CLOUDS[state.cloudIndex]!;
export const retainedFraction = (state: DiverState): number =>
  state.evaluation.retained;
export const shareOfOptimum = (state: DiverState): number =>
  state.evaluation.share;

/**
 * The 2D shadow. Derived, never stored — it changes on every slider tick.
 *
 * NOT a selector. It builds a fresh array, so passing it to `useDiverStore(...)`
 * hands `useSyncExternalStore` a new snapshot on every render and the component
 * re-renders forever — React error 185, which is exactly how this was found. Call it
 * with a state snapshot inside `useMemo`, keyed on `points` and `angles`, which are
 * the stable slices it actually depends on.
 */
export function shadowFor(state: DiverState): Shadow[] {
  return project(state.points, state.angles);
}

/**
 * A cloud, its analysis, and its untouched verdict — built once per page.
 *
 * All three are pure functions of the cloud id and the fixed seed, so rebuilding
 * them on every retry and every return to a cloud was the same work producing the
 * same answer. Reusing the same `points` array also means the 3D view keeps its
 * WebGL renderer across a retry instead of building a new one.
 */
const cloudCache = new Map<
  number,
  { points: Point3D[]; analysis: Analysis; evaluation: Evaluation }
>();

function cloudData(cloudIndex: number) {
  const cached = cloudCache.get(cloudIndex);
  if (cached !== undefined) return cached;
  const points = buildCloud(CLOUDS[cloudIndex]!.id, CLOUD_SEED);
  const analysis = analyse(points);
  const entry = {
    points,
    analysis,
    evaluation: evaluate({
      points,
      analysis,
      angles: DEFAULT_ANGLES,
      submitted: false,
    }),
  };
  cloudCache.set(cloudIndex, entry);
  return entry;
}

/** The api's `separationAt`, one allocation-free probe per cloud. */
const probes = new WeakMap<readonly Point3D[], ReturnType<typeof separationProbe>>();
function probeFor(points: readonly Point3D[]) {
  let probe = probes.get(points);
  if (probe === undefined) {
    probe = separationProbe(points);
    probes.set(points, probe);
  }
  return probe;
}

function freshCloud(cloudIndex: number, hintedIds: readonly string[] = []) {
  const { points, analysis, evaluation } = cloudData(cloudIndex);
  return {
    cloudIndex,
    points,
    analysis,
    angles: { ...DEFAULT_ANGLES },
    submitted: false,
    hintUsed: hintedIds.includes(CLOUDS[cloudIndex]!.id),
    showGroups: false,
    evaluation,
    phase: "diving" as Phase,
    failure: null,
    codeRecorded: false,
  };
}

/**
 * Wrap into -180..180 so the sliders never wander off into multiples of a turn.
 *
 * One modulo, not a loop. The first version subtracted 360 until the value was in
 * range, and a code-lane angle of 1e20 never got there — 1e20 − 360 is 1e20 in
 * floating point — so the tab hung inside the api where `checkBudget` cannot
 * reach. In-range values pass through untouched, so both ends of the slider stay
 * where the player put them.
 */
export const normalise = (degrees: number): number => {
  if (degrees >= -180 && degrees <= 180) return degrees;
  const wrapped = ((degrees % 360) + 360) % 360;
  return wrapped > 180 ? wrapped - 360 : wrapped;
};

/** The hint's price: the same 0.7 Backprop Blitz charges for its trace. */
export const HINT_FACTOR = 0.7;

/**
 * What a surfaced dive is worth to progression, 0–1: the verdict's own score,
 * the hint's price, and a campaign term so surfacing more clouds is worth more.
 * Rounded to six places so a formula that lands on 0.8 lands on it exactly.
 */
export function diveScore({
  evaluationScore,
  hinted,
  surfacedCount,
}: {
  evaluationScore: number;
  hinted: boolean;
  surfacedCount: number;
}): number {
  const raw =
    evaluationScore *
    (hinted ? HINT_FACTOR : 1) *
    (0.75 + 0.25 * (clamp(surfacedCount, 0, CLOUDS.length) / CLOUDS.length));
  return Math.round(clamp(raw, 0, 1) * 1e6) / 1e6;
}

export const useDiverStore = create<DiverState>((set, get) => ({
  ...freshCloud(0),
  hintedIds: [],
  surfacedIds: [],
  whyCard: whyCardFor({ kind: "briefing", cloud: CLOUDS[0]! }),
  lane: "visual" as Lane,

  setLane: (lane) => set({ lane }),

  setAngle: (axis, degrees) => {
    get().setAngles({ [axis]: degrees });
  },

  setAngles: (partial) => {
    const state = get();
    const angles: Angles = {
      yaw: normalise(partial.yaw ?? state.angles.yaw),
      pitch: clamp(partial.pitch ?? state.angles.pitch, -90, 90),
      roll: normalise(partial.roll ?? state.angles.roll),
    };
    // A NaN would poison every derived number on screen. The code lane refuses
    // one by name before it gets here; this is the store refusing it too.
    if (![angles.yaw, angles.pitch, angles.roll].every(Number.isFinite)) return;
    if (
      angles.yaw === state.angles.yaw &&
      angles.pitch === state.angles.pitch &&
      angles.roll === state.angles.roll
    ) {
      return;
    }

    // Turning the cloud un-submits it: the verdict belonged to a plane that is no
    // longer on screen, and leaving a red failure strip attached to a projection the
    // player has since moved away from would be lying about the current state.
    const evaluation = evaluate({
      points: state.points,
      analysis: state.analysis,
      angles,
      submitted: false,
    });

    // Roll is the exception worth calling out, so the card fires for it.
    const rolledOnly =
      partial.roll !== undefined &&
      partial.yaw === undefined &&
      partial.pitch === undefined;

    set({
      angles,
      submitted: false,
      phase: "diving",
      failure: null,
      evaluation,
      whyCard: rolledOnly
        ? whyCardFor({
            kind: "rolled",
            retained: evaluation.retained,
            roll: angles.roll,
          })
        : state.whyCard,
    });
  },

  nudge: (axis, delta) => {
    const state = get();
    get().setAngles({ [axis]: state.angles[axis] + delta });
  },

  submit: (source = "visual") => {
    const state = get();
    // Already judged, and nothing has moved since: the same verdict again, so it
    // must not be recorded again (running a snippet twice, say). Except once: a
    // code commit of a projection the sliders surfaced is the round's first
    // code-lane surfacing, and dropping it would silently withhold the third star.
    const firstCodeSurfacing =
      source === "code" && state.phase === "surfaced" && !state.codeRecorded;
    if (state.submitted && !firstCodeSurfacing) return;
    const evaluation = evaluate({
      points: state.points,
      analysis: state.analysis,
      angles: state.angles,
      submitted: true,
    });
    const cloud = cloudSpec(state);
    const surfaced = evaluation.outcome === "surfaced";

    set({
      submitted: true,
      evaluation,
      // Revealing the groups on submit is the payoff: the whole dive was a guess
      // about structure the player could not see, and this is when they find out.
      showGroups: true,
      phase: surfaced ? "surfaced" : "diving",
      failure: evaluation.failure,
      codeRecorded: state.codeRecorded || (surfaced && source === "code"),
      surfacedIds:
        surfaced && !state.surfacedIds.includes(cloud.id)
          ? [...state.surfacedIds, cloud.id]
          : state.surfacedIds,
      whyCard: whyCardFor({
        kind: "submitted",
        cloud,
        evaluation,
        analysis: state.analysis,
        hintUsed: state.hintUsed,
      }),
    });

    if (surfaced) {
      const surfacedCount = state.surfacedIds.includes(cloud.id)
        ? state.surfacedIds.length
        : state.surfacedIds.length + 1;
      useProgression.getState().recordResult({
        slug: SLUG,
        score: diveScore({
          evaluationScore: evaluation.score,
          hinted: state.hintUsed,
          surfacedCount,
        }),
        lane: source,
        completed: true,
        codeLaneCleared: source === "code",
      });
    }
  },

  usePcaHint: () => {
    const state = get();
    const angles = state.analysis.pcaAngles;
    const evaluation = evaluate({
      points: state.points,
      analysis: state.analysis,
      angles,
      submitted: false,
    });
    const cloud = cloudSpec(state);
    set({
      angles,
      hintUsed: true,
      hintedIds: state.hintedIds.includes(cloud.id)
        ? state.hintedIds
        : [...state.hintedIds, cloud.id],
      submitted: false,
      phase: "diving",
      failure: null,
      evaluation,
      whyCard: whyCardFor({
        kind: "hinted",
        analysis: state.analysis,
        retained: evaluation.retained,
      }),
    });
  },

  toggleGroups: () => set({ showGroups: !get().showGroups }),

  nextCloud: () => {
    const state = get();
    const next = (state.cloudIndex + 1) % CLOUDS.length;
    set({
      ...freshCloud(next, state.hintedIds),
      surfacedIds: state.surfacedIds,
      whyCard: whyCardFor({ kind: "briefing", cloud: CLOUDS[next]! }),
    });
  },

  reset: () => {
    const state = get();
    set({
      ...freshCloud(state.cloudIndex, state.hintedIds),
      surfacedIds: state.surfacedIds,
      whyCard: whyCardFor({ kind: "briefing", cloud: CLOUDS[state.cloudIndex]! }),
    });
  },
}));

/**
 * What the code lane can do, and nothing more.
 *
 * Every verb writes the same store the sliders write (CLAUDE.md two-lane rule).
 * `api.pca` hands over the eigendecomposition, which the visual lane keeps behind a
 * button — in here that is the point rather than a cheat: computing the principal
 * axes and comparing them against the angle you found by eye is the exercise.
 */
export interface DiverCodeApi {
  setAngles: (angles: Partial<Angles>) => void;
  submit: () => void;
  nextCloud: () => void;
  reset: () => void;

  cloud: () => { id: string; title: string; points: number };
  angles: () => Angles;
  /** Retained variance and separation at the current angles. */
  score: () => {
    retained: number;
    share: number;
    separation: number;
    separationShare: number;
    outcome: string;
  };
  /** Retained variance at arbitrary angles, without moving the cloud. */
  retainedAt: (angles: Partial<Angles>) => number;
  /** Separation at arbitrary angles, without moving the cloud. */
  separationAt: (angles: Partial<Angles>) => number;
  /** The real eigendecomposition: values, vectors, and the optimal angles. */
  pca: () => {
    eigenvalues: [number, number, number];
    eigenvectors: [number[], number[], number[]];
    bestRetained: number;
    angles: Angles;
  };
  /** The projected points, with their hidden group ids. */
  shadow: () => Array<{ u: number; v: number; groupId: number }>;
  points: () => Array<{ x: number; y: number; z: number; groupId: number }>;
}

export function createCodeApi(): DiverCodeApi {
  const store = useDiverStore;

  const anglesFrom = (partial: Partial<Angles>): Angles => {
    if (partial === null || typeof partial !== "object" || Array.isArray(partial)) {
      throw new Error(
        "Angles are an object of degrees, like { yaw: 30, pitch: -15, roll: 0 }.",
      );
    }
    for (const key of Object.keys(partial)) {
      if (key !== "yaw" && key !== "pitch" && key !== "roll") {
        throw new Error(`Unknown angle "${key}". Use yaw, pitch and roll.`);
      }
    }
    const current = store.getState().angles;
    const merged = {
      yaw: partial.yaw ?? current.yaw,
      pitch: partial.pitch ?? current.pitch,
      roll: partial.roll ?? current.roll,
    };
    for (const [key, value] of Object.entries(merged)) {
      if (!Number.isFinite(value)) {
        throw new Error(`${key} must be a finite number of degrees.`);
      }
    }
    return merged;
  };

  return {
    setAngles: (angles) => store.getState().setAngles(anglesFrom(angles)),
    submit: () => store.getState().submit("code"),
    nextCloud: () => store.getState().nextCloud(),
    reset: () => store.getState().reset(),

    cloud: () => {
      const state = store.getState();
      const spec = CLOUDS[state.cloudIndex]!;
      return { id: spec.id, title: spec.title, points: state.points.length };
    },

    angles: () => ({ ...store.getState().angles }),

    score: () => {
      const { evaluation } = store.getState();
      return {
        retained: evaluation.retained,
        share: evaluation.share,
        separation: evaluation.separation.ratio,
        separationShare: evaluation.separationShare,
        outcome: evaluation.outcome,
      };
    },

    retainedAt: (angles) => {
      const state = store.getState();
      return varianceRetained(state.analysis.covariance, anglesFrom(angles));
    },

    separationAt: (angles) => {
      const state = store.getState();
      // Same number as `separation(project(...)).ratio`, held to it by the tests,
      // without building a shadow — the starter snippet asks for 3,660 of these.
      return probeFor(state.points)(anglesFrom(angles));
    },

    pca: () => {
      const { analysis } = store.getState();
      return {
        eigenvalues: [...analysis.eigen.values] as [number, number, number],
        eigenvectors: analysis.eigen.vectors.map((vector) => [...vector]) as [
          number[],
          number[],
          number[],
        ],
        bestRetained: analysis.best,
        angles: { ...analysis.pcaAngles },
      };
    },

    shadow: () =>
      shadowFor(store.getState()).map((point) => ({
        u: point.u,
        v: point.v,
        groupId: point.groupId,
      })),

    points: () =>
      store.getState().points.map((point) => ({
        x: point.x,
        y: point.y,
        z: point.z,
        groupId: point.groupId,
      })),
  };
}
