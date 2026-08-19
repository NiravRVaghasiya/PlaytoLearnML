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
  separation,
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
 * It contains a sweep over a thousand candidate planes to find the best reachable
 * separation, which is milliseconds but is also completely independent of the player
 * — recomputing it on every slider tick would be work done to produce the same
 * answer. It is invalidated by exactly one thing: changing cloud.
 */
export interface DiverState {
  cloudIndex: number;
  points: Point3D[];
  analysis: Analysis;

  angles: Angles;
  /** True once the player has committed, which is when a verdict may be named. */
  submitted: boolean;
  /** Whether the PCA hint was used on this cloud. Costs the third star. */
  hintUsed: boolean;
  /** Cloud ids surfaced this session. */
  surfacedIds: string[];

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
  submit: () => void;
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

function freshCloud(cloudIndex: number) {
  const points = buildCloud(CLOUDS[cloudIndex]!.id, CLOUD_SEED);
  const analysis = analyse(points);
  return {
    cloudIndex,
    points,
    analysis,
    angles: { ...DEFAULT_ANGLES },
    submitted: false,
    hintUsed: false,
    showGroups: false,
    evaluation: evaluate({
      points,
      analysis,
      angles: DEFAULT_ANGLES,
      submitted: false,
    }),
    phase: "diving" as Phase,
    failure: null,
  };
}

const normalise = (degrees: number): number => {
  // Wrap into -180..180 so the sliders never wander off into multiples of a turn.
  let value = degrees;
  while (value > 180) value -= 360;
  while (value < -180) value += 360;
  return value;
};

export const useDiverStore = create<DiverState>((set, get) => ({
  ...freshCloud(0),
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

  submit: () => {
    const state = get();
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
        score: clamp(
          evaluation.score *
            (state.hintUsed ? 0.7 : 1) *
            (0.75 + 0.25 * (surfacedCount / CLOUDS.length)),
          0,
          1,
        ),
        lane: state.lane,
        completed: true,
        codeLaneCleared: state.lane === "code",
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
    set({
      angles,
      hintUsed: true,
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
      ...freshCloud(next),
      surfacedIds: state.surfacedIds,
      whyCard: whyCardFor({ kind: "briefing", cloud: CLOUDS[next]! }),
    });
  },

  reset: () => {
    const state = get();
    set({
      ...freshCloud(state.cloudIndex),
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
    submit: () => store.getState().submit(),
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
      return separation(project(state.points, anglesFrom(angles))).ratio;
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
