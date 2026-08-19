"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";
import { project } from "./ml";
import { cloudSpec, useDiverStore } from "./store";
import { ShadowPlane2D } from "./ShadowPlane2D";

/**
 * Three.js is loaded only in the browser and only for this lane.
 *
 * It is 600kB and it is the sole dependency in this project that cannot run without
 * WebGL, so it must not be part of the initial bundle or of any server render. The
 * shadow — the view that actually carries the meaning — is plain SVG and appears
 * immediately, so the lane is usable before this arrives and remains usable if it
 * never does.
 */
const PointCloud3D = dynamic(
  () => import("./PointCloud3D").then((module_) => module_.PointCloud3D),
  {
    ssr: false,
    loading: () => (
      <div
        className="w-full animate-pulse rounded-md border border-border bg-surface-2"
        style={{ height: 300 }}
      />
    ),
  },
);

export function VisualLane() {
  const points = useDiverStore((s) => s.points);
  const angles = useDiverStore((s) => s.angles);
  const showGroups = useDiverStore((s) => s.showGroups);
  const submitted = useDiverStore((s) => s.submitted);
  const evaluation = useDiverStore((s) => s.evaluation);
  const cloud = useDiverStore(cloudSpec);

  /**
   * Derived here, not selected from the store.
   *
   * The shadow changes on every slider tick and a stale one would show a projection
   * the player is no longer holding — but it is also a fresh array, so selecting it
   * through the store hook gives `useSyncExternalStore` a new snapshot every render
   * and the component loops until React throws error 185. Which it did. Select the
   * stable slices; memoise the derivation.
   */
  const shadow = useMemo(() => project(points, angles), [points, angles]);

  const discardedLabel = useMemo(
    () =>
      `yaw ${angles.yaw.toFixed(0)}°, pitch ${angles.pitch.toFixed(0)}°`,
    [angles.yaw, angles.pitch],
  );

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-sm font-semibold">
          {cloud.title}
          <span className="ml-2 font-normal text-text-muted">
            {points.length} points, 3 hidden groups
          </span>
        </h2>
        <span className="text-xs text-text-muted">
          {submitted ? "committed" : "turning"}
        </span>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <PointCloud3D
          points={points}
          angles={angles}
          showGroups={showGroups}
          discardedLabel={discardedLabel}
        />
        <div>
          <h3 className="mb-1.5 text-xs font-semibold">
            The shadow it casts
          </h3>
          <ShadowPlane2D
            shadow={shadow}
            showGroups={showGroups}
            retained={evaluation.retained}
          />
        </div>
      </div>
    </div>
  );
}
