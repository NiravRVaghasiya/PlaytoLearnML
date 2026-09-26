"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { isWebGLAvailable } from "@/lib/utils";
import { GROUP_LABELS, frameFor, type Angles, type Point3D } from "./ml";
import { GROUP_COLOUR_TOKENS, UNREVEALED_COLOUR_TOKEN } from "./ShadowPlane2D";

export interface PointCloud3DProps {
  points: readonly Point3D[];
  angles: Angles;
  showGroups: boolean;
  /** The direction being discarded, drawn as an arrow pointing at the viewer. */
  discardedLabel: string;
}

/**
 * A design token as a three.js colour, read from the same CSS custom property
 * the shadow fills with.
 *
 * The first version hardcoded three hexes of its own (a softer blue, orange and
 * green), so a group was one colour in 3D and another in the shadow directly
 * beneath it. Reading the token means the two views cannot drift apart.
 */
export function tokenColour({ token, fallback }: { token: string; fallback: string }) {
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue(token)
    .trim();
  return new THREE.Color(value === "" ? fallback : value);
}

/** Free everything the GPU holds for a scene graph: geometry and materials. */
function disposeTree(root: THREE.Object3D) {
  root.traverse((object) => {
    if (object instanceof THREE.Points || object instanceof THREE.LineSegments) {
      object.geometry.dispose();
      const material = object.material as THREE.Material | THREE.Material[];
      if (Array.isArray(material)) material.forEach((entry) => entry.dispose());
      else material.dispose();
    }
  });
}

/**
 * The cloud in 3D (spec: `<PointCloud3D>`, Three.js).
 *
 * This is the one place in the whole site where a real 3D renderer earns its keep:
 * the player's task is to turn a cloud in space, and depth, perspective and
 * occlusion are the cues that make an orientation legible. So Three.js, as the spec
 * asks.
 *
 * ── What that costs, and how it is paid ─────────────────────────────────────
 * A WebGL canvas is invisible to assistive technology. That is acceptable HERE and
 * nowhere else in this game, because every judgement the player actually makes is
 * made from `<ShadowPlane2D>`, which is SVG with the group counts and spread in its
 * label, and from the gauge, which is text. This view shows what you are holding;
 * the shadow shows what it means. If this canvas were missing entirely the game
 * would still be playable and still be winnable — which is the test for whether a
 * canvas is decoration or content.
 *
 * WebGL is also not guaranteed. If context creation fails the component says so and
 * gets out of the way rather than leaving an empty box, and the rest of the game
 * carries on working.
 *
 * ── Rendering and releasing ────────────────────────────────────────────────
 * There is no render loop: a frame is drawn only when something it shows has
 * changed — the angles, the points, the grouping, the size — so an idle cloud
 * costs nothing. The canvas is sized in CSS pixels (`setSize` writes its style)
 * with the backing store at up to 2× for sharpness, so a high-density screen can
 * never double its layout width. And on unmount the renderer is disposed AND its
 * context released with `forceContextLoss()`: `dispose()` alone frees three's
 * caches but leaves the context alive until GC — the same leak the capability
 * probe used to have, before it became the shared, self-releasing
 * `isWebGLAvailable`. The renderer lives as long as the cloud's `points` array,
 * which the store now reuses across a retry, so "Start this cloud again" no
 * longer builds a new one.
 */
export function PointCloud3D({
  points,
  angles,
  showGroups,
  discardedLabel,
}: PointCloud3DProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const groupRef = useRef<THREE.Group | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  /**
   * WebGL support is detected once, during render, rather than by failing inside an
   * effect and calling setState — which is what the first version did and which
   * React's lint rule correctly objects to, because it turns a capability check into
   * a cascading re-render. A read-only probe in a lazy initialiser is the standard
   * shape for this, and it also means the fallback renders on the first pass instead
   * of flashing an empty box first. The probe itself is `isWebGLAvailable`, run
   * once per page and shared with Gradient Descent Skier's terrain.
   */
  const [supported] = useState(isWebGLAvailable);

  // Scene setup, once.
  useEffect(() => {
    const mount = mountRef.current;
    if (mount === null || !supported) return;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      return;
    }

    const width = mount.clientWidth || 340;
    const height = 300;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.setClearColor(0x000000, 0);
    mount.appendChild(renderer.domElement);
    renderer.domElement.setAttribute("aria-hidden", "true");

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 200);
    // Looking down -z, so the discarded axis of the frame points at the viewer.
    camera.position.set(0, 0, 26);
    camera.lookAt(0, 0, 0);

    const cloud = new THREE.Group();
    scene.add(cloud);

    // A faint box so the rotation is readable even where points are sparse.
    const bounds = points.reduce(
      (max, point) =>
        Math.max(max, Math.abs(point.x), Math.abs(point.y), Math.abs(point.z)),
      1,
    );
    const cage = new THREE.LineSegments(
      new THREE.EdgesGeometry(
        new THREE.BoxGeometry(bounds * 2, bounds * 2, bounds * 2),
      ),
      new THREE.LineBasicMaterial({ color: 0x8a8f98, transparent: true, opacity: 0.22 }),
    );
    cloud.add(cage);

    rendererRef.current = renderer;
    sceneRef.current = scene;
    cameraRef.current = camera;
    groupRef.current = cloud;

    const onResize = () => {
      const next = mount.clientWidth || width;
      renderer.setSize(next, height);
      camera.aspect = next / height;
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);
    };
    window.addEventListener("resize", onResize);

    return () => {
      window.removeEventListener("resize", onResize);
      disposeTree(cloud);
      renderer.dispose();
      // After dispose(), which removes three's context-lost listener, so the
      // deliberate loss is not logged as if it were an accident.
      renderer.forceContextLoss();
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
      rendererRef.current = null;
      sceneRef.current = null;
      cameraRef.current = null;
      groupRef.current = null;
    };
    // Scene lifetime is tied to the cloud identity; angles are applied separately.
  }, [points, supported]);

  // Points, rebuilt when the cloud or the grouping changes.
  useEffect(() => {
    const cloud = groupRef.current;
    if (cloud === null) return;

    // Clear previous point objects, keeping the cage (index 0).
    for (let index = cloud.children.length - 1; index >= 1; index -= 1) {
      const child = cloud.children[index]!;
      cloud.remove(child);
      if (child instanceof THREE.Points) {
        child.geometry.dispose();
        (child.material as THREE.Material).dispose();
      }
    }

    // Read per rebuild rather than once, so they are always the tokens in force.
    const groupColours = GROUP_COLOUR_TOKENS.map(tokenColour);
    const unrevealed = tokenColour(UNREVEALED_COLOUR_TOKEN);

    const byGroup: Point3D[][] = GROUP_LABELS.map(() => []);
    for (const point of points) byGroup[point.groupId]?.push(point);

    byGroup.forEach((group, groupId) => {
      if (group.length === 0) return;
      const positions = new Float32Array(group.length * 3);
      group.forEach((point, index) => {
        positions[index * 3] = point.x;
        positions[index * 3 + 1] = point.y;
        positions[index * 3 + 2] = point.z;
      });
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(positions, 3),
      );
      const material = new THREE.PointsMaterial({
        color: showGroups ? (groupColours[groupId] ?? unrevealed) : unrevealed,
        size: 0.24,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.9,
      });
      cloud.add(new THREE.Points(geometry, material));
    });

    const renderer = rendererRef.current;
    const scene = sceneRef.current;
    const camera = cameraRef.current;
    if (renderer && scene && camera) renderer.render(scene, camera);
  }, [points, showGroups]);

  // Orientation. Applied as the inverse of the projection frame, so that the axis
  // the projection discards ends up pointing straight at the camera — the shadow on
  // screen is then literally what the maths keeps.
  useEffect(() => {
    const cloud = groupRef.current;
    const renderer = rendererRef.current;
    const scene = sceneRef.current;
    const camera = cameraRef.current;
    if (!cloud || !renderer || !scene || !camera) return;

    const frame = frameFor(angles);
    const matrix = new THREE.Matrix4().set(
      frame[0][0], frame[0][1], frame[0][2], 0,
      frame[1][0], frame[1][1], frame[1][2], 0,
      frame[2][0], frame[2][1], frame[2][2], 0,
      0, 0, 0, 1,
    );
    cloud.quaternion.setFromRotationMatrix(matrix);
    renderer.render(scene, camera);
  }, [angles]);

  if (!supported) {
    return (
      <p className="rounded-md border border-border bg-surface-2 p-3 text-xs text-text-muted">
        This browser could not start WebGL, so the 3D view is unavailable. The
        shadow below is the part that matters and still works — it is what the
        projection actually produces.
      </p>
    );
  }

  return (
    <figure className="m-0">
      <div
        ref={mountRef}
        className="w-full overflow-hidden rounded-md border border-border bg-surface-2"
        style={{ height: 300 }}
      />
      <figcaption className="mt-1.5 text-xs text-text-muted">
        The cloud, turned by your sliders. The axis pointing at you —{" "}
        {discardedLabel} — is the one being discarded, and it is the only thing the
        gauge responds to.
      </figcaption>
    </figure>
  );
}
