"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import {
  DOMAIN,
  GLOBAL_MINIMUM,
  LOCAL_MINIMA,
  lossAt,
  type Vec2,
} from "./ml";

export interface LossTerrainProps {
  trail: readonly Vec2[];
  position: Vec2;
  diverged: boolean;
  /**
   * WebGL turned out not to work after all — the renderer could not be built,
   * or the browser took the context away (a GPU reset, a backgrounded tab on
   * iOS). The lane falls back to the contour view rather than a dead canvas.
   */
  onUnavailable?: () => void;
}

/**
 * The 3-D loss surface (spec: `<LossTerrain>`, Three.js).
 *
 * Presentation only. Every control lives in the rail as a real form widget, and
 * the camera is fixed — no orbit controls, deliberately. A rotatable camera adds a
 * mouse-only interaction that a keyboard or switch user cannot perform, and
 * nothing about the lesson requires changing the viewing angle. Players who want a
 * different read of the surface get the contour view, which is a full equal and
 * not a fallback.
 *
 * The lane asks the shared `isWebGLAvailable()` in `@/lib/utils` so it can
 * choose the contour view *before* mounting this, rather than showing a broken
 * canvas.
 *
 * ── GPU etiquette ────────────────────────────────────────────────────────────
 * Nothing in this scene moves on its own — the camera is fixed and the skier
 * only moves when the store does — so it renders on demand: after setup, after
 * a resize, and after each change to the skier or the trail. It used to redraw
 * a 9,216-vertex mesh twice over, 60 times a second, for as long as the lane
 * was open: battery and heat on a phone, for a picture that wasn't changing.
 *
 * Browsers cap live WebGL contexts (Chrome: 16) and silently kill the oldest
 * when the cap is hit — which in a single-page session can be the TF.js
 * backend of another game. So the capability probe runs once per page and
 * releases its context (the shared helper does both), and unmount forces the
 * renderer's context loss rather than waiting for garbage collection.
 * `renderer.dispose()` alone frees buffers, not the context.
 */

const GRID = 96;
/** World units the domain maps onto. */
const SPAN = 10;
/** Vertical exaggeration, so the wells read as valleys. */
const HEIGHT = 1.5;

function toWorld(x: number, y: number): THREE.Vector3 {
  const u = (x - DOMAIN.minX) / (DOMAIN.maxX - DOMAIN.minX);
  const v = (y - DOMAIN.minY) / (DOMAIN.maxY - DOMAIN.minY);
  return new THREE.Vector3(
    (u - 0.5) * SPAN,
    lossAt(x, y) * HEIGHT,
    (v - 0.5) * SPAN * 0.75,
  );
}

export function LossTerrain({
  trail,
  position,
  diverged,
  onUnavailable,
}: LossTerrainProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const skierRef = useRef<THREE.Mesh | null>(null);
  const trailRef = useRef<THREE.Line | null>(null);
  /** Draw one frame. Null until the scene exists, and again after unmount. */
  const renderRef = useRef<(() => void) | null>(null);
  const onUnavailableRef = useRef(onUnavailable);
  useEffect(() => {
    onUnavailableRef.current = onUnavailable;
  });

  // Scene setup, once.
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      // The lane checks isWebGLAvailable() first, but a context can still be
      // refused here (too many live contexts, a blocklisted GPU). Fall back.
      onUnavailableRef.current?.();
      return;
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 200);
    camera.position.set(0, 9.5, 12);
    camera.lookAt(0, 0.6, 0);

    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    mount.appendChild(renderer.domElement);
    renderer.domElement.setAttribute("aria-hidden", "true");
    // Sized in CSS pixels here, not by the drawing buffer: the buffer is
    // width × devicePixelRatio, and letting it size the element made the
    // canvas twice the lane's width on a phone.
    renderer.domElement.style.display = "block";
    renderer.domElement.style.width = "100%";

    const onContextLost = (event: Event) => {
      event.preventDefault();
      onUnavailableRef.current?.();
    };
    renderer.domElement.addEventListener("webglcontextlost", onContextLost);

    // Terrain mesh: a plane whose vertices are displaced by the real loss.
    const geometry = new THREE.PlaneGeometry(
      SPAN,
      SPAN * 0.75,
      GRID - 1,
      GRID - 1,
    );
    const positions = geometry.attributes.position!;
    const colors = new Float32Array(positions.count * 3);

    let low = Infinity;
    let high = -Infinity;
    const heights: number[] = [];

    for (let index = 0; index < positions.count; index += 1) {
      const px = positions.getX(index);
      const py = positions.getY(index);
      const dataX =
        DOMAIN.minX + (px / SPAN + 0.5) * (DOMAIN.maxX - DOMAIN.minX);
      const dataY =
        DOMAIN.minY + (py / (SPAN * 0.75) + 0.5) * (DOMAIN.maxY - DOMAIN.minY);
      const loss = lossAt(dataX, dataY);
      heights.push(loss);
      low = Math.min(low, loss);
      high = Math.max(high, loss);
      positions.setZ(index, loss * HEIGHT);
    }

    // Height-mapped lightness only — the same greyscale-safe scheme the contour
    // view uses, so switching views doesn't change how the surface reads.
    for (let index = 0; index < heights.length; index += 1) {
      const t = (heights[index]! - low) / Math.max(1e-9, high - low);
      const lightness = 0.1 + Math.pow(t, 0.55) * 0.42;
      const color = new THREE.Color().setHSL(0.6, 0.3, lightness);
      colors[index * 3] = color.r;
      colors[index * 3 + 1] = color.g;
      colors[index * 3 + 2] = color.b;
    }
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.rotateX(-Math.PI / 2);
    geometry.computeVertexNormals();

    scene.add(
      new THREE.Mesh(
        geometry,
        new THREE.MeshStandardMaterial({
          vertexColors: true,
          roughness: 0.9,
          metalness: 0,
          side: THREE.DoubleSide,
        }),
      ),
    );

    // Wireframe over the top, so the shape is legible without relying on shading.
    scene.add(
      new THREE.Mesh(
        geometry,
        new THREE.MeshBasicMaterial({
          color: 0x2c3444,
          wireframe: true,
          transparent: true,
          opacity: 0.28,
        }),
      ),
    );

    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const key = new THREE.DirectionalLight(0xffffff, 0.9);
    key.position.set(-6, 12, 8);
    scene.add(key);

    // Flags at both minima, shape-coded the same way as the contour view.
    for (const minimum of [GLOBAL_MINIMUM, ...LOCAL_MINIMA]) {
      const isGlobal = minimum.kind === "global";
      const marker = new THREE.Mesh(
        isGlobal
          ? new THREE.TorusGeometry(0.32, 0.07, 8, 24)
          : new THREE.BoxGeometry(0.42, 0.42, 0.42),
        new THREE.MeshBasicMaterial({
          color: isGlobal ? 0x2ecc71 : 0xf1c40f,
        }),
      );
      const world = toWorld(minimum.x, minimum.y);
      marker.position.set(world.x, world.y + 0.3, world.z);
      if (isGlobal) marker.rotation.x = Math.PI / 2;
      scene.add(marker);
    }

    // The skier.
    const skier = new THREE.Mesh(
      new THREE.SphereGeometry(0.22, 20, 16),
      new THREE.MeshStandardMaterial({ color: 0xffd54f, roughness: 0.4 }),
    );
    scene.add(skier);
    skierRef.current = skier;

    // The path trail.
    const trailGeometry = new THREE.BufferGeometry();
    const line = new THREE.Line(
      trailGeometry,
      new THREE.LineBasicMaterial({ color: 0x4fc3f7 }),
    );
    scene.add(line);
    trailRef.current = line;

    const render = () => renderer.render(scene, camera);
    renderRef.current = render;

    const resize = () => {
      const width = mount.clientWidth;
      if (width === 0) return;
      const height = Math.max(220, Math.round(width * 0.62));
      // updateStyle false: the buffer follows the pixel ratio, while the
      // element stays at 100% of the lane (above) and this CSS height.
      renderer.setSize(width, height, false);
      renderer.domElement.style.height = `${height}px`;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      render();
    };

    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(mount);

    return () => {
      renderRef.current = null;
      observer.disconnect();
      renderer.domElement.removeEventListener("webglcontextlost", onContextLost);
      // Three.js holds GPU buffers that survive garbage collection, so dispose
      // explicitly — the same discipline useModel applies to tensors.
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh || object instanceof THREE.Line) {
          object.geometry.dispose();
          const material = object.material;
          if (Array.isArray(material)) material.forEach((m) => m.dispose());
          else material.dispose();
        }
      });
      renderer.dispose();
      // dispose() frees buffers but leaves the context alive until GC; give
      // it back now, so lane toggles and Retries don't stack up contexts.
      renderer.forceContextLoss();
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
      skierRef.current = null;
      trailRef.current = null;
    };
  }, []);

  // Move the skier and redraw the trail as the descent proceeds.
  useEffect(() => {
    const skier = skierRef.current;
    if (skier) {
      const world = toWorld(
        Math.min(Math.max(position.x, DOMAIN.minX), DOMAIN.maxX),
        Math.min(Math.max(position.y, DOMAIN.minY), DOMAIN.maxY),
      );
      skier.position.set(world.x, world.y + 0.24, world.z);
      skier.visible = !diverged;
      (skier.material as THREE.MeshStandardMaterial).color.set(
        diverged ? 0xe74c3c : 0xffd54f,
      );
    }

    const line = trailRef.current;
    if (line) {
      const points = trail.map((point) => {
        const world = toWorld(
          Math.min(Math.max(point.x, DOMAIN.minX), DOMAIN.maxX),
          Math.min(Math.max(point.y, DOMAIN.minY), DOMAIN.maxY),
        );
        return new THREE.Vector3(world.x, world.y + 0.12, world.z);
      });
      line.geometry.dispose();
      line.geometry = new THREE.BufferGeometry().setFromPoints(points);
    }

    // The one frame this change needs.
    renderRef.current?.();
  }, [trail, position, diverged]);

  return <div ref={mountRef} className="w-full" />;
}
