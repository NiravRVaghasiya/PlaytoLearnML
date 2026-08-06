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
 * `isWebGLAvailable` is exported so the lane can choose the contour view *before*
 * mounting this, rather than showing a broken canvas.
 */

/** Can this browser actually give us a WebGL context? */
export function isWebGLAvailable(): boolean {
  if (typeof document === "undefined") return false;
  try {
    const canvas = document.createElement("canvas");
    return Boolean(
      canvas.getContext("webgl2") ??
        canvas.getContext("webgl") ??
        canvas.getContext("experimental-webgl"),
    );
  } catch {
    return false;
  }
}

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

export function LossTerrain({ trail, position, diverged }: LossTerrainProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const skierRef = useRef<THREE.Mesh | null>(null);
  const trailRef = useRef<THREE.Line | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);

  // Scene setup, once.
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      // The lane checks isWebGLAvailable() first, so this is belt-and-braces.
      return;
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 200);
    camera.position.set(0, 9.5, 12);
    camera.lookAt(0, 0.6, 0);

    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    mount.appendChild(renderer.domElement);
    renderer.domElement.setAttribute("aria-hidden", "true");

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

    sceneRef.current = scene;
    rendererRef.current = renderer;

    const resize = () => {
      const width = mount.clientWidth;
      const height = Math.max(220, Math.round(width * 0.62));
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);
    };

    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(mount);

    let frame = 0;
    const loop = () => {
      renderer.render(scene, camera);
      frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
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
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
      sceneRef.current = null;
      rendererRef.current = null;
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
  }, [trail, position, diverged]);

  return <div ref={mountRef} className="w-full" />;
}
