/** Small shared helpers. Nothing game-specific belongs here. */

/** Join conditional class names. Falsy values are dropped. */
export function cx(
  ...parts: Array<string | false | null | undefined>
): string {
  return parts.filter(Boolean).join(" ");
}

/** Clamp `n` into [min, max]. */
export function clamp(n: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, n));
}

/** Format a 0..1 ratio as a whole-percent string, e.g. 0.8421 → "84%". */
export function formatPercent(ratio: number, digits = 0): string {
  return `${(ratio * 100).toFixed(digits)}%`;
}

/**
 * Deterministic PRNG (mulberry32). Games use this so a "seed" reproduces the
 * exact same dataset — essential for testing ML logic and for fair leaderboards.
 */
export function seededRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * One standard-normal draw from a uniform source, by the Box–Muller transform.
 * Consumes exactly two values from `random`, so a seeded generator still
 * reproduces a dataset exactly.
 *
 * Lives here because seven games needed it and each had its own copy — and the
 * copies had already drifted apart on the one line that matters. `log(0)` is
 * `-Infinity`, so the first draw has to be floored above zero; the game-local
 * versions disagreed about the floor (`Number.EPSILON` in six of them, `1e-12`
 * in the seventh), which quietly made one game's clouds a different shape from
 * everyone else's in the degenerate case.
 */
export function gaussian(random: () => number): number {
  const u = Math.max(random(), Number.EPSILON);
  const v = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Resolve after the browser has had a chance to paint.
 *
 * `await Promise.resolve()` is not enough: microtasks run before rendering, so a
 * "Running…" label set just before a long synchronous job would never reach the
 * screen. A `requestAnimationFrame` callback runs right before a paint, and the
 * `setTimeout` inside it lands after that paint — which is the point.
 *
 * rAF is paused in background tabs, so a plain timeout races it: a job started
 * from a hidden tab (autoRun on a tab opened in the background) still starts,
 * a frame or so late, instead of waiting until the player comes back.
 *
 * Games chunking a heavy loop use this between batches so the UI stays live:
 * `if (i % 10 === 0) await yieldToPaint();`
 */
export function yieldToPaint(fallbackMs = 100): Promise<void> {
  return new Promise<void>((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    if (typeof requestAnimationFrame === "function") {
      requestAnimationFrame(() => setTimeout(finish, 0));
    }
    setTimeout(finish, fallbackMs);
  });
}

/** Cached result of the one WebGL probe per page. `null` = not probed yet. */
let webglSupport: boolean | null = null;

/**
 * Can this browser create a WebGL context? Probed once per page, then cached.
 *
 * Every probe is a real GL context, and browsers cap live contexts (Chrome: 16)
 * by force-losing the OLDEST one — which in a long session can be TF.js's own
 * WebGL backend. So the probe releases its context straight away through
 * `WEBGL_lose_context` rather than waiting for garbage collection, and it runs
 * once, not on every mount of a 3D view (Gradient Descent Skier's terrain,
 * Dimension Diver's point cloud).
 */
export function isWebGLAvailable(): boolean {
  if (webglSupport !== null) return webglSupport;
  // Server render: can't know, and must not cache a guess for the client.
  if (typeof document === "undefined") return false;

  try {
    const canvas = document.createElement("canvas");
    const gl = (canvas.getContext("webgl2") ?? canvas.getContext("webgl")) as
      | WebGLRenderingContext
      | WebGL2RenderingContext
      | null
      | undefined;
    webglSupport = Boolean(gl);
    gl?.getExtension("WEBGL_lose_context")?.loseContext();
  } catch {
    webglSupport = false;
  }
  return webglSupport;
}

/** Test seam: forget the cached WebGL probe. Never needed by app code. */
export function resetWebGLProbeForTests(): void {
  webglSupport = null;
}
