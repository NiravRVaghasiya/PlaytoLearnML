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
