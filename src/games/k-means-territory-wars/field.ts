/**
 * Pointer geometry for the village map. Pure functions, kept out of the
 * component so the two mistakes they exist to prevent are unit-tested.
 *
 * ── Why the SVG's own matrix, not its bounding box ──────────────────────────
 * The map is a `viewBox="0 0 100 100"` SVG filling a flex column. On a desktop
 * the column is much taller than it is wide (1329 px tall for 878 px wide at
 * 1440×900), and the default `preserveAspectRatio` ("xMidYMid meet")
 * letterboxes the square map with empty bands above and below. Mapping a
 * pointer through `getBoundingClientRect()` ignored those bands, so a flag
 * dragged to y = 90 landed at y = 76. `getScreenCTM()` is the transform the
 * browser actually painted with, so inverting it puts the flag under the
 * pointer at any aspect ratio.
 *
 * (Sort-It Arcade carries the same few lines in its own `field.ts`: each game
 * folder is self-contained.)
 */

/** The slice of `DOMMatrix` this needs — a 2-D affine transform. */
export interface AffineMatrix {
  a: number;
  b: number;
  c: number;
  d: number;
  e: number;
  f: number;
}

/** What `SVGGraphicsElement.getScreenCTM()` returns, structurally. */
export interface ScreenMatrix extends AffineMatrix {
  inverse(): AffineMatrix;
}

/**
 * Client (CSS pixel) coordinates → the SVG's user units (its viewBox), or null
 * when there is no usable transform (element not rendered, degenerate scale).
 */
export function clientToUser(
  clientX: number,
  clientY: number,
  ctm: ScreenMatrix | null,
): { x: number; y: number } | null {
  if (!ctm) return null;
  const inverse = ctm.inverse();
  const x = inverse.a * clientX + inverse.c * clientY + inverse.e;
  const y = inverse.b * clientX + inverse.d * clientY + inverse.f;
  return Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null;
}

/**
 * Half of DESIGN.md's 44 px minimum touch target. A flag is grabbable anywhere
 * within this many CSS pixels of its centre, whatever size the map renders at —
 * the old fixed 6-unit circle came to about 39 px across on a 324 px phone map.
 */
export const GRAB_RADIUS_PX = 22;

/**
 * The grab radius never shrinks below this in viewBox units, so a large
 * desktop map keeps the generous target it always had.
 */
export const MIN_GRAB_RADIUS_UNITS = 6;

/**
 * Which flag a press at `point` grabs: the NEAREST one within `radius`, or null.
 *
 * Nearest, not topmost. Two flags within a hit radius of each other used to
 * resolve by SVG paint order, so pressing squarely on flag 1 could pick up
 * flag 2 simply because it was drawn later.
 */
export function flagUnderPointer(
  point: { x: number; y: number },
  flags: ReadonlyArray<{ x: number; y: number }>,
  radius: number,
): number | null {
  let best: number | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < flags.length; index += 1) {
    const flag = flags[index]!;
    const distance = Math.hypot(flag.x - point.x, flag.y - point.y);
    // Strictly nearer wins, so an exact tie goes to the lower index — the same
    // rule `nearestCentroid` uses for villages.
    if (distance <= radius && distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  }
  return best;
}
