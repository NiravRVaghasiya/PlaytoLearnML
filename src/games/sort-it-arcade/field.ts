import { clamp } from "@/lib/utils";

/**
 * Pointer geometry for the Sort-It field. Pure functions, kept out of the
 * component so the two mistakes they exist to prevent are unit-tested.
 *
 * ── Why the SVG's own matrix, not its bounding box ──────────────────────────
 * The field is a `viewBox="0 0 100 100"` SVG that fills a flex column. On a
 * desktop the column is taller than it is wide, and the default
 * `preserveAspectRatio` ("xMidYMid meet") letterboxes the square content with
 * empty bands above and below. Mapping a pointer through
 * `getBoundingClientRect()` ignores those bands, so a drag to the bottom of the
 * field landed 10–20% of the way up it. `getScreenCTM()` is the transform the
 * browser actually painted with, letterboxing included, so inverting it puts the
 * handle exactly under the pointer at any aspect ratio.
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
 * Which knot owns field position `x` (0–1): the nearest one, with knots evenly
 * spaced from 0 to 1.
 *
 * ── Why a column, not a hit circle ──────────────────────────────────────────
 * Each knot used to carry its own transparent hit circle, 5 units in radius.
 * With 25 knots they sit 3.7 units apart, so every circle overlapped the next,
 * and SVG paint order sent a press on handle i to handle i + 1. Owning a column
 * instead makes the whole height of the field a target for each knot, and a
 * column can't overlap its neighbour. Height is not width, though: a column
 * is about 88 / (knots − 1) viewBox units wide, so the curve's are about 70
 * CSS px on a 324 px phone field and the wiggle's only about 12 px.
 */
export function nearestKnotIndex(x: number, knotCount: number): number {
  if (knotCount <= 1) return 0;
  return Math.round(clamp(x, 0, 1) * (knotCount - 1));
}

/** Half of DESIGN.md's 44 px minimum touch target. */
export const GRAB_RADIUS_PX = 22;
