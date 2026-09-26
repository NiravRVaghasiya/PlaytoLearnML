import { WEIGHT_MAX } from "./ml";

/**
 * How much of the class colour a weight's cell gets, in percent, over
 * `--surface-2`: the faintest non-zero weight at the floor, ±WEIGHT_MAX at the
 * ceiling, the same scale for both signs.
 *
 * The ceiling is a contrast limit, not a taste. The number in each cell is drawn
 * in `--text`, and the ceiling used to be 76%, where a -2 cell's orange left that
 * text at 2.8:1 — under WCAG AA's 4.5:1 on every "Vertical edge" preset and on
 * every learned grid whose strongest weight is negative. At 46% the worst cell,
 * orange, measures 5.0:1 (tint.test.tsx computes it from the tokens). Switching to
 * dark text on strong tints does not work instead: between about 51% and 65%
 * orange, neither `--text` nor `--bg` reaches 4.5:1.
 *
 * Only `--text` may sit on a tint. `--text-muted` on the strongest orange is
 * 2.4:1, below even the 3:1 an icon needs, which is why the steppers' plus and
 * minus glyphs are `--text` too.
 */
export const TINT_FLOOR = 14;
export const TINT_CEILING = 46;

/**
 * A kernel cell's background: blue for positive, orange for negative (a pair a
 * red-green colour deficiency does not hide), stronger for a larger magnitude.
 * Shared by `<KernelDesigner>` and `<LearnedKernels>` so a learned grid reads
 * exactly like a drawn one. Anything that rounds to 0.0 gets no tint.
 */
export function weightTint(weight: number): string {
  if (!Number.isFinite(weight) || Math.abs(weight) < 0.05) {
    return "var(--surface-2)";
  }
  const strength =
    (Math.min(Math.abs(weight), WEIGHT_MAX) / WEIGHT_MAX) *
      (TINT_CEILING - TINT_FLOOR) +
    TINT_FLOOR;
  return `color-mix(in srgb, var(${
    weight > 0 ? "--class-a" : "--class-b"
  }) ${strength}%, var(--surface-2))`;
}
