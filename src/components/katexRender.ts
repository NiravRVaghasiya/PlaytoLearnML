/**
 * KaTeX, loaded only when a Math drawer first opens.
 *
 * `MathDrawer` imports this module with a dynamic `import()`, so KaTeX's
 * ~270 KB of JavaScript and its stylesheet split into their own chunk instead
 * of shipping with every game — most players never open the drawer, and those
 * who do pay for it once. Keep every KaTeX import in this file; a static
 * `import katex` anywhere else would pull it back into the game bundles.
 */
import katex, { type KatexOptions } from "katex";
import "katex/dist/katex.min.css";

/**
 * Every game's equation, as KaTeX actually receives it.
 *
 * Games write multi-line equations with top-level `\\` (often `\\[1.2em]`).
 * In display mode KaTeX treats a bare `\\` exactly as LaTeX does — it does
 * nothing, with a `newLineInDisplayMode` warning — so most of the games'
 * equations were rendering as one run-on line. `gathered` is the amsmath environment built for
 * stacked, centred display lines, and inside it `\\` (with an optional
 * `[space]`) is a real line break. A single-line equation renders identically.
 */
export function wrapEquation(equation: string): string {
  return `\\begin{gathered}${equation}\\end{gathered}`;
}

/**
 * Runtime options.
 *
 * - `htmlAndMathml`: the HTML is for the eye (and `aria-hidden` by KaTeX); the
 *   MathML beside it is what a screen reader reads, instead of the raw LaTeX a
 *   plain-text fallback would give it.
 * - `throwOnError: true` so bad input reaches the caller's fallback (the LaTeX
 *   source as text) instead of KaTeX's inline red error, which fails contrast
 *   on our dark surface.
 * - `strict: "ignore"` at runtime only: a stray non-standard construct should
 *   render, not spam the console. The unit test re-runs every game's equation
 *   with strict checking, so real problems are still caught — in CI.
 */
export const EQUATION_OPTIONS: KatexOptions = {
  displayMode: true,
  output: "htmlAndMathml",
  throwOnError: true,
  strict: "ignore",
};

/** Render one equation to HTML. Returns "" when KaTeX can't parse it. */
export function renderEquation(equation: string): string {
  try {
    return katex.renderToString(wrapEquation(equation), EQUATION_OPTIONS);
  } catch {
    // Never let a bad LaTeX string take down a game.
    return "";
  }
}
