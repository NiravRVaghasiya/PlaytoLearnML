/**
 * Shared engine types. These are the contract between the shell and the 14 game
 * modules — no game-specific shapes belong here.
 */

import type { HighlightLanguage } from "@/lib/highlight";

/** The two-lane rule (CLAUDE.md): every game ships both. */
export type Lane = "visual" | "code";

/**
 * The "Reveal the Math" payload (DESIGN.md §5/§6).
 *
 * `ml-verifier` checks that `equation` matches what `code` actually does and
 * that `code` matches the real implementation in the game's `ml.ts`. Paraphrase
 * freely, but never show an equation the simulation doesn't run.
 */
export interface MathReveal {
  /** KaTeX/LaTeX source, rendered in display mode. */
  equation: string;
  /** The real code driving the sim — copy it from `ml.ts`, don't invent it. */
  code: string;
  codeLanguage?: HighlightLanguage;
  /** Optional prose: what the symbols mean, in plain language. */
  notes?: string;
}

/**
 * A named ML failure (pedagogy contract #4).
 *
 * `name` is the failure's real name ("Overfitting", "Divergence"). `detail`
 * must carry the numbers that justify it ("99% train, 61% test"). A generic
 * "Game Over" is a contract violation.
 */
export interface NamedFailure {
  name: string;
  detail: string;
}

/** Mastery stars, per spec §4: complete / high score / code-lane challenge. */
export type StarCount = 0 | 1 | 2 | 3;
