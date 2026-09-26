"use client";

import { useMemo } from "react";
import { cx } from "@/lib/utils";
import {
  TOKEN_COLOR,
  highlight,
  type HighlightLanguage,
} from "@/lib/highlight";

export interface CodeBlockProps {
  code: string;
  language?: HighlightLanguage;
  /** Show 1-based line numbers in a non-selectable gutter. */
  showLineNumbers?: boolean;
  className?: string;
  /** Accessible name, e.g. "Gradient descent step, JavaScript". */
  ariaLabel?: string;
}

/**
 * Read-only syntax-coloured code, used by `<MathDrawer>` and anywhere a game
 * needs to show its real implementation.
 *
 * The colouring is decorative — the code reads identically without it, so no
 * meaning is carried by colour alone (DESIGN.md §9). The block is focusable and
 * scrollable so keyboard users can reach long snippets.
 */
export function CodeBlock({
  code,
  language = "javascript",
  showLineNumbers = true,
  className,
  ariaLabel,
}: CodeBlockProps) {
  const lines = useMemo(() => code.replace(/\n$/, "").split("\n"), [code]);

  return (
    <pre
      tabIndex={0}
      aria-label={ariaLabel}
      className={cx(
        "overflow-auto rounded-md border border-border bg-bg p-3 font-mono text-sm leading-relaxed",
        className,
      )}
    >
      <code>
        {lines.map((line, lineIndex) => (
          <span key={lineIndex} className="block whitespace-pre">
            {showLineNumbers ? (
              <span
                aria-hidden="true"
                // Full-strength --text-muted: at /60 the gutter measured 3.41:1 on
                // --bg, under the 4.5:1 AA bar for text. Full strength is 7.5:1.
                className="mr-4 inline-block w-6 shrink-0 text-right select-none text-text-muted"
              >
                {lineIndex + 1}
              </span>
            ) : null}
            {highlight(line, language).map((token, tokenIndex) => (
              <span
                key={tokenIndex}
                style={{ color: TOKEN_COLOR[token.kind] }}
              >
                {token.text}
              </span>
            ))}
          </span>
        ))}
      </code>
    </pre>
  );
}
