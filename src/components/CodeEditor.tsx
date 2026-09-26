"use client";

import { useId, useMemo, useRef } from "react";
import { AlertCircle } from "lucide-react";
import { cx } from "@/lib/utils";
import {
  TOKEN_COLOR,
  highlight,
  type HighlightLanguage,
} from "@/lib/highlight";

export interface CodeEditorProps {
  label: string;
  value: string;
  onChange: (code: string) => void;
  language?: HighlightLanguage;
  readOnly?: boolean;
  /** Runtime/parse error from the last run. Shown, not thrown. */
  error?: string | null;
  hint?: string;
  rows?: number;
  className?: string;
}

/**
 * The code lane's editable snippet (DESIGN.md §6).
 *
 * A transparent `<textarea>` sits over a syntax-coloured `<pre>` so editing uses
 * the platform's own text field — real caret, real selection, real IME, real
 * screen-reader support — while still showing colour. The `<pre>` is
 * `aria-hidden`, so assistive tech reads the textarea only and never sees the
 * decoration twice.
 *
 * Tab is deliberately NOT captured. Trapping Tab to insert indentation strands
 * keyboard users inside the field, which fails DESIGN.md §9. Players indent with
 * spaces; the snippets we ship are short by design.
 */
export function CodeEditor({
  label,
  value,
  onChange,
  language = "javascript",
  readOnly = false,
  error,
  hint,
  rows = 14,
  className,
}: CodeEditorProps) {
  const id = useId();
  const errorId = `${id}-error`;
  const hintId = `${id}-hint`;
  const preRef = useRef<HTMLPreElement>(null);

  const lines = useMemo(() => value.split("\n"), [value]);

  const describedBy =
    [hint ? hintId : null, error ? errorId : null].filter(Boolean).join(" ") ||
    undefined;

  // Shared metrics: the overlay only lines up if both layers agree exactly.
  const shared =
    "m-0 p-3 font-mono text-sm leading-6 whitespace-pre-wrap break-words";

  return (
    <div className={cx("flex min-h-0 flex-col", className)}>
      <label htmlFor={id} className="mb-2 text-sm font-medium">
        {label}
      </label>

      <div className="relative min-h-0 flex-1 overflow-hidden rounded-md border border-border bg-bg focus-within:border-primary">
        {/* Colour layer. Decorative only. */}
        <pre
          ref={preRef}
          aria-hidden="true"
          className={cx(shared, "pointer-events-none absolute inset-0 overflow-auto")}
        >
          <code>
            {lines.map((line, lineIndex) => (
              <span key={lineIndex} className="block">
                {highlight(line, language).map((token, tokenIndex) => (
                  <span
                    key={tokenIndex}
                    style={{ color: TOKEN_COLOR[token.kind] }}
                  >
                    {token.text}
                  </span>
                ))}
                {/* Keep empty lines tall so the layers stay in sync. */}
                {line.length === 0 ? "\u200b" : null}
              </span>
            ))}
          </code>
        </pre>

        {/* Input layer: transparent text, visible caret. */}
        <textarea
          id={id}
          value={value}
          readOnly={readOnly}
          onChange={(event) => onChange(event.target.value)}
          onScroll={(event) => {
            const pre = preRef.current;
            if (!pre) return;
            pre.scrollTop = event.currentTarget.scrollTop;
            pre.scrollLeft = event.currentTarget.scrollLeft;
          }}
          rows={rows}
          spellCheck={false}
          autoCapitalize="off"
          autoCorrect="off"
          aria-describedby={describedBy}
          aria-invalid={error ? true : undefined}
          className={cx(
            shared,
            "relative block h-full w-full resize-none bg-transparent text-transparent caret-[var(--text)]",
            "outline-none selection:bg-primary/30",
          )}
        />
      </div>

      {hint ? (
        <p id={hintId} className="mt-2 text-xs text-text-muted">
          {hint}
        </p>
      ) : null}

      {error ? (
        // Red carries the state (border, tint, icon); the message itself is in
        // --text. --wrong text on its own 10% tint measured 4.1:1 at 12px,
        // under the 4.5:1 AA bar — and this box is in all fourteen code lanes.
        // pre-wrap keeps a multi-line message (a Python traceback summary, a
        // snippet's own thrown text) on separate lines instead of one run-on.
        <p
          id={errorId}
          role="alert"
          className="mt-2 flex items-start gap-2 rounded-md border border-wrong/50 bg-wrong/10 p-2 font-mono text-xs text-text"
        >
          <AlertCircle aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-wrong" />
          <span className="min-w-0 break-words whitespace-pre-wrap">{error}</span>
        </p>
      ) : null}
    </div>
  );
}
