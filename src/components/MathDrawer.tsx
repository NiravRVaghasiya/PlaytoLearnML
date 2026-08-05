"use client";

import { useEffect, useMemo, useRef } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import { X } from "lucide-react";
import { Button } from "./Button";
import { CodeBlock } from "./CodeBlock";
import { cx } from "@/lib/utils";
import type { MathReveal } from "@/engine/types";

export interface MathDrawerProps {
  open: boolean;
  onClose: () => void;
  math: MathReveal;
  /** Game title, used in the drawer's accessible name. */
  title: string;
}

/**
 * The "ƒ Math" reveal drawer (DESIGN.md §5/§6).
 *
 * Shows the real equation (KaTeX) beside the real code driving the simulation.
 * `ml-verifier` fails a game when these disagree with `ml.ts`, so treat the
 * contents as documentation of the implementation, not marketing.
 *
 * Accessibility: rendered as a modal dialog with a focus trap, Escape to close,
 * and focus returned to the trigger on dismiss.
 */
export function MathDrawer({ open, onClose, math, title }: MathDrawerProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const restoreRef = useRef<HTMLElement | null>(null);

  const equationHtml = useMemo(() => {
    try {
      return katex.renderToString(math.equation, {
        displayMode: true,
        throwOnError: false,
        output: "html",
      });
    } catch {
      // Never let a bad LaTeX string take down a game.
      return "";
    }
  }, [math.equation]);

  // Remember the trigger, move focus in, restore on close.
  useEffect(() => {
    if (!open) return;

    restoreRef.current = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();

    return () => restoreRef.current?.focus?.();
  }, [open]);

  // Escape to close + Tab trapped inside the panel.
  useEffect(() => {
    if (!open) return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab") return;

      const focusable = panelRef.current?.querySelectorAll<HTMLElement>(
        'button, [href], input, textarea, select, [tabindex]:not([tabindex="-1"])',
      );
      if (!focusable || focusable.length === 0) return;

      const first = focusable[0]!;
      const last = focusable[focusable.length - 1]!;

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      {/* Backdrop. Presentational — Escape and the close button are the
          keyboard paths out, so this needs no interactive role. */}
      <div
        className="absolute inset-0 bg-bg/80"
        onClick={onClose}
        aria-hidden="true"
      />

      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={`Reveal the math: ${title}`}
        className={cx(
          "relative flex h-full w-full max-w-xl flex-col overflow-y-auto",
          "border-l border-border bg-surface p-6 shadow-raised",
        )}
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="font-mono text-sm text-primary">ƒ Reveal the math</p>
            <h2 className="mt-1 font-display text-xl font-semibold">{title}</h2>
          </div>
          <Button
            ref={closeRef}
            variant="ghost"
            size="sm"
            onClick={onClose}
            icon={<X className="size-4" />}
          >
            Close
          </Button>
        </div>

        <section className="mt-6" aria-labelledby="math-equation-heading">
          <h3
            id="math-equation-heading"
            className="text-sm font-semibold tracking-wide text-text-muted uppercase"
          >
            The equation
          </h3>
          <div className="mt-2 overflow-x-auto rounded-md border border-border bg-surface-2 p-4">
            {equationHtml ? (
              <div dangerouslySetInnerHTML={{ __html: equationHtml }} />
            ) : (
              <p className="font-mono text-sm text-text-muted">
                {math.equation}
              </p>
            )}
          </div>
          {/* KaTeX output is decorative markup to a screen reader; give it a
              plain-text fallback that can actually be read aloud. */}
          <p className="sr-only-live">LaTeX source: {math.equation}</p>
        </section>

        {math.notes ? (
          <section className="mt-6" aria-labelledby="math-notes-heading">
            <h3
              id="math-notes-heading"
              className="text-sm font-semibold tracking-wide text-text-muted uppercase"
            >
              In plain language
            </h3>
            <p className="mt-2 text-sm leading-relaxed">{math.notes}</p>
          </section>
        ) : null}

        <section className="mt-6" aria-labelledby="math-code-heading">
          <h3
            id="math-code-heading"
            className="text-sm font-semibold tracking-wide text-text-muted uppercase"
          >
            The code that actually runs
          </h3>
          <CodeBlock
            className="mt-2"
            code={math.code}
            language={math.codeLanguage ?? "javascript"}
            ariaLabel={`Implementation of ${title}`}
          />
        </section>
      </div>
    </div>
  );
}
