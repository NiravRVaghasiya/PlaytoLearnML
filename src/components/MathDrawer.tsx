"use client";

import { useEffect, useRef, useState } from "react";
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

/** What's in the equation box: rendered HTML, or "" = show the source. */
interface RenderedEquation {
  equation: string;
  html: string;
  /**
   * KaTeX itself didn't arrive (as opposed to "" from a parse failure, which
   * no retry can fix). The source shows meanwhile, and the next open tries
   * the download again instead of settling for it for the rest of the visit.
   */
  loadFailed?: boolean;
}

const FOCUSABLE =
  'button:not([disabled]), [href], input, textarea, select, [tabindex]:not([tabindex="-1"])';

/**
 * The "ƒ Math" reveal drawer (DESIGN.md §5/§6).
 *
 * Shows the real equation (KaTeX) beside the real code driving the simulation.
 * `ml-verifier` fails a game when these disagree with `ml.ts`, so treat the
 * contents as documentation of the implementation, not marketing.
 *
 * Accessibility: rendered as a modal dialog with a focus trap, Escape to close,
 * and focus returned to the trigger on dismiss. Three details make it actually
 * modal rather than only labelled as one:
 *
 * - The panel itself is focusable (`tabIndex={-1}`), so clicking its text keeps
 *   focus inside instead of dropping it on the page behind — from where Tab used
 *   to walk straight out into the game.
 * - Tab from anywhere that isn't inside the panel is pulled back in.
 * - While open, keydown events stop at the dialog (a capture listener on the
 *   document), so a game's document-level shortcuts — Data Detox sorts rows on
 *   1–4 — can't change the game behind an `aria-modal` dialog. Default actions
 *   (Tab, Enter on a button, scrolling) are untouched. `GameShell` also marks
 *   the rest of the page `inert`, which covers pointer input and the virtual
 *   cursor.
 *
 * KaTeX is loaded the first time the drawer opens, not with the game — see
 * `katexRender.ts`. The code block renders immediately; the equation follows a
 * moment later on first open, and instantly after that.
 */
export function MathDrawer({ open, onClose, math, title }: MathDrawerProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const restoreRef = useRef<HTMLElement | null>(null);

  const [rendered, setRendered] = useState<RenderedEquation | null>(null);
  // Stays true after a failed load while the drawer is open (so it isn't
  // retried in a loop), and is true again on the next open.
  const needsRender =
    open && (rendered?.equation !== math.equation || rendered.loadFailed === true);

  // Lazy-load KaTeX on first open (and again only if the equation changes).
  useEffect(() => {
    if (!needsRender) return;
    let live = true;
    const equation = math.equation;

    import("./katexRender")
      .then(({ renderEquation }) => {
        if (live) setRendered({ equation, html: renderEquation(equation) });
      })
      .catch(() => {
        // Chunk failed to load (offline, deploy mid-session). The LaTeX source
        // is still the truth, so show that rather than nothing.
        if (live) setRendered({ equation, html: "", loadFailed: true });
      });

    return () => {
      live = false;
    };
  }, [needsRender, math.equation]);

  const equationHtml =
    rendered && rendered.equation === math.equation ? rendered.html : null;
  /** Stable hook for browser checks: wait for "rendered", not for a timeout. */
  const equationState =
    equationHtml === null ? "loading" : equationHtml ? "rendered" : "source";

  // Remember the trigger, move focus in, restore on close.
  useEffect(() => {
    if (!open) return;

    restoreRef.current = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();

    return () => restoreRef.current?.focus?.();
  }, [open]);

  // Escape to close, Tab trapped inside the panel, and nothing leaks out.
  useEffect(() => {
    if (!open) return;

    const onKeyDown = (event: KeyboardEvent) => {
      // Capture phase on the document: runs before any target or bubbling
      // listener, so stopping it here keeps the page behind from reacting.
      event.stopPropagation();

      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab") return;

      const panel = panelRef.current;
      const focusable = panel?.querySelectorAll<HTMLElement>(FOCUSABLE);
      if (!panel || !focusable || focusable.length === 0) return;

      const first = focusable[0]!;
      const last = focusable[focusable.length - 1]!;
      const active = document.activeElement;

      // Focus on the panel itself (after a click on its text) or, somehow,
      // outside it: the browser's next Tab stop may be in the page behind.
      if (active === panel || !panel.contains(active)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
        return;
      }

      if (event.shiftKey && active === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && active === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", onKeyDown, true);
    return () => document.removeEventListener("keydown", onKeyDown, true);
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
        data-testid="math-dialog"
        // Focusable so a click on the panel's text keeps focus in the dialog.
        tabIndex={-1}
        className={cx(
          "relative flex h-full w-full max-w-xl flex-col overflow-y-auto outline-none",
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
          {/* Focusable because it scrolls: a long equation overflows sideways,
              and a scroller that can't take focus can't be scrolled from the
              keyboard in Safari or Firefox. */}
          <div
            role="group"
            aria-labelledby="math-equation-heading"
            tabIndex={0}
            data-testid="math-equation"
            data-state={equationState}
            className="mt-2 overflow-x-auto rounded-md border border-border bg-surface-2 p-4"
          >
            {equationHtml === null ? (
              <p className="font-mono text-sm text-text-muted">
                Rendering the equation…
              </p>
            ) : equationHtml ? (
              // KaTeX emits MathML beside the HTML, so a screen reader reads
              // real maths here rather than backslash commands.
              <div dangerouslySetInnerHTML={{ __html: equationHtml }} />
            ) : (
              // KaTeX couldn't render it: the source is still the truth.
              <p className="font-mono text-sm break-words text-text-muted">
                {math.equation}
              </p>
            )}
          </div>
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
