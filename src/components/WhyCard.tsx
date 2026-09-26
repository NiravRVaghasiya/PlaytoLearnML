"use client";

import { AlertTriangle, CheckCircle2, Info, XCircle } from "lucide-react";
import Link from "next/link";
import { cx } from "@/lib/utils";

export type WhyTone = "info" | "good" | "bad" | "warn";

export interface WhyCardContent {
  /**
   * Stable key for the action that produced this card. Changing it re-triggers
   * the fade-in, which is how the player knows the explanation is *new*.
   */
  key: string;
  /** Short headline. On a loss this MUST be the named failure mode. */
  title: string;
  /** Two lines of plain language. No jargon without a gloss. */
  body: string;
  tone?: WhyTone;
  /** Deep link into the Concept Library (spec §6). */
  conceptHref?: string;
  conceptLabel?: string;
}

export interface WhyCardProps {
  content: WhyCardContent | null;
  className?: string;
}

const TONE: Record<
  WhyTone,
  { border: string; text: string; Icon: typeof Info; srPrefix: string }
> = {
  info: {
    border: "border-border",
    text: "text-primary",
    Icon: Info,
    srPrefix: "Explanation",
  },
  good: {
    border: "border-correct/40",
    text: "text-correct",
    Icon: CheckCircle2,
    srPrefix: "Good news",
  },
  bad: {
    border: "border-wrong/50",
    text: "text-wrong",
    Icon: XCircle,
    srPrefix: "Failure",
  },
  warn: {
    border: "border-warn/50",
    text: "text-warn",
    Icon: AlertTriangle,
    srPrefix: "Warning",
  },
};

/**
 * The one line a screen reader should hear when this card changes: its tone
 * prefix and headline, e.g. "Failure: You overfit". Deliberately not the body —
 * the body is two sentences, and reading it after every move would bury the
 * metric. `GameShell` announces this politely whenever the card's key changes.
 */
export function whyCardHeadline(content: WhyCardContent): string {
  return `${TONE[content.tone ?? "info"].srPrefix}: ${content.title}`;
}

/**
 * "Why did that happen?" — the 2-line explanation tied to the player's last
 * action (DESIGN.md §6, spec §4).
 *
 * This is the component that turns a metric move into a lesson. It is also
 * where a loss is explained: on failure the game passes `tone="bad"` and a
 * `title` naming the real ML failure ("You overfit — 99% train, 61% test"),
 * which is how the pedagogy contract's fourth point is met on screen.
 *
 * Tone is carried by an icon and a screen-reader prefix as well as colour, so
 * it survives colourblindness and greyscale (DESIGN.md §9).
 */
export function WhyCard({ content, className }: WhyCardProps) {
  if (!content) {
    return (
      <div
        className={cx(
          "rounded-md border border-dashed border-border bg-surface/60 p-4",
          className,
        )}
      >
        <p className="text-sm text-text-muted">
          Make a move — the explanation for what happens lands here.
        </p>
      </div>
    );
  }

  const tone = TONE[content.tone ?? "info"];
  const { Icon } = tone;

  return (
    <div
      // Re-keying on the action restarts the fade-in animation.
      key={content.key}
      className={cx(
        "why-card-in rounded-md border bg-surface p-4",
        tone.border,
        className,
      )}
    >
      <div className="flex gap-3">
        <Icon
          aria-hidden="true"
          className={cx("mt-0.5 size-5 shrink-0", tone.text)}
          strokeWidth={2}
        />
        <div className="min-w-0">
          <h3 className={cx("font-display text-base font-semibold", tone.text)}>
            <span className="sr-only-live">{tone.srPrefix}: </span>
            {content.title}
          </h3>
          <p className="mt-1 text-sm leading-relaxed text-text">
            {content.body}
          </p>
          {content.conceptHref ? (
            <Link
              href={content.conceptHref}
              // min-h-11: a standalone link, not one inside a sentence, so the
              // WCAG 2.5.8 inline exception does not cover it (DESIGN.md §9).
              className="mt-2 inline-flex min-h-11 items-center text-sm font-medium text-primary underline decoration-dotted underline-offset-4"
            >
              {content.conceptLabel ?? "Read the concept"}
            </Link>
          ) : null}
        </div>
      </div>
    </div>
  );
}
