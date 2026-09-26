"use client";

import { Info, Star } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";
import { cx } from "@/lib/utils";
import type { StarCount } from "@/engine/types";

export interface StarRatingProps {
  earned: StarCount;
  max?: number;
  size?: "sm" | "md";
  /** What each star requires, in order. Shown on demand, not only on hover. */
  criteria?: readonly string[];
  className?: string;
}

/**
 * Mastery stars (DESIGN.md §6, spec §4): 1 = completed, 2 = high score,
 * 3 = cleared the code-lane challenge.
 *
 * Filled vs empty is carried by the icon's fill AND the accessible label, not
 * by colour alone. Newly earned stars pop once; the animation is decorative and
 * disabled under `prefers-reduced-motion` by the global rule in globals.css.
 *
 * What each star requires used to live only in a `title` tooltip, which touch
 * and keyboard users can't open. It is now behind a small disclosure button
 * that works for every input, and each criterion says whether it is earned in
 * words — so the list is useful to a screen reader too, not just the icons.
 * The hover tooltip stays as a shortcut for mouse users.
 *
 * Empty stars are drawn in --text-muted (about 7:1 on the footer), not
 * --border: at 1.4:1 the empty ones were close to invisible, so "1 star" read
 * as the whole rating instead of "1 of 3".
 */
export function StarRating({
  earned,
  max = 3,
  size = "md",
  criteria,
  className,
}: StarRatingProps) {
  const previousRef = useRef(earned);
  const [popIndex, setPopIndex] = useState<number | null>(null);
  const [open, setOpen] = useState(false);
  const listId = useId();

  useEffect(() => {
    if (earned > previousRef.current) {
      setPopIndex(earned - 1);
      const timer = setTimeout(() => setPopIndex(null), 400);
      previousRef.current = earned;
      return () => clearTimeout(timer);
    }
    previousRef.current = earned;
  }, [earned]);

  const dimension = size === "sm" ? "size-4" : "size-5";
  const hasCriteria = Boolean(criteria && criteria.length > 0);

  return (
    <div className={cx("inline-flex flex-col items-start gap-1", className)}>
      <div className="inline-flex items-center gap-1">
        <div
          role="img"
          aria-label={`${earned} of ${max} mastery stars earned`}
          className="inline-flex items-center gap-1"
          title={criteria?.join(" · ")}
        >
          {Array.from({ length: max }, (_, index) => {
            const filled = index < earned;
            return (
              <Star
                key={index}
                aria-hidden="true"
                className={cx(
                  dimension,
                  filled ? "text-star" : "text-text-muted",
                  popIndex === index ? "star-pop" : null,
                )}
                fill={filled ? "currentColor" : "none"}
                strokeWidth={2}
              />
            );
          })}
        </div>

        {hasCriteria ? (
          <button
            type="button"
            aria-label="What earns each star"
            aria-expanded={open}
            aria-controls={open ? listId : undefined}
            onClick={() => setOpen((value) => !value)}
            className={cx(
              "inline-flex size-11 items-center justify-center rounded-md",
              "text-text-muted transition-colors dur-micro hover:bg-surface-2 hover:text-text",
            )}
          >
            <Info aria-hidden="true" className="size-4" />
          </button>
        ) : null}
      </div>

      {hasCriteria && open ? (
        <ol id={listId} className="max-w-72 space-y-1 text-xs text-text-muted">
          {criteria!.map((criterion, index) => {
            const done = index < earned;
            return (
              <li key={index} className="flex items-start gap-1.5">
                <Star
                  aria-hidden="true"
                  className={cx(
                    "mt-px size-3.5 shrink-0",
                    done ? "text-star" : "text-text-muted",
                  )}
                  fill={done ? "currentColor" : "none"}
                  strokeWidth={2}
                />
                <span>
                  <span className={done ? "text-text" : undefined}>{criterion}</span>
                  <span className="sr-only-live">
                    {done ? " (earned)" : " (not yet)"}
                  </span>
                </span>
              </li>
            );
          })}
        </ol>
      ) : null}
    </div>
  );
}
