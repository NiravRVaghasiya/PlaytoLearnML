"use client";

import { Star } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { cx } from "@/lib/utils";
import type { StarCount } from "@/engine/types";

export interface StarRatingProps {
  earned: StarCount;
  max?: number;
  size?: "sm" | "md";
  /** Tooltip/label per star, e.g. what each one requires. */
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

  return (
    <div
      role="img"
      aria-label={`${earned} of ${max} mastery stars earned`}
      className={cx("inline-flex items-center gap-1", className)}
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
              filled ? "text-star" : "text-border",
              popIndex === index ? "star-pop" : null,
            )}
            fill={filled ? "currentColor" : "none"}
            strokeWidth={2}
          />
        );
      })}
    </div>
  );
}
