"use client";

import { useEffect, useSyncExternalStore } from "react";
import { StarRating } from "@/components/StarRating";
// Straight from the module, not the `@/engine` barrel: the barrel re-exports
// useModel, and the home page must not pull TF.js in to draw a few stars.
import { useProgression } from "@/engine/progression";

/**
 * Per-game progress on the home roster (spec §4: mastery stars), read from the
 * progression service.
 *
 * Progress lives in the player's localStorage, which the server cannot see, so
 * the rule that keeps this hydration-safe is: render NOTHING until after mount.
 * The server HTML and the first client render are both empty; stars appear on
 * the next render, once the store has read storage. A card the player has not
 * cleared stays exactly as the server drew it.
 */

const noopSubscribe = () => () => {};

/**
 * `false` on the server and during hydration, `true` afterwards — without the
 * extra render a `useState` + `useEffect` mount flag costs.
 */
function useHasMounted(): boolean {
  return useSyncExternalStore(
    noopSubscribe,
    () => true,
    () => false,
  );
}

/**
 * Loads saved progress once for the whole roster. Rendered a single time on the
 * page, rather than once per card, so fourteen cards do not mean fourteen reads
 * of storage.
 */
export function ProgressionHydrator() {
  const hydrated = useProgression((s) => s.hydrated);
  const hydrate = useProgression((s) => s.hydrate);

  useEffect(() => {
    if (!hydrated) void hydrate();
  }, [hydrated, hydrate]);

  return null;
}

/** Stars and a "Cleared" label for a completed game; nothing otherwise. */
export function RosterProgress({ slug }: { slug: string }) {
  const mounted = useHasMounted();
  const hydrated = useProgression((s) => s.hydrated);
  const progress = useProgression((s) => s.games[slug]);

  if (!mounted || !hydrated || !progress?.completed) return null;

  return (
    // A <div>: StarRating renders one, and a <div> cannot sit inside a <span>.
    <div className="mt-3 flex items-center gap-2 text-xs text-text-muted">
      {/* The accessible name ("2 of 3 mastery stars earned") comes from
          StarRating, and joins the card link's name. */}
      <StarRating earned={progress.stars} size="sm" />
      <span>Cleared</span>
    </div>
  );
}
