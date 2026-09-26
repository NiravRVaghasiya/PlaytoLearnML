"use client"; // Error boundaries must be Client Components.

import { ErrorPanel } from "./_components/ErrorPanel";

/**
 * Route-level error UI for every page under the root layout.
 *
 * Games have their own boundary in `GameMount`, which keeps the failure inside
 * the game and names it; this catches everything else — the roster, the Concept
 * Library, and a game page's own server render — so no crash ever ends on Next's
 * unbranded "Application error" screen with no way home. It renders inside the
 * root layout, so the skip link and fonts still work.
 *
 * `retry` (Next 16.3+) re-fetches the segment and re-renders it; `reset` only
 * re-renders. Both are accepted, `retry` preferred, so a transient failure in
 * fetching the page's server payload is recoverable too.
 */
export default function RouteError({
  error,
  retry,
  reset,
}: {
  error: Error & { digest?: string };
  retry?: () => void;
  reset: () => void;
}) {
  return <ErrorPanel error={error} retry={retry ?? reset} />;
}
