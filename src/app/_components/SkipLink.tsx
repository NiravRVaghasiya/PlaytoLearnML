"use client";

import type { MouseEvent } from "react";

/**
 * DESIGN.md §9 — keyboard users get a way past the chrome.
 *
 * The link has to work before any JavaScript runs, so its `href` targets
 * `#main-content`, which every route renders on the SERVER. Pointing the href
 * at the game canvas looked right once and was broken: games mount via
 * `dynamic(ssr: false)`, so the canvas doesn't exist when the page loads.
 *
 * But on a game page `#main-content` is a wrapper around the whole game, the
 * GameShell header included, so following the href alone lands on "Back" and
 * skips nothing. Hence the click handler: once a game has mounted, activating
 * the link moves focus to GameShell's `<main id="game-canvas">` (focusable via
 * `tabIndex={-1}`, asserted in GameShell.test.tsx) — past the header. Before a
 * game mounts, and on every other page, it behaves as a plain in-page link.
 *
 * Styling uses Tailwind's `sr-only` / `focus:not-sr-only` pair, which are built
 * to undo each other. The project's `.sr-only-live` helper is NOT
 * interchangeable: `not-sr-only` doesn't reset its `clip-path`, so the link
 * would stay clipped while focused.
 */
export function SkipLink() {
  const onClick = (event: MouseEvent<HTMLAnchorElement>) => {
    const target =
      document.getElementById("game-canvas") ??
      document.getElementById("main-content");
    // Only take over when the target can actually hold focus; otherwise the
    // browser's own fragment navigation (which moves the sequential-focus
    // starting point even to an unfocusable element) is the better behaviour.
    if (!target || !target.hasAttribute("tabindex")) return;
    event.preventDefault();
    target.focus();
  };

  return (
    <a
      href="#main-content"
      onClick={onClick}
      className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:rounded-md focus:bg-primary focus:px-4 focus:py-2 focus:text-primary-ink"
    >
      Skip to content
    </a>
  );
}
