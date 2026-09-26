"use client";

import { useEffect } from "react";
import { RefreshCw, RotateCcw } from "lucide-react";
import { isChunkLoadError } from "@/lib/chunk-load-error";

/**
 * The body shared by `app/error.tsx` and `app/global-error.tsx`: what went
 * wrong in plain language, a retry, a reload for the case a retry can't fix,
 * and a way home.
 *
 * Plain `<a href="/">` rather than `next/link` for the home link on purpose. By
 * the time this renders, client-side routing may itself be what broke — a
 * chunk that failed to download, or a global error that took the root layout
 * with it — and a full navigation is the one route home that can't depend on
 * it.
 *
 * The buttons are native, not the shared `<Button>`, for the same reason:
 * global-error replaces the root layout and must not depend on anything it can
 * avoid. They copy `<Button>`'s classes, so the page still looks like GameML.
 */
export function ErrorPanel({
  error,
  retry,
  headingLevel = 1,
}: {
  error: Error & { digest?: string };
  retry: () => void;
  headingLevel?: 1 | 2;
}) {
  // Never swallowed. Next logs errors that reach a boundary in development
  // only; this makes sure production consoles (and anything reporting from
  // them) see it too.
  useEffect(() => {
    console.error("[GameML] Unhandled error:", error);
  }, [error]);

  const chunkFailed = isChunkLoadError(error);
  const Heading = headingLevel === 1 ? "h1" : "h2";

  return (
    <main
      id="main-content"
      tabIndex={-1}
      className="mx-auto max-w-2xl px-6 py-16"
    >
      <p className="font-mono text-sm text-primary">Something went wrong</p>
      <Heading className="mt-2 font-display text-3xl font-bold">
        {chunkFailed
          ? "Part of the site didn't finish downloading"
          : "This page hit an unexpected error"}
      </Heading>
      <p className="mt-4 text-lg text-text-muted">
        {chunkFailed
          ? "That usually means the connection dropped, or the site was updated while this tab was open. Reloading fetches a fresh copy."
          : "That's our bug, not something you did. Trying again usually fixes it; if not, reloading the page gives it a clean start."}
      </p>
      {error.digest ? (
        // The digest is what matches this to the server's log; it reveals
        // nothing about the error itself.
        <p className="mt-2 font-mono text-xs text-text-muted">
          Reference: {error.digest}
        </p>
      ) : null}

      <div className="mt-8 flex flex-wrap gap-3">
        {chunkFailed ? null : (
          <button
            type="button"
            onClick={retry}
            className="inline-flex min-h-11 items-center justify-center gap-2 rounded-md bg-primary px-4 font-medium text-primary-ink transition-[filter] dur-micro hover:brightness-110"
          >
            <RotateCcw aria-hidden="true" className="size-4" />
            Try again
          </button>
        )}
        <button
          type="button"
          onClick={() => window.location.reload()}
          className={
            chunkFailed
              ? "inline-flex min-h-11 items-center justify-center gap-2 rounded-md bg-primary px-4 font-medium text-primary-ink transition-[filter] dur-micro hover:brightness-110"
              : "inline-flex min-h-11 items-center justify-center gap-2 rounded-md border border-border bg-surface-2 px-4 font-medium text-text transition-[border-color] dur-micro hover:border-primary/60"
          }
        >
          <RefreshCw aria-hidden="true" className="size-4" />
          Reload the page
        </button>
        {/* eslint-disable-next-line @next/next/no-html-link-for-pages -- a full
            navigation on purpose; see the component comment. */}
        <a
          href="/"
          className="inline-flex min-h-11 items-center justify-center gap-2 rounded-md px-4 font-medium text-primary underline decoration-dotted underline-offset-4"
        >
          Back to all games
        </a>
      </div>
    </main>
  );
}
