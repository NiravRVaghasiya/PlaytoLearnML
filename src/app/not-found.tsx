import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft, BookOpen } from "lucide-react";
import { getGameMeta, START_SLUG } from "@/lib/catalog";

/**
 * The site's 404. Rendered for any unmatched URL and for unknown `/play/<slug>`
 * and `/concepts/<slug>` (both routes set `dynamicParams = false`, so these are
 * served statically with a real 404 status).
 *
 * Next's default had no main landmark, nothing for the layout's skip link to
 * land on, and no way back — a mistyped game link was a dead end. This keeps
 * the three things a lost visitor needs: what happened, the way home, and the
 * one place worth going if they don't know which game they wanted.
 */

export const metadata: Metadata = {
  title: "Page not found",
  // Next adds `noindex` to 404 responses itself; saying so here as well keeps
  // the page out of search results even if it is ever served with a 200.
  robots: { index: false },
};

export default function NotFound() {
  const start = getGameMeta(START_SLUG);

  return (
    <main
      id="main-content"
      tabIndex={-1}
      className="mx-auto max-w-2xl px-6 py-16"
    >
      <p className="font-mono text-sm text-primary">404</p>
      <h1 className="mt-2 font-display text-3xl font-bold">
        That page isn&apos;t here
      </h1>
      <p className="mt-4 text-lg text-text-muted">
        The link may be mistyped, or the page may have moved. Every game and
        every explainer is one click from the pages below.
      </p>

      <ul className="mt-8 flex flex-wrap gap-3">
        <li>
          <Link
            href="/"
            className="inline-flex min-h-11 items-center gap-2 rounded-md bg-primary px-4 font-medium text-primary-ink transition-[filter] dur-micro hover:brightness-110"
          >
            <ArrowLeft aria-hidden="true" className="size-4" />
            All games
          </Link>
        </li>
        <li>
          <Link
            href="/concepts"
            className="inline-flex min-h-11 items-center gap-2 rounded-md border border-border bg-surface-2 px-4 font-medium text-text transition-[border-color] dur-micro hover:border-primary/60"
          >
            <BookOpen aria-hidden="true" className="size-4" />
            Concept Library
          </Link>
        </li>
        {start ? (
          <li>
            <Link
              href={`/play/${start.slug}`}
              className="inline-flex min-h-11 items-center gap-2 rounded-md px-4 font-medium text-primary underline decoration-dotted underline-offset-4"
            >
              New here? Start with {start.title}
            </Link>
          </li>
        ) : null}
      </ul>
    </main>
  );
}
