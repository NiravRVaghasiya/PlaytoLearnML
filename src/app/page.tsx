import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight, BookOpen } from "lucide-react";
import {
  GAME_CATALOG,
  START_SLUG,
  getGameMeta,
  rosterByCategory,
} from "@/lib/catalog";
import { SITE_DESCRIPTION, SITE_TITLE, pageSocialMetadata } from "@/lib/site";
import {
  ProgressionHydrator,
  RosterProgress,
} from "./_components/RosterProgress";

export const metadata: Metadata = pageSocialMetadata({
  title: SITE_TITLE,
  description: SITE_DESCRIPTION,
  path: "/",
});

/**
 * The first-visit explainer. Three steps because that is the loop every game
 * shares: act in the Visual lane (pedagogy contract #2), watch the live metric
 * and the named failure (#3, #4), then drive the same game from code (the
 * two-lane rule). Nothing here is specific to one game, so none of it can go
 * stale when a game is retuned.
 */
const HOW_IT_WORKS = [
  {
    title: "Play the algorithm",
    body: "Drag, click and slide in the Visual lane. Each move you make is a step the real algorithm takes.",
  },
  {
    title: "Watch the live metric",
    body: "Accuracy, loss or inertia moves with every action. Lose, and the game names the ML mistake behind it.",
  },
  {
    title: "Switch to the Code lane",
    body: "Edit the JavaScript or Python that drives the same game, and watch the same metric answer.",
  },
] as const;

const primaryLink =
  "inline-flex min-h-11 items-center gap-2 rounded-md bg-primary px-4 font-medium text-primary-ink transition-[filter] dur-micro hover:brightness-110";

export default function Home() {
  // catalog.test.ts asserts START_SLUG is in the catalog (and is a genuinely
  // gentle first game), so this is never undefined in practice.
  const start = getGameMeta(START_SLUG);
  const groups = rosterByCategory();

  return (
    <main
      id="main-content"
      tabIndex={-1}
      className="mx-auto max-w-6xl px-6 py-12"
    >
      <ProgressionHydrator />

      <header className="mb-12">
        <p className="font-mono text-sm text-primary">The ML Continent</p>
        <h1 className="mt-2 text-3xl font-bold">GameML</h1>
        <p className="mt-3 max-w-2xl text-lg text-text-muted">
          Fourteen games that teach real Machine Learning. Every model trains in
          your browser — no server, no GPU.
        </p>

        {/* A first-time visitor's question is "which one do I open?", and a
            grid of fourteen doesn't answer it. */}
        <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-3">
          {start ? (
            <Link href={`/play/${start.slug}`} className={primaryLink}>
              Start here: {start.title}
              <ArrowRight aria-hidden="true" className="size-4" />
            </Link>
          ) : null}
          {/* The games link *into* the library from their WhyCards, so it
              needs a way in from the top level too — otherwise the only route
              to it is mid-game, and it never gets crawled. */}
          <Link
            href="/concepts"
            className="inline-flex min-h-11 items-center gap-2 text-sm font-medium text-primary underline decoration-dotted underline-offset-4"
          >
            <BookOpen aria-hidden="true" className="size-4" />
            Read the Concept Library
          </Link>
        </div>
        {start ? (
          <p className="mt-2 text-sm text-text-muted">
            {start.difficulty} level with no prerequisites — the game most of
            the others build on.
          </p>
        ) : null}
      </header>

      <section aria-labelledby="how-heading" className="mb-12">
        <h2 id="how-heading" className="text-xl font-semibold">
          How it works
        </h2>
        <ol className="mt-4 grid gap-3 sm:grid-cols-3">
          {HOW_IT_WORKS.map((step, index) => (
            <li
              key={step.title}
              className="rounded-md border border-border bg-surface p-4"
            >
              <p className="font-mono text-xs text-primary">Step {index + 1}</p>
              <h3 className="mt-1 font-display font-semibold">{step.title}</h3>
              <p className="mt-1 text-sm text-text-muted">{step.body}</p>
            </li>
          ))}
        </ol>
        <p className="mt-3 text-sm text-text-muted">
          Everything runs in your browser: no server, no GPU, no account. Your
          progress is saved on this device.
        </p>
      </section>

      <section aria-labelledby="roster-heading">
        <h2 id="roster-heading" className="text-xl font-semibold">
          All {GAME_CATALOG.length} games
        </h2>
        <p className="mt-1 text-sm text-text-muted">
          Grouped by topic, easiest first. Every game is open, so play them in
          any order.
        </p>

        {/* Grouped by category, as spec §6 asks, rather than by build phase —
            "Phase 1 — MVP" is a roadmap milestone and meant nothing to a
            learner. `rosterByCategory` owns the order; catalog.test.ts pins it. */}
        {groups.map(({ category, games }) => (
          <div key={category} className="mt-8">
            <h3 className="font-mono text-sm tracking-wide text-text-muted uppercase">
              {category}
            </h3>
            {/* Every catalog game is built, so every card is a link. That is
                an assumption, and it is the one `registry.test.ts` asserts:
                add a game to the catalog ahead of its module and the suite
                fails rather than the roster shipping a dead link. */}
            <ul className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {games.map((game) => (
                <li key={game.slug}>
                  <Link
                    href={`/play/${game.slug}`}
                    className="block h-full rounded-md border border-border bg-surface p-4 transition-colors dur-micro hover:border-primary"
                  >
                    <p className="font-display font-semibold">{game.title}</p>
                    <p className="mt-1 text-sm text-text-muted">
                      {game.concept}
                    </p>
                    <p className="mt-3 flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-xs text-text-muted">
                      <span className="rounded-full border border-border px-2 py-0.5">
                        {game.difficulty}
                      </span>
                      {game.slug === START_SLUG ? (
                        <span className="rounded-full bg-primary/15 px-2 py-0.5 text-primary">
                          Start here
                        </span>
                      ) : null}
                      <span>Live metric: {game.metricLabel}</span>
                    </p>
                    {/* Client-only: stars appear after mount, never on the
                        server, so there is nothing to mismatch. */}
                    <RosterProgress slug={game.slug} />
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </section>
    </main>
  );
}
