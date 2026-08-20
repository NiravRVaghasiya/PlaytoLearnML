import Link from "next/link";
import { BookOpen } from "lucide-react";
import { GAME_CATALOG, PHASES } from "@/lib/catalog";

export default function Home() {
  return (
    <main
      id="main-content"
      tabIndex={-1}
      className="mx-auto max-w-6xl px-6 py-12"
    >
      <header className="mb-12">
        <p className="font-mono text-sm text-primary">The ML Continent</p>
        <h1 className="mt-2 text-3xl font-bold">GameML</h1>
        <p className="mt-3 max-w-2xl text-lg text-text-muted">
          Fourteen games that teach real Machine Learning. Every model trains in
          your browser — no server, no GPU.
        </p>
        {/* The games link *into* the library from their WhyCards, so it needs a
            way in from the top level too — otherwise the only route to it is
            mid-game, and it never gets crawled. */}
        <p className="mt-4">
          <Link
            href="/concepts"
            className="inline-flex items-center gap-2 text-sm font-medium text-primary underline decoration-dotted underline-offset-4"
          >
            <BookOpen aria-hidden="true" className="size-4" />
            Read the Concept Library
          </Link>
        </p>
      </header>

      <section aria-labelledby="roster-heading">
        <h2 id="roster-heading" className="text-xl font-semibold">
          Game roster
        </h2>

        {PHASES.map((phase) => (
          <div key={phase} className="mt-8">
            <h3 className="font-mono text-sm tracking-wide text-text-muted uppercase">
              Phase {phase}
              {phase === 1 ? " — MVP" : null}
            </h3>
            {/* Every catalog game is built, so every card is a link. That is an
                assumption, and it is the one `registry.test.ts` asserts: add a
                game to the catalog ahead of its module and the suite fails
                rather than the roster shipping a dead link. */}
            <ul className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {GAME_CATALOG.filter((g) => g.phase === phase).map((game) => (
                <li key={game.slug}>
                  <Link
                    href={`/play/${game.slug}`}
                    className="block h-full rounded-md border border-border bg-surface p-4 transition-colors dur-micro hover:border-primary"
                  >
                    <p className="font-display font-semibold">{game.title}</p>
                    <p className="mt-1 text-sm text-text-muted">
                      {game.concept}
                    </p>
                    <p className="mt-3 font-mono text-xs text-text-muted">
                      Play · {game.metricLabel}
                    </p>
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
