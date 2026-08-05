import Link from "next/link";
import { Lock } from "lucide-react";
import { GAME_CATALOG, PHASES } from "@/lib/catalog";
import { isPlayable } from "@/games/registry";


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
            <ul className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {GAME_CATALOG.filter((g) => g.phase === phase).map((game) => {
                const playable = isPlayable(game.slug);

                const card = (
                  <>
                    <p className="flex items-center gap-2 font-display font-semibold">
                      {!playable ? (
                        <Lock
                          aria-hidden="true"
                          className="size-4 shrink-0 text-text-muted"
                        />
                      ) : null}
                      {game.title}
                    </p>
                    <p className="mt-1 text-sm text-text-muted">
                      {game.concept}
                    </p>
                    <p className="mt-3 font-mono text-xs text-text-muted">
                      {playable ? `Play · ${game.metricLabel}` : "Not built yet"}
                    </p>
                  </>
                );

                return (
                  <li key={game.slug}>
                    {playable ? (
                      <Link
                        href={`/play/${game.slug}`}
                        className="block h-full rounded-md border border-border bg-surface p-4 transition-colors dur-micro hover:border-primary"
                      >
                        {card}
                      </Link>
                    ) : (
                      // No `opacity` here on purpose: dimming the whole card
                      // drops --text-muted below AA contrast. The dashed border,
                      // lock icon and "Not built yet" label carry the state.
                      <div className="h-full rounded-md border border-dashed border-border bg-surface p-4">
                        {card}
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </section>
    </main>
  );
}
