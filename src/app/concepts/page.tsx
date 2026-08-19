import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { CONCEPT_LIBRARY } from "@/lib/concepts";
import { getGameMeta } from "@/lib/catalog";

export const metadata: Metadata = {
  title: "Concept Library",
  description:
    "Short, plain-language explainers for the ML ideas behind the games — decision boundaries, overfitting, gradient descent, clustering and more.",
};

export default function ConceptLibraryPage() {
  return (
    <main
      id="main-content"
      tabIndex={-1}
      className="mx-auto max-w-4xl px-6 py-12"
    >
      <Link
        href="/"
        className="inline-flex items-center gap-2 text-sm text-text-muted underline decoration-dotted underline-offset-4 hover:text-text"
      >
        <ArrowLeft aria-hidden="true" className="size-4" />
        GameML
      </Link>

      <header className="mt-6 mb-10">
        <p className="font-mono text-sm text-primary">Concept Library</p>
        <h1 className="mt-2 font-display text-3xl font-bold">
          The ideas behind the games
        </h1>
        <p className="mt-3 max-w-2xl text-lg text-text-muted">
          One short explainer per concept. Each one answers a question a game
          asks you, names the failure modes you will meet, and links straight
          into the game that teaches it.
        </p>
      </header>

      <ul className="grid gap-4 sm:grid-cols-2">
        {CONCEPT_LIBRARY.map((concept) => {
          const gameTitles = concept.games
            .map((slug) => getGameMeta(slug)?.title)
            .filter((title): title is string => Boolean(title));

          return (
            <li key={concept.slug}>
              <Link
                href={`/concepts/${concept.slug}`}
                className="dur-micro block h-full rounded-md border border-border bg-surface p-5 transition-colors hover:border-primary"
              >
                <h2 className="font-display text-lg font-semibold">
                  {concept.question}
                </h2>
                <p className="mt-2 text-sm leading-relaxed text-text-muted">
                  {concept.summary}
                </p>
                {gameTitles.length > 0 ? (
                  <p className="mt-3 font-mono text-xs text-text-muted">
                    Taught in {gameTitles.join(" · ")}
                  </p>
                ) : null}
              </Link>
            </li>
          );
        })}
      </ul>
    </main>
  );
}
