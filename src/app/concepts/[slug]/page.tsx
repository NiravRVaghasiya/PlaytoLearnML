import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft, Gamepad2 } from "lucide-react";
import { CONCEPT_LIBRARY, conceptSlugs, getConcept } from "@/lib/concepts";
import { getGameMeta } from "@/lib/catalog";
import { pageSocialMetadata } from "@/lib/site";

interface PageProps {
  params: Promise<{ slug: string }>;
}

/**
 * Static at build time. The library is content, so there is nothing to render
 * per-request and every page should be a cacheable document — which is also the
 * point of it existing (spec §8, Concept Library as the SEO growth loop).
 */
export function generateStaticParams() {
  return conceptSlugs().map((slug) => ({ slug }));
}

/**
 * The whole library is known at build time, so an unknown slug is a static 404
 * rather than an on-demand render written to the ISR cache.
 */
export const dynamicParams = false;

export async function generateMetadata({
  params,
}: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const concept = getConcept(slug);
  // Defensive only: dynamicParams = false keeps unknown slugs out.
  if (!concept) return { title: "Concept not found" };

  return {
    // The layout template appends " · GameML". `question` rather than `title`
    // because the searched-for phrase is "what is overfitting", not "overfitting".
    title: concept.question,
    description: concept.summary,
    ...pageSocialMetadata({
      title: concept.question,
      description: concept.summary,
      path: `/concepts/${slug}`,
      type: "article",
    }),
  };
}

export default async function ConceptPage({ params }: PageProps) {
  const { slug } = await params;
  const concept = getConcept(slug);
  if (!concept) notFound();

  const games = concept.games
    .map((gameSlug) => getGameMeta(gameSlug))
    .filter((game): game is NonNullable<typeof game> => game !== undefined);

  const related = concept.related
    .map((relatedSlug) => CONCEPT_LIBRARY.find((c) => c.slug === relatedSlug))
    .filter((c): c is NonNullable<typeof c> => c !== undefined);

  return (
    <main
      id="main-content"
      tabIndex={-1}
      className="mx-auto max-w-3xl px-6 py-12"
    >
      <Link
        href="/concepts"
        className="inline-flex min-h-6 items-center gap-2 text-sm text-text-muted underline decoration-dotted underline-offset-4 hover:text-text"
      >
        <ArrowLeft aria-hidden="true" className="size-4" />
        Concept Library
      </Link>

      <article className="mt-6">
        <header>
          <h1 className="font-display text-3xl font-bold">{concept.title}</h1>
          <p className="mt-4 text-lg leading-relaxed text-text">
            {concept.answer}
          </p>
        </header>

        {concept.sections.map((section) => (
          <section key={section.heading} className="mt-10">
            <h2 className="font-display text-xl font-semibold">
              {section.heading}
            </h2>
            {section.body.map((paragraph, index) => (
              <p
                // Paragraphs are static content with no identity of their own;
                // the heading scopes the key, so index is stable here.
                key={`${section.heading}-${index}`}
                className="mt-3 leading-relaxed text-text-muted"
              >
                {paragraph}
              </p>
            ))}
          </section>
        ))}

        {concept.failureModes.length > 0 ? (
          <section className="mt-10">
            <h2 className="font-display text-xl font-semibold">
              How this goes wrong
            </h2>
            <p className="mt-2 text-sm text-text-muted">
              These are the names the games put on screen when you hit them, so
              the failure you read about here is the failure you will recognise
              in play.
            </p>
            <dl className="mt-4 space-y-4">
              {concept.failureModes.map((mode) => (
                <div
                  key={mode.name}
                  className="rounded-md border border-border bg-surface p-4"
                >
                  <dt className="font-display font-semibold text-warn">
                    {mode.name}
                  </dt>
                  <dd className="mt-1 text-sm leading-relaxed text-text-muted">
                    {mode.gloss}
                  </dd>
                </div>
              ))}
            </dl>
          </section>
        ) : null}

        {games.length > 0 ? (
          <section className="mt-10">
            <h2 className="font-display text-xl font-semibold">
              Play it instead of reading about it
            </h2>
            <ul className="mt-4 grid gap-3 sm:grid-cols-2">
              {games.map((game) => (
                <li key={game.slug}>
                  <Link
                    href={`/play/${game.slug}`}
                    className="dur-micro block h-full rounded-md border border-border bg-surface p-4 transition-colors hover:border-primary"
                  >
                    <p className="flex items-center gap-2 font-display font-semibold">
                      <Gamepad2
                        aria-hidden="true"
                        className="size-4 shrink-0 text-primary"
                      />
                      {game.title}
                    </p>
                    <p className="mt-1 text-sm text-text-muted">
                      {game.coreIntuition}
                    </p>
                    <p className="mt-3 font-mono text-xs text-text-muted">
                      Play · {game.metricLabel}
                    </p>
                  </Link>
                </li>
              ))}
            </ul>
          </section>
        ) : null}
      </article>

      {related.length > 0 ? (
        <nav aria-labelledby="related-heading" className="mt-12">
          <h2
            id="related-heading"
            className="font-display text-xl font-semibold"
          >
            Read next
          </h2>
          <ul className="mt-3 space-y-2">
            {related.map((other) => (
              <li key={other.slug}>
                <Link
                  href={`/concepts/${other.slug}`}
                  className="inline-flex min-h-6 items-center text-primary underline decoration-dotted underline-offset-4"
                >
                  {other.question}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      ) : null}
    </main>
  );
}
