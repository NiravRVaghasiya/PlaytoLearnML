import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { getGameMeta } from "@/lib/catalog";
import { pageSocialMetadata } from "@/lib/site";
import { isPlayable, playableSlugs } from "@/games/registry";
import { GameMount } from "@/games/GameMount";

interface PageProps {
  params: Promise<{ slug: string }>;
}

/** Applied only with scripting off; hides GameMount's `data-game-loading` box. */
const NOSCRIPT_CSS = "[data-game-loading]{display:none}";

/** Pre-render a route for every playable game. */
export function generateStaticParams() {
  return playableSlugs().map((slug) => ({ slug }));
}

/**
 * Every valid slug is known at build time (above), so anything else is a 404 —
 * served as the static not-found page. With the default (`true`), each unknown
 * slug a crawler or a typo produced was rendered on demand by a server function
 * and written to the ISR cache, only to say "not found".
 */
export const dynamicParams = false;

export async function generateMetadata({
  params,
}: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const meta = getGameMeta(slug);
  // Defensive only: with dynamicParams = false an unknown slug never reaches
  // this function.
  if (!meta) return { title: "Game not found" };

  return {
    title: meta.title,
    description: meta.coreIntuition,
    ...pageSocialMetadata({
      title: meta.title,
      description: meta.coreIntuition,
      path: `/play/${slug}`,
    }),
  };
}

export default async function PlayPage({ params }: PageProps) {
  const { slug } = await params;

  // Defensive, like the metadata guard: `dynamicParams = false` already 404s
  // any slug outside generateStaticParams, and registry.test.ts keeps the
  // registry and the mount map in step. Kept so the page can never render a
  // shell around nothing if those two ever drift.
  if (!isPlayable(slug)) notFound();

  // The skip link in the root layout targets #main-content, so it has to exist
  // in the server-rendered HTML. The game itself mounts client-side, so this
  // wrapper — not anything inside GameShell — is what provides the anchor.
  //
  // No `tabIndex` on purpose. A focusable wrapper around the whole game steals
  // focus on every click on plain text inside it: the next Tab then restarts
  // at the header's "Back" link, and a click inside the Math dialog moved
  // focus OUT of the dialog. The skip link doesn't need it — once the game has
  // mounted it focuses GameShell's canvas directly (see SkipLink), and before
  // that, plain fragment navigation moves the Tab starting point here.
  const meta = getGameMeta(slug);

  return (
    <div id="main-content">
      {/* Without JavaScript the game can never mount (all of its ML runs in
          the browser), and the page used to sit on "Loading game…" forever.
          This says why, names the game and what it teaches, and offers the
          one part of the site that works without scripts. It is also the only
          server-rendered text on a game page, so it is what a non-rendering
          crawler reads. */}
      <noscript>
        {/* GameMount server-renders its "Loading game…" status as well, and
            without scripts it would sit under this explanation announcing a
            game that can never load. Only a browser with scripting off parses
            this as a stylesheet. The selector is GameLoading's attribute. */}
        <style>{NOSCRIPT_CSS}</style>
        <main className="mx-auto max-w-2xl px-6 py-12">
          <h1 className="font-display text-2xl font-bold">{meta?.title}</h1>
          <p className="mt-3 text-lg text-text-muted">{meta?.coreIntuition}</p>
          <p className="mt-4 text-text-muted">
            This game runs its machine learning in your browser, so it needs
            JavaScript. Turn it on and reload, or read the{" "}
            <Link
              href="/concepts"
              className="text-primary underline decoration-dotted underline-offset-4"
            >
              Concept Library
            </Link>
            , which works without it.
          </p>
        </main>
      </noscript>
      <GameMount slug={slug} />
    </div>
  );
}
