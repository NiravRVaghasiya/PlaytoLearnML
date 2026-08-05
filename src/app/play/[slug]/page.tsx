import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { getGameMeta } from "@/lib/catalog";
import { isPlayable, playableSlugs } from "@/games/registry";
import { GameMount } from "@/games/GameMount";

interface PageProps {
  params: Promise<{ slug: string }>;
}

/** Pre-render a route for every playable game. */
export function generateStaticParams() {
  return playableSlugs().map((slug) => ({ slug }));
}

export async function generateMetadata({
  params,
}: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const meta = getGameMeta(slug);
  if (!meta) return { title: "Game not found" };

  return {
    title: meta.title,
    description: meta.coreIntuition,
  };
}

export default async function PlayPage({ params }: PageProps) {
  const { slug } = await params;

  // A slug in the catalog but not the registry is a game we haven't built yet.
  if (!isPlayable(slug)) notFound();

  // The skip link in the root layout targets #main-content, so it has to exist
  // in the server-rendered HTML. The game itself mounts client-side, so this
  // wrapper — not anything inside GameShell — is what provides the anchor.
  return (
    <div id="main-content" tabIndex={-1}>
      <GameMount slug={slug} />
    </div>
  );
}
