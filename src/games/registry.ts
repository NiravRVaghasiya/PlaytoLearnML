/**
 * Which games are actually playable.
 *
 * Deliberately free of React and of `next/dynamic`: this module is imported by
 * Server Components (the home roster, the `/play/[slug]` route), and
 * `next/dynamic` with `ssr: false` is not allowed there. The component map lives
 * in `GameMount.tsx`, which is a Client Component.
 *
 * `src/lib/catalog.ts` lists all 14 games for the roster; this list is what's
 * built. `registry.test.ts` checks both directions so an unbuilt game can never
 * present a dead link, and a built game can never be missing its metadata.
 */
export const PLAYABLE_SLUGS = [
  "sort-it-arcade",
  "k-means-territory-wars",
  "data-detox",
  "gradient-descent-skier",
  "neuron-forge",
  "overfit-tower-defense",
  "confusion-matrix-chef",
  "decision-tree-architect",
] as const;

export type PlayableSlug = (typeof PLAYABLE_SLUGS)[number];

export function isPlayable(slug: string): slug is PlayableSlug {
  return (PLAYABLE_SLUGS as readonly string[]).includes(slug);
}

export function playableSlugs(): string[] {
  return [...PLAYABLE_SLUGS];
}
