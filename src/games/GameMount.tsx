"use client";

import dynamic from "next/dynamic";
import type { ComponentType } from "react";

/**
 * Mounts a game by slug.
 *
 * A Client Component, because each entry is a `next/dynamic` import with
 * `ssr: false` — which Next only permits on the client. That flag is not
 * incidental: games lean on WebGL, pointer capture, and `requestAnimationFrame`,
 * so there is nothing to gain from server rendering them and plenty to break.
 *
 * The loading fallback matches the shell's canvas box so the layout doesn't jump
 * while a game's chunk arrives.
 */

const GAME_COMPONENTS: Record<string, ComponentType> = {
  "sort-it-arcade": dynamic(() => import("./sort-it-arcade"), {
    ssr: false,
    loading: () => <GameLoading />,
  }),
  "k-means-territory-wars": dynamic(
    () => import("./k-means-territory-wars"),
    { ssr: false, loading: () => <GameLoading /> },
  ),
  "data-detox": dynamic(() => import("./data-detox"), {
    ssr: false,
    loading: () => <GameLoading />,
  }),
  "gradient-descent-skier": dynamic(
    () => import("./gradient-descent-skier"),
    { ssr: false, loading: () => <GameLoading /> },
  ),
  "neuron-forge": dynamic(() => import("./neuron-forge"), {
    ssr: false,
    loading: () => <GameLoading />,
  }),
  "overfit-tower-defense": dynamic(
    () => import("./overfit-tower-defense"),
    { ssr: false, loading: () => <GameLoading /> },
  ),
  "confusion-matrix-chef": dynamic(() => import("./confusion-matrix-chef"), {
    ssr: false,
    loading: () => <GameLoading />,
  }),
  "decision-tree-architect": dynamic(
    () => import("./decision-tree-architect"),
    { ssr: false, loading: () => <GameLoading /> },
  ),
  "hyperparameter-heist": dynamic(() => import("./hyperparameter-heist"), {
    ssr: false,
    loading: () => <GameLoading />,
  }),
  "feature-forge": dynamic(() => import("./feature-forge"), {
    ssr: false,
    loading: () => <GameLoading />,
  }),
  "agent-academy": dynamic(() => import("./agent-academy"), {
    ssr: false,
    loading: () => <GameLoading />,
  }),
  "convolution-kitchen": dynamic(() => import("./convolution-kitchen"), {
    ssr: false,
    loading: () => <GameLoading />,
  }),
  "backprop-blitz": dynamic(() => import("./backprop-blitz"), {
    ssr: false,
    loading: () => <GameLoading />,
  }),
  "dimension-diver": dynamic(() => import("./dimension-diver"), {
    ssr: false,
    loading: () => <GameLoading />,
  }),
};

/** Slugs this module can actually mount. Compared against the registry in tests. */
export const MOUNTED_SLUGS = Object.keys(GAME_COMPONENTS);

function GameLoading() {
  return (
    <div
      role="status"
      className="flex min-h-dvh items-center justify-center bg-bg text-text-muted"
    >
      Loading game…
    </div>
  );
}

export function GameMount({ slug }: { slug: string }) {
  const Game = GAME_COMPONENTS[slug];

  // Unreachable by design, and kept only because the index signature is
  // optional: `/play/[slug]` calls notFound() for anything outside the registry,
  // and registry.test.ts asserts MOUNTED_SLUGS and playableSlugs() are the same
  // set. Reaching here means those two drifted apart, which is a bug in the
  // wiring rather than a state to render an apology for — so it throws instead
  // of shipping a "not built yet" screen no test can exercise.
  if (!Game) {
    throw new Error(
      `GameMount: no module registered for "${slug}". MOUNTED_SLUGS and the registry have drifted.`,
    );
  }

  return <Game />;
}
