import { describe, expect, it } from "vitest";
import { GAME_CATALOG, getGameMeta } from "@/lib/catalog";
import { MOUNTED_SLUGS } from "./GameMount";
import { isPlayable, playableSlugs } from "./registry";

describe("game registry", () => {
  it("only registers slugs that exist in the catalog", () => {
    // A registry entry with no catalog metadata would render a game with no
    // title, concept, or core intuition.
    for (const slug of playableSlugs()) {
      expect(getGameMeta(slug), `no catalog entry for ${slug}`).toBeDefined();
    }
  });

  it("reports playability honestly", () => {
    // All 14 catalog games are built, so the registry should say yes to every
    // one of them — and still say no to a slug that isn't a game at all.
    for (const game of GAME_CATALOG) {
      expect(isPlayable(game.slug), `${game.slug} should be playable`).toBe(
        true,
      );
    }
    expect(isPlayable("not-a-game")).toBe(false);
  });

  it("respects the phase roadmap — no phase started before the previous one finished", () => {
    // CLAUDE.md: "Don't start a Phase 3 game while Phase 1 is incomplete."
    //
    // Stated as a rule rather than a hardcoded phase number, so it keeps holding
    // as later phases land: for every phase that has any game built, every
    // earlier phase must be built in full.
    const built = new Set(playableSlugs());
    const startedPhases = new Set(
      playableSlugs().map((slug) => getGameMeta(slug)!.phase),
    );

    for (const phase of startedPhases) {
      for (const game of GAME_CATALOG) {
        if (game.phase < phase) {
          expect(
            built.has(game.slug),
            `phase ${phase} is underway but ${game.slug} (phase ${game.phase}) is not built`,
          ).toBe(true);
        }
      }
    }
  });

  it("keeps the server-side list and the client mount map in sync", () => {
    // The list is split across two modules because next/dynamic with
    // ssr: false is illegal in a Server Component. If they drift, a route
    // passes validation and then fails to mount anything.
    expect([...MOUNTED_SLUGS].sort()).toEqual(playableSlugs().sort());
  });

  it("registers the whole catalog, and never mounts what it hasn't registered", () => {
    const registered = new Set(playableSlugs());
    const unbuilt = GAME_CATALOG.filter((g) => !registered.has(g.slug));
    // The catalog is now complete, so nothing should be left behind. Named
    // rather than counted so a regression says *which* game went missing.
    expect(unbuilt.map((g) => g.slug)).toEqual([]);
    // Kept for the next game added to the catalog ahead of its implementation:
    // an unregistered game must be absent from the mount map too, so the route
    // refuses it cleanly instead of rendering a shell around nothing.
    for (const game of unbuilt) {
      expect(MOUNTED_SLUGS).not.toContain(game.slug);
    }
  });
});
