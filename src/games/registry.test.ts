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
    expect(isPlayable("sort-it-arcade")).toBe(true);
    expect(isPlayable("dimension-diver")).toBe(false);
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

  it("leaves the unbuilt games unregistered rather than broken", () => {
    const registered = new Set(playableSlugs());
    const unbuilt = GAME_CATALOG.filter((g) => !registered.has(g.slug));
    // Sanity: this is a work in progress, so most of the catalog is unbuilt.
    expect(unbuilt.length).toBe(GAME_CATALOG.length - registered.size);
    expect(unbuilt.length).toBeGreaterThan(0);
    for (const game of unbuilt) {
      expect(MOUNTED_SLUGS).not.toContain(game.slug);
    }
  });
});
