import { existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  playableSlugsFromSource,
  siteRoutes,
} from "../../scripts/harness.mjs";
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

describe("a game module on disk is a game on the site", () => {
  // Everything above compares the registry, the catalog and the mount map with
  // EACH OTHER, so a game built under src/games/<slug>/ and never registered in
  // any of them passed the whole suite while being unreachable. These start
  // from the directory instead.
  const GAMES_DIR = join(process.cwd(), "src", "games");
  const gameDirs = readdirSync(GAMES_DIR, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .filter((name) => existsSync(join(GAMES_DIR, name, "index.tsx")))
    .sort();

  it("registers every src/games/<slug>/index.tsx", () => {
    expect(gameDirs.length, "no game modules found to check").toBeGreaterThan(0);
    expect(gameDirs).toEqual([...playableSlugs()].sort());
  });

  it("gives every playable game a browser playthrough", () => {
    // scripts/verify-playthrough.mjs is what proves a game is wired to the
    // screen; a game without one is never driven in a real browser.
    for (const slug of playableSlugs()) {
      expect(
        existsSync(join(process.cwd(), "scripts", "playthroughs", `${slug}.mjs`)),
        `scripts/playthroughs/${slug}.mjs is missing`,
      ).toBe(true);
    }
  });

  it("puts every playable game in the browser harnesses' route list", () => {
    // The a11y audit and the mobile harness read the registry's source with a
    // regex (they run under plain node). The parse has to equal the registry.
    expect(playableSlugsFromSource()).toEqual(playableSlugs());
    const routes = siteRoutes().map((route) => route.path);
    for (const slug of playableSlugs()) {
      expect(routes).toContain(`/play/${slug}`);
    }
    // ...and the harnesses always include a route that must 404.
    expect(siteRoutes().some((route) => route.kind === "not-found")).toBe(true);
  });
});
