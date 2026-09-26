import { describe, expect, it } from "vitest";
import {
  CATEGORIES,
  DIFFICULTIES,
  GAME_CATALOG,
  PHASE_1_BUILD_ORDER,
  START_SLUG,
  getGameMeta,
  rosterByCategory,
  type GameMeta,
} from "./catalog";

describe("catalog", () => {
  it("holds all 14 spec games", () => {
    expect(GAME_CATALOG).toHaveLength(14);
  });

  it("uses unique kebab-case slugs", () => {
    const slugs = GAME_CATALOG.map((g) => g.slug);
    expect(new Set(slugs).size).toBe(slugs.length);
    for (const slug of slugs) {
      expect(slug).toMatch(/^[a-z0-9]+(-[a-z0-9]+)*$/);
    }
  });

  it("gives every game a core intuition, metric, and named failure mode", () => {
    // Pedagogy contract points 1, 3 and 4 must be declared for every game
    // before it can be built. Missing copy here is a build blocker.
    for (const g of GAME_CATALOG) {
      expect(g.coreIntuition.length, `${g.slug} coreIntuition`).toBeGreaterThan(
        20,
      );
      expect(g.metricLabel.length, `${g.slug} metricLabel`).toBeGreaterThan(0);
      expect(g.failureMode.length, `${g.slug} failureMode`).toBeGreaterThan(0);
    }
  });

  it("only references prerequisites that exist and are not later-phase", () => {
    for (const g of GAME_CATALOG) {
      for (const req of g.requires) {
        const dep = getGameMeta(req);
        expect(dep, `${g.slug} requires unknown ${req}`).toBeDefined();
        expect(dep!.phase, `${g.slug} requires later-phase ${req}`).toBeLessThanOrEqual(
          g.phase,
        );
      }
    }
  });

  it("keeps the Phase-1 MVP to the four spec games", () => {
    const phase1 = GAME_CATALOG.filter((g) => g.phase === 1).map((g) => g.slug);
    expect(phase1.sort()).toEqual([...PHASE_1_BUILD_ORDER].sort());
  });

  it("sends a first-time visitor to a game that is genuinely a first game", () => {
    // The home page's "Start here" links to START_SLUG. Held to every axis the
    // catalog records, so an edit that makes it a bad first game fails here
    // instead of quietly sending beginners into the deep end.
    const start = getGameMeta(START_SLUG);
    expect(start, "START_SLUG is not in the catalog").toBeDefined();
    expect(start!.difficulty).toBe("Beginner");
    expect(start!.complexity).toBe("Easy");
    expect(start!.requires).toEqual([]);
    expect(PHASE_1_BUILD_ORDER[0]).toBe(START_SLUG);

    // ...and more of the unlock graph depends on it than on any other game.
    const dependents = (slug: string) =>
      GAME_CATALOG.filter((g) => g.requires.includes(slug)).length;
    const most = Math.max(...GAME_CATALOG.map((g) => dependents(g.slug)));
    expect(dependents(START_SLUG)).toBe(most);
  });
});

describe("rosterByCategory", () => {
  it("lists every game exactly once", () => {
    const listed = rosterByCategory().flatMap((group) =>
      group.games.map((g) => g.slug),
    );
    expect([...listed].sort()).toEqual(GAME_CATALOG.map((g) => g.slug).sort());
  });

  it("groups by category in the spec's order and never shows an empty heading", () => {
    const groups = rosterByCategory();
    // Every category a game uses is a known, ordered category...
    for (const game of GAME_CATALOG) {
      expect(CATEGORIES, `${game.slug} has an unordered category`).toContain(
        game.category,
      );
    }
    // ...and the groups come out in exactly that order.
    const order = groups.map((g) => CATEGORIES.indexOf(g.category));
    expect(order).toEqual([...order].sort((a, b) => a - b));
    for (const group of groups) {
      expect(group.games.length, `${group.category} is empty`).toBeGreaterThan(0);
      for (const game of group.games) expect(game.category).toBe(group.category);
    }
  });

  it("puts easier games first within a category, keeping catalog order on ties", () => {
    for (const group of rosterByCategory()) {
      const ranks = group.games.map((g) => DIFFICULTIES.indexOf(g.difficulty));
      expect(ranks, `${group.category} is not easiest-first`).toEqual(
        [...ranks].sort((a, b) => a - b),
      );
    }
    // A tie keeps catalog order. Built so that sorting by anything else (title,
    // slug) would flip it, and with an easier game listed last so the
    // difficulty sort has to move something.
    const base = GAME_CATALOG[0]!;
    const tie: GameMeta[] = [
      { ...base, slug: "zz-first", title: "Zed", difficulty: "Advanced" },
      { ...base, slug: "aa-second", title: "Alpha", difficulty: "Advanced" },
      { ...base, slug: "mm-easy", title: "Mid", difficulty: "Beginner" },
    ];
    const [group] = rosterByCategory(tie);
    expect(group!.games.map((g) => g.slug)).toEqual([
      "mm-easy",
      "zz-first",
      "aa-second",
    ]);
  });

  it("drops a category no game uses", () => {
    const onlyOne = GAME_CATALOG.filter((g) => g.category === "Neural Networks");
    const groups = rosterByCategory(onlyOne);
    expect(groups.map((g) => g.category)).toEqual(["Neural Networks"]);
  });
});
