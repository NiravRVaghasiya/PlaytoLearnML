import { describe, expect, it } from "vitest";
import {
  GAME_CATALOG,
  PHASE_1_BUILD_ORDER,
  getGameMeta,
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
});
