import { beforeEach, describe, expect, it } from "vitest";
import { GAME_CATALOG } from "@/lib/catalog";
import {
  BASE_XP,
  CODE_LANE_MULTIPLIER,
  CONCEPT_BADGES,
  EMPTY_PROGRESSION,
  HIGH_SCORE_THRESHOLD,
  applyResult,
  badgeLabel,
  createMemoryAdapter,
  isUnlocked,
  levelFromXp,
  starsFor,
  unlockedSlugs,
  useProgression,
  xpForScore,
  xpToClearLevel,
  type ProgressionState,
} from "./progression";

describe("xpForScore", () => {
  it("scales with score", () => {
    expect(xpForScore(0, "visual")).toBe(0);
    expect(xpForScore(0.5, "visual")).toBe(BASE_XP / 2);
    expect(xpForScore(1, "visual")).toBe(BASE_XP);
  });

  it("pays a premium for the code lane (spec §4)", () => {
    expect(xpForScore(1, "code")).toBe(BASE_XP * CODE_LANE_MULTIPLIER);
    expect(xpForScore(1, "code")).toBeGreaterThan(xpForScore(1, "visual"));
  });

  it("clamps out-of-range scores", () => {
    expect(xpForScore(-3, "visual")).toBe(0);
    expect(xpForScore(9, "visual")).toBe(BASE_XP);
  });
});

describe("levelFromXp", () => {
  it("starts at level 1 with no XP", () => {
    expect(levelFromXp(0)).toEqual({
      level: 1,
      xpIntoLevel: 0,
      xpForNextLevel: xpToClearLevel(1),
    });
  });

  it("levels up exactly on the threshold", () => {
    const first = xpToClearLevel(1);
    expect(levelFromXp(first - 1).level).toBe(1);
    expect(levelFromXp(first).level).toBe(2);
    expect(levelFromXp(first).xpIntoLevel).toBe(0);
  });

  it("keeps the remainder inside the current level", () => {
    const info = levelFromXp(xpToClearLevel(1) + 40);
    expect(info.level).toBe(2);
    expect(info.xpIntoLevel).toBe(40);
    expect(info.xpForNextLevel).toBe(xpToClearLevel(2));
  });

  it("never reports a remainder beyond the level requirement", () => {
    for (let xp = 0; xp < 3000; xp += 37) {
      const info = levelFromXp(xp);
      expect(info.xpIntoLevel).toBeLessThan(info.xpForNextLevel);
      expect(info.xpIntoLevel).toBeGreaterThanOrEqual(0);
    }
  });
});

describe("starsFor", () => {
  it("gives nothing until the game is completed", () => {
    expect(
      starsFor({ completed: false, bestScore: 1, codeLaneCleared: true }),
    ).toBe(0);
  });

  it("gives one star for completion at a low score", () => {
    expect(
      starsFor({ completed: true, bestScore: 0.2, codeLaneCleared: false }),
    ).toBe(1);
  });

  it("gives two stars at the high-score threshold", () => {
    expect(
      starsFor({
        completed: true,
        bestScore: HIGH_SCORE_THRESHOLD,
        codeLaneCleared: false,
      }),
    ).toBe(2);
  });

  it("gives three only when the code lane is also cleared", () => {
    expect(
      starsFor({ completed: true, bestScore: 0.95, codeLaneCleared: true }),
    ).toBe(3);
    // Code lane cleared but score too low is still one star — stars are
    // cumulative, so you can't skip the second.
    expect(
      starsFor({ completed: true, bestScore: 0.3, codeLaneCleared: true }),
    ).toBe(1);
  });
});

describe("CONCEPT_BADGES", () => {
  it("covers every game in the catalog", () => {
    // Guards the one duplication in the codebase: badges are keyed by slug in
    // progression.ts while the roster lives in catalog.ts.
    for (const game of GAME_CATALOG) {
      expect(CONCEPT_BADGES[game.slug], `missing badge for ${game.slug}`).toBeDefined();
    }
  });

  it("has no badges for games that don't exist", () => {
    const slugs = new Set(GAME_CATALOG.map((g) => g.slug));
    for (const key of Object.keys(CONCEPT_BADGES)) {
      expect(slugs.has(key), `badge for unknown game ${key}`).toBe(true);
    }
  });

  it("renders a shareable label", () => {
    expect(badgeLabel("overfit-tower-defense")).toBe(
      "I understand overfitting",
    );
    expect(badgeLabel("not-a-game")).toBeNull();
  });
});

describe("applyResult", () => {
  it("awards XP and a first star for a completed run", () => {
    const applied = applyResult(EMPTY_PROGRESSION, {
      slug: "sort-it-arcade",
      score: 0.5,
      lane: "visual",
      completed: true,
    });

    expect(applied.xpGained).toBe(50);
    expect(applied.state.xp).toBe(50);
    expect(applied.stars).toBe(1);
    expect(applied.state.games["sort-it-arcade"]?.playCount).toBe(1);
  });

  it("does not pay twice for the same score", () => {
    const first = applyResult(EMPTY_PROGRESSION, {
      slug: "sort-it-arcade",
      score: 0.9,
      lane: "visual",
      completed: true,
    });
    const second = applyResult(first.state, {
      slug: "sort-it-arcade",
      score: 0.9,
      lane: "visual",
      completed: true,
    });

    expect(first.xpGained).toBe(90);
    expect(second.xpGained).toBe(0);
    expect(second.state.xp).toBe(90);
    // The replay still counts as a play.
    expect(second.state.games["sort-it-arcade"]?.playCount).toBe(2);
  });

  it("pays the difference when the player beats their best", () => {
    const first = applyResult(EMPTY_PROGRESSION, {
      slug: "data-detox",
      score: 0.4,
      lane: "visual",
      completed: true,
    });
    const better = applyResult(first.state, {
      slug: "data-detox",
      score: 0.9,
      lane: "visual",
      completed: true,
    });

    expect(better.xpGained).toBe(50);
    expect(better.state.xp).toBe(90);
    expect(better.state.games["data-detox"]?.bestScore).toBeCloseTo(0.9);
  });

  it("keeps the best score when a later run is worse", () => {
    const good = applyResult(EMPTY_PROGRESSION, {
      slug: "data-detox",
      score: 0.9,
      lane: "visual",
      completed: true,
    });
    const worse = applyResult(good.state, {
      slug: "data-detox",
      score: 0.1,
      lane: "visual",
      completed: true,
    });

    expect(worse.state.games["data-detox"]?.bestScore).toBeCloseTo(0.9);
    expect(worse.stars).toBe(2);
  });

  it("awards the concept badge at two stars, once", () => {
    const first = applyResult(EMPTY_PROGRESSION, {
      slug: "gradient-descent-skier",
      score: 0.95,
      lane: "visual",
      completed: true,
    });
    expect(first.newBadge).toBe("gradient-descent-skier");
    expect(first.state.badges).toEqual(["gradient-descent-skier"]);

    const again = applyResult(first.state, {
      slug: "gradient-descent-skier",
      score: 0.99,
      lane: "visual",
      completed: true,
    });
    expect(again.newBadge).toBeNull();
    expect(again.state.badges).toEqual(["gradient-descent-skier"]);
  });

  it("withholds the badge at one star", () => {
    const applied = applyResult(EMPTY_PROGRESSION, {
      slug: "gradient-descent-skier",
      score: 0.2,
      lane: "visual",
      completed: true,
    });
    expect(applied.newBadge).toBeNull();
    expect(applied.state.badges).toEqual([]);
  });

  it("remembers a cleared code lane across later visual runs", () => {
    const codeRun = applyResult(EMPTY_PROGRESSION, {
      slug: "k-means-territory-wars",
      score: 0.9,
      lane: "code",
      completed: true,
      codeLaneCleared: true,
    });
    expect(codeRun.stars).toBe(3);

    const visualRun = applyResult(codeRun.state, {
      slug: "k-means-territory-wars",
      score: 0.85,
      lane: "visual",
      completed: true,
    });
    expect(visualRun.stars).toBe(3);
  });
});

describe("unlocks", () => {
  const withCompleted = (slug: string): ProgressionState =>
    applyResult(EMPTY_PROGRESSION, {
      slug,
      score: 1,
      lane: "visual",
      completed: true,
    }).state;

  it("opens games with no prerequisites", () => {
    expect(isUnlocked("sort-it-arcade", EMPTY_PROGRESSION)).toBe(true);
    expect(isUnlocked("data-detox", EMPTY_PROGRESSION)).toBe(true);
    expect(isUnlocked("gradient-descent-skier", EMPTY_PROGRESSION)).toBe(true);
  });

  it("gates games behind their prerequisites", () => {
    expect(isUnlocked("k-means-territory-wars", EMPTY_PROGRESSION)).toBe(false);
    expect(isUnlocked("k-means-territory-wars", withCompleted("sort-it-arcade"))).toBe(
      true,
    );
  });

  it("requires ALL prerequisites, not just one", () => {
    // neuron-forge needs sort-it-arcade AND gradient-descent-skier.
    const partial = withCompleted("sort-it-arcade");
    expect(isUnlocked("neuron-forge", partial)).toBe(false);
  });

  it("rejects unknown slugs", () => {
    expect(isUnlocked("not-a-game", EMPTY_PROGRESSION)).toBe(false);
  });

  it("lists the Phase-1 entry points for a new player", () => {
    expect(unlockedSlugs(EMPTY_PROGRESSION).sort()).toEqual(
      ["data-detox", "gradient-descent-skier", "sort-it-arcade"].sort(),
    );
  });
});

describe("useProgression store", () => {
  beforeEach(() => {
    useProgression.setState({
      ...EMPTY_PROGRESSION,
      hydrated: false,
      syncError: null,
      lastGain: null,
    });
    useProgression.getState().setAdapter(createMemoryAdapter());
  });

  it("records a result and exposes the derived level", () => {
    const applied = useProgression.getState().recordResult({
      slug: "sort-it-arcade",
      score: 1,
      lane: "code",
      completed: true,
    });

    expect(applied.xpGained).toBe(150);
    expect(useProgression.getState().xp).toBe(150);
    expect(useProgression.getState().lastGain).toBe(150);
    expect(useProgression.getState().level().level).toBe(1);
    expect(useProgression.getState().starsFor("sort-it-arcade")).toBe(2);
  });

  it("hydrates from its adapter", async () => {
    useProgression.getState().setAdapter(
      createMemoryAdapter({ xp: 640, games: {}, badges: ["data-detox"] }),
    );

    await useProgression.getState().hydrate();

    const state = useProgression.getState();
    expect(state.hydrated).toBe(true);
    expect(state.xp).toBe(640);
    expect(state.badges).toEqual(["data-detox"]);
    expect(state.level().level).toBe(3); // 200 + 300 consumed, 140 into level 3
  });

  it("persists through the adapter and can be cleared", async () => {
    const adapter = createMemoryAdapter();
    useProgression.getState().setAdapter(adapter);

    useProgression.getState().recordResult({
      slug: "data-detox",
      score: 0.9,
      lane: "visual",
      completed: true,
    });

    // Writes are optimistic; let the microtask queue drain.
    await Promise.resolve();
    await Promise.resolve();

    expect((await adapter.load())?.xp).toBe(90);

    await useProgression.getState().clear();
    expect(useProgression.getState().xp).toBe(0);
    expect((await adapter.load())?.xp).toBe(0);
  });

  it("gates unlocks through the store", () => {
    expect(useProgression.getState().isUnlocked("k-means-territory-wars")).toBe(
      false,
    );

    useProgression.getState().recordResult({
      slug: "sort-it-arcade",
      score: 1,
      lane: "visual",
      completed: true,
    });

    expect(useProgression.getState().isUnlocked("k-means-territory-wars")).toBe(
      true,
    );
  });
});
