import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GAME_CATALOG } from "@/lib/catalog";
import {
  BASE_XP,
  CODE_LANE_MULTIPLIER,
  CONCEPT_BADGES,
  EMPTY_PROGRESSION,
  GAIN_FLASH_MS,
  HIGH_SCORE_THRESHOLD,
  STORAGE_KEY,
  applyResult,
  badgeLabel,
  createLocalAdapter,
  createMemoryAdapter,
  isUnlocked,
  mergeProgression,
  levelFromXp,
  starsFor,
  unlockedSlugs,
  useProgression,
  xpForScore,
  xpToClearLevel,
  type GameProgress,
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

describe("useProgression — the transient XP flash", () => {
  beforeEach(() => {
    useProgression.setState({
      ...EMPTY_PROGRESSION,
      hydrated: true,
      syncError: null,
      lastGain: null,
    });
    useProgression.getState().setAdapter(createMemoryAdapter());
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("clears lastGain after the flash, so an equal second gain flashes again", () => {
    vi.useFakeTimers();
    const store = useProgression.getState();

    store.recordResult({ slug: "sort-it-arcade", score: 0.2, lane: "visual", completed: true });
    expect(useProgression.getState().lastGain).toBe(20);

    vi.advanceTimersByTime(GAIN_FLASH_MS);
    // Cleared: the XP bar unmounts the "+20 XP" span, so the NEXT +20 remounts
    // it (replaying the animation and changing the live region's content).
    expect(useProgression.getState().lastGain).toBeNull();

    store.recordResult({ slug: "sort-it-arcade", score: 0.4, lane: "visual", completed: true });
    expect(useProgression.getState().lastGain).toBe(20);
  });

  it("dismissGain drops a pending flash (GameShell does this on unmount)", () => {
    vi.useFakeTimers();
    useProgression
      .getState()
      .recordResult({ slug: "data-detox", score: 0.5, lane: "visual", completed: true });
    expect(useProgression.getState().lastGain).toBe(50);

    useProgression.getState().dismissGain();
    expect(useProgression.getState().lastGain).toBeNull();

    // And the old timer can't fire later and clobber a newer gain.
    useProgression
      .getState()
      .recordResult({ slug: "data-detox", score: 0.9, lane: "visual", completed: true });
    vi.advanceTimersByTime(GAIN_FLASH_MS - 1);
    expect(useProgression.getState().lastGain).toBe(40);
  });
});

describe("useProgression — persistence failures are surfaced", () => {
  beforeEach(() => {
    useProgression.setState({
      ...EMPTY_PROGRESSION,
      hydrated: true,
      syncError: null,
      lastGain: null,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    useProgression.getState().setAdapter(createMemoryAdapter());
  });

  it("reports a localStorage quota error through syncError instead of swallowing it", async () => {
    useProgression.getState().setAdapter(createLocalAdapter());
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("The quota has been exceeded.", "QuotaExceededError");
    });

    const applied = useProgression
      .getState()
      .recordResult({ slug: "k-means-territory-wars", score: 1, lane: "visual", completed: true });

    // Gameplay is unaffected: the award landed in memory.
    expect(applied.xpGained).toBe(100);
    expect(useProgression.getState().xp).toBe(100);

    await vi.waitFor(() =>
      expect(useProgression.getState().syncError).toMatch(/quota/i),
    );
  });

  it("clears syncError once a later save succeeds", async () => {
    useProgression.setState({ syncError: "earlier failure" });
    useProgression.getState().setAdapter(createMemoryAdapter());

    useProgression
      .getState()
      .recordResult({ slug: "k-means-territory-wars", score: 1, lane: "visual", completed: true });

    await vi.waitFor(() => expect(useProgression.getState().syncError).toBeNull());
  });
});

describe("mergeProgression (cross-tab)", () => {
  const game = (slug: string, over: Partial<GameProgress> = {}): GameProgress => ({
    slug,
    bestScore: 0,
    stars: 0,
    completed: false,
    codeLaneCleared: false,
    playCount: 1,
    xpAwarded: 0,
    updatedAt: "2026-01-01T00:00:00.000Z",
    ...over,
  });

  it("keeps what each tab earned, so neither overwrites the other", () => {
    const tabA: ProgressionState = {
      xp: 100,
      games: { "sort-it-arcade": game("sort-it-arcade", { bestScore: 1, completed: true, stars: 2, xpAwarded: 100 }) },
      badges: ["sort-it-arcade"],
    };
    const tabB: ProgressionState = {
      xp: 60,
      games: { "data-detox": game("data-detox", { bestScore: 0.6, completed: true, stars: 1, xpAwarded: 60 }) },
      badges: [],
    };

    const merged = mergeProgression(tabA, tabB);
    expect(Object.keys(merged.games).sort()).toEqual(["data-detox", "sort-it-arcade"]);
    expect(merged.xp).toBe(160);
    expect(merged.badges).toEqual(["sort-it-arcade"]);
  });

  it("takes the best of each field for a game both tabs played, and recomputes stars", () => {
    const merged = mergeProgression(
      {
        xp: 90,
        games: { "neuron-forge": game("neuron-forge", { bestScore: 0.9, completed: true, stars: 2, xpAwarded: 90, playCount: 3 }) },
        badges: [],
      },
      {
        xp: 45,
        games: { "neuron-forge": game("neuron-forge", { bestScore: 0.3, completed: true, codeLaneCleared: true, stars: 1, xpAwarded: 45, playCount: 5 }) },
        badges: [],
      },
    );

    const forge = merged.games["neuron-forge"]!;
    expect(forge.bestScore).toBe(0.9);
    expect(forge.codeLaneCleared).toBe(true);
    // 0.9 best score + code lane cleared (from the other tab) = three stars.
    expect(forge.stars).toBe(3);
    expect(forge.playCount).toBe(5);
    expect(forge.xpAwarded).toBe(90);
    expect(merged.xp).toBe(90);
  });
});

describe("useProgression — storage events from another tab", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useProgression.getState().setAdapter(createLocalAdapter());
    useProgression.setState({
      ...EMPTY_PROGRESSION,
      hydrated: true,
      syncError: null,
      lastGain: null,
    });
  });

  afterEach(() => {
    window.localStorage.clear();
    useProgression.getState().setAdapter(createMemoryAdapter());
  });

  const otherTabSaves = (state: ProgressionState | null) => {
    window.dispatchEvent(
      new StorageEvent("storage", {
        key: STORAGE_KEY,
        newValue: state === null ? null : JSON.stringify(state),
      }),
    );
  };

  it("folds the other tab's progress into this one instead of losing it later", () => {
    useProgression
      .getState()
      .recordResult({ slug: "sort-it-arcade", score: 1, lane: "visual", completed: true });

    otherTabSaves({
      xp: 70,
      games: {
        "data-detox": {
          slug: "data-detox",
          bestScore: 0.7,
          stars: 1,
          completed: true,
          codeLaneCleared: false,
          playCount: 1,
          xpAwarded: 70,
          updatedAt: "2026-01-01T00:00:00.000Z",
        },
      },
      badges: [],
    });

    const state = useProgression.getState();
    expect(state.games["sort-it-arcade"]?.completed).toBe(true);
    expect(state.games["data-detox"]?.completed).toBe(true);
    expect(state.xp).toBe(170);
  });

  it("keeps both tabs' progress in storage when they save before hearing from each other", async () => {
    // This tab earns and saves.
    useProgression
      .getState()
      .recordResult({ slug: "sort-it-arcade", score: 1, lane: "visual", completed: true });
    const stored = () =>
      JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "null") as ProgressionState | null;
    await vi.waitFor(() => expect(stored()?.games["sort-it-arcade"]).toBeDefined());

    // The other tab hydrated before that save and never heard about it, then
    // saved its own copy. It runs the same adapter code this tab does.
    const otherTab = applyResult(EMPTY_PROGRESSION, {
      slug: "data-detox",
      score: 0.7,
      lane: "visual",
      completed: true,
    }).state;
    await createLocalAdapter().save(otherTab);

    // Storage kept both, so a reload (or a third tab) loses neither.
    expect(Object.keys(stored()!.games).sort()).toEqual(["data-detox", "sort-it-arcade"]);
    expect(stored()!.xp).toBe(170);
    await useProgression.getState().hydrate();
    expect(useProgression.getState().xp).toBe(170);

    // And a deliberate reset is still a reset, not merged away.
    await useProgression.getState().clear();
    expect(stored()).toEqual(EMPTY_PROGRESSION);
  });

  it("adopts a reset from another tab rather than merging it away", () => {
    useProgression
      .getState()
      .recordResult({ slug: "sort-it-arcade", score: 1, lane: "visual", completed: true });

    otherTabSaves(EMPTY_PROGRESSION);

    expect(useProgression.getState().xp).toBe(0);
    expect(useProgression.getState().games).toEqual({});
  });

  it("ignores other keys", () => {
    useProgression
      .getState()
      .recordResult({ slug: "sort-it-arcade", score: 1, lane: "visual", completed: true });
    window.dispatchEvent(
      new StorageEvent("storage", { key: "something-else", newValue: "{}" }),
    );
    expect(useProgression.getState().xp).toBe(100);
  });
});
