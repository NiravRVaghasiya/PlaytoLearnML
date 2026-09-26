"use client";

import { create } from "zustand";
import { GAME_CATALOG, getGameMeta } from "@/lib/catalog";
import { getSupabaseClient } from "@/lib/supabase";
import { clamp } from "@/lib/utils";
import type { Lane, StarCount } from "./types";

/**
 * Progression service: XP, mastery stars, concept badges, and unlocks
 * (spec §4).
 *
 * Two design decisions worth knowing:
 *
 * 1. **It works with no backend.** Progression persists to localStorage, per
 *    browser, and the games play identically with or without it. Progression
 *    must never be a hard dependency of gameplay — the first-run flow in spec
 *    §6 is an explicit no-signup demo.
 *
 *    Supabase is a seam, not a feature: `createSupabaseAdapter` exists and is
 *    tested, but nothing calls it yet, because there is no sign-in flow to
 *    supply the user id it needs. Setting `NEXT_PUBLIC_SUPABASE_*` alone
 *    therefore changes nothing. Wiring it up means adding auth first, then
 *    calling `setAdapter()` with it before `hydrate()`.
 * 2. **Writes are optimistic.** Local state updates immediately so the XP bar
 *    springs the instant a player earns something; persistence happens after and
 *    surfaces failures in `syncError` rather than blocking the UI.
 *
 * The pure scoring functions below are exported separately so they can be
 * unit-tested without React or a store.
 */

// ── Tuning constants ────────────────────────────────────────────────────────

/** XP for a perfect run in the visual lane. */
export const BASE_XP = 100;
/** Spec §4: "the code lane pays more". */
export const CODE_LANE_MULTIPLIER = 1.5;
/** Normalised score needed for the second mastery star. */
export const HIGH_SCORE_THRESHOLD = 0.8;

// ── Data model ─────────────────────────────────────────────────────────────

export interface GameProgress {
  slug: string;
  /** Best normalised score, 0–1. */
  bestScore: number;
  stars: StarCount;
  completed: boolean;
  /** Third star: the player cleared the code-lane challenge. */
  codeLaneCleared: boolean;
  playCount: number;
  /** XP already granted for this game, so replays can't farm it. */
  xpAwarded: number;
  updatedAt: string;
}

export interface ProgressionState {
  xp: number;
  games: Record<string, GameProgress>;
  /** Earned concept-badge ids (keys of `CONCEPT_BADGES`). */
  badges: string[];
}

export const EMPTY_PROGRESSION: ProgressionState = {
  xp: 0,
  games: {},
  badges: [],
};

/**
 * Shareable concept badges (spec §4/§8: "I understand overfitting").
 *
 * Kept here rather than in the catalog because a badge is a progression reward,
 * not part of a game's identity. `progression.test.ts` asserts this map covers
 * every catalog slug, so the two can't drift.
 */
export const CONCEPT_BADGES: Record<string, string> = {
  "sort-it-arcade": "decision boundaries",
  "k-means-territory-wars": "clustering without labels",
  "data-detox": "why data cleaning changes everything",
  "gradient-descent-skier": "gradient descent and learning rate",
  "neuron-forge": "network architecture",
  "overfit-tower-defense": "overfitting",
  "confusion-matrix-chef": "precision versus recall",
  "decision-tree-architect": "information gain",
  "hyperparameter-heist": "hyperparameter search",
  "feature-forge": "feature engineering",
  "agent-academy": "reward design",
  "convolution-kitchen": "convolutional features",
  "backprop-blitz": "backpropagation",
  "dimension-diver": "dimensionality reduction",
};

/** Human label for a badge, e.g. "I understand overfitting". */
export function badgeLabel(slug: string): string | null {
  const concept = CONCEPT_BADGES[slug];
  return concept ? `I understand ${concept}` : null;
}

// ── Pure scoring ───────────────────────────────────────────────────────────

/**
 * XP for a run. Scales with the normalised score and pays a premium for the
 * code lane (spec §4).
 */
export function xpForScore(score: number, lane: Lane): number {
  const multiplier = lane === "code" ? CODE_LANE_MULTIPLIER : 1;
  return Math.round(BASE_XP * clamp(score, 0, 1) * multiplier);
}

/** XP needed to clear a given level. Grows linearly: 200, 300, 400, … */
export function xpToClearLevel(level: number): number {
  return 200 + Math.max(0, level - 1) * 100;
}

export interface LevelInfo {
  level: number;
  xpIntoLevel: number;
  xpForNextLevel: number;
}

export function levelFromXp(totalXp: number): LevelInfo {
  let level = 1;
  let remaining = Math.max(0, Math.floor(totalXp));

  while (remaining >= xpToClearLevel(level)) {
    remaining -= xpToClearLevel(level);
    level += 1;
  }

  return {
    level,
    xpIntoLevel: remaining,
    xpForNextLevel: xpToClearLevel(level),
  };
}

/**
 * Mastery stars (spec §4). Cumulative, so three stars implies the first two:
 *   ⭐ finished the game
 *   ⭐⭐ … and scored at least `HIGH_SCORE_THRESHOLD`
 *   ⭐⭐⭐ … and cleared the code-lane challenge
 */
export function starsFor(input: {
  completed: boolean;
  bestScore: number;
  codeLaneCleared: boolean;
}): StarCount {
  if (!input.completed) return 0;
  if (input.bestScore < HIGH_SCORE_THRESHOLD) return 1;
  return input.codeLaneCleared ? 3 : 2;
}

/** A game is unlocked once every prerequisite in the catalog is completed. */
export function isUnlocked(slug: string, state: ProgressionState): boolean {
  const meta = getGameMeta(slug);
  if (!meta) return false;
  return meta.requires.every((req) => state.games[req]?.completed === true);
}

export function unlockedSlugs(state: ProgressionState): string[] {
  return GAME_CATALOG.filter((g) => isUnlocked(g.slug, state)).map(
    (g) => g.slug,
  );
}

// ── Result reduction (pure) ────────────────────────────────────────────────

export interface GameResult {
  slug: string;
  /** Normalised 0–1. Each game maps its own scoring onto this. */
  score: number;
  lane: Lane;
  /** Did the player finish the objective? */
  completed: boolean;
  /** Did they clear the code-lane challenge (third star)? */
  codeLaneCleared?: boolean;
}

export interface AppliedResult {
  state: ProgressionState;
  /** XP actually granted — 0 when the run didn't beat the previous best. */
  xpGained: number;
  /** Badge id newly earned by this run, if any. */
  newBadge: string | null;
  stars: StarCount;
}

/**
 * Fold a run into progression state. Pure: no storage, no React.
 *
 * XP is granted as the *increment* over what this game has already paid out, so
 * replaying an easy level can't farm levels — but beating your own best always
 * pays the difference.
 */
export function applyResult(
  state: ProgressionState,
  result: GameResult,
): AppliedResult {
  const previous = state.games[result.slug];
  const score = clamp(result.score, 0, 1);

  const bestScore = Math.max(previous?.bestScore ?? 0, score);
  const completed = (previous?.completed ?? false) || result.completed;
  const codeLaneCleared =
    (previous?.codeLaneCleared ?? false) || result.codeLaneCleared === true;

  const stars = starsFor({ completed, bestScore, codeLaneCleared });

  // Value the best run at the better of the two lanes' rates.
  const earnable = Math.max(
    xpForScore(bestScore, result.lane),
    previous?.xpAwarded ?? 0,
  );
  const xpGained = Math.max(0, earnable - (previous?.xpAwarded ?? 0));

  const nextGame: GameProgress = {
    slug: result.slug,
    bestScore,
    stars,
    completed,
    codeLaneCleared,
    playCount: (previous?.playCount ?? 0) + 1,
    xpAwarded: (previous?.xpAwarded ?? 0) + xpGained,
    updatedAt: new Date().toISOString(),
  };

  // Concept badges land at two stars — you understood it, not just finished it.
  const badgeEarned =
    stars >= 2 && CONCEPT_BADGES[result.slug] && !state.badges.includes(result.slug)
      ? result.slug
      : null;

  return {
    state: {
      xp: state.xp + xpGained,
      games: { ...state.games, [result.slug]: nextGame },
      badges: badgeEarned ? [...state.badges, badgeEarned] : state.badges,
    },
    xpGained,
    newBadge: badgeEarned,
    stars,
  };
}

/**
 * Combine two progression states without losing anything either one earned.
 *
 * Used when another tab saves: both tabs hydrated from the same storage, then
 * each earned things on its own, so neither copy is "the truth". Per game the
 * best of each field wins (best score, completion, code-lane clear, XP already
 * paid out, play count), stars are recomputed from those, and badges are the
 * union. Total XP is the sum of what every game has paid out — which is how
 * `applyResult` accrues it — and never less than either side's total.
 */
export function mergeProgression(
  a: ProgressionState,
  b: ProgressionState,
): ProgressionState {
  const games: Record<string, GameProgress> = { ...a.games };
  for (const [slug, theirs] of Object.entries(b.games)) {
    const ours = games[slug];
    if (!ours) {
      games[slug] = theirs;
      continue;
    }
    const bestScore = Math.max(ours.bestScore, theirs.bestScore);
    const completed = ours.completed || theirs.completed;
    const codeLaneCleared = ours.codeLaneCleared || theirs.codeLaneCleared;
    games[slug] = {
      slug,
      bestScore,
      completed,
      codeLaneCleared,
      stars: starsFor({ completed, bestScore, codeLaneCleared }),
      playCount: Math.max(ours.playCount, theirs.playCount),
      xpAwarded: Math.max(ours.xpAwarded, theirs.xpAwarded),
      updatedAt: ours.updatedAt > theirs.updatedAt ? ours.updatedAt : theirs.updatedAt,
    };
  }

  const paidOut = Object.values(games).reduce(
    (sum, game) => sum + (Number.isFinite(game.xpAwarded) ? game.xpAwarded : 0),
    0,
  );

  return {
    xp: Math.max(a.xp, b.xp, paidOut),
    games,
    badges: [...new Set([...a.badges, ...b.badges])],
  };
}

function isEmptyProgression(state: ProgressionState): boolean {
  return (
    state.xp === 0 &&
    Object.keys(state.games).length === 0 &&
    state.badges.length === 0
  );
}

// ── Persistence adapters ───────────────────────────────────────────────────

export interface ProgressionAdapter {
  readonly name: "local" | "supabase" | "memory";
  load(): Promise<ProgressionState | null>;
  save(state: ProgressionState): Promise<void>;
}

export const STORAGE_KEY = "gameml:progression:v1";

/** Parse a stored blob. Corrupt input reads as "nothing saved", never a throw. */
function parseStored(raw: string | null): ProgressionState | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as Partial<ProgressionState> | null;
    if (!parsed || typeof parsed !== "object") return null;
    return {
      xp: typeof parsed.xp === "number" ? parsed.xp : 0,
      games:
        parsed.games && typeof parsed.games === "object" ? parsed.games : {},
      badges: Array.isArray(parsed.badges) ? parsed.badges : [],
    };
  } catch {
    return null;
  }
}

/** Default adapter. Works offline, needs no account. */
export function createLocalAdapter(): ProgressionAdapter {
  return {
    name: "local",
    async load() {
      if (typeof window === "undefined") return null;
      try {
        return parseStored(window.localStorage.getItem(STORAGE_KEY));
      } catch {
        // Blocked storage (some privacy modes throw on access) must not break
        // gameplay.
        return null;
      }
    },
    async save(state) {
      if (typeof window === "undefined") return;
      // Deliberately NOT swallowed. A quota or privacy-mode failure is still
      // non-fatal — gameplay carries on from memory — but the player deserves
      // to know their progress isn't being kept, and `persist()` can only say
      // so (via `syncError`) if the failure reaches it.
      //
      // Read, merge, write. Another tab may have saved since this one last
      // heard from it (its `storage` event is still in flight), and writing
      // this tab's copy over the blob would erase what that tab earned. An
      // empty state is a deliberate reset (`clear()`) and is written as-is.
      const next = isEmptyProgression(state)
        ? state
        : mergeProgression(
            parseStored(window.localStorage.getItem(STORAGE_KEY)) ?? EMPTY_PROGRESSION,
            state,
          );
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    },
  };
}

/**
 * Supabase adapter. Requires an authenticated user id and the browser-safe
 * anon key (see `src/lib/supabase.ts` — the service-role key must never reach
 * the client). Returns `null` when Supabase isn't configured so callers can
 * fall back to local storage.
 *
 * Expected table, protected by RLS so a player can only touch their own row:
 *
 * ```sql
 * create table public.progression (
 *   user_id    uuid primary key references auth.users on delete cascade,
 *   xp         integer not null default 0,
 *   games      jsonb   not null default '{}'::jsonb,
 *   badges     jsonb   not null default '[]'::jsonb,
 *   updated_at timestamptz not null default now()
 * );
 * alter table public.progression enable row level security;
 * create policy "own row" on public.progression
 *   for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
 * ```
 */
export function createSupabaseAdapter(
  userId: string,
): ProgressionAdapter | null {
  const client = getSupabaseClient();
  if (!client) return null;

  return {
    name: "supabase",
    async load() {
      const { data, error } = await client
        .from("progression")
        .select("xp, games, badges")
        .eq("user_id", userId)
        .maybeSingle();

      if (error || !data) return null;
      return {
        xp: typeof data.xp === "number" ? data.xp : 0,
        games: (data.games ?? {}) as Record<string, GameProgress>,
        badges: (data.badges ?? []) as string[],
      };
    },
    async save(state) {
      const { error } = await client.from("progression").upsert({
        user_id: userId,
        xp: state.xp,
        games: state.games,
        badges: state.badges,
        updated_at: new Date().toISOString(),
      });
      if (error) throw new Error(error.message);
    },
  };
}

/** In-memory adapter for tests. */
export function createMemoryAdapter(
  seed: ProgressionState = EMPTY_PROGRESSION,
): ProgressionAdapter {
  let held: ProgressionState = seed;
  return {
    name: "memory",
    async load() {
      return held;
    },
    async save(state) {
      held = state;
    },
  };
}

// ── React store ────────────────────────────────────────────────────────────

export interface ProgressionStore extends ProgressionState {
  hydrated: boolean;
  /** Non-fatal persistence failure, surfaced for a quiet "not saved" hint. */
  syncError: string | null;
  /**
   * XP delta from the most recent award, for the XP bar's flash. Transient: it
   * clears itself after `GAIN_FLASH_MS`, and `GameShell` clears it when a game
   * unmounts, so the "+N XP" never replays on arrival in another game, and a
   * second award of the same size still flashes and is announced again.
   */
  lastGain: number | null;

  setAdapter: (adapter: ProgressionAdapter) => void;
  /** Drop the pending XP flash (GameShell calls this on unmount). */
  dismissGain: () => void;
  hydrate: () => Promise<void>;
  recordResult: (result: GameResult) => AppliedResult;
  isUnlocked: (slug: string) => boolean;
  level: () => LevelInfo;
  starsFor: (slug: string) => StarCount;
  clear: () => Promise<void>;
}

/**
 * How long a "+N XP" stays in `lastGain`. A little longer than the
 * `xp-gain-flash` animation in globals.css (1600 ms), so the flash always
 * finishes before the value clears.
 */
export const GAIN_FLASH_MS = 2000;

/**
 * A save failure as a sentence. Read structurally rather than with
 * `instanceof Error`: a storage quota failure is a `DOMException`, which is not
 * an `Error` in every realm (jsdom, some embedded webviews).
 */
function saveErrorMessage(cause: unknown): string {
  const message = (cause as { message?: unknown } | null)?.message;
  return typeof message === "string" && message.length > 0
    ? message
    : "Could not save progress";
}

export const useProgression = create<ProgressionStore>((set, get) => {
  let adapter: ProgressionAdapter = createLocalAdapter();
  let gainTimer: ReturnType<typeof setTimeout> | null = null;

  const clearGainTimer = () => {
    if (gainTimer !== null) clearTimeout(gainTimer);
    gainTimer = null;
  };

  const persist = (state: ProgressionState) => {
    // Optimistic: local state is already updated; report failure, don't revert.
    // `Promise.resolve().then` so an adapter that throws synchronously is
    // reported the same way as one that rejects.
    Promise.resolve()
      .then(() => adapter.save(state))
      .then(() => set({ syncError: null }))
      .catch((cause: unknown) =>
        set({
          syncError:
            saveErrorMessage(cause),
        }),
      );
  };

  /*
   * Cross-tab sync. Two tabs each hydrate once; without this, the second tab
   * to save silently overwrote everything the first had earned. The browser
   * fires `storage` in every OTHER tab when one writes, so each tab folds the
   * other's save into its own state. That keeps memory whole; storage is kept
   * whole by the local adapter, which merges with the stored blob on every
   * save, so a near-simultaneous save in both tabs loses nothing even before
   * either event arrives. A save that empties progression is a deliberate
   * reset and is adopted as-is rather than merged away.
   */
  if (typeof window !== "undefined") {
    window.addEventListener("storage", (event) => {
      if (event.key !== null && event.key !== STORAGE_KEY) return;
      if (adapter.name !== "local" || !get().hydrated) return;

      const incoming = parseStored(event.newValue) ?? EMPTY_PROGRESSION;
      if (isEmptyProgression(incoming)) {
        set({ ...EMPTY_PROGRESSION });
        return;
      }
      const { xp, games, badges } = get();
      set(mergeProgression({ xp, games, badges }, incoming));
    });
  }

  return {
    ...EMPTY_PROGRESSION,
    hydrated: false,
    syncError: null,
    lastGain: null,

    setAdapter(next) {
      adapter = next;
    },

    dismissGain() {
      clearGainTimer();
      if (get().lastGain !== null) set({ lastGain: null });
    },

    async hydrate() {
      const loaded = await adapter.load();
      set({
        xp: loaded?.xp ?? 0,
        games: loaded?.games ?? {},
        badges: loaded?.badges ?? [],
        hydrated: true,
      });
    },

    recordResult(result) {
      const { xp, games, badges } = get();
      const applied = applyResult({ xp, games, badges }, result);

      set({ ...applied.state, lastGain: applied.xpGained || null });
      clearGainTimer();
      if (applied.xpGained > 0) {
        gainTimer = setTimeout(() => {
          gainTimer = null;
          set({ lastGain: null });
        }, GAIN_FLASH_MS);
      }
      persist(applied.state);

      return applied;
    },

    isUnlocked(slug) {
      const { xp, games, badges } = get();
      return isUnlocked(slug, { xp, games, badges });
    },

    level() {
      return levelFromXp(get().xp);
    },

    starsFor(slug) {
      return get().games[slug]?.stars ?? 0;
    },

    async clear() {
      clearGainTimer();
      set({ ...EMPTY_PROGRESSION, lastGain: null, syncError: null });
      try {
        await adapter.save(EMPTY_PROGRESSION);
      } catch (cause) {
        set({
          syncError:
            saveErrorMessage(cause),
        });
      }
    },
  };
});
