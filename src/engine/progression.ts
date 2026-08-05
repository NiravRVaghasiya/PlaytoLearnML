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
 * 1. **It works with no backend.** Supabase is optional. When it isn't
 *    configured, everything persists to localStorage and the games play
 *    identically. Progression must never be a hard dependency of gameplay —
 *    the first-run flow in spec §6 is an explicit no-signup demo.
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

// ── Persistence adapters ───────────────────────────────────────────────────

export interface ProgressionAdapter {
  readonly name: "local" | "supabase" | "memory";
  load(): Promise<ProgressionState | null>;
  save(state: ProgressionState): Promise<void>;
}

export const STORAGE_KEY = "gameml:progression:v1";

/** Default adapter. Works offline, needs no account. */
export function createLocalAdapter(): ProgressionAdapter {
  return {
    name: "local",
    async load() {
      if (typeof window === "undefined") return null;
      try {
        const raw = window.localStorage.getItem(STORAGE_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw) as Partial<ProgressionState>;
        return {
          xp: typeof parsed.xp === "number" ? parsed.xp : 0,
          games: parsed.games ?? {},
          badges: Array.isArray(parsed.badges) ? parsed.badges : [],
        };
      } catch {
        // Corrupt or blocked storage must not break gameplay.
        return null;
      }
    },
    async save(state) {
      if (typeof window === "undefined") return;
      try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
      } catch {
        // Private-mode quota errors are non-fatal.
      }
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
  /** XP delta from the most recent award, for the XP bar's flash. */
  lastGain: number | null;

  setAdapter: (adapter: ProgressionAdapter) => void;
  hydrate: () => Promise<void>;
  recordResult: (result: GameResult) => AppliedResult;
  isUnlocked: (slug: string) => boolean;
  level: () => LevelInfo;
  starsFor: (slug: string) => StarCount;
  clear: () => Promise<void>;
}

export const useProgression = create<ProgressionStore>((set, get) => {
  let adapter: ProgressionAdapter = createLocalAdapter();

  const persist = (state: ProgressionState) => {
    // Optimistic: local state is already updated; report failure, don't revert.
    adapter
      .save(state)
      .then(() => set({ syncError: null }))
      .catch((cause: unknown) =>
        set({
          syncError:
            cause instanceof Error ? cause.message : "Could not save progress",
        }),
      );
  };

  return {
    ...EMPTY_PROGRESSION,
    hydrated: false,
    syncError: null,
    lastGain: null,

    setAdapter(next) {
      adapter = next;
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
      set({ ...EMPTY_PROGRESSION, lastGain: null, syncError: null });
      await adapter.save(EMPTY_PROGRESSION);
    },
  };
});
