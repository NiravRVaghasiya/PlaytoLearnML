import { createClient, type SupabaseClient } from "@supabase/supabase-js";

/**
 * Supabase access for GameML.
 *
 * GameML is playable with NO Supabase project at all — the progression service
 * falls back to localStorage when this returns `null` (see
 * `src/engine/progression.ts`). Nothing here touches ML: per CLAUDE.md all
 * training/inference stays in the browser.
 *
 * SECURITY
 * --------
 * `SUPABASE_SERVICE_ROLE_KEY` bypasses Row Level Security. It is deliberately
 * NOT read in this module, because this module is imported by client
 * components. Service-role work belongs in a Route Handler / Server Action
 * that reads `process.env.SUPABASE_SERVICE_ROLE_KEY` directly and never
 * returns the key to the caller.
 */

let cached: SupabaseClient | null | undefined;

/** True when browser-safe Supabase credentials are configured. */
export function isSupabaseConfigured(): boolean {
  return Boolean(
    process.env.NEXT_PUBLIC_SUPABASE_URL &&
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
  );
}

/**
 * Returns a browser-safe Supabase client, or `null` when the project isn't
 * configured. Callers MUST handle `null` — never assume a backend exists.
 */
export function getSupabaseClient(): SupabaseClient | null {
  if (cached !== undefined) return cached;

  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  cached = url && anonKey ? createClient(url, anonKey) : null;
  return cached;
}

/** Test seam: forget the memoized client. */
export function resetSupabaseClient(): void {
  cached = undefined;
}
