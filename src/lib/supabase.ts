import { createClient, type SupabaseClient } from "@supabase/supabase-js";

/**
 * Supabase access for GameML — a seam for a backend that is NOT wired up yet.
 *
 * Today progression is per-browser localStorage, full stop. The store in
 * `src/engine/progression.ts` always starts on `createLocalAdapter()`;
 * `createSupabaseAdapter()` exists, but nothing in the app calls
 * it or `setAdapter()`, and there is no sign-in flow to supply the user id it
 * needs. So setting `NEXT_PUBLIC_SUPABASE_URL` / `NEXT_PUBLIC_SUPABASE_ANON_KEY`
 * on a deploy changes nothing, and progression needs no environment variables
 * at all. Wiring it means: an auth flow, then `setAdapter()` with the signed-in
 * user's adapter before `hydrate()`.
 *
 * GameML is and must stay playable with NO Supabase project — the first-run
 * flow in spec §6 is an explicit no-signup demo. Nothing here touches ML: per
 * CLAUDE.md all training/inference stays in the browser.
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
