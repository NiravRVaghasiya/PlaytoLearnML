/**
 * Did this error come from a JavaScript chunk that failed to download, rather
 * than from the app's own code?
 *
 * The distinction decides the advice a crash screen gives. A code bug may clear
 * on a retry; a missing chunk never will, because `next/dynamic` sits on
 * `React.lazy`, which caches the rejected import — only a full reload fetches it
 * again. It is also the COMMON failure in production: a dropped mobile
 * connection mid-download (the TF.js chunk alone is about 250 KB gzipped), or a
 * redeploy that removed the chunk an open tab was built against (version skew).
 *
 * Its own tiny module because both the per-game boundary (`GameMount`) and the
 * route-level `app/error.tsx` / `app/global-error.tsx` need it, and the route
 * boundaries ship on every page — they must not drag the game mount map in.
 *
 * Browsers and bundlers word it differently, so this matches all the shapes:
 * webpack/Turbopack's `ChunkLoadError` ("Loading chunk 123 failed", "Failed to
 * load chunk /_next/static/chunks/…"), and native dynamic `import()` failures
 * from Chrome ("Failed to fetch dynamically imported module"), Safari
 * ("Importing a module script failed") and Firefox ("error loading dynamically
 * imported module").
 */
export function isChunkLoadError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  if (error.name === "ChunkLoadError") return true;
  return /Loading chunk [\w/.-]+ failed|Failed to load chunk|Loading CSS chunk|Failed to fetch dynamically imported module|Importing a module script failed|error loading dynamically imported module/i.test(
    error.message,
  );
}
