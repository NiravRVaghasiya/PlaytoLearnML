/**
 * Lighthouse accessibility audit for GameML.
 *
 * CLAUDE.md's Definition of Done requires a Lighthouse accessibility score of at
 * least 95 per game. This script runs the real Lighthouse accessibility category
 * against a running server, using the Chromium that Playwright installed.
 *
 * Usage:
 *   node scripts/a11y-audit.mjs [url ...]
 *
 * Defaults to the home page and every playable game route. Exits non-zero if any
 * page scores below the threshold, so it can gate a merge.
 */

import { chromium } from "playwright";
import lighthouse from "lighthouse";

const THRESHOLD = 95;
const BASE = process.env.AUDIT_BASE ?? "http://localhost:3000";

/** Every playable route, plus the home page. Keep in sync with the registry. */
const DEFAULT_PATHS = [
  "/",
  "/play/sort-it-arcade",
  "/play/k-means-territory-wars",
  "/play/data-detox",
  "/play/gradient-descent-skier",
  "/play/neuron-forge",
  "/play/overfit-tower-defense",
];

const urls =
  process.argv.slice(2).length > 0
    ? process.argv.slice(2)
    : DEFAULT_PATHS.map((path) => `${BASE}${path}`);

const browser = await chromium.launch({
  args: ["--remote-debugging-port=9222"],
});

let failures = 0;

try {
  for (const url of urls) {
    const result = await lighthouse(
      url,
      {
        port: 9222,
        output: "json",
        logLevel: "error",
        onlyCategories: ["accessibility"],
        // Games are interactive canvases, not documents; desktop is the primary
        // target per DESIGN.md §10.
        formFactor: "desktop",
        screenEmulation: { disabled: true },
      },
      undefined,
    );

    const lhr = result?.lhr;
    if (!lhr) {
      console.log(`${url}\n  ERROR: Lighthouse returned no result`);
      failures += 1;
      continue;
    }

    console.log(`\n${url}`);

    // A score of 0 with no failed audits means Lighthouse itself failed to run
    // the page. Surface that rather than reporting a misleading zero.
    if (lhr.runtimeError) {
      console.log(
        `  RUNTIME ERROR: ${lhr.runtimeError.code} — ${lhr.runtimeError.message}`,
      );
      failures += 1;
      continue;
    }

    const score = Math.round((lhr.categories.accessibility.score ?? 0) * 100);
    const pass = score >= THRESHOLD;
    if (!pass) failures += 1;

    console.log(`  accessibility: ${score}  ${pass ? "PASS" : "FAIL"}`);

    const failed = Object.values(lhr.audits).filter(
      (audit) =>
        audit.score !== null &&
        audit.score < 1 &&
        audit.scoreDisplayMode !== "notApplicable" &&
        audit.scoreDisplayMode !== "informative",
    );

    if (failed.length === 0) {
      console.log("  no failed audits");
    } else {
      for (const audit of failed) {
        console.log(`  FAILED AUDIT: ${audit.id} — ${audit.title}`);
        const items = audit.details?.items ?? [];
        for (const item of items.slice(0, 4)) {
          const snippet = item.node?.snippet ?? item.node?.selector ?? "";
          if (snippet) console.log(`      ${String(snippet).slice(0, 160)}`);
        }
      }
    }

    // Manual-only checks Lighthouse can't automate; listed so they aren't
    // mistaken for passes.
    const manual = Object.values(lhr.audits).filter(
      (audit) => audit.scoreDisplayMode === "manual",
    );
    if (manual.length > 0) {
      console.log(
        `  ${manual.length} checks require manual verification (Lighthouse cannot automate these)`,
      );
    }
  }
} finally {
  await browser.close();
}

console.log(
  `\n${failures === 0 ? "ALL PASS" : `${failures} page(s) below ${THRESHOLD}`}`,
);
process.exit(failures === 0 ? 0 : 1);
