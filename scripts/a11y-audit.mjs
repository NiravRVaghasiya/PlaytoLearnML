/**
 * Lighthouse accessibility audit for GameML.
 *
 * CLAUDE.md's Definition of Done requires a Lighthouse accessibility score of at
 * least 95 per game. This script runs the real Lighthouse accessibility category
 * against a running server, using the Chromium that Playwright installed.
 *
 * Usage:
 *   node scripts/a11y-audit.mjs [url ...]
 *   AUDIT_FORM_FACTORS=mobile node scripts/a11y-audit.mjs   # one pass only
 *
 * Defaults to every route on the site — see `siteRoutes()` in harness.mjs —
 * each audited twice, once as a desktop and once as a phone. Exits non-zero if
 * any page scores below the threshold in either pass, so it can gate a merge.
 *
 * What a 100 here does NOT prove: Lighthouse audits the page as it first loads.
 * It never opens the Math drawer, types into a code lane or triggers a named
 * failure, and it cannot check target size, drag alternatives, reflow or focus
 * obscured by sticky chrome. `verify-mobile.mjs` and `verify-playthrough.mjs`
 * cover some of that — the latter opens every game's Math dialog and checks
 * that KaTeX rendered, focus stays trapped and Escape hands it back
 * (`checkMathDialog` in harness.mjs), though no Lighthouse or axe scan runs on
 * the open dialog; the rest is manual.
 */

import { chromium } from "playwright";
import lighthouse from "lighthouse";
import { routeFor, siteRoutes } from "./harness.mjs";

const THRESHOLD = 95;
const BASE = process.env.AUDIT_BASE ?? "http://localhost:3000";

/**
 * Explicit emulation for both passes. The previous config said "desktop" but
 * disabled emulation, so Lighthouse measured whatever window the Playwright
 * browser happened to open — 800×600, neither the `lg` desktop layout nor a
 * phone — and the mobile layout, where the sticky metric and the stacked rail
 * live, was never audited at all.
 */
const FORM_FACTORS = {
  // Lighthouse's own desktop preset metrics: wide enough for the `lg` layout
  // (the right rail) that DESIGN.md §10 calls the primary target.
  desktop: {
    formFactor: "desktop",
    screenEmulation: {
      mobile: false,
      width: 1350,
      height: 940,
      deviceScaleFactor: 1,
      disabled: false,
    },
    // Keep Chromium's own UA; the default would claim to be a phone.
    emulatedUserAgent: false,
  },
  // A small Android phone, the same 360×740 at 2× that verify-mobile.mjs uses,
  // so the two harnesses are judging the same layout.
  mobile: {
    formFactor: "mobile",
    screenEmulation: {
      mobile: true,
      width: 360,
      height: 740,
      deviceScaleFactor: 2,
      disabled: false,
    },
  },
};

const requestedFactors = (process.env.AUDIT_FORM_FACTORS ?? "desktop,mobile")
  .split(",")
  .map((name) => name.trim())
  .filter(Boolean);
for (const name of requestedFactors) {
  if (!(name in FORM_FACTORS)) {
    console.error(
      `Unknown form factor "${name}" in AUDIT_FORM_FACTORS. Use desktop, mobile or both.`,
    );
    process.exit(1);
  }
}

const urls =
  process.argv.slice(2).length > 0
    ? process.argv.slice(2)
    : siteRoutes().map((route) => `${BASE}${route.path}`);

const browser = await chromium.launch({
  args: ["--remote-debugging-port=9222"],
});

let failures = 0;
let audited = 0;

try {
  for (const factor of requestedFactors) {
    console.log(`\n${"=".repeat(66)}\n${factor} pass\n${"=".repeat(66)}`);

    for (const url of urls) {
      // The 404 page is audited on purpose — it is where a mistyped link lands,
      // and it used to have no main landmark and no way home. Lighthouse treats
      // a 404 document as a failed load unless told the status is expected.
      const expectNotFound = routeFor(url)?.kind === "not-found";

      const result = await lighthouse(
        url,
        {
          port: 9222,
          output: "json",
          logLevel: "error",
          onlyCategories: ["accessibility"],
          ...FORM_FACTORS[factor],
          ...(expectNotFound ? { ignoreStatusCode: true } : {}),
        },
        undefined,
      );
      audited += 1;

      const lhr = result?.lhr;
      if (!lhr) {
        console.log(`${url} [${factor}]\n  ERROR: Lighthouse returned no result`);
        failures += 1;
        continue;
      }

      console.log(`\n${url} [${factor}]`);

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
  }
} finally {
  await browser.close();
}

console.log(
  `\n${audited} audits (${urls.length} pages × ${requestedFactors.join(" + ")})`,
);
console.log(
  `${failures === 0 ? "ALL PASS" : `${failures} audit(s) below ${THRESHOLD} or failed to run`}`,
);
process.exit(failures === 0 ? 0 : 1);
