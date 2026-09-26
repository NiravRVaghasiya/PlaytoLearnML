/**
 * Browser playthrough runner.
 *
 * Unit tests prove a game's ML is correct; this proves it's wired to the screen.
 * Sort-It Arcade shipped with two silent pedagogical bugs that every unit test
 * passed — a 25-parameter boundary that could win by 0.002, and a verdict that
 * depended on the player's route to it. Both were found here. So this runs
 * before any game is called done.
 *
 * Each game supplies a module in `scripts/playthroughs/<slug>.mjs` exporting
 * `slug`, `title`, and `run({ page, check, metricText })`.
 *
 * On top of each game's own checks, every game's ƒ Math dialog is opened and
 * checked (`checkMathDialog`: KaTeX rendered, focus trapped, Escape returns
 * focus), and every playthrough fails on a console
 * error, an uncaught page error, or a warning that means a real bug (a KaTeX
 * equation LaTeX would render differently, a React hydration/key warning).
 * Headless-GPU chatter and TF.js's WebGL-to-CPU fallback are ignored. The
 * rules are `classifyConsoleMessage` in harness.mjs, shared with
 * verify-mobile.mjs.
 *
 * Usage:
 *   node scripts/verify-playthrough.mjs               # every game
 *   node scripts/verify-playthrough.mjs sort-it-arcade
 */

import { chromium } from "playwright";
import { readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { checkMathDialog, classifyConsoleMessage } from "./harness.mjs";

const BASE = process.env.AUDIT_BASE ?? "http://localhost:3000";
const here = dirname(fileURLToPath(import.meta.url));
const playthroughDir = join(here, "playthroughs");

const requested = process.argv.slice(2);
const files = (await readdir(playthroughDir)).filter((name) =>
  name.endsWith(".mjs"),
);
const playthroughs = [];
for (const file of files) {
  const loaded = await import(`file://${join(playthroughDir, file)}`);
  if (requested.length === 0 || requested.includes(loaded.slug)) {
    playthroughs.push(loaded);
  }
}

if (playthroughs.length === 0) {
  console.error(
    `No playthrough matched. Available: ${files.map((f) => f.replace(".mjs", "")).join(", ")}`,
  );
  process.exit(1);
}

/**
 * Everything currently sitting in an ARIA live region, read via textContent so
 * the screen-reader-only clipping used by `.sr-only-live` cannot hide it.
 */
const liveRegionText = (page) =>
  page.$$eval("[aria-live]", (nodes) =>
    nodes.map((node) => node.textContent ?? "").join(" | "),
  );

const browser = await chromium.launch();
let totalFailed = 0;
let totalChecks = 0;

for (const playthrough of playthroughs) {
  const results = [];
  const consoleErrors = [];
  const consoleWarnings = [];

  const page = await browser.newPage();
  // Errors always fail. Warnings fail only when they are the kind that means a
  // real bug — a KaTeX equation rendering differently from LaTeX, a React
  // hydration or key warning — and headless-GPU / TF.js-fallback chatter is
  // ignored. The split lives in harness.mjs so verify-mobile agrees with it.
  page.on("console", (message) => {
    const verdict = classifyConsoleMessage({
      type: message.type(),
      text: message.text(),
      resourceUrl: message.location()?.url,
    });
    const line = `${message.type()}: ${message.text().slice(0, 240)}`;
    if (verdict === "fail") consoleErrors.push(line);
    else if (verdict === "warn") consoleWarnings.push(line);
  });
  page.on("pageerror", (error) =>
    consoleErrors.push(`pageerror: ${error.message}`),
  );

  const check = (name, passed, detail = "") => {
    results.push({ name, passed });
    console.log(
      `  ${passed ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`,
    );
  };

  const metricText = () =>
    page.locator('[data-testid="metric-value"]').first().innerText();

  console.log(
    `\n${"=".repeat(66)}\n${playthrough.title} (${playthrough.slug})\n${"=".repeat(66)}`,
  );

  try {
    await page.goto(`${BASE}/play/${playthrough.slug}`, { waitUntil: "load" });
    // Games mount client-side; wait for the shell heading.
    await page
      .getByRole("heading", { name: playthrough.title })
      .waitFor({ timeout: 25000 });

    // The metric announcement is debounced by 600ms, so give the opening one
    // time to land before reading it. Sampling here matters: a metric that has
    // not been measured yet is exactly where a NaN leaks into speech, and by the
    // end of a playthrough it has been overwritten by a real value.
    await page.waitForTimeout(900);
    const openingSpeech = await liveRegionText(page);

    await playthrough.run({ page, check, metricText });

    // After the game's own checks, so it can't disturb their starting state,
    // and before the console check, so a KaTeX warning is counted.
    await checkMathDialog(page, check);

    console.log("\nAnnouncement hygiene");
    const closingSpeech = await liveRegionText(page);
    const spoken = `${openingSpeech} | ${closingSpeech}`;
    // Screen-reader-only text is easy to get wrong precisely because nobody
    // looks at it. The visible readout renders an unmeasured metric as "—";
    // this catches the case where the spoken version says "NaN percent" instead.
    const junk = ["NaN", "undefined", "Infinity", "null"].filter((token) =>
      spoken.includes(token),
    );
    check(
      "no NaN or undefined reaches a screen reader",
      junk.length === 0,
      junk.length > 0 ? `found ${junk.join(", ")} in: ${spoken.slice(0, 200)}` : "",
    );

    console.log("\nConsole hygiene");
    check(
      "no console errors or bug-shaped warnings during the playthrough",
      consoleErrors.length === 0,
      consoleErrors.slice(0, 3).join(" | "),
    );
    if (consoleWarnings.length > 0) {
      // Printed, not failed: an unclassified warning is worth a look but is not
      // proof of a bug. Promote its pattern in harness.mjs if it turns out to be.
      console.log(
        `  WARN  ${consoleWarnings.length} other console warning(s): ${consoleWarnings.slice(0, 3).join(" | ")}`,
      );
    }
  } catch (error) {
    check(`playthrough completed without throwing`, false, error.message);
  } finally {
    await page.close();
  }

  const failed = results.filter((result) => !result.passed);
  totalFailed += failed.length;
  totalChecks += results.length;
  console.log(
    `\n${results.length - failed.length}/${results.length} checks passed for ${playthrough.slug}`,
  );
}

await browser.close();

console.log(
  `\n${"=".repeat(66)}\n${totalChecks - totalFailed}/${totalChecks} checks passed overall`,
);
process.exit(totalFailed === 0 ? 0 : 1);
