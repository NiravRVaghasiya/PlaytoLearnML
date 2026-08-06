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
 * Usage:
 *   node scripts/verify-playthrough.mjs               # every game
 *   node scripts/verify-playthrough.mjs sort-it-arcade
 */

import { chromium } from "playwright";
import { readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

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

const browser = await chromium.launch();
let totalFailed = 0;
let totalChecks = 0;

for (const playthrough of playthroughs) {
  const results = [];
  const consoleErrors = [];

  const page = await browser.newPage();
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
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

    await playthrough.run({ page, check, metricText });

    console.log("\nConsole hygiene");
    check(
      "no console errors during the playthrough",
      consoleErrors.length === 0,
      consoleErrors.slice(0, 3).join(" | "),
    );
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
