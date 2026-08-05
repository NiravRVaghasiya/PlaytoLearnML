/**
 * End-to-end playthrough check for Sort-It Arcade.
 *
 * Unit tests prove the ML is right; this proves the ML is *wired to the screen*.
 * It drives a real browser and asserts the things the pedagogy contract and the
 * CLAUDE.md Definition of Done actually claim:
 *
 *   - the live metric moves in response to a keyboard-only player action
 *   - the visual lane is playable without a mouse
 *   - the code lane drives the same state and reaches the same verdict
 *   - the named failure mode appears, with numbers, when the player overfits
 *   - no console errors along the way
 *
 * Usage: node scripts/verify-playthrough.mjs   (needs a server on AUDIT_BASE)
 */

import { chromium } from "playwright";

const BASE = process.env.AUDIT_BASE ?? "http://localhost:3200";
const URL = `${BASE}/play/sort-it-arcade`;

const results = [];
const consoleErrors = [];

function check(name, passed, detail = "") {
  results.push({ name, passed, detail });
  console.log(`  ${passed ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
}

const browser = await chromium.launch();
const page = await browser.newPage();

page.on("console", (message) => {
  if (message.type() === "error") consoleErrors.push(message.text());
});
page.on("pageerror", (error) => consoleErrors.push(`pageerror: ${error.message}`));

const metricText = () =>
  page.locator('[data-testid="metric-value"]').first().innerText();

try {
  await page.goto(URL, { waitUntil: "load" });
  // Wait for the client-only game to mount.
  await page.getByRole("heading", { name: "Sort-It Arcade" }).waitFor({
    timeout: 20000,
  });

  console.log("\nShell anatomy");
  check(
    "metric readout present",
    (await page.locator('[data-testid="metric-value"]').count()) > 0,
  );
  check(
    "lane toggle present",
    await page.getByRole("radiogroup", { name: "Lane" }).isVisible(),
  );
  check(
    "math button present",
    await page.getByRole("button", { name: /math/i }).isVisible(),
  );
  check(
    "why-card docked",
    await page.getByRole("region", { name: /why did that happen/i }).isVisible(),
  );

  // ── contract #3: the metric moves on a player action, keyboard only ──
  console.log("\nLive feedback (pedagogy contract #3)");
  const before = await metricText();

  const handle = page.getByRole("slider", { name: /Boundary handle 1 of 2/ });
  await handle.focus();
  check("boundary handle is focusable", await handle.evaluate((el) => el === document.activeElement));

  // `Home` drives the handle to the top of the field — a large, deterministic
  // move. Note that accuracy is NOT monotonic in handle height (tilting the line
  // one way hurts before it helps), so a few small nudges can legitimately
  // return to the starting accuracy. Asserting on a nudge would be flaky for a
  // reason that has nothing to do with the wiring.
  await page.keyboard.press("Home");
  await page.waitForTimeout(700);

  const after = await metricText();
  const handleValue = await handle.getAttribute("aria-valuenow");
  check(
    "keyboard action moves the boundary",
    handleValue === "100",
    `handle at ${handleValue}`,
  );
  check(
    "metric changes after a keyboard-only action",
    before !== after,
    `${before} -> ${after}`,
  );

  const announcement = await page
    .locator('[aria-live="polite"]')
    .first()
    .innerText();
  check(
    "metric change is announced to screen readers",
    /accuracy/i.test(announcement),
    announcement.trim(),
  );

  // ── the win path, via the visual lane ──
  console.log("\nVisual lane: the intended solution wins");
  await page.getByRole("button", { name: /^Curve/ }).click();
  await page.getByRole("button", { name: /Fit it for me/ }).click();
  await page.waitForTimeout(300);
  await page.getByRole("button", { name: /Check generalization/ }).click();
  await page.waitForTimeout(600);

  const whyTitle = await page
    .getByRole("region", { name: /why did that happen/i })
    .innerText();
  check(
    "mid-capacity curve clears the round",
    /cleared/i.test(whyTitle),
    whyTitle.split("\n")[0],
  );
  check(
    "no failure alert on a win",
    (await page.locator('[data-testid="named-failure"]').count()) === 0,
  );
  check(
    "XP bar reflects progress",
    await page.getByRole("progressbar", { name: "Experience points" }).isVisible(),
  );

  // ── contract #4: the named failure, driven from the CODE lane ──
  console.log("\nCode lane + named failure (pedagogy contracts #2 and #4)");
  await page.getByRole("radio", { name: /code/i }).click();
  const editor = page.getByLabel("Boundary-fitting script");
  check("code lane editor present", await editor.isVisible());

  await editor.fill(
    [
      "api.setBoundaryType('wiggle');",
      "api.autoFit();",
      "const r = api.check();",
      "log('train', r.trainAccuracy.toFixed(3));",
      "log('test ', r.testAccuracy.toFixed(3));",
      "log('verdict', r.outcome);",
    ].join("\n"),
  );
  await page.getByRole("button", { name: /^Run/ }).click();
  await page.waitForTimeout(1500);

  // Read the output panel specifically. Matching on page text would also match
  // the snippet inside the editor, which would pass even if nothing ran.
  const output = await page
    .getByRole("region", { name: "Script output" })
    .innerText();
  check(
    "code lane executed and logged the verdict",
    /verdict\s+overfit/.test(output),
    output.replace(/\n/g, " | "),
  );

  const laneError = await page.locator('[role="alert"][id$="-error"]').count();
  check("code lane reported no script error", laneError === 0);

  // Scoped by testid: Next ships its own always-present role="alert" route
  // announcer, so an unscoped getByRole("alert") matches two elements.
  const alert = page.locator('[data-testid="named-failure"]');
  const alertVisible = (await alert.count()) > 0;
  check("named failure alert appears after overfitting", alertVisible);
  check(
    "the failure strip is exposed as an alert to assistive tech",
    alertVisible && (await alert.getAttribute("role")) === "alert",
  );

  if (alertVisible) {
    const alertText = await alert.innerText();
    check(
      "failure is NAMED, not generic",
      /overfit/i.test(alertText) && !/game over/i.test(alertText),
      alertText.split("\n")[0],
    );
    check(
      "failure detail carries real numbers",
      /\d+%/.test(alertText),
      alertText.replace(/\n/g, " ").slice(0, 110),
    );
    check(
      "retry is one click away inside the alert",
      (await alert.getByRole("button", { name: /retry/i }).count()) > 0,
    );
  }

  // Metric must be red on failure, and still visible.
  const metricClass = await page
    .locator('[data-testid="metric-value"]')
    .first()
    .getAttribute("class");
  check(
    "metric turns red on failure and stays visible",
    (metricClass ?? "").includes("text-wrong"),
  );

  console.log("\nConsole hygiene");
  check(
    "no console errors during the playthrough",
    consoleErrors.length === 0,
    consoleErrors.slice(0, 3).join(" | "),
  );
} finally {
  await browser.close();
}

const failed = results.filter((r) => !r.passed);
console.log(
  `\n${results.length - failed.length}/${results.length} checks passed`,
);
process.exit(failed.length === 0 ? 0 : 1);
