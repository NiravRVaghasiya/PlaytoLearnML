/**
 * Browser playthrough for Sort-It Arcade.
 *
 * This harness earned its keep here: it found a 25-parameter boundary that could
 * win by 0.002, and an overfit verdict that depended on the player's route to it.
 * Both passed every unit test.
 */

export const slug = "sort-it-arcade";
export const title = "Sort-It Arcade";

export async function run({ page, check, metricText }) {
  const whyRegion = page.getByRole("region", { name: /why did that happen/i });
  const failure = page.locator('[data-testid="named-failure"]');

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
  check("why-card docked", await whyRegion.isVisible());

  // ── contract #3: live metric responds to keyboard-only input ──
  console.log("\nLive feedback (pedagogy contract #3)");
  const before = await metricText();

  const handle = page.getByRole("slider", { name: /Boundary handle 1 of 2/ });
  await handle.focus();
  check(
    "boundary handle is focusable",
    await handle.evaluate((el) => el === document.activeElement),
  );

  // `Home` is a large, deterministic move. Accuracy is NOT monotonic in handle
  // height — tilting the line one way hurts before it helps — so a few small
  // nudges can legitimately return to the starting accuracy.
  await page.keyboard.press("Home");
  await page.waitForTimeout(700);

  check(
    "keyboard action moves the boundary",
    (await handle.getAttribute("aria-valuenow")) === "100",
  );
  const after = await metricText();
  check(
    "metric changes after a keyboard-only action",
    before !== after,
    `${before} -> ${after}`,
  );
  check(
    "metric change is announced to screen readers",
    /accuracy/i.test(
      await page.locator('[aria-live="polite"]').first().innerText(),
    ),
  );

  // ── the win path, via the visual lane ──
  console.log("\nVisual lane: the intended solution wins");
  await page.getByRole("button", { name: /^Curve/ }).click();
  await page.getByRole("button", { name: /Fit it for me/ }).click();
  await page.waitForTimeout(300);
  await page.getByRole("button", { name: /Check generalization/ }).click();
  await page.waitForTimeout(600);

  const whyTitle = await whyRegion.innerText();
  check(
    "mid-capacity curve clears the round",
    /cleared/i.test(whyTitle),
    whyTitle.split("\n")[0],
  );
  check("no failure alert on a win", (await failure.count()) === 0);
  check(
    "XP bar reflects progress",
    await page
      .getByRole("progressbar", { name: "Experience points" })
      .isVisible(),
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

  // Read the output panel specifically — matching page text would also match
  // the snippet inside the editor, passing even if nothing ran.
  const output = await page
    .getByRole("region", { name: "Script output" })
    .innerText();
  check(
    "code lane executed and logged the verdict",
    /verdict\s+overfit/.test(output),
    output.replace(/\n/g, " | "),
  );

  const alertVisible = (await failure.count()) > 0;
  check("named failure alert appears after overfitting", alertVisible);

  if (alertVisible) {
    const text = await failure.innerText();
    check(
      "failure is NAMED, not generic",
      /overfit/i.test(text) && !/game over/i.test(text),
      text.split("\n")[0],
    );
    check(
      "failure detail carries real numbers",
      /\d+%/.test(text),
      text.replace(/\n/g, " ").slice(0, 110),
    );
    check(
      "retry is one click away inside the alert",
      (await failure.getByRole("button", { name: /retry/i }).count()) > 0,
    );
  }

  const metricClass = await page
    .locator('[data-testid="metric-value"]')
    .first()
    .getAttribute("class");
  check(
    "metric turns red on failure and stays visible",
    (metricClass ?? "").includes("text-wrong"),
  );
}
