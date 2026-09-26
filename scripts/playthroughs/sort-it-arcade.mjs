/**
 * Browser playthrough for Sort-It Arcade.
 *
 * This harness earned its keep here: it found a 25-parameter boundary that could
 * win by 0.002, and an overfit verdict that depended on the player's route to it.
 * Both passed every unit test.
 */

export const slug = "sort-it-arcade";
export const title = "Sort-It Arcade";

/**
 * Screen point of a field coordinate (0–1 on both axes), projected through the
 * SVG's own painted transform — the same `getScreenCTM()` the lane inverts. A
 * bounding-box projection would share the bug it is meant to catch.
 */
async function fieldPoint(page, x, y) {
  return page.evaluate(
    ([fx, fy]) => {
      const svg = document.querySelector(
        'svg[aria-label="Classification field"]',
      );
      const m = svg.getScreenCTM();
      // viewBox units, matching the lane's scales (PAD = 6 of 100).
      const vx = 6 + fx * 88;
      const vy = 94 - fy * 88;
      return { x: m.a * vx + m.c * vy + m.e, y: m.b * vx + m.d * vy + m.f };
    },
    [x, y],
  );
}

/** Height (0–100) of a handle, from its slider value. */
async function handleValue(page, index, count) {
  return Number(
    await page
      .getByRole("slider", { name: `Boundary handle ${index} of ${count}` })
      .getAttribute("aria-valuenow"),
  );
}

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

  // `End` is a large, deterministic move. Accuracy is NOT monotonic in handle
  // height — tilting the line one way hurts before it helps — so a few small
  // nudges can legitimately return to the starting accuracy.
  await page.keyboard.press("End");
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

  // ARIA slider pattern: Home is the minimum. It used to be inverted.
  await page.keyboard.press("Home");
  await page.waitForTimeout(150);
  check(
    "Home sends a handle to the bottom (aria-valuemin)",
    (await handle.getAttribute("aria-valuenow")) === "0",
  );
  await page.keyboard.press("End");

  // ── pointer: drag, tap, and the 25-handle wiggle ──
  // 1024×768 is where the field renders taller than wide and letterboxes, so a
  // bounding-box mapping put a drag to 90 at 80. The desktop layout starts at
  // 1024, so no sticky metric overlaps the field here.
  console.log("\nPointer input (drag, single tap, dense handles)");
  const viewport = page.viewportSize();
  await page.setViewportSize({ width: 1024, height: 768 });
  // Scroll so the painted square (not the taller element box) sits just under
  // the top of the viewport: every target below is then on screen.
  await page.evaluate(() => {
    const svg = document.querySelector(
      'svg[aria-label="Classification field"]',
    );
    window.scrollBy(0, svg.getScreenCTM().f - 80);
  });
  await page.waitForTimeout(300);

  let from = await fieldPoint(page, 1, 0.5);
  let to = await fieldPoint(page, 1, 0.9);
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  await page.mouse.move(to.x, to.y, { steps: 6 });
  await page.mouse.up();
  await page.waitForTimeout(150);
  const dragged = await handleValue(page, 2, 2);
  check(
    "a drag lands where the pointer is on a letterboxed field",
    Math.abs(dragged - 90) <= 1,
    `handle 2 at ${dragged}, target 90`,
  );

  // WCAG 2.5.7: one tap, no drag, moves the nearest handle there.
  const tap = await fieldPoint(page, 0.1, 0.2);
  await page.mouse.click(tap.x, tap.y);
  await page.waitForTimeout(150);
  const tapped = await handleValue(page, 1, 2);
  check(
    "a single tap moves the nearest handle to the tapped height",
    Math.abs(tapped - 20) <= 1,
    `handle 1 at ${tapped}, target 20`,
  );

  // 25 handles are 3.7 viewBox units apart; overlapping hit circles used to
  // send a press on handle 6 to handle 7.
  await page.getByRole("button", { name: /^Wiggle/ }).click();
  await page.waitForTimeout(300);
  const sixBefore = await handleValue(page, 6, 25);
  const sevenBefore = await handleValue(page, 7, 25);
  from = await fieldPoint(page, 5 / 24, sixBefore / 100);
  to = await fieldPoint(page, 5 / 24, 0.15);
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  await page.mouse.move(to.x, to.y, { steps: 6 });
  await page.mouse.up();
  await page.waitForTimeout(150);
  const six = await handleValue(page, 6, 25);
  const seven = await handleValue(page, 7, 25);
  check(
    "pressing a wiggle handle drags that handle, not its neighbour",
    Math.abs(six - 15) <= 1 && seven === sevenBefore,
    `handle 6 ${sixBefore}->${six}, handle 7 ${sevenBefore}->${seven}`,
  );

  // Back to the route the rest of this playthrough is screened on: a flat
  // line with handle 1 at the top.
  if (viewport) await page.setViewportSize(viewport);
  await page.getByRole("button", { name: /^Line/ }).click();
  await page.getByRole("button", { name: "Retry" }).last().click();
  await handle.focus();
  await page.keyboard.press("End");
  await page.waitForTimeout(300);

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
