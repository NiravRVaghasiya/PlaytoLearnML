/**
 * Browser playthrough for K-Means Territory Wars.
 *
 * Unit tests prove the clustering is right; this proves it's wired to the screen.
 * Asserts the shell anatomy, that the live metric moves on keyboard-only input,
 * that both lanes drive the same board, and that the named failures appear with
 * real numbers — plus no console errors.
 */

export const slug = "k-means-territory-wars";
export const title = "K-Means Territory Wars";

export async function run({ page, check, metricText }) {
  const outputText = () =>
    page.getByRole("region", { name: "Script output" }).innerText();
  const whyText = () =>
    page.getByRole("region", { name: /why did that happen/i }).innerText();
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
    "elbow chart present",
    await page.getByRole("heading", { name: "Elbow chart" }).isVisible(),
  );
  // The chart's data is also exposed as a real table for screen-reader users.
  await page.getByText("Inertia at each k (table)").click();
  check(
    "elbow data available as a table, not just a picture",
    await page.getByRole("table").first().isVisible(),
  );
  check(
    "assign and update are separate controls",
    (await page.getByRole("button", { name: /1\. Assign/ }).count()) === 1 &&
      (await page.getByRole("button", { name: /2\. Update/ }).count()) === 1,
  );

  // ── contract #3: live metric responds to keyboard-only input ──
  console.log("\nLive feedback (pedagogy contract #3)");
  const before = await metricText();

  const flag = page.getByRole("button", { name: /^Flag 1 of/ });
  await flag.focus();
  check(
    "flag is focusable",
    await flag.evaluate((el) => el === document.activeElement),
  );

  for (let i = 0; i < 8; i += 1) await page.keyboard.press("ArrowRight");
  await page.waitForTimeout(700);

  const after = await metricText();
  check(
    "metric changes after a keyboard-only action",
    before !== after,
    `${before} -> ${after}`,
  );
  check(
    "metric change is announced",
    /inertia/i.test(
      await page.locator('[aria-live="polite"]').first().innerText(),
    ),
  );

  // ── contract #2: assign and update are visibly distinct steps ──
  console.log("\nThe assign/update loop (pedagogy contract #2)");
  check(
    "moving a flag marks the assignment stale",
    /out of date/i.test(await page.locator("main").innerText()),
  );

  await page.getByRole("button", { name: /1\. Assign/ }).click();
  await page.waitForTimeout(400);
  check(
    "assign explains what it did",
    /assigned/i.test(await whyText()),
    (await whyText()).split("\n")[0],
  );

  await page.getByRole("button", { name: /2\. Update/ }).click();
  await page.waitForTimeout(400);
  const afterUpdate = await whyText();
  check(
    "update explains what it did",
    /moved|converged/i.test(afterUpdate),
    afterUpdate.split("\n")[0],
  );

  await page.getByRole("button", { name: /Run to convergence/ }).click();
  await page.waitForTimeout(600);
  check("settling reports convergence", /settled/i.test(await whyText()));

  // ── the "Bad k" trap: two flags is the wrong answer ──
  console.log("\nNamed failure: Bad k (pedagogy contract #4)");
  await page.getByRole("button", { name: /Score this map/ }).click();
  await page.waitForTimeout(600);

  const badKVisible = (await failure.count()) > 0;
  check("scoring two flags produces a failure", badKVisible);
  if (badKVisible) {
    const text = await failure.innerText();
    check(
      "failure is NAMED, not generic",
      /bad k/i.test(text) && !/game over/i.test(text),
      text.split("\n")[0],
    );
    check(
      "failure detail carries real numbers",
      /\d/.test(text),
      text.replace(/\n/g, " ").slice(0, 110),
    );
    check(
      "retry is one click away inside the alert",
      (await failure.getByRole("button", { name: /retry/i }).count()) > 0,
    );
  }

  // ── the code lane drives the same board, and can win ──
  console.log("\nCode lane (two-lane rule)");
  await page.getByRole("radio", { name: /code/i }).click();
  const editor = page.getByLabel("Clustering script");
  check("code lane editor present", await editor.isVisible());

  await editor.fill(
    [
      "api.reset();",
      "api.setK(3);",
      "api.scatterFlags();",
      "api.settle();",
      "const r = api.check();",
      "log('k', r.k);",
      "log('inertia', r.inertia.toFixed(3));",
      "log('verdict', r.outcome);",
    ].join("\n"),
  );
  // Exact name: the controls rail also has a "Run to convergence" button, and a
  // /^Run/ pattern matches both.
  const runButton = page.getByRole("button", { name: "Run", exact: true });

  await runButton.click();
  await page.waitForTimeout(2000);

  const output = await outputText();
  check(
    "code lane executed and reached the right answer",
    /verdict\s+win/.test(output),
    output.replace(/\n/g, " | "),
  );
  check(
    "no failure alert after a win",
    (await failure.count()) === 0,
  );
  check(
    "XP bar reflects progress",
    await page
      .getByRole("progressbar", { name: "Experience points" })
      .isVisible(),
  );

  // ── inertia falls as k rises: the trap, demonstrated live ──
  console.log("\nThe trap, demonstrated in the browser");
  await editor.fill(
    [
      "api.reset(); api.setK(3); api.scatterFlags(); api.settle();",
      "const good = api.inertia();",
      "api.setK(8); api.scatterFlags(); api.settle();",
      "const greedy = api.inertia();",
      "log('k=3 inertia', good.toFixed(3));",
      "log('k=8 inertia', greedy.toFixed(3));",
      "log('lower with more flags:', greedy < good);",
      "const r = api.check();",
      "log('verdict', r.outcome);",
    ].join("\n"),
  );
  await runButton.click();
  await page.waitForTimeout(2500);

  const trapOutput = await outputText();
  check(
    "more flags genuinely lower inertia",
    /lower with more flags:\s*true/.test(trapOutput),
    trapOutput.replace(/\n/g, " | "),
  );
  check(
    "and it is still judged a bad k, not rewarded",
    /verdict\s+bad-k/.test(trapOutput),
  );

  const metricClass = await page
    .locator('[data-testid="metric-value"]')
    .first()
    .getAttribute("class");
  check(
    "metric turns red on failure and stays visible",
    (metricClass ?? "").includes("text-wrong"),
  );
}
