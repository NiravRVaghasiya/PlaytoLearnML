/**
 * Browser playthrough for Data Detox.
 *
 * The first game whose live metric comes from a real TensorFlow.js retrain, so
 * this also checks that the model actually trains in the browser, that the meter
 * moves when the cleaning changes, and that repeated retrains don't fall over.
 */

export const slug = "data-detox";
export const title = "Data Detox";

export async function run({ page, check, metricText }) {
  const whyRegion = page.getByRole("region", { name: /why did that happen/i });
  const failure = page.locator('[data-testid="named-failure"]');
  const outputText = () =>
    page.getByRole("region", { name: "Script output" }).innerText();
  const runButton = page.getByRole("button", { name: "Run", exact: true });

  console.log("\nShell anatomy");
  check(
    "metric readout present",
    (await page.locator('[data-testid="metric-value"]').count()) > 0,
  );
  check(
    "conveyor belt present",
    await page.getByRole("region", { name: "Conveyor belt" }).isVisible(),
  );
  check(
    "four cleaning bins present",
    (await page
      .getByRole("region", { name: "Cleaning bins" })
      .getByRole("button")
      .count()) >= 4,
  );
  check("why-card docked", await whyRegion.isVisible());

  // ── the metric starts genuinely unknown, not faked at zero ──
  console.log("\nLive feedback (pedagogy contract #3)");
  const initial = await metricText();
  check(
    "accuracy is blank before any model exists",
    initial.trim() === "—",
    `showed "${initial.trim()}"`,
  );

  // ── contract #2: sorting rows IS the pipeline ──
  console.log("\nSorting rows (pedagogy contract #2)");
  await page.keyboard.press("2"); // Impute
  await page.waitForTimeout(250);
  const afterOne = await whyRegion.innerText();
  check(
    "sorting a row explains the tradeoff it just made",
    afterOne.length > 40,
    afterOne.split("\n").slice(0, 2).join(" / "),
  );

  // Sort a full batch by keyboard alone, which should trigger a retrain.
  for (let i = 0; i < 12; i += 1) {
    await page.keyboard.press("2");
    await page.waitForTimeout(60);
  }

  // A real TF.js fit takes a moment.
  await page.waitForFunction(
    () => {
      const el = document.querySelector('[data-testid="metric-value"]');
      return el && el.textContent && el.textContent.trim() !== "—";
    },
    undefined,
    { timeout: 30000 },
  );

  const trained = await metricText();
  check(
    "model trains in the browser and the meter fills",
    trained.trim() !== "—" && /%$/.test(trained.trim()),
    `accuracy ${trained.trim()}`,
  );
  check(
    "metric change is announced to screen readers",
    /accuracy/i.test(
      await page.locator('[aria-live="polite"]').first().innerText(),
    ),
  );

  // ── the code lane writes the whole pipeline, and wins ──
  console.log("\nCode lane: a pipeline as a function (two-lane rule)");
  await page.getByRole("radio", { name: /code/i }).click();
  const editor = page.getByLabel("Preprocessing pipeline");
  check("code lane editor present", await editor.isVisible());

  await editor.fill(
    [
      "api.reset();",
      "api.forEachRow((row) => {",
      "  if (row.isNull) return 'impute';",
      "  if (row.isOutlier) return 'cap';",
      "  return 'keep';",
      "});",
      "const acc = await api.retrain();",
      "log('kept', api.kept());",
      "log('blanks left', api.blanksLeft());",
      "log('drift', api.balanceDrift().toFixed(3));",
      "log('accuracy', acc.toFixed(3));",
      "log('verdict', api.check().outcome);",
    ].join("\n"),
  );
  await runButton.click();
  await page.waitForTimeout(6000);

  const goodOutput = await outputText();
  check(
    "a clean pipeline clears the round",
    /verdict\s+win/.test(goodOutput),
    goodOutput.replace(/\n/g, " | "),
  );
  check("no failure alert on a win", (await failure.count()) === 0);
  check(
    "XP bar reflects progress",
    await page
      .getByRole("progressbar", { name: "Experience points" })
      .isVisible(),
  );

  // ── contract #4: the named failure, driven by dropping blanks ──
  console.log("\nNamed failure: Selection bias (pedagogy contract #4)");
  await editor.fill(
    [
      "api.reset();",
      "api.forEachRow((row) => {",
      "  if (row.isNull) return 'drop';", // the mistake
      "  if (row.isOutlier) return 'cap';",
      "  return 'keep';",
      "});",
      "const acc = await api.retrain();",
      "log('kept', api.kept());",
      "log('drift', api.balanceDrift().toFixed(3));",
      "log('accuracy', acc.toFixed(3));",
      "log('verdict', api.check().outcome);",
    ].join("\n"),
  );
  await runButton.click();
  await page.waitForTimeout(6000);

  const biasOutput = await outputText();
  check(
    "dropping the blank rows is diagnosed as bias",
    /verdict\s+selection-bias/.test(biasOutput),
    biasOutput.replace(/\n/g, " | "),
  );
  check(
    "and it genuinely skewed the class balance",
    /drift\s+-0\.[12]/.test(biasOutput),
  );

  const alertVisible = (await failure.count()) > 0;
  check("named failure alert appears", alertVisible);
  if (alertVisible) {
    const text = await failure.innerText();
    check(
      "failure is NAMED, not generic",
      /selection bias/i.test(text) && !/game over/i.test(text),
      text.split("\n")[0],
    );
    check(
      "failure detail carries real numbers",
      /\d+% healthy/.test(text),
      text.replace(/\n/g, " ").slice(0, 120),
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

  // ── starvation, the catalog's named failure for this game ──
  console.log("\nNamed failure: Data starvation");
  await editor.fill(
    [
      "api.reset();",
      "api.forEachRow(() => 'drop');",
      "log('kept', api.kept());",
      "log('verdict', api.check().outcome);",
    ].join("\n"),
  );
  await runButton.click();
  await page.waitForTimeout(3000);

  const starvedOutput = await outputText();
  check(
    "dropping everything starves the model",
    /verdict\s+starved/.test(starvedOutput),
    starvedOutput.replace(/\n/g, " | "),
  );
  check(
    "starvation is named",
    /data starvation/i.test(await failure.innerText()),
  );
}
