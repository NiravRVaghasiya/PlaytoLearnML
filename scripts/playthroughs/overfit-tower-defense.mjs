/**
 * Browser playthrough for Overfit Tower Defense.
 *
 * The two named failures are checked as a pair, on purpose. They have opposite
 * fixes — one says add regularization, the other says take it away — so a game
 * that can only demonstrate one of them has not taught the tradeoff at all.
 *
 * Every deployment below trains a real model on the CPU backend (TF.js declines
 * the software WebGL rasteriser in headless Chromium), hence the long waits.
 */

export const slug = "overfit-tower-defense";
export const title = "Overfit Tower Defense";

/** Deploying swaps the buttons for a Stop button. Wait for it to come back. */
async function waitForTraining(page) {
  await page
    .getByRole("button", { name: "Stop training" })
    .waitFor({ state: "visible", timeout: 25000 })
    .catch(() => {
      // A fast run can finish before we look. Not a failure.
    });
  await page
    .getByRole("button", { name: /Deploy against wave|Send wave|New run/ })
    .first()
    .waitFor({ state: "visible", timeout: 240000 });
}

export async function run({ page, check, metricText }) {
  const whyRegion = page.getByRole("region", { name: /why did that happen/i });
  const failure = page.locator('[data-testid="named-failure"]');
  const complexity = page.getByRole("slider", { name: /Model complexity/ });
  const core = page.getByRole("progressbar", { name: /Core health/ });
  // Scoped tightly: the battlefield's label also mentions the gap, so a looser
  // name matches two images.
  const gapMeter = () =>
    page.getByRole("img", { name: /Training accuracy|No model trained yet/i });
  const outputRegion = page.getByRole("region", { name: "Script output" });

  // ── shell anatomy ──
  console.log("\nShell anatomy");
  check(
    "metric readout present",
    (await page.locator('[data-testid="metric-value"]').count()) > 0,
  );
  check("complexity slider is a real slider", await complexity.isVisible());
  check("core health is a real progressbar", await core.isVisible());
  check(
    "core starts at full health",
    (await core.getAttribute("aria-valuenow")) === "100",
    `aria-valuenow=${await core.getAttribute("aria-valuenow")}`,
  );
  check("why-card docked", await whyRegion.isVisible());
  check("gap meter present before any training", await gapMeter().isVisible());
  check(
    "battlefield present",
    await page.getByRole("img", { name: /attackers/i }).isVisible(),
  );

  // ── contract #2: the controls are the model ──
  console.log("\nPlayer action = the bias-variance knob (contract #2)");
  await complexity.focus();
  check(
    "complexity slider is keyboard focusable",
    await complexity.evaluate((el) => el === document.activeElement),
  );
  // Read inputValue, not aria-valuenow: this is a native range input, so the
  // browser derives the ARIA value from the DOM value rather than an attribute.
  const before = await complexity.inputValue();
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("ArrowRight");
  check(
    "arrow keys change complexity",
    (await complexity.inputValue()) !== before,
    `${before} -> ${await complexity.inputValue()}`,
  );
  check(
    "parameter count is stated against the data size",
    /parameters for \d+ training points/i.test(
      await page.locator("main, aside, form").first().innerText().catch(() => ""),
    ) || /per point/i.test(await page.locator("body").innerText()),
  );

  const addL2 = page.getByRole("button", { name: /Add a L2 tower/ });
  check("L2 tower control present", await addL2.isVisible());
  await addL2.click();
  await page.waitForTimeout(200);
  check(
    "adding a tower reports the lambda it produces",
    /λ₂\s*0\.012/.test(await page.locator("body").innerText()),
    (await page.locator("body").innerText()).match(/λ₂[^\n]*/)?.[0] ?? "",
  );
  check(
    "the why-card explains what L2 actually does",
    /square/i.test(await whyRegion.innerText()),
  );

  // ── contract #3: live gap while a real model trains ──
  console.log("\nLive train/validation gap (contract #3)");
  const gapBefore = await metricText();
  await page.getByRole("button", { name: /Trial run/ }).click();
  await waitForTraining(page);
  const gapAfter = await metricText();

  check(
    "training moves the live gap metric",
    gapBefore !== gapAfter,
    `${gapBefore} -> ${gapAfter}`,
  );
  check(
    "a trial costs the core nothing",
    (await core.getAttribute("aria-valuenow")) === "100",
    `aria-valuenow=${await core.getAttribute("aria-valuenow")}`,
  );
  check(
    "the gap meter now reports both accuracies",
    /Training accuracy \d+ percent, validation accuracy \d+ percent/i.test(
      (await gapMeter().getAttribute("aria-label")) ?? "",
    ),
    ((await gapMeter().getAttribute("aria-label")) ?? "").slice(0, 110),
  );
  check(
    "bias is reported separately from the gap",
    /Bias/.test(await page.locator("body").innerText()),
  );
  // The announcement is debounced (~600ms), so poll for it rather than reading
  // once and racing the timer.
  const announced = await page
    .waitForFunction(
      () =>
        /gap/i.test(
          document.querySelector('[aria-live="polite"]')?.textContent ?? "",
        ),
      null,
      { timeout: 10000 },
    )
    .then(() => true)
    .catch(() => false);
  check(
    "the result is announced to screen readers",
    announced,
    (await page.locator('[aria-live="polite"]').first().innerText()).slice(0, 60),
  );

  // ── the code lane: the sweep that shows the U-shape ──
  console.log("\nCode lane: a real regularization sweep");
  await page.getByRole("radio", { name: /code/i }).click();
  const editor = page.getByLabel("Defence script");
  check("code lane editor present", await editor.isVisible());

  // Jump to the famine wave, where regularization has the most to prove.
  await editor.fill(
    [
      "api.restart();",
      "// Walk to wave 4 by deploying a well-behaved model each time.",
      "for (let w = 1; w <= 3; w++) {",
      "  api.setComplexity(8); api.setTowers({ l1: 0, l2: 1, dropout: 0 });",
      "  await api.deploy();",
      "  api.nextWave();",
      "}",
      "log('wave', api.wave(), '| points', api.trainPoints(), '| core', Math.round(api.coreHp()));",
      "api.setComplexity(32);",
      "api.setTowers({ l1: 0, l2: 0, dropout: 0 });",
      "const bare = await api.trial();",
      "log('no towers  val', pct(bare.validationAccuracy), 'gap', pct(bare.gap), 'bias', pct(bare.bias));",
      "api.setTowers({ l1: 3, l2: 0, dropout: 0 });",
      "const held = await api.trial();",
      "log('L1 x3      val', pct(held.validationAccuracy), 'gap', pct(held.gap), 'bias', pct(held.bias));",
      "api.setTowers({ l1: 6, l2: 6, dropout: 6 });",
      "const over = await api.trial();",
      "log('everything val', pct(over.validationAccuracy), 'gap', pct(over.gap), 'bias', pct(over.bias));",
      "log('verdict', held.validationAccuracy > bare.validationAccuracy ? 'regularization helped' : 'no help');",
      "log('overdone', over.bias > held.bias ? 'yes' : 'no');",
      "function pct(x) { return (x * 100).toFixed(1) + '%'; }",
    ].join("\n"),
  );
  await page.getByRole("button", { name: "Run", exact: true }).click();
  await outputRegion.getByText(/overdone/).waitFor({ timeout: 600000 });

  const output = await outputRegion.innerText();
  check(
    "the sweep reaches the famine wave",
    /wave 4 \| points 90/.test(output),
    output.split("\n").find((l) => l.startsWith("wave")) ?? "",
  );
  check(
    "regularization raises validation accuracy, not just lowers the gap",
    /verdict regularization helped/.test(output),
    output.replace(/\n/g, " | "),
  );
  check(
    "over-regularizing raises bias — the other failure mode",
    /overdone yes/.test(output),
    output.replace(/\n/g, " | "),
  );
  check(
    "the code lane wrote to the same store the visual lane reads",
    (await core.getAttribute("aria-valuenow")) !== "100",
    `core aria-valuenow=${await core.getAttribute("aria-valuenow")}`,
  );

  // ── contract #4a: Overfitting ──
  console.log("\nNamed failure: Overfitting");
  // Maximum capacity, no defence, every wave. One bad deployment is survivable
  // by design — it takes a sustained refusal to regularise to lose the core.
  await editor.fill(
    [
      "api.restart();",
      "api.setComplexity(32);",
      "api.setTowers({ l1: 0, l2: 0, dropout: 0 });",
      "for (let w = 1; w <= 5; w++) {",
      "  await api.deploy();",
      "  log('wave ' + w, 'core', Math.round(api.coreHp()),",
      "      'gap', (api.lastResult().gap * 100).toFixed(0) + '%',",
      "      api.lastResult().dominant);",
      "  if (api.phase() !== 'resolved') break;",
      "  api.nextWave();",
      "}",
      "log('phase', api.phase());",
      // Unique sentinel: the log panel keeps earlier output, so a shared marker
      // would match the previous run and the wait would return instantly.
      "log('NODEFENCE', api.lastResult().dominant);",
    ].join("\n"),
  );
  await page.getByRole("button", { name: "Run", exact: true }).click();
  await outputRegion.getByText(/NODEFENCE/).waitFor({ timeout: 900000 });

  const overfitOut = await outputRegion.innerText();
  check(
    "an unregularised model is dominated by the gap",
    /NODEFENCE overfit/.test(overfitOut),
    overfitOut.replace(/\n/g, " | "),
  );
  check(
    "never regularising destroys the core",
    /phase lost/.test(overfitOut),
    overfitOut.replace(/\n/g, " | "),
  );

  const overfitFailed = (await failure.count()) > 0;
  if (overfitFailed) {
    const text = await failure.innerText();
    check(
      "failure is NAMED 'Overfitting', not generic",
      /overfitting/i.test(text) && !/game over/i.test(text),
      text.split("\n")[0],
    );
    check(
      "failure detail carries real numbers",
      /\d+%/.test(text),
      text.replace(/\n/g, " ").slice(0, 130),
    );
    check(
      "it advises ADDING regularization",
      /L2 or dropout|not enough/i.test(text),
    );
    check(
      "retry is one click away inside the alert",
      (await failure.getByRole("button", { name: /retry/i }).count()) > 0,
    );
  } else {
    check("named failure alert appears after the core falls", false);
  }

  // ── contract #4b: Underfitting, the opposite diagnosis ──
  console.log("\nNamed failure: Underfitting (opposite fix)");
  // Over-defend everywhere: tiny model, every tower up. The gap stays near zero
  // the whole way, which is exactly why the core still dies.
  await editor.fill(
    [
      "api.restart();",
      "api.setComplexity(1);",
      "api.setTowers({ l1: 6, l2: 6, dropout: 6 });",
      "for (let w = 1; w <= 5; w++) {",
      "  await api.deploy();",
      "  log('wave ' + w, 'core', Math.round(api.coreHp()),",
      "      'gap', (api.lastResult().gap * 100).toFixed(0) + '%',",
      "      'bias', (api.lastResult().bias * 100).toFixed(0) + '%',",
      "      api.lastResult().dominant);",
      "  if (api.phase() !== 'resolved') break;",
      "  api.nextWave();",
      "}",
      "log('phase', api.phase());",
      "log('OVERDEFENDED', api.lastResult().dominant);",
    ].join("\n"),
  );
  await page.getByRole("button", { name: "Run", exact: true }).click();
  await outputRegion.getByText(/OVERDEFENDED/).waitFor({ timeout: 900000 });

  const underfitOut = await outputRegion.innerText();
  check(
    "over-defending is dominated by bias, not by the gap",
    /OVERDEFENDED underfit/.test(underfitOut),
    underfitOut.replace(/\n/g, " | "),
  );
  check(
    "over-defending also destroys the core — the opposite mistake, same result",
    /phase lost/.test(underfitOut),
    underfitOut.replace(/\n/g, " | "),
  );

  if ((await failure.count()) > 0) {
    const text = await failure.innerText();
    check(
      "this failure is named 'Underfitting' — a different diagnosis",
      /underfitting/i.test(text),
      text.split("\n")[0],
    );
    check(
      "and it advises REMOVING regularization — the opposite advice",
      /take towers down|raise complexity/i.test(text),
      text.replace(/\n/g, " ").slice(0, 130),
    );
    check(
      "it explicitly rules out overfitting as the cause",
      /overfitting was never your problem/i.test(text),
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
  check(
    "XP bar present",
    await page
      .getByRole("progressbar", { name: "Experience points" })
      .isVisible(),
  );
}
