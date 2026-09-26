/**
 * Browser playthrough for Neuron Forge.
 *
 * Every training run below is a real `model.fit`, so the waits are generous:
 * TensorFlow.js falls back to the CPU backend in headless Chromium here (the
 * software WebGL rasteriser lacks the float-texture extensions it needs), which
 * makes each run take seconds rather than milliseconds.
 *
 * The two named failures are checked as a *pair* on the same puzzle, because the
 * whole lesson is that they need different fixes: one wants a different
 * activation, the other wants more neurons.
 */

export const slug = "neuron-forge";
export const title = "Neuron Forge";

/** Training swaps the Train button for a Stop button. Wait for it to swap back. */
async function waitForTraining(page) {
  const train = page.getByRole("button", { name: "Train network" });
  await page
    .getByRole("button", { name: "Stop training" })
    .waitFor({ state: "visible", timeout: 20000 })
    .catch(() => {
      // A very fast run can finish before we look. Not a failure.
    });
  await train.waitFor({ state: "visible", timeout: 180000 });
}

export async function run({ page, check, metricText }) {
  const whyRegion = page.getByRole("region", { name: /why did that happen/i });
  const failure = page.locator('[data-testid="named-failure"]');
  const patternSelect = page.getByLabel("Pattern", { exact: true });
  const addLayer = page.getByRole("button", { name: "Add a layer" });
  const trainButton = page.getByRole("button", { name: "Train network" });
  const surface = () => page.getByRole("img", { name: /surface|Scatter plot/ });

  // ── shell anatomy ──
  console.log("\nShell anatomy");
  check(
    "metric readout present",
    (await page.locator('[data-testid="metric-value"]').count()) > 0,
  );
  check("pattern selector present", await patternSelect.isVisible());
  check(
    "compute budget is shown as a real progressbar",
    await page
      .getByRole("progressbar", { name: /Compute budget/ })
      .isVisible(),
  );
  check("why-card docked", await whyRegion.isVisible());
  check(
    "decision surface renders before any training",
    await surface().isVisible(),
  );
  check(
    "network diagram present",
    /no hidden layers|hidden neuron/i.test(await page.locator("main").innerText()),
  );

  // ── contract #2: the controls ARE the architecture ──
  console.log("\nPlayer action = architecture (pedagogy contract #2)");
  await patternSelect.selectOption("circle");
  await page.waitForTimeout(250);

  const budget = page.getByRole("progressbar", { name: /Compute budget/ });
  check(
    "switching pattern resets the budget to unspent",
    (await budget.getAttribute("aria-valuenow")) === "0",
    `aria-valuenow=${await budget.getAttribute("aria-valuenow")}`,
  );
  check(
    "circle puzzle advertises its 8-neuron budget",
    (await budget.getAttribute("aria-valuemax")) === "8",
  );

  await addLayer.click();
  await page.waitForTimeout(200);
  check(
    "adding a layer spends budget",
    (await budget.getAttribute("aria-valuenow")) === "1",
    `aria-valuenow=${await budget.getAttribute("aria-valuenow")}`,
  );

  const grow = page.getByRole("button", { name: /Add a neuron to layer 1/ });
  await grow.focus();
  check(
    "neuron stepper is keyboard focusable",
    await grow.evaluate((el) => el === document.activeElement),
  );
  await page.keyboard.press("Enter");
  await page.keyboard.press("Enter");
  await page.keyboard.press("Enter");
  await page.waitForTimeout(200);
  check(
    "keyboard alone can build a 4-neuron layer",
    (await budget.getAttribute("aria-valuenow")) === "4",
    `aria-valuenow=${await budget.getAttribute("aria-valuenow")}`,
  );
  check(
    "the diagram describes the architecture in text",
    /4 relu neurons/i.test(await page.locator("main").innerText()),
  );

  // ── contract #3: live feedback while a real network trains ──
  console.log("\nLive feedback while training (pedagogy contract #3)");
  const lossBefore = await metricText();
  await trainButton.click();
  await waitForTraining(page);
  const lossAfter = await metricText();

  check(
    "training moves the live loss metric",
    lossBefore !== lossAfter,
    `${lossBefore} -> ${lossAfter}`,
  );
  check(
    "the loss curve is drawn from real epoch logs",
    await page.getByRole("img", { name: /Loss curve over \d+ /i }).isVisible(),
  );
  check(
    "the decision surface now shows a learned boundary",
    /Decision surface/i.test((await surface().getAttribute("aria-label")) ?? ""),
  );
  // Any polite live region, not just the first: the shell may announce the
  // why-card headline in a region of its own, ahead of the metric's in the DOM.
  check(
    "result is announced to screen readers",
    (await page.locator('[aria-live="polite"]').filter({ hasText: /loss/i }).count()) > 0,
  );

  // 4 relu neurons is the measured minimal solution for the circle.
  check(
    "4 relu neurons solves the circle",
    (await failure.count()) === 0,
    (await failure.count()) === 0 ? "no failure alert" : await failure.innerText(),
  );
  check(
    // Located by its text rather than by container: GameShell renders the
    // secondary metrics in a grid sibling of <main>, not inside it.
    "held-out accuracy is reported, not training accuracy",
    await page.getByText("Held-out accuracy").first().isVisible(),
  );
  check(
    "the why-card headline carries the outcome and its number",
    /Solved: \d+% held-out/.test(await whyRegion.innerText()),
    (await whyRegion.innerText()).split("\n")[0],
  );

  // ── contract #4a: No non-linearity ──
  console.log("\nNamed failure: No non-linearity");
  // Same neurons, same depth. One dropdown.
  await page.getByLabel("Activation for layer 1").selectOption("linear");
  await page.waitForTimeout(200);
  check(
    "the why-card warns before training, as a prediction prompt",
    /linear/i.test(await whyRegion.innerText()),
  );

  await trainButton.click();
  await waitForTraining(page);

  const linearFailed = (await failure.count()) > 0;
  check("changing only the activation breaks it", linearFailed);
  if (linearFailed) {
    const text = await failure.innerText();
    check(
      "failure is NAMED 'No non-linearity', not generic",
      /no non-linearity/i.test(text) && !/game over/i.test(text),
      text.split("\n")[0],
    );
    check(
      "failure detail carries real numbers",
      /\d+%/.test(text),
      text.replace(/\n/g, " ").slice(0, 140),
    );
    check(
      "failure points at the activation, not at the neuron count",
      /activation/i.test(text),
    );
    check(
      "retry is one click away inside the alert",
      (await failure.getByRole("button", { name: /retry/i }).count()) > 0,
    );

    // The failure says "change one activation". Retry must leave the network
    // it is talking about in place, not wipe it.
    await failure.getByRole("button", { name: /retry/i }).click();
    await page.waitForTimeout(250);
    check(
      "retry clears the result but keeps the architecture",
      (await failure.count()) === 0 &&
        (await budget.getAttribute("aria-valuenow")) === "4" &&
        (await page.getByLabel("Activation for layer 1").inputValue()) === "linear",
      `failures=${await failure.count()} budget=${await budget.getAttribute("aria-valuenow")}`,
    );
  }

  // ── contract #4b: Insufficient capacity, a DIFFERENT diagnosis ──
  console.log("\nNamed failure: Insufficient capacity (different fix)");
  await page.getByLabel("Activation for layer 1").selectOption("relu");
  // Shrink back to 2 neurons: non-linear, but not enough of it.
  const shrink = page.getByRole("button", { name: /Remove a neuron from layer 1/ });
  await shrink.click();
  await shrink.click();
  await page.waitForTimeout(200);
  check(
    "back to a 2-neuron relu layer",
    (await budget.getAttribute("aria-valuenow")) === "2",
    `aria-valuenow=${await budget.getAttribute("aria-valuenow")}`,
  );

  await trainButton.click();
  await waitForTraining(page);

  const capacityFailed = (await failure.count()) > 0;
  check("too few neurons fails", capacityFailed);
  if (capacityFailed) {
    const text = await failure.innerText();
    check(
      "this failure is named 'Insufficient capacity' — a different diagnosis",
      /insufficient capacity/i.test(text),
      text.split("\n")[0],
    );
    check(
      "and it advises spending budget, since budget remains",
      /budget to spend/i.test(text),
      text.replace(/\n/g, " ").slice(0, 140),
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

  // ── budget is a hard cap, enforced by disabling ──
  console.log("\nThe compute budget is a cap, not a penalty");
  await page.getByRole("button", { name: /Add a neuron to layer 1/ }).click();
  await page.getByRole("button", { name: /Add a neuron to layer 1/ }).click();
  await page.getByRole("button", { name: /Add a neuron to layer 1/ }).click();
  await page.getByRole("button", { name: /Add a neuron to layer 1/ }).click();
  await page.getByRole("button", { name: /Add a neuron to layer 1/ }).click();
  await page.getByRole("button", { name: /Add a neuron to layer 1/ }).click();
  await page.waitForTimeout(250);
  check(
    "neurons stop at the 8-neuron budget",
    (await budget.getAttribute("aria-valuenow")) === "8",
    `aria-valuenow=${await budget.getAttribute("aria-valuenow")}`,
  );
  check(
    "the grow button is disabled rather than failing the player",
    await page
      .getByRole("button", { name: /Add a neuron to layer 1/ })
      .isDisabled(),
  );
  check(
    "adding another layer is disabled at the cap",
    await addLayer.isDisabled(),
  );

  // React Flow only fits the nodes of its FIRST render. The diagram started
  // with three nodes; eight hidden neurons later it must still show them all.
  const clipped = await page.evaluate(() => {
    const pane = document.querySelector(".react-flow")?.getBoundingClientRect();
    if (!pane) return -1;
    return [...document.querySelectorAll(".react-flow__node")].filter((node) => {
      const box = node.getBoundingClientRect();
      return (
        box.left < pane.left - 1 ||
        box.right > pane.right + 1 ||
        box.top < pane.top - 1 ||
        box.bottom > pane.bottom + 1
      );
    }).length;
  });
  check(
    "the network diagram refits: every neuron is inside the frame",
    clipped === 0,
    `${clipped} node(s) outside the pane`,
  );

  // ── the code lane, sharing the same store and the same trainer ──
  console.log("\nCode lane: same store, same model");
  await page.getByRole("radio", { name: /code/i }).click();
  const editor = page.getByLabel("Architecture script");
  check("code lane editor present", await editor.isVisible());

  // ── the capacity ladder, measured in the browser the player uses ──
  // The unit tests measure it on Node's CPU backend; numbers differ by 1–3
  // points between backends, so the lesson is re-proved here with a margin.
  console.log("\nThe capacity ladder, in the browser");
  await editor.fill(
    [
      "const rungs = [",
      "  ['circle', [{ neurons: 3, activation: 'relu' }], [{ neurons: 4, activation: 'relu' }]],",
      "  ['xor',    [{ neurons: 3, activation: 'relu' }], [{ neurons: 4, activation: 'relu' }]],",
      "  ['spiral', [{ neurons: 8, activation: 'relu' }],",
      "             [{ neurons: 8, activation: 'relu' }, { neurons: 8, activation: 'relu' }]],",
      "];",
      "for (const [pattern, below, minimal] of rungs) {",
      "  api.setPattern(pattern);",
      "  api.setLayers(below);   const low = await api.train();",
      "  api.setLayers(minimal); const high = await api.train();",
      "  log('RUNG', pattern, low.accuracy.toFixed(3), high.accuracy.toFixed(3), api.target());",
      "}",
      "// The winning 8→8 with a third layer of 3 behind it: it holds the solution.",
      "api.setLayers([{ neurons: 8, activation: 'relu' }, { neurons: 8, activation: 'relu' },",
      "               { neurons: 3, activation: 'relu' }]);",
      "const deep = await api.train();",
      "log('CONTAINED', deep.outcome, deep.accuracy.toFixed(3));",
      "// The whole spiral budget in three layers: measured to die, not to lack capacity.",
      "api.setLayers([{ neurons: 8, activation: 'relu' }, { neurons: 8, activation: 'relu' },",
      "               { neurons: 4, activation: 'relu' }]);",
      "const dead = await api.train();",
      "log('DEADNET', dead.outcome, dead.failure && dead.failure.name);",
    ].join("\n"),
  );
  await page.getByRole("button", { name: "Run", exact: true }).click();
  await page
    .getByRole("region", { name: "Script output" })
    .getByText(/DEADNET/)
    .waitFor({ timeout: 900000 });
  const ladder = await page
    .getByRole("region", { name: "Script output" })
    .innerText();
  const rungs = ladder.split("\n").filter((l) => l.startsWith("RUNG"));
  check("all three puzzles were measured", rungs.length === 3, `${rungs.length} rung(s)`);
  for (const line of rungs) {
    const [, pattern, low, high, target] = line.split(/\s+/);
    check(
      `${pattern}: the rung below misses and the minimal solution clears, 2+ points either side`,
      Number(low) <= Number(target) - 0.02 && Number(high) >= Number(target) + 0.02,
      line,
    );
  }
  // Whatever this backend makes of 8→8→3 — a win, a dead layer, or a miss that
  // still varies — it holds the minimal solution, so a capacity verdict is wrong.
  const contained = ladder.split("\n").find((l) => l.startsWith("CONTAINED")) ?? "";
  check(
    "a network holding the minimal solution is never called Insufficient capacity",
    /^CONTAINED (win|dead-network|optimisation-failure) /.test(contained),
    contained || ladder.slice(0, 120),
  );
  check(
    "a dead relu stack is named Dying ReLU, not Insufficient capacity",
    /DEADNET dead-network Dying ReLU/.test(ladder),
    ladder.split("\n").find((l) => l.startsWith("DEADNET")) ?? ladder.slice(0, 120),
  );
  check(
    "the Dying ReLU failure strip is shown",
    /Dying ReLU/.test((await failure.count()) > 0 ? await failure.innerText() : ""),
  );

  await editor.fill(
    [
      "api.setPattern('circle');",
      "api.setLayers([{ neurons: 4, activation: 'linear' },",
      "               { neurons: 4, activation: 'linear' }]);",
      "const flat = await api.train();",
      "log('8 linear:', (flat.accuracy * 100).toFixed(1) + '%', flat.outcome);",
      "api.setLayers([{ neurons: 4, activation: 'relu' }]);",
      "const bent = await api.train();",
      "log('4 relu:  ', (bent.accuracy * 100).toFixed(1) + '%', bent.outcome);",
      "log('neurons used', api.neurons(), 'of', api.budget());",
    ].join("\n"),
  );
  await page.getByRole("button", { name: "Run", exact: true }).click();

  // Two real trainings.
  await page
    .getByRole("region", { name: "Script output" })
    .getByText(/4 relu:/)
    .waitFor({ timeout: 240000 });
  const output = await page
    .getByRole("region", { name: "Script output" })
    .innerText();

  check(
    "the whole budget spent on linear layers fails",
    /8 linear:.*no-nonlinearity/.test(output),
    output.replace(/\n/g, " | "),
  );
  check(
    "HALF the neurons with relu wins",
    /4 relu:.*win/.test(output),
    output.replace(/\n/g, " | "),
  );
  check(
    "the code lane wrote to the same store the visual lane reads",
    /neurons used 4 of 8/.test(output),
  );
  check("no failure alert after the winning run", (await failure.count()) === 0);
  check(
    "XP bar reflects progress",
    await page
      .getByRole("progressbar", { name: "Experience points" })
      .isVisible(),
  );
}
