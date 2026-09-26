/**
 * Browser playthrough for Convolution Kitchen.
 *
 * What this verifies that no other playthrough can:
 *
 *   1. The sliding window is a real control with real arithmetic — nine products
 *      and a sum, recomputed as it moves. That is pedagogy contract #2 for this
 *      game, and it has to be operable by keyboard.
 *   2. A working feature map stops looking like the photograph. Checked through the
 *      gallery's reported peak and mean, since the tiles themselves are canvases.
 *   3. The single-layer ceiling is real IN THE BROWSER: good kernels, four of them,
 *      still fail, and the failure names the reason rather than blaming the player.
 *   4. Adding a second layer passes with three filters, and TF.js is doing genuine
 *      conv work — the raw-pixel baseline sits near chance, so nothing but the
 *      convolution could have earned the score.
 *
 * The stack is rescored on every edit, so most waits are for the metric to settle.
 */

export const slug = "convolution-kitchen";
export const title = "Convolution Kitchen";

/**
 * Scoring is async and the caption says so while it runs.
 *
 * The raw-pixel baseline is fitted once, AFTER the first score is on screen (it
 * used to hold the first score back), so it has its own caption to wait out.
 */
async function settle(page, timeout = 90000) {
  await page
    .waitForFunction(
      () => {
        const caption = document.body.innerText;
        return (
          !/refitting the classifier/i.test(caption) &&
          !/scoring the opening filter/i.test(caption) &&
          !/fitting the baseline/i.test(caption)
        );
      },
      null,
      { timeout },
    )
    .catch(() => {});
  await page.waitForTimeout(200);
}

/** Load a preset into layer `layerIndex`, slot `slot` (both 0-based). */
async function preset(page, layerIndex, slot, name) {
  await page
    .getByLabel(
      new RegExp(
        `Load a preset into .*, layer ${layerIndex + 1} slot ${slot + 1}`,
      ),
    )
    .selectOption(name);
  await settle(page);
}

const metricNumber = async (page) =>
  Number(
    (
      (await page.locator('[data-testid="metric-value"]').first().innerText()) ??
      ""
    ).replace(/[^\d.-]/g, ""),
  );

export async function run({ page, check }) {
  const whyRegion = page.getByRole("region", { name: /why did that happen/i });
  const failure = page.locator('[data-testid="named-failure"]');
  const controls = page.locator("#game-canvas + div");
  const main = page.locator("#game-canvas");
  const outputRegion = page.getByRole("region", { name: "Script output" });
  const windowControl = page.getByRole("button", {
    name: /Kernel window on the input picture/,
  });

  // ── shell anatomy ──
  console.log("\nShell anatomy");
  await windowControl.waitFor({ timeout: 120000 });
  await settle(page, 180000);

  check(
    "metric readout present",
    (await page.locator('[data-testid="metric-value"]').count()) > 0,
  );
  check(
    "the live metric is the detection score, per the catalog",
    /Detection score/.test(await page.locator("body").innerText()),
  );
  check("sliding window control present", await windowControl.isVisible());
  check(
    "the kernel is nine steppers, reachable by keyboard",
    (await page.getByRole("button", { name: /^Increase row \d column \d/ }).count()) ===
      9,
    `${await page.getByRole("button", { name: /^Increase row \d column \d/ }).count()} increase buttons`,
  );
  check(
    "pooling is a radio group",
    (await page.getByRole("radio", { name: /Average 2x2/ }).count()) > 0,
  );
  check(
    "the feature map gallery is present",
    /What each filter sees/.test(await main.innerText()),
  );
  check("why-card docked", await whyRegion.isVisible());
  // The opening blur is scored on mount, so by now the why-card has already moved
  // from the briefing to the verdict on it. The verdict is the stronger message.
  check(
    "the why-card explains why the opening filter cannot work",
    /looks like the photograph/.test(await whyRegion.innerText()) &&
      /no sign change/.test(await whyRegion.innerText()),
    (await whyRegion.innerText()).split("\n")[1] ?? "",
  );

  // ── the premise: convolution is doing the work, not the classifier ──
  console.log("\nThe premise: the classifier cannot do this alone");
  const bodyText = await page.locator("body").innerText();
  const rawMatch = bodyText.match(/Raw pixels score[\s\S]{0,40}?(\d+)%/);
  const raw = rawMatch ? Number(rawMatch[1]) : NaN;
  check(
    "the same classifier on raw pixels sits near chance",
    Number.isFinite(raw) && raw < 45,
    `raw pixels ${raw}% (chance is 25%)`,
  );
  check(
    "and the game says so where the player will read it",
    /no fixed pixel template fits/.test(await main.innerText()),
  );

  // ── named failure 1: the blur it opens on ──
  console.log("\nNamed failure: You built a blur");
  const blurFailed = (await failure.count()) > 0;
  check("the opening blur fails, as designed", blurFailed);
  if (blurFailed) {
    const text = await failure.innerText();
    check(
      "failure is NAMED 'You built a blur', not generic",
      /you built a blur/i.test(text) && !/game over/i.test(text),
      text.split("\n")[0],
    );
    check(
      "it explains that an average measures the one randomised quantity",
      /AVERAGE/.test(text) && /brightness/i.test(text),
      text.replace(/\n/g, " ").slice(0, 170),
    );
    check(
      "it tells the player the actual fix: a sign change",
      /positive weight and at least one negative weight/i.test(text),
    );
  }
  check(
    "the kernel panel flags it inline too",
    /brightness meter, not a detector/.test(await controls.innerText()),
  );

  // ── contract #2: sliding the kernel IS the convolution ──
  console.log("\nPlayer action = sliding the kernel (contract #2)");
  check(
    "the nine multiplies are a real table, not a picture",
    /The nine multiplies under the window/.test(await main.innerText()),
  );

  const sumBefore = (await main.innerText()).match(/Sum\s+([-\d.]+)/)?.[1];
  await windowControl.focus();
  for (let step = 0; step < 3; step += 1) {
    await windowControl.press("ArrowRight");
  }
  await page.waitForTimeout(250);
  const sumAfter = (await main.innerText()).match(/Sum\s+([-\d.]+)/)?.[1];

  check(
    "arrow keys move the window and the arithmetic follows",
    sumBefore !== undefined && sumAfter !== undefined && sumBefore !== sumAfter,
    `sum ${sumBefore} -> ${sumAfter}`,
  );
  check(
    "the window reports its own position to assistive tech",
    /at row \d+, column 12\./.test(
      (await windowControl.getAttribute("aria-label")) ?? "",
    ),
    ((await windowControl.getAttribute("aria-label")) ?? "").slice(0, 90),
  );
  check(
    "and the sum is explained rather than just displayed",
    /looks like the kernel|cancelled|opposite/.test(await main.innerText()),
  );

  // ── the fix: an edge detector, and a feature map that stops looking like a photo ──
  console.log("\nA detector instead of a blur");
  await preset(page, 0, 0, "Vertical edge");

  const afterEdge = await metricNumber(page);
  check(
    "one edge detector lifts the score well above chance",
    afterEdge > 50,
    `detection score ${afterEdge}%`,
  );
  check(
    "the brightness-meter warning clears once the filter is a real detector",
    !/brightness meter, not a detector/.test(await controls.innerText()),
  );
  check(
    "the kernel panel reports its measured mean output",
    /mean output 0\.\d+/.test(await controls.innerText()),
    (await controls.innerText()).match(/sum [+-]?\d+ · mean output [\d.]+/)?.[0] ??
      "",
  );

  const galleryLabel =
    (await page.getByRole("img", { name: /Vertical edge: feature map/ }).getAttribute(
      "aria-label",
    )) ?? "";
  check(
    "the feature map reports a real peak and mean, not just a picture",
    /brightest response [\d.]+, mean [\d.]+/.test(galleryLabel),
    galleryLabel.slice(0, 110),
  );
  // Anchored to digit-dot-digit rather than [\d.]+, which swallows the sentence's
  // closing full stop and turns "mean 0.04." into NaN.
  const peak = Number(galleryLabel.match(/brightest response (\d+\.\d+)/)?.[1] ?? NaN);
  const mean = Number(galleryLabel.match(/mean (\d+\.\d+)/)?.[1] ?? NaN);
  check(
    "a detector's map is mostly dark with bright lines — mean far below peak",
    Number.isFinite(peak) && Number.isFinite(mean) && mean < peak * 0.5,
    `peak ${peak}, mean ${mean} (a blur's map would sit near its peak)`,
  );

  // ── named failure 2: Dead filters — the catalog's declared failure mode ──
  console.log("\nNamed failure: Dead filters");
  await page.getByRole("button", { name: /^Add a filter/ }).first().click();
  await settle(page);
  await preset(page, 0, 1, "Centre spot");

  const dead = (await failure.count()) > 0;
  check("a filter that never fires is caught", dead);
  if (dead) {
    const text = await failure.innerText();
    check(
      "failure is NAMED 'Dead filters' — the catalog's declared mode",
      /dead filters/i.test(text),
      text.split("\n")[0],
    );
    check(
      "it blames the weight sum and the ReLU, with the number",
      /-6/.test(text) && /ReLU/.test(text),
      text.replace(/\n/g, " ").slice(0, 190),
    );
    check(
      "it notes that a sensible kernel can still be dead on this data",
      /no dots in this kitchen/.test(text),
    );
  }
  check(
    "the gallery shows the dead map as all zero, in words",
    (await page.getByRole("img", { name: /feature map is entirely zero/ }).count()) >
      0,
  );
  check(
    "and the kernel panel prints its mean activation",
    /Outputs nothing: mean activation 0\.0000/.test(await controls.innerText()),
  );

  // ── named failure 3: duplicates ──
  console.log("\nNamed failure: Duplicate filters");
  await preset(page, 0, 1, "Vertical edge");

  const dup = (await failure.count()) > 0;
  check("two identical filters are caught", dup);
  if (dup) {
    const text = await failure.innerText();
    check(
      "failure is NAMED 'Duplicate filters' — a different diagnosis",
      /duplicate filters/i.test(text) && !/dead filters/i.test(text),
      text.split("\n")[0],
    );
    check(
      "it reports the measured correlation, not a guess about the weights",
      /correlate at 1\.00/.test(text) && /activations rather than on the weights/.test(text),
      text.replace(/\n/g, " ").slice(0, 190),
    );
  }

  // ── named failure 4: the ceiling. Good kernels, still not enough. ──
  console.log("\nNamed failure: One layer is not enough");
  await preset(page, 0, 1, "Horizontal edge");
  await page.getByRole("button", { name: /^Add a filter/ }).first().click();
  await settle(page);
  await preset(page, 0, 2, "Diagonal edge");

  const ceilingScore = await metricNumber(page);
  const ceiling = (await failure.count()) > 0;
  check(
    "three good, distinct, live filters still fail",
    ceiling && ceilingScore < 90,
    `detection score ${ceilingScore}%`,
  );
  if (ceiling) {
    const text = await failure.innerText();
    check(
      "failure is NAMED 'One layer is not enough' — not the player's fault",
      /one layer is not enough/i.test(text) &&
        /not bad kernel design/i.test(text),
      text.split("\n")[0],
    );
    check(
      "it blames the global average, which is the actual reason",
      /global average/.test(text) && /one number per filter/.test(text),
      text.replace(/\n/g, " ").slice(0, 200),
    );
    check(
      "it proves it from the confusion, not from a hunch",
      /mistakes land on each OTHER/.test(text) &&
        /came back wearing the sibling's name/.test(text),
    );
    check(
      "it offers the experiment rather than asking to be believed",
      /Let it learn/.test(text),
    );
  }
  check(
    "the per-class table shows exactly which dishes collapse",
    /Vertical on top/.test(await main.innerText()) &&
      /Horizontal on top/.test(await main.innerText()),
  );

  // ── the reference run: gradient descent hits the same wall ──
  console.log("\nLet it learn: the ceiling is the architecture, not the design");
  await page.getByRole("button", { name: /Let it learn/ }).click();
  await page
    .getByRole("button", { name: /^Let it learn$/ })
    .waitFor({ state: "visible", timeout: 300000 });
  await page.waitForTimeout(500);

  const learnedText = await whyRegion.innerText();
  const learnedScore = Number(
    learnedText.match(/got (\d+)%/)?.[1] ?? NaN,
  );
  check(
    "training the same shape end to end also fails to pass",
    Number.isFinite(learnedScore) && learnedScore < 90,
    `learned kernels ${learnedScore}%`,
  );
  check(
    "and the why-card says the shape is the limit",
    /shape is the limit/.test(learnedText),
    learnedText.split("\n")[0] ?? "",
  );
  check(
    "the learned score is reported as a metric too",
    /Learned kernels/.test(await page.locator("body").innerText()),
  );

  // The why-card sends the player to the grids under the button, so they must be
  // there: one 3x3 table per filter in the learned shape (three, here), each
  // scaled so its strongest weight reads ±2 as the note beneath them says.
  const picked = page.getByRole("region", { name: "What gradient descent picked" });
  const pickedGrids = picked.getByRole("table");
  const gridCount = await pickedGrids.count();
  const cellsPerGrid = [];
  const peaks = [];
  for (let index = 0; index < gridCount; index += 1) {
    const cells = await pickedGrids.nth(index).locator("td").allInnerTexts();
    cellsPerGrid.push(cells.length);
    peaks.push(Math.max(...cells.map((cell) => Math.abs(Number(cell)))));
  }
  check(
    "the kernels gradient descent picked are drawn, one grid per filter",
    gridCount === 3 && cellsPerGrid.every((count) => count === 9),
    `${gridCount} grids, cells ${cellsPerGrid.join("/")}`,
  );
  check(
    "each learned grid is scaled so its strongest weight reads 2",
    peaks.length === gridCount && peaks.every((peak) => peak === 2),
    `peaks ${peaks.join(", ")}`,
  );

  // ── the win: depth, on fewer filters ──
  console.log("\nAdding a second layer");
  await page.getByRole("button", { name: /Add layer 2/ }).click();
  await settle(page, 180000);

  check(
    "the win explains what the second layer actually contributed",
    /a position, converted into a quantity/.test(await whyRegion.innerText()) &&
      /edges, then textures, then arrangement/.test(await whyRegion.innerText()),
    (await whyRegion.innerText()).split("\n")[1]?.slice(0, 110) ?? "",
  );
  check(
    "the receptive field grew, and the game says so as a number",
    /Receptive field/.test(await page.locator("body").innerText()) &&
      /one output cell sees/.test(await controls.innerText()),
    (await controls.innerText()).match(/one output cell sees\s+\d+px/)?.[0] ?? "",
  );

  const finalScore = await metricNumber(page);
  check(
    "two layers pass the target that no single layer could reach",
    finalScore >= 90,
    `detection score ${finalScore}% (single-layer best was ${ceilingScore}%)`,
  );
  check(
    "no failure remains",
    (await failure.count()) === 0,
    (await failure.count()) === 0 ? "clear" : await failure.innerText(),
  );
  check(
    "the win credits the filters and their reuse, not the classifier",
    /reused at every position/.test(await whyRegion.innerText()),
    (await whyRegion.innerText()).split("\n")[0] ?? "",
  );
  check(
    "XP bar present",
    await page
      .getByRole("progressbar", { name: "Experience points" })
      .isVisible(),
  );

  const announced = await page
    .waitForFunction(
      () =>
        /detection score/i.test(
          document.querySelector('[aria-live="polite"]')?.textContent ?? "",
        ),
      null,
      { timeout: 15000 },
    )
    .then(() => true)
    .catch(() => false);
  check(
    "the metric is announced to screen readers",
    announced,
    (await page.locator('[aria-live="polite"]').first().innerText()).slice(0, 64),
  );

  // ── two lanes, one store ──
  console.log("\nCode lane: the search the steppers cannot do");
  await page.getByRole("radio", { name: "Code", exact: true }).click();

  const editor = page.getByLabel("Kitchen script");
  check("code lane editor present", await editor.isVisible());

  await editor.fill(
    [
      "// Sweep one-layer stacks, then compare against depth.",
      "let bestOne = 0;",
      "const sets = [",
      "  ['Vertical edge'],",
      "  ['Vertical edge', 'Horizontal edge'],",
      "  ['Vertical edge', 'Horizontal edge', 'Diagonal edge'],",
      "  ['Vertical edge', 'Horizontal edge', 'Diagonal edge', 'Sharpen'],",
      "];",
      "for (const kernels of sets) {",
      "  checkBudget();",
      "  const r = await api.trial([{ pool: 'avg', kernels }]);",
      "  bestOne = Math.max(bestOne, r.accuracy);",
      "  log('one layer,', r.filters, 'filters:', pct(r.accuracy));",
      "}",
      "log('BESTONE', pct(bestOne));",
      "const deep = await api.trial([",
      "  { pool: 'avg', kernels: ['Vertical edge'] },",
      "  { pool: 'avg', kernels: ['Pass through', 'Horizontal edge'] },",
      "]);",
      "log('two layers,', deep.filters, 'filters:', pct(deep.accuracy),",
      "    'channels', deep.channels);",
      "log('per class', deep.perClass.map(pct).join(' '));",
      "// Build it in the real kitchen, through the same store.",
      "api.reset();",
      "api.applyPreset(0, 0, 'Vertical edge');",
      "api.addLayer();",
      "log('my stack:', JSON.stringify(api.layers().map(l => ",
      "  l.kernels.map(k => k.label))));",
      "log('JSDONE');",
      "function pct(x) { return (x * 100).toFixed(1) + '%'; }",
    ].join("\n"),
  );

  await page.getByRole("button", { name: /^Run$/ }).click();
  await page
    .getByRole("button", { name: /^Run$/ })
    .waitFor({ state: "visible", timeout: 300000 });
  await page.waitForTimeout(500);

  const output = await outputRegion.innerText();
  check(
    "the script completed without an error",
    /JSDONE/.test(output),
    output.replace(/\n/g, " | ").slice(0, 220),
  );
  const bestOne = Number(output.match(/BESTONE ([\d.]+)%/)?.[1] ?? NaN);
  check(
    "no one-layer stack in the sweep passes the target",
    Number.isFinite(bestOne) && bestOne < 90,
    `best one-layer score ${bestOne}%`,
  );
  check(
    "and three filters across two layers do",
    /two layers, 3 filters: 9\d\.\d%|two layers, 3 filters: 100/.test(output),
    output.split("\n").find((line) => line.startsWith("two layers")) ?? "",
  );
  check(
    "the code lane wrote the player's real stack through the same store",
    /my stack: \[\["Vertical edge"\],\["Pass through","Horizontal edge"\]\]/.test(
      output,
    ),
    output.split("\n").find((line) => line.startsWith("my stack")) ?? "",
  );

  await page.getByRole("radio", { name: "Visual", exact: true }).click();
  await settle(page, 180000);
  check(
    "the visual lane shows the stack the code lane built — one store, two lanes",
    /Layer 2/.test(await controls.innerText()) &&
      /Pass through/.test(await controls.innerText()),
  );
  check(
    "and it scores as a win",
    (await metricNumber(page)) >= 90,
    `detection score ${await metricNumber(page)}%`,
  );
}
