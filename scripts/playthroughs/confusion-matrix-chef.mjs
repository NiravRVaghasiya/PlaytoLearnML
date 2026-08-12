/**
 * Browser playthrough for Confusion Matrix Chef.
 *
 * Nothing trains here, so this runs fast — the whole game is arithmetic over
 * fixed scores, which is the spec's design ("metrics recomputed on threshold
 * change, no retrain needed").
 *
 * The two conceptual failures are checked as a pair on DIFFERENT shifts, because
 * that is the only way to show they are different mistakes: the accuracy paradox
 * needs imbalanced data to bite, and the wrong-side-of-the-tradeoff failure needs
 * a shift where accuracy is not lying.
 */

export const slug = "confusion-matrix-chef";
export const title = "Confusion Matrix Chef";

/**
 * Read a cell of the confusion matrix by its abbreviation.
 *
 * By test id rather than by text: the cell's textContent has no separator between
 * the count and the abbreviation, so a word-boundary match on "TP" never fires.
 */
async function matrixCell(page, abbreviation) {
  const text = await page
    .locator(`[data-testid="cell-${abbreviation.toLowerCase()}"]`)
    .first()
    .innerText();
  return Number.parseInt(text.trim().split(/\s+/)[0] ?? "", 10);
}

/**
 * Move the threshold with the keyboard.
 *
 * `fill()` rejects `<input type="range">`, and poking `.value` directly does not
 * reach React's onChange. Home/End plus arrow keys is both what actually works and
 * the path a keyboard-only player takes, so it exercises the real interaction.
 */
async function setThreshold(page, slider, target) {
  const stepsFromTop = Math.round((1 - target) * 100);
  const stepsFromBottom = Math.round(target * 100);

  if (stepsFromTop <= stepsFromBottom) {
    await slider.press("End");
    for (let step = 0; step < stepsFromTop; step += 1) {
      await slider.press("ArrowDown");
    }
  } else {
    await slider.press("Home");
    for (let step = 0; step < stepsFromBottom; step += 1) {
      await slider.press("ArrowUp");
    }
  }
  await page.waitForTimeout(200);
}

export async function run({ page, check, metricText }) {
  const whyRegion = page.getByRole("region", { name: /why did that happen/i });
  const failure = page.locator('[data-testid="named-failure"]');
  const slider = page.getByRole("slider", { name: /Decision threshold/ });
  const serve = page.getByRole("button", { name: /Serve this cutoff/ });
  const outputRegion = page.getByRole("region", { name: "Script output" });
  const runButton = page.getByRole("button", { name: "Run", exact: true });

  // ── shell anatomy ──
  console.log("\nShell anatomy");
  check(
    "metric readout present",
    (await page.locator('[data-testid="metric-value"]').count()) > 0,
  );
  check("threshold slider is a real slider", await slider.isVisible());
  check(
    "confusion matrix is a real table, not a grid of divs",
    (await page.getByRole("table").count()) > 0,
  );
  check(
    "matrix rows are labelled for screen readers",
    (await page.getByRole("rowheader").count()) >= 2,
    `${await page.getByRole("rowheader").count()} row headers`,
  );
  check(
    "ROC curve present",
    await page.getByRole("img", { name: /ROC curve/i }).isVisible(),
  );
  // .first(): the shift name appears in both the critic panel and the why-card.
  check(
    "critic brief present",
    await page.getByText(/Shift 1 of 4/).first().isVisible(),
  );
  check("why-card docked", await whyRegion.isVisible());

  // ── contract #2: the slider IS the decision rule ──
  console.log("\nPlayer action = the decision cutoff (contract #2)");
  await slider.focus();
  check(
    "threshold slider is keyboard focusable",
    await slider.evaluate((el) => el === document.activeElement),
  );

  const caughtBefore = await matrixCell(page, "TP");
  const missedBefore = await matrixCell(page, "FN");
  // Native range input: read the DOM value, not an ARIA attribute.
  const thresholdBefore = await slider.inputValue();
  for (let press = 0; press < 10; press += 1) {
    await page.keyboard.press("ArrowUp");
  }
  await page.waitForTimeout(250);

  check(
    "arrow keys move the cutoff",
    (await slider.inputValue()) !== thresholdBefore,
    `${thresholdBefore} -> ${await slider.inputValue()}`,
  );

  const caughtAfter = await matrixCell(page, "TP");
  const missedAfter = await matrixCell(page, "FN");
  check(
    "raising the cutoff catches fewer and misses more",
    caughtAfter < caughtBefore && missedAfter > missedBefore,
    `caught ${caughtBefore}->${caughtAfter}, missed ${missedBefore}->${missedAfter}`,
  );
  check(
    "every case is still accounted for",
    caughtAfter + missedAfter === caughtBefore + missedBefore,
    `${caughtAfter + missedAfter} positives both times`,
  );
  check(
    "the why-card names the cell that paid for the change",
    /false alarm|miss|flagged pile/i.test(await whyRegion.innerText()),
  );

  // ── contract #3: live metric, and accuracy visible next to it ──
  console.log("\nLive metrics (contract #3)");
  const headline = await metricText();
  await setThreshold(page, slider, 0.3);
  check(
    "the cutoff reached the requested value by keyboard alone",
    Math.abs(Number(await slider.inputValue()) - 0.3) < 0.005,
    `value=${await slider.inputValue()}`,
  );
  check(
    "the headline metric responds to the cutoff",
    (await metricText()) !== headline,
    `${headline} -> ${await metricText()}`,
  );
  check(
    "the headline metric is this shift's cost-driven one, not a fixed one",
    /Precision/.test(await page.locator("body").innerText()),
  );
  check(
    "accuracy is shown alongside, so it can be caught lying later",
    /Accuracy/.test(await page.locator("body").innerText()),
  );
  check(
    "the baseline for ignoring the model is stated",
    /for ignoring the model/i.test(await page.locator("body").innerText()),
  );

  const announced = await page
    .waitForFunction(
      () =>
        /precision/i.test(
          document.querySelector('[aria-live="polite"]')?.textContent ?? "",
        ),
      null,
      { timeout: 10000 },
    )
    .then(() => true)
    .catch(() => false);
  check(
    "the metric is announced to screen readers",
    announced,
    (await page.locator('[aria-live="polite"]').first().innerText()).slice(0, 60),
  );

  check(
    "the ROC label says the curve is fixed and only the point moves",
    /moving the threshold only moves the point/i.test(
      (await page
        .getByRole("img", { name: /ROC curve/i })
        .getAttribute("aria-label")) ?? "",
    ),
  );

  // ── contract #4a: Wrong side of the tradeoff, on a balanced-ish shift ──
  console.log("\nNamed failure: Wrong side of the tradeoff");
  await setThreshold(page, slider, 0.9);
  await serve.click();
  await page.waitForTimeout(400);

  const wrongSide = (await failure.count()) > 0;
  check("a lopsided cutoff fails the brief", wrongSide);
  if (wrongSide) {
    const text = await failure.innerText();
    check(
      "failure is NAMED 'Wrong side of the tradeoff'",
      /wrong side of the tradeoff/i.test(text) && !/game over/i.test(text),
      text.split("\n")[0],
    );
    check(
      "failure carries real numbers",
      /\d+%/.test(text),
      text.replace(/\n/g, " ").slice(0, 130),
    );
    check(
      "it names the direction to move the cutoff",
      /lower the threshold|raise the threshold/i.test(text),
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

  // ── code lane: the sweep, and the accuracy trap it exposes ──
  console.log("\nCode lane: sweep every cutoff");
  await page.getByRole("radio", { name: /code/i }).click();
  const editor = page.getByLabel("Threshold script");
  check("code lane editor present", await editor.isVisible());

  await editor.fill(
    [
      "api.restart();",
      "// Clear shift 1 by searching for the band rather than guessing.",
      "function band() {",
      "  const out = [];",
      "  for (let t = 0; t <= 100; t++) {",
      "    const m = api.metricsAt(t / 100);",
      "    if (api.constraints().every(c => m[c.metric] >= c.floor)) out.push(t / 100);",
      "  }",
      "  return out;",
      "}",
      "const b1 = band();",
      "log('shift 1 band', b1[0].toFixed(2), 'to', b1[b1.length - 1].toFixed(2));",
      "// What does chasing accuracy alone pick?",
      "let byAcc = { t: 0, acc: -1 };",
      "for (let t = 0; t <= 100; t++) {",
      "  const m = api.metricsAt(t / 100);",
      "  if (m.accuracy > byAcc.acc) byAcc = { t: t / 100, acc: m.accuracy, rec: m.recall };",
      "}",
      "log('most accurate cutoff', byAcc.t.toFixed(2), 'acc', (byAcc.acc*100).toFixed(1) + '%');",
      "log('accurate cutoff in band?', b1.includes(byAcc.t));",
      "api.setThreshold(b1[Math.floor(b1.length / 2)]);",
      "let r = await api.serve();",
      "log('shift 1', r.outcome);",
      "api.nextShift();",
      "log('now on shift', api.shift(), '| baseline', (api.majorityBaseline()*100).toFixed(0) + '%');",
      "SENTINEL_ONE();",
      "function SENTINEL_ONE() { log('SWEEPDONE'); }",
    ].join("\n"),
  );
  await runButton.click();
  await outputRegion.getByText(/SWEEPDONE/).waitFor({ timeout: 60000 });

  const sweepOut = await outputRegion.innerText();
  check(
    "the sweep finds shift one's band",
    /shift 1 band 0\.\d+ to 0\.\d+/.test(sweepOut),
    sweepOut.split("\n").find((l) => l.startsWith("shift 1 band")) ?? "",
  );
  check(
    "the most accurate cutoff is NOT the one that satisfies the brief",
    /accurate cutoff in band\? false/.test(sweepOut),
    sweepOut.replace(/\n/g, " | "),
  );
  check(
    "searching for the band clears the shift",
    /shift 1 win/.test(sweepOut),
    sweepOut.replace(/\n/g, " | "),
  );
  check(
    "the code lane advanced the same store the visual lane reads",
    /now on shift 2 \| baseline 94%/.test(sweepOut),
    sweepOut.split("\n").find((l) => l.startsWith("now on shift")) ?? "",
  );

  // ── contract #4b: the Accuracy paradox, on the imbalanced shift ──
  console.log("\nNamed failure: Accuracy paradox (needs imbalance to bite)");
  await editor.fill(
    [
      "// Shift 2 is 6% positives. Flag nothing and accuracy reads 94%.",
      "api.setThreshold(1);",
      "const m = api.metricsAt(1);",
      "log('accuracy', (m.accuracy * 100).toFixed(1) + '%',",
      "    '| recall', (m.recall * 100).toFixed(1) + '%',",
      "    '| baseline', (api.majorityBaseline() * 100).toFixed(0) + '%');",
      "const r = await api.serve();",
      "log('PARADOXDONE', r.outcome);",
    ].join("\n"),
  );
  await runButton.click();
  await outputRegion.getByText(/PARADOXDONE/).waitFor({ timeout: 60000 });

  const paradoxOut = await outputRegion.innerText();
  check(
    "flagging nothing on rare positives still scores 94% accuracy",
    /accuracy 94\.0% \| recall 0\.0%/.test(paradoxOut),
    paradoxOut.replace(/\n/g, " | "),
  );
  check(
    "and that is judged the accuracy paradox",
    /PARADOXDONE accuracy-paradox/.test(paradoxOut),
    paradoxOut.replace(/\n/g, " | "),
  );

  if ((await failure.count()) > 0) {
    const text = await failure.innerText();
    check(
      "this failure is NAMED 'Accuracy paradox' — a different diagnosis",
      /accuracy paradox/i.test(text),
      text.split("\n")[0],
    );
    check(
      "it explains that accuracy counts the easy majority class",
      /ignoring the scores|counts both classes/i.test(text),
      text.replace(/\n/g, " ").slice(0, 130),
    );
    check(
      "it does not advise moving the threshold — the scoreboard is the problem",
      /wrong scoreboard/i.test(text),
    );
  } else {
    check("named failure alert appears for the paradox", false);
  }

  check(
    "XP bar present",
    await page
      .getByRole("progressbar", { name: "Experience points" })
      .isVisible(),
  );
}
