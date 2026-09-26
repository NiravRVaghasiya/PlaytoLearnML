/**
 * Browser playthrough for Decision Tree Architect.
 *
 * Nothing trains: a tree is built by counting, so this runs fast.
 *
 * The two named failures have opposite fixes — prune versus add gates — so both
 * are exercised, and the ridge plot is checked separately because it is the one
 * where the greedy learner fails outright and a human can beat it.
 */

export const slug = "decision-tree-architect";
export const title = "Decision Tree Architect";

/**
 * Wait until the tree diagram shows at least `minNodes` nodes, every one of
 * them wholly inside the canvas frame.
 *
 * The canvas is read-only — no pan, no zoom — so a node outside the frame is a
 * node the player can never see. React Flow's `fitView` prop only fits the
 * first render, and for a while every gate past the first grew off the bottom
 * of the frame while the figcaption still read "1 gate and 2 leaves".
 */
async function treeFitsCanvas(page, minNodes) {
  const fitted = await page
    .waitForFunction(
      (min) => {
        const pane = document.querySelector(".react-flow");
        if (!pane) {
          window.__treeFit = "no canvas";
          return false;
        }
        const frame = pane.getBoundingClientRect();
        const nodes = [...pane.querySelectorAll(".react-flow__node")];
        const outside = nodes.filter((node) => {
          const box = node.getBoundingClientRect();
          return (
            box.left < frame.left - 1 ||
            box.right > frame.right + 1 ||
            box.top < frame.top - 1 ||
            box.bottom > frame.bottom + 1
          );
        }).length;
        // Kept on the page so the last measurement can be reported on failure.
        window.__treeFit = `${nodes.length} nodes, ${outside} outside the frame`;
        return nodes.length >= min && outside === 0;
      },
      minNodes,
      { timeout: 5000 },
    )
    .then(() => true)
    .catch(() => false);
  const detail = await page.evaluate(() => window.__treeFit ?? "not measured");
  return { fitted, detail };
}

export async function run({ page, check, metricText }) {
  const whyRegion = page.getByRole("region", { name: /why did that happen/i });
  const failure = page.locator('[data-testid="named-failure"]');
  // The alert's text, not its element count: holds whether the shell unmounts
  // the alert between failures or keeps an emptied live region mounted.
  const failureText = async () =>
    (await failure.allInnerTexts()).join(" ").trim();
  const ghost = page.locator('[data-testid="overfit-ghost"]');
  const signOff = page.getByRole("button", { name: /Sign off this tree/ });
  const outputRegion = page.getByRole("region", { name: "Script output" });
  const runButton = page.getByRole("button", { name: "Run", exact: true });

  // ── shell anatomy ──
  console.log("\nShell anatomy");
  check(
    "metric readout present",
    (await page.locator('[data-testid="metric-value"]').count()) > 0,
  );
  check(
    "the live metric is validation accuracy, per the catalog",
    /Validation accuracy/.test(await page.locator("body").innerText()),
  );
  check(
    "training accuracy sits beside it so divergence is visible",
    /Training accuracy/.test(await page.locator("body").innerText()),
  );
  check(
    "the gain table is a real table",
    (await page.getByRole("table").count()) > 0,
  );
  check(
    "it has a lookahead column, not just gain",
    (await page.getByRole("columnheader", { name: /Lookahead/ }).count()) > 0,
  );
  check(
    "leaf picker is a radio group, reachable by keyboard",
    (await page.getByRole("radio").count()) > 0,
  );
  check(
    "accuracy pair present",
    await page.getByRole("img", { name: /Training accuracy .* validation/i }).isVisible(),
  );
  check("why-card docked", await whyRegion.isVisible());
  check("no overfit ghost on an empty tree", (await ghost.count()) === 0);

  // ── contract #2: building a gate IS choosing a split ──
  console.log("\nPlayer action = choosing the split (contract #2)");
  const buildSlope = page.getByRole("button", { name: /Build a gate on slope/ });
  check("gain table offers a build button per reading", await buildSlope.isVisible());
  const slopeLabel = (await buildSlope.getAttribute("aria-label")) ?? "";
  check(
    "the button states the threshold, gain and lookahead it would use",
    /at 0\.\d+, gain 0\.\d+, lookahead 0\.\d+/.test(slopeLabel),
    slopeLabel,
  );

  const valBefore = await metricText();
  await buildSlope.click();
  await page.waitForTimeout(300);
  check(
    "building a gate moves validation accuracy",
    (await metricText()) !== valBefore,
    `${valBefore} -> ${await metricText()}`,
  );
  check(
    "the why-card reports the gain and how the plots split",
    /gain/i.test(await whyRegion.innerText()),
  );
  check(
    "the tree diagram now shows a gate",
    /1 gate and 2 leaves/.test(await page.locator("main").innerText()),
  );
  {
    const { fitted, detail } = await treeFitsCanvas(page, 3);
    check("the diagram re-fits so the new leaves are in view", fitted, detail);
  }
  check(
    "the tree is stated in words for screen readers",
    /gate n0, \S+( \S+)? < 0\.\d\d/.test(
      (await page
        .getByRole("list", { name: "The tree, gate by gate" })
        .textContent()) ?? "",
    ),
  );

  const announced = await page
    .waitForFunction(
      () =>
        /validation accuracy/i.test(
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
    (await page.locator('[aria-live="polite"]').first().innerText()).slice(0, 64),
  );

  // ── the box plot signs off in two gates ──
  console.log("\nTwo gates clear the terrace plot");
  const bedrockRadios = await page.getByRole("radio").count();
  check("splitting produced more leaves to choose from", bedrockRadios >= 2);

  // Select the leaf holding the most plots, then take the greedy gate there.
  await page.getByRole("radio").first().check();
  await page.waitForTimeout(200);
  await page.getByRole("button", { name: /Take the greedy gate/ }).click();
  await page.waitForTimeout(300);

  await signOff.click();
  await page.waitForTimeout(400);
  check(
    "a two-gate tree signs the plot off",
    (await failureText()) === "",
    (await failureText()) === "" ? "no failure" : await failureText(),
  );
  check(
    "the next plot is offered",
    (await page.getByRole("button", { name: /Next plot/ }).count()) > 0,
  );

  // ── contract #4a: Overfit depth, via the code lane ──
  console.log("\nNamed failure: Overfit depth");
  await page.getByRole("radio", { name: /code/i }).click();
  const editor = page.getByLabel("Tree script");
  check("code lane editor present", await editor.isVisible());

  await editor.fill(
    [
      "api.restart();",
      "// Reach the peak first, so there is something to give away. Not signed",
      "// off: the peak is tracked as the tree grows, and a signed-off plot is closed.",
      "api.growGreedy(3);",
      "log('peak depth', api.depth(), 'val', pct(api.validationAccuracy()));",
      "log('ghost at the peak?', api.peak().validation > api.validationAccuracy());",
      "// Now keep going. Training can only rise; validation need not.",
      "api.growGreedy(api.round().maxDepth);",
      "log('deep depth', api.depth(), 'train', pct(api.trainAccuracy()),",
      "    'val', pct(api.validationAccuracy()), 'starved', api.starved());",
      "const deep = api.signOff();",
      "log('DEEPDONE', deep.outcome);",
      "function pct(x) { return (x * 100).toFixed(1) + '%'; }",
    ].join("\n"),
  );
  await runButton.click();
  await outputRegion.getByText(/DEEPDONE/).waitFor({ timeout: 60000 });

  const deepOut = await outputRegion.innerText();
  check(
    "growing straight to the peak is not reported as overfitting",
    /ghost at the peak\? false/.test(deepOut),
    deepOut.replace(/\n/g, " | "),
  );
  check(
    "growing past the peak is judged Overfit depth",
    /DEEPDONE overfit-depth/.test(deepOut),
    deepOut.replace(/\n/g, " | "),
  );
  check(
    "training rose while validation fell",
    (() => {
      const deep = deepOut.match(/deep depth \d+ train ([\d.]+)% val ([\d.]+)%/);
      const peak = deepOut.match(/peak depth \d+ val ([\d.]+)%/);
      if (!deep || !peak) return false;
      return (
        Number(deep[1]) > Number(deep[2]) &&
        Number(deep[2]) < Number(peak[1])
      );
    })(),
    deepOut.split("\n").filter((l) => /depth/.test(l)).join(" | "),
  );

  if ((await failureText()) !== "") {
    const text = await failureText();
    check(
      "failure is NAMED 'Overfit depth', not generic",
      /overfit depth/i.test(text) && !/game over/i.test(text),
      text.split("\n")[0],
    );
    check(
      "it names the depth to prune back to",
      /prune back to depth \d+/i.test(text),
      text.replace(/\n/g, " ").slice(0, 140),
    );
    check(
      "it carries real numbers",
      /\d+\.\d%/.test(text),
    );
    check(
      "retry is one click away inside the alert",
      (await failure.getByRole("button", { name: /retry/i }).count()) > 0,
    );
  } else {
    check("named failure alert appears for overfit depth", false);
  }

  // The ghost should be up too, since validation is below the player's best.
  await page.getByRole("radio", { name: /visual/i }).click();
  await page.waitForTimeout(300);
  check("the overfit ghost appears when validation dips", (await ghost.count()) > 0);
  const ghostText = (await ghost.count()) > 0 ? await ghost.innerText() : "";
  check(
    "the ghost says what depth the player had it at",
    /at depth \d+/i.test(ghostText),
    ghostText.replace(/\n/g, " ").slice(0, 120),
  );
  {
    // A full-depth greedy tree is the widest this plot grows.
    const { fitted, detail } = await treeFitsCanvas(page, 9);
    check("every node of the deep tree is inside the canvas", fitted, detail);
  }
  {
    // Same tree, narrower pane: a phone turned upright, a window made
    // smaller. React Flow keeps the old transform on a resize, so without a
    // re-fit keyed on the pane's size the right-hand leaves stayed clipped.
    const viewport = page.viewportSize();
    if (viewport) {
      await page.setViewportSize({ width: 390, height: viewport.height });
      const { fitted, detail } = await treeFitsCanvas(page, 9);
      check("the deep tree re-fits when the window narrows", fitted, detail);
      await page.setViewportSize(viewport);
    }
  }

  // ── following the advice works: prune back to the peak and sign off ──
  console.log("\nPruning back to the peak signs the plot off");
  await page.getByRole("radio", { name: /code/i }).click();
  await editor.fill(
    [
      "api.growGreedy(3);",
      "const back = api.signOff();",
      "log('PEAKDONE', back.outcome, 'depth', api.depth());",
    ].join("\n"),
  );
  await runButton.click();
  await outputRegion.getByText(/PEAKDONE/).waitFor({ timeout: 60000 });
  check(
    "the shallow tree at the peak signs off",
    /PEAKDONE win depth 3/.test(await outputRegion.innerText()),
    (await outputRegion.innerText()).replace(/\n/g, " | "),
  );

  // ── pruning recovers it: the experiment, not the claim ──
  console.log("\nPruning raises accuracy on unseen plots");
  await editor.fill(
    [
      "api.restart();",
      "api.growGreedy(8);",
      "const before = api.validationAccuracy();",
      "log('deep  val', pct(before), 'depth', api.depth());",
      "// Cut everything below depth 2.",
      "for (const id of api.leaves()) { /* touch to force recompute */ }",
      "api.clear();",
      "api.growGreedy(2);",
      "const after = api.validationAccuracy();",
      "log('shallow val', pct(after), 'depth', api.depth());",
      "log('PRUNEDONE', after > before ? 'smaller tree scored higher' : 'no gain');",
      "function pct(x) { return (x * 100).toFixed(1) + '%'; }",
    ].join("\n"),
  );
  await runButton.click();
  await outputRegion.getByText(/PRUNEDONE/).waitFor({ timeout: 60000 });

  const pruneOut = await outputRegion.innerText();
  check(
    "a smaller tree scores higher on plots it never saw",
    /PRUNEDONE smaller tree scored higher/.test(pruneOut),
    pruneOut.replace(/\n/g, " | "),
  );

  // ── contract #4b: Underfit stump, the opposite fix ──
  console.log("\nNamed failure: Underfit stump (opposite fix)");
  await editor.fill(
    [
      "api.restart();",
      "// No gates at all: every plot gets the same verdict.",
      "const r = api.signOff();",
      "log('train', pct(api.trainAccuracy()), 'val', pct(api.validationAccuracy()));",
      "log('STUMPDONE', r.outcome);",
      "function pct(x) { return (x * 100).toFixed(1) + '%'; }",
    ].join("\n"),
  );
  await runButton.click();
  await outputRegion.getByText(/STUMPDONE/).waitFor({ timeout: 60000 });

  const stumpOut = await outputRegion.innerText();
  check(
    "a tree with no gates is judged Underfit stump",
    /STUMPDONE underfit-stump/.test(stumpOut),
    stumpOut.replace(/\n/g, " | "),
  );

  if ((await failureText()) !== "") {
    const text = await failureText();
    check(
      "this failure is NAMED 'Underfit stump' — a different diagnosis",
      /underfit stump/i.test(text),
      text.split("\n")[0],
    );
    check(
      "and it advises ADDING gates — the opposite advice",
      /add gates/i.test(text) && !/prune/i.test(text),
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

  // ── the ridge plot: greedy fails, a human wins ──
  console.log("\nThe greedy learner's blind spot on the ridge plot");
  await editor.fill(
    [
      "api.restart();",
      "// Walk to plot 3.",
      "api.growGreedy(2); api.signOff(); api.nextRound();",
      "api.growGreedy(4); api.signOff(); api.nextRound();",
      "log('plot', api.round().index, api.round().name, api.round().boundary);",
      "",
      "// What does the gain table say at the root?",
      "for (const row of api.gainTable('n0')) {",
      "  log(row.name.padEnd(8), 'gain', row.gain.toFixed(4),",
      "      'lookahead', row.lookaheadGain.toFixed(3));",
      "}",
      "",
      "// Greedy CART, at every depth it is allowed:",
      "const g = api.tryGreedy(5);",
      "log('greedy depth 5 -> val', pct(g.validationAccuracy));",
      "",
      "// The tree a human builds, two gates on the two real readings:",
      "api.clear();",
      "api.split('n0', 0, 0.5);",
      "api.split('n0L', 1, 0.5);",
      "api.split('n0R', 1, 0.5);",
      "log('hand-built  -> val', pct(api.validationAccuracy()), 'depth', api.depth());",
      "const r = api.signOff();",
      "log('RIDGEDONE', r.outcome, 'beat greedy by',",
      "    ((api.validationAccuracy() - g.validationAccuracy) * 100).toFixed(1) + ' points');",
      "function pct(x) { return (x * 100).toFixed(1) + '%'; }",
    ].join("\n"),
  );
  await runButton.click();
  await outputRegion.getByText(/RIDGEDONE/).waitFor({ timeout: 90000 });

  const ridgeOut = await outputRegion.innerText();
  check(
    "reached the ridge plot",
    /plot 3 Split ridge quadrants/.test(ridgeOut),
    ridgeOut.split("\n").find((l) => l.startsWith("plot 3")) ?? "",
  );
  check(
    "every reading looks worthless in the gain column",
    (() => {
      const gains = [...ridgeOut.matchAll(/gain (0\.\d+) lookahead/g)].map((m) =>
        Number(m[1]),
      );
      return gains.length === 4 && gains.every((gain) => gain < 0.05);
    })(),
    ridgeOut.split("\n").filter((l) => /lookahead/.test(l)).join(" | "),
  );
  // The punchline: the two columns disagree, and only one of them is right.
  // Immediate gain — the only thing CART looks at — ranks a useless survey
  // reading top. Lookahead ranks a real one top, by a clear multiple.
  const ridgeRows = [
    ...ridgeOut.matchAll(/^(\S+(?:\s\S+)?)\s+gain (0\.\d+) lookahead (0\.\d+)$/gm),
  ].map((row) => ({
    name: row[1].trim(),
    gain: Number(row[2]),
    lookahead: Number(row[3]),
  }));
  const informative = ["slope", "bedrock"];
  const topByGain = ridgeRows.reduce((top, row) =>
    row.gain > top.gain ? row : top,
  );
  const byLookahead = [...ridgeRows].sort((a, b) => b.lookahead - a.lookahead);

  check(
    "immediate gain ranks a USELESS reading top — exactly CART's mistake",
    !informative.includes(topByGain.name),
    `gain picks ${topByGain.name} (${topByGain.gain})`,
  );
  check(
    "lookahead ranks a real reading top instead",
    informative.includes(byLookahead[0].name),
    `lookahead picks ${byLookahead[0].name} (${byLookahead[0].lookahead})`,
  );
  check(
    "and it stands out clearly from the rest",
    byLookahead[0].lookahead > byLookahead[1].lookahead * 3,
    `${byLookahead[0].name} ${byLookahead[0].lookahead} vs ${byLookahead[1].name} ${byLookahead[1].lookahead}`,
  );
  check(
    "greedy CART is stuck near chance here",
    (() => {
      const match = ridgeOut.match(/greedy depth 5 -> val ([\d.]+)%/);
      return match !== null && Number(match[1]) < 65;
    })(),
    ridgeOut.split("\n").find((l) => l.startsWith("greedy")) ?? "",
  );
  check(
    "a hand-built two-gate tree signs the plot off",
    /RIDGEDONE win/.test(ridgeOut),
    ridgeOut.split("\n").find((l) => l.startsWith("RIDGEDONE")) ?? "",
  );
  check(
    "and beats greedy by a wide margin",
    (() => {
      const match = ridgeOut.match(/beat greedy by ([\d.]+) points/);
      return match !== null && Number(match[1]) > 20;
    })(),
    ridgeOut.split("\n").find((l) => l.startsWith("RIDGEDONE")) ?? "",
  );
  // ── Retry means this plot again, not the whole survey ──
  console.log("\nRetry keeps the plots already signed off");
  await editor.fill(
    [
      "api.restart();",
      "api.growGreedy(2); api.signOff(); api.nextRound();",
      "// No gates on plot 2: a named failure to retry from.",
      "const r = api.signOff();",
      "log('RETRYSETUP', api.round().index, r.outcome);",
    ].join("\n"),
  );
  await runButton.click();
  await outputRegion.getByText(/RETRYSETUP/).waitFor({ timeout: 60000 });
  check(
    "a gateless plot two fails with a named diagnosis",
    /RETRYSETUP 2 underfit-stump/.test(await outputRegion.innerText()),
    (await outputRegion.innerText()).replace(/\n/g, " | "),
  );
  const retryButton = failure.getByRole("button", { name: /retry/i });
  if ((await retryButton.count()) > 0) {
    await retryButton.click();
    await page.waitForTimeout(300);
    // Text, not element count: holds whether the shell removes the alert or
    // keeps an emptied live region mounted.
    check(
      "retry clears the named failure",
      !/underfit stump/i.test((await failure.allInnerTexts()).join(" ")),
    );
    check(
      "retry keeps the player on plot two",
      await page.getByText(/Plot 2 of 3/).first().isVisible(),
    );
    await editor.fill("log('RETRYDONE', api.round().index, api.cleared());");
    await runButton.click();
    await outputRegion.getByText(/RETRYDONE/).waitFor({ timeout: 60000 });
    check(
      "retry does not wipe the plot already signed off",
      /RETRYDONE 2 1/.test(await outputRegion.innerText()),
      (await outputRegion.innerText()).replace(/\n/g, " | "),
    );
  } else {
    check("retry is offered inside the failure alert", false);
  }

  check(
    "XP bar present",
    await page
      .getByRole("progressbar", { name: "Experience points" })
      .isVisible(),
  );
}
