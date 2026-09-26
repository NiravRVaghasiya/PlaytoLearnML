/**
 * Browser playthrough for Dimension Diver.
 *
 * What this verifies that no other playthrough can:
 *
 *   1. Three.js actually renders a WebGL cloud in a real browser, and the game
 *      remains playable from the SVG shadow if it does not.
 *   2. The roll slider provably cannot move the gauge — the geometric fact the
 *      whole game rests on, checked in the DOM rather than only in a unit test.
 *   3. "Lost variance" and "High variance, mixed groups" are distinct diagnoses
 *      with different fixes, and the second one points at a plane that actually
 *      passes rather than at one that would fail the first.
 *   4. The nested shells win WITHOUT separating the groups, and the game says why —
 *      the t-SNE/UMAP motivation, stated as a limit of linear projection.
 */

export const slug = "dimension-diver";
export const title = "Dimension Diver";

/**
 * Set a range input by keyboard.
 *
 * `fill()` refuses `<input type="range">`, and `aria-valuenow` is not set by the
 * native control, so the value is read back with `inputValue()`.
 */
async function setSlider(page, label, target, { min, step }) {
  const slider = page.getByLabel(label);
  await slider.focus();
  await slider.press("Home");
  const clicks = Math.round((target - min) / step);
  for (let index = 0; index < clicks; index += 1) {
    await slider.press("ArrowRight");
  }
  await page.waitForTimeout(80);
  return Number(await slider.inputValue());
}

/**
 * Nudge one slider from wherever it is, without the Home reset.
 *
 * `setSlider` jumps to the minimum first, which is fine for setting a value but
 * wrong for the roll-invariance check: that check has to show the gauge holding
 * still while ONE slider moves, so it must not disturb the others and must not
 * depend on a Home keypress landing where expected.
 */
async function nudgeSlider(page, label, presses) {
  const slider = page.getByLabel(label);
  await slider.focus();
  for (let index = 0; index < presses; index += 1) {
    await slider.press("ArrowRight");
  }
  await page.waitForTimeout(80);
  return Number(await slider.inputValue());
}

const readAngles = async (page) => ({
  yaw: Number(await page.getByLabel(/^Yaw/).inputValue()),
  pitch: Number(await page.getByLabel(/^Pitch/).inputValue()),
  roll: Number(await page.getByLabel(/^Roll/).inputValue()),
});

const YAW = { min: -180, step: 1 };
const PITCH = { min: -90, step: 1 };

const metricNumber = async (page) =>
  Number(
    (
      (await page.locator('[data-testid="metric-value"]').first().innerText()) ??
      ""
    ).replace(/[^\d.-]/g, ""),
  );

/**
 * Read the metric only once it has stopped moving.
 *
 * The readout animates between values, so sampling it straight after a change
 * captures a mid-transition frame. That produced a false failure on the
 * roll-invariance check — the maths is exactly invariant, and the "before" reading
 * was a number the gauge was still travelling through.
 */
async function settledMetric(page, tries = 40) {
  let previous = await metricNumber(page);
  for (let attempt = 0; attempt < tries; attempt += 1) {
    await page.waitForTimeout(80);
    const current = await metricNumber(page);
    if (current === previous) return current;
    previous = current;
  }
  return previous;
}

export async function run({ page, check }) {
  const whyRegion = page.getByRole("region", { name: /why did that happen/i });
  const failure = page.locator('[data-testid="named-failure"]');
  const controls = page.locator("#game-canvas + div");
  const main = page.locator("#game-canvas");
  const outputRegion = page.getByRole("region", { name: "Script output" });
  const shadow = () => page.getByRole("img", { name: /^Shadow of the cloud/ });
  const gauge = () =>
    page.getByRole("progressbar", { name: /spread kept by this projection/ });

  // ── shell anatomy ──
  console.log("\nShell anatomy");
  await shadow().waitFor({ timeout: 60000 });

  check(
    "metric readout present",
    (await page.locator('[data-testid="metric-value"]').count()) > 0,
  );
  check(
    "the live metric is variance retained, per the catalog",
    /Variance retained/.test(await page.locator("body").innerText()),
  );
  check("the 2D shadow is present and is SVG", await shadow().isVisible());
  check("the variance gauge is a real progressbar", await gauge().isVisible());
  check(
    "three rotation sliders",
    (await page.locator('input[type="range"]').count()) === 3,
    `${await page.locator('input[type="range"]').count()} range inputs`,
  );
  check(
    "the eigenvalue table is a real table",
    /Variance along the cloud/.test(await controls.innerText()) &&
      (await page.getByRole("table").count()) > 0,
  );
  check("why-card docked", await whyRegion.isVisible());
  check(
    "the briefing warns that roll cannot change the score",
    /roll slider will spin the picture/.test(await whyRegion.innerText()),
    (await whyRegion.innerText()).split("\n")[1]?.slice(0, 90) ?? "",
  );

  // Three.js is the spec's renderer and it arrives via a dynamic import, so wait
  // for either the canvas or the explicit no-WebGL fallback — a bare count races
  // the chunk load and flakes.
  await Promise.race([
    main.locator("canvas").first().waitFor({ timeout: 30000 }),
    page
      .getByText(/could not start WebGL/)
      .waitFor({ timeout: 30000 }),
  ]).catch(() => {});

  const canvasCount = await main.locator("canvas").count();
  const fellBack = await page
    .getByText(/could not start WebGL/)
    .isVisible()
    .catch(() => false);

  check(
    "the 3D view either rendered with Three.js or said why it could not",
    canvasCount > 0 || fellBack,
    canvasCount > 0
      ? `${canvasCount} canvas element(s)`
      : fellBack
        ? "WebGL unavailable, fallback message shown"
        : "neither",
  );
  if (canvasCount > 0) {
    const size = await main.locator("canvas").first().boundingBox();
    check(
      "and the canvas has real dimensions",
      size !== null && size.width > 50 && size.height > 50,
      size === null ? "no box" : `${Math.round(size.width)}x${Math.round(size.height)}`,
    );
  }
  check(
    "the game is playable from the SVG shadow regardless of WebGL",
    await shadow().isVisible(),
  );
  check(
    "the groups are hidden until the player commits",
    /Groups are hidden until you commit/.test(await main.innerText()),
  );

  const shadowLabel = (await shadow().getAttribute("aria-label")) ?? "";
  check(
    "the shadow is described in words, including what it retains",
    /300 points/.test(shadowLabel) && /percent of the cloud/.test(shadowLabel),
    shadowLabel.slice(0, 110),
  );

  // ── the geometric fact the game rests on ──
  console.log("\nRoll cannot change the score");
  await setSlider(page, /^Yaw/, 30, YAW);
  await setSlider(page, /^Pitch/, -20, PITCH);
  const anglesBefore = await readAngles(page);
  const beforeRoll = await settledMetric(page);

  await nudgeSlider(page, /^Roll/, 90);
  const anglesAfter = await readAngles(page);
  const afterRoll = await settledMetric(page);

  check(
    "only the roll slider moved",
    anglesAfter.yaw === anglesBefore.yaw &&
      anglesAfter.pitch === anglesBefore.pitch &&
      anglesAfter.roll === anglesBefore.roll + 90,
    `yaw ${anglesBefore.yaw}->${anglesAfter.yaw}, pitch ${anglesBefore.pitch}->${anglesAfter.pitch}, roll ${anglesBefore.roll}->${anglesAfter.roll}`,
  );
  check(
    "rolling leaves the retained variance untouched",
    beforeRoll === afterRoll,
    `${beforeRoll}% before, ${afterRoll}% after a 90° roll`,
  );
  check(
    "and the why-card explains why, in terms of the discarded axis",
    /discarded axis never moved/.test(await whyRegion.innerText()),
    (await whyRegion.innerText()).split("\n")[0] ?? "",
  );
  check(
    "it also names the real number of degrees of freedom",
    /two real degrees of freedom/.test(await whyRegion.innerText()),
  );

  // ── contract #2: turning the cloud IS choosing a projection ──
  console.log("\nPlayer action = discovering a principal plane (contract #2)");
  const beforeTurn = await metricNumber(page);
  await setSlider(page, /^Yaw/, 90, YAW);
  const afterTurn = await metricNumber(page);
  check(
    "yaw moves the live gauge",
    beforeTurn !== afterTurn,
    `${beforeTurn}% -> ${afterTurn}%`,
  );

  const announced = await page
    .waitForFunction(
      () =>
        /variance retained/i.test(
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

  // ── named failure: Lost variance ──
  console.log("\nNamed failure: Lost variance");
  await page.getByRole("button", { name: /Commit this projection/ }).click();
  await page.waitForTimeout(300);

  const lost = (await failure.count()) > 0;
  check("committing a poor plane fails", lost);
  if (lost) {
    const text = await failure.innerText();
    check(
      "failure is NAMED 'Lost variance', per the catalog",
      /lost variance/i.test(text) && !/game over/i.test(text),
      text.split("\n")[0],
    );
    check(
      "it quotes the eigenvalues that explain the ceiling",
      /three variances are/.test(text),
      text.replace(/\n/g, " ").slice(0, 180),
    );
    check(
      "and tells the player what to aim for",
      /thinnest along/.test(text),
    );
  }
  check(
    "committing reveals the hidden groups — the payoff for guessing",
    /Alpha/.test(await main.innerText()) && /Beta/.test(await main.innerText()),
  );
  check(
    "and the shadow now describes the groups it is showing",
    /groups/.test((await shadow().getAttribute("aria-label")) ?? ""),
  );

  // ── the win on the gentle cloud, via the PCA hint ──
  console.log("\nThe principal plane");
  await page.getByRole("button", { name: /Show me PCA's answer/ }).click();
  await page.waitForTimeout(300);

  const hinted = await metricNumber(page);
  check(
    "the hint snaps to a near-perfect projection",
    hinted > 99,
    `retained ${hinted}%`,
  );
  check(
    "and says it was a closed-form answer, not a search",
    /closed-form answer/.test(await whyRegion.innerText()),
    (await whyRegion.innerText()).split("\n")[0] ?? "",
  );
  check(
    "the hint states its cost up front",
    /Caps your score|Snapped to yaw/.test(await controls.innerText()),
  );

  await page.getByRole("button", { name: /Commit this projection/ }).click();
  await page.waitForTimeout(300);
  check(
    "the pancake surfaces",
    (await failure.count()) === 0 &&
      /Next cloud/.test(await controls.innerText()),
    (await whyRegion.innerText()).split("\n")[0] ?? "",
  );
  check(
    "the win credits the eigenvalues rather than the search",
    /principal variances are/.test(await whyRegion.innerText()),
  );
  check(
    "XP bar present",
    await page
      .getByRole("progressbar", { name: "Experience points" })
      .isVisible(),
  );

  // ── named failure: the conflict. Maximum variance, nothing to see. ──
  console.log("\nNamed failure: High variance, mixed groups");
  await page.getByRole("button", { name: /Next cloud/ }).click();
  await page.waitForTimeout(400);
  check(
    "cloud 2 is the loud axis",
    /loud axis/i.test(await main.innerText()),
    (await main.innerText()).split("\n")[0]?.slice(0, 60) ?? "",
  );

  await page.getByRole("button", { name: /Show me PCA's answer/ }).click();
  await page.waitForTimeout(300);
  const pcaRetained = await metricNumber(page);
  await page.getByRole("button", { name: /Commit this projection/ }).click();
  await page.waitForTimeout(400);

  const mixed = (await failure.count()) > 0;
  check(
    "PCA's own plane fails on this cloud",
    mixed && pcaRetained > 99,
    `retained ${pcaRetained}% and still failed`,
  );
  if (mixed) {
    const text = await failure.innerText();
    check(
      "failure is NAMED for the conflict — a different diagnosis",
      /High variance, mixed groups/i.test(text) && !/Lost variance/.test(text),
      text.split("\n")[0],
    );
    check(
      "it says the failure is the method's, not the player's",
      /not your mistake, it is the limitation/.test(text),
      text.replace(/\n/g, " ").slice(0, 190),
    );
    check(
      "it points at a plane that still passes the variance target",
      /still inside the target/.test(text) &&
        /Discard the middle axis/.test(text),
      text.match(/there is another plane[^.]*\./)?.[0]?.slice(0, 130) ?? "",
    );
  }
  check(
    "the separation metric reports the gap",
    /Group separation/.test(await page.locator("body").innerText()) &&
      /best reachable/.test(await page.locator("body").innerText()),
  );

  // ── the nested shells: a win without separation, and the reason ──
  console.log("\nNested shells: no linear projection can separate them");
  await page.getByRole("button", { name: /Start this cloud again/ }).click();
  await page.waitForTimeout(200);
  await page.getByRole("button", { name: /Next cloud/ }).click().catch(() => {});
  await page.waitForTimeout(400);

  // The needle was not surfaced, so Next cloud may be absent; navigate by name.
  if (!/Nested shells/i.test(await main.innerText())) {
    await page.getByRole("radio", { name: "Code", exact: true }).click();
    await page.getByLabel("Projection script").fill("api.nextCloud();");
    await page.getByRole("button", { name: /^Run$/ }).click();
    await page.waitForTimeout(600);
    await page.getByRole("radio", { name: "Visual", exact: true }).click();
    await page.waitForTimeout(300);
  }
  check(
    "cloud 3 is the nested shells",
    /Nested shells/i.test(await main.innerText()),
    (await main.innerText()).split("\n")[0]?.slice(0, 60) ?? "",
  );
  check(
    "the game says up front that no flat shadow separates these",
    /no flat shadow|cannot be separated/i.test(
      (await controls.innerText()) + (await page.locator("body").innerText()),
    ),
  );

  await page.getByRole("button", { name: /Show me PCA's answer/ }).click();
  await page.waitForTimeout(300);
  await page.getByRole("button", { name: /Commit this projection/ }).click();
  await page.waitForTimeout(400);

  check(
    "it surfaces even though the groups are still one blob",
    (await failure.count()) === 0,
    (await whyRegion.innerText()).split("\n")[0] ?? "",
  );
  const shellsCard = await whyRegion.innerText();
  check(
    "and explains that a weighted sum of coordinates cannot express a radius",
    /no weighted sum of coordinates/.test(shellsCard),
    shellsCard.replace(/\n/g, " ").slice(0, 190),
  );
  check(
    "naming t-SNE and UMAP as the methods that exist for exactly this",
    /t-SNE and UMAP/.test(shellsCard),
  );
  check(
    "and noting the cloud has barely any preferred plane",
    /nearly equal/.test(shellsCard) && /barely any preferred plane/.test(shellsCard),
  );

  // ── two lanes, one store ──
  console.log("\nCode lane: sweeping the sphere for both objectives");
  await page.getByRole("radio", { name: "Code", exact: true }).click();

  const editor = page.getByLabel("Projection script");
  check("code lane editor present", await editor.isVisible());

  await editor.fill(
    [
      "// Back to the loud axis, then sweep for both objectives.",
      "let guard = 0;",
      "while (api.cloud().id !== 'needle' && guard++ < 5) api.nextCloud();",
      "api.reset();",
      "const pca = api.pca();",
      "log('cloud', api.cloud().id);",
      "log('eigenvalues', pca.eigenvalues.map(v => v.toFixed(2)).join(' '));",
      "log('ceiling', (pca.bestRetained * 100).toFixed(2) + '%');",
      "",
      "// Roll is provably irrelevant.",
      "const r0 = api.retainedAt({ yaw: 30, pitch: -20, roll: 0 });",
      "const r1 = api.retainedAt({ yaw: 30, pitch: -20, roll: 137 });",
      "log('ROLLDIFF', Math.abs(r0 - r1).toExponential(1));",
      "",
      "let bestVar = { v: -1 }, bestBoth = { s: -1 };",
      "for (let yaw = -90; yaw < 90; yaw += 3) {",
      "  for (let pitch = -90; pitch <= 90; pitch += 3) {",
      "    checkBudget();",
      "    const v = api.retainedAt({ yaw, pitch, roll: 0 });",
      "    const s = api.separationAt({ yaw, pitch, roll: 0 });",
      "    if (v > bestVar.v) bestVar = { v, s, yaw, pitch };",
      "    if (v >= pca.bestRetained * 0.98 && s > bestBoth.s) {",
      "      bestBoth = { v, s, yaw, pitch };",
      "    }",
      "  }",
      "}",
      "log('MAXVAR retained', (bestVar.v * 100).toFixed(2) +",
      "    '% separation ' + bestVar.s.toFixed(3));",
      "log('BOTH   retained', (bestBoth.v * 100).toFixed(2) +",
      "    '% separation ' + bestBoth.s.toFixed(3));",
      "",
      "api.setAngles({ yaw: bestBoth.yaw, pitch: bestBoth.pitch, roll: 0 });",
      "api.submit();",
      "log('OUTCOME', api.score().outcome);",
      "log('JSDONE');",
    ].join("\n"),
  );

  await page.getByRole("button", { name: /^Run$/ }).click();
  await page
    .getByRole("button", { name: /^Run$/ })
    .waitFor({ state: "visible", timeout: 180000 });
  await page.waitForTimeout(500);

  const output = await outputRegion.innerText();
  check(
    "the script completed without an error",
    /JSDONE/.test(output),
    output.replace(/\n/g, " | ").slice(0, 200),
  );
  check(
    "roll changes the retained variance by exactly nothing",
    /ROLLDIFF 0\.0e\+0/.test(output),
    output.split("\n").find((line) => line.startsWith("ROLLDIFF")) ?? "",
  );
  check(
    "the variance-optimal plane separates nothing",
    (() => {
      const match = output.match(/MAXVAR retained [\d.]+% separation ([\d.]+)/);
      return match !== null && Number(match[1]) < 0.35;
    })(),
    output.split("\n").find((line) => line.startsWith("MAXVAR")) ?? "",
  );
  check(
    "and a plane inside the same variance target separates them well",
    (() => {
      const match = output.match(/BOTH   retained [\d.]+% separation ([\d.]+)/);
      return match !== null && Number(match[1]) > 1;
    })(),
    output.split("\n").find((line) => line.startsWith("BOTH")) ?? "",
  );
  check(
    "committing that plane surfaces the cloud through the same store",
    /OUTCOME surfaced/.test(output),
    output.split("\n").find((line) => line.startsWith("OUTCOME")) ?? "",
  );

  await page.getByRole("radio", { name: "Visual", exact: true }).click();
  await page.waitForTimeout(400);
  check(
    "the visual lane shows the projection the code lane committed — one store, two lanes",
    (await metricNumber(page)) > 95,
    `retained ${await metricNumber(page)}%`,
  );
  check(
    "and the groups are visible in the shadow",
    /Alpha/.test(await main.innerText()),
  );

  // ── WebGL contexts are released, not leaked ──
  // Every lane toggle unmounts the 3D view. It used to leave its probe context
  // and its renderer's context for GC, and Chrome caps live contexts and kills
  // the OLDEST when the cap is hit — which can be another game's TF.js backend.
  console.log("\nWebGL contexts are released on unmount");
  const contextWarnings = [];
  const onConsole = (message) => {
    if (/Too many active WebGL contexts|Oldest context will be lost/i.test(message.text())) {
      contextWarnings.push(message.text());
    }
  };
  page.on("console", onConsole);
  for (let round = 0; round < 12; round += 1) {
    await page.getByRole("radio", { name: "Code", exact: true }).click();
    await page.waitForTimeout(80);
    await page.getByRole("radio", { name: "Visual", exact: true }).click();
    await page.waitForTimeout(200);
  }
  for (let round = 0; round < 6; round += 1) {
    await page.getByRole("button", { name: /Start this cloud again/ }).click();
    await page.waitForTimeout(120);
  }
  await page.waitForTimeout(400);
  page.off("console", onConsole);
  check(
    "twelve lane toggles and six restarts exhaust no WebGL context budget",
    contextWarnings.length === 0,
    contextWarnings.length === 0 ? "no context-limit warnings" : contextWarnings[0].slice(0, 120),
  );
  check(
    "and the 3D view is still there afterwards",
    (await main.locator("canvas").count()) > 0 ||
      (await page.getByText(/could not start WebGL/).isVisible().catch(() => false)),
  );
}
