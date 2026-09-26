/**
 * Browser playthrough for Gradient Descent Skier.
 *
 * Checks both views of the surface. Headless Chromium here provides WebGL through
 * software rendering, so the Three.js terrain does mount — note that TensorFlow.js
 * refuses the same backend, because it needs float-texture extensions the software
 * rasteriser lacks. The contour view is verified by switching to it explicitly,
 * since it must carry the entire game on its own (DESIGN.md §10).
 */

export const slug = "gradient-descent-skier";
export const title = "Gradient Descent Skier";

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
    "learning-rate dial present and is a real slider",
    await page.getByRole("slider", { name: /Learning rate/ }).isVisible(),
  );
  check(
    "momentum control present",
    await page.getByRole("slider", { name: /Momentum/ }).isVisible(),
  );
  check("why-card docked", await whyRegion.isVisible());

  // ── both views of the surface ──
  console.log("\nSurface views, 3D and the DESIGN.md §10 fallback");
  const webglAvailable = await page.evaluate(() => {
    try {
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
      // Release the probe's context rather than leaving it to count against
      // the browser's cap of live WebGL contexts.
      context?.getExtension("WEBGL_lose_context")?.loseContext();
      return Boolean(context);
    } catch {
      return false;
    }
  });

  const toggle3D = page.getByRole("button", { name: "3D" });

  if (webglAvailable) {
    check(
      "3D terrain mounts when WebGL is available",
      (await page.locator("canvas").count()) > 0,
    );
    check("the 3D toggle is offered", await toggle3D.isEnabled());
    check(
      "the terrain canvas is hidden from assistive tech",
      (await page.locator("canvas").first().getAttribute("aria-hidden")) ===
        "true",
    );
    // The drawing buffer is width × devicePixelRatio; the element must still
    // be exactly as wide as the lane, or a phone's layout doubles in width.
    const sizes = await page.evaluate(() => {
      const canvas = document.querySelector("canvas");
      if (!canvas || !canvas.parentElement) return null;
      return {
        canvas: canvas.getBoundingClientRect().width,
        lane: canvas.parentElement.getBoundingClientRect().width,
      };
    });
    check(
      "the terrain canvas is sized to its lane in CSS pixels",
      sizes !== null && Math.abs(sizes.canvas - sizes.lane) <= 1,
      sizes ? `${sizes.canvas}px in ${sizes.lane}px` : "no canvas",
    );
    // Now switch to the fallback and confirm it stands alone.
    await page.getByRole("button", { name: "Contour" }).click();
    await page.waitForTimeout(400);
  } else {
    check(
      "missing WebGL is stated, not hidden behind a dead canvas",
      /no WebGL/i.test(await page.locator("main").innerText()),
    );
    check("the 3D toggle is disabled rather than broken", await toggle3D.isDisabled());
  }

  const contour = page.getByRole("img", { name: /Contour map of the loss/ });
  check("contour view renders the surface", await contour.isVisible());
  check(
    "contour names both minima in its accessible description",
    /deepest valley[\s\S]*shallow one/i.test(
      (await contour.getAttribute("aria-label")) ?? "",
    ),
    ((await contour.getAttribute("aria-label")) ?? "").slice(0, 100),
  );
  check(
    "contour states where the skier is, in text",
    /You are at x/.test((await contour.getAttribute("aria-label")) ?? ""),
  );

  // ── contract #2 + #3: the dial IS the step size, and loss responds ──
  console.log("\nLive feedback (pedagogy contracts #2 and #3)");
  const dial = page.getByRole("slider", { name: /Learning rate/ });
  await dial.focus();
  check(
    "dial is keyboard focusable",
    await dial.evaluate((el) => el === document.activeElement),
  );

  const rateBefore = await dial.getAttribute("aria-valuenow");
  await page.keyboard.press("ArrowUp");
  await page.keyboard.press("ArrowUp");
  check(
    "arrow keys change the learning rate",
    (await dial.getAttribute("aria-valuenow")) !== rateBefore,
    `${rateBefore} -> ${await dial.getAttribute("aria-valuenow")}`,
  );

  const lossBefore = await metricText();
  await page.getByRole("button", { name: "One step" }).click();
  await page.waitForTimeout(600);
  const lossAfter = await metricText();
  check(
    "one gradient step moves the loss",
    lossBefore !== lossAfter,
    `${lossBefore} -> ${lossAfter}`,
  );
  check(
    "loss change is announced to screen readers",
    /loss/i.test(
      await page.locator('[aria-live="polite"]').first().innerText(),
    ),
  );

  // ── contract #4: the local-minimum trap, then momentum escaping it ──
  console.log("\nNamed failure: Local minimum (pedagogy contract #4)");
  await page.getByRole("radio", { name: /code/i }).click();
  const editor = page.getByLabel("Descent script");
  check("code lane editor present", await editor.isVisible());

  await editor.fill(
    [
      "api.reset();",
      "api.setLearningRate(0.1);",
      "api.setMomentum(0);",
      "api.runToEnd();",
      "const r = api.check();",
      "log('loss', api.loss().toFixed(3));",
      "log('at x', api.position().x.toFixed(3));",
      "log('deepest at x', api.globalMinimum().x.toFixed(3));",
      "log('verdict', r.outcome);",
    ].join("\n"),
  );
  await runButton.click();
  await page.waitForTimeout(2500);

  const trapped = await outputText();
  check(
    "plain descent gets stuck in the shallow valley",
    /verdict\s+local-minimum/.test(trapped),
    trapped.replace(/\n/g, " | "),
  );

  const trapVisible = (await failure.count()) > 0;
  check("named failure alert appears", trapVisible);
  if (trapVisible) {
    const text = await failure.innerText();
    check(
      "failure is NAMED, not generic",
      /local minimum/i.test(text) && !/game over/i.test(text),
      text.split("\n")[0],
    );
    check(
      "failure detail carries real numbers",
      /\d+\.\d+/.test(text),
      text.replace(/\n/g, " ").slice(0, 120),
    );
    check(
      "and names a change that was actually run and reaches the bottom",
      /momentum 0\.85 reaches the deepest valley/.test(text),
      text.replace(/\n/g, " ").slice(-120),
    );
    check(
      "retry is one click away inside the alert",
      (await failure.getByRole("button", { name: /retry/i }).count()) > 0,
    );
  }

  // The same rate, plus momentum. This is the lesson in one comparison.
  console.log("\nMomentum escapes the trap — same rate, different outcome");
  await editor.fill(
    [
      "api.reset();",
      "api.setLearningRate(0.1);",
      "api.setMomentum(0.85);", // the only change
      "api.runToEnd();",
      "const r = api.check();",
      "log('loss', api.loss().toFixed(3));",
      "log('at x', api.position().x.toFixed(3));",
      "log('score', (r.score * 100).toFixed(0) + '%');",
      "log('verdict', r.outcome);",
    ].join("\n"),
  );
  await runButton.click();
  await page.waitForTimeout(2500);

  const freed = await outputText();
  check(
    "the same rate with momentum reaches the global minimum",
    /verdict\s+win/.test(freed),
    freed.replace(/\n/g, " | "),
  );
  check("no failure alert on a win", (await failure.count()) === 0);
  check(
    "XP bar reflects progress",
    await page
      .getByRole("progressbar", { name: "Experience points" })
      .isVisible(),
  );

  // ── divergence, the catalog's named failure for this game ──
  console.log("\nNamed failure: Divergence");
  await editor.fill(
    [
      "api.reset();",
      "api.setLearningRate(1.5);",
      "api.runToEnd();",
      "const r = api.check();",
      "log('steps', api.steps());",
      "log('diverged', api.diverged());",
      "log('verdict', r.outcome);",
    ].join("\n"),
  );
  await runButton.click();
  await page.waitForTimeout(2000);

  const blown = await outputText();
  check(
    "too large a rate diverges",
    /verdict\s+diverged/.test(blown),
    blown.replace(/\n/g, " | "),
  );
  check(
    "divergence is named",
    /divergence/i.test(await failure.innerText()),
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
