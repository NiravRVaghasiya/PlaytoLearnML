/**
 * Browser playthrough for Hyperparameter Heist.
 *
 * Nothing trains — the objective is closed form — so this runs fast.
 *
 * The point being checked is not that any one run succeeds. It is that the three
 * strategies rank in the order the spec claims, and that the reason (resolution on
 * the dial that matters) is visible on screen rather than only in the maths.
 */

export const slug = "hyperparameter-heist";
export const title = "Hyperparameter Heist";

export async function run({ page, check, metricText }) {
  const whyRegion = page.getByRole("region", { name: /why did that happen/i });
  const failure = page.locator('[data-testid="named-failure"]');
  const budgetBar = page.getByRole("progressbar", { name: /Compute budget/ });
  const outputRegion = page.getByRole("region", { name: "Script output" });
  const runButton = page.getByRole("button", { name: "Run", exact: true });
  const tryButton = page.getByRole("button", { name: /Try these dials/ });

  // ── shell anatomy ──
  console.log("\nShell anatomy");
  check(
    "metric readout present",
    (await page.locator('[data-testid="metric-value"]').count()) > 0,
  );
  check(
    "the live metric is the best objective, per the catalog",
    /Best objective/.test(await page.locator("body").innerText()),
  );
  check("compute budget is a real progressbar", await budgetBar.isVisible());
  check(
    "budget starts full",
    (await budgetBar.getAttribute("aria-valuenow")) === "0",
    `aria-valuenow=${await budgetBar.getAttribute("aria-valuenow")}`,
  );
  check(
    "four dials, all real sliders",
    (await page.getByRole("slider").count()) === 4,
    `${await page.getByRole("slider").count()} sliders`,
  );
  check(
    "strategy toggle is a radio group",
    (await page.getByRole("radio", { name: /Grid/ }).count()) > 0,
  );
  check(
    "the grid's cost in tries is stated up front",
    /2\s*4\s*=\s*16|16/.test(await page.locator("body").innerText()),
  );
  check(
    // No table yet — with no tries there is nothing to tabulate, and the trail
    // says so rather than rendering empty headers.
    "the trail admits it has nothing to show yet",
    /No tries yet/i.test(await page.locator("main").innerText()),
  );
  check("why-card docked", await whyRegion.isVisible());

  // ── the surface must NOT give the answer away ──
  console.log("\nThe objective surface stays hidden while cracking");
  const surface = page.getByRole("img", { name: /objective surface|surrogate/i });
  check("objective surface present", await surface.isVisible());
  const initialLabel = (await surface.getAttribute("aria-label")) ?? "";
  check(
    "nothing is known before the first try",
    /unknown/i.test(initialLabel),
    initialLabel.slice(0, 90),
  );
  check(
    "the true surface is not on show",
    !/true objective surface/i.test(initialLabel),
  );

  // ── contract #2: a try IS one objective evaluation ──
  console.log("\nPlayer action = spending a try (contract #2)");
  const lrSlider = page.getByRole("slider").first();
  await lrSlider.focus();
  check(
    "dials are keyboard focusable",
    await lrSlider.evaluate((el) => el === document.activeElement),
  );
  const before = await lrSlider.inputValue();
  await lrSlider.press("ArrowRight");
  await lrSlider.press("ArrowRight");
  check(
    "arrow keys move a dial",
    (await lrSlider.inputValue()) !== before,
    `${before} -> ${await lrSlider.inputValue()}`,
  );

  const metricBefore = await metricText();
  await tryButton.click();
  await page.waitForTimeout(300);
  check(
    "spending a try moves the best-objective metric",
    (await metricText()) !== metricBefore,
    `${metricBefore} -> ${await metricText()}`,
  );
  check(
    "the budget went down by exactly one",
    (await budgetBar.getAttribute("aria-valuenow")) === "1",
    `aria-valuenow=${await budgetBar.getAttribute("aria-valuenow")}`,
  );
  check(
    "the why-card reports coverage, not just the score",
    /distinct/i.test(await whyRegion.innerText()),
  );
  check(
    "the trail records the try",
    /1 of 16 tries spent/.test(await page.locator("main").innerText()),
  );
  check(
    "and now renders as a real table",
    (await page.getByRole("table").count()) > 0,
  );
  check(
    "centring every dial does not open the safe",
    !/cracked|Open, on try/i.test(await whyRegion.innerText()),
    (await whyRegion.innerText()).split("\n")[0] ?? "",
  );

  const announced = await page
    .waitForFunction(
      () =>
        /best objective/i.test(
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
    "distinct learning-rate values is kept permanently visible",
    /Distinct lr values/i.test(await page.locator("body").innerText()),
  );

  // ── contract #4: the Grid trap ──
  console.log("\nNamed failure: Grid trap");
  await page.getByRole("radio", { name: /Grid/ }).check();
  await page.waitForTimeout(200);
  check(
    "choosing grid warns about its cost before it is run",
    /full factorial|thorough/i.test(await whyRegion.innerText()),
  );

  // "Spend up to": a strategy run stops the moment the safe opens. The grid
  // never opens it, so here it still spends everything.
  await page.getByRole("button", { name: /Spend up to .* on grid/ }).click();
  await page.waitForTimeout(500);

  check(
    "the grid spends the entire budget",
    (await budgetBar.getAttribute("aria-valuenow")) === "16",
    `aria-valuenow=${await budgetBar.getAttribute("aria-valuenow")}`,
  );

  const gridFailed = (await failure.count()) > 0;
  check("the grid fails to crack the safe", gridFailed);
  if (gridFailed) {
    const text = await failure.innerText();
    check(
      "failure is NAMED 'Grid trap', not generic",
      /grid trap/i.test(text) && !/game over/i.test(text),
      text.split("\n")[0],
    );
    check(
      "it names the dial that was under-explored",
      /learning rate/i.test(text),
      text.replace(/\n/g, " ").slice(0, 140),
    );
    // A count, not a specific number: this run spent one manual try before the
    // grid, so the mix has three distinct rates rather than the grid's own two.
    const distinct = Number(text.match(/only (\d+) distinct/)?.[1] ?? NaN);
    check(
      "it states how few distinct values were tried",
      Number.isFinite(distinct) && distinct <= 3,
      `reported ${distinct} distinct values`,
    );
    check(
      "it contrasts that with what the same budget buys at random",
      /at random/i.test(text),
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

  // ── the reveal: the payoff for a finished run ──
  console.log("\nThe surface is revealed once the run is over");
  const revealedLabel = (await surface.getAttribute("aria-label")) ?? "";
  check(
    "the true surface is now on show",
    /true objective surface/i.test(revealedLabel),
    revealedLabel.slice(0, 100),
  );
  check(
    "and all sixteen tries are marked on it",
    /16 tries/.test(revealedLabel),
    revealedLabel.slice(0, 110),
  );

  // ── Bayesian mode surfaces the acquisition function ──
  console.log("\nBayesian mode surfaces the acquisition suggestion");
  await page.getByRole("button", { name: /Crack it again/ }).click();
  await page.waitForTimeout(300);
  await page.getByRole("radio", { name: /Bayesian/ }).check();
  await page.waitForTimeout(300);

  check(
    "the acquisition function's pick is shown",
    /Acquisition function suggests/i.test(await page.locator("body").innerText()),
  );
  check(
    "it admits it is still warming up before it has data",
    /warming up/i.test(await page.locator("body").innerText()),
  );

  await page.getByRole("button", { name: /Try where it points/ }).click();
  await page.waitForTimeout(300);
  check(
    "following the hint spends a try",
    (await budgetBar.getAttribute("aria-valuenow")) === "1",
    `aria-valuenow=${await budgetBar.getAttribute("aria-valuenow")}`,
  );

  await page.getByRole("button", { name: /Spend up to .* on bayesian/ }).click();
  await page.waitForTimeout(1500);
  check(
    "Bayesian search cracks the safe",
    (await failure.count()) === 0,
    (await failure.count()) === 0
      ? "no failure alert"
      : await failure.innerText(),
  );
  check(
    "cracking is announced as a win",
    /open/i.test(await whyRegion.innerText()),
    (await whyRegion.innerText()).split("\n")[0] ?? "",
  );
  const spentOnBayes = Number(await budgetBar.getAttribute("aria-valuenow"));
  const openedOn = Number(
    (await page.locator("body").innerText()).match(/Open on try (\d+)/)?.[1] ??
      NaN,
  );
  check(
    // The run stops at the crack, so the tries left are the score's tries to
    // spare, and the readout names the try that actually opened it.
    "the run stops at the crack, and the readout names that try",
    Number.isFinite(openedOn) && openedOn === spentOnBayes && spentOnBayes < 16,
    `opened on ${openedOn}, spent ${spentOnBayes} of 16`,
  );

  // ── the code lane settles the claim over many runs ──
  console.log("\nCode lane: the comparison, over many runs");
  await page.getByRole("radio", { name: /code/i }).click();
  const editor = page.getByLabel("Cracking script");
  check("code lane editor present", await editor.isVisible());

  const budgetBeforeSnippet = await budgetBar.getAttribute("aria-valuenow");
  await editor.fill(
    [
      "const RUNS = 40;",
      "const out = {};",
      "for (const s of ['grid', 'random', 'bayesian']) {",
      "  const runs = s === 'grid' ? 1 : RUNS;",
      "  let total = 0, cracked = 0, lr = 0;",
      "  for (let seed = 1; seed <= runs; seed++) {",
      "    const r = api.simulate(s, seed * 7919);",
      "    total += r.best;",
      "    if (r.best >= api.crackThreshold()) cracked++;",
      "    lr += r.distinctLearningRates;",
      "  }",
      "  out[s] = { mean: total / runs, crack: cracked / runs, lr: lr / runs };",
      "  log(s, 'mean', pct(out[s].mean), 'crack', pct(out[s].crack),",
      "      'lrSeen', out[s].lr.toFixed(1));",
      "}",
      "log('ordered', out.grid.mean < out.random.mean &&",
      "    out.random.mean < out.bayesian.mean ? 'grid < random < bayesian' : 'NO');",
      "log('resolution', out.random.lr > out.grid.lr * 3 ? 'random has more' : 'NO');",
      "log('HEISTDONE');",
      "function pct(x) { return (x * 100).toFixed(1) + '%'; }",
    ].join("\n"),
  );
  await runButton.click();
  await outputRegion.getByText(/HEISTDONE/).waitFor({ timeout: 120000 });

  const output = await outputRegion.innerText();
  check(
    "grid never cracks the safe",
    /grid mean .* crack\s+0\.0%/.test(output),
    output.split("\n").find((l) => l.startsWith("grid")) ?? "",
  );
  check(
    "random cracks it most of the time",
    (() => {
      const match = output.match(/random mean [\d.]+% crack ([\d.]+)%/);
      return match !== null && Number(match[1]) > 50;
    })(),
    output.split("\n").find((l) => l.startsWith("random")) ?? "",
  );
  check(
    "the strategies rank grid < random < bayesian",
    /ordered grid < random < bayesian/.test(output),
    output.replace(/\n/g, " | "),
  );
  check(
    "and the reason is resolution on the dial that matters",
    /resolution random has more/.test(output),
    output.split("\n").filter((l) => /lrSeen|resolution/.test(l)).join(" | "),
  );
  check(
    "simulating strategies costs the player no budget",
    (await budgetBar.getAttribute("aria-valuenow")) === budgetBeforeSnippet,
    `aria-valuenow ${budgetBeforeSnippet} -> ${await budgetBar.getAttribute("aria-valuenow")}`,
  );
  check(
    "XP bar present",
    await page
      .getByRole("progressbar", { name: "Experience points" })
      .isVisible(),
  );
}
