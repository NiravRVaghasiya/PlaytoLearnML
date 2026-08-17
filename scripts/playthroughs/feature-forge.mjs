/**
 * Browser playthrough for Feature Forge.
 *
 * Two things are being verified here that no other playthrough covers:
 *
 *   1. The model is genuinely held still, so a score change is attributable to the
 *      features and nothing else.
 *   2. The Python lane actually works — real CPython, real pandas, loaded from
 *      /public, calling back into the same store the visual lane writes.
 *
 * The Python run downloads ~20 MB and fits several models, so its waits are long.
 */

export const slug = "feature-forge";
export const title = "Feature Forge";

/** Retrains are async; the forge button reads "Retraining…" while one is in flight. */
async function settle(page) {
  await page
    .getByRole("button", { name: /Forge it/ })
    .waitFor({ state: "visible", timeout: 120000 })
    .catch(() => {});
  await page.waitForTimeout(250);
}

/** Pick a transform, then the columns, then forge. */
async function forge(page, transform, ...columns) {
  // Anchored for the same reason as the columns: /Standardise/ also matches the
  // "Log + scale" option, whose blurb says "log1p, then standardise".
  await page.getByRole("radio", { name: new RegExp(`^${transform}\\b`) }).check();
  await page.waitForTimeout(150);
  for (const column of columns) {
    // Anchored: a bare /income/ also matches the household checkbox, whose
    // description ends "...next to income".
    await page.getByRole("checkbox", { name: new RegExp(`^${column}\\b`) }).check();
  }
  await page.getByRole("button", { name: /Forge it/ }).click();
  await settle(page);
}

export async function run({ page, check, metricText }) {
  const whyRegion = page.getByRole("region", { name: /why did that happen/i });
  const failure = page.locator('[data-testid="named-failure"]');
  const gauge = () =>
    page.getByRole("img", { name: /Validation accuracy|Establishing the baseline/i });
  const outputRegion = page.getByRole("region", { name: "Script output" });

  // ── shell anatomy, once the baseline has been fitted ──
  console.log("\nShell anatomy");
  await page
    .getByRole("img", { name: /Validation accuracy \d/ })
    .waitFor({ timeout: 120000 });

  check(
    "metric readout present",
    (await page.locator('[data-testid="metric-value"]').count()) > 0,
  );
  check(
    "the live metric is metric lift, per the catalog",
    /Metric lift/.test(await page.locator("body").innerText()),
  );
  check("metric gauge present", await gauge().isVisible());
  check(
    "transform picker is a radio group",
    (await page.getByRole("radio", { name: /Bin/ }).count()) > 0,
  );
  check(
    "column tray is a checkbox group",
    (await page.getByRole("checkbox").count()) >= 7,
    `${await page.getByRole("checkbox").count()} checkboxes`,
  );
  check(
    "the leaky column is flagged as risky before it is used",
    /risky/i.test(await page.locator("main, form, aside").first().innerText()) ||
      /risky/i.test(await page.locator("body").innerText()),
  );
  check(
    "importance bars are a real table",
    (await page.getByRole("table").count()) > 0,
  );
  check("why-card docked", await whyRegion.isVisible());
  check(
    "the briefing states the model is fixed",
    /fixed/i.test(await whyRegion.innerText()),
  );

  const baselineLabel = (await gauge().getAttribute("aria-label")) ?? "";
  const baseline = Number(
    baselineLabel.match(/baseline of ([\d.]+) percent/)?.[1] ?? NaN,
  );
  check(
    "a baseline was established by training the fixed model",
    Number.isFinite(baseline) && baseline > 50 && baseline < 80,
    `baseline ${baseline}%`,
  );

  // ── contract #2: forging a feature IS feature engineering ──
  console.log("\nPlayer action = forging a feature (contract #2)");
  const liftBefore = await metricText();
  await forge(page, "Day of week", "signup_ts");

  check(
    "forging a feature moves the lift metric",
    (await metricText()) !== liftBefore,
    `${liftBefore} -> ${await metricText()}`,
  );
  check(
    "it is recognised as a legendary transform, with a reason",
    /Legendary/i.test(await whyRegion.innerText()) &&
      /weekday/i.test(await whyRegion.innerText()),
    (await whyRegion.innerText()).split("\n")[0] ?? "",
  );
  check(
    // The matrix columns are labelled from the transform's display name, so they
    // read "day of week(signup_ts)=0" rather than using the transform id.
    "the importance table re-ranks to include the new columns",
    /day of week\(signup_ts\)=\d/.test(await page.locator("main").innerText()),
  );

  const announced = await page
    .waitForFunction(
      () =>
        /metric lift/i.test(
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

  // ── contract #4a: No lift — a transform that only changes units ──
  console.log("\nNamed failure: No lift");
  await page.getByRole("button", { name: /Empty the forge/ }).click();
  await settle(page);
  await forge(page, "Log \\+ scale", "income");
  await page.getByRole("button", { name: /Score this forge/ }).click();
  await page.waitForTimeout(400);

  const noLift = (await failure.count()) > 0;
  check("a units-only transform fails to lift", noLift);
  if (noLift) {
    const text = await failure.innerText();
    check(
      "failure is NAMED 'No lift', not generic",
      /no lift/i.test(text) && !/game over/i.test(text),
      text.split("\n")[0],
    );
    check(
      "it explains that scaling changes units, not meaning",
      /units/i.test(text),
      text.replace(/\n/g, " ").slice(0, 140),
    );
    check(
      "it points out the baseline was already standardised",
      /already standardised/i.test(text),
    );
  }

  // ── contract #4b: Leakage — the opposite problem, a score that is too good ──
  console.log("\nNamed failure: Leakage (a score that is too good)");
  await page.getByRole("button", { name: /Empty the forge/ }).click();
  await settle(page);
  await forge(page, "Standardise", "refund_issued");

  const leakLabel = (await gauge().getAttribute("aria-label")) ?? "";
  const leakScore = Number(
    leakLabel.match(/Validation accuracy ([\d.]+) percent/)?.[1] ?? NaN,
  );
  check(
    "the leaky column produces the best score in the game",
    Number.isFinite(leakScore) && leakScore > 85,
    `${leakScore}% validation accuracy`,
  );
  check(
    "and the gauge says out loud that it is not real",
    /not real/i.test(leakLabel),
    leakLabel.slice(0, 120),
  );

  const leaked = (await failure.count()) > 0;
  check("leakage is caught immediately, without waiting to be scored", leaked);
  if (leaked) {
    const text = await failure.innerText();
    check(
      "this failure is NAMED 'Leakage' — a different diagnosis",
      /leakage/i.test(text),
      text.split("\n")[0],
    );
    check(
      "it names the offending column and when it gets set",
      /refund_issued/.test(text) && /after/i.test(text),
      text.replace(/\n/g, " ").slice(0, 150),
    );
    check(
      "the metric goes red even though the number went up",
      ((await page
        .locator('[data-testid="metric-value"]')
        .first()
        .getAttribute("class")) ?? "").includes("text-wrong"),
    );
  }

  // ── the Python lane: real pandas, and the same store ──
  console.log("\nPython lane: real CPython with pandas");
  await page.getByRole("button", { name: /Empty the forge/ }).click();
  await settle(page);
  // Exact: /code/i also matches transform blurbs containing "one-hot encoded".
  await page.getByRole("radio", { name: "Code", exact: true }).click();

  const editor = page.getByLabel("Feature script");
  check("python lane editor present", await editor.isVisible());
  check(
    "it warns about the download before fetching anything",
    /20 MB|WebAssembly/i.test(await page.locator("main").innerText()),
  );

  await editor.fill(
    [
      "import sys, pandas as pd",
      "df = table()",
      "print('python', sys.version.split()[0])",
      "print('pandas', pd.__version__)",
      "print('rows', len(df))",
      "# The weekday effect, straight out of the data.",
      "df['wd'] = (df.signup_ts // 86400) % 7",
      "rates = df.groupby('wd').churned.mean()",
      "print('weekend', round(float(rates[[0, 6]].mean()), 3),",
      "      'midweek', round(float(rates[[2, 3, 4]].mean()), 3))",
      "before = score()",
      "after = await forge('day_of_week', 'signup_ts')",
      "print('forged from python', round(before, 4), '->', round(after, 4))",
      "print('features', [f['label'] for f in features()])",
      "print('PYDONE')",
    ].join("\n"),
  );

  await page.getByRole("button", { name: /Run Python/ }).click();

  // Downloads the runtime plus pandas, then fits a model. Waits for the run to
  // FINISH rather than for the success marker, so a Python error is reported as
  // itself instead of as a timeout.
  await page
    .getByRole("button", { name: "Run Python" })
    .waitFor({ state: "visible", timeout: 300000 });

  const output = await outputRegion.innerText();
  check(
    "the Python run completed without an error",
    /PYDONE/.test(output),
    output.replace(/\n/g, " | ").slice(0, 220),
  );
  check(
    "real CPython is running",
    /python 3\.\d+/.test(output),
    output.split("\n").find((l) => l.startsWith("python")) ?? "",
  );
  check(
    "real pandas is running",
    /pandas \d+\./.test(output),
    output.split("\n").find((l) => l.startsWith("pandas")) ?? "",
  );
  check(
    "pandas read the training table",
    /rows 1200/.test(output),
    output.split("\n").find((l) => l.startsWith("rows")) ?? "",
  );
  check(
    "a groupby found the weekend effect the transform exists to capture",
    (() => {
      const match = output.match(/weekend ([\d.]+) midweek ([\d.]+)/);
      return match !== null && Number(match[1]) > Number(match[2]) + 0.05;
    })(),
    output.split("\n").find((l) => l.startsWith("weekend")) ?? "",
  );
  check(
    "Python forged a feature into the same store",
    /forged from python/.test(output) && /day of week\(signup_ts\)/.test(output),
    output.split("\n").find((l) => l.startsWith("forged")) ?? "",
  );

  // Back to the visual lane: the Python-forged feature must be there.
  await page.getByRole("radio", { name: "Visual", exact: true }).click();
  await page.waitForTimeout(400);
  check(
    "the visual lane shows the feature Python forged — one store, two lanes",
    /day of week\(signup_ts\)/.test(await page.locator("main").innerText()),
  );

  // ── the win: lift from representation alone ──
  console.log("\nForging a winning set");
  await forge(page, "Bin", "age");
  await forge(page, "One-hot", "city_code");
  await forge(page, "Ratio", "income", "household");

  const finalLabel = (await gauge().getAttribute("aria-label")) ?? "";
  const finalLift = Number(
    finalLabel.match(/lift of ([-\d.]+) points/)?.[1] ?? NaN,
  );
  check(
    "all four legendary transforms lift past the target",
    Number.isFinite(finalLift) && finalLift >= 6,
    `lift ${finalLift} points`,
  );

  await page.getByRole("button", { name: /Score this forge/ }).click();
  await page.waitForTimeout(400);
  check(
    "the forge is accepted",
    (await failure.count()) === 0,
    (await failure.count()) === 0 ? "no failure" : await failure.innerText(),
  );
  check(
    "and the win credits representation, not the model",
    /model never got bigger|Representation did all of it/i.test(
      await whyRegion.innerText(),
    ),
    (await whyRegion.innerText()).split("\n")[0] ?? "",
  );
  check(
    "XP bar present",
    await page
      .getByRole("progressbar", { name: "Experience points" })
      .isVisible(),
  );
}
