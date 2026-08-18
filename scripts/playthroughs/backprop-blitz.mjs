/**
 * Browser playthrough for Backprop Blitz.
 *
 * What this verifies that no other playthrough can:
 *
 *   1. The player really performs the chain rule: each step is a rule choice, and
 *      the numbers it produces are shown against a real autograd trace.
 *   2. A wrong gradient is ALLOWED to keep flowing, and the panel distinguishes
 *      "your rule made this wrong" from "this inherited someone else's mistake".
 *   3. The two hardest lessons land as measurements: a halved gradient points 0°
 *      from the truth at half the length, and a sign flip diverges.
 *   4. A wrong rule that coincidentally gives the right answer is caught and said
 *      out loud, then collapses in the scenario that shuts the gate.
 */

export const slug = "backprop-blitz";
export const title = "Backprop Blitz";

/**
 * Pick a rule at the currently focused step.
 *
 * Choosing deliberately does NOT advance — the player is meant to read the
 * gradient their rule produced — so advancing is a separate, explicit act.
 */
async function pick(page, labelPattern) {
  await page.getByRole("radio", { name: labelPattern }).check();
  await page.waitForTimeout(120);
}

/** Choose, then move to the next node. */
async function pickAndAdvance(page, labelPattern) {
  await pick(page, labelPattern);
  const forward = page.getByRole("button", { name: /^(Next: |Forward$)/ });
  if ((await forward.count()) > 0 && (await forward.first().isEnabled())) {
    await forward.first().click();
    await page.waitForTimeout(100);
  }
}

/** The nine correct choices, in walk order: L d y s v acc(h) h z u. */
const CORRECT_WALK = [
  /g × 2d/,
  /pass g through unchanged/,
  /copy g to both inputs/,
  /copy g to both inputs/,
  /each input gets g × the OTHER input/,
  /add the arriving gradients/,
  /pass g only if the input was positive/,
  /copy g to both inputs/,
  /each input gets g × the OTHER input/,
];

async function walkCorrectly(page, { except } = {}) {
  for (let step = 0; step < CORRECT_WALK.length; step += 1) {
    const rule = except?.step === step ? except.rule : CORRECT_WALK[step];
    if (step === CORRECT_WALK.length - 1) {
      await pick(page, rule);
    } else {
      await pickAndAdvance(page, rule);
    }
  }
  await page.waitForTimeout(200);
}

const metricNumber = async (page) =>
  Number(
    (
      (await page.locator('[data-testid="metric-value"]').first().innerText()) ??
      ""
    ).replace(/[^\d.-]/g, ""),
  );

const secondary = async (page, label) => {
  const text = await page.locator("body").innerText();
  const match = text.match(
    new RegExp(`${label}[\\s\\S]{0,60}?(-?[\\d.]+|Infinity)`),
  );
  return match ? match[1] : null;
};

export async function run({ page, check }) {
  const whyRegion = page.getByRole("region", { name: /why did that happen/i });
  const failure = page.locator('[data-testid="named-failure"]');
  const controls = page.locator("#game-canvas + div");
  const main = page.locator("#game-canvas");
  const outputRegion = page.getByRole("region", { name: "Script output" });
  const graph = () => page.getByRole("img", { name: /^Computation graph/ });

  // ── shell anatomy ──
  console.log("\nShell anatomy");
  await graph().waitFor({ timeout: 60000 });

  check(
    "metric readout present",
    (await page.locator('[data-testid="metric-value"]').count()) > 0,
  );
  check(
    "the live metric is gradient correctness, per the catalog",
    /Gradient correctness/.test(await page.locator("body").innerText()),
  );
  check("compute graph present", await graph().isVisible());
  check(
    "the graph is also a table, because the numbers are the content",
    (await page.locator("figure table").count()) > 0,
  );
  check(
    "the correctness meter is a real progressbar",
    await page
      .getByRole("progressbar", { name: /matching the autograd trace/ })
      .isVisible(),
  );
  check("why-card docked", await whyRegion.isVisible());
  check(
    "the briefing says wrong gradients are allowed to flow",
    /poisons everything behind it/.test(await whyRegion.innerText()),
    (await whyRegion.innerText()).split("\n")[1]?.slice(0, 90) ?? "",
  );

  const graphLabel = (await graph().getAttribute("aria-label")) ?? "";
  check(
    "the graph reports every node's value and gradient state in words",
    /L equals 5\.06/.test(graphLabel) &&
      /gradient not yet routed/.test(graphLabel),
    graphLabel.slice(0, 110),
  );
  check(
    "the forward pass has already run and the loss is shown",
    /loss 5\.063/.test(await main.innerText()),
  );
  check(
    "nothing is routed yet, and the meter says why the loss is the exception",
    /derivative of anything with respect to itself/.test(
      await controls.innerText(),
    ),
  );

  // ── contract #2: routing the error backward IS the chain rule ──
  console.log("\nPlayer action = executing the chain rule (contract #2)");
  check(
    "step 1 of 9 is the loss node",
    /Step 1 of 9 · node L/.test(await main.innerText()),
    (await main.innerText()).match(/Step \d of \d · node \w+/)?.[0] ?? "",
  );

  const before = await metricNumber(page);
  await pick(page, /g × 2d/);
  const after = await metricNumber(page);

  check(
    "choosing a rule moves the live metric",
    after > before,
    `${before}% -> ${after}%`,
  );
  check(
    "the panel stays put so the result of the choice can be read",
    /Step 1 of 9 · node L/.test(await main.innerText()),
  );
  check(
    "and offers the next node explicitly",
    await page.getByRole("button", { name: /^Next: d$/ }).isVisible(),
  );
  check(
    "the why-card explains the rule, not just that it was right",
    /derivative of d/.test(await whyRegion.innerText()),
    (await whyRegion.innerText()).split("\n")[1]?.slice(0, 90) ?? "",
  );
  check(
    "the gradient it sends is shown against the autograd value",
    /What that sends onward/.test(await main.innerText()) &&
      /\+4\.500/.test(await main.innerText()),
  );

  const announced = await page
    .waitForFunction(
      () =>
        /gradient correctness/i.test(
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

  // ── a wrong rule is allowed to flow, and its damage is attributed ──
  console.log("\nA wrong rule keeps flowing");
  await page.getByRole("button", { name: /^Next: d$/ }).click();
  await pickAndAdvance(page, /pass g through unchanged/);
  await pickAndAdvance(page, /split g in half/);

  check(
    "the why-card says the wrong number is still flowing",
    /still flowing/.test(await whyRegion.innerText()),
    (await whyRegion.innerText()).split("\n")[0] ?? "",
  );
  check(
    "the immediate victims of the bad rule are attributed to it",
    /wrong here/.test(await controls.innerText()),
    (await controls.innerText()).match(/\d+ wrong here[^\n]*/)?.[0] ?? "",
  );

  // Inheritance only exists once the walk has gone further back than the mistake,
  // so carry on a few nodes before asking about it.
  await pickAndAdvance(page, CORRECT_WALK[3]);
  await pickAndAdvance(page, CORRECT_WALK[4]);
  await pickAndAdvance(page, CORRECT_WALK[5]);

  check(
    "and nodes further back are marked as having inherited it, not as new mistakes",
    /only wrong because something downstream is/.test(
      await controls.innerText(),
    ) && /inherited/.test(await controls.innerText()),
    (await controls.innerText()).match(/\d+ wrong here[^\n]*/)?.[0] ?? "",
  );

  // ── named failure: halving at an add is a learning-rate bug in disguise ──
  console.log("\nNamed failure: Add nodes copy, they do not split");
  for (let step = 6; step < CORRECT_WALK.length; step += 1) {
    if (step === CORRECT_WALK.length - 1) {
      await pick(page, CORRECT_WALK[step]);
    } else {
      await pickAndAdvance(page, CORRECT_WALK[step]);
    }
  }
  await page.waitForTimeout(200);

  const halved = (await failure.count()) > 0;
  check("finishing the walk wrong names the failure", halved);
  if (halved) {
    const text = await failure.innerText();
    check(
      "failure is NAMED for the misconception, not 'incorrect'",
      /Add nodes copy, they do not split/.test(text) && !/game over/i.test(text),
      text.split("\n")[0],
    );
    check(
      "it explains that a gradient is a rate, not a quantity to share out",
      /it is a rate/.test(text),
      text.replace(/\n/g, " ").slice(0, 170),
    );
    check(
      "it reports the direction is RIGHT and only the length is wrong",
      /right DIRECTION/.test(text) && /0\.50 times its length/.test(text),
      text.match(/right DIRECTION[^.]*\./)?.[0] ?? "",
    );
    check(
      "and calls that a learning-rate bug in disguise",
      /learning-rate bug in disguise/.test(text),
    );
    check(
      "it does NOT claim the loss went up, because measured it does not",
      !/loss.{0,12}UP/i.test(text),
    );
  }

  const angle = await secondary(page, "Gradient angle");
  check(
    "the angle metric reads zero degrees for a halved gradient",
    angle !== null && Math.abs(Number(angle)) < 0.5,
    `angle ${angle}°`,
  );

  // ── named failure: a sign flip points uphill and diverges ──
  console.log("\nNamed failure: wrong sign — this one really does diverge");
  await page.getByRole("button", { name: /Start this scenario again/ }).click();
  await page.waitForTimeout(250);
  await walkCorrectly(page, { except: { step: 1, rule: /negate g/ } });

  const flipped = (await failure.count()) > 0;
  check("a flipped sign is caught", flipped);
  if (flipped) {
    const text = await failure.innerText();
    check(
      "failure is NAMED for the sign, a different diagnosis",
      /wrong sign/i.test(text) && !/Add nodes copy/.test(text),
      text.split("\n")[0],
    );
    check(
      "it says the gradient points UPHILL at 180 degrees",
      /UPHILL/.test(text) && /180°/.test(text),
      text.match(/points UPHILL[^.]*\./)?.[0] ?? "",
    );
    check(
      "and that training does not converge at all",
      /does not converge at all/.test(text),
    );
  }
  check(
    "the training metric reports divergence rather than a number",
    /diverged — the gradient pointed uphill/.test(
      await page.locator("body").innerText(),
    ),
  );

  // ── the coincidence: a wrong rule that gets the right answer ──
  console.log("\nThe coincidence: right answer, wrong rule");
  await page.getByRole("button", { name: /Start this scenario again/ }).click();
  await page.waitForTimeout(250);
  await walkCorrectly(page, {
    except: { step: 6, rule: /always pass g through/ },
  });

  check(
    "every gradient matches, so the scenario clears",
    (await metricNumber(page)) === 100 && (await failure.count()) === 0,
    `correctness ${await metricNumber(page)}%`,
  );
  check(
    "but the game says a rule is wrong and got away with it",
    /rules is wrong/.test(await whyRegion.innerText()),
    (await whyRegion.innerText()).split("\n")[0] ?? "",
  );
  check(
    "it explains that the gate being open hides the mistake",
    /works for the wrong reason/.test(await whyRegion.innerText()),
  );
  check(
    "the meter flags the coincidence too",
    /got away with it/.test(await controls.innerText()),
  );

  // ── the same rule collapses when the gate shuts ──
  console.log("\nAnd the same rule collapses in the next scenario");
  await page.getByRole("button", { name: /Next scenario/ }).click();
  await page.waitForTimeout(300);
  check(
    "scenario 2 is the shut gate",
    /gate is shut/i.test(await main.innerText()),
    (await main.innerText()).split("\n")[0]?.slice(0, 60) ?? "",
  );

  await walkCorrectly(page, {
    except: { step: 6, rule: /always pass g through/ },
  });
  const collapsed = (await failure.count()) > 0;
  check("the shortcut that worked before now fails", collapsed);
  if (collapsed) {
    const text = await failure.innerText();
    check(
      "and it is named for inventing gradient through a shut gate",
      /shut gate/i.test(text),
      text.split("\n")[0],
    );
  }

  // ── the win ──
  console.log("\nRouting it correctly");
  await page.getByRole("button", { name: /Start this scenario again/ }).click();
  await page.waitForTimeout(250);
  await walkCorrectly(page);

  check(
    "every gradient matches the autograd trace",
    (await metricNumber(page)) === 100,
    `correctness ${await metricNumber(page)}%`,
  );
  check(
    "no failure remains",
    (await failure.count()) === 0,
    (await failure.count()) === 0 ? "clear" : await failure.innerText(),
  );
  check(
    "the win reports no coincidences this time",
    !/rules is wrong/.test(await whyRegion.innerText()),
    (await whyRegion.innerText()).split("\n")[0] ?? "",
  );
  check(
    "XP bar present",
    await page
      .getByRole("progressbar", { name: "Experience points" })
      .isVisible(),
  );

  // The reveal: the real trace alongside the player's.
  await page.getByRole("button", { name: /Show the real trace/ }).click();
  await page.waitForTimeout(250);
  check(
    "the reveal shows the autograd column",
    /Autograd/.test(await controls.innerText()),
  );
  check(
    "and explains that a backward pass has no global step",
    /no global calculation anywhere/.test(await whyRegion.innerText()),
  );

  // ── two lanes, one store ──
  console.log("\nCode lane: measuring every misconception at once");
  await page.getByRole("radio", { name: "Code", exact: true }).click();

  const editor = page.getByLabel("Routing script");
  check("code lane editor present", await editor.isVisible());

  await editor.fill(
    [
      "// Verify the reference trace, then price every wrong rule.",
      "// reset() keeps the current scenario, so get back to the open gate first.",
      "let guard = 0;",
      "while (api.scenario().id !== 'open' && guard++ < 5) api.nextScenario();",
      "api.reset();",
      "log('scenario', api.scenario().id);",
      "const truth = api.autograd();",
      "log('dw1', truth.grads.w1.toFixed(4),",
      "    'db1', truth.grads.b1.toFixed(4),",
      "    'dw2', truth.grads.w2.toFixed(4),",
      "    'db2', truth.grads.b2.toFixed(4));",
      "log('branch: h =', truth.grads.h.toFixed(4),",
      "    '= 4.5 from s + 2.25 from v');",
      "",
      "// Greedy search for the routing that matches the reference.",
      "for (const step of api.steps()) {",
      "  let best = null, bestWrong = Infinity;",
      "  for (const option of step.options) {",
      "    checkBudget();",
      "    api.choose(step.index, option.id);",
      "    const wrong = api.correctness().wrong.length;",
      "    if (wrong < bestWrong) { bestWrong = wrong; best = option.id; }",
      "  }",
      "  api.choose(step.index, best);",
      "}",
      "const c = api.correctness();",
      "log('SOLVED', (c.fraction * 100).toFixed(0) + '%',",
      "    'angle', c.angle.toFixed(1), 'coincidences', c.coincidences.length);",
      "",
      "// Break one rule and price it.",
      "api.choose(2, 'add-split');",
      "const broken = api.correctness();",
      "log('HALVED angle', broken.angle.toFixed(1),",
      "    'length', broken.ratio.toFixed(2));",
      "const run = api.train({ steps: 20 });",
      "const good = api.train({ useTruth: true, steps: 20 });",
      "log('20 steps: mine', run.final.toFixed(5),",
      "    'truth', good.final.toFixed(5));",
      "log('JSDONE');",
    ].join("\n"),
  );

  await page.getByRole("button", { name: /^Run$/ }).click();
  await page
    .getByRole("button", { name: /^Run$/ })
    .waitFor({ state: "visible", timeout: 120000 });
  await page.waitForTimeout(400);

  const output = await outputRegion.innerText();
  check(
    "the script completed without an error",
    /JSDONE/.test(output),
    output.replace(/\n/g, " | ").slice(0, 200),
  );
  check(
    "the reference trace is the one the tests pin to tf.grads",
    /dw1 13\.5000 db1 6\.7500 dw2 9\.0000 db2 4\.5000/.test(output),
    output.split("\n").find((line) => line.startsWith("dw1")) ?? "",
  );
  check(
    "the branch gradient is the sum of both paths",
    /h = 6\.7500 = 4\.5 from s \+ 2\.25 from v/.test(output),
    output.split("\n").find((line) => line.startsWith("branch")) ?? "",
  );
  check(
    "a greedy search finds a routing that matches the reference exactly",
    /SOLVED 100% angle 0\.0 coincidences 0/.test(output),
    output.split("\n").find((line) => line.startsWith("SOLVED")) ?? "",
  );
  check(
    "halving at an add is 0 degrees off at half the length",
    /HALVED angle 0\.0 length 0\.50/.test(output),
    output.split("\n").find((line) => line.startsWith("HALVED")) ?? "",
  );
  check(
    "and 20 steps of descent separates the two trajectories",
    (() => {
      const match = output.match(/20 steps: mine ([\d.]+) truth ([\d.]+)/);
      return match !== null && match[1] !== match[2];
    })(),
    output.split("\n").find((line) => line.startsWith("20 steps")) ?? "",
  );

  await page.getByRole("radio", { name: "Visual", exact: true }).click();
  await page.waitForTimeout(300);
  check(
    "the visual lane shows the routing the code lane wrote — one store, two lanes",
    /split g in half/.test(await main.innerText()) ||
      (await metricNumber(page)) < 100,
    `correctness ${await metricNumber(page)}%`,
  );
}
