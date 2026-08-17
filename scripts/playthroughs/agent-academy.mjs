/**
 * Browser playthrough for Agent Academy.
 *
 * What this verifies that no other playthrough can:
 *
 *   1. The player never touches the agent. Every control writes a reward or an
 *      epsilon, and the agent's behaviour follows from those alone.
 *   2. Reward hacking is legible on screen BEFORE training — the reward editor
 *      solves the reward function by value iteration and says out loud when the
 *      optimum stops being the exit.
 *   3. The headline metric goes UP while the task goes unfinished. That
 *      contradiction is the lesson, so it has to be visible in the DOM.
 *   4. Two failures that look identical get different names, with numbers.
 */

export const slug = "agent-academy";
export const title = "Agent Academy";

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
  await page.waitForTimeout(120);
  return Number(await slider.inputValue());
}

const EPSILON = { min: 0, step: 0.05 };
const CHEESE = { min: 0, step: 0.5 };
const STEP_COST = { min: -2, step: 0.1 };
const PIT = { min: -40, step: 1 };

const EPSILON_LABEL = /chance of a random move/;

async function train(page, times = 1) {
  for (let index = 0; index < times; index += 1) {
    const button = page.getByRole("button", { name: /^Train \d+ episodes/ });
    if ((await button.count()) === 0) return;
    await button.click();
    await page.waitForTimeout(220);
  }
}

export async function run({ page, check, metricText }) {
  const whyRegion = page.getByRole("region", { name: /why did that happen/i });
  const failure = page.locator('[data-testid="named-failure"]');
  const grid = () => page.getByRole("img", { name: /^A 4 by 7 grid/ });
  const curve = () => page.getByRole("img", { name: /Reward per episode|No episodes yet/ });
  // The shell renders the control rail as a plain div immediately after <main>.
  const controls = page.locator("#game-canvas + div");
  const outputRegion = page.getByRole("region", { name: "Script output" });

  // ── shell anatomy ──
  console.log("\nShell anatomy");
  await grid().waitFor({ timeout: 60000 });

  check(
    "metric readout present",
    (await page.locator('[data-testid="metric-value"]').count()) > 0,
  );
  check(
    "the live metric is episode reward, per the catalog",
    /Episode reward/.test(await page.locator("body").innerText()),
  );
  check("grid world present", await grid().isVisible());
  check("reward curve present", await curve().isVisible());
  check(
    "five reward sliders, one per reward",
    (await page.locator('input[type="range"]').count()) === 6,
    `${await page.locator('input[type="range"]').count()} range inputs (5 rewards + epsilon)`,
  );
  check(
    "the Q-value table is a real table, not a picture",
    (await page.getByRole("table").count()) > 0,
  );
  check("why-card docked", await whyRegion.isVisible());
  check(
    "the briefing says the player does not move the agent",
    /do not get to move the agent/i.test(await whyRegion.innerText()),
    (await whyRegion.innerText()).split("\n")[0] ?? "",
  );

  const gridLabel = (await grid().getAttribute("aria-label")) ?? "";
  check(
    "the grid describes itself in words, not only in pixels",
    /cheese/.test(gridLabel) && /pit/.test(gridLabel) && /exit/.test(gridLabel),
    gridLabel.slice(0, 120),
  );
  check(
    "and reports that nothing has trained yet",
    /has not trained yet/.test(gridLabel),
  );

  // ── contract #1/#3: the reward arithmetic is on screen before anything runs ──
  console.log("\nThe reward function is solved before a single episode");
  const controlText = await controls.innerText();
  check(
    "the round trip to the cheese is shown, and it is exactly break-even",
    /Round trip to the cheese[\s\S]{0,40}\+0\.0/.test(controlText),
    controlText.match(/Round trip to the cheese[\s\S]{0,20}/)?.[0]?.replace(/\n/g, " ") ??
      "",
  );
  check(
    "the starting rewards are stated to ask for the exit",
    /do ask for the exit/i.test(controlText),
  );
  check(
    "the epsilon slider starts below what the maze needs",
    Number(await page.getByLabel(EPSILON_LABEL).inputValue()) <= 0.1,
    `epsilon ${await page.getByLabel(EPSILON_LABEL).inputValue()}`,
  );

  // ── contract #2: setting rewards + epsilon IS defining the RL problem ──
  console.log("\nPlayer action = writing the reward function (contract #2)");
  const before = await metricText();
  await train(page, 2);

  check(
    "training moves the live metric",
    (await metricText()) !== before,
    `${before} -> ${await metricText()}`,
  );
  check(
    "the reward curve now has data",
    /Reward per episode over \d+ episodes/.test(
      (await curve().getAttribute("aria-label")) ?? "",
    ),
    ((await curve().getAttribute("aria-label")) ?? "").slice(0, 110),
  );

  const announced = await page
    .waitForFunction(
      () =>
        /episode reward/i.test(
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

  // ── named failure 1: No exploration ──
  console.log("\nNamed failure: No exploration");
  const lowEpsilon = (await failure.count()) > 0;
  check("the default epsilon fails, as designed", lowEpsilon);
  if (lowEpsilon) {
    const text = await failure.innerText();
    check(
      "failure is NAMED 'No exploration', not generic",
      /no exploration/i.test(text) && !/game over/i.test(text),
      text.split("\n")[0],
    );
    check(
      "it proves the rewards were not the problem, with a number",
      /rewards are fine/i.test(text) && /\d+%/.test(text),
      text.replace(/\n/g, " ").slice(0, 170),
    );
    check(
      "it explains that an unvisited Q-value never changes",
      /never revisits|keeps whatever it was initialised to/i.test(text),
    );
  }

  check(
    "the grid shows the agent pacing by the cheese, not travelling",
    /Cheese per episode/.test(await page.locator("body").innerText()) &&
      /farming, not travelling/.test(await page.locator("body").innerText()),
  );

  // The Q-value panel is the mechanism behind that failure: a 0 is not an opinion.
  await page.getByRole("button", { name: "Learned values" }).click();
  await page.waitForTimeout(200);
  const heatLabel = (await grid().getAttribute("aria-label")) ?? "";
  check(
    "the learned-values overlay reports its range in numbers",
    /Q-value overlay on: highest [-\d.]+, lowest [-\d.]+/.test(heatLabel),
    heatLabel.match(/Q-value overlay[^.]*/)?.[0] ?? "",
  );
  check(
    "the grid is also available as a table of cells and values",
    /value [-\d.]+, best action (up|down|left|right)/.test(
      await page.locator("figure table").first().innerText(),
    ),
  );

  // ── the fix: raise epsilon ──
  console.log("\nThe fix: more exploration");
  const raised = await setSlider(page, EPSILON_LABEL, 0.3, EPSILON);
  check("epsilon raised to 0.30", Math.abs(raised - 0.3) < 1e-9, `epsilon ${raised}`);
  check(
    "changing epsilon threw the old agent away, so the comparison is clean",
    /0 of 1200 episodes spent/.test(await controls.innerText()),
    (await controls.innerText()).match(/\d+ of \d+ episodes spent/)?.[0] ?? "",
  );

  await train(page, 3);
  check(
    "the agent graduates on the same rewards that just failed",
    (await failure.count()) === 0 &&
      /New academy/.test(await controls.innerText()),
    (await whyRegion.innerText()).split("\n")[0] ?? "",
  );

  const winLabel = (await grid().getAttribute("aria-label")) ?? "";
  check(
    "and the grid shows a nine-step walk to the exit",
    /greedy walk is 9 steps long and reaches the exit/.test(winLabel),
    winLabel.match(/greedy walk[^.]*/)?.[0] ?? "",
  );
  check(
    "the win credits trial and error from a table of zeros",
    /table of zeros|no idea the exit existed/i.test(await whyRegion.innerText()),
  );
  check(
    "XP bar present",
    await page
      .getByRole("progressbar", { name: "Experience points" })
      .isVisible(),
  );

  // ── named failure 2: Reward hacking — the metric climbs, the task stops ──
  console.log("\nNamed failure: Reward hacking (the metric goes UP)");
  await page.getByRole("button", { name: /New academy/ }).click();
  await page.waitForTimeout(250);

  const cheese = await setSlider(page, /Eat cheese/, 3, CHEESE);
  check("cheese raised to +3.0", Math.abs(cheese - 3) < 1e-9, `cheese ${cheese}`);

  const afterCheese = await controls.innerText();
  check(
    "the editor says the rewards no longer ask for the exit — before any training",
    /do not ask for the exit/i.test(afterCheese),
    afterCheese.match(/Best possible policy:[^.]*\./)?.[0]?.replace(/\n/g, " ") ?? "",
  );
  check(
    "the why-card warns without spending an episode",
    /never leaves/i.test(await whyRegion.innerText()),
    (await whyRegion.innerText()).split("\n")[0] ?? "",
  );
  check(
    "the round-trip readout now shows a profit",
    /Round trip to the cheese[\s\S]{0,40}\+2\.0/.test(afterCheese),
  );

  await setSlider(page, EPSILON_LABEL, 0.3, EPSILON);
  await train(page, 4);

  const hackedMetric = Number(
    (await metricText()).replace(/[^\d.-]/g, ""),
  );
  check(
    "the headline reward is far higher than a winning run scores",
    Number.isFinite(hackedMetric) && hackedMetric > 20,
    `episode reward ${hackedMetric} (a graduate scores 16.5)`,
  );
  check(
    "and the caption says nobody is reaching the exit",
    /0% of rollouts reach the exit/.test(await page.locator("body").innerText()),
  );
  check(
    "the metric goes red even though the number went up",
    ((await page
      .locator('[data-testid="metric-value"]')
      .first()
      .getAttribute("class")) ?? "").includes("text-wrong"),
  );

  const hacked = (await failure.count()) > 0;
  check("reward hacking is caught", hacked);
  if (hacked) {
    const text = await failure.innerText();
    check(
      "this failure is NAMED 'Reward hacking' — a different diagnosis",
      /reward hacking/i.test(text),
      text.split("\n")[0],
    );
    check(
      "it is proved by solving the reward function, not guessed from the run",
      /solved exactly/i.test(text) && /value iteration|no learning involved|four actions/i.test(text),
      text.replace(/\n/g, " ").slice(0, 190),
    );
    check(
      "it states that more episodes and more exploration will not help",
      /More episodes will not help/i.test(text) &&
        /More exploration will not help/i.test(text),
    );
    check(
      "it shows the round-trip arithmetic that makes farming pay",
      /round trip to the cheese clears \+2\.0/i.test(text),
      text.match(/round trip[^.]*\./)?.[0] ?? "",
    );
  }

  // The "what your rewards ask for" overlay should now loop rather than exit.
  await page.getByRole("button", { name: /What your rewards ask for/ }).click();
  await page.waitForTimeout(200);
  check(
    "the overlay legend distinguishes learned behaviour from requested behaviour",
    /what your rewards ask for/i.test(await page.locator("figure").first().innerText()),
  );

  // ── named failure 3: same symptom, different name and different fix ──
  console.log("\nNamed failure: Quitting beats trying (same symptom, other cause)");
  await page.getByRole("button", { name: /New academy/ }).click().catch(() => {});
  await page.waitForTimeout(250);
  await setSlider(page, /Eat cheese/, 1, CHEESE);
  await setSlider(page, /Fall in the pit/, 0, PIT);
  await setSlider(page, /Per move/, -1, STEP_COST);
  await setSlider(page, EPSILON_LABEL, 0.3, EPSILON);

  const punishing = await controls.innerText();
  check(
    "these rewards STILL ask for the exit — so this is not reward hacking",
    /do ask for the exit/i.test(punishing),
    punishing.match(/Best possible policy:[^.]*\./)?.[0]?.replace(/\n/g, " ") ?? "",
  );

  await train(page, 4);

  const quitting = (await failure.count()) > 0;
  check("the agent still fails, for a third distinct reason", quitting);
  if (quitting) {
    const text = await failure.innerText();
    check(
      "failure is NAMED 'Quitting beats trying' — not hacking, not exploration",
      /quitting beats trying/i.test(text) &&
        !/reward hacking/i.test(text) &&
        !/^no exploration/i.test(text),
      text.split("\n")[0],
    );
    check(
      "it distinguishes achievable from learnable",
      /achievable in principle is not the same as being findable/i.test(text),
    );
    check(
      "it cites the cost of searching against the cost of stopping",
      /-9\.0/.test(text) && /pit/i.test(text),
      text.replace(/\n/g, " ").slice(0, 190),
    );
    check(
      "and the rollouts really do end in the pit",
      /% fall in/.test(await page.locator("body").innerText()),
    );
  }

  // ── two lanes, one store ──
  console.log("\nCode lane: the sweep the grid cannot show");
  await page.getByRole("radio", { name: "Code", exact: true }).click();

  const editor = page.getByLabel("Training script");
  check("code lane editor present", await editor.isVisible());

  await editor.fill(
    [
      "// Prove the epsilon claim over seeds, then prove exploration cannot",
      "// rescue a hacked reward function.",
      "// api.simulate inherits the CURRENT rewards, so put them back first.",
      "api.setReward('step', -0.5);",
      "api.setReward('trap', -20);",
      "api.setReward('pellet', 1);",
      "log('rewards restored:', JSON.stringify(api.rewards()));",
      "let lowSolved = 0, highSolved = 0, hackedSolved = 0;",
      "for (let seed = 1; seed <= 8; seed++) {",
      "  checkBudget();",
      "  if (api.simulate({ epsilon: 0, episodes: 600, seed })?.solved) lowSolved++;",
      "  if (api.simulate({ epsilon: 0.3, episodes: 300, seed })?.solved) highSolved++;",
      "  if (api.simulate({ rewards: { pellet: 3 }, epsilon: 0.9, episodes: 600, seed })?.solved) hackedSolved++;",
      "}",
      "log('epsilon 0        solved', lowSolved, 'of 8');",
      "log('epsilon 0.3      solved', highSolved, 'of 8');",
      "log('cheese 3, e 0.9  solved', hackedSolved, 'of 8');",
      "const plain = api.solve({ pellet: 1 });",
      "const rich = api.solve({ pellet: 3 });",
      "log('cheese 1 best policy', plain.reachesGoal ? 'walks out' : 'never leaves',",
      "    'for', plain.episodeReward.toFixed(1));",
      "log('cheese 3 best policy', rich.reachesGoal ? 'walks out' : 'never leaves',",
      "    'for', rich.episodeReward.toFixed(1));",
      "api.setEpsilon(0.3);",
      "api.train(300);",
      "log('my agent goal rate', api.report().goalRate.toFixed(2),",
      "    'after', api.episodes(), 'episodes');",
      "log('JSDONE');",
    ].join("\n"),
  );

  await page.getByRole("button", { name: /^Run$/ }).click();
  await page
    .getByRole("button", { name: /^Run$/ })
    .waitFor({ state: "visible", timeout: 180000 });
  await page.waitForTimeout(400);

  const output = await outputRegion.innerText();
  check(
    "the script completed without an error",
    /JSDONE/.test(output),
    output.replace(/\n/g, " | ").slice(0, 220),
  );
  check(
    "epsilon 0 solves nothing, on every seed",
    /epsilon 0\s+solved 0 of 8/.test(output),
    output.split("\n").find((line) => line.startsWith("epsilon 0 ")) ?? "",
  );
  check(
    "epsilon 0.3 solves everything, on half the episodes",
    /epsilon 0\.3\s+solved 8 of 8/.test(output),
    output.split("\n").find((line) => line.startsWith("epsilon 0.3")) ?? "",
  );
  check(
    "and no amount of exploration rescues an over-paid cheese",
    /cheese 3, e 0\.9\s+solved 0 of 8/.test(output),
    output.split("\n").find((line) => line.startsWith("cheese 3,")) ?? "",
  );
  check(
    "value iteration explains why, with no training at all",
    /cheese 1 best policy walks out for 16\.5/.test(output) &&
      /cheese 3 best policy never leaves for 80\.0/.test(output),
    output.split("\n").filter((line) => line.includes("best policy")).join(" | "),
  );
  check(
    "the code lane can rewrite the reward function through the same store",
    /rewards restored: \{"goal":20,"trap":-20,"step":-0\.5,"pellet":1,"bump":-1\}/.test(
      output,
    ),
    output.split("\n").find((line) => line.startsWith("rewards restored")) ?? "",
  );
  check(
    "the code lane trained the player's own agent through the same store",
    /my agent goal rate 1\.00 after 300 episodes/.test(output),
    output.split("\n").find((line) => line.startsWith("my agent")) ?? "",
  );

  await page.getByRole("radio", { name: "Visual", exact: true }).click();
  await page.waitForTimeout(400);
  check(
    "the visual lane shows the agent the code lane trained — one store, two lanes",
    /300 of 1200 episodes spent/.test(await controls.innerText()),
    (await controls.innerText()).match(/\d+ of \d+ episodes spent/)?.[0] ?? "",
  );
  check(
    "and it graduated",
    /100% of rollouts reach the exit|New academy/.test(
      await page.locator("body").innerText(),
    ),
  );
}
