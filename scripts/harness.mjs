/**
 * Shared by the browser harnesses: `a11y-audit.mjs`, `verify-playthrough.mjs`
 * and `verify-mobile.mjs`.
 *
 * Two things live here because each harness had, or would have had, its own
 * copy, and copies drift:
 *
 * 1. **The route list.** The harnesses run under plain node and cannot import
 *    the TypeScript modules that define the site (`src/games/registry.ts`,
 *    `src/lib/concepts.ts`), so the a11y audit used to keep a hand-written
 *    list — and a page missing from a hand-written list is a page nobody
 *    audits. Instead the slugs are read out of those two source files.
 *    Parsing source with a regex is only safe if something checks the parse,
 *    so `src/lib/concepts.test.ts` and `src/games/registry.test.ts` assert
 *    these functions return exactly `conceptSlugs()` and `playableSlugs()`.
 *
 * 2. **What counts as a real console problem.** Headless Chromium on a CI box
 *    has no GPU, and says so, loudly, on every WebGL page. Those lines are
 *    noise; a KaTeX or React warning is a bug. Both harnesses that watch the
 *    console need the same answer, so the answer is written once.
 *
 * 3. **The ƒ Math dialog's contract** (`checkMathDialog`), which every game
 *    shares through GameShell and so is checked once, the same way, per game.
 *
 * Plain ESM with no dependencies beyond node, so the unit tests can import it.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

/** @param {string} relative */
function readSource(relative) {
  return readFileSync(join(ROOT, relative), "utf8");
}

/**
 * Slugs in `PLAYABLE_SLUGS` in `src/games/registry.ts`, in order.
 * @param {string} [source] the registry's source text (tests pass their own)
 * @returns {string[]}
 */
export function playableSlugsFromSource(
  source = readSource("src/games/registry.ts"),
) {
  const block = source.match(/PLAYABLE_SLUGS\s*=\s*\[([\s\S]*?)\]/);
  const slugs = block
    ? [...(block[1] ?? "").matchAll(/["']([a-z0-9-]+)["']/g)].map((m) => m[1])
    : [];
  if (slugs.length === 0) {
    throw new Error(
      "scripts/harness.mjs: found no PLAYABLE_SLUGS in src/games/registry.ts — did its shape change?",
    );
  }
  return /** @type {string[]} */ (slugs);
}

/**
 * Slugs of the entries in `CONCEPT_LIBRARY` in `src/lib/concepts.ts`, in order.
 * Only string-literal `slug: "..."` properties after the declaration count; the
 * interface's `slug: string;` above it is not a match.
 * @param {string} [source] the library's source text (tests pass their own)
 * @returns {string[]}
 */
export function conceptSlugsFromSource(
  source = readSource("src/lib/concepts.ts"),
) {
  const start = source.indexOf("CONCEPT_LIBRARY");
  const body = start === -1 ? "" : source.slice(start);
  const slugs = [...body.matchAll(/\bslug:\s*["']([a-z0-9-]+)["']/g)].map(
    (m) => m[1],
  );
  if (slugs.length === 0) {
    throw new Error(
      "scripts/harness.mjs: found no concept slugs in src/lib/concepts.ts — did its shape change?",
    );
  }
  return /** @type {string[]} */ (slugs);
}

/**
 * @typedef {object} SiteRoute
 * @property {string} path
 * @property {"page" | "game" | "not-found"} kind
 * @property {number} status the HTTP status a healthy deploy returns
 * @property {string} [slug] the game slug, for `kind: "game"`
 */

/**
 * Two deliberately-unknown URLs, one per dynamic segment. Both must be real
 * 404s: `dynamicParams = false` on each route turns an unknown slug into the
 * static not-found page instead of an on-demand render.
 */
export const NOT_FOUND_PATHS = ["/play/does-not-exist", "/concepts/does-not-exist"];

/**
 * Every route a harness should visit: the home page, every playable game, the
 * Concept Library and each of its pages, and the 404 page.
 * @returns {SiteRoute[]}
 */
export function siteRoutes() {
  return [
    { path: "/", kind: "page", status: 200 },
    ...playableSlugsFromSource().map((slug) => ({
      path: `/play/${slug}`,
      kind: /** @type {const} */ ("game"),
      status: 200,
      slug,
    })),
    { path: "/concepts", kind: "page", status: 200 },
    ...conceptSlugsFromSource().map((slug) => ({
      path: `/concepts/${slug}`,
      kind: /** @type {const} */ ("page"),
      status: 200,
    })),
    ...NOT_FOUND_PATHS.map((path) => ({
      path,
      kind: /** @type {const} */ ("not-found"),
      status: 404,
    })),
  ];
}

/**
 * The route a URL or path refers to, if it is one of ours. Lets a harness that
 * was handed an explicit URL still know it is looking at the 404 page.
 * @param {string} urlOrPath
 * @returns {SiteRoute | undefined}
 */
export function routeFor(urlOrPath) {
  const path = new URL(urlOrPath, "http://harness.invalid").pathname;
  return siteRoutes().find((route) => route.path === path);
}

// ── Console hygiene ──────────────────────────────────────────────────────────

/**
 * Headless Chromium with no GPU. Chrome narrates its SwiftShader fallback on
 * every WebGL page, and none of it is ours to fix.
 */
const HEADLESS_GPU_NOISE = [
  /GroupMarkerNotSet/,
  /Automatic fallback to software WebGL/i,
  /--enable-unsafe-swiftshader/,
  /SwiftShader/i,
  /GL Driver Message/,
  /GPU stall due to ReadPixels/,
];

/**
 * TF.js tries the WebGL backend first and falls back to CPU when it can't get a
 * context. The fallback is designed behaviour and the games work on it; the two
 * warnings it prints on the way are not a bug.
 */
const TFJS_BACKEND_FALLBACK = [
  /Initialization of backend webgl failed/,
  /WebGL is not supported on this device/,
];

/**
 * Warnings that mean something in the app is actually wrong. KaTeX prints a
 * warning (not an error) for input it renders differently from LaTeX, which is
 * exactly how a broken equation reached production unnoticed; React reports
 * hydration mismatches and key/update-loop bugs the same way.
 */
const BUG_WARNINGS = [
  /LaTeX-incompatible input/,
  /No character metrics for/,
  /\bKaTeX\b/,
  /^Warning: /,
  /hydrat/i,
  /unique "key" prop/,
  /Maximum update depth/,
  /Cannot update a component/,
];

/**
 * @typedef {object} ConsoleMessage
 * @property {string} type Playwright's `ConsoleMessage.type()`
 * @property {string} text
 * @property {string} [resourceUrl] `ConsoleMessage.location().url`, if any
 */

/**
 * @typedef {object} ConsoleContext
 * @property {string} [pageUrl] the URL the harness navigated to
 * @property {boolean} [expectNotFound] true on a route that should 404
 */

/**
 * `"fail"` — a real problem; the harness should fail.
 * `"warn"` — worth printing, not proof of a bug.
 * `"ignore"` — known environment noise, or not a problem at all.
 *
 * @param {ConsoleMessage} message
 * @param {ConsoleContext} [context]
 * @returns {"fail" | "warn" | "ignore"}
 */
export function classifyConsoleMessage(message, context = {}) {
  const { type, text, resourceUrl } = message;
  const noise = [...HEADLESS_GPU_NOISE, ...TFJS_BACKEND_FALLBACK];

  if (type === "error") {
    // On the 404 route, the browser logs the document's own 404 as a failed
    // resource. That is the page working. Any OTHER 404 there is still a bug.
    if (
      context.expectNotFound &&
      /status of 404/.test(text) &&
      (!resourceUrl || !context.pageUrl || sameDocument(resourceUrl, context.pageUrl))
    ) {
      return "ignore";
    }
    if (noise.some((pattern) => pattern.test(text))) return "ignore";
    return "fail";
  }

  if (type === "warning") {
    if (noise.some((pattern) => pattern.test(text))) return "ignore";
    if (BUG_WARNINGS.some((pattern) => pattern.test(text))) return "fail";
    return "warn";
  }

  return "ignore";
}

/** @param {string} a @param {string} b */
function sameDocument(a, b) {
  try {
    const left = new URL(a);
    const right = new URL(b);
    return left.origin === right.origin && left.pathname === right.pathname;
  } catch {
    return false;
  }
}

// ── The ƒ Math dialog ────────────────────────────────────────────────────────

/**
 * The ƒ Math dialog, checked the same way on every game. verify-playthrough
 * runs it after each game's own playthrough. MathDrawer's unit tests run in
 * jsdom; this runs each game's real equation in a real browser:
 *
 * - KaTeX actually rendered it. Its chunk loads on first open, so a closed-state
 *   audit never sees it, and a parse failure only shows as the LaTeX source.
 * - Focus moves in, and neither Tab nor Shift+Tab can walk out — including
 *   after a click on the dialog's text, which used to drop focus onto the page
 *   behind, from where Tab walked straight back into the game.
 * - Escape closes it, and focus returns to the button that opened it.
 *
 * The button and the dialog are found by role and accessible name (the dialog
 * is labelled "Reveal the math: <title>"), so a markup change that keeps the
 * contract keeps passing; the equation by MathDrawer's `math-equation` test
 * hook and KaTeX's own class names. Lives here, with no Playwright import, so
 * it needs nothing but the `page` it is handed.
 *
 * @param {import("playwright").Page} page a game page, with the game mounted
 * @param {(name: string, passed: boolean, detail?: string) => void} check
 */
export async function checkMathDialog(page, check) {
  console.log("\nReveal the Math dialog (shared)");
  const triggers = page.getByRole("button", { name: /^(ƒ\s*)?Math$/ });
  const triggerCount = await triggers.count();
  check("one Math button in the game header", triggerCount === 1, `found ${triggerCount}`);
  if (triggerCount !== 1) return;
  // Held as a handle: the header goes `inert` while the dialog is open, which
  // takes the button out of the accessibility tree a role query searches.
  const trigger = await triggers.elementHandle();
  if (!trigger) return;

  try {
    await triggers.click();
    const dialog = page.getByRole("dialog", { name: /reveal the math/i });
    const opened = await dialog
      .waitFor({ state: "visible", timeout: 5000 })
      .then(() => true, () => false);
    check("Math button opens a named dialog", opened);
    if (!opened) return;
    check(
      "Math button reports the dialog as expanded",
      (await trigger.getAttribute("aria-expanded")) === "true",
    );

    // MathDrawer marks the equation box `data-testid="math-equation"` with
    // `data-state` loading | rendered | source ("source" = KaTeX couldn't parse
    // it, or its chunk didn't load, so the LaTeX is shown as text). Wait for it
    // to leave "loading" rather than for a fixed time; without the hook, wait
    // for KaTeX's own markup instead.
    const box = dialog.locator('[data-testid="math-equation"]');
    let state = null;
    if ((await box.count()) > 0) {
      await box
        .and(page.locator(':not([data-state="loading"])'))
        .waitFor({ state: "attached", timeout: 15000 })
        .catch(() => {});
      state = await box.getAttribute("data-state");
    } else {
      await dialog
        .locator(".katex")
        .first()
        .waitFor({ state: "attached", timeout: 15000 })
        .catch(() => {});
    }
    const rendered =
      (state === null || state === "rendered") &&
      (await dialog.locator(".katex").count()) > 0;
    check(
      "the equation rendered with KaTeX, not the LaTeX fallback",
      rendered,
      rendered
        ? ""
        : `${state ? `data-state=${state}: ` : ""}${(await dialog.innerText())
            .replace(/\s+/g, " ")
            .slice(0, 140)}`,
    );
    if (rendered) {
      check(
        "the rendered equation carries MathML for screen readers",
        (await dialog.locator(".katex-mathml math").count()) > 0,
      );
    }

    const focusInside = () =>
      page.evaluate(() => {
        const open = document.querySelector('[role="dialog"][aria-modal="true"]');
        return Boolean(open && open.contains(document.activeElement));
      });
    const where = () =>
      page.evaluate(() => {
        const el = document.activeElement;
        if (!el || el === document.body) return "body";
        const label = (el.getAttribute("aria-label") ?? el.textContent ?? "")
          .trim()
          .slice(0, 30);
        return `${el.tagName.toLowerCase()} "${label}"`;
      });

    check("focus moves into the dialog on open", await focusInside(), await where());

    // More presses than the dialog has stops, so every wrap is exercised.
    const escapes = [];
    for (const key of [...Array(6).fill("Tab"), ...Array(6).fill("Shift+Tab")]) {
      await page.keyboard.press(key);
      if (!(await focusInside())) escapes.push(`${key} → ${await where()}`);
    }
    check(
      "Tab and Shift+Tab stay inside the dialog",
      escapes.length === 0,
      escapes.slice(0, 3).join(" | "),
    );

    await dialog.locator("h2").first().click();
    const afterClick = await focusInside();
    const clickedTo = await where();
    await page.keyboard.press("Tab");
    const afterTab = await focusInside();
    check(
      "after a click on the dialog's text, focus and Tab stay inside",
      afterClick && afterTab,
      !afterClick
        ? `the click left focus on ${clickedTo}`
        : afterTab
          ? ""
          : `Tab → ${await where()}`,
    );

    await page.keyboard.press("Escape");
    const closed = await dialog
      .waitFor({ state: "detached", timeout: 3000 })
      .then(() => true, () => false);
    check("Escape closes the dialog", closed);
    if (!closed) return;
    // Focus is handed back in an effect after the close commits.
    const returned = await page
      .waitForFunction((el) => el === document.activeElement, trigger, {
        timeout: 2000,
      })
      .then(() => true, () => false);
    check("focus returns to the Math button", returned, returned ? "" : await where());
    check(
      "Math button reports the dialog as collapsed",
      (await trigger.getAttribute("aria-expanded")) === "false",
    );
  } finally {
    await trigger.dispose();
  }
}
