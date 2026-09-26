/**
 * Phone-layout harness.
 *
 * Loads every route on the site (see `siteRoutes()` in harness.mjs) as a small
 * Android phone — 360×740 CSS px at 2× density, touch, mobile viewport — and
 * fails on the things that make a page unusable on one:
 *
 *   - horizontal overflow: `documentElement.scrollWidth > clientWidth`, i.e. the
 *     page scrolls sideways;
 *   - a forced-wide layout viewport: `window.innerWidth > 360`. On a mobile
 *     viewport, content wider than the screen does not always scroll — Chrome
 *     can instead zoom the whole page out to fit it, which widens the layout
 *     viewport and shrinks every control. The scrollWidth check alone misses
 *     that case (it measured 621/621 on one game), so both are checked;
 *   - console errors, uncaught page errors, and bug-shaped warnings (KaTeX,
 *     React) — classified by harness.mjs, which ignores headless-GPU noise and
 *     the TF.js WebGL-to-CPU fallback, and the 404 route's own 404;
 *   - the wrong HTTP status (every page 200, the not-found routes 404);
 *   - no `<main>` landmark or no `<h1>`;
 *   - on a `/play/*` route, no primary live metric on screen (pedagogy contract
 *     #3: the metric never hides).
 *
 * Targets smaller than 24×24 CSS px (WCAG 2.2 SC 2.5.8) are REPORTED as
 * warnings, not failures: the SC has spacing and inline-text exceptions a
 * bounding box cannot judge, so each one needs a human look.
 *
 * Usage:
 *   node scripts/verify-mobile.mjs                 # every route
 *   node scripts/verify-mobile.mjs /play/data-detox /concepts
 *   AUDIT_BASE=http://localhost:3200 node scripts/verify-mobile.mjs
 *
 * Needs a running server (`bun run build && bun run start`). Exits non-zero on
 * any failure.
 */

import { chromium } from "playwright";
import { classifyConsoleMessage, routeFor, siteRoutes } from "./harness.mjs";

const BASE = process.env.AUDIT_BASE ?? "http://localhost:3000";
const WIDTH = 360;
const HEIGHT = 740;
/** WCAG 2.2 SC 2.5.8, Target Size (Minimum). */
const MIN_TARGET = 24;

const requested = process.argv.slice(2);
const routes =
  requested.length > 0
    ? requested.map(
        (arg) =>
          routeFor(arg) ?? {
            path: new URL(arg, BASE).pathname,
            kind: "page",
            status: 200,
          },
      )
    : siteRoutes();

const browser = await chromium.launch();
const context = await browser.newContext({
  viewport: { width: WIDTH, height: HEIGHT },
  deviceScaleFactor: 2,
  isMobile: true,
  hasTouch: true,
});

let failedRoutes = 0;
let warningCount = 0;

try {
  for (const route of routes) {
    const url = `${BASE}${route.path}`;
    const expectNotFound = route.kind === "not-found";
    const failures = [];
    const warnings = [];

    const page = await context.newPage();
    page.on("console", (message) => {
      const verdict = classifyConsoleMessage(
        {
          type: message.type(),
          text: message.text(),
          resourceUrl: message.location()?.url,
        },
        { pageUrl: url, expectNotFound },
      );
      const line = `console ${message.type()}: ${message.text().slice(0, 200)}`;
      if (verdict === "fail") failures.push(line);
      else if (verdict === "warn") warnings.push(line);
    });
    page.on("pageerror", (error) =>
      failures.push(`pageerror: ${error.message.slice(0, 200)}`),
    );

    try {
      const response = await page.goto(url, { waitUntil: "load" });
      const status = response?.status() ?? 0;
      if (status !== route.status) {
        failures.push(`HTTP ${status}, expected ${route.status}`);
      }

      if (route.kind === "game") {
        // Games mount client-side (`ssr: false`), so the metric appears only
        // once the game's chunk has loaded. Waiting for it IS the check.
        const metric = page.locator('[data-testid="metric-value"]').first();
        try {
          await metric.waitFor({ state: "visible", timeout: 25_000 });
        } catch {
          failures.push("no primary metric became visible within 25s");
        }
        // Let 3D scenes and TF.js backends settle before measuring the layout
        // they may have widened.
        await page.waitForTimeout(1_500);
      } else {
        await page.waitForTimeout(500);
      }

      const layout = await page.evaluate(
        ({ width, minTarget }) => {
          const doc = document.documentElement;

          /** Something visibly wider than the viewport, for the report. */
          const wide = [...document.querySelectorAll("body *")]
            .filter((el) => el.getBoundingClientRect().right > window.innerWidth + 1)
            .filter((el) => !el.closest(".sr-only, .sr-only-live"))
            .slice(0, 4)
            .map((el) => {
              const cls = String(el.className?.baseVal ?? el.className ?? "").slice(0, 50);
              return `${el.tagName.toLowerCase()}${el.id ? `#${el.id}` : ""}${cls ? `.${cls}` : ""} right=${Math.round(el.getBoundingClientRect().right)}`;
            });

          /**
           * The area a pointer can actually hit. A checkbox or radio is also
           * activated by clicking its <label>, so a 13px native box inside a
           * full-width label is not a 13px target — measuring the input alone
           * reported every labelled checkbox on the site as a miss.
           */
          const hitBox = (el) => {
            const boxes = [el, ...(el.labels ?? [])].map((node) =>
              node.getBoundingClientRect(),
            );
            const left = Math.min(...boxes.map((b) => b.left));
            const right = Math.max(...boxes.map((b) => b.right));
            const top = Math.min(...boxes.map((b) => b.top));
            const bottom = Math.max(...boxes.map((b) => b.bottom));
            return { width: right - left, height: bottom - top };
          };

          const small = [
            ...document.querySelectorAll(
              "a[href], button, input, select, textarea, [role=button], [role=slider], [role=radio], [role=checkbox], [tabindex='0']",
            ),
          ]
            .filter((el) => {
              const own = el.getBoundingClientRect();
              // Visually-hidden (the skip link before it is focused) and
              // not-rendered elements are not targets anyone can miss.
              if (own.width <= 1 || own.height <= 1) return false;
              if (el.closest(".sr-only, .sr-only-live, [hidden], [inert]")) return false;
              const box = hitBox(el);
              return box.width < minTarget || box.height < minTarget;
            })
            .map((el) => {
              const box = hitBox(el);
              const name = (
                el.getAttribute("aria-label") ??
                el.textContent ??
                el.getAttribute("name") ??
                ""
              )
                .trim()
                .replace(/\s+/g, " ")
                .slice(0, 40);
              return `${el.tagName.toLowerCase()} "${name}" ${Math.round(box.width)}×${Math.round(box.height)}`;
            });

          return {
            scrollWidth: doc.scrollWidth,
            clientWidth: doc.clientWidth,
            innerWidth: window.innerWidth,
            expectedWidth: width,
            hasMain: document.querySelector("main, [role=main]") !== null,
            h1Count: document.querySelectorAll("h1").length,
            wide,
            small,
          };
        },
        { width: WIDTH, minTarget: MIN_TARGET },
      );

      if (layout.scrollWidth > layout.clientWidth) {
        failures.push(
          `horizontal overflow: scrollWidth ${layout.scrollWidth} > clientWidth ${layout.clientWidth}`,
        );
      }
      if (layout.innerWidth > layout.expectedWidth) {
        failures.push(
          `layout viewport forced to ${layout.innerWidth}px on a ${layout.expectedWidth}px screen (content wider than the phone zoomed the page out)`,
        );
      }
      if ((layout.scrollWidth > layout.clientWidth || layout.innerWidth > WIDTH) && layout.wide.length > 0) {
        failures.push(`  widest elements: ${layout.wide.join(" | ")}`);
      }
      if (!layout.hasMain) failures.push("no <main> landmark");
      if (layout.h1Count === 0) failures.push("no <h1>");

      for (const target of layout.small) {
        warnings.push(`target under ${MIN_TARGET}px: ${target}`);
      }
    } catch (error) {
      failures.push(`harness error: ${error.message}`);
    } finally {
      await page.close();
    }

    const ok = failures.length === 0;
    if (!ok) failedRoutes += 1;
    warningCount += warnings.length;

    console.log(`\n${ok ? "PASS" : "FAIL"}  ${route.path}`);
    for (const failure of failures) console.log(`  FAIL  ${failure}`);
    for (const warning of warnings.slice(0, 6)) console.log(`  WARN  ${warning}`);
    if (warnings.length > 6) {
      console.log(`  WARN  …and ${warnings.length - 6} more`);
    }
  }
} finally {
  await browser.close();
}

console.log(
  `\n${routes.length - failedRoutes}/${routes.length} routes passed at ${WIDTH}×${HEIGHT}` +
    (warningCount > 0 ? ` (${warningCount} warning(s) to review)` : ""),
);
process.exit(failedRoutes === 0 ? 0 : 1);
