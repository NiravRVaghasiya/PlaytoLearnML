import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import {
  NOT_FOUND_PATHS,
  checkMathDialog,
  classifyConsoleMessage,
  playableSlugsFromSource,
  routeFor,
  siteRoutes,
} from "../../scripts/harness.mjs";

/**
 * `scripts/harness.mjs` decides what the browser harnesses visit and which
 * console lines fail them. Both decisions are easy to get quietly wrong — a
 * route that drops out of the list is never checked again, and a noise filter
 * that is too broad hides real bugs — so they are pinned here. (Whether the
 * route parse matches the real registry and library is asserted next to those
 * modules, in registry.test.ts and concepts.test.ts.)
 */

describe("siteRoutes", () => {
  it("covers home, every game, the library, and the 404s, each with its expected status", () => {
    const routes = siteRoutes();
    expect(routes[0]).toEqual({ path: "/", kind: "page", status: 200 });
    expect(routes.filter((r) => r.kind === "game").length).toBeGreaterThan(0);
    for (const path of NOT_FOUND_PATHS) {
      expect(routes).toContainEqual({ path, kind: "not-found", status: 404 });
    }
    const paths = routes.map((r) => r.path);
    expect(new Set(paths).size, "duplicate route").toBe(paths.length);
  });

  it("recognises one of its routes from a full URL", () => {
    expect(routeFor("http://localhost:3000/play/does-not-exist")?.kind).toBe(
      "not-found",
    );
    expect(routeFor("http://localhost:3000/play/sort-it-arcade")?.kind).toBe("game");
    expect(routeFor("/nowhere")).toBeUndefined();
  });

  it("parses the registry without being fooled by other string arrays", () => {
    const source = `
      const OTHER = ["not-a-game"];
      export const PLAYABLE_SLUGS = [
        "alpha-game",
        'beta',
      ] as const;`;
    expect(playableSlugsFromSource(source)).toEqual(["alpha-game", "beta"]);
    expect(() => playableSlugsFromSource("export const X = 1;")).toThrow(
      /no PLAYABLE_SLUGS/,
    );
  });
});

describe("classifyConsoleMessage", () => {
  it("fails on any ordinary console error", () => {
    expect(classifyConsoleMessage({ type: "error", text: "TypeError: x is undefined" })).toBe(
      "fail",
    );
  });

  it("fails on warnings that mean a real bug: KaTeX and React", () => {
    for (const text of [
      "LaTeX-incompatible input and strict mode is set to 'warn': In LaTeX, \\\\ or \\newline does nothing in display mode [newLineInDisplayMode]",
      "No character metrics for 'ƒ' in style 'Main-Regular' and mode 'math'",
      "Warning: Each child in a list should have a unique \"key\" prop.",
      "A tree hydrated but some attributes of the server rendered HTML didn't match the client properties.",
      "Maximum update depth exceeded.",
    ]) {
      expect(classifyConsoleMessage({ type: "warning", text }), text).toBe("fail");
    }
  });

  it("ignores headless-GPU chatter and the TF.js WebGL-to-CPU fallback", () => {
    for (const text of [
      "[GroupMarkerNotSet(crbug.com/242999)!:A80019001C590000]Automatic fallback to software WebGL has been deprecated. Please use the --enable-unsafe-swiftshader (about:flags#enable-unsafe-swiftshader) flag",
      "[.WebGL-0x73c4006fe900]GL Driver Message (OpenGL, Performance, GL_CLOSE_PATH_NV, High): GPU stall due to ReadPixels",
      "Initialization of backend webgl failed",
      "Error: WebGL is not supported on this device\n    at new Ix (http://localhost:3200/_next/static/chunks/3av8lr41mc7lc.js:434:441)",
    ]) {
      expect(classifyConsoleMessage({ type: "warning", text }), text).toBe("ignore");
    }
  });

  it("reports an unclassified warning without failing on it", () => {
    expect(
      classifyConsoleMessage({ type: "warning", text: "THREE.WebGLRenderer: something new" }),
    ).toBe("warn");
  });

  it("ignores the 404 page's own 404, and only that one", () => {
    const pageUrl = "http://localhost:3000/play/does-not-exist";
    const text = "Failed to load resource: the server responded with a status of 404 (Not Found)";
    expect(
      classifyConsoleMessage(
        { type: "error", text, resourceUrl: pageUrl },
        { pageUrl, expectNotFound: true },
      ),
    ).toBe("ignore");
    // A missing asset on the 404 page is still a bug...
    expect(
      classifyConsoleMessage(
        { type: "error", text, resourceUrl: "http://localhost:3000/icon.svg" },
        { pageUrl, expectNotFound: true },
      ),
    ).toBe("fail");
    // ...and so is any 404 on a page that should exist.
    expect(
      classifyConsoleMessage(
        { type: "error", text, resourceUrl: "http://localhost:3000/play/sort-it-arcade" },
        { pageUrl: "http://localhost:3000/play/sort-it-arcade" },
      ),
    ).toBe("fail");
  });

  it("does not treat log or info output as a problem", () => {
    expect(classifyConsoleMessage({ type: "log", text: "Warning: this is a log" })).toBe(
      "ignore",
    );
  });
});

describe("checkMathDialog", () => {
  it("is run on every game by verify-playthrough, after the game's own checks", () => {
    const runner = readFileSync(
      join(process.cwd(), "scripts", "verify-playthrough.mjs"),
      "utf8",
    );
    expect(runner).toMatch(/import \{[^}]*\bcheckMathDialog\b[^}]*\} from "\.\/harness\.mjs"/);
    const ownChecks = runner.indexOf("await playthrough.run(");
    const shared = runner.indexOf("await checkMathDialog(page, check)");
    expect(ownChecks).toBeGreaterThan(-1);
    expect(shared, "checkMathDialog is never called").toBeGreaterThan(ownChecks);
  });

  it("fails, rather than skips, a game with no Math button", async () => {
    // A stand-in page with no matching button. The check must record a failure;
    // a silent return would let a game lose its Math dialog unnoticed.
    const page = {
      getByRole: () => ({ count: async () => 0 }),
    } as unknown as Parameters<typeof checkMathDialog>[0];
    const results: { name: string; passed: boolean }[] = [];
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    try {
      await checkMathDialog(page, (name, passed) => results.push({ name, passed }));
    } finally {
      log.mockRestore();
    }
    expect(results).toEqual([
      { name: "one Math button in the game header", passed: false },
    ]);
  });
});
