/// <reference types="vite/client" />
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import katex from "katex";
import { describe, expect, it, vi } from "vitest";
import { PLAYABLE_SLUGS } from "@/games/registry";
import { MathDrawer } from "./MathDrawer";
import { EQUATION_OPTIONS, renderEquation, wrapEquation } from "./katexRender";

/**
 * Every game's real `MATH_EQUATION`, straight from its `ml.ts` — the same
 * string the game passes to the drawer. Globbed rather than listed so a new
 * game is covered the moment it exists.
 */
const mlModules = import.meta.glob("../games/*/ml.ts", {
  eager: true,
}) as Record<string, { MATH_EQUATION?: unknown }>;

const equations = Object.entries(mlModules).map(([path, module]) => ({
  slug: path.split("/").at(-2)!,
  equation: module.MATH_EQUATION,
}));

/**
 * Render with KaTeX's strict checking turned all the way up.
 *
 * `strict: "error"` on its own is not enough for this regression: KaTeX treats
 * `newLineInDisplayMode` as a "strict behaviour" code, and in "error" mode it
 * silently applies LaTeX's behaviour (drop the line break) instead of throwing.
 * A strict FUNCTION sees every code, so it records them all and still answers
 * "error" — which makes genuinely non-strict input throw, as `throwOnError`
 * demands.
 */
function strictCodes(source: string): string[] {
  const codes: string[] = [];
  katex.renderToString(source, {
    ...EQUATION_OPTIONS,
    throwOnError: true,
    strict: (errorCode: string) => {
      codes.push(errorCode);
      return "error";
    },
  });
  return codes;
}

describe("every game's equation renders cleanly", () => {
  it("covers every playable game", () => {
    expect(equations.map((e) => e.slug).sort()).toEqual([...PLAYABLE_SLUGS].sort());
  });

  it.each(equations)("$slug: renders in strict mode with real line breaks", ({ equation }) => {
    expect(typeof equation).toBe("string");
    expect((equation as string).length).toBeGreaterThan(0);
    expect(strictCodes(wrapEquation(equation as string))).toEqual([]);
    // And the runtime path produces real output, not the "" fallback.
    expect(renderEquation(equation as string)).toContain("katex");
  });

  it("is a real check: an unwrapped multi-line equation is flagged", () => {
    // Without `gathered`, a top-level \\ in display mode does nothing. If this
    // stopped being detected, the test above would pass for the wrong reason.
    const multiLine = String.raw`a = b \\[1.2em] c = d`;
    expect(strictCodes(multiLine)).toContain("newLineInDisplayMode");
    expect(strictCodes(wrapEquation(multiLine))).toEqual([]);
  });
});

const MATH = {
  equation: String.raw`y = wx + b \\ L = (y - \hat{y})^2`,
  code: "const y = w * x + b;",
};

function renderDrawer(overrides: Partial<Parameters<typeof MathDrawer>[0]> = {}) {
  const onClose = vi.fn();
  const utils = render(
    <div>
      <button type="button">Behind the dialog</button>
      <MathDrawer open title="Test Game" math={MATH} onClose={onClose} {...overrides} />
    </div>,
  );
  return { ...utils, onClose, user: userEvent.setup() };
}

describe("<MathDrawer> rendering", () => {
  it("renders nothing, and no KaTeX, while closed", () => {
    const { container } = render(
      <MathDrawer open={false} title="Test Game" math={MATH} onClose={() => {}} />,
    );
    expect(container).toBeEmptyDOMElement();
    expect(document.querySelector(".katex")).toBeNull();
  });

  it("renders the equation on open, with MathML for screen readers", async () => {
    renderDrawer();

    // The code is there immediately; the equation follows once KaTeX loads.
    expect(screen.getByRole("dialog")).toHaveTextContent("const y = w * x + b;");
    await waitFor(() => expect(document.querySelector(".katex")).not.toBeNull());
    expect(document.querySelector(".katex math")).not.toBeNull();
    // Two display lines, not one run-on line.
    expect(document.querySelectorAll(".katex-html .mtable .vlist-t").length).toBeGreaterThan(0);
  });

  it("falls back to the LaTeX source when KaTeX can't parse it", async () => {
    renderDrawer({ math: { equation: String.raw`\frac{1}{`, code: "x" } });
    const region = screen.getByRole("group", { name: "The equation" });
    await waitFor(() => expect(region).toHaveTextContent(String.raw`\frac{1}{`));
    expect(region.querySelector(".katex")).toBeNull();
  });

  it("exposes stable hooks for browser checks: the dialog, the equation, and its state", async () => {
    // scripts/verify-playthrough.mjs waits on these instead of a timeout.
    renderDrawer();
    expect(screen.getByTestId("math-dialog")).toBe(screen.getByRole("dialog"));
    const equation = screen.getByTestId("math-equation");
    expect(equation).toBe(screen.getByRole("group", { name: "The equation" }));
    expect(equation).toHaveAttribute("data-state", "loading");
    await waitFor(() => expect(equation).toHaveAttribute("data-state", "rendered"));
    expect(equation.querySelector(".katex math")).not.toBeNull();
  });

  it("marks a parse failure as showing the source", async () => {
    renderDrawer({ math: { equation: String.raw`\frac{1}{`, code: "x" } });
    await waitFor(() =>
      expect(screen.getByTestId("math-equation")).toHaveAttribute("data-state", "source"),
    );
  });

  it("makes the scrollable equation box keyboard-reachable and named", () => {
    renderDrawer();
    const region = screen.getByRole("group", { name: "The equation" });
    expect(region).toHaveAttribute("tabindex", "0");
  });
});

describe("<MathDrawer> is actually modal", () => {
  it("keeps Tab inside after a click on the dialog's text", async () => {
    // Regression: clicking non-focusable text dropped focus on the page behind,
    // and the next Tab walked out into the game.
    const { user } = renderDrawer();
    await user.click(screen.getByText("The code that actually runs"));
    expect(screen.getByRole("dialog")).toHaveFocus();

    await user.tab();
    expect(screen.getByRole("dialog")).toContainElement(
      document.activeElement as HTMLElement,
    );
    expect(screen.getByRole("button", { name: "Behind the dialog" })).not.toHaveFocus();

    await user.tab({ shift: true });
    expect(screen.getByRole("dialog")).toContainElement(
      document.activeElement as HTMLElement,
    );
  });

  it("wraps Tab from the last control back to the first", async () => {
    const { user } = renderDrawer();
    const dialog = screen.getByRole("dialog");
    const focusables = dialog.querySelectorAll<HTMLElement>(
      'button, [tabindex]:not([tabindex="-1"])',
    );
    focusables[focusables.length - 1]!.focus();

    await user.tab();
    expect(focusables[0]).toHaveFocus();
  });

  it("stops page-level keyboard shortcuts from reaching the game behind it", async () => {
    const shortcut = vi.fn();
    document.addEventListener("keydown", shortcut);
    try {
      const { user, onClose } = renderDrawer();
      await user.keyboard("2");
      expect(shortcut).not.toHaveBeenCalled();

      // Escape still closes it.
      await user.keyboard("{Escape}");
      expect(onClose).toHaveBeenCalledOnce();
    } finally {
      document.removeEventListener("keydown", shortcut);
    }
  });

  it("lets page shortcuts work again once closed", async () => {
    const shortcut = vi.fn();
    document.addEventListener("keydown", shortcut);
    try {
      const { user, rerender } = renderDrawer();
      rerender(
        <div>
          <button type="button">Behind the dialog</button>
          <MathDrawer open={false} title="Test Game" math={MATH} onClose={() => {}} />
        </div>,
      );
      await user.keyboard("2");
      expect(shortcut).toHaveBeenCalled();
    } finally {
      document.removeEventListener("keydown", shortcut);
    }
  });
});
