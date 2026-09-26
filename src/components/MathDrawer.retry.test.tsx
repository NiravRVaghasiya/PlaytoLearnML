import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { MathDrawer } from "./MathDrawer";

/**
 * Its own file because it replaces the KaTeX chunk for every test in it: the
 * first import fails the way a dropped connection does, and later ones work.
 */
const chunk = vi.hoisted(() => ({ attempts: 0 }));
vi.mock("./katexRender", async (importOriginal) => {
  chunk.attempts += 1;
  if (chunk.attempts === 1) {
    throw new TypeError("Failed to fetch dynamically imported module");
  }
  return importOriginal();
});

const MATH = { equation: "y = wx + b", code: "const y = w * x + b;" };

describe("<MathDrawer> after KaTeX failed to download", () => {
  it("shows the source meanwhile, and retries on the next open", async () => {
    const drawer = (open: boolean) => (
      <MathDrawer open={open} title="Test Game" math={MATH} onClose={() => {}} />
    );
    const { rerender } = render(drawer(true));

    // First open: the chunk didn't arrive, so the LaTeX source stands in.
    const region = () => screen.getByTestId("math-equation");
    await waitFor(() => expect(region()).toHaveAttribute("data-state", "source"));
    expect(region()).toHaveTextContent("y = wx + b");
    expect(region().querySelector(".katex")).toBeNull();
    expect(chunk.attempts).toBe(1);

    // Close and reopen once the connection is back: it tries again, rather
    // than showing raw LaTeX for the rest of the visit.
    rerender(drawer(false));
    rerender(drawer(true));
    await waitFor(() => expect(region()).toHaveAttribute("data-state", "rendered"));
    expect(region().querySelector(".katex math")).not.toBeNull();
    expect(chunk.attempts).toBe(2);
  });
});
