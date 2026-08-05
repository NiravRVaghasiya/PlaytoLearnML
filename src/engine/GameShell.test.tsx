import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { GameShell, type GameShellProps } from "./GameShell";

/**
 * Structural tests for the shared frame.
 *
 * These assert the DESIGN.md §5 anatomy is actually on screen — every game
 * inherits whatever is verified here, so a regression in the shell is a
 * regression in all fourteen games at once.
 */
function renderShell(overrides: Partial<GameShellProps> = {}) {
  const props: GameShellProps = {
    slug: "sort-it-arcade",
    title: "Sort-It Arcade",
    metric: {
      label: "Accuracy",
      value: 0.84,
      format: "percent",
      goodDirection: "up",
    },
    math: {
      equation: "y = wx + b",
      code: "const y = w * x + b;",
    },
    controls: <button type="button">Reset boundary</button>,
    visual: <div>visual lane canvas</div>,
    code: <div>code lane editor</div>,
    ...overrides,
  };

  return { user: userEvent.setup(), ...render(<GameShell {...props} />), props };
}

describe("<GameShell> anatomy", () => {
  it("renders the title, lane toggle and math button in the top bar", () => {
    renderShell();

    expect(
      screen.getByRole("heading", { name: "Sort-It Arcade" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /back/i })).toBeInTheDocument();
    expect(screen.getByRole("radiogroup", { name: "Lane" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /math/i })).toBeInTheDocument();
  });

  it("shows the visual lane by default and the controls rail", () => {
    renderShell();

    expect(screen.getByText("visual lane canvas")).toBeInTheDocument();
    expect(screen.queryByText("code lane editor")).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Reset boundary" }),
    ).toBeInTheDocument();
  });

  it("keeps the metric visible with its live region", () => {
    renderShell();

    expect(screen.getByText("Accuracy")).toBeInTheDocument();
    expect(screen.getByTestId("metric-value")).toHaveTextContent("84%");
    expect(document.querySelector('[aria-live="polite"]')).toBeInTheDocument();
  });

  it("exposes the canvas as the skip-link target", () => {
    const { container } = renderShell();

    const canvas = container.querySelector("#game-canvas");
    expect(canvas).not.toBeNull();
    // The layout's "Skip to game" link needs somewhere focusable to land.
    expect(canvas).toHaveAttribute("tabindex", "-1");
  });

  it("announces only the primary metric, not the secondary ones", () => {
    renderShell({
      secondaryMetrics: [
        { label: "Complexity", value: 7, format: "integer", goodDirection: "down" },
      ],
    });

    expect(screen.getByText("Complexity")).toBeInTheDocument();
    // One live region, not three.
    expect(document.querySelectorAll('[aria-live="polite"]')).toHaveLength(1);
  });
});

describe("<GameShell> lane switching", () => {
  it("swaps lanes on click and only mounts the active one", async () => {
    const { user } = renderShell();

    await user.click(screen.getByRole("radio", { name: /code/i }));

    expect(screen.getByText("code lane editor")).toBeInTheDocument();
    expect(screen.queryByText("visual lane canvas")).not.toBeInTheDocument();
  });

  it("supports arrow-key navigation between lanes", async () => {
    const { user } = renderShell();

    await user.tab();
    await user.tab();
    expect(screen.getByRole("radio", { name: /visual/i })).toHaveFocus();

    await user.keyboard("{ArrowRight}");
    expect(screen.getByText("code lane editor")).toBeInTheDocument();
  });

  it("honours a controlled lane", async () => {
    const onLaneChange = vi.fn();
    const { user } = renderShell({ lane: "code", onLaneChange });

    expect(screen.getByText("code lane editor")).toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: /visual/i }));
    expect(onLaneChange).toHaveBeenCalledWith("visual");
    // Still on code — the parent owns the state.
    expect(screen.getByText("code lane editor")).toBeInTheDocument();
  });

  it("blocks the code lane when a game says it isn't wired", async () => {
    const { user } = renderShell({
      codeLaneDisabled: true,
      codeLaneDisabledReason: "Code lane lands in the next PR",
    });

    const codeRadio = screen.getByRole("radio", { name: /code/i });
    expect(codeRadio).toHaveAttribute("aria-disabled", "true");

    await user.click(codeRadio);
    expect(screen.getByText("visual lane canvas")).toBeInTheDocument();
  });
});

describe("<GameShell> named failure (pedagogy contract #4)", () => {
  it("names the failure, shows the numbers, and offers one-click retry", async () => {
    const onRetry = vi.fn();
    const { user } = renderShell({
      failure: { name: "Overfitting", detail: "99% train, 61% test" },
      onRetry,
    });

    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("Overfitting");
    expect(alert).toHaveTextContent("99% train, 61% test");

    // Retry is inside the alert — no dead-end Game Over screen to dismiss first.
    await user.click(
      screen.getAllByRole("button", { name: /retry/i })[0]!,
    );
    expect(onRetry).toHaveBeenCalled();
  });

  it("turns the metric red even when the game forgot to set state", () => {
    renderShell({
      failure: { name: "Divergence", detail: "loss = Infinity" },
    });

    expect(screen.getByTestId("metric-value").className).toContain(
      "text-wrong",
    );
  });

  it("shows no alert when there is no failure", () => {
    renderShell();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});

describe("<GameShell> WhyCard dock", () => {
  it("prompts before the first action", () => {
    renderShell();
    expect(screen.getByText(/make a move/i)).toBeInTheDocument();
  });

  it("renders the explanation for the last action", () => {
    renderShell({
      whyCard: {
        key: "wiggle-1",
        title: "That boundary is memorising",
        body: "You gained 3% accuracy but doubled the complexity cost.",
        tone: "warn",
      },
    });

    expect(
      screen.getByText("That boundary is memorising"),
    ).toBeInTheDocument();
    expect(screen.getByText(/doubled the complexity cost/)).toBeInTheDocument();
  });
});

describe("<GameShell> progression footer", () => {
  it("renders XP and stars when progress is supplied", () => {
    renderShell({
      progress: {
        level: 3,
        xpIntoLevel: 140,
        xpForNextLevel: 400,
        stars: 2,
        starCriteria: ["Finish", "Score 80%", "Clear the code lane"],
      },
    });

    const bar = screen.getByRole("progressbar");
    expect(bar).toHaveAttribute("aria-valuenow", "140");
    expect(bar).toHaveAttribute("aria-valuemax", "400");
    // A name says what is measured; valuetext says where you are in it.
    expect(bar).toHaveAccessibleName("Experience points");
    expect(bar).toHaveAttribute("aria-valuetext", "Level 3, 140 of 400 XP");

    expect(
      screen.getByRole("img", { name: "2 of 3 mastery stars earned" }),
    ).toBeInTheDocument();
  });

  it("omits the footer widgets when there is no progress", () => {
    renderShell();
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
  });
});

describe("<GameShell> math drawer", () => {
  it("opens as a modal dialog and closes on Escape", async () => {
    const { user } = renderShell();

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /math/i }));

    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveAccessibleName(/Reveal the math: Sort-It Arcade/);
    // The real implementation is shown, not a paraphrase.
    expect(dialog).toHaveTextContent("const y = w * x + b;");

    await user.keyboard("{Escape}");
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    );
  });

  it("returns focus to the trigger after closing", async () => {
    const { user } = renderShell();
    const trigger = screen.getByRole("button", { name: /math/i });

    await user.click(trigger);
    await user.keyboard("{Escape}");

    await waitFor(() => expect(trigger).toHaveFocus());
  });
});
