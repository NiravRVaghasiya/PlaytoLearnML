import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GameErrorBoundary } from "./GameMount";

/**
 * The per-game crash boundary. Before it existed, any render/effect error in a
 * game — or a game chunk that failed to download — replaced the whole page
 * with Next's unbranded fallback and no link back to the roster.
 */

let consoleError: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  // React logs caught render errors itself; the boundary logs them too. Both
  // are captured here so the test output stays readable, and asserted below.
  consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  consoleError.mockRestore();
});

/** A game that throws `error` for as long as `state.broken` is true. */
function makeFlakyGame(error: Error) {
  const state = { broken: true };
  function FlakyGame() {
    if (state.broken) throw error;
    return <p>game is running</p>;
  }
  return { Game: FlakyGame, state };
}

describe("<GameErrorBoundary>", () => {
  it("replaces a crashed game with a named recovery panel, not a blank page", () => {
    const { Game } = makeFlakyGame(new Error("WebGL context lost"));
    render(
      <GameErrorBoundary slug="sort-it-arcade">
        <Game />
      </GameErrorBoundary>,
    );

    // Named after the game, in a landmark, with a way home.
    const heading = screen.getByRole("heading", {
      level: 1,
      name: "Sort-It Arcade hit an unexpected error",
    });
    expect(screen.getByRole("main")).toContainElement(heading);
    expect(screen.getByRole("link", { name: /back to all games/i })).toHaveAttribute(
      "href",
      "/",
    );
    expect(screen.getByRole("button", { name: /try again/i })).toBeInTheDocument();
    // Focus moves to the explanation, since the control in use just vanished.
    expect(heading).toHaveFocus();
  });

  it("logs the error instead of swallowing it", () => {
    const error = new Error("tensor shape mismatch");
    const { Game } = makeFlakyGame(error);
    render(
      <GameErrorBoundary slug="neuron-forge">
        <Game />
      </GameErrorBoundary>,
    );

    const ours = consoleError.mock.calls.find(
      (call: unknown[]) =>
        typeof call[0] === "string" && call[0].includes('"neuron-forge" crashed'),
    );
    expect(ours, "the boundary did not console.error the crash").toBeDefined();
    expect(ours).toContain(error);
  });

  it("'Try again' remounts the game, so a transient crash recovers", async () => {
    const user = userEvent.setup();
    const { Game, state } = makeFlakyGame(new Error("first frame failed"));
    render(
      <GameErrorBoundary slug="data-detox">
        <Game />
      </GameErrorBoundary>,
    );
    expect(screen.queryByText("game is running")).not.toBeInTheDocument();

    // Whatever broke has cleared (a lost WebGL context restored, say).
    state.broken = false;
    await user.click(screen.getByRole("button", { name: /try again/i }));

    expect(screen.getByText("game is running")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /try again/i })).not.toBeInTheDocument();
  });

  it("recognises a failed chunk download and offers a reload rather than a futile retry", () => {
    const chunkError = new Error(
      "Failed to load chunk /_next/static/chunks/02me8g73r4htw.js from module 1234",
    );
    const { Game } = makeFlakyGame(chunkError);
    render(
      <GameErrorBoundary slug="gradient-descent-skier">
        <Game />
      </GameErrorBoundary>,
    );

    expect(
      screen.getByRole("heading", {
        name: "Gradient Descent Skier didn't finish downloading",
      }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /reload the page/i })).toBeInTheDocument();
    // React.lazy caches the rejected import, so a remount can never succeed.
    expect(screen.queryByRole("button", { name: /try again/i })).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: /back to all games/i })).toBeInTheDocument();
  });
});
