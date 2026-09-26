import { render, screen, waitFor, within } from "@testing-library/react";
import { renderToString } from "react-dom/server";
import { afterEach, describe, expect, it } from "vitest";
import {
  EMPTY_PROGRESSION,
  createLocalAdapter,
  createMemoryAdapter,
  useProgression,
  type ProgressionState,
} from "@/engine/progression";
import { CATEGORIES, GAME_CATALOG, START_SLUG, getGameMeta } from "@/lib/catalog";
import Home from "../page";

/**
 * The home roster is a first-time visitor's whole introduction, so these pin
 * what it must do for one: say where to start, explain the loop, group games
 * in learner terms rather than build phases, and show progress without ever
 * rendering it on the server (where it can't be known, so it would mismatch).
 */

/** The roster card for a game (not the "Start here" link in the header). */
function rosterCard(slug: string): HTMLElement {
  const roster = screen.getByRole("region", { name: /All \d+ games/ });
  const card = within(roster)
    .getAllByRole("link")
    .find((link) => link.getAttribute("href") === `/play/${slug}`);
  if (!card) throw new Error(`no roster card for ${slug}`);
  return card;
}

function resetProgression(adapterSeed?: ProgressionState) {
  useProgression.setState({ ...EMPTY_PROGRESSION, hydrated: false });
  useProgression
    .getState()
    .setAdapter(adapterSeed ? createMemoryAdapter(adapterSeed) : createLocalAdapter());
}

afterEach(() => {
  resetProgression();
});

const clearedSortIt: ProgressionState = {
  xp: 150,
  badges: [],
  games: {
    "sort-it-arcade": {
      slug: "sort-it-arcade",
      bestScore: 0.9,
      stars: 2,
      completed: true,
      codeLaneCleared: false,
      playCount: 1,
      xpAwarded: 90,
      updatedAt: "2026-01-01T00:00:00.000Z",
    },
  },
};

describe("home page", () => {
  it("gives every game in the catalog exactly one roster card, and it is a link", () => {
    resetProgression(EMPTY_PROGRESSION);
    render(<Home />);
    const roster = screen.getByRole("region", { name: /All \d+ games/ });
    const hrefs = within(roster)
      .getAllByRole("link")
      .map((link) => link.getAttribute("href"));
    expect([...hrefs].sort()).toEqual(
      GAME_CATALOG.map((game) => `/play/${game.slug}`).sort(),
    );
    for (const game of GAME_CATALOG) {
      expect(rosterCard(game.slug)).toHaveTextContent(game.title);
    }
  });

  it("tells a first-time visitor where to start", () => {
    resetProgression(EMPTY_PROGRESSION);
    render(<Home />);
    const start = getGameMeta(START_SLUG)!;
    const cta = screen.getByRole("link", { name: `Start here: ${start.title}` });
    expect(cta).toHaveAttribute("href", `/play/${START_SLUG}`);
  });

  it("explains the loop in three steps: play, watch the metric, switch to code", () => {
    resetProgression(EMPTY_PROGRESSION);
    render(<Home />);
    const how = screen.getByRole("region", { name: "How it works" });
    const steps = within(how).getAllByRole("listitem");
    expect(steps).toHaveLength(3);
    expect(steps[0]).toHaveTextContent(/Visual lane/);
    expect(steps[1]).toHaveTextContent(/live metric/i);
    expect(steps[2]).toHaveTextContent(/Code lane/);
    expect(how).toHaveTextContent(/runs in your browser/);
  });

  it("groups games by category in spec order, not by internal build phase", () => {
    resetProgression(EMPTY_PROGRESSION);
    const { container } = render(<Home />);
    expect(container).not.toHaveTextContent(/Phase \d/);
    expect(container).not.toHaveTextContent(/MVP/);

    const roster = screen.getByRole("region", { name: /All \d+ games/ });
    const headings = within(roster)
      .getAllByRole("heading", { level: 3 })
      .map((h) => h.textContent);
    expect(headings).toEqual(
      CATEGORIES.filter((c) => GAME_CATALOG.some((g) => g.category === c)),
    );
    // Difficulty is on the card, so "easiest first" is visible, not implied.
    expect(rosterCard("backprop-blitz")).toHaveTextContent("Advanced");
  });

  it("shows stars on a cleared game once saved progress has loaded", async () => {
    resetProgression(clearedSortIt);
    render(<Home />);

    const card = rosterCard("sort-it-arcade");
    await waitFor(() =>
      expect(
        within(card).getByRole("img", { name: "2 of 3 mastery stars earned" }),
      ).toBeInTheDocument(),
    );
    expect(card).toHaveTextContent("Cleared");

    // A game the player hasn't cleared shows no progress at all.
    const other = rosterCard("data-detox");
    expect(within(other).queryByRole("img")).not.toBeInTheDocument();
    expect(other).not.toHaveTextContent("Cleared");
  });

  it("never renders progress on the server, so hydration cannot mismatch", () => {
    // Even with the store already holding progress, the server HTML must be
    // the empty state — it is what the client's first render has to match.
    useProgression.setState({ ...clearedSortIt, hydrated: true });
    const html = renderToString(<Home />);
    expect(html).not.toContain("mastery stars");
    expect(html).not.toContain("Cleared");
    // ...while the rest of the page is fully there.
    expect(html).toContain('href="/play/sort-it-arcade"');
  });
});
