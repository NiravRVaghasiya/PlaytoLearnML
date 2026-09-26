import { readFileSync } from "node:fs";
import { join } from "node:path";
import { useState } from "react";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { GameShell, type GameShellProps } from "./GameShell";
import { EMPTY_PROGRESSION, useProgression } from "./progression";
import type { NamedFailure } from "./types";
import { useCodeLane } from "./useCodeLane";

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
    // One metric live region, not three: the primary announces, the
    // secondaries stay silent. (The shell's WhyCard announcer is a separate,
    // deliberate channel, so this counts only inside the metric regions.)
    const primary = screen.getByRole("region", { name: "Live metric" });
    const secondary = screen.getByRole("region", { name: "More metrics" });
    expect(primary.querySelectorAll("[aria-live]")).toHaveLength(1);
    expect(secondary.querySelectorAll("[aria-live]")).toHaveLength(0);
    expect(secondary).toHaveTextContent("Complexity");
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

    // APG radio pattern: focus moves WITH the selection. Leaving it behind on
    // the now-unchecked radio meant a screen reader said "Visual, not checked"
    // and never announced the lane that was picked.
    const codeRadio = screen.getByRole("radio", { name: /code/i });
    expect(codeRadio).toHaveFocus();
    expect(codeRadio).toHaveAttribute("aria-checked", "true");

    await user.keyboard("{ArrowLeft}");
    expect(screen.getByRole("radio", { name: /visual/i })).toHaveFocus();
  });

  it("keeps the player's code-lane edits across a switch to the visual lane and back", async () => {
    // Only the active lane is mounted, so this is the case that used to throw
    // the player's edits away.
    function ScriptLane() {
      const lane = useCodeLane({ initialCode: "api.step();", api: {} });
      return (
        <textarea
          aria-label="Script"
          value={lane.code}
          onChange={(event) => lane.setCode(event.target.value)}
        />
      );
    }
    const { user } = renderShell({ code: <ScriptLane /> });

    await user.click(screen.getByRole("radio", { name: /code/i }));
    await user.clear(screen.getByRole("textbox", { name: "Script" }));
    await user.type(screen.getByRole("textbox", { name: "Script" }), "api.setK(3);");

    await user.click(screen.getByRole("radio", { name: /visual/i }));
    expect(screen.queryByRole("textbox", { name: "Script" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("radio", { name: /code/i }));

    expect(screen.getByRole("textbox", { name: "Script" })).toHaveValue("api.setK(3);");
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

    // Announced assertively, name and numbers together.
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("Overfitting");
    expect(alert).toHaveTextContent("99% train, 61% test");

    // And shown, with Retry right in the strip — no dead-end Game Over screen
    // to dismiss first.
    const strip = screen.getByTestId("named-failure");
    expect(strip).toHaveTextContent("Overfitting");
    expect(strip).toHaveTextContent("99% train, 61% test");
    await user.click(within(strip).getByRole("button", { name: /retry/i }));
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

  it("shows and announces nothing when there is no failure", () => {
    renderShell();
    expect(screen.queryByTestId("named-failure")).not.toBeInTheDocument();
    // The alert region is persistent (so its content changes are announced
    // reliably) and empty.
    expect(screen.getByRole("alert")).toBeEmptyDOMElement();
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

describe("<GameShell> named-failure announcements", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  const OVERFIT = { name: "Overfitting", detail: "99% train, 61% test" };

  it("keeps one persistent alert region instead of remounting it with the strip", () => {
    const { rerender, props } = renderShell({ failure: OVERFIT });
    const region = screen.getByRole("alert");

    // A game re-rendering the same failure (a new object with the same words)
    // must not produce a new announcement.
    rerender(<GameShell {...props} failure={{ ...OVERFIT }} />);
    expect(screen.getByRole("alert")).toBe(region);
    expect(region).toHaveTextContent("Overfitting. 99% train, 61% test");
  });

  it("does not re-announce the same failure when only its live numbers move", () => {
    const { rerender, props } = renderShell({ failure: OVERFIT });
    const region = screen.getByRole("alert");

    // Convolution Kitchen's detail quotes per-class accuracy, which changes on
    // nearly every edit while the diagnosis does not.
    rerender(
      <GameShell {...props} failure={{ ...OVERFIT, detail: "98% train, 63% test" }} />,
    );
    expect(region).toHaveTextContent("Overfitting. 99% train, 61% test");
    // The visible strip still shows the current numbers.
    expect(screen.getByTestId("named-failure")).toHaveTextContent("98% train, 63% test");
  });

  it("announces a different failure at once, without it clearing first", () => {
    const { rerender, props } = renderShell({ failure: OVERFIT });
    const region = screen.getByRole("alert");

    rerender(
      <GameShell {...props} failure={{ name: "Underfitting", detail: "58% train, 57% test" }} />,
    );
    expect(region).toHaveTextContent("Underfitting. 58% train, 57% test");
  });

  it("does not re-announce a failure that clears and comes straight back", () => {
    vi.useFakeTimers();
    const { rerender, props } = renderShell({ failure: OVERFIT });
    const region = screen.getByRole("alert");
    const before = region.textContent;

    // Convolution Kitchen clears its failure on every edit and re-derives the
    // same one ~1–2 s later. The visible strip goes, the words stay put.
    rerender(<GameShell {...props} failure={null} />);
    expect(screen.queryByTestId("named-failure")).not.toBeInTheDocument();
    act(() => vi.advanceTimersByTime(1500));
    expect(region.textContent).toBe(before);

    rerender(<GameShell {...props} failure={{ ...OVERFIT }} />);
    expect(region.textContent).toBe(before);
  });

  it("announces the same failure again when it comes back after a Retry", () => {
    vi.useFakeTimers();
    // A game whose Retry clears its failure, the way every game's store does.
    let failAgain!: () => void;
    function Game(props: GameShellProps) {
      const [failure, setFailure] = useState<NamedFailure | null>(OVERFIT);
      failAgain = () => setFailure({ ...OVERFIT });
      return <GameShell {...props} failure={failure} onRetry={() => setFailure(null)} />;
    }
    // renderShell only for its default props.
    const { props, unmount: drop } = renderShell();
    drop();
    const { unmount } = render(<Game {...props} />);
    const region = screen.getByRole("alert");
    expect(region).toHaveTextContent("Overfitting. 99% train, 61% test");

    // Retry is a new attempt. The Skier's "Run to end" is synchronous, so the
    // same divergence can be back well inside the 3 s grace period.
    act(() => within(screen.getByTestId("named-failure")).getByRole("button", { name: /retry/i }).click());
    expect(screen.queryByTestId("named-failure")).not.toBeInTheDocument();
    expect(region).toBeEmptyDOMElement();

    act(() => vi.advanceTimersByTime(800));
    act(() => failAgain());
    expect(region).toHaveTextContent("Overfitting. 99% train, 61% test");
    unmount();
  });

  it("does not blank a failure that a Retry left in place", () => {
    const { props } = renderShell({ failure: OVERFIT, onRetry: () => {} });
    const region = screen.getByRole("alert");
    act(() => screen.getAllByRole("button", { name: /retry/i })[0]!.click());
    // Same props: the game kept the failure. Still said, still on screen.
    expect(props.failure).toEqual(OVERFIT);
    expect(region).toHaveTextContent("Overfitting. 99% train, 61% test");
  });

  it("empties after the grace period, and announces a different failure at once", () => {
    vi.useFakeTimers();
    const { rerender, props } = renderShell({ failure: OVERFIT });
    const region = screen.getByRole("alert");

    rerender(<GameShell {...props} failure={null} />);
    act(() => vi.advanceTimersByTime(3000));
    expect(region).toBeEmptyDOMElement();

    rerender(
      <GameShell
        {...props}
        failure={{ name: "Underfitting", detail: "58% train, 57% test" }}
      />,
    );
    expect(region).toHaveTextContent("Underfitting. 58% train, 57% test");
  });
});

describe("<GameShell> WhyCard announcements (WCAG 4.1.3)", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  const politeRegions = () =>
    [...document.querySelectorAll('[aria-live="polite"]')].filter(
      (node) => !node.closest('[aria-label="Live metric"]'),
    );

  it("announces the new card's headline politely — not its body", () => {
    vi.useFakeTimers();
    const { rerender, props } = renderShell();

    rerender(
      <GameShell
        {...props}
        whyCard={{
          key: "split-1",
          title: "That split was worth it",
          body: "Gini fell from 0.50 to 0.18 on the left branch.",
          tone: "good",
        }}
      />,
    );

    const texts = () => politeRegions().map((node) => node.textContent);
    // Debounced, so it lands after the metric's own sentence.
    expect(texts()).not.toContain("Good news: That split was worth it");
    act(() => vi.advanceTimersByTime(700));
    expect(texts()).toContain("Good news: That split was worth it");
    expect(texts().join(" ")).not.toContain("Gini fell");
  });

  it("announces only the last card of a quick burst", () => {
    vi.useFakeTimers();
    const { rerender, props } = renderShell();
    const card = (n: number) => ({ key: `k${n}`, title: `Card ${n}`, body: "…" });

    rerender(<GameShell {...props} whyCard={card(1)} />);
    act(() => vi.advanceTimersByTime(200));
    rerender(<GameShell {...props} whyCard={card(2)} />);
    act(() => vi.advanceTimersByTime(700));

    const spoken = politeRegions().map((node) => node.textContent).join(" | ");
    expect(spoken).toContain("Explanation: Card 2");
    expect(spoken).not.toContain("Card 1");
  });

  it("stays out of the WhyCard region, so its visible text is unchanged", () => {
    vi.useFakeTimers();
    renderShell({ whyCard: { key: "a", title: "Headline", body: "Body." } });
    act(() => vi.advanceTimersByTime(700));
    const dock = screen.getByRole("region", { name: /why did that happen/i });
    for (const node of politeRegions()) expect(dock).not.toContainElement(node as HTMLElement);
  });
});

describe("<GameShell> layout contract", () => {
  it("pins only the primary metric, and keeps it and the controls in landmarks", () => {
    renderShell({
      secondaryMetrics: [
        { label: "Complexity", value: 7, format: "integer", goodDirection: "down" },
      ],
    });

    const primary = screen.getByRole("region", { name: "Live metric" });
    const secondary = screen.getByRole("region", { name: "More metrics" });
    expect(primary.className).toContain("sticky");
    // Secondary readouts scroll with the page: pinning them too used to take
    // ~60% of a phone screen.
    expect(secondary.className).not.toContain("sticky");
    expect(primary).not.toContainElement(secondary);

    expect(
      screen.getByRole("region", { name: "Controls" }),
    ).toContainElement(screen.getByRole("button", { name: "Reset boundary" }));
  });

  it("keeps the header from pinning below lg, where it wraps and covered the metric", () => {
    renderShell();
    const header = document.querySelector("header")!;
    const classes = header.className.split(/\s+/);
    expect(classes).toContain("lg:sticky");
    expect(classes).not.toContain("sticky");
  });

  it("unpins the metric only on zoom-short viewports, at the same height the focus padding drops", () => {
    // jsdom evaluates no media queries, so this pins the numbers instead. A
    // browser check of the same rule (320×256 static; 740×360, 844×390 and
    // 360×740 pinned and hit-testable) lives in the engine fix notes.
    renderShell();
    const primary = screen.getByRole("region", { name: "Live metric" });
    const variant = primary.className.match(/\[@media\(max-height:(\d+)px\)\]:static/);
    expect(variant).not.toBeNull();
    const unpinAt = Number(variant![1]);

    // 400% zoom: 1280×1024 → 320×256, 1920×1080 → 480×270. Nothing may pin.
    expect(unpinAt).toBeGreaterThanOrEqual(270);
    // Landscape phones: 740×360, 844×390, and ~340 px with Safari's toolbars.
    // DESIGN.md: the metric is never hidden, so they must keep the pin.
    expect(unpinAt).toBeLessThan(340);

    // globals.css must drop its 9rem scroll-padding at exactly that height:
    // lower, and focus lands under a pinned card (WCAG 2.4.11); higher, and
    // there is dead padding with nothing pinned.
    const css = readFileSync(join(process.cwd(), "src", "app", "globals.css"), "utf8");
    const padding = css.match(
      /@media \(max-height: (\d+)px\) \{\s*html:has\(\[data-game\]\) \{\s*scroll-padding-top: 1rem;/,
    );
    expect(padding).not.toBeNull();
    expect(Number(padding![1])).toBe(unpinAt);
  });

  it("keeps the controls rail as the div right after the canvas (playthroughs rely on it)", () => {
    const { container } = renderShell();
    const rail = container.querySelector("#game-canvas + div");
    expect(rail).not.toBeNull();
    expect(rail).toHaveTextContent("Controls");
  });
});

describe("<GameShell> Math dialog makes the page inert", () => {
  it("marks header, content and footer inert while open, and not the announcers", async () => {
    const { user } = renderShell({ failure: { name: "Divergence", detail: "loss = Infinity" } });
    await user.click(screen.getByRole("button", { name: /math/i }));

    expect(document.querySelector("header")).toHaveAttribute("inert");
    expect(screen.getByTestId("named-failure").closest("[inert]")).not.toBeNull();
    expect(document.querySelector("footer")).toHaveAttribute("inert");
    // The dialog and the live regions stay live.
    expect(screen.getByRole("dialog").closest("[inert]")).toBeNull();
    expect(screen.getByRole("alert").closest("[inert]")).toBeNull();

    await user.keyboard("{Escape}");
    await waitFor(() => expect(document.querySelector("[inert]")).toBeNull());
  });
});

describe("<GameShell> a crashing lane", () => {
  it("keeps the shell (title, metric, controls) up and offers a way back", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    let shouldThrow = true;
    function Boom() {
      if (shouldThrow) throw new Error("WebGL context lost");
      return <div>recovered canvas</div>;
    }

    try {
      const { user } = renderShell({ visual: <Boom /> });

      expect(screen.getByText(/visual lane stopped working/i)).toBeInTheDocument();
      expect(screen.getByText(/WebGL context lost/)).toBeInTheDocument();
      expect(screen.getByRole("heading", { name: "Sort-It Arcade" })).toBeInTheDocument();
      expect(screen.getByTestId("metric-value")).toHaveTextContent("84%");

      shouldThrow = false;
      await user.click(screen.getByRole("button", { name: /try the visual lane again/i }));
      expect(screen.getByText("recovered canvas")).toBeInTheDocument();
    } finally {
      consoleError.mockRestore();
    }
  });
});

describe("<GameShell> progression extras", () => {
  const progress = {
    level: 1,
    xpIntoLevel: 20,
    xpForNextLevel: 200,
    stars: 2 as const,
    starCriteria: ["Finish", "Score 80%", "Clear the code lane"],
  };

  afterEach(() => {
    useProgression.setState({
      ...EMPTY_PROGRESSION,
      syncError: null,
      lastGain: null,
    });
  });

  it("tells the player when progress isn't being saved", () => {
    useProgression.setState({ syncError: "QuotaExceededError" });
    renderShell({ progress });
    expect(screen.getByRole("status")).toHaveTextContent(
      "Progress isn't being saved on this device.",
    );
  });

  it("shows this game's concept badge once earned", () => {
    useProgression.setState({ badges: ["sort-it-arcade"] });
    renderShell({ progress });
    expect(screen.getByRole("status")).toHaveTextContent(
      "I understand decision boundaries",
    );
  });

  it("drops a pending XP flash when the game unmounts", () => {
    useProgression.setState({ lastGain: 40 });
    const { unmount } = renderShell({ progress: { ...progress, recentGain: 40 } });
    unmount();
    expect(useProgression.getState().lastGain).toBeNull();
  });

  it("makes each star's criterion reachable without hover", async () => {
    const { user } = renderShell({ progress });
    const toggle = screen.getByRole("button", { name: "What earns each star" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");

    await user.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    const items = screen.getAllByRole("listitem").map((li) => li.textContent);
    expect(items).toEqual([
      "Finish (earned)",
      "Score 80% (earned)",
      "Clear the code lane (not yet)",
    ]);
  });
});
