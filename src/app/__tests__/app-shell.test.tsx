import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GameLoading } from "@/games/GameMount";
import { playableSlugs } from "@/games/registry";
import { conceptSlugs } from "@/lib/concepts";
import { SkipLink } from "../_components/SkipLink";
import RouteError from "../error";
import NotFound from "../not-found";
import PlayPage from "../play/[slug]/page";
import robots from "../robots";
import sitemap from "../sitemap";

/**
 * The site-level pieces around the games: the skip link, the 404 and error
 * pages, and the crawler files. Each one used to be missing or a dead end.
 */

describe("<SkipLink>", () => {
  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("skips past the game header to the canvas once a game has mounted", async () => {
    const user = userEvent.setup();
    render(
      <>
        <SkipLink />
        <div id="main-content">
          <header>
            <button type="button">Back</button>
          </header>
          <main id="game-canvas" tabIndex={-1}>
            canvas
          </main>
        </div>
      </>,
    );
    await user.click(screen.getByRole("link", { name: "Skip to content" }));
    // Not the wrapper, whose first stop is the header's Back link.
    expect(document.getElementById("game-canvas")).toHaveFocus();
  });

  it("focuses #main-content on pages without a game", async () => {
    const user = userEvent.setup();
    render(
      <>
        <SkipLink />
        <main id="main-content" tabIndex={-1}>
          page
        </main>
      </>,
    );
    await user.click(screen.getByRole("link", { name: "Skip to content" }));
    expect(document.getElementById("main-content")).toHaveFocus();
  });

  it("is a real in-page link, so it works before JavaScript loads", () => {
    render(<SkipLink />);
    expect(screen.getByRole("link", { name: "Skip to content" })).toHaveAttribute(
      "href",
      "#main-content",
    );
  });
});

describe("404 page", () => {
  it("has a main landmark for the skip link, a heading, and ways back", () => {
    render(<NotFound />);
    const main = screen.getByRole("main");
    expect(main).toHaveAttribute("id", "main-content");
    expect(screen.getByRole("heading", { level: 1 })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /all games/i })).toHaveAttribute(
      "href",
      "/",
    );
    expect(screen.getByRole("link", { name: /concept library/i })).toHaveAttribute(
      "href",
      "/concepts",
    );
  });
});

describe("route error page", () => {
  let consoleError: ReturnType<typeof vi.spyOn>;
  beforeEach(() => {
    consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
  });
  afterEach(() => {
    consoleError.mockRestore();
  });

  it("explains, retries via retry() when Next provides it, and links home", async () => {
    const user = userEvent.setup();
    const retry = vi.fn();
    const reset = vi.fn();
    const error = new Error("boom");
    render(<RouteError error={error} retry={retry} reset={reset} />);

    expect(screen.getByRole("main")).toHaveAttribute("id", "main-content");
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      /unexpected error/i,
    );
    expect(screen.getByRole("link", { name: /back to all games/i })).toHaveAttribute(
      "href",
      "/",
    );
    await user.click(screen.getByRole("button", { name: /try again/i }));
    expect(retry).toHaveBeenCalledTimes(1);
    expect(reset).not.toHaveBeenCalled();
    // Logged, not swallowed.
    expect(consoleError).toHaveBeenCalledWith("[GameML] Unhandled error:", error);
  });

  it("falls back to reset() on a Next without retry()", async () => {
    const user = userEvent.setup();
    const reset = vi.fn();
    render(<RouteError error={new Error("boom")} reset={reset} />);
    await user.click(screen.getByRole("button", { name: /try again/i }));
    expect(reset).toHaveBeenCalledTimes(1);
  });

  it("advises a reload, not a retry, when a chunk failed to download", () => {
    const error = Object.assign(new Error("Loading chunk 9 failed."), {
      name: "ChunkLoadError",
    });
    render(<RouteError error={error} reset={vi.fn()} />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      /didn't finish downloading/i,
    );
    expect(screen.getByRole("button", { name: /reload the page/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /try again/i })).not.toBeInTheDocument();
  });
});

describe("crawler files", () => {
  it("lists the home page, the library, every concept and every playable game", () => {
    const urls = sitemap().map((entry) => new URL(entry.url).pathname);
    expect(urls).toContain("/");
    expect(urls).toContain("/concepts");
    for (const slug of conceptSlugs()) expect(urls).toContain(`/concepts/${slug}`);
    for (const slug of playableSlugs()) expect(urls).toContain(`/play/${slug}`);
    expect(new Set(urls).size, "duplicate sitemap entry").toBe(urls.length);
    expect(urls).toHaveLength(2 + conceptSlugs().length + playableSlugs().length);
    // Absolute URLs, as the sitemap protocol requires.
    for (const entry of sitemap()) expect(entry.url).toMatch(/^https?:\/\//);
  });

  it("allows crawling and points at the sitemap on the same origin", () => {
    const file = robots();
    expect(file.rules).toEqual({ userAgent: "*", allow: "/" });
    const sitemapUrl = new URL(String(file.sitemap));
    expect(sitemapUrl.pathname).toBe("/sitemap.xml");
    expect(sitemapUrl.origin).toBe(new URL(sitemap()[0]!.url).origin);
  });
});

describe("game page without JavaScript", () => {
  it("hides the loading status, which would promise a game that never arrives", async () => {
    const page = await PlayPage({
      params: Promise.resolve({ slug: "sort-it-arcade" }),
    });
    const html = renderToStaticMarkup(page);

    // The explanation and its stylesheet are inside <noscript>, so only a
    // browser with scripting off ever applies them.
    const noscript = html.match(/<noscript>([\s\S]*?)<\/noscript>/)?.[1] ?? "";
    expect(noscript).toContain("needs");
    const css = noscript.match(/<style>([^<]*)<\/style>/)?.[1];
    expect(css, "no stylesheet inside <noscript>").toBeDefined();
    const rule = css!.match(/^\s*([^{]+)\{\s*display\s*:\s*none\s*;?\s*\}\s*$/);
    expect(rule, `unexpected noscript CSS: ${css}`).not.toBeNull();

    // …and that rule really selects the loading box GameMount renders, so the
    // two can't drift apart silently.
    render(<GameLoading />);
    const loading = screen.getByRole("status");
    expect(loading).toHaveTextContent(/loading game/i);
    expect(loading.matches(rule![1]!.trim())).toBe(true);
  });
});
