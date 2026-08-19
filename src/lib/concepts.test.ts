import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { GAME_CATALOG, getGameMeta } from "./catalog";
import { CONCEPT_LIBRARY, conceptSlugs, getConcept } from "./concepts";

/**
 * The Concept Library's integrity, and specifically the bug that prompted it:
 * all four Phase-1 games shipped WhyCards linking to `/concepts/*` before any
 * such route existed, so the "read the concept" affordance 404'd.
 *
 * A test that hardcoded the eight known slugs would have caught that once. This
 * one reads the game sources instead, so it catches the *next* one too — add a
 * WhyCard link to a concept nobody has written and the suite fails.
 */

const GAMES_DIR = join(process.cwd(), "src", "games");

interface GameLink {
  /** Game slug, from the directory name. */
  game: string;
  /** Concept slug the card links to. */
  concept: string;
  /** The link text, when the card supplies one. */
  label?: string;
}

/** Every `src/games/<slug>/why-cards.ts` on disk. */
function whyCardSources(): { game: string; source: string }[] {
  return readdirSync(GAMES_DIR, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => ({
      game: entry.name,
      path: join(GAMES_DIR, entry.name, "why-cards.ts"),
    }))
    .filter((entry) => existsSync(entry.path))
    .map((entry) => ({
      game: entry.game,
      source: readFileSync(entry.path, "utf8"),
    }));
}

/**
 * Concept slugs appearing as string literals in a why-cards module. Deliberately
 * naive: it matches the URL wherever it occurs, so it cannot be fooled by how
 * the games choose to store the href.
 */
function literalConceptSlugs(source: string): string[] {
  // `noUncheckedIndexedAccess` is on, so a capture group reads as
  // `string | undefined` even when the pattern guarantees it matched.
  return [...source.matchAll(/["'`]\/concepts\/([a-z0-9-]+)["'`]/g)].flatMap(
    (match) => (match[1] ? [match[1]] : []),
  );
}

/**
 * `conceptHref` / `conceptLabel` pairs, with the href resolved through the
 * module's own `const NAME = "/concepts/..."` declarations. Pairs whose href is
 * computed rather than declared are skipped — `linkedConceptSlugs` still covers
 * the literal, this only relaxes the label check.
 */
function labelledLinks(game: string, source: string): GameLink[] {
  const consts = new Map<string, string>();
  for (const match of source.matchAll(
    /const\s+(\w+)\s*=\s*["']\/concepts\/([a-z0-9-]+)["']/g,
  )) {
    const [, name, slug] = match;
    if (name && slug) consts.set(name, slug);
  }

  const links: GameLink[] = [];
  // The quote styles are spelled out separately on purpose: label copy contains
  // apostrophes ("Why dropping isn't neutral"), so a `["']` character class
  // ends the match in the middle of the word.
  for (const match of source.matchAll(
    /conceptHref:\s*([^,\n]+),(?:\s*conceptLabel:\s*(?:"([^"]*)"|'([^']*)'))?/g,
  )) {
    const expr = match[1]?.trim();
    if (!expr) continue;
    const direct = expr.match(/^["']\/concepts\/([a-z0-9-]+)["']$/);
    const concept = direct ? direct[1] : consts.get(expr);
    if (!concept) continue;
    links.push({ game, concept, label: match[2] ?? match[3] });
  }
  return links;
}

const SOURCES = whyCardSources();
const ALL_LINKS = SOURCES.flatMap(({ game, source }) =>
  labelledLinks(game, source),
);

describe("concept library", () => {
  it("has a unique, URL-safe slug per concept", () => {
    const slugs = conceptSlugs();
    expect(new Set(slugs).size, "duplicate concept slug").toBe(slugs.length);
    for (const slug of slugs) {
      expect(slug, `${slug} is not kebab-case`).toMatch(/^[a-z0-9]+(-[a-z0-9]+)*$/);
    }
  });

  it("gives every concept a question, an answer and real sections", () => {
    for (const concept of CONCEPT_LIBRARY) {
      expect(concept.title.length, `${concept.slug} title`).toBeGreaterThan(0);
      // The <title> tag is the searched-for phrase, so it has to read as one.
      expect(concept.question, `${concept.slug} question`).toMatch(/\?$/);
      expect(
        concept.answer.length,
        `${concept.slug} answer is too short to answer anything`,
      ).toBeGreaterThan(80);
      // Search engines truncate past ~160 characters; a cut-off description is
      // worse than a short one.
      expect(
        concept.summary.length,
        `${concept.slug} summary will be truncated`,
      ).toBeLessThanOrEqual(160);
      expect(
        concept.sections.length,
        `${concept.slug} has no sections`,
      ).toBeGreaterThan(0);

      for (const section of concept.sections) {
        expect(section.heading.length, `${concept.slug} section heading`)
          .toBeGreaterThan(0);
        expect(
          section.body.length,
          `${concept.slug} / ${section.heading} has no body`,
        ).toBeGreaterThan(0);
        for (const paragraph of section.body) {
          expect(
            paragraph.trim().length,
            `${concept.slug} / ${section.heading} has an empty paragraph`,
          ).toBeGreaterThan(0);
        }
      }
    }
  });

  it("only points at games that exist, and never at nothing", () => {
    for (const concept of CONCEPT_LIBRARY) {
      expect(
        concept.games.length,
        `${concept.slug} sends the reader nowhere`,
      ).toBeGreaterThan(0);
      for (const slug of concept.games) {
        expect(
          getGameMeta(slug),
          `${concept.slug} lists ${slug}, which is not in the catalog`,
        ).toBeDefined();
      }
      expect(
        new Set(concept.games).size,
        `${concept.slug} lists a game twice`,
      ).toBe(concept.games.length);
    }
  });

  it("cross-links only to concepts that exist, and not to itself", () => {
    for (const concept of CONCEPT_LIBRARY) {
      for (const slug of concept.related) {
        expect(
          getConcept(slug),
          `${concept.slug} links to ${slug}, which is not in the library`,
        ).toBeDefined();
        expect(slug, `${concept.slug} lists itself as related`).not.toBe(
          concept.slug,
        );
      }
    }
  });

  it("names failure modes the way the games name them", () => {
    // Not an exhaustive match against the catalog: the games use finer-grained
    // names than the catalog's one-per-game summary ("Sent the gradient backward
    // with the wrong sign" vs "Broken chain rule"). What matters is that a mode
    // is named and glossed, because an unnamed failure is the thing the pedagogy
    // contract forbids.
    for (const concept of CONCEPT_LIBRARY) {
      for (const mode of concept.failureModes) {
        expect(mode.name.length, `${concept.slug} failure mode name`)
          .toBeGreaterThan(0);
        expect(
          mode.gloss.length,
          `${concept.slug} / ${mode.name} has no explanation`,
        ).toBeGreaterThan(20);
      }
    }
  });
});

describe("concept links from the games", () => {
  it("found why-cards to scan", () => {
    // Guards the two regexes below: if a refactor moved the copy elsewhere, the
    // scan would silently pass by matching nothing at all.
    expect(SOURCES.length, "no why-cards.ts found under src/games").toBeGreaterThan(
      0,
    );
    expect(ALL_LINKS.length, "no conceptHref links parsed").toBeGreaterThan(0);
  });

  it("resolves every /concepts/ link in every game", () => {
    // The original bug, stated as a test. This is why the library exists.
    for (const { game, source } of SOURCES) {
      for (const slug of literalConceptSlugs(source)) {
        expect(
          getConcept(slug),
          `${game} links to /concepts/${slug}, which would 404`,
        ).toBeDefined();
      }
    }
  });

  it("keeps the promise a link's text makes", () => {
    // A card offering "What momentum does" has to land on a page with a section
    // headed exactly that. Otherwise the link resolves and still dead-ends.
    for (const link of ALL_LINKS) {
      if (!link.label) continue;
      const concept = getConcept(link.concept);
      expect(
        concept,
        `${link.game} links to /concepts/${link.concept}, which would 404`,
      ).toBeDefined();

      const headings = concept!.sections.map((s) => s.heading);
      expect(
        headings,
        `${link.game} offers "${link.label}" but /concepts/${link.concept} has no such section`,
      ).toContain(link.label);
    }
  });

  it("funnels back into the game that raised the question", () => {
    // Spec §8: the library's job is to hand the reader into the game. If a game
    // links out to a concept, that concept has to link back.
    for (const link of ALL_LINKS) {
      const concept = getConcept(link.concept);
      // Checked first so a missing concept reports as a missing concept, rather
      // than as `toContain` being handed an undefined subject.
      expect(
        concept,
        `${link.game} links to /concepts/${link.concept}, which would 404`,
      ).toBeDefined();
      expect(
        concept!.games,
        `/concepts/${link.concept} is linked from ${link.game} but does not list it`,
      ).toContain(link.game);
    }
  });

  it("is audited for accessibility, every page of it", () => {
    // `scripts/a11y-audit.mjs` lists its routes by hand, because it runs under
    // plain node and cannot import this TypeScript module. Hand-written lists
    // drift, and a concept page absent from the list is a page whose contrast
    // and heading order nobody checks. So the drift is a test failure.
    const audit = readFileSync(
      join(process.cwd(), "scripts", "a11y-audit.mjs"),
      "utf8",
    );
    expect(audit, "the audit script has no concepts index route").toContain(
      '"/concepts"',
    );
    for (const slug of conceptSlugs()) {
      expect(
        audit,
        `/concepts/${slug} is not in the a11y audit's route list`,
      ).toContain(`"/concepts/${slug}"`);
    }
  });

  it("covers every Phase-1 game, which is where the 404s were", () => {
    const phase1 = GAME_CATALOG.filter((g) => g.phase === 1).map((g) => g.slug);
    const linked = new Set(ALL_LINKS.map((l) => l.game));
    for (const slug of phase1) {
      expect(
        linked.has(slug),
        `${slug} has no concept links — it used to have broken ones`,
      ).toBe(true);
    }
  });
});
