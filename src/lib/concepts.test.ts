import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { conceptSlugsFromSource, siteRoutes } from "../../scripts/harness.mjs";
import * as heist from "@/games/hyperparameter-heist/ml";
import * as kmeans from "@/games/k-means-territory-wars/ml";
import { useKMeansStore } from "@/games/k-means-territory-wars/store";
import * as skier from "@/games/gradient-descent-skier/ml";
import * as sortIt from "@/games/sort-it-arcade/ml";
import { useSortItStore } from "@/games/sort-it-arcade/store";
import * as tree from "@/games/decision-tree-architect/ml";
import { useArchitectStore } from "@/games/decision-tree-architect/store";
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
    // The browser harnesses run under plain node and cannot import this
    // TypeScript module, so `scripts/harness.mjs` reads the concept slugs out
    // of this file's source with a regex. A regex over source is only as good
    // as the check on it: if the parse ever misses an entry, that page silently
    // drops out of the a11y audit and the mobile harness. So the parse must
    // equal the real library, exactly and in order.
    expect(conceptSlugsFromSource()).toEqual(conceptSlugs());

    const routes = siteRoutes().map((route) => route.path);
    expect(routes, "the harness route list has no concepts index").toContain(
      "/concepts",
    );
    for (const slug of conceptSlugs()) {
      expect(
        routes,
        `/concepts/${slug} is not in the harness route list`,
      ).toContain(`/concepts/${slug}`);
    }

    // And the audit really does use that list, rather than a hand-kept copy.
    const audit = readFileSync(
      join(process.cwd(), "scripts", "a11y-audit.mjs"),
      "utf8",
    );
    expect(audit).toMatch(/import \{[^}]*\bsiteRoutes\b[^}]*\} from "\.\/harness\.mjs"/);
  });

  it("is parsed by the harness regex, not fooled by it", () => {
    // The parse ignores the interface's `slug: string;` and anything before the
    // library, and fails loudly rather than returning an empty route list.
    expect(
      conceptSlugsFromSource(
        'interface X { slug: string; }\nexport const CONCEPT_LIBRARY = [\n  { slug: "a-b" },\n  { slug: "c" },\n];',
      ),
    ).toEqual(["a-b", "c"]);
    expect(() => conceptSlugsFromSource("export const NOTHING = [];")).toThrow(
      /no concept slugs/,
    );
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

describe("numbers the library quotes from the games", () => {
  // The library's house rule is "never invent a number: every figure quoted is
  // one the shipped games actually produce". Figures had drifted from what the
  // games compute (a grid said to see three learning rates that sees two, a
  // tree "two levels shallower" that was three, a Sort-It verdict from a round
  // that was since re-seeded, a momentum run no slider setting produces). These
  // tests recompute the quoted figures from the games' own ML and stores, so a
  // retune that changes them fails here and the prose gets updated with it —
  // instead of the page quietly teaching a number the game no longer shows.
  //
  // Only the pure, TF.js-free games are pinned, so this stays fast. Data
  // Detox's figures are pinned in data-detox.test.ts.

  const prose = (slug: string) =>
    getConcept(slug)!
      .sections.flatMap((section) => section.body)
      .join(" ");
  const pct = (ratio: number) => `${Math.round(ratio * 100)}%`;
  const WORDS = ["zero", "one", "two", "three", "four", "five", "six"];

  it("Hyperparameter Heist: the grid's learning rates, and random search's", () => {
    const dial = heist.dominantDial();
    // The dominant dial is the learning rate, which is what the prose names.
    expect(heist.DIALS[dial]?.name ?? "").toMatch(/learning rate/i);

    const asTrials = (points: number[][]) =>
      points.map((params, index) => ({
        params,
        objectiveValue: 0,
        index: index + 1,
        source: "grid" as const,
      }));
    const gridSeen = heist.distinctValuesTried(
      asTrials(heist.gridPoints(heist.BUDGET)),
      dial,
    );
    let randomSeen = 0;
    const runs = 200;
    for (let seed = 1; seed <= runs; seed += 1) {
      randomSeen += heist.distinctValuesTried(
        asTrials(heist.randomPoints(heist.BUDGET, seed)),
        dial,
      );
    }
    const text = prose("learning-rate");
    expect(text).toContain(`only ${WORDS[gridSeen]} distinct learning rates`);
    expect(heist.BUDGET).toBe(16);
    expect(text).toContain("sixteen-try budget");
    // "around twelve": the average over many seeds rounds to it.
    expect(Math.round(randomSeen / runs)).toBe(12);
    expect(text).toContain("sees around twelve");

    // "never cracks the safe, while random search … usually does".
    expect(text).toContain("never cracks the safe");
    expect(heist.gridBest(heist.BUDGET)).toBeLessThan(heist.CRACK_THRESHOLD);
    expect(text).toContain("usually does");
    expect(heist.randomCrackRate(heist.BUDGET)).toBeGreaterThan(0.5);
  });

  it("Decision Tree Architect: the first plot's depth curve", () => {
    // The store's own first plot, exactly as a player meets it.
    useArchitectStore.getState().restart();
    const { dataset, roundIndex } = useArchitectStore.getState();
    const round = tree.roundAt(roundIndex);
    expect(round.index).toBe(1);
    const curve = tree.depthCurve(dataset, round.maxDepth);
    const peak = curve.reduce((best, point) =>
      point.validationAccuracy > best.validationAccuracy ? point : best,
    );
    const deepest = curve[curve.length - 1]!;

    expect(prose("overfitting")).toContain(
      `growing the tree from depth ${peak.depth} to depth ${deepest.depth} takes training accuracy from ${pct(
        peak.trainAccuracy,
      )} to ${pct(deepest.trainAccuracy)} while validation falls from ${pct(
        peak.validationAccuracy,
      )} to ${pct(deepest.validationAccuracy)}`,
    );
  });

  it("Sort-It Arcade: the first round's overfit verdict, word for word", () => {
    // The route the prose describes, through the real store: a fresh first
    // round, Wiggle, "Fit it for me", Check generalization.
    useSortItStore.getState().startRound(1);
    useSortItStore.getState().setBoundaryType("wiggle");
    useSortItStore.getState().autoFit();
    const evaluation = useSortItStore.getState().check();

    expect(evaluation.outcome).toBe("overfit");
    expect(useSortItStore.getState().boundary.params).toHaveLength(
      sortIt.KNOT_COUNTS.wiggle,
    );
    // Read the percentages off the verdict the player is shown, so the prose
    // is held to the game's own rounding rather than to this test's.
    const shown = evaluation.failure?.detail.match(
      /^(\d+)% on the (\d+) points you fitted, (\d+)% on (\d+) points it never saw/,
    );
    expect(shown, evaluation.failure?.detail).toBeTruthy();
    const [, train, fitted, test, unseen] = shown!;

    const text = prose("overfitting");
    expect(text).toContain(`${sortIt.KNOT_COUNTS.wiggle}-parameter Wiggle`);
    expect(text).toContain(
      `the overfit verdict reads ${train}% on the ${fitted} points you fitted and ${test}% on ${unseen} points it never saw — the ${train}% on its own looks like success, and it is the ${
        Number(train) - Number(test)
      }-point gap`,
    );
  });

  it("Sort-It Arcade: on every round the wiggle wins on seen points and loses on unseen ones", () => {
    const flat = (count: number) => Array.from({ length: count }, () => 0.5);
    for (let round = 1; round <= sortIt.ROUND_SEEDS.length; round += 1) {
      const seed = sortIt.seedForRound(round);
      const train = sortIt.generateTrainPoints(seed);
      const test = sortIt.generateTestPoints(seed);
      // From a flat start, which is what switching capacity on a fresh round
      // hands "Fit it for me".
      const fit = (type: sortIt.BoundaryType) =>
        sortIt.fitKnots(train, flat(sortIt.KNOT_COUNTS[type]));
      const line = fit("line");
      const curve = fit("curve");
      const wiggle = fit("wiggle");

      const seen = (params: number[]) => sortIt.accuracyOf(train, params);
      const unseen = (params: number[]) => sortIt.accuracyOf(test, params);
      expect(seen(wiggle), `round ${round}: wiggle vs curve, seen`).toBeGreaterThan(
        seen(curve),
      );
      expect(seen(wiggle), `round ${round}: wiggle vs line, seen`).toBeGreaterThan(
        seen(line),
      );
      expect(unseen(wiggle), `round ${round}: wiggle vs curve, unseen`).toBeLessThan(
        unseen(curve),
      );
    }

    const text = prose("decision-boundaries");
    expect(sortIt.KNOT_COUNTS.curve).toBe(5);
    expect(text).toContain(
      "on every round, the wiggliest boundary scores best on the points you can see, and worse than the five-parameter curve on the ones you cannot",
    );
    // "A boundary with 25 free parameters fitted to 200 points" is the Wiggle.
    expect(text).toContain(
      `${sortIt.KNOT_COUNTS.wiggle} free parameters fitted to ${sortIt.TRAIN_SIZE} points`,
    );
  });

  it("K-Means Territory Wars: the first map's elbow table, and eight flags judged a bad k", () => {
    useKMeansStore.getState().startRound(1);
    const { elbowPoints, trueK, points, seed } = useKMeansStore.getState();
    const inertiaAt = (k: number) =>
      elbowPoints.find((point) => point.k === k)!.inertia;
    const three = inertiaAt(3);
    const eight = inertiaAt(8);

    expect(trueK).toBe(3);
    // The elbow table prints inertia to two places, as the prose does.
    const text = prose("choosing-k");
    expect(text).toContain("which has three real groups");
    expect(text).toContain(
      `${three.toFixed(2)} for three flags and ${eight.toFixed(2)} for eight`,
    );
    expect(three / eight).toBeGreaterThan(2);
    expect(text).toContain("more than twice as good");

    // Settle eight flags on the best eight-flag answer: still a bad k.
    const best = kmeans.solveKMeans(points, 8, { seed });
    while (useKMeansStore.getState().centroids.length < 8) {
      useKMeansStore.getState().addFlag();
    }
    best.centroids.forEach((centroid, index) =>
      useKMeansStore.getState().moveFlag(index, centroid.x, centroid.y),
    );
    useKMeansStore.getState().settle();
    expect(useKMeansStore.getState().inertia.toFixed(2)).toBe(eight.toFixed(2));
    const verdict = useKMeansStore.getState().check();
    expect(verdict.failure?.name).toBe("Bad k");
    expect(text).toContain(`the ${verdict.failure!.name} verdict`);
  });

  it("Overfit Tower Defense: L1 is not said to reach exactly zero", () => {
    // Its model trains with Adam, which leaves L1-quieted weights close to zero
    // but not on it (measured: none of 64 distractor weights is exactly 0), and
    // its L1 card says so. The page it links to must not contradict the card.
    const text = prose("overfitting");
    expect(text).not.toMatch(/L1 [^.]*to exactly zero/);
    expect(text).toContain("L1 pushes the unhelpful ones close to zero");
  });

  /** One full Skier run from the top, and the verdict the game gives it. */
  const skierRun = (learningRate: number, momentum: number) => {
    const run = skier.descend(learningRate, momentum);
    const verdict = skier.evaluate({
      pos: run.final.pos,
      finalLoss: run.finalLoss,
      steps: run.steps,
      diverged: run.diverged,
      settled: run.settled,
      learningRate,
      momentum,
      overshoots: run.overshoots,
      rises: run.rises,
      minX: run.minX,
    });
    return { run, verdict };
  };

  it("Gradient Descent Skier: where plain descent stops, and where the deep valley is", () => {
    const plain = skier.descend(skier.DEFAULT_LEARNING_RATE, 0);
    const global = skier.GLOBAL_MINIMUM;
    expect(prose("gradient-descent")).toContain(
      `plain descent settles at loss ${plain.finalLoss.toFixed(2)}, in the shallow valley at x ${plain.final.pos.x.toFixed(
        2,
      )}, while the deepest valley is at x ${global.x.toFixed(2)} with a loss of ${global.loss.toFixed(2)}`,
    );
  });

  it("Gradient Descent Skier: the same rate with momentum 0.9 settles in the deep valley", () => {
    // A stop the momentum slider really has (0 to 0.95 in steps of 0.01).
    const MOMENTUM = 0.9;
    expect(MOMENTUM).toBeLessThanOrEqual(skier.MAX_MOMENTUM);
    const plain = skier.descend(skier.DEFAULT_LEARNING_RATE, 0);
    const { run } = skierRun(skier.DEFAULT_LEARNING_RATE, MOMENTUM);

    expect(run.settled).toBe(true);
    expect(run.diverged).toBe(false);
    // "— the global minimum": within the game's own reach tolerance, and at the
    // same two-place x the local-minima section gives the deep valley.
    expect(
      skier.distanceTo(run.final.pos, skier.GLOBAL_MINIMUM),
    ).toBeLessThanOrEqual(skier.REACH_TOLERANCE);
    expect(run.final.pos.x.toFixed(2)).toBe(skier.GLOBAL_MINIMUM.x.toFixed(2));

    expect(prose("gradient-descent")).toContain(
      `the identical learning rate that got stuck at ${plain.finalLoss.toFixed(
        2,
      )} and set momentum to ${MOMENTUM}, and the skier is carried through the shallow valley and settles at a loss of ${run.finalLoss.toFixed(
        2,
      )}, at x ${run.final.pos.x.toFixed(2)} — the global minimum`,
    );
  });

  it("Gradient Descent Skier: where plain descent stops holding, and where it explodes", () => {
    const text = prose("learning-rate");
    // The 2/L line the prose names, as the game computes it.
    expect(text).toContain(
      `puts that line at a plain-descent rate of about ${skier.STABLE_RATE.toFixed(2)}`,
    );

    // Walk the rate up in hundredths, as far as the dial goes.
    const rates: number[] = [];
    for (
      let hundredths = Math.max(1, Math.round(skier.MIN_LEARNING_RATE * 100));
      hundredths <= Math.round(skier.MAX_LEARNING_RATE * 100);
      hundredths += 1
    ) {
      rates.push(hundredths / 100);
    }

    // Just past the line the skier bounces rather than explodes, and the game
    // names that Oscillation.
    const justPast = rates.filter(
      (rate) => rate > skier.STABLE_RATE && rate <= skier.STABLE_RATE + 0.1,
    );
    expect(justPast.length).toBeGreaterThan(0);
    for (const rate of justPast) {
      const { run, verdict } = skierRun(rate, 0);
      expect(run.diverged, `rate ${rate} diverged`).toBe(false);
      expect(verdict.failure?.name, `rate ${rate}`).toBe("Oscillation");
    }
    expect(text).toContain("which the game names Oscillation");

    // Where the loss first runs away, and how fast.
    const firstDiverging = rates.find((rate) => skierRun(rate, 0).run.diverged);
    const firstStepDiverging = rates.find((rate) => {
      const { run } = skierRun(rate, 0);
      return run.diverged && run.steps === 1;
    });
    expect(firstDiverging).toBeDefined();
    expect(firstStepDiverging).toBeDefined();
    const runaway = skierRun(firstDiverging!, 0);
    expect(runaway.verdict.failure?.name).toBe("Divergence");
    // From there up every rate diverges, so it really is a line, not a fluke,
    // and from the second one up every run lasts a single step.
    for (const rate of rates.filter((r) => r > firstDiverging!)) {
      expect(skierRun(rate, 0).run.diverged, `rate ${rate}`).toBe(true);
    }
    for (const rate of rates.filter((r) => r >= firstStepDiverging!)) {
      expect(skierRun(rate, 0).run.steps, `rate ${rate}`).toBe(1);
    }

    expect(text).toContain(
      `The loss runs away from a rate of about ${firstDiverging!.toFixed(
        2,
      )}, within ${WORDS[runaway.run.steps]} steps, and from ${firstStepDiverging!.toFixed(
        2,
      )} on the very first one`,
    );
  });
});
