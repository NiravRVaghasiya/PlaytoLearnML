---
name: new-game-module
description: Scaffold and implement a new GameML game module from its spec entry, following the fixed file layout, two-lane rule, and pedagogy contract. Use whenever adding a game under src/games/.
---

# New Game Module

## Overview

This skill turns a game's entry in `docs/GameML_Build_Spec.md` into a working, tested, **registered** module under `src/games/<slug>/`. It enforces:

- the two-lane rule: visual and code lanes sharing one state store;
- the pedagogy contract: core intuition, player-action = algorithm, live feedback, named failure mode;
- client-side-only ML.

Use it for every new game so all modules stay structurally identical and reuse the shared engine. The API contract is `docs/ENGINE_API.md`. `src/games/sort-it-arcade/` is the smallest complete reference.

## Workflow

1. **Locate the spec.** Open `docs/GameML_Build_Spec.md`, find the game, and extract its fields: concept, core intuition, player-action = algorithm, gameplay loop, win/lose/scoring, live feedback, data model, components, ML logic and tech. Also read spec §9, which lists where shipped games deliberately differ and why; the same reasoning applies to a new one.
2. **Pick the slug.** kebab-case (e.g. `k-means-territory-wars`). Create `src/games/<slug>/`.
3. **Store first (`store.ts`).** Implement the spec's data model as a typed Zustand store. This is the single source of truth both lanes share. It also holds `lane`, `whyCard`, `failure` and the round state.
   - The scoring action (e.g. `check`) takes `source?: Lane`, defaulting to `"visual"`.
   - It records progress with `recordResult({ lane: source, codeLaneCleared: source === "code", … })`, so the third star follows the action, not the visible tab.
   - It records a clear once per board, but still lets a first code-lane clear through. See `sort-it-arcade/store.ts` `check()`.
4. **ML logic (`ml.ts`).** Implement the real computation that produces the live metric. No server calls. Export pure, testable functions.
   - Use TF.js only where a model actually trains, through `useModel` or the `tfjs-model-lifecycle` rules. Plain TypeScript is the norm for everything else: 9 of the 14 games have no TF.js.
   - Build procedural data from `seededRandom` / `gaussian` in `@/lib/utils`, so rounds are reproducible.
   - Export `MATH_EQUATION` (LaTeX), `MATH_CODE` (the real code, copied from this file) and `MATH_NOTES`. `MathDrawer.test.tsx` fails if a playable game has no `MATH_EQUATION`, or if KaTeX flags it in strict mode.
5. **Visual lane (`VisualLane.tsx`).** No-code interaction (drag/click/slider) bound to the store.
   - Draw 2D views in **SVG** (D3 for scales and shapes), not canvas: *when the numbers are the content, use SVG*. Use Three.js only when depth is the content, React Flow for node graphs. There is no p5.js, Phaser or Highcharts in the project; don't add them without asking.
   - Every drag needs a keyboard path and a single-pointer path (a tap, or − / + buttons; WCAG 2.5.7).
   - Map pointer events into SVG coordinates with `getScreenCTM()`, not the bounding box. See `sort-it-arcade/field.ts`.
6. **Code lane (`CodeLane.tsx`).** Editable snippet driving the SAME store via `useCodeLane`.
   - Build the api in a plain function (`createCodeApi()` in `store.ts` or `code-api.ts`), so it is testable without React.
   - Verbs are small and named like the algorithm. They validate their arguments, throw named errors, and pass `"code"` as the source of scoring actions.
   - Export `STARTER_CODE` so a test can run it through `createJsExecutor`.
   - Size `maxRunMs` for the slowest honest run: TF.js's CPU backend is much slower than WebGL. If the api is where the time goes (a `train()` or `trial()` verb), also pass `budgetApiCalls: true`, as Neuron Forge and Overfit TD do, so `for (;;) await api.train()` times out.
   - The player's draft survives lane switches, keyed by `initialCode` by default, so keep `STARTER_CODE` unique to the game (or pass `persistKey`).
   - For a Python lane, use `createPyodideExecutor({ packages, prelude, onStatus })`, and add every top-level package to `LANE_PACKAGES` in `scripts/setup-pyodide.mjs`, or the deploy ships without its wheels.
7. **Why-cards (`why-cards.ts`).** Map player actions and failure modes to 2-line, plain-language explanations, each with a new `key`. On a loss, the card's `title` is the named failure.
   - Add `conceptHref` / `conceptLabel` only where a `src/lib/concepts.ts` section heading equals `conceptLabel` exactly.
   - Also add the slug to that concept's `games` array. `concepts.test.ts` enforces both.
8. **Entry (`index.tsx`).** Mount inside `<GameShell>` with its real props: `slug`, `title`, `metric`, `secondaryMetrics`, `math`, `controls`, `visual`, `code`, `whyCard`, `failure`, `lane`/`onLaneChange`, `progress`, `onRetry`, `onNext`.
   - Hydrate progression on mount.
   - Build `STAR_CRITERIA` from `HIGH_SCORE_THRESHOLD`; don't restate 80%.
   - Default-export the component; `GameMount` imports it dynamically.
9. **Register it — or it is invisible.** Five places (CLAUDE.md, "Repo layout"):
   1. `src/lib/catalog.ts`: a `GAME_CATALOG` entry. `catalog.test.ts` pins the count at 14, so update it for a 15th game.
   2. `src/games/registry.ts`: `PLAYABLE_SLUGS`.
   3. `src/games/GameMount.tsx`: a `dynamic(() => import("./<slug>"), { ssr: false, loading: … })` entry.
   4. `src/engine/progression.ts`: `CONCEPT_BADGES`.
   5. `scripts/playthroughs/<slug>.mjs`: the browser playthrough (`slug`, `title`, `run({ page, check, metricText })`).

   Routes, the sitemap and the a11y/phone harness lists are derived from these. Never add `src/app/play/<slug>/page.tsx`.
10. **Tests (`<slug>.test.ts`).** Unit-test:
    - `ml.ts` correctness;
    - that the metric updates when the store changes;
    - the named failure fires on the right inputs and only then;
    - the code-lane api's argument checks;
    - the attribution of the third star.

    Anything that trains must return `tensorCount()` to baseline.
11. **Verify.**
    - Run `bun run verify`: tsc, eslint and the whole suite, including the registry, badge, KaTeX and concept-link tests.
    - Then run `bun run build && bun run start`, and against that server `bun run verify:play <slug>`, `bun run verify:mobile /play/<slug>` and `bun run audit:a11y http://localhost:3000/play/<slug>`. The audit needs ≥ 95 in both passes. The harnesses need Chromium, installed once with `bunx playwright install chromium`.
    - Finally run `/verify-game <slug>` and walk the CLAUDE.md Definition of Done.

## Reference

- File layout, registration, tech stack and Definition of Done: `CLAUDE.md`.
- Engine API (GameShell props, useModel, useCodeLane, progression): `docs/ENGINE_API.md`.
- Design tokens, GameShell anatomy, accessibility rules: `DESIGN.md`.
- TF.js model lifecycle patterns: the `tfjs-model-lifecycle` skill.
- See `template.md` in this skill folder for a copy-paste module skeleton.
