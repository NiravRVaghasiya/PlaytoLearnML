# CLAUDE.md — GameML

> Operating manual for AI coding agents (Claude Code / Cursor / Windsurf) working in this repo.
> Read this first, every session. It defines how we build, the guardrails, and where things live.

## What we're building

**GameML** — an interactive website that teaches Machine Learning through games and hands-on challenges. 14 games across 6 ML categories. Full product spec lives in `docs/GameML_Build_Spec.md`. Design system in `DESIGN.md`.

**North star:** every game must teach a *specific, identifiable* ML concept. If a mechanic can't complete the pedagogy contract (below), it doesn't ship.

## The pedagogy contract (non-negotiable)

Every game module MUST satisfy all four before it's considered done:

1. **Core intuition** — one sentence the learner walks away with.
2. **Player action = algorithm** — the player's action must mirror what the ML algorithm actually does.
3. **Live feedback** — an always-visible metric (accuracy/loss/inertia) that moves in response to the player's action. It never hides.
4. **Named failure mode** — losing shows the *named* ML failure ("You overfit — 99% train, 61% test"), not a generic "Game Over."

If you're implementing a game and can't point to all four in code, stop and flag it.

## Golden rules

- **Two-lane rule.** Every game ships with a **visual lane** (no-code: drag/click/slider) AND a **code lane** (editable TF.js/Pyodide snippet driving the *same* game state). Never build one without the other.
- **Client-side ML only.** All training/inference runs in-browser via TensorFlow.js / Pyodide. No ML server, no GPU dependency. This is a hard cost constraint — do not add a backend inference endpoint.
- **Build the shared engine before the games.** `<GameShell>`, `useModel()`, `useCodeLane()`, `<MetricReadout>`, `<WhyCard>`, and the `progression` service are built once and reused. Never fork these per-game.
- **Progressive complexity.** Respect the phase roadmap. Don't start a Phase 3 game while Phase 1 is incomplete.
- **Accessibility is not optional.** Keyboard-navigable, ARIA-labeled, colorblind-safe palettes (see DESIGN.md). Games must be playable without a mouse where feasible.

## Tech stack (do not substitute without asking)

| Layer | Choice |
|---|---|
| Framework | React 19 + Next.js 16 (App Router, TypeScript), every route statically generated |
| In-browser ML | TensorFlow.js, in the five games that train a model (Data Detox, Feature Forge, Neuron Forge, Overfit Tower Defense, Convolution Kitchen). The other nine compute their ML in plain TypeScript: boundary fitting, confusion-matrix metrics, CART, gradient descent, a Gaussian process, k-means, PCA, tabular Q-learning, autograd. |
| Python-in-browser | Pyodide + pandas, self-hosted under `/pyodide/` (Feature Forge's code lane) |
| Data viz and 2D | SVG, with D3 for scales and shapes |
| 3D | Three.js (Gradient Descent Skier, Dimension Diver) |
| Node editors | React Flow (Neuron Forge, Decision Tree Architect) |
| State | Zustand (per-game stores) |
| Persistence | localStorage via the `progression` service. Supabase is a tested seam that nothing calls yet. |
| Styling | Tailwind CSS v4 + tokens from DESIGN.md |
| Test | Vitest + React Testing Library; Playwright scripts for playthroughs and the phone layout; Lighthouse for accessibility |
| Package manager | bun 1.3.13, with Node 22 |

**What changed from the original table, and why.** The original table named
**Highcharts**, **p5.js** and **Phaser.js**. None was ever imported, and all
three have been **removed from `package.json`**. Highcharts is also commercially
licensed. Every chart, grid, graph and battlefield is SVG, on a rule worth keeping:
*when the numbers are the content, use SVG*. An SVG chart is in the
accessibility tree; a canvas is an opaque bitmap that would have to be built a
second time for screen readers. It is also part of why every route scores 100 on
Lighthouse accessibility, in both the desktop and the mobile pass. Lighthouse
only audits first load, though, so that score is a floor, not proof. Don't add a
canvas or game-engine dependency for a 2D view without asking.

Three.js earns its place in two views, and neither is a drag-to-orbit scene:

- **Gradient Descent Skier's loss terrain** has a fixed camera, deliberately (no
  orbit controls). A 3D / Contour toggle switches to a top-down SVG contour map of
  the same surface. The contour map is an equal view, not a fallback, and it also
  takes over when WebGL is missing or the context is lost.
- **Dimension Diver's point cloud** is rotated with three sliders (yaw, pitch,
  roll) in the rail. The judgement is made from the SVG 2D shadow and the gauges;
  without WebGL the 3D panel says so and the game still works.

Both render on demand rather than in a loop, and release their WebGL context on
unmount.

The Supabase row reflects what shipped. There is no auth UI, so
`createSupabaseAdapter` is never called, and setting `NEXT_PUBLIC_SUPABASE_*`
changes nothing about progress.

## Repo layout

```
gameml/
├── CLAUDE.md                 ← you are here
├── DESIGN.md                 ← design system + tokens
├── README.md                 ← setup, env vars, verification, deploying, known limitations
├── docs/
│   ├── GameML_Build_Spec.md  ← full product spec (source of truth for games; §9 = deviations)
│   └── ENGINE_API.md         ← shared engine API reference
├── .claude/
│   ├── agents/               ← subagent definitions
│   ├── skills/               ← reusable skills (new-game-module, tfjs-model-lifecycle)
│   ├── commands/             ← slash commands
│   └── settings.json         ← permissions + MCP allowlist
├── .mcp.json                 ← MCP server config
├── .github/workflows/ci.yml  ← typecheck, lint, 3 test shards, build + browser harnesses
├── next.config.ts            ← security headers / CSP, /pyodide cache policy
├── src/
│   ├── app/                  ← Next.js routes
│   │   ├── layout.tsx            ← fonts, metadata, <SkipLink/>
│   │   ├── page.tsx              ← the roster (grouped by category, "Start here")
│   │   ├── play/[slug]/          ← a game (static, dynamicParams = false)
│   │   ├── concepts/             ← the Concept Library index
│   │   ├── concepts/[slug]/      ← one explainer per concept (8)
│   │   ├── not-found.tsx         ← the 404 (has <main>, a way home)
│   │   ├── error.tsx             ← route-level error boundary
│   │   ├── global-error.tsx      ← last-resort boundary for a root-layout crash
│   │   ├── sitemap.ts            ← generated from the registry + Concept Library
│   │   ├── robots.ts             ← allow all, points at the sitemap
│   │   ├── icon.svg, favicon.ico, apple-icon.png
│   │   ├── _components/          ← SkipLink, ErrorPanel, RosterProgress
│   │   └── __tests__/            ← home page + app-shell tests
│   ├── engine/               ← SHARED: GameShell, LaneErrorBoundary, useModel, useCodeLane, progression
│   ├── components/           ← shared UI primitives (MetricReadout, WhyCard, ...);
│   │                           katexRender.ts is the only file that imports KaTeX
│   ├── games/                ← one folder per game (self-contained)
│   │   ├── registry.ts           ← which slugs are playable (server-safe)
│   │   ├── GameMount.tsx         ← slug → component map (client, ssr:false) + GameErrorBoundary
│   │   └── <game-slug>/
│   │       ├── index.tsx         ← entry, mounts in GameShell
│   │       ├── store.ts          ← Zustand state (the data model); most games also build the
│   │       │                       code lane's api here (createCodeApi) or in code-api.ts
│   │       ├── VisualLane.tsx
│   │       ├── CodeLane.tsx      ← useCodeLane + the editor (and the api, where store.ts doesn't build it)
│   │       ├── ml.ts             ← the ML logic (TF.js where a model trains, plain TS otherwise)
│   │       ├── why-cards.ts      ← feedback copy tied to actions
│   │       ├── <game>.test.ts
│   │       └── …                 ← game components; some games add code-api.ts or field.ts
│   └── lib/                  ← catalog, concepts, site (origin + metadata), highlight,
│                               chunk-load-error, utils, supabase seam, security-headers.test.ts
├── scripts/
│   ├── harness.mjs           ← shared route list (parsed from registry + concepts) + console rules
│   ├── verify-playthrough.mjs    ← drives each game in a real browser
│   ├── playthroughs/<slug>.mjs   ← one per game
│   ├── verify-mobile.mjs     ← every route at 360×740: overflow, landmarks, metric, console
│   ├── a11y-audit.mjs        ← Lighthouse over every route, desktop + mobile
│   ├── setup-pyodide.mjs     ← stages the Python runtime + sha256-checked wheels into public/
│   └── pyodide-smoke.mjs     ← standalone Pyodide + pandas check in Node
└── public/datasets/          ← empty placeholder (README only); every game generates its data in code
```

A new game has to be registered in **five** places or it is invisible:

1. **`src/lib/catalog.ts`**: a `GAME_CATALOG` entry (title, concept, core
   intuition, metric label, failure mode, phase, `requires`). `catalog.test.ts`
   currently pins the catalog at 14 games, so a fifteenth updates that test too.
2. **`src/games/registry.ts`**: add the slug to `PLAYABLE_SLUGS`.
3. **`src/games/GameMount.tsx`**: a `next/dynamic` entry with `ssr: false`.
4. **`src/engine/progression.ts`**: a `CONCEPT_BADGES` entry for the badge.
5. **`scripts/playthroughs/<slug>.mjs`**: the browser playthrough.

Everything else is derived. The `/play/<slug>` page comes from
`generateStaticParams`, the sitemap from the registry, and the a11y and phone
harnesses' route lists from `scripts/harness.mjs`, which parses `registry.ts`.
Never add a static `src/app/play/<slug>/page.tsx`; it would bypass the registry.

Tests enforce these:

- `registry.test.ts` fails when the registry, the catalog and the mount map
  disagree, when a `src/games/<dir>/index.tsx` isn't registered, when a playable
  game has no playthrough, or when the harness parse differs from the registry.
- `progression.test.ts` fails when a catalog slug has no badge.
- `MathDrawer.test.tsx` fails when a playable game's `ml.ts` doesn't export a
  `MATH_EQUATION` that KaTeX renders cleanly in strict mode.

Concept Library links are expected but not required. Add
`conceptHref`/`conceptLabel` to a WhyCard only where an existing concept section
heading matches `conceptLabel` exactly, and list the slug in that concept's
`games` array. `concepts.test.ts` fails on a link that 404s, a label with no
matching heading, or a concept that doesn't list the game back. It also pins
the game figures the explainers quote against the games' own code (Data
Detox's are pinned in `data-detox.test.ts`), so a game change that moves a
quoted number fails there first. Today 11 of the
14 games link in. Agent Academy, Confusion Matrix Chef and Dimension Diver don't,
because no concept page covers them yet. The coverage test only requires the
Phase-1 games.

## Workflow for adding a game

1. Read the game's entry in `docs/GameML_Build_Spec.md`.
2. Invoke the **new-game-module** skill (`.claude/skills/new-game-module`).
3. Scaffold under `src/games/<slug>/` following the layout above.
4. Implement `store.ts` from the spec's data model → `ml.ts` → `VisualLane.tsx` → `CodeLane.tsx` → `why-cards.ts`.
5. Verify the pedagogy contract (all 4 points) in code.
6. Write tests: ML logic correctness + the live-feedback binding.
7. Wire XP/stars into the `progression` service. Credit the third star by the
   lane the clearing action came from. The store's scoring action takes a
   `source` argument (the code lane's api passes `"code"`, the visual lane's
   button passes nothing), and it calls
   `recordResult({ lane: source, codeLaneCleared: source === "code" })`. Which
   tab is visible must not decide it. `sort-it-arcade/store.ts` `check()` is the
   reference.
8. Register the game in the five places above, and write its playthrough.
9. Run `/verify-game <slug>` before marking done.

## Definition of Done (per game)

- [ ] Both lanes implemented and share one state store.
- [ ] Pedagogy contract: all 4 points demonstrable.
- [ ] Live metric updates within ~1s of player action.
- [ ] Named failure mode on loss.
- [ ] "Why did that happen?" cards wired to the last action.
- [ ] "Reveal the Math" toggle shows the real equation + code.
- [ ] Keyboard + ARIA accessible; colorblind-safe.
- [ ] ML logic unit-tested; no server calls.
- [ ] XP/stars/badges wired to progression service; the third star comes from a
      clear made through the code lane's api.
- [ ] Registered in all five places; `bun run verify` is green.
- [ ] `bun run verify:play <slug>` passes. That includes no console errors, no
      bug-shaped warnings (KaTeX, React), no `NaN`/`undefined` in a live region,
      and the shared Math-dialog check (`checkMathDialog` in
      `scripts/harness.mjs`): KaTeX renders the equation with MathML, focus is
      trapped, Escape closes it, and focus returns to the Math button.
- [ ] `bun run verify:mobile /play/<slug>` passes: no sideways scroll at 360 px,
      and the metric is on screen.
- [ ] Lighthouse a11y ≥ 95 in **both** the desktop and the mobile pass of
      `bun run audit:a11y`. Lighthouse only sees first load, so check the open
      Math drawer's contrast, the code-lane error state and the named failure
      by hand (`verify:play` covers the drawer's behaviour, not its contrast).

## Guardrails / never do

- ❌ Never add a server-side ML inference endpoint.
- ❌ Never ship a game with only one lane.
- ❌ Never hardcode a metric — it must be computed from real (client-side) ML.
- ❌ Never fork the shared engine components per-game.
- ❌ Never commit secrets. The app needs none today. Any future keys go in `.env.local` (gitignored). The Supabase service-role key must never reach client code, and since nothing reads it today, don't set it on a deploy.
- ❌ Never fetch or import from a third-party origin without changing the CSP in `next.config.ts` (`security-headers.test.ts` fails on an absolute URL in app code).
- ❌ Never bypass the pedagogy contract to "make it fun" — fun that teaches nothing is out of scope.

## Commit & PR conventions

- Conventional Commits (`feat(games): add k-means territory wars`).
- One game per PR where possible. PR description must list which pedagogy-contract points are satisfied and where.
- Run `/verify-game` and tests before requesting review.

## When unsure

The spec (`docs/GameML_Build_Spec.md`) is the source of truth for *what* each game teaches and its data model. This file is the source of truth for *how* we build. DESIGN.md is the source of truth for *how it looks*. If they conflict, ask rather than guess.
