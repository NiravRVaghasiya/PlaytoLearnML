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
| Framework | React + Next.js (App Router, TypeScript) |
| In-browser ML | TensorFlow.js |
| Python-in-browser | Pyodide (code lane) |
| Data viz | D3.js + Highcharts |
| 2D canvas | p5.js |
| 3D | Three.js |
| Game engine | Phaser.js |
| Node editors | React Flow |
| State | Zustand (per-game stores) |
| Backend | Supabase (auth, XP, leaderboards) |
| Styling | Tailwind CSS + tokens from DESIGN.md |
| Test | Vitest + React Testing Library; Playwright for e2e |

## Repo layout

```
gameml/
├── CLAUDE.md                 ← you are here
├── DESIGN.md                 ← design system + tokens
├── docs/
│   └── GameML_Build_Spec.md  ← full product spec (source of truth for games)
├── .claude/
│   ├── agents/               ← subagent definitions
│   ├── skills/               ← reusable skills (new-game-module, tfjs-model-lifecycle)
│   ├── commands/             ← slash commands
│   └── settings.json         ← permissions + MCP allowlist
├── .mcp.json                 ← MCP server config
├── src/
│   ├── app/                  ← Next.js routes
│   ├── engine/               ← SHARED: GameShell, useModel, useCodeLane, progression
│   ├── components/           ← shared UI primitives (MetricReadout, WhyCard, ...)
│   ├── games/                ← one folder per game (self-contained)
│   │   └── <game-slug>/
│   │       ├── index.tsx         ← entry, mounts in GameShell
│   │       ├── store.ts          ← Zustand state (the data model)
│   │       ├── VisualLane.tsx
│   │       ├── CodeLane.tsx
│   │       ├── ml.ts             ← the ML logic (TF.js)
│   │       ├── why-cards.ts      ← feedback copy tied to actions
│   │       └── <game>.test.ts
│   └── lib/                  ← utils, supabase client, datasets
└── public/datasets/          ← Titanic, Iris, MNIST subsets
```

## Workflow for adding a game

1. Read the game's entry in `docs/GameML_Build_Spec.md`.
2. Invoke the **new-game-module** skill (`.claude/skills/new-game-module`).
3. Scaffold under `src/games/<slug>/` following the layout above.
4. Implement `store.ts` from the spec's data model → `ml.ts` → `VisualLane.tsx` → `CodeLane.tsx` → `why-cards.ts`.
5. Verify the pedagogy contract (all 4 points) in code.
6. Write tests: ML logic correctness + the live-feedback binding.
7. Wire XP/stars into the `progression` service.
8. Run `/verify-game <slug>` before marking done.

## Definition of Done (per game)

- [ ] Both lanes implemented and share one state store.
- [ ] Pedagogy contract: all 4 points demonstrable.
- [ ] Live metric updates within ~1s of player action.
- [ ] Named failure mode on loss.
- [ ] "Why did that happen?" cards wired to the last action.
- [ ] "Reveal the Math" toggle shows the real equation + code.
- [ ] Keyboard + ARIA accessible; colorblind-safe.
- [ ] ML logic unit-tested; no server calls.
- [ ] XP/stars/badges wired to progression service.
- [ ] Lighthouse a11y ≥ 95; no console errors.

## Guardrails / never do

- ❌ Never add a server-side ML inference endpoint.
- ❌ Never ship a game with only one lane.
- ❌ Never hardcode a metric — it must be computed from real (client-side) ML.
- ❌ Never fork the shared engine components per-game.
- ❌ Never commit secrets. Supabase keys go in `.env.local` (gitignored).
- ❌ Never bypass the pedagogy contract to "make it fun" — fun that teaches nothing is out of scope.

## Commit & PR conventions

- Conventional Commits (`feat(games): add k-means territory wars`).
- One game per PR where possible. PR description must list which pedagogy-contract points are satisfied and where.
- Run `/verify-game` and tests before requesting review.

## When unsure

The spec (`docs/GameML_Build_Spec.md`) is the source of truth for *what* each game teaches and its data model. This file is the source of truth for *how* we build. DESIGN.md is the source of truth for *how it looks*. If they conflict, ask rather than guess.
