# 🎮 GameML

An interactive site that teaches Machine Learning by playing it. Fourteen games
across six ML categories, and **every model trains in the browser** — no ML
server, no GPU, nothing to install to play.

All 14 games are built. There is also a Concept Library of short explainers that
the games link into from their feedback cards.

## Running it

The lockfile is `bun.lock`, so `bun` is the expected package manager.

```bash
bun install
bun run dev          # http://localhost:3000 — also primes the Pyodide runtime
bun run build        # production build
bun run start        # serve the production build
```

`predev` and `prebuild` copy the Pyodide runtime into `public/pyodide`, which is
what makes the Python code lane work offline. If `import pandas` ever fails,
`bun run verify:pyodide` re-primes the wheels.

## Verifying it

Three layers, because they catch different things. Unit tests prove the ML is
correct; the playthroughs prove it is wired to the screen; Lighthouse proves it
is reachable.

```bash
bun run verify       # tsc --noEmit && eslint . && vitest run
bun run build && bun run start   # the two below need a running server

bun run verify:play  # drives all 14 games in a real browser
bun run audit:a11y   # Lighthouse accessibility on every route
```

Both browser harnesses read `AUDIT_BASE` (default `http://localhost:3000`), so
point them at whichever port you started:

```bash
AUDIT_BASE=http://localhost:3001 bun run verify:play sort-it-arcade
```

`verify:play` exists because Sort-It Arcade once shipped two silent pedagogical
bugs that every unit test passed. It asserts the metric actually moves, that a
loss names a real ML failure mode, that both lanes share one store, and that
nothing like `NaN` reaches a screen reader.

## The four load-bearing ideas

1. **Pedagogy contract** — every game has a core intuition, player-action =
   algorithm, always-visible live feedback, and a *named* failure mode.
2. **Two-lane rule** — every game ships a visual (no-code) lane and a code lane
   sharing one state store.
3. **Client-side ML only** — TensorFlow.js and Pyodide in-browser. No inference
   endpoint, which is what makes hosting cheap.
4. **Shared engine** — `GameShell`, `useModel`, `useCodeLane` and `progression`
   are built once and reused, never forked per game.

## Layout

```
├── CLAUDE.md                     ← how we build: rules, guardrails, DoD
├── DESIGN.md                     ← how it looks: tokens, shell anatomy, a11y
├── docs/
│   ├── GameML_Build_Spec.md      ← what we build: the 14 games and their models
│   └── ENGINE_API.md             ← shared engine API reference
├── src/
│   ├── app/                      ← routes: /, /play/[slug], /concepts/[slug]
│   ├── engine/                   ← SHARED: GameShell, useModel, useCodeLane, progression
│   ├── components/               ← shared primitives (MetricReadout, WhyCard, …)
│   ├── games/<slug>/             ← one self-contained folder per game
│   │                               store · ml · VisualLane · CodeLane · why-cards · tests
│   └── lib/                      ← catalog, concepts, utils, supabase client
├── scripts/                      ← a11y audit, playthrough runner, pyodide setup
└── .claude/                      ← agents, skills and slash commands for adding a game
```

## Docs, in reading order

1. **`CLAUDE.md`** — how we build. The pedagogy contract, the golden rules and
   the Definition of Done. Cited throughout the code.
2. **`DESIGN.md`** — how it looks. Colour and type tokens, `GameShell` anatomy,
   and the accessibility requirements. The most-cited document in the codebase;
   code comments reference its sections by number.
3. **`docs/GameML_Build_Spec.md`** — what each game teaches. `src/lib/catalog.ts`
   is a transcription of it, and the spec wins if they disagree.
4. **`docs/ENGINE_API.md`** — the shared engine's API, for extending it.

If they conflict: CLAUDE = how, DESIGN = look, SPEC = what. Ask rather than guess.

## Adding a game

The spec carries one unbuilt stretch entry, GAN Duel, and `.claude/` still holds
the tooling for it: the `new-game-module` skill, the `game-builder`, `ml-verifier`
and `design-reviewer` subagents, and `/new-game` + `/verify-game`. A game is not
done until all four pedagogy-contract points are demonstrable in code and the
Definition of Done in `CLAUDE.md` passes.

## Setup notes

- Secrets never get committed. `SUPABASE_*` and `GITHUB_TOKEN` go in `.env.local`,
  which is gitignored; see `.env.example`. Progression falls back to
  `localStorage`, so the app runs with no Supabase project at all.
- MCP servers are configured in `.mcp.json`; `github` and `supabase` are gated
  behind "ask" in `.claude/settings.json`.

## Provenance

The 14-game catalog came out of an ideation prompt, which produced
`docs/GameML_Build_Spec.md`, which in turn produced this kit's `CLAUDE.md`,
`DESIGN.md` and `.claude/` tooling. The original ideation and kickoff prompts
were removed once the build was complete — they described work that is now done,
and are recoverable from git history if the provenance is ever needed.
