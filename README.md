# 🎮 GameML — Vibe Coding Kit

Everything an AI coding agent (Claude Code, Cursor, Windsurf) needs to build **GameML** — an interactive site that teaches Machine Learning through games — with minimal hand-holding. Drop this at the root of your repo and start vibe coding.

## What's inside

```
vibecodingkit/
├── CLAUDE.md                       ← agent operating manual (rules, guardrails, DoD)
├── DESIGN.md                       ← design system (tokens, GameShell anatomy, a11y)
├── README.md                       ← you are here
├── .mcp.json                       ← MCP servers (filesystem, git, github, playwright, supabase, context7)
├── docs/
│   └── GameML_Build_Spec.md        ← full product spec: 14 games, data models, roadmap
└── .claude/
    ├── settings.json               ← permissions + MCP allowlist (safe defaults)
    ├── agents/                     ← subagents
    │   ├── engine-architect.md     ← builds the shared engine (do this first)
    │   ├── game-builder.md         ← implements one game end-to-end
    │   ├── ml-verifier.md          ← checks the ML is real + honest
    │   └── design-reviewer.md      ← checks DESIGN.md + accessibility
    ├── skills/
    │   ├── new-game-module/        ← scaffold a game from its spec entry (+ template)
    │   └── tfjs-model-lifecycle/   ← memory-safe TF.js patterns
    └── commands/                   ← slash commands
        ├── scaffold-engine.md      ← /scaffold-engine
        ├── new-game.md             ← /new-game <name>
        └── verify-game.md          ← /verify-game <slug>

```

## The four load-bearing ideas

1. **Pedagogy contract** — every game must have: a core intuition, player-action = algorithm, always-visible live feedback, and a *named* failure mode. Enforced by `CLAUDE.md` + `ml-verifier`.
2. **Two-lane rule** — every game ships a visual (no-code) lane AND a code lane sharing one state store.
3. **Client-side ML only** — TensorFlow.js + Pyodide in-browser. No ML server → cheap to host.
4. **Shared engine first** — `GameShell`, `useModel`, `useCodeLane`, `progression` built once, reused by all 14 games.

## Quick start

```bash
# 1. Copy this kit to your repo root, then open Claude Code there.
# 2. Trust the project MCP servers when prompted (see .claude/settings.json).

# 3. Build the foundation:
/scaffold-engine

# 4. Build Phase-1 games (per the spec roadmap):
/new-game Sort-It Arcade
/new-game K-Means Territory Wars
/new-game Data Detox
/new-game Gradient Descent Skier

# 5. Verify each:
/verify-game gradient-descent-skier

```

## Build order (from the spec roadmap)

- **Phase 1 (MVP):** Sort-It Arcade · K-Means Territory Wars · Data Detox · Gradient Descent Skier
- **Phase 2:** Neuron Forge · Overfit Tower Defense · Confusion Matrix Chef · Decision Tree Architect
- **Phase 3:** Hyperparameter Heist · Feature Forge · Agent Academy · Convolution Kitchen
- **Phase 4:** Backprop Blitz · Dimension Diver · (bonus) GAN Duel

Always build the shared engine (`/scaffold-engine`) before any game.

## Setup notes

- **Env vars** (never commit): `GITHUB_TOKEN`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`. Put them in `.env.local` (gitignored). `settings.json` denies reading `.env*`.
- **MCP servers** run via `npx` — Node 18+ required. `github` and `supabase` are gated behind "ask" permissions by default.
- **Playwright MCP** powers `/verify-game`'s accessibility + "does the metric actually move" checks.

## Read order for a new agent

1. `CLAUDE.md` (how we build) → 2. `DESIGN.md` (how it looks) → 3. `docs/GameML_Build_Spec.md` (what each game teaches). If they ever conflict: CLAUDE = how, DESIGN = look, SPEC = what. Ask rather than guess.

