# Kickoff Prompt — paste into Claude Code / Cursor

> Open this folder as your project root first, then paste the prompt below into the agent. It orients the agent to the kit and starts the Phase-1 build. Copy everything inside the fenced block.

---

```
You are the lead engineer building GameML — an interactive website that teaches
Machine Learning through games. This repo already contains a complete "vibe coding
kit". Before writing any code, read these in order and treat them as authoritative:

1. CLAUDE.md              — how we build (rules, guardrails, Definition of Done)
2. DESIGN.md              — how it looks (tokens, GameShell anatomy, accessibility)
3. docs/GameML_Build_Spec.md — what we build (14 games, data models, roadmap)

Also load the available subagents (.claude/agents/), skills (.claude/skills/),
slash commands (.claude/commands/), and MCP servers (.mcp.json).

Non-negotiable rules (from CLAUDE.md):
- Pedagogy contract: every game needs a core intuition, player-action = algorithm,
  always-visible live feedback, and a NAMED failure mode.
- Two-lane rule: every game ships a visual (no-code) lane AND a code lane sharing
  ONE state store.
- Client-side ML only (TensorFlow.js + Pyodide). No ML server. No GPU dependency.
- Build the shared engine ONCE and reuse it; never fork engine components per game.
- Accessibility: keyboard nav, ARIA live region on the metric, colorblind-safe.

Your task, in this order:

STEP 0 — Confirm understanding.
Summarize the pedagogy contract, the two-lane rule, and the Phase-1 game list back
to me in <10 lines. Ask me any blocking questions. Do NOT scaffold yet — wait for
my "go".

STEP 1 — Project scaffold.
On my "go": initialize a Next.js (App Router, TypeScript) + Tailwind project,
install the stack from CLAUDE.md (tensorflow/tfjs, zustand, d3, three, phaser,
reactflow, highcharts, @supabase/supabase-js), and create the repo layout from
CLAUDE.md. Add .gitignore, .env.example (GITHUB_TOKEN, SUPABASE_URL,
SUPABASE_SERVICE_ROLE_KEY), and wire Tailwind to the DESIGN.md color/type tokens.

STEP 2 — Shared engine.
Run /scaffold-engine (engine-architect subagent): build src/components primitives,
src/engine/GameShell.tsx, useModel.ts (memory-safe per the tfjs-model-lifecycle
skill), useCodeLane.ts, and progression.ts. Produce a short API reference.

STEP 3 — Phase-1 games (one PR each, in this order):
  /new-game Sort-It Arcade
  /new-game K-Means Territory Wars
  /new-game Data Detox
  /new-game Gradient Descent Skier
For each: follow the new-game-module skill (store → ml → VisualLane → CodeLane →
why-cards → index → tests), then run /verify-game <slug> (ml-verifier +
design-reviewer + tests + Lighthouse a11y >= 95). Do not mark a game done until
all four pedagogy-contract points are demonstrable in code and the Definition of
Done checklist passes.

Work incrementally. After each step, stop and report what you built, which
pedagogy-contract points are satisfied (with file+line refs), and what's next.
Never commit secrets. Ask before pushing or touching GitHub/Supabase.

```

---

## Shorter variant (for a quick start)

```
Read CLAUDE.md, DESIGN.md, and docs/GameML_Build_Spec.md — they're authoritative.
Load the .claude/ agents, skills, and commands and the .mcp.json servers.

Confirm you understand the pedagogy contract, the two-lane rule, and client-side-ML-only,
and list the Phase-1 games. Then wait for my "go".

On "go": run /scaffold-engine, then build the Phase-1 games one at a time with
/new-game, verifying each with /verify-game. Report after every step; never commit secrets.

```

## Tips

- In **Cursor**, add CLAUDE.md, DESIGN.md, and the spec to context (@-mention them) so they stay pinned.
- In **Claude Code**, CLAUDE.md loads automatically; the /commands and subagents are picked up from `.claude/`.
- Trust the project MCP servers when prompted (see `.claude/settings.json` — github/supabase are gated behind "ask").
- Keep PRs to one game each; let `/verify-game` gate merges.

```


```

