---
description: Run full ML + design + a11y verification on a game
argument-hint: <game-slug>
---

Verify the GameML game: **$ARGUMENTS**

Prerequisites: Chromium for Playwright (installed once with `bunx playwright install chromium`), and a production server. Run `bun run build && bun run start`, which serves on `http://localhost:3000`; set `AUDIT_BASE` if you use another port.

Run and collect reports:
1. `bun run test:run src/games/$ARGUMENTS` — the game's unit tests. Then `bun run verify` (tsc + eslint + the whole suite), because the registry, badge, KaTeX-equation and concept-link tests live outside the game folder.
2. `bun run verify:play $ARGUMENTS` — the browser playthrough. This is what proves the game is wired to the screen. It fails on the game's own checks, on the shared Math-dialog check (KaTeX rendered, focus trapped, Escape closes it, focus returns), and on any console error, bug-shaped warning (KaTeX, React) or `NaN`/`undefined` reaching a live region.
3. `bun run verify:mobile /play/$ARGUMENTS` — the phone layout at 360×740. It fails on sideways scroll, a forced-wide viewport, console errors, a wrong status, a missing `<main>`/`<h1>`, or a live metric that never appears. It warns on targets under 24px; review each warning.
4. `bun run audit:a11y http://localhost:3000/play/$ARGUMENTS` — Lighthouse accessibility. It must score ≥ 95 in both the desktop and the mobile pass.
5. In parallel, the review subagents:
   - `ml-verifier` — is the ML real, does player-action mirror the algorithm, is the failure mode named correctly, does the math reveal match the code, do the stars match the copy?
   - `design-reviewer` — DESIGN.md token adherence, GameShell anatomy, the "aha" moment, and the accessibility checks Lighthouse can't do (the Math drawer, the code-lane error and named-failure states, drag alternatives, target sizes). It can use the Playwright MCP if connected.

Then check the CLAUDE.md **Definition of Done** checklist for this game.

Output a single consolidated PASS/FAIL report, citing the harness output, with concrete fixes for every FAIL. Do not edit code in this command — hand fixes to `game-builder`.
