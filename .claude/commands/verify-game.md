---
description: Run full ML + design + a11y verification on a game
argument-hint: <game-slug>
---

Verify the GameML game: **$ARGUMENTS**

Run in parallel and collect reports:
1. `ml-verifier` subagent — is the ML real, does player-action mirror the algorithm, is the failure mode named correctly, does the math reveal match the code?
2. `design-reviewer` subagent — DESIGN.md token adherence, GameShell anatomy, the "aha" moment, and accessibility (Lighthouse a11y ≥ 95) via the Playwright MCP.
3. Run the test suite for `src/games/$ARGUMENTS/`.

Then check the CLAUDE.md **Definition of Done** checklist for this game.

Output a single consolidated PASS/FAIL report with concrete fixes for every FAIL. Do not edit code in this command — hand fixes to `game-builder`.
