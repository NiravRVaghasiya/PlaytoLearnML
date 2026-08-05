---
description: Scaffold and implement a new GameML game from its spec entry
argument-hint: <game-name-or-slug>
---

Build the GameML game: **$ARGUMENTS**

1. Open `docs/GameML_Build_Spec.md` and find this game's entry.
2. Use the `new-game-module` skill and delegate implementation to the `game-builder` subagent.
3. Follow the strict build order: store → ml → VisualLane → CodeLane → why-cards → index → tests.
4. Enforce: two-lane rule, client-side ML only, shared engine reuse, DESIGN.md tokens.
5. When built, run `/verify-game $ARGUMENTS`.

Report: files created, which pedagogy-contract points are satisfied (with line refs), and open questions.
