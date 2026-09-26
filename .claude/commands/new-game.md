---
description: Scaffold and implement a new GameML game from its spec entry
argument-hint: <game-name-or-slug>
---

Build the GameML game: **$ARGUMENTS**

1. Open `docs/GameML_Build_Spec.md` and find this game's entry (and read §9, the deviations shipped games made and why).
2. Use the `new-game-module` skill and delegate implementation to the `game-builder` subagent.
3. Follow the strict build order: store → ml → VisualLane → CodeLane → why-cards → index → **registration** → playthrough → tests.
4. Register the game in all five places, or it is invisible:
   - `src/lib/catalog.ts` (and the 14-game count in `catalog.test.ts`);
   - `src/games/registry.ts`;
   - `src/games/GameMount.tsx`;
   - `CONCEPT_BADGES` in `src/engine/progression.ts`;
   - `scripts/playthroughs/<slug>.mjs`.

   Do not add a static `src/app/play/<slug>/page.tsx`.
5. Enforce: two-lane rule, client-side ML only, shared engine reuse, DESIGN.md tokens, SVG for 2D views, and third-star credit from the code lane's api (not the visible tab).
6. Run `bun run verify` until it is green. It includes the registry, badge, KaTeX-equation and concept-link tests.
7. When built, run `/verify-game $ARGUMENTS`.

Report: files created, commands run and their results, which pedagogy-contract points are satisfied (with line refs), and open questions.
