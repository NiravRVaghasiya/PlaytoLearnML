---
name: new-game-module
description: Scaffold and implement a new GameML game module from its spec entry, following the fixed file layout, two-lane rule, and pedagogy contract. Use whenever adding a game under src/games/.
---

# New Game Module

## Overview

This skill turns a game's entry in `docs/GameML_Build_Spec.md` into a working, tested module under `src/games/<slug>/`. It enforces the two-lane rule (visual + code sharing one state store), the pedagogy contract (core intuition, player-action=algorithm, live feedback, named failure mode), and client-side-only ML. Use it for every new game so all 14 modules stay structurally identical and reuse the shared engine.

## Workflow

1. **Locate the spec.** Open `docs/GameML_Build_Spec.md`, find the game, and extract: concept, core intuition, player-action=algorithm, gameplay loop, win/lose/scoring, live feedback, data model, components, ML logic, tech.
2. **Pick the slug.** kebab-case (e.g. `k-means-territory-wars`). Create `src/games/<slug>/`.
3. **Store first (`store.ts`).** Implement the spec's data model as a typed Zustand store. This is the single source of truth both lanes share.
4. **ML logic (`ml.ts`).** Implement the real computation that produces the live metric. TF.js or pure-JS per the spec's "Suggested Tech". No server calls. Export pure, testable functions.
5. **Visual lane (`VisualLane.tsx`).** No-code interaction (drag/click/slider) bound to the store; use the library from the spec (p5/D3/Three/Phaser/React Flow).
6. **Code lane (`CodeLane.tsx`).** Editable snippet (TF.js/Pyodide) that drives the SAME store state via `useCodeLane`.
7. **Why-cards (`why-cards.ts`).** Map player actions + failure modes → 2-line plain-language explanations. Wire to `<WhyCard>`.
8. **Entry (`index.tsx`).** Mount inside `<GameShell>`; wire lane toggle, `<MathDrawer>` (equation + real code), and XP/stars via `progression`.
9. **Tests (`<slug>.test.ts`).** Unit-test `ml.ts` correctness + that the metric updates when the store changes.
10. **Verify.** Run the pedagogy contract check and the CLAUDE.md Definition of Done. Run `/verify-game <slug>`.

## Reference

- File layout, tech stack, and Definition of Done: `CLAUDE.md`.
- Design tokens + GameShell anatomy: `DESIGN.md`.
- TF.js model lifecycle patterns: the `tfjs-model-lifecycle` skill.
- See `template.md` in this skill folder for a copy-paste module skeleton.
