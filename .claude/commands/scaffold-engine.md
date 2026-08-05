---
description: Build the shared engine layer before any games
---

Build the GameML shared engine using the `engine-architect` subagent.

Deliver, in order:
1. `src/components/` primitives: MetricReadout, WhyCard, LaneToggle, MathDrawer, XPBar, StarRating, Slider, Dial, DatasetChip (DESIGN.md §6).
2. `src/engine/GameShell.tsx` — fixed layout (DESIGN.md §5), lane toggle + ƒ Math drawer host.
3. `src/engine/useModel.ts` — TF.js lifecycle (see `tfjs-model-lifecycle` skill), leak-free.
4. `src/engine/useCodeLane.ts` — binds editable snippet to shared state.
5. `src/engine/progression.ts` — XP/stars/badges/unlocks via Supabase.

Constraints: no game-specific logic in the engine; typed + documented APIs; accessibility contract met. Output a short API reference for `game-builder` to consume.
