---
description: Extend or harden the shared engine layer (it is already built)
---

The GameML shared engine already exists and all 14 games run on it. Use the `engine-architect` subagent for any change to it, and do not rebuild it from scratch.

The engine, for reference:
1. `src/components/` primitives: MetricReadout, WhyCard, LaneToggle, MathDrawer (+ katexRender), XPBar, StarRating, Slider, Dial, DatasetChip, Button, CodeEditor, CodeBlock (DESIGN.md §6).
2. `src/engine/GameShell.tsx` (+ `LaneErrorBoundary.tsx`) — the fixed layout (DESIGN.md §5), the lane toggle, the Math drawer host, and the failure and WhyCard announcers.
3. `src/engine/useModel.ts` — TF.js lifecycle (see the `tfjs-model-lifecycle` skill), leak-free, latest-wins training.
4. `src/engine/useCodeLane.ts` — binds an editable snippet to shared state; JS and Pyodide executors.
5. `src/engine/progression.ts` — XP/stars/badges/unlocks. It persists to localStorage; Supabase is an unwired seam.

Constraints:
- no game-specific logic in the engine;
- changes are additive and typed;
- the accessibility contract and the harness hooks (`data-testid`s, `#game-canvas + div`) are kept;
- `docs/ENGINE_API.md` is updated in the same change;
- `bun run verify` is green;
- and, for anything visual, `bun run verify:play` and `bun run verify:mobile` pass against a production build.
