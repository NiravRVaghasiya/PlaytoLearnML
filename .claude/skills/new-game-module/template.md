# Game module skeleton

Copy this structure into `src/games/<slug>/`. Replace `<Slug>` / `<slug>` / `<Game Name>`.
It is a skeleton, not a game: every `…` is yours to fill from the spec. The real,
complete reference is `src/games/sort-it-arcade/`; check signatures against
`docs/ENGINE_API.md`.

## ml.ts
```ts
// Real, client-side ML. Pure + testable. No server calls.
// TF.js only if a model actually trains (see the tfjs-model-lifecycle skill).
import { seededRandom } from "@/lib/utils";

export function generateData(seed: number) {
  const random = seededRandom(seed);          // reproducible rounds
  // …
}

export function computeMetric(/* … */): number {
  // accuracy / loss / inertia — computed for real, never hardcoded
}

export interface Evaluation {
  outcome: "win" | "failure" | "near-miss";
  score: number;                               // normalised 0–1 for progression
  failure: { name: string; detail: string } | null;   // NAMED failure, with numbers
}

export function evaluate(/* … */): Evaluation { /* … */ }

// Required by MathDrawer.test.tsx (it globs every src/games/*/ml.ts).
export const MATH_EQUATION = String.raw`…`;   // multi-line: separate lines with \\
export const MATH_CODE = `…`;                  // copied from the functions above
export const MATH_NOTES = "…";                 // what the symbols mean, plainly
```

## store.ts
```ts
"use client";

import { create } from "zustand";
import { useProgression } from "@/engine/progression";
import type { Lane, NamedFailure } from "@/engine/types";
import type { WhyCardContent } from "@/components";
import { evaluate, type Evaluation } from "./ml";
import { whyCardFor } from "./why-cards";

export const SLUG = "<slug>";

// Mirror the spec's Data Model; note any deliberate extension in a comment.
interface <Slug>State {
  metric: number;                    // the live metric
  failure: NamedFailure | null;      // named ML failure when lost
  whyCard: WhyCardContent | null;
  lane: Lane;
  won: boolean;
  clearedFrom: Lane | null;          // record a clear once per board
  setLane: (lane: Lane) => void;
  step: () => void;                  // a player action = an algorithm step
  check: (source?: Lane) => Evaluation;
  reset: () => void;
}

export const use<Slug>Store = create<<Slug>State>((set, get) => ({
  metric: 0,
  failure: null,
  whyCard: null,
  lane: "visual",
  won: false,
  clearedFrom: null,
  setLane: (lane) => set({ lane }),
  step: () => {
    /* apply the action → recompute via ml.ts → set metric + a new whyCard key;
       any edit that changes the board resets clearedFrom to null */
  },
  // `source` is where the ACTION came from: the code lane's api passes "code".
  // The visible tab must not decide the third star.
  check: (source = "visual") => {
    const evaluation = evaluate(/* … */);
    const win = evaluation.outcome === "win";
    const { clearedFrom } = get();
    const record =
      win && (clearedFrom === null || (source === "code" && clearedFrom !== "code"));
    set({
      failure: evaluation.failure,
      won: win,
      clearedFrom: win ? (record ? source : clearedFrom) : null,
      whyCard: whyCardFor({ kind: "checked", evaluation }),
    });
    if (record) {
      useProgression.getState().recordResult({
        slug: SLUG,
        score: evaluation.score,
        lane: source,
        completed: true,
        codeLaneCleared: source === "code",
      });
    }
    return evaluation;
  },
  reset: () => set({ metric: 0, failure: null, whyCard: null, won: false, clearedFrom: null }),
}));

// The code lane's api, testable without React. Small, verb-shaped, validated.
export function createCodeApi() {
  const store = use<Slug>Store;
  return {
    step: () => store.getState().step(),
    metric: () => store.getState().metric,
    check: () => store.getState().check("code"),
    // setK: (k: unknown) => { if (!Number.isInteger(k)) throw new RangeError(`…`); … },
  };
}
```

## why-cards.ts
```ts
import type { WhyCardContent } from "@/components";

// A new `key` per action replays the card and triggers the headline announcement.
// On a loss the title IS the named failure. Only add conceptHref/conceptLabel
// when conceptLabel equals a section heading in src/lib/concepts.ts exactly,
// and add this slug to that concept's `games` array.
export function whyCardFor(event: /* … */): WhyCardContent { /* … */ }
```

## VisualLane.tsx
```tsx
"use client";
// SVG (+ D3 scales) for 2D. Read and write the SAME store as the code lane.
// Every drag also has a keyboard path and a single-pointer path (tap, or − / +).
// Map pointers with svg.getScreenCTM() — see sort-it-arcade/field.ts.
export function VisualLane() { /* … */ }
```

## CodeLane.tsx
```tsx
"use client";

import { useMemo } from "react";
import { Play, RotateCcw } from "lucide-react";
import { Button, CodeEditor } from "@/components";
import { useCodeLane } from "@/engine/useCodeLane";
import { createCodeApi } from "./store";

export const STARTER_CODE = `// Drives the same game state as the visual lane.
api.step();
log('metric', api.metric());
const result = api.check();
log('verdict', result.outcome);`;

export function CodeLane() {
  const api = useMemo(() => createCodeApi(), []);
  // maxRunMs: size for the slowest honest run (TF.js CPU backend is slow).
  const lane = useCodeLane({ initialCode: STARTER_CODE, api, maxRunMs: 6000 });

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="primary" size="sm" onClick={() => void lane.run()}
          disabled={lane.running} icon={<Play className="size-4" />}>
          {lane.running ? "Running…" : "Run"}
        </Button>
        <Button variant="ghost" size="sm" onClick={lane.reset}
          icon={<RotateCcw className="size-4" />}>
          Restore snippet
        </Button>
      </div>
      <CodeEditor label="<Game Name> script" value={lane.code} onChange={lane.setCode}
        language="javascript" error={lane.error} hint="api.step · api.metric · api.check" />
      <section aria-label="Script output">
        <h2 className="mb-1 text-xs font-semibold uppercase text-text-muted">Output</h2>
        {/* render lane.logs */}
      </section>
    </div>
  );
}
```

## index.tsx
```tsx
"use client";

import { useEffect } from "react";
import type { MetricSpec } from "@/components";
import { GameShell } from "@/engine/GameShell";
import { HIGH_SCORE_THRESHOLD, levelFromXp, useProgression } from "@/engine/progression";
import { getGameMeta } from "@/lib/catalog";
import { MATH_CODE, MATH_EQUATION, MATH_NOTES } from "./ml";
import { SLUG, use<Slug>Store } from "./store";
import { VisualLane } from "./VisualLane";
import { CodeLane } from "./CodeLane";

const STAR_CRITERIA = [
  "…complete the objective",
  `Score ${Math.round(HIGH_SCORE_THRESHOLD * 100)}% or better`,
  "Clear it from the code lane",
];

function Controls() { /* sliders / dials / buttons from @/components */ return null; }

export default function <Slug>() {
  const meta = getGameMeta(SLUG);
  const metricValue = use<Slug>Store((s) => s.metric);
  const failure = use<Slug>Store((s) => s.failure);
  const whyCard = use<Slug>Store((s) => s.whyCard);
  const lane = use<Slug>Store((s) => s.lane);
  const setLane = use<Slug>Store((s) => s.setLane);
  const reset = use<Slug>Store((s) => s.reset);

  const hydrate = useProgression((s) => s.hydrate);
  const hydrated = useProgression((s) => s.hydrated);
  const lastGain = useProgression((s) => s.lastGain);
  const games = useProgression((s) => s.games);
  const xp = useProgression((s) => s.xp);           // subscribe, then derive
  const level = levelFromXp(xp);
  useEffect(() => {
    if (!hydrated) void hydrate();
  }, [hydrated, hydrate]);

  const metric: MetricSpec = {
    label: "Accuracy",
    value: metricValue,              // NaN = "not measured yet" (renders —)
    format: "percent",
    goodDirection: "up",
  };

  return (
    <GameShell
      slug={SLUG}
      title={meta?.title ?? "<Game Name>"}
      metric={metric}
      math={{ equation: MATH_EQUATION, code: MATH_CODE, codeLanguage: "javascript", notes: MATH_NOTES }}
      controls={<Controls />}
      visual={<VisualLane />}
      code={<CodeLane />}
      whyCard={whyCard}
      failure={failure}
      lane={lane}
      onLaneChange={setLane}
      progress={{
        level: level.level,
        xpIntoLevel: level.xpIntoLevel,
        xpForNextLevel: level.xpForNextLevel,
        stars: games[SLUG]?.stars ?? 0,
        recentGain: lastGain,
        starCriteria: STAR_CRITERIA,
      }}
      onRetry={reset}
    />
  );
}
```

## <slug>.test.ts
```ts
import { describe, it, expect } from "vitest";
import { computeMetric, evaluate } from "./ml";

describe("<slug> ml", () => {
  it("computes the metric correctly", () => { /* against a hand-checked case */ });
  it("names the failure mode, with numbers, only when it happens", () => { /* … */ });
});
describe("<slug> store + code api", () => {
  it("moves the metric when the store changes", () => { /* … */ });
  it("credits the third star only to a code-lane clear", () => { /* … */ });
  it("rejects bad api arguments with a named error", () => { /* … */ });
});
```

## Registration (all five, or the game is invisible)
```ts
// src/lib/catalog.ts        → GAME_CATALOG: { slug: "<slug>", title, category, concept,
//                              coreIntuition, metricLabel, failureMode, difficulty,
//                              complexity, phase, requires: [...] }
//                              (catalog.test.ts pins the count at 14 — update it)
// src/games/registry.ts     → PLAYABLE_SLUGS: [..., "<slug>"]
// src/games/GameMount.tsx   → "<slug>": dynamic(() => import("./<slug>"),
//                                { ssr: false, loading: () => <GameLoading /> }),
// src/engine/progression.ts → CONCEPT_BADGES: { ..., "<slug>": "<concept, lower case>" }
```

## scripts/playthroughs/<slug>.mjs
```js
export const slug = "<slug>";
export const title = "<Game Name>";          // must match the GameShell <h1>

export async function run({ page, check, metricText }) {
  const failure = page.locator('[data-testid="named-failure"]');
  const whyRegion = page.getByRole("region", { name: /why did that happen/i });
  const controls = page.locator("#game-canvas + div");

  console.log("\nShell anatomy");
  check("metric readout present",
    (await page.locator('[data-testid="metric-value"]').count()) > 0);
  check("why-card docked", await whyRegion.isVisible());

  console.log("\nThe metric moves with the player's action");
  const before = await metricText();
  // … act through the real controls (getByRole), as a player would …
  check("metric changed", (await metricText()) !== before);

  console.log("\nNamed failure");
  // … drive the game into its failure mode …
  // check("names the failure", /Overfitting/.test(await failure.innerText()));

  console.log("\nBoth lanes share one store");
  // … switch to Code, run the snippet, switch back, assert the visual lane changed …
}
```
