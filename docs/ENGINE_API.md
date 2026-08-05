# GameML Engine API Reference

The shared engine, built once and reused by all 14 games (CLAUDE.md). This is the
contract `game-builder` codes against — you shouldn't need to read the engine
source to build a game.

**Never fork anything in here.** If a game needs different behaviour, add a prop
and document it (and if it's visual, add the pattern to DESIGN.md first).

```ts
import { GameShell, useModel, useCodeLane, useProgression } from "@/engine";
import { MetricReadout, Slider, Dial, WhyCard, DatasetChip } from "@/components";
```

---

## 1. `<GameShell>`

The fixed frame every game renders inside (DESIGN.md §5). Presentational — it
computes nothing. If the metric moves, it's because your game recomputed it from
real client-side ML and passed a new value down.

```ts
interface GameShellProps {
  slug: string;                          // catalog slug
  title: string;
  backHref?: string;                     // default "/"

  metric: MetricSpec;                    // REQUIRED — the always-visible metric
  secondaryMetrics?: readonly MetricSpec[];

  math: MathReveal;                      // ƒ Math drawer contents

  controls: ReactNode;                   // right rail
  visual: ReactNode;                     // no-code lane
  code: ReactNode;                       // code lane

  whyCard?: WhyCardContent | null;
  failure?: NamedFailure | null;         // named failure mode

  progress?: GameShellProgress;

  lane?: Lane;                           // omit both to let the shell manage it
  onLaneChange?: (lane: Lane) => void;
  codeLaneDisabled?: boolean;
  codeLaneDisabledReason?: string;

  onRetry?: () => void;
  onNext?: () => void;
  nextLabel?: string;                    // default "Next"
}

interface GameShellProgress {
  level: number;
  xpIntoLevel: number;
  xpForNextLevel: number;
  stars: StarCount;                      // 0 | 1 | 2 | 3
  recentGain?: number | null;            // flashes "+120 XP"
  starCriteria?: readonly string[];
}
```

### Two behaviours to design around

**Only the active lane is mounted.** Switching lanes unmounts the other one. Game
state survives because it lives in your Zustand store, not in the lane
components. This is why the two-lane rule *requires* a shared store — and it's
what stops a Three.js or Phaser canvas from sitting in memory while the player
reads code.

**`failure` drives the loss presentation.** Setting it turns the metric red
automatically (you don't have to also set `metric.state`) and renders an inline
`role="alert"` strip with the failure name, the numbers, and a one-click Retry.
There is deliberately no blocking Game Over screen (DESIGN.md §8).

```tsx
<GameShell
  slug="sort-it-arcade"
  title="Sort-It Arcade"
  metric={{ label: "Accuracy", value: accuracy, format: "percent", goodDirection: "up" }}
  secondaryMetrics={[{ label: "Complexity", value: penalty, format: "integer", goodDirection: "down" }]}
  math={{ equation: String.raw`\hat{y} = \operatorname{sign}(w^\top x + b)`, code: BOUNDARY_SOURCE }}
  controls={<BoundaryControls />}
  visual={<VisualLane />}
  code={<CodeLane />}
  whyCard={whyCard}
  failure={failure}
  progress={progress}
  onRetry={reset}
/>
```

---

## 2. `useModel()` — TF.js lifecycle

```ts
function useModel(options: UseModelOptions): UseModelApi;

interface UseModelOptions {
  build: () => tf.LayersModel;                  // pure factory, may run twice
  onEpoch?: (metrics: EpochMetrics) => void;    // the live-metric wire
  onDone?: (last: EpochMetrics | null) => void;
  autoBuild?: boolean;                          // default false
}

interface UseModelApi {
  status: "idle" | "ready" | "training" | "error";
  error: string | null;
  epoch: number;            // 1-based, for "epoch 7 of 50"
  totalEpochs: number;
  latest: EpochMetrics | null;

  build(): void;                                       // rebuild disposes first
  train(request: TrainRequest): Promise<EpochMetrics | null>;
  predict(xs: number[][]): Float32Array | null;
  stop(): void;                                        // halts at epoch boundary
  reset(): void;
  tensorCount(): number;                               // tf.memory().numTensors
}

interface EpochMetrics {
  epoch: number;            // 0-based, as TF.js reports it
  loss: number;
  accuracy: number | null;  // null unless compiled with metrics: ["accuracy"]
  valLoss: number | null;
  valAccuracy: number | null;
}

interface TrainRequest {
  xs: number[][];
  ys: number[][] | number[];   // flat arrays are lifted to one column
  epochs?: number;             // default 50
  batchSize?: number;          // default 32
  validationSplit?: number;    // enables valLoss / valAccuracy
  shuffle?: boolean;           // default true
}
```

### Rules this hook enforces for you

- **No tensors cross the boundary.** You pass plain arrays and get plain arrays
  back. Nothing that needs disposing can be captured in a render closure.
- **Disposal on rebuild, reset, and unmount** — including the optimizer. TF.js's
  `model.dispose()` frees the weights but *not* Adam's accumulator variables
  (two per weight), which is the leak that eventually kills a tab.
- **Overlapping trainings are serialised.** A player mashing "Train" stops the
  in-flight fit and awaits it before the next starts, so two fits never share a
  model. Stale `onEpoch` callbacks from a superseded run are dropped.
- **`predict` runs inside `tf.tidy`.**

`src/engine/useModel.test.ts` asserts `tensorCount()` returns to baseline after a
full round, after unmounting mid-training, and after five rebuilds. If you add a
game that trains, keep that promise.

```ts
const model = useModel({
  build: () => buildClassifier(2),
  onEpoch: ({ loss, accuracy }) => store.setLiveMetrics(loss, accuracy),
});

await model.train({ xs, ys, epochs: 40, validationSplit: 0.2 });
const probabilities = model.predict(gridPoints);
```

---

## 3. `useCodeLane()` — the code lane

```ts
function useCodeLane<TApi extends object>(
  options: UseCodeLaneOptions<TApi>,
): UseCodeLaneApi;

interface UseCodeLaneOptions<TApi extends object> {
  initialCode: string;                  // reset() restores exactly this
  api: TApi;                            // your store, as callable functions
  language?: "javascript" | "python";   // default "javascript"
  executor?: CodeExecutor<TApi>;
  maxRunMs?: number;                    // default 4000
  autoRun?: boolean;                    // default false
  maxLogs?: number;                     // default 100
}

interface UseCodeLaneApi {
  code: string;
  setCode(code: string): void;
  dirty: boolean;                       // edited away from initialCode
  running: boolean;
  error: string | null;                 // shown, never thrown
  logs: readonly CodeRunLog[];          // { level: "log" | "error", message }
  run(): Promise<void>;                 // always resolves
  reset(): void;
  clearLogs(): void;
  language: CodeLanguage;
  available: boolean;                   // false for python until Pyodide lands
}
```

The snippet body receives three parameters: `api`, `log(...)`, and
`checkBudget()`. It may use `await`.

Keep the `api` small and verb-shaped so the snippet reads like the algorithm:

```ts
const lane = useCodeLane({
  initialCode: [
    "// One k-means iteration. Run it again and watch inertia fall.",
    "api.assignPointsToNearestCentroid();",
    "api.moveCentroidsToClusterMean();",
    "log('inertia', api.inertia());",
  ].join("\n"),
  api: {
    assignPointsToNearestCentroid: store.assign,
    moveCentroidsToClusterMean: store.update,
    inertia: () => store.getState().inertia,
  },
});
```

### Security — read this before wiring a share feature

The JavaScript executor uses `new AsyncFunction(...)`: the player's own code, in
their own tab, same trust model as a browser console. It is **not** a security
sandbox.

1. **Never** feed this hook code that came from another user, a URL parameter, or
   a shared-snippet feature. That converts a playground into stored XSS. Such a
   feature needs a Web Worker or iframe isolate first.
2. Runaway loops are only *cooperatively* interruptible. `checkBudget()` throws
   `CodeLaneTimeoutError` once the budget is spent, but only when the snippet
   calls back in. A bare `while (true) {}` will still hang the tab.

`createPyodideExecutor()` is the seam for Python lanes (Feature Forge, Phase 3).
It currently throws a clear player-facing message rather than failing silently.

---

## 4. `progression` — XP, stars, badges, unlocks

### Pure functions (no React, no storage)

```ts
xpForScore(score: number, lane: Lane): number      // BASE_XP=100, code lane ×1.5
xpToClearLevel(level: number): number              // 200, 300, 400, …
levelFromXp(totalXp: number): LevelInfo            // { level, xpIntoLevel, xpForNextLevel }
starsFor({ completed, bestScore, codeLaneCleared }): StarCount
isUnlocked(slug: string, state: ProgressionState): boolean
unlockedSlugs(state: ProgressionState): string[]
applyResult(state: ProgressionState, result: GameResult): AppliedResult
badgeLabel(slug: string): string | null            // "I understand overfitting"
```

Stars are **cumulative** (spec §4): ⭐ finished · ⭐⭐ scored ≥ `HIGH_SCORE_THRESHOLD`
(0.8) · ⭐⭐⭐ also cleared the code-lane challenge. Clearing the code lane at a low
score is still one star.

XP is granted as the **increment** over what a game has already paid out, so
replaying an easy level can't farm levels — but beating your own best always pays
the difference.

### Store

```ts
const { recordResult, level, starsFor, isUnlocked, hydrate, xp, badges } =
  useProgression();

const { xpGained, stars, newBadge } = recordResult({
  slug: "gradient-descent-skier",
  score: 0.92,          // normalised 0–1; each game maps its own scoring
  lane: "visual",
  completed: true,
  codeLaneCleared: false,
});
```

Writes are optimistic: local state updates immediately so the XP bar springs at
once, then persistence happens behind it. Failures land in `syncError` rather
than blocking or reverting.

### Adapters

| Adapter | When |
|---|---|
| `createLocalAdapter()` | Default. localStorage, works offline, no account. |
| `createSupabaseAdapter(userId)` | Returns `null` unless Supabase is configured. |
| `createMemoryAdapter(seed?)` | Tests. |

Progression is never a hard dependency of gameplay — spec §6's first-run flow is
an explicit no-signup demo. The Supabase adapter's docblock carries the table DDL
and its RLS policy. It uses the **anon** key only; the service-role key must
never reach the browser.

---

## 5. Component primitives

From `@/components` (DESIGN.md §6).

| Component | Notes for game authors |
|---|---|
| `<MetricReadout>` | `{ label, value, format?: "percent"\|"decimal"\|"integer", precision?, goodDirection?: "up"\|"down", state?, caption? }`. `GameShell` renders this for you — use it directly only for extra readouts. `goodDirection` decides which way is green. |
| `<WhyCard>` | `{ key, title, body, tone?, conceptHref? }`. Change `key` to replay the fade-in — that's how the player knows the explanation is new. On a loss, `title` must be the named failure. |
| `<LaneToggle>` | Rendered by `GameShell`. Real radiogroup with arrow-key support. |
| `<MathDrawer>` | Rendered by `GameShell`. KaTeX equation + real code. |
| `<XPBar>`, `<StarRating>` | Rendered by `GameShell` when you pass `progress`. |
| `<Slider>` | `{ label, value, min, max, step?, onChange, scale?: "linear"\|"log", format?, unit?, hint? }`. Native `<input type="range">` underneath. **Use `scale="log"` for anything spanning orders of magnitude** — learning rate especially. |
| `<Dial>` | Same shape as `Slider` plus `keyStep`. Vertical drag; full keyboard (arrows, Page Up/Down, Home/End). |
| `<DatasetChip>` | `{ name, detail?, selected?, locked?, lockedReason?, onSelect }`. |
| `<Button>` | `variant: "primary" \| "secondary" \| "ghost" \| "danger"`, `size: "sm" \| "md"`. 44px min target. |
| `<CodeEditor>` | `{ label, value, onChange, language?, error?, hint?, rows? }`. Tab is intentionally not trapped. |
| `<CodeBlock>` | Read-only, syntax-coloured. Used by `MathDrawer`. |

Helpers: `useAnimatedNumber(target, { durationMs?, disabled? })`,
`formatMetric(value, format, precision?)`,
`metricAnnouncement(label, value, previous, format, precision?)`,
`fromPosition` / `toPosition` (the linear/log scale maths).

---

## 6. Shared types

```ts
type Lane = "visual" | "code";
type StarCount = 0 | 1 | 2 | 3;

interface MathReveal {
  equation: string;                     // KaTeX/LaTeX, display mode
  code: string;                         // the REAL code from your ml.ts
  codeLanguage?: "javascript" | "python";
  notes?: string;                       // plain-language gloss
}

interface NamedFailure {
  name: string;                         // "Overfitting", "Divergence"
  detail: string;                       // "99% train, 61% test"
}
```

`ml-verifier` checks that `MathReveal.equation` matches what `MathReveal.code`
does, and that the code matches your actual `ml.ts`. Copy it; don't retype it
from memory.

---

## 7. What the engine does NOT do

Deliberate gaps, so you don't go looking:

- **No ML.** The engine runs *your* model; it has no opinion on the algorithm.
  `ml.ts` is yours.
- **No game state.** Every game owns its Zustand store. The engine holds only
  lane selection and drawer open/closed.
- **No scoring.** You map your game's scoring onto a normalised 0–1 `score`
  before calling `recordResult`.
- **No canvas.** D3 / p5 / Three / Phaser / React Flow live in your lane
  components. The shell just gives them a sized, focusable, labelled box.
- **No routing.** Wire `src/app/play/<slug>/page.tsx` yourself.

---

## 8. Accessibility already handled

You inherit these; don't undo them.

- Focus rings: global `:focus-visible`, 2px `--primary`.
- Metric announcements: one debounced `aria-live="polite"` region per
  `MetricReadout`. The tweening number is `aria-hidden` on purpose — `<output>`
  has an implicit `role="status"`, so leaving it exposed would fire an
  announcement on every animation frame.
- Direction of travel is a glyph (▲ ▼ –) plus words in the announcement, never
  colour alone.
- 44px minimum targets on every interactive primitive.
- `prefers-reduced-motion` respected globally and in `useAnimatedNumber`.
- Skip link to `#game-canvas`, which `GameShell` marks focusable.

**What you still own per game:** keyboard access to the canvas itself. A drag-only
boundary or a pointer-only dial fails the contract. DESIGN.md §10 requires
button-based fallbacks for 3D games specifically.
