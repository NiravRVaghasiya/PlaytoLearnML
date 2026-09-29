# GameML Engine API Reference

The shared engine, built once and reused by all 14 games (CLAUDE.md). This is the
contract `game-builder` codes against — you shouldn't need to read the engine
source to build a game. When this file and the source disagree, the source wins
and this file is the bug.

**Never fork anything in here.** If a game needs different behaviour, add a prop
and document it (and if it's visual, add the pattern to DESIGN.md first).

```ts
import { GameShell } from "@/engine/GameShell";
import { useModel } from "@/engine/useModel";
import { useCodeLane, createPyodideExecutor } from "@/engine/useCodeLane";
import { useProgression, levelFromXp, HIGH_SCORE_THRESHOLD } from "@/engine/progression";
import { MetricReadout, Slider, Dial, Button, CodeEditor, type MetricSpec } from "@/components";
```

The `@/engine` barrel re-exports the public API of the four engine modules above
(`@/components` is its own barrel). Games may use it, but it re-exports
`useModel`, which imports TensorFlow.js. **Code outside a game (the
home page, the Concept Library) must import from the specific module**, for
example `@/engine/progression`, or it drags TF.js into that page's bundle.

---

## 1. `<GameShell>`

The fixed frame every game renders inside (DESIGN.md §5). Presentational — it
computes nothing about the game. If the metric moves, it's because your game
recomputed it from real client-side ML and passed a new value down. (It does read
the progression store, for the "not saved" hint and the concept badge — never for
game logic.)

```ts
interface GameShellProps {
  slug: string;                          // catalog slug (progression key, data-game attr)
  title: string;
  backHref?: string;                     // default "/"

  metric: MetricSpec;                    // REQUIRED — the always-visible metric
  secondaryMetrics?: readonly MetricSpec[];

  math: MathReveal;                      // Math drawer contents (REQUIRED)

  controls: ReactNode;                   // the rail (REQUIRED)
  visual: ReactNode;                     // no-code lane
  code: ReactNode;                       // code lane

  whyCard?: WhyCardContent | null;
  failure?: NamedFailure | null;         // named failure mode

  progress?: GameShellProgress;          // omit to hide the XP/stars widgets

  lane?: Lane;                           // omit both to let the shell manage it
  onLaneChange?: (lane: Lane) => void;
  codeLaneDisabled?: boolean;
  codeLaneDisabledReason?: string;

  onRetry?: () => void;                  // shows Retry in the failure strip and the footer
  onNext?: () => void;
  nextLabel?: string;                    // default "Next"
}

interface GameShellProgress {
  level: number;
  xpIntoLevel: number;
  xpForNextLevel: number;
  stars: StarCount;                      // 0 | 1 | 2 | 3
  recentGain?: number | null;            // pass useProgression's lastGain; flashes "+120 XP"
  starCriteria?: readonly string[];      // shown by StarRating's "What earns each star"
}
```

### Behaviours to design around

**Only the active lane is mounted.** Switching lanes unmounts the other one. Game
state survives because it lives in your Zustand store, not in the lane
components. This is why the two-lane rule *requires* a shared store — and it's
what stops a Three.js canvas from sitting in memory while the player reads code.
The code lane's own state (the edited snippet, the last output and error) is
kept by `useCodeLane` in a module-level draft cache keyed by `persistKey`
(default: `initialCode`), so a lane switch, or leaving the game and coming back,
restores it for the page's lifetime. `reset()` restores `initialCode` and
forgets the draft. Nothing is written to storage.

**Each lane renders inside a `LaneErrorBoundary`,** keyed by lane. A render-time
throw in a lane (a WebGL canvas that can't get a context, a TF.js call on a
disposed model) replaces only that lane with "The visual lane stopped working",
a retry button and a pointer to the other lane. The header, metric and controls
stay up. Outside it, `GameMount`'s `GameErrorBoundary` catches crashes in the
controls, the shell itself, or a game chunk that failed to download. Route-level
`src/app/error.tsx` backs both up.

**`failure` drives the loss presentation.**

- Setting it turns the metric red automatically; you don't also have to set
  `metric.state`. An explicit `metric.state` wins.
- It renders an inline strip (`data-testid="named-failure"`) with the failure
  name, the numbers, and a one-click Retry when `onRetry` is set. There is
  deliberately no blocking Game Over screen (DESIGN.md §8).
- The strip itself is **not** a live region. The shell keeps one persistent
  `role="alert"` region, filled with `name. detail` when a failure arrives or
  its **name** changes. A detail whose numbers move under the same name
  (Convolution Kitchen quotes live per-class accuracy) is not re-read; the strip
  shows the current detail. Re-setting the same failure, or clearing it and
  bringing it back within 3 s, is not re-read either. The region empties 3 s after the
  failure clears, so the same failure returning later is announced again.
- Retry (the strip's or the footer's) empties the region before calling
  `onRetry`, so the same failure straight after a Retry is announced as a new
  loss, grace or not.

**WhyCard headlines are announced by the shell.** When `whyCard.key` changes, a
persistent polite region receives `whyCardHeadline(card)` after 700 ms: the tone
prefix and the title, never the body. A burst of cards (a drag) announces only
the last one. A `tone: "bad"` card shown during a failure isn't announced,
because the alert already said it. Change `key` on every action; that is also
what replays the card's fade-in.

**Layout contract.**

- At `lg` and up: the header is sticky, the controls sit above the metric in a
  320 px right rail, and the canvas is on the left.
- Below `lg`: only the primary metric pins (`sticky top-0`). The header scrolls
  away, secondary metrics scroll, and the order is metric → canvas → controls.
- Below `lg`, on viewports 320 px tall or less (400% zoom), nothing pins.
  Landscape phones (about 340–412 px tall) keep the pin.

Pages with a shell get `scroll-padding-top` to match (globals.css), so keyboard
focus isn't hidden under the pinned metric. Its short-viewport query must use
the same height as the shell's unpin rule; `GameShell.test.tsx` fails if they
drift apart.

**Landmarks and hooks the harnesses rely on.** Keep these when you touch the
shell.

- The canvas is `<main id="game-canvas" tabIndex={-1}>`. The skip link focuses
  it once a game has mounted.
- The primary metric is a region labelled "Live metric", and the secondaries are
  "More metrics".
- The rail is a `role="region"` labelled "Controls", and it stays the `<div>`
  immediately after `<main>`: the playthroughs find it as `#game-canvas + div`.
- The WhyCard dock is a region labelled "Why did that happen?".
- The header's Math button is named exactly "Math" and carries
  `data-testid="math-open"`. The drawer is a `role="dialog"` with `aria-modal`,
  named "Reveal the math: <title>" (`data-testid="math-dialog"`). Its equation
  box is `data-testid="math-equation"`, with `data-state` `loading`, `rendered`
  or `source`. `checkMathDialog` in `scripts/harness.mjs` waits on that state
  rather than a timeout.

**The Math drawer is modal.** While it is open, the header, content and footer
are `inert`. The two announcement regions sit outside them and keep working.

**Footer extras.**

- Once this game's concept badge is earned, the footer shows it ("I understand
  overfitting").
- If a save fails, it shows "Progress isn't being saved on this device."
- Both are in a polite status region.
- On unmount the shell calls `useProgression.getState().dismissGain()`, so a
  "+N XP" never replays in the next game.

```tsx
<GameShell
  slug={SLUG}
  title={meta?.title ?? "Sort-It Arcade"}
  metric={{ label: "Accuracy", value: accuracy, format: "percent", goodDirection: "up" }}
  secondaryMetrics={[{ label: "Complexity", value: cost, format: "integer", goodDirection: "down" }]}
  math={{ equation: MATH_EQUATION, code: MATH_CODE, codeLanguage: "javascript", notes: MATH_NOTES }}
  controls={<Controls />}
  visual={<VisualLane />}
  code={<CodeLane />}
  whyCard={whyCard}
  failure={failure}
  lane={lane}
  onLaneChange={setLane}
  progress={{ ...levelFromXp(xp), stars: games[SLUG]?.stars ?? 0, recentGain: lastGain, starCriteria: STAR_CRITERIA }}
  onRetry={reset}
  onNext={won ? newRound : undefined}
  nextLabel="Next round"
/>
```

`src/games/sort-it-arcade/index.tsx` is the reference wiring, including
progression hydration.

---

## 2. `useModel()` — TF.js lifecycle

Five games train a TF.js model: Data Detox, Feature Forge, Neuron Forge, Overfit
Tower Defense and Convolution Kitchen. `useModel` is used by Data Detox, Neuron
Forge and Overfit TD. The others run TF.js directly in their store or `ml.ts`,
under the same disposal rules.

```ts
function useModel(options: UseModelOptions): UseModelApi;

interface UseModelOptions {
  build: () => tf.LayersModel;                  // pure factory, may run twice (Strict Mode)
  onEpoch?: (metrics: EpochMetrics) => void;    // the live-metric wire
  onDone?: (last: EpochMetrics | null) => void; // after a fit finishes or is stopped (not on error or takeover)
  autoBuild?: boolean;                          // default false
}

interface UseModelApi {
  status: "idle" | "ready" | "training" | "error";
  error: string | null;
  epoch: number;            // 1-based, for "epoch 7 of 50"
  totalEpochs: number;
  latest: EpochMetrics | null;

  build(): void;                                       // (re)build; old model freed safely
  train(request: TrainRequest): Promise<EpochMetrics | null>;
  predict(xs: number[][]): Float32Array | null;        // null when no model / empty input
  stop(): void;                                        // halts at the next epoch boundary
  reset(): void;                                       // dispose + clear, cancels queued trains
  tensorCount(): number;                               // tf.memory().numTensors
}

interface EpochMetrics {
  epoch: number;            // 0-based, as TF.js reports it
  loss: number;             // NaN if TF.js reported none
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
  rebuild?: boolean;           // fresh weights, built AFTER any in-flight fit stops
}
```

### Rules this hook enforces for you

- **No tensors cross the boundary.** You pass plain arrays and get plain arrays
  back. Nothing that needs disposing can be captured in a render closure.
- **Disposal on rebuild, reset, and unmount** — including the optimizer. TF.js's
  `model.dispose()` frees the weights but *not* Adam's accumulator variables
  (two per weight), which is the leak that eventually kills a tab.
- **A model is never disposed mid-fit.** `build()` or `reset()` during a fit
  stops it, and frees the old weights once it has actually stopped. So
  `build(); await train(…)` is safe even while a previous fit is running.
  `train({ …, rebuild: true })` does the same thing in one call.
- **Overlapping trainings are serialised, latest wins.** A new `train()` stops
  the in-flight fit and waits for it. Of all the calls that queued up in the
  meantime, only the newest runs; the others resolve `null`. So does the fit it
  stopped: a fit taken over mid-run by a newer `train()`, `build()` or `reset()`
  resolves `null` and doesn't fire `onDone`. `reset()` and unmount cancel
  anything still queued. Stale `onEpoch` callbacks from a superseded fit are
  dropped. Three rapid calls are covered by a test.
- **`train()` resolving `null` means "not trained".** That happens on bad input
  (row counts differ), a factory failure, a takeover by a newer call or a
  reset, or an unmount while the call was still queued. Never score whatever
  the model currently holds after a `null`.
- **A fit cut short by `stop()` or an unmount resolves with its last completed
  epoch** (`null` if none finished) and fires `onDone`, because nothing replaced
  the model. The run is partial, so check the epoch count before scoring it.
- **`predict` runs inside `tf.tidy`.**

Keep your own store-level guard as well (`if (state.training) return`, or a
disabled Train button). The hook makes overlapping calls safe, but only your game
knows whether a second click should restart or be ignored.

`src/engine/useModel.test.ts` asserts `tensorCount()` returns to baseline:

- after a full build → train → predict → reset round,
- after unmounting mid-training,
- after repeated rebuilds.

If you add a game that trains, keep that promise.

```ts
const model = useModel({
  build: () => buildClassifier(2),
  onEpoch: ({ loss, accuracy }) => store.setLiveMetrics(loss, accuracy),
});

const last = await model.train({ xs, ys, epochs: 40, validationSplit: 0.2, rebuild: true });
if (last === null) return;               // superseded or cancelled — don't score
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
  executor?: CodeExecutor<TApi>;        // default: JS executor, or a bare Pyodide one for python
  maxRunMs?: number;                    // default 4000 — JavaScript only, cooperative
  budgetApiCalls?: boolean;             // default false — check the budget on every api call
  autoRun?: boolean;                    // default false — run once on mount
  maxLogs?: number;                     // default 100
  persistKey?: string | null;           // draft-cache key; default initialCode, null opts out
}

interface UseCodeLaneApi {
  code: string;
  setCode(code: string): void;
  dirty: boolean;                       // edited away from initialCode
  running: boolean;
  error: string | null;                 // shown, never thrown (see "How a run behaves")
  logs: readonly CodeRunLog[];          // { level: "log" | "error", message }
  run(): Promise<void>;                 // always resolves; ignored while a run is in flight
  reset(): void;                        // restores initialCode, clears error and logs, forgets the draft
  clearLogs(): void;
  language: CodeLanguage;
  available: boolean;                   // always true for "javascript" and "python"
}
```

### How a run behaves

1. `run()` is ignored if a run is already in flight. A double click can't
   interleave two snippets' api calls against one store.
2. It sets `running`, then **yields a paint** (`yieldToPaint()`), so "Running…"
   is on screen before a long synchronous snippet starts.
3. The time budget starts **after** that yield.
4. Logs are collected during the run and committed when it ends.
   `log(x)` formats each argument with the exported `formatLogArg`:
   - strings are passed through;
   - an `Error` shows as `Name: message`, not `{}`;
   - functions and symbols are named;
   - a `Map`, a `Set` or a typed array shows its contents;
   - a circular structure doesn't throw.
5. Any throw becomes `error` and a final `level: "error"` log line. It is never
   rethrown. The player's own errors read `${name}: ${message}`, because
   "TypeError" or "NameError" is the clue. The engine's own player-facing errors
   (`CodeLaneTimeoutError`, `PyodideLoadError`) are already sentences, and show
   only their message.
6. The edited code, and the run's logs and error, are written through to the
   draft cache, even if the lane unmounted mid-run, so they are there when the
   player comes back.

### The JavaScript executor

`createJsExecutor()` is the default. The snippet body receives three parameters:
`api`, `log(...)` and `checkBudget()`, and it may use `await`.

`checkBudget()` throws `CodeLaneTimeoutError` once `maxRunMs` is spent: "Your
code ran longer than 4000ms and was stopped. Look for a loop that never ends, or
do less work per run." With `budgetApiCalls: true`, every call into `api` checks
the budget first. That is how `for (;;) await api.train()` gets stopped when the
snippet never calls `checkBudget()` itself. It is off by default; Neuron Forge
and Overfit TD turn it on, because their api is where the time goes. Size
`maxRunMs` for the slowest honest run on TF.js's CPU backend (no WebGL), which
is far slower than WebGL. Neuron Forge and Overfit TD allow 15 minutes, and
Convolution Kitchen 2 minutes.

Keep the `api` small and verb-shaped so the snippet reads like the algorithm.
Build it in a plain function (most games export `createCodeApi()` from `store.ts`
or `code-api.ts`), so it can be unit-tested without React. Validate arguments
and throw named errors; the error text is what the player reads.

```ts
const api = useMemo(() => createCodeApi(), []);
const lane = useCodeLane({
  initialCode: STARTER_CODE,
  api,                                   // e.g. { assign, update, inertia, check }
  maxRunMs: 6000,
});
```

### The Python executor: `createPyodideExecutor(options)`

This is real CPython, with packages, in the tab. Feature Forge uses it, with
pandas.

```ts
interface PyodideExecutorOptions {
  packages?: readonly string[];   // e.g. ["pandas"]; loaded before each run (a no-op once in)
  prelude?: string;               // Python run before the player's code each time (your helpers)
  indexURL?: string;              // default: NEXT_PUBLIC_PYODIDE_INDEX_URL || "/pyodide/"
  onStatus?: (status: string) => void;
  loadTimeoutMs?: number;         // default 180_000, per download step
  loadModule?: (indexURL: string) => Promise<PyodideModule>;  // test seam only
}
```

- **Loading.** The runtime (about 13 MB) loads from `indexURL` the first time a
  Python lane runs, never on page load. It is loaded with a bundler-ignored
  dynamic `import()`, so it is not in the client bundle. One runtime is shared
  per `indexURL` across runs and games.
- **Status.** `onStatus` reports, in order:
  1. "Downloading the Python runtime (about 13 MB, once per visit)…"
  2. "Loading pandas…", only while packages are missing.
  3. "Starting pandas — the page may pause for a few seconds…". This appears on
     the first run with a prelude and packages, and a paint is yielded before
     the freeze.
  4. "Python <version> ready.", where the version is the runtime's
     `pyodide.version`.

  The hook only commits logs at the end of a run, so wire `onStatus` to your own
  state to show progress.
- **Failures are recoverable, not cached.**
  - A runtime load that fails or stalls past `loadTimeoutMs` rejects with
    `PyodideLoadError` ("… Check your connection and press Run again.") and is
    evicted, so the next Run starts fresh.
  - `loadPackage` doesn't reject when a wheel 404s, so the executor checks
    `loadedPackages` afterwards and throws `PyodideLoadError`. The next Run
    retries only the missing packages.
- **Python errors.** An exception is reduced to `PythonRunError`, whose `name` is
  the Python type, so the lane shows "NameError: name 'x' is not defined
  (line 3)". The full traceback goes to the output log. Tracebacks name the
  player's code `<your code>` and the prelude `<lane setup>`, so a line number
  always refers to the right one. `formatPythonError(message)` is exported if
  you need the one-line summary yourself.
- **What the snippet sees.** A global `api`, which is your JS object. `print()`
  and stderr go to the output panel. There is no `log` and no `checkBudget` in
  Python.
- **Deploying it.** The wheels must be staged by `scripts/setup-pyodide.mjs`.
  Its `LANE_PACKAGES` must list every top-level package any lane passes in
  `packages`; dependencies are resolved from the lockfile. See
  [DEVELOPMENT.md](DEVELOPMENT.md#the-python-lane-and-its-wheels).

### Limitations — read this before wiring a share feature

1. **Not a security sandbox.** The JavaScript executor uses
   `new AsyncFunction(...)`, and Python runs with full access to the page through
   Pyodide's JS bridge. It is the player's own code in their own tab, the same
   trust model as a browser console. **Never** feed this hook code that came from
   another user, a URL parameter or a shared-snippet feature; that turns a
   playground into stored XSS. Such a feature needs a Web Worker or iframe
   isolate first. (The CSP allows `'unsafe-eval'` precisely because of this
   executor.)
2. **JavaScript is only cooperatively interruptible.** The budget can only throw
   when the snippet calls `checkBudget()`, or calls `api` with `budgetApiCalls`
   on. A bare `while (true) {}` hangs the tab.
3. **Python is not interruptible at all, and runs on the main thread.**
   `maxRunMs`, `budgetApiCalls` and `checkBudget()` don't apply to it: Python
   executes inside wasm and never yields. A `while True:` hangs the tab, and the
   first `import pandas` freezes the page for a few seconds. The real fixes are a
   Worker-hosted runtime plus Pyodide's interrupt buffer (a `SharedArrayBuffer`,
   which needs COOP+COEP; COEP is not set). Neither is built.

---

## 4. `progression` — XP, stars, badges, unlocks

Everything persists to **localStorage**, per browser (key `gameml:progression:v1`).
There are no accounts. See "Adapters" below for the Supabase seam.

### Pure functions (no React, no storage)

```ts
xpForScore(score: number, lane: Lane): number      // BASE_XP=100 × score, code lane ×1.5
xpToClearLevel(level: number): number              // 200, 300, 400, …
levelFromXp(totalXp: number): LevelInfo            // { level, xpIntoLevel, xpForNextLevel }
starsFor({ completed, bestScore, codeLaneCleared }): StarCount
isUnlocked(slug: string, state: ProgressionState): boolean
unlockedSlugs(state: ProgressionState): string[]
applyResult(state: ProgressionState, result: GameResult): AppliedResult
mergeProgression(a: ProgressionState, b: ProgressionState): ProgressionState
badgeLabel(slug: string): string | null            // "I understand overfitting"
```

Constants: `BASE_XP`, `CODE_LANE_MULTIPLIER`, `HIGH_SCORE_THRESHOLD` (0.8),
`GAIN_FLASH_MS` (2000), `CONCEPT_BADGES` (one per catalog slug, enforced by a
test), `EMPTY_PROGRESSION`, `STORAGE_KEY`.

Stars are **cumulative** (spec §4): ⭐ finished · ⭐⭐ scored ≥ `HIGH_SCORE_THRESHOLD`
(0.8) · ⭐⭐⭐ also cleared the code-lane challenge. Clearing the code lane at a low
score is still one star. Import `HIGH_SCORE_THRESHOLD` into your `STAR_CRITERIA`
copy rather than restating 80%, so the words can't drift from the rule.

Concept badges land at two stars.

XP is granted as the **increment** over what a game has already paid out, so
replaying an easy level can't farm levels — but beating your own best always pays
the difference.

`isUnlocked`/`unlockedSlugs` implement the catalog's `requires` graph and are
tested, but **nothing in the UI gates on them**. Every game is open, and the home
page says so.

### Store

```ts
const {
  xp, games, badges,          // ProgressionState
  hydrated, syncError, lastGain,
  hydrate, recordResult, dismissGain, setAdapter, clear,
  isUnlocked, level, starsFor,
} = useProgression();

useEffect(() => { if (!hydrated) void hydrate(); }, [hydrated, hydrate]);

const { xpGained, stars, newBadge } = useProgression.getState().recordResult({
  slug: "sort-it-arcade",
  score: 0.92,          // normalised 0–1; each game maps its own scoring
  lane: source,         // the lane the clearing ACTION came from
  completed: true,
  codeLaneCleared: source === "code",
});
```

- **Attribute the lane to the action, not the tab.** The code lane's api should
  call the store's scoring action with `"code"`, and the visual lane's button
  with nothing (the default, `"visual"`). The third star must follow what cleared
  the round, never which tab happens to be showing. `sort-it-arcade/store.ts`
  `check(source)` is the reference, including recording a clear once per board
  while still letting a first code-lane clear earn the third star.
- **Derive the level from `xp`.** Subscribe to `xp` and call `levelFromXp(xp)`.
  Calling the store's `level()` reads the right value but doesn't subscribe the
  component.
- **Writes are optimistic.** State updates at once, so the XP bar springs
  immediately, and the save runs after. A save that throws, such as a
  localStorage quota or a privacy mode, sets `syncError`, which `GameShell` shows
  as a quiet footer hint. A later successful save clears it. Nothing reverts.
- **`lastGain` is transient.** It is set by `recordResult` and cleared after
  `GAIN_FLASH_MS`, so an equal second gain still flashes. `dismissGain()` clears
  it early, and `GameShell` calls it on unmount.
- **Cross-tab sync.** A `storage` event from another tab is folded in with
  `mergeProgression`, so nothing either tab earned is lost:
  - per game, the best of each field wins, and stars are recomputed;
  - badges are the union;
  - XP is never less than either side.

  An emptied save from another tab is adopted as a reset. This only applies to
  the local adapter, and only after hydration. The local adapter's `save` merges
  as well: it reads the stored blob, merges it with the state being written and
  writes the result, so two tabs saving at nearly the same moment, before
  either hears the other's `storage` event, both persist. An empty state (from
  `clear()`) is written as-is.

### Adapters

| Adapter | When |
|---|---|
| `createLocalAdapter()` | The store's adapter from the start. localStorage; `load` treats blocked or corrupt storage as "nothing saved"; `save` merges with what is stored (see "Cross-tab sync") and throws on failure, which surfaces as `syncError`. |
| `createSupabaseAdapter(userId)` | A seam, **not wired**. It returns `null` unless `NEXT_PUBLIC_SUPABASE_URL` and `…_ANON_KEY` are set. Nothing in the app calls it or `setAdapter()`: there is no sign-in flow to supply `userId`. Wiring it needs auth first, then `setAdapter(adapter)` before `hydrate()`. Its docblock carries the table DDL and the RLS policy. It uses the **anon** key only; the service-role key must never reach the browser. |
| `createMemoryAdapter(seed?)` | Tests. |

Progression is never a hard dependency of gameplay — spec §6's first-run flow is
an explicit no-signup demo.

---

## 5. Component primitives

From `@/components` (DESIGN.md §6).

| Component | Props | Notes for game authors |
|---|---|---|
| `<MetricReadout>` | `MetricSpec & { size?: "md" \| "lg", announce?: boolean, className? }` | `MetricSpec = { label, value, format?: "percent" \| "decimal" \| "integer", precision?, goodDirection?: "up" \| "down", state?: "neutral" \| "good" \| "bad" \| "warn", caption? }`. `GameShell` renders these for you from `metric`/`secondaryMetrics`; use it directly only for extra readouts. `goodDirection` decides which way pulses green, and an improvement also plays the `metric-sparkle` glow. `state` pins the colour over the pulse and suppresses the sparkle unless it is `"good"`; the ▲/▼ glyph still shows direction. Use `"neutral"` for direction-less counts (K-Means' k and iterations). Pass `NaN` for "not measured yet": it renders "—", is spoken as "<label> not measured yet", and shows no direction. Only the primary announces (`announce`, debounced 600 ms); `GameShell` sets `announce={false}` on secondaries. The number carries `data-testid="metric-value"` and the arrow `data-testid="metric-trend"`. |
| `<WhyCard>` | `{ content: WhyCardContent \| null, className? }` | Rendered by `GameShell` from `whyCard`. `WhyCardContent = { key, title, body, tone?: "info" \| "good" \| "bad" \| "warn", conceptHref?, conceptLabel? }`. Change `key` to replay the fade-in and trigger the headline announcement. On a loss, `title` must be the named failure. **`conceptLabel` must equal a section heading of the target concept exactly**, and the concept's `games` array must list your slug. `concepts.test.ts` checks both, for hrefs written as a literal or a `const X = "/concepts/…"`. `whyCardHeadline(content)` gives the announced string ("Failure: You overfit"). |
| `<LaneToggle>` | `{ value, onChange, codeDisabled?, codeDisabledReason?, className? }` | Rendered by `GameShell`. Real radiogroup, one tab stop; arrow keys move focus and selection together. |
| `<MathDrawer>` | `{ open, onClose, math, title }` | Rendered by `GameShell`. Modal dialog with a focus trap. Keydown is captured at the document while open, so game shortcuts can't fire behind it. KaTeX is lazy-loaded on first open from `katexRender.ts`, **the only file allowed to import `katex`**. A static import anywhere else puts KaTeX back into every game bundle. Equations are wrapped in `\begin{gathered}…\end{gathered}`, so top-level `\\` line breaks render. The output is HTML plus MathML, and it falls back to the LaTeX source if KaTeX throws. If KaTeX's chunk fails to download, the source shows and the next open retries the download. Test hooks: see "Landmarks and hooks" in §1. |
| `<XPBar>` | `{ level, xpIntoLevel, xpForNextLevel, recentGain?, className? }` | Rendered by `GameShell` when you pass `progress`. |
| `<StarRating>` | `{ earned, max? = 3, size?: "sm" \| "md", criteria?, className? }` | Rendered by `GameShell` when you pass `progress`. With `criteria` it adds a 44 px "What earns each star" disclosure listing each criterion as earned or not yet. |
| `<Slider>` | `{ label, value, min, max, step?, onChange, scale?: "linear" \| "log", format?, unit?, hint?, disabled?, className? }` | Native `<input type="range">` underneath, 44 px tall. **Use `scale="log"` for anything spanning orders of magnitude** — learning rate especially (requires `min > 0`). `hint` is the input's description; say what the control does to the algorithm. |
| `<Dial>` | `{ label, value, min, max, onChange, scale?, format?, unit?, hint?, disabled?, keyStep?, className? }` | No `step`. `keyStep` is a fraction of the range per arrow key or −/+ tap (default 0.02); Page Up/Down move 5×, Home/End go to the ends. It has built-in 44 px − / + buttons (the single-pointer alternative to dragging) with a polite announcement of the new value, plus vertical drag. The whole control is about 44 + 64 + 44 px wide, plus gaps. |
| `<DatasetChip>` | `{ name, detail?, selected?, locked?, lockedReason?, onSelect?, className? }` | `aria-pressed` for selection. Sort-It Arcade uses it for its capacity choice. |
| `<Button>` | `button` props + `{ variant?: "primary" \| "secondary" \| "ghost" \| "danger", size?: "sm" \| "md", icon?, children }` | 44 px min target at both sizes. `ref` is a normal prop. Games must not hand-roll buttons. |
| `<CodeEditor>` | `{ label, value, onChange, language?, readOnly?, error?, hint?, rows? = 14, className? }` | Tab is intentionally not trapped. `error` renders a `role="alert"` box. |
| `<CodeBlock>` | `{ code, language?, showLineNumbers? = true, ariaLabel?, className? }` | Read-only, syntax-coloured, focusable. Used by `MathDrawer`. |

Helpers:

- `useAnimatedNumber(target, { durationMs? = 240, disabled? })`, which snaps
  under reduced motion and for non-finite targets.
- `formatMetric(value, format, precision?)`, which gives "—" for non-finite
  values.
- `metricAnnouncement(label, value, previous, format, precision?)`.
- `fromPosition` / `toPosition`, the linear/log scale maths.

From `@/lib/utils`:

- `cx`, `clamp`, `formatPercent`.
- `seededRandom(seed)` (mulberry32) and `gaussian(random)` (Box–Muller). Use
  these for every procedural dataset, so rounds are reproducible.
- `yieldToPaint(fallbackMs = 100)`: resolves after the browser has painted. Put
  it between batches of a long synchronous loop, e.g.
  `if (i % 10 === 0) await yieldToPaint()`.
- `isWebGLAvailable()`: one probe per page, cached, whose context is released at
  once. Use it for any new 3D view. Browsers cap live WebGL contexts and kill the
  oldest, which can be TF.js's. Gradient Descent Skier and Dimension Diver both
  use it.

---

## 6. Shared types

```ts
type Lane = "visual" | "code";
type StarCount = 0 | 1 | 2 | 3;

interface MathReveal {
  equation: string;                     // KaTeX/LaTeX, display mode (wrapped in `gathered`)
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

Export the equation from `ml.ts` as `MATH_EQUATION`; the games also export
`MATH_CODE` and `MATH_NOTES` by convention. `MathDrawer.test.tsx` globs every
`src/games/*/ml.ts`, fails if a playable game has no `MATH_EQUATION`, and
renders each one with KaTeX's strict checks fully on. So LaTeX that KaTeX would
render differently from LaTeX fails the suite. At runtime KaTeX runs with
`strict: "ignore"`, so the test is where this gets caught.

---

## 7. What the engine does NOT do

Deliberate gaps, so you don't go looking:

- **No ML.** The engine runs *your* model; it has no opinion on the algorithm.
  `ml.ts` is yours.
- **No game state.** Every game owns its Zustand store. The engine holds only
  lane selection (when uncontrolled), drawer open/closed, the two announcement
  buffers, and the code lane's in-memory drafts.
- **No scoring.** You map your game's scoring onto a normalised 0–1 `score`
  before calling `recordResult`.
- **No canvas.** SVG / D3 / Three / React Flow live in your lane components.
  The shell just gives them a sized, focusable, labelled box.
- **No per-game routing.** There is one dynamic route, `src/app/play/[slug]`,
  statically generated from `src/games/registry.ts` with `dynamicParams = false`.
  A game appears there once it is registered in the five places CLAUDE.md lists:
  `catalog.ts`, `registry.ts`, `GameMount.tsx`, `CONCEPT_BADGES`, and
  `scripts/playthroughs/<slug>.mjs`. **Do not add a static
  `src/app/play/<slug>/page.tsx`.** It would bypass the registry, its tests and
  the error boundaries.

---

## 8. Accessibility already handled

You inherit these; don't undo them.

- Focus rings: global `:focus-visible`, 2px `--primary`.
- The skip link ("Skip to content", `src/app/_components/SkipLink.tsx`, rendered
  by the root layout) targets `#main-content`. Every route renders that on the
  server; on a game page it is a plain wrapper with no `tabIndex`. Once a game
  has mounted, activating the link focuses `GameShell`'s
  `<main id="game-canvas">` directly, past the header.
- Metric announcements:
  - One debounced (600 ms) `aria-live="polite"` region per announcing
    `MetricReadout`.
  - The tweening number is `aria-hidden` on purpose. `<output>` has an implicit
    `role="status"`, so leaving it exposed would fire an announcement on every
    animation frame.
  - The same rule applies to `Slider`'s and `Dial`'s visible `<output>`: the
    control's `aria-valuetext` is the one spoken channel.
- Named failures (assertive, persistent) and WhyCard headlines (polite, 700 ms)
  are announced by `GameShell`, as §1 describes.
- Direction of travel is a glyph (▲ ▼ –) plus words in the announcement, never
  colour alone.
- 44px minimum targets on every interactive primitive. The Dial's − / + buttons
  are the single-pointer alternative to its drag (WCAG 2.5.7).
- `prefers-reduced-motion` is respected globally and in `useAnimatedNumber`, and
  `.flash-wrong` is off entirely under it.
- `scroll-padding-top` on shell pages keeps focus clear of the pinned metric
  (WCAG 2.4.11).
- Equations reach screen readers as MathML.

**What you still own per game:**

- Keyboard access to the canvas itself. A drag-only boundary or a pointer-only
  control fails the contract; every drag needs a keyboard path and a
  single-pointer path (Sort-It's tap-to-place handles, the Dial's buttons).
- SVG-to-pointer mapping through `getScreenCTM()`, not the bounding box.
  `preserveAspectRatio` letterboxing breaks bounding-box maths; see the two
  `field.ts` files.
- Screen-reader-only tables (a text version of a grid or graph): wrap the
  `<table>` in a `<div className="sr-only-live">` rather than putting the class
  on the table, because `overflow` clipping doesn't apply to table boxes.
