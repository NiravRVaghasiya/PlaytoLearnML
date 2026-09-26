---
name: tfjs-model-lifecycle
description: Correct, memory-safe TensorFlow.js model patterns for GameML games — build, train with live metric callbacks, predict, and dispose. Use whenever a game trains or runs a model in-browser.
---

# TensorFlow.js Model Lifecycle

## Overview

GameML runs all ML client-side. Games build and destroy models repeatedly as players tweak and retry, so memory management and live-metric callbacks are critical. This skill defines the canonical patterns so every game behaves consistently and doesn't leak GPU memory.

Only five games use TF.js at all: Data Detox, Feature Forge, Neuron Forge, Overfit Tower Defense and Convolution Kitchen. If the algorithm isn't a tensor computation (k-means, a tree, a Q-table, a closed-form surface), write it in plain TypeScript and skip TF.js entirely.

## First choice: the engine's `useModel`

`src/engine/useModel.ts` already implements every rule below, and is tested for it. Data Detox, Neuron Forge and Overfit TD use it. Reach for it first; the full API is in `docs/ENGINE_API.md` §2.

```ts
import { useModel } from "@/engine/useModel";

const model = useModel({
  build: () => buildModel(2),                         // pure factory; may run twice
  onEpoch: ({ loss, accuracy }) => store.getState().setLive(loss, accuracy),
});

const last = await model.train({ xs, ys, epochs: 40, validationSplit: 0.2, rebuild: true });
if (last === null) return;   // taken over, reset, unmounted while queued or bad input: NOT trained, don't score
const probs = model.predict(grid);                    // plain arrays in, Float32Array out
```

What it guarantees:

- Tensors never cross into React state.
- Weights **and the optimizer** are disposed on rebuild, reset and unmount.
- A model is never disposed mid-fit.
- Overlapping `train()` calls are serialised, and the latest wins; the others resolve `null`, including a fit a newer `train()`, `build()` or `reset()` took over mid-run (no `onDone` for it either).
- A fit cut short by `stop()` or an unmount resolves with its last completed epoch, so check the epoch count before scoring it.
- `predict` runs inside `tf.tidy`.

Still keep a store-level guard (`if (state.training) return`) so a second click does what your game means.

## When a game drives TF.js directly

Feature Forge refits many small models per action, and Convolution Kitchen runs conv ops. Neither fits `useModel`'s one-model shape, so both call TF.js from `ml.ts`/`store.ts`. Then these rules are yours:

1. **Build** in a factory function (no side effects, returns the model). Seed the initializers if results must be reproducible across runs.
2. **Train** with `model.fit` and an `onEpochEnd` callback that pushes the REAL logs into the store. If a log is missing, report it as missing (`null` / `NaN`); never substitute `0`.
3. **Predict** inside `tf.tidy`, and read results out with `dataSync()`/`data()` before leaving the tidy.
4. **Dispose** everything you created: the model, its **optimizer**, and every tensor. `model.dispose()` does not free Adam's accumulator variables (two per weight). Wrap the fit in `try/finally`, so a throw can't leak.
5. **Guard against stale async work.**
   - Take a ticket or generation number before an `await`.
   - After it, drop the result if a newer request, a retry or an unmount has happened since.
   - Never dispose a model that a `fit` is still reading. Stop it (`model.stopTraining = true`), await the fit, then dispose.

```ts
import * as tf from "@tensorflow/tfjs";

export async function fitOnce(xs: number[][], ys: number[], epochs: number) {
  const model = buildModel(xs[0]!.length);
  const x = tf.tensor2d(xs);
  const y = tf.tensor2d(ys.map((v) => [v]));
  try {
    const history = await model.fit(x, y, { epochs, verbose: 0 });
    const loss = history.history.loss?.at(-1);
    return typeof loss === "number" ? loss : Number.NaN;
  } finally {
    const optimizer = model.optimizer;
    x.dispose();
    y.dispose();
    model.dispose();
    optimizer?.dispose();   // the leak model.dispose() leaves behind
  }
}
```

## Rules

- Always `tf.tidy` around inference; always dispose (model **and** optimizer) on reset/unmount.
- Feed the live metric through `onEpochEnd`, and never fake progress.
- Keep tensors out of React state: store scalars and arrays, not `tf.Tensor`.
- Check `tf.memory().numTensors` in tests, to assert no leaks after a round (`useModel`'s `tensorCount()`).
- Keep the umbrella `import * as tf from "@tensorflow/tfjs"` in the game's own modules. They are already lazy chunks (via `GameMount`), so don't import TF.js from anything the home page or Concept Library loads. That includes the `@/engine` barrel, which re-exports `useModel`.
- Without WebGL, TF.js falls back to its (much slower) CPU backend. Size `maxRunMs` and test timeouts for that, not for a GPU.

## Reference

- Build rules: `CLAUDE.md`. Engine API: `docs/ENGINE_API.md` §2. Metric display: `<MetricReadout>` (DESIGN.md §6).
