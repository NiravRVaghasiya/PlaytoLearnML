---
name: tfjs-model-lifecycle
description: Correct, memory-safe TensorFlow.js model patterns for GameML games — build, train with live metric callbacks, predict, and dispose. Use whenever a game trains or runs a model in-browser.
---

# TensorFlow.js Model Lifecycle

## Overview

GameML runs all ML client-side. Games build and destroy models repeatedly as players tweak and retry, so memory management and live-metric callbacks are critical. This skill defines the canonical patterns so every game's `ml.ts` behaves consistently and doesn't leak GPU memory.

## Workflow

1. **Build** the model in a factory function (no side effects, returns the model).
2. **Train** with `model.fit` using the `onEpochEnd` callback to push the live metric into the store every epoch — this is what makes the metric "move" during play.
3. **Predict** with `tf.tidy` to auto-dispose intermediate tensors.
4. **Dispose** the model and any retained tensors on reset/unmount. Never leave models in memory between rounds.
5. **Guard** against overlapping trainings (a player spamming "train"): cancel/await the previous fit before starting a new one.

## Canonical patterns

```ts
import * as tf from '@tensorflow/tfjs';

export function buildModel(inputDim: number): tf.LayersModel {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 8, activation: 'relu', inputShape: [inputDim] }));
  m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  m.compile({ optimizer: tf.train.adam(0.03), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  return m;
}

export async function train(
  model: tf.LayersModel,
  xs: tf.Tensor, ys: tf.Tensor,
  onMetric: (epoch: number, loss: number, acc: number) => void,
) {
  await model.fit(xs, ys, {
    epochs: 50,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        onMetric(epoch, logs?.loss ?? 0, logs?.acc ?? logs?.accuracy ?? 0);
      },
    },
  });
}

export function predict(model: tf.LayersModel, x: tf.Tensor): Float32Array {
  return tf.tidy(() => (model.predict(x) as tf.Tensor).dataSync() as Float32Array);
}

export function dispose(model: tf.LayersModel, ...tensors: tf.Tensor[]) {
  model.dispose();
  tensors.forEach(t => t.dispose());
}
```

## Rules

- Always `tf.tidy` around inference; always `dispose` on reset/unmount.
- Feed the live metric through `onEpochEnd` — never fake progress.
- Keep tensors out of React state (store scalars/arrays, not `tf.Tensor`).
- Check `tf.memory().numTensors` in tests to assert no leaks after a round.

## Reference

- Build rules: `CLAUDE.md`. Metric display: `<MetricReadout>` (DESIGN.md).
