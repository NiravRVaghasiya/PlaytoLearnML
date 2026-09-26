"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";

/**
 * Shared TensorFlow.js model lifecycle (the `tfjs-model-lifecycle` skill).
 *
 * Games build and destroy models constantly — every retry, every slider nudge,
 * every architecture change. On a WebGL backend a leaked model is leaked GPU
 * memory, and fourteen games each leaking a little adds up to a tab that dies
 * mid-lesson. So this hook owns five things games keep getting wrong:
 *
 * 1. **No tensors in React state.** The model lives in a ref; the public API
 *    takes and returns plain arrays. Nothing that needs disposing can be
 *    captured by a render closure.
 * 2. **Live metrics come from `onEpochEnd`.** Real logs, every epoch. This is
 *    the wire that makes the metric move during training — never fake progress.
 * 3. **Inference is wrapped in `tf.tidy`.**
 * 4. **Disposal on rebuild, reset, and unmount.**
 * 5. **Overlapping trainings are serialised, latest wins.** A player mashing
 *    "Train" stops the in-flight fit and waits for it; of all the calls that
 *    queued up meanwhile, only the newest runs. Every other one resolves
 *    `null`, the stopped fit included, so two fits never share a model and no
 *    caller scores weights a newer request now owns.
 * 6. **A model is never disposed mid-fit.** `build()` or `reset()` during a
 *    fit stops it and frees the old weights once it has actually stopped, so
 *    the common "fresh weights, then train" (`build(); await train(…)`) is safe
 *    even while a previous fit is still running.
 *
 * Tests assert `tensorCount()` returns to baseline after a full round.
 */

export type ModelStatus = "idle" | "ready" | "training" | "error";

export interface EpochMetrics {
  /** Zero-based epoch index as reported by TF.js. */
  epoch: number;
  loss: number;
  /** Null when the model wasn't compiled with an accuracy metric. */
  accuracy: number | null;
  valLoss: number | null;
  valAccuracy: number | null;
}

export interface TrainRequest {
  /** Features, one row per sample. Plain arrays only. */
  xs: number[][];
  /** Targets. A flat array is treated as one column. */
  ys: number[][] | number[];
  epochs?: number;
  batchSize?: number;
  /** 0–1. Enables `valLoss`/`valAccuracy` in the epoch metrics. */
  validationSplit?: number;
  shuffle?: boolean;
  /**
   * Train from fresh weights: rebuild the model AFTER any in-flight fit has
   * been stopped and awaited, then fit. Equivalent to `build(); train()`, but
   * the rebuild happens once the queue is clear rather than while another fit
   * may still hold the model.
   */
  rebuild?: boolean;
}

export interface UseModelOptions {
  /**
   * Pure factory returning a compiled model. Called by `build()`. Must have no
   * side effects — it may be called more than once (React Strict Mode).
   */
  build: () => tf.LayersModel;
  /** Fired every epoch with real logs. Wire this to the game's store. */
  onEpoch?: (metrics: EpochMetrics) => void;
  /**
   * Fired once when a fit finishes or is stopped. Not fired for a fit that a
   * newer `train()`, `build()` or `reset()` took over: like its epoch updates,
   * its end is stale.
   */
  onDone?: (last: EpochMetrics | null) => void;
  /** Build on mount. Defaults to false so games control the timing. */
  autoBuild?: boolean;
}

export interface UseModelApi {
  status: ModelStatus;
  error: string | null;
  /** 1-based epoch count for display ("epoch 7 of 50"). */
  epoch: number;
  totalEpochs: number;
  latest: EpochMetrics | null;
  /** Build or rebuild. Disposes any previous model first. */
  build: () => void;
  /**
   * Fit the model. Resolves with the last epoch's metrics — partial if `stop()`
   * or an unmount cut it short, so check the epoch count — or `null` when the
   * request never ran or was taken over: bad input, a factory failure, a reset
   * or unmount while it waited, or a newer `train()`, `build()` or `reset()`
   * that superseded it, queued or mid-fit. Treat `null` as "not trained" —
   * don't score whatever the model currently holds.
   */
  train: (request: TrainRequest) => Promise<EpochMetrics | null>;
  /** Synchronous inference. Returns null when no model is built. */
  predict: (xs: number[][]) => Float32Array | null;
  /** Ask the current fit to stop at the next epoch boundary. */
  stop: () => void;
  /** Dispose the model and clear all state. */
  reset: () => void;
  /** Live tensor count from `tf.memory()` — used by leak tests. */
  tensorCount: () => number;
}

function readLogs(epoch: number, logs?: tf.Logs): EpochMetrics {
  const pick = (...keys: string[]): number | null => {
    for (const key of keys) {
      const value = logs?.[key];
      if (typeof value === "number" && Number.isFinite(value)) return value;
    }
    return null;
  };

  return {
    epoch,
    loss: pick("loss") ?? Number.NaN,
    accuracy: pick("acc", "accuracy"),
    valLoss: pick("val_loss"),
    valAccuracy: pick("val_acc", "val_accuracy"),
  };
}

/** Normalise targets to 2-D so `[0, 1, 1]` and `[[0], [1], [1]]` both work. */
function normalizeTargets(ys: number[][] | number[]): number[][] {
  if (ys.length === 0) return [];
  return Array.isArray(ys[0])
    ? (ys as number[][])
    : (ys as number[]).map((y) => [y]);
}

export function useModel(options: UseModelOptions): UseModelApi {
  const modelRef = useRef<tf.LayersModel | null>(null);
  /** In-flight fit, so a second train() can await it instead of racing. */
  const pendingRef = useRef<Promise<unknown> | null>(null);
  /** The model that in-flight fit is using, so it isn't disposed under it. */
  const fittingModelRef = useRef<tf.LayersModel | null>(null);
  /**
   * Ticket per train() call. Only the newest ticket may start a fit once the
   * queue clears; reset and unmount take a ticket too, which cancels anything
   * still waiting. A single `if (pending) await pending` was not enough: with
   * three overlapping calls, two waiters both woke up and raced into
   * `model.fit` ("another fit() call is ongoing").
   */
  const requestRef = useRef(0);
  /** Bumped per fit; stale epoch callbacks are dropped. */
  const generationRef = useRef(0);
  const mountedRef = useRef(true);
  /**
   * Sticky stop request. `model.fit()` clears `stopTraining` when it starts, so
   * a `stop()` issued in the window between calling `train()` and fit actually
   * beginning would otherwise be silently discarded. We keep the intent here and
   * re-apply it at every epoch boundary.
   */
  const stopRequestedRef = useRef(false);

  /**
   * Latest options, kept in a ref so inline `onEpoch={(m) => …}` props don't
   * invalidate `train`/`build` on every render. Synced in an effect rather than
   * during render — writing a ref while rendering is unsafe under concurrent
   * React, and every consumer of this ref runs from an event or an effect.
   */
  const optionsRef = useRef(options);
  useEffect(() => {
    optionsRef.current = options;
  });

  const [status, setStatus] = useState<ModelStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [epoch, setEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(0);
  const [latest, setLatest] = useState<EpochMetrics | null>(null);

  const disposeInstance = useCallback((model: tf.LayersModel) => {
    // `LayersModel.dispose()` frees the weights but NOT the optimizer, which
    // allocates its own accumulator variables during `fit` (Adam keeps two per
    // weight). Leaving those behind is the leak that eventually kills the tab,
    // so drop the optimizer too. Each `build()` makes a fresh optimizer, so
    // nothing else can be holding a reference.
    const optimizer = model.optimizer as tf.Optimizer | undefined;
    try {
      model.dispose();
    } catch {
      // Already disposed — nothing to do.
    }
    try {
      optimizer?.dispose();
    } catch {
      // Optimizers throw if disposed twice; harmless.
    }
  }, []);

  /**
   * Take the current model out of service and free it — now if it is idle, or
   * once its fit has actually stopped if it is mid-fit. Disposing weights that
   * `fit` is still reading throws "LayersVariable … is already disposed" inside
   * the fit and can leave the next fit reading freed memory.
   */
  const disposeModel = useCallback(() => {
    const model = modelRef.current;
    modelRef.current = null;
    if (!model) return;

    const pending = pendingRef.current;
    if (pending && fittingModelRef.current === model) {
      model.stopTraining = true;
      const release = () => disposeInstance(model);
      void pending.then(release, release);
      return;
    }
    disposeInstance(model);
  }, [disposeInstance]);

  const build = useCallback(() => {
    try {
      // A fit still running on the old model is now obsolete: drop its
      // remaining epoch callbacks so they can't report on weights nobody uses.
      if (pendingRef.current) generationRef.current += 1;
      // Rebuilding must not orphan the previous model's weights.
      disposeModel();
      modelRef.current = optionsRef.current.build();
      setStatus("ready");
      setError(null);
      setEpoch(0);
      setLatest(null);
    } catch (cause) {
      setStatus("error");
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [disposeModel]);

  const stop = useCallback(() => {
    stopRequestedRef.current = true;
    const model = modelRef.current;
    if (model) model.stopTraining = true;
  }, []);

  const reset = useCallback(() => {
    generationRef.current += 1;
    // Cancel any train() still queued behind the in-flight fit.
    requestRef.current += 1;
    stop();
    disposeModel();
    setStatus("idle");
    setError(null);
    setEpoch(0);
    setTotalEpochs(0);
    setLatest(null);
  }, [stop, disposeModel]);

  const train = useCallback(
    async (request: TrainRequest): Promise<EpochMetrics | null> => {
      // --- serialise overlapping trainings -------------------------------
      const ticket = ++requestRef.current;
      while (pendingRef.current) {
        const pending = pendingRef.current;
        stop();
        try {
          await pending;
        } catch {
          // The previous fit's failure is already reported; don't mask this one.
        }
        // A newer train(), a reset or an unmount arrived while this one
        // waited. The newest request wins; this one never runs.
        if (ticket !== requestRef.current) return null;
      }
      if (ticket !== requestRef.current) return null;

      if (request.rebuild || !modelRef.current) build();
      const model = modelRef.current;
      if (!model) return null;

      const targets = normalizeTargets(request.ys);
      if (request.xs.length === 0 || targets.length !== request.xs.length) {
        const message = `train(): xs has ${request.xs.length} rows, ys has ${targets.length}`;
        setStatus("error");
        setError(message);
        return null;
      }

      const epochs = request.epochs ?? 50;
      const generation = ++generationRef.current;

      // Clear the stop request only now — after any previous fit has been
      // stopped and awaited above — so this run starts with a clean slate.
      stopRequestedRef.current = false;
      model.stopTraining = false;
      setStatus("training");
      setError(null);
      setEpoch(0);
      setTotalEpochs(epochs);

      let last: EpochMetrics | null = null;
      /** Tensors this call owns. Collected in an array so the `finally` block
       *  can free them without TypeScript narrowing them away. */
      const owned: tf.Tensor[] = [];

      const run = (async () => {
        await tf.ready();

        const xs = tf.tensor2d(request.xs);
        const ys = tf.tensor2d(targets);
        owned.push(xs, ys);

        await model.fit(xs, ys, {
          epochs,
          batchSize: request.batchSize ?? 32,
          validationSplit: request.validationSplit,
          shuffle: request.shuffle ?? true,
          callbacks: {
            onEpochEnd: (epochIndex, logs) => {
              // Re-assert a stop that was requested before fit took over.
              if (stopRequestedRef.current) model.stopTraining = true;

              // A newer fit (or a reset) has taken over — drop this update.
              if (generation !== generationRef.current) return;

              const metrics = readLogs(epochIndex, logs);
              last = metrics;

              if (mountedRef.current) {
                setEpoch(epochIndex + 1);
                setLatest(metrics);
              }
              // The live-feedback wire: real logs into the game's store.
              optionsRef.current.onEpoch?.(metrics);
            },
          },
        });
      })();

      pendingRef.current = run;
      fittingModelRef.current = model;

      try {
        await run;
        // Taken over mid-fit: a newer train() is queued for this model (its
        // ticket is newer), or build()/reset() replaced it (the generation
        // moved). This fit's partial epochs describe weights nobody will use,
        // so it reports "not trained", like a request that never ran. An
        // unmount replaces nothing; like stop(), it reports how far it got.
        const superseded =
          mountedRef.current &&
          (ticket !== requestRef.current || generation !== generationRef.current);
        if (superseded) return null;
        if (mountedRef.current) setStatus("ready");
        optionsRef.current.onDone?.(last);
        return last;
      } catch (cause) {
        if (generation === generationRef.current && mountedRef.current) {
          setStatus("error");
          setError(cause instanceof Error ? cause.message : String(cause));
        }
        return null;
      } finally {
        // Training tensors are ours; free them regardless of outcome.
        for (const tensor of owned) tensor.dispose();
        owned.length = 0;
        if (pendingRef.current === run) {
          pendingRef.current = null;
          fittingModelRef.current = null;
        }
      }
    },
    [build, stop],
  );

  const predict = useCallback((xs: number[][]): Float32Array | null => {
    const model = modelRef.current;
    if (!model || xs.length === 0) return null;

    // tidy() disposes the input tensor and every intermediate for us.
    return tf.tidy(() => {
      const input = tf.tensor2d(xs);
      const output = model.predict(input) as tf.Tensor;
      return output.dataSync() as Float32Array;
    });
  }, []);

  const tensorCount = useCallback(() => tf.memory().numTensors, []);

  // Unmount: stop the fit and free the weights.
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      generationRef.current += 1;
      requestRef.current += 1;
      const model = modelRef.current;
      if (model) model.stopTraining = true;
      disposeModel();
    };
  }, [disposeModel]);

  useEffect(() => {
    if (optionsRef.current.autoBuild) build();
  }, [build]);

  return {
    status,
    error,
    epoch,
    totalEpochs,
    latest,
    build,
    train,
    predict,
    stop,
    reset,
    tensorCount,
  };
}
