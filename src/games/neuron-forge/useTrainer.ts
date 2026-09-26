"use client";

import { useCallback, useEffect, useRef } from "react";
import type * as tf from "@tensorflow/tfjs";
import { useModel, type UseModelApi } from "@/engine";
import {
  MODEL_SEED,
  TRAIN_BATCH,
  TRAIN_EPOCHS,
  accuracyFromPredictions,
  architectureOf,
  buildModel,
  diagnoseCollapse,
  surfaceGrid,
  toMatrix,
  type Collapse,
  type Evaluation,
} from "./ml";
import { useNeuronForgeStore } from "./store";

/** Refresh the decision surface every N epochs while training. */
const SURFACE_EVERY = 16;

export interface TrainOptions {
  /** Set by the code lane's `api.train()`, so its clear is credited to it. */
  fromCode?: boolean;
}

export interface TrainerApi {
  /**
   * Resolves with the verdict, so the code lane can print it. Null when no run
   * started (one was already in flight) or the run was superseded.
   */
  train: (options?: TrainOptions) => Promise<Evaluation | null>;
  stop: () => void;
  /** Stop any in-flight fit and wait until it has fully resolved. */
  stopAndWait: () => Promise<void>;
  training: boolean;
  epoch: number;
  totalEpochs: number;
  error: string | null;
}

/**
 * Drives real TF.js training for whichever lane is showing.
 *
 * Called exactly once, from `index.tsx`, and handed to the controls. Both lanes
 * therefore train through this one path — the code lane sets the architecture and
 * presses the same button, so there is no second training implementation that
 * could drift from the first.
 */
export function useNeuronTrainer(): TrainerApi {
  const beginTraining = useNeuronForgeStore((s) => s.beginTraining);
  const recordEpoch = useNeuronForgeStore((s) => s.recordEpoch);
  const setSurface = useNeuronForgeStore((s) => s.setSurface);
  const finishTraining = useNeuronForgeStore((s) => s.finishTraining);

  /** Built once: the grid never changes. */
  const gridRef = useRef<number[][] | null>(null);
  gridRef.current ??= surfaceGrid();

  /**
   * The model API, for use inside `onEpoch`. Synced in an effect rather than
   * assigned during render, matching `useModel`'s own reasoning about refs.
   */
  const apiRef = useRef<UseModelApi | null>(null);

  /**
   * The model the last `build()` produced. `useModel` deliberately hides its
   * model behind plain-array methods, but the dead-network check has to look
   * INSIDE the network — at each hidden layer's output — which `predict` cannot
   * do. The factory hands the model to `useModel` and keeps a reference, read
   * only right after a fit this hook itself started, while `useModel` still owns
   * it. Nothing here disposes it.
   */
  const builtRef = useRef<tf.LayersModel | null>(null);

  /** The in-flight `train()` call, so a Retry can stop it and wait it out. */
  const inFlightRef = useRef<Promise<Evaluation | null> | null>(null);

  const model = useModel({
    // Read the architecture from the store at build time rather than closing over
    // it. A closure would be one render stale if the architecture and the train
    // click land in the same tick, which is exactly what the code lane does.
    build: () => {
      const built = buildModel(
        architectureOf(useNeuronForgeStore.getState().layers),
        MODEL_SEED,
      );
      builtRef.current = built;
      return built;
    },
    onEpoch: (metrics) => {
      recordEpoch(metrics.epoch, metrics.loss);
      // The spec asks for a live surface, not just a final one: watching the
      // boundary fail to bend is the lesson for the linear architectures.
      if ((metrics.epoch + 1) % SURFACE_EVERY === 0 && gridRef.current) {
        const predictions = apiRef.current?.predict(gridRef.current);
        if (predictions) setSurface(predictions);
      }
    },
  });

  useEffect(() => {
    apiRef.current = model;
  });

  const train = useCallback(
    ({ fromCode = false }: TrainOptions = {}): Promise<Evaluation | null> => {
      const state = useNeuronForgeStore.getState();
      if (state.training) return Promise.resolve(null);

      const run = (async (): Promise<Evaluation | null> => {
        // Fresh weights every run. Without this, "train again" would silently
        // continue from where the last run stopped, and the same architecture
        // would stop giving the same answer.
        model.build();
        const attempt = beginTraining();
        const architecture = architectureOf(state.layers);

        const trainSet = toMatrix(state.dataset.train);
        const last = await model.train({
          xs: trainSet.xs,
          ys: trainSet.ys,
          epochs: TRAIN_EPOCHS,
          batchSize: TRAIN_BATCH,
          shuffle: false,
        });
        // `train` resolves normally on Stop, and with null if the fit threw.
        // Both are reported, neither is scored as if it had finished.
        const epochsRun = last === null ? 0 : last.epoch + 1;

        // Score on points the network has never seen — the only number that can
        // support a claim about what the architecture can represent.
        const testSet = toMatrix(state.dataset.test);
        const predictions = last === null ? null : model.predict(testSet.xs);
        const accuracy =
          predictions === null
            ? null
            : accuracyFromPredictions(predictions, testSet.ys);

        let collapse: Collapse | null = null;
        const built = builtRef.current;
        if (predictions !== null && built !== null && epochsRun >= TRAIN_EPOCHS) {
          try {
            collapse = diagnoseCollapse(
              built,
              architecture,
              trainSet.xs,
              predictions,
            );
          } catch {
            // The model was disposed under us (the page unmounted mid-run).
            // No diagnosis is better than a wrong one.
            collapse = null;
          }
        }

        return finishTraining(
          {
            accuracy,
            surface:
              last !== null && gridRef.current
                ? model.predict(gridRef.current)
                : null,
            collapse,
            epochsRun,
            fromCode,
          },
          attempt,
        );
      })();

      inFlightRef.current = run;
      const settle = () => {
        if (inFlightRef.current === run) inFlightRef.current = null;
      };
      // Both handlers, so this bookkeeping branch can never become an
      // unhandled rejection of its own; the caller still sees `run` reject.
      run.then(settle, settle);
      return run;
    },
    [model, beginTraining, finishTraining],
  );

  const { stop } = model;
  const stopAndWait = useCallback(async () => {
    const pending = inFlightRef.current;
    if (!pending) return;
    stop();
    try {
      await pending;
    } catch {
      // The failure is already on screen via `model.error`.
    }
  }, [stop]);

  return {
    train,
    stop,
    stopAndWait,
    training: model.status === "training",
    epoch: model.epoch,
    totalEpochs: model.totalEpochs || TRAIN_EPOCHS,
    error: model.error,
  };
}
