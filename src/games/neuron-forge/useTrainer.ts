"use client";

import { useCallback, useEffect, useRef } from "react";
import { useModel, type UseModelApi } from "@/engine";
import {
  MODEL_SEED,
  TRAIN_BATCH,
  TRAIN_EPOCHS,
  accuracyFromPredictions,
  architectureOf,
  buildModel,
  surfaceGrid,
  toMatrix,
  type Evaluation,
} from "./ml";
import { useNeuronForgeStore } from "./store";

/** Refresh the decision surface every N epochs while training. */
const SURFACE_EVERY = 16;

export interface TrainerApi {
  /** Resolves with the verdict, so the code lane can print it. */
  train: () => Promise<Evaluation | null>;
  stop: () => void;
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

  const model = useModel({
    // Read the architecture from the store at build time rather than closing over
    // it. A closure would be one render stale if the architecture and the train
    // click land in the same tick, which is exactly what the code lane does.
    build: () =>
      buildModel(
        architectureOf(useNeuronForgeStore.getState().layers),
        MODEL_SEED,
      ),
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

  const train = useCallback(async (): Promise<Evaluation | null> => {
    const state = useNeuronForgeStore.getState();
    if (state.training) return null;

    // Fresh weights every run. Without this, "train again" would silently
    // continue from where the last run stopped, and the same architecture would
    // stop giving the same answer.
    model.build();
    beginTraining();

    const trainSet = toMatrix(state.dataset.train);
    await model.train({
      xs: trainSet.xs,
      ys: trainSet.ys,
      epochs: TRAIN_EPOCHS,
      batchSize: TRAIN_BATCH,
      shuffle: false,
    });

    // Score on points the network has never seen — the only number that can
    // support a claim about what the architecture can represent.
    const testSet = toMatrix(state.dataset.test);
    const predictions = model.predict(testSet.xs);
    const accuracy =
      predictions === null
        ? null
        : accuracyFromPredictions(predictions, testSet.ys);

    return finishTraining({
      accuracy,
      surface: gridRef.current ? model.predict(gridRef.current) : null,
    });
  }, [model, beginTraining, finishTraining]);

  return {
    train,
    stop: model.stop,
    training: model.status === "training",
    epoch: model.epoch,
    totalEpochs: model.totalEpochs || TRAIN_EPOCHS,
    error: model.error,
  };
}
