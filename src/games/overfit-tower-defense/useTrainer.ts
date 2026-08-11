"use client";

import { useCallback, useEffect, useRef } from "react";
import { useModel, type UseModelApi } from "@/engine";
import {
  MODEL_SEED,
  TRAIN_BATCH,
  TRAIN_EPOCHS,
  accuracyFromPredictions,
  buildModel,
  toMatrix,
  type WaveResult,
} from "./ml";
import { useTowerDefenseStore } from "./store";

/** How often to measure both accuracies while training. */
const SAMPLE_EVERY = 10;

export interface Measurement {
  trainAccuracy: number;
  validationAccuracy: number;
  gap: number;
  bias: number;
}

export interface TrainerApi {
  /** Train the configured model and resolve the wave. */
  deploy: () => Promise<WaveResult | null>;
  /** Train and measure, without letting the wave touch the core. */
  trial: () => Promise<Measurement | null>;
  stop: () => void;
  training: boolean;
  epoch: number;
  totalEpochs: number;
  error: string | null;
}

/**
 * Trains the player's model and resolves the wave against it.
 *
 * Used once, in `index.tsx`, and handed to the controls, so both lanes deploy
 * through this single path.
 *
 * The live train/validation sampling is why this measures accuracy itself rather
 * than using `validationSplit`. A split would carve the validation set out of the
 * player's already-scarce training data, so the number on screen during training
 * would not be the number the wave is scored on — and on the famine wave it would
 * also take points away from the thing being learned.
 */
export function useNeuronDefenseTrainer(): TrainerApi {
  const beginTraining = useTowerDefenseStore((s) => s.beginTraining);
  const recordProgress = useTowerDefenseStore((s) => s.recordProgress);
  const resolveWave = useTowerDefenseStore((s) => s.resolveWave);
  const finishTrial = useTowerDefenseStore((s) => s.finishTrial);

  const apiRef = useRef<UseModelApi | null>(null);

  const model = useModel({
    // Read from the store at build time: the player may retune and deploy within
    // one tick from the code lane, and a closure would be a render stale.
    build: () => {
      const { modelComplexity, towers } = useTowerDefenseStore.getState();
      return buildModel(modelComplexity, towers, MODEL_SEED);
    },
    onEpoch: (metrics) => {
      if ((metrics.epoch + 1) % SAMPLE_EVERY !== 0) return;
      const api = apiRef.current;
      if (!api) return;

      const { dataset } = useTowerDefenseStore.getState();
      const train = toMatrix(dataset.train);
      const validation = toMatrix(dataset.validation);

      const trainPredictions = api.predict(train.xs);
      const validationPredictions = api.predict(validation.xs);
      if (!trainPredictions || !validationPredictions) return;

      recordProgress({
        epoch: metrics.epoch + 1,
        trainAccuracy: accuracyFromPredictions(trainPredictions, train.ys),
        validationAccuracy: accuracyFromPredictions(
          validationPredictions,
          validation.ys,
        ),
      });
    },
  });

  useEffect(() => {
    apiRef.current = model;
  });

  /** Shared by deploy and trial: build fresh, fit, measure both sides. */
  const runOnce = useCallback(async (): Promise<Measurement | null> => {
    const state = useTowerDefenseStore.getState();

    // Fresh weights every run: a wave must score the model the player
    // configured, not that plus whatever the previous run already learned.
    model.build();
    beginTraining();

    const train = toMatrix(state.dataset.train);
    await model.train({
      xs: train.xs,
      ys: train.ys,
      epochs: TRAIN_EPOCHS,
      batchSize: TRAIN_BATCH,
      shuffle: false,
    });

    const validation = toMatrix(state.dataset.validation);
    const trainPredictions = model.predict(train.xs);
    const validationPredictions = model.predict(validation.xs);
    if (!trainPredictions || !validationPredictions) return null;

    const trainAccuracy = accuracyFromPredictions(trainPredictions, train.ys);
    const validationAccuracy = accuracyFromPredictions(
      validationPredictions,
      validation.ys,
    );

    return {
      trainAccuracy,
      validationAccuracy,
      gap: Math.max(0, trainAccuracy - validationAccuracy),
      bias: Math.max(0, state.dataset.achievable - trainAccuracy),
    };
  }, [model, beginTraining]);

  const deploy = useCallback(async (): Promise<WaveResult | null> => {
    if (useTowerDefenseStore.getState().phase === "training") return null;
    const measured = await runOnce();
    if (!measured) return null;
    return resolveWave({
      trainAccuracy: measured.trainAccuracy,
      validationAccuracy: measured.validationAccuracy,
    });
  }, [runOnce, resolveWave]);

  const trial = useCallback(async (): Promise<Measurement | null> => {
    if (useTowerDefenseStore.getState().phase === "training") return null;
    const measured = await runOnce();
    if (!measured) return null;
    finishTrial({
      trainAccuracy: measured.trainAccuracy,
      validationAccuracy: measured.validationAccuracy,
    });
    return measured;
  }, [runOnce, finishTrial]);

  return {
    deploy,
    trial,
    stop: model.stop,
    training: model.status === "training",
    epoch: model.epoch,
    totalEpochs: model.totalEpochs || TRAIN_EPOCHS,
    error: model.error,
  };
}
