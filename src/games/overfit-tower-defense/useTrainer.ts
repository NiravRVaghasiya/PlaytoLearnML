"use client";

import { useCallback, useEffect, useRef } from "react";
import { useModel, type UseModelApi } from "@/engine";
import {
  MODEL_SEED,
  SAMPLE_EVERY,
  TRAIN_BATCH,
  TRAIN_EPOCHS,
  accuracyFromPredictions,
  buildModel,
  fitEnd,
  toMatrix,
  type WaveResult,
} from "./ml";
import { useTowerDefenseStore } from "./store";

export interface Measurement {
  trainAccuracy: number;
  validationAccuracy: number;
  gap: number;
  bias: number;
}

export interface RunOptions {
  /** Set by the code lane's `api.deploy()`, so its clear is credited to it. */
  fromCode?: boolean;
}

export interface TrainerApi {
  /**
   * Train the configured model and resolve the wave. Null when nothing was
   * scored: no open wave, the fit was stopped, or a Retry superseded it.
   */
  deploy: (options?: RunOptions) => Promise<WaveResult | null>;
  /** Train and measure, without letting the wave touch the core. */
  trial: () => Promise<Measurement | null>;
  stop: () => void;
  /** Stop any in-flight fit and wait until it has fully resolved. */
  stopAndWait: () => Promise<void>;
  /**
   * Why the latest `deploy` or `trial` came back null: the player stopped it,
   * or the fit failed (the message is in `error`). Null for any other reason.
   * Read from a ref, so a script that awaited the call sees the current answer.
   */
  unscoredReason: () => "stopped" | "failed" | null;
  training: boolean;
  epoch: number;
  totalEpochs: number;
  error: string | null;
}

type RunOutcome =
  | { kind: "measured"; measured: Measurement; attempt: number }
  | { kind: "stopped"; epochsRun: number; attempt: number }
  /** The fit threw, or the model was gone before it could be measured. */
  | { kind: "failed"; attempt: number }
  /** Superseded by a restart, or no run could start. Nothing to report. */
  | { kind: "void" };

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
  const abortTraining = useTowerDefenseStore((s) => s.abortTraining);

  const apiRef = useRef<UseModelApi | null>(null);
  /** The in-flight run, so the next one can wait for it to wind down. */
  const inFlightRef = useRef<Promise<RunOutcome> | null>(null);
  /** That run's `attempt` token, so its live samples are tagged with it. */
  const attemptRef = useRef(0);
  /** See `TrainerApi.unscoredReason`. */
  const unscoredRef = useRef<"stopped" | "failed" | null>(null);

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

      // The store drops the sample unless this run is still the one training.
      recordProgress(
        {
          epoch: metrics.epoch + 1,
          trainAccuracy: accuracyFromPredictions(trainPredictions, train.ys),
          validationAccuracy: accuracyFromPredictions(
            validationPredictions,
            validation.ys,
          ),
        },
        attemptRef.current,
      );
    },
  });

  useEffect(() => {
    apiRef.current = model;
  });

  const { stop } = model;

  /** Shared by deploy and trial: build fresh, fit, measure both sides. */
  const runOnce = useCallback(async (): Promise<RunOutcome> => {
    // A Retry can leave the previous fit winding down while the store has
    // already moved on. Building now would dispose the model under it, so stop
    // it and wait for it to resolve first; its result is then dropped by token.
    const previous = inFlightRef.current;
    if (previous) {
      stop();
      await previous.catch(() => undefined);
    }

    const state = useTowerDefenseStore.getState();
    if (state.phase === "training") return { kind: "void" };

    const run = (async (): Promise<RunOutcome> => {
      // Fresh weights every run: a wave must score the model the player
      // configured, not that plus whatever the previous run already learned.
      model.build();
      const attempt = beginTraining();
      attemptRef.current = attempt;

      const train = toMatrix(state.dataset.train);
      const last = await model.train({
        xs: train.xs,
        ys: train.ys,
        epochs: TRAIN_EPOCHS,
        batchSize: TRAIN_BATCH,
        shuffle: false,
      });

      if (useTowerDefenseStore.getState().attempt !== attempt) {
        return { kind: "void" };
      }
      // `train` resolves normally on Stop and with null if the fit threw.
      // Neither is a finished model, so neither may be scored — and they are
      // reported apart, because only one of them is something the player did.
      const ended = fitEnd(last);
      if (ended.kind === "failed") return { kind: "failed", attempt };
      if (ended.kind === "stopped") {
        return { kind: "stopped", epochsRun: ended.epochsRun, attempt };
      }

      const validation = toMatrix(state.dataset.validation);
      const trainPredictions = model.predict(train.xs);
      const validationPredictions = model.predict(validation.xs);
      if (!trainPredictions || !validationPredictions) {
        return { kind: "failed", attempt };
      }

      const trainAccuracy = accuracyFromPredictions(trainPredictions, train.ys);
      const validationAccuracy = accuracyFromPredictions(
        validationPredictions,
        validation.ys,
      );

      return {
        kind: "measured",
        attempt,
        measured: {
          trainAccuracy,
          validationAccuracy,
          gap: Math.max(0, trainAccuracy - validationAccuracy),
          bias: Math.max(0, state.dataset.achievable - trainAccuracy),
        },
      };
    })();

    inFlightRef.current = run;
    try {
      return await run;
    } finally {
      if (inFlightRef.current === run) inFlightRef.current = null;
    }
  }, [model, stop, beginTraining]);

  const deploy = useCallback(
    async ({ fromCode = false }: RunOptions = {}): Promise<WaveResult | null> => {
      unscoredRef.current = null;
      // Only an open wave can be fought. The store refuses a resolve for any
      // other phase too; checking here saves training a model for nothing.
      if (useTowerDefenseStore.getState().phase !== "tuning") return null;
      const outcome = await runOnce();
      if (outcome.kind === "void") return null;
      if (outcome.kind === "stopped" || outcome.kind === "failed") {
        unscoredRef.current = outcome.kind;
        abortTraining(outcome.attempt, {
          epochsRun: outcome.kind === "stopped" ? outcome.epochsRun : 0,
          deploy: true,
          failed: outcome.kind === "failed",
        });
        return null;
      }
      return resolveWave(outcome.measured, outcome.attempt, { fromCode });
    },
    [runOnce, resolveWave, abortTraining],
  );

  const trial = useCallback(async (): Promise<Measurement | null> => {
    unscoredRef.current = null;
    if (useTowerDefenseStore.getState().phase === "training") return null;
    const outcome = await runOnce();
    if (outcome.kind === "void") return null;
    if (outcome.kind === "stopped" || outcome.kind === "failed") {
      unscoredRef.current = outcome.kind;
      abortTraining(outcome.attempt, {
        epochsRun: outcome.kind === "stopped" ? outcome.epochsRun : 0,
        deploy: false,
        failed: outcome.kind === "failed",
      });
      return null;
    }
    finishTrial(outcome.measured, outcome.attempt);
    return outcome.measured;
  }, [runOnce, finishTrial, abortTraining]);

  const stopAndWait = useCallback(async () => {
    const pending = inFlightRef.current;
    if (!pending) return;
    stop();
    await pending.catch(() => undefined);
  }, [stop]);

  const unscoredReason = useCallback(() => unscoredRef.current, []);

  return {
    deploy,
    trial,
    stop,
    stopAndWait,
    unscoredReason,
    training: model.status === "training",
    epoch: model.epoch,
    totalEpochs: model.totalEpochs || TRAIN_EPOCHS,
    error: model.error,
  };
}
