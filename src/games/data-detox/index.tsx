"use client";

import { useCallback, useEffect, useRef } from "react";
import { Sparkles } from "lucide-react";
import { Button, type MetricSpec } from "@/components";
import { GameShell } from "@/engine/GameShell";
import {
  HIGH_SCORE_THRESHOLD,
  levelFromXp,
  useProgression,
} from "@/engine/progression";
import { useModel } from "@/engine/useModel";
import { getGameMeta } from "@/lib/catalog";
import {
  BATCH_SIZE,
  MATH_CODE,
  MATH_EQUATION,
  MATH_NOTES,
  MAX_BALANCE_DRIFT,
  TRAIN_BATCH,
  TRAIN_EPOCHS,
  WIN_ACCURACY,
  accuracyFromPredictions,
  buildModel,
} from "./ml";
import { SLUG, useAttemptClock, useDataDetoxStore } from "./store";
import { VisualLane } from "./VisualLane";
import { CodeLane } from "./CodeLane";

/**
 * The engine's star rule, stated exactly: ★ cleared, ★★ best score at least
 * HIGH_SCORE_THRESHOLD (score is accuracy minus the time penalty — not the
 * game's 70% win bar), ★★★ that plus a pipeline scored from the code lane.
 */
const STAR_CRITERIA = [
  "Clean a dataset",
  `Score ${Math.round(HIGH_SCORE_THRESHOLD * 100)}% or better`,
  `Score ${Math.round(HIGH_SCORE_THRESHOLD * 100)}% with a pipeline checked from the code lane`,
];

function Controls({
  retrain,
  onNewRound,
}: {
  retrain: () => Promise<number | null>;
  onNewRound: () => void;
}) {
  const pipeline = useDataDetoxStore((s) => s.pipeline);
  const retrains = useDataDetoxStore((s) => s.retrains);
  const training = useDataDetoxStore((s) => s.training);
  const check = useDataDetoxStore((s) => s.check);
  const won = useDataDetoxStore((s) => s.won);
  const round = useDataDetoxStore((s) => s.round);

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h3 className="mb-2 text-sm font-medium">Your pipeline so far</h3>
        <dl className="grid grid-cols-2 gap-2 font-mono text-xs">
          {(
            [
              ["keep", pipeline.binCounts.keep],
              ["impute", pipeline.binCounts.impute],
              ["cap", pipeline.binCounts.cap],
              ["drop", pipeline.binCounts.drop],
            ] as const
          ).map(([label, count]) => (
            <div
              key={label}
              className="rounded-md border border-border bg-surface-2 px-2 py-1"
            >
              <dt className="text-text-muted">{label}</dt>
              <dd className="text-base tabular-nums">{count}</dd>
            </div>
          ))}
        </dl>
      </div>

      <div className="border-t border-border pt-4">
        <h3 className="mb-1 text-sm font-medium">What the model will see</h3>
        <ul className="space-y-1 font-mono text-xs text-text-muted">
          <li>{pipeline.kept} training rows</li>
          <li>{pipeline.keptWithNaiveFill} still carry a blank read as 0</li>
          <li>{pipeline.keptWithOutlier} still carry an extreme value</li>
          <li>
            class balance drift{" "}
            <span
              className={
                Math.abs(pipeline.balanceDrift) > MAX_BALANCE_DRIFT
                  ? "text-wrong"
                  : undefined
              }
            >
              {(pipeline.balanceDrift * 100).toFixed(0)}%
            </span>
          </li>
        </ul>
        <p className="mt-2 text-xs text-text-muted">
          Retrains every {BATCH_SIZE} decisions. {retrains} so far
          {training ? " · training…" : ""}.
        </p>
      </div>

      <div className="flex flex-col gap-2 border-t border-border pt-4">
        <Button
          variant="secondary"
          onClick={() => void retrain()}
          disabled={training || pipeline.kept === 0}
        >
          Retrain now
        </Button>
        <Button
          variant="primary"
          // The rail is visible from both lanes, so a press here is credited to
          // the visual lane whichever tab is showing; api.check() is the
          // code-lane route.
          onClick={() => check("visual")}
          disabled={training}
          icon={<Sparkles className="size-4" />}
        >
          Score this pipeline
        </Button>
        {won ? (
          <Button variant="secondary" onClick={onNewRound}>
            New batch ({round} cleaned)
          </Button>
        ) : null}
      </div>
    </div>
  );
}

export default function DataDetox() {
  const meta = getGameMeta(SLUG);

  const pipeline = useDataDetoxStore((s) => s.pipeline);
  const modelAccuracy = useDataDetoxStore((s) => s.modelAccuracy);
  const training = useDataDetoxStore((s) => s.training);
  const stale = useDataDetoxStore(
    (s) => s.modelAccuracy !== null && s.trainedVersion !== s.pipelineVersion,
  );
  const lastEvaluation = useDataDetoxStore((s) => s.lastEvaluation);
  const failure = useDataDetoxStore((s) => s.failure);
  const whyCard = useDataDetoxStore((s) => s.whyCard);
  const lane = useDataDetoxStore((s) => s.lane);
  const setLane = useDataDetoxStore((s) => s.setLane);
  const reset = useDataDetoxStore((s) => s.reset);
  const newRound = useDataDetoxStore((s) => s.newRound);
  const won = useDataDetoxStore((s) => s.won);

  const hydrate = useProgression((s) => s.hydrate);
  const hydrated = useProgression((s) => s.hydrated);
  const lastGain = useProgression((s) => s.lastGain);
  const games = useProgression((s) => s.games);
  const xp = useProgression((s) => s.xp);
  const level = levelFromXp(xp);

  useEffect(() => {
    if (!hydrated) void hydrate();
  }, [hydrated, hydrate]);

  /**
   * The downstream model. Owned here rather than in the store because TF.js
   * tensors must never live in React state (the tfjs-model-lifecycle skill), and
   * `useModel` handles disposal, overlapping retrains, and unmount for us.
   */
  const model = useModel({ build: () => buildModel(1) });

  /** The fit currently running, so the next retrain can wait for it. */
  const inFlightRef = useRef<Promise<unknown> | null>(null);

  /**
   * Retrain on the current pipeline and push held-out accuracy into the store.
   *
   * Resolves with the accuracy, or null when this fit's result was not used —
   * a newer retrain, a Retry or a new batch took over, or the page unmounted.
   *
   * The ticket is claimed BEFORE waiting on any running fit, so from that
   * moment the older fit's result is stale in the store's eyes, whichever of
   * the two finishes first. And the model is only rebuilt once the older fit
   * has actually stopped: `build()` disposes the current model, and disposing
   * one mid-fit throws inside TF.js and left the next retrain predicting with
   * an untrained net.
   */
  const retrain = useCallback(async (): Promise<number | null> => {
    const store = useDataDetoxStore.getState();
    if (store.pipeline.kept === 0) return null;

    const ticket = store.beginTraining();

    while (inFlightRef.current) {
      model.stop();
      await inFlightRef.current.catch(() => undefined);
    }
    if (!useDataDetoxStore.getState().isCurrentTraining(ticket.id)) return null;

    const run = (async (): Promise<number | null> => {
      // Rebuild every time: this is a fresh fit on a different dataset, not a
      // continuation. Reusing warm weights would let an earlier, dirtier
      // pipeline flatter a later one.
      model.build();
      const last = await model.train({
        xs: ticket.xs,
        ys: ticket.ys,
        epochs: TRAIN_EPOCHS,
        batchSize: TRAIN_BATCH,
        shuffle: false,
      });
      // A fit that failed, or was stopped short of its epochs, is not the
      // model the meter claims to describe.
      if (!last || last.epoch + 1 < TRAIN_EPOCHS) return null;

      const predictions = model.predict(ticket.test.xs);
      return predictions
        ? accuracyFromPredictions(predictions, ticket.test.ys)
        : null;
    })();

    inFlightRef.current = run;
    let accuracy: number | null = null;
    try {
      accuracy = await run;
    } catch {
      accuracy = null;
    } finally {
      if (inFlightRef.current === run) inFlightRef.current = null;
    }

    const latest = useDataDetoxStore.getState();
    if (accuracy === null) {
      latest.abandonTraining(ticket.id);
      return null;
    }
    return latest.reportAccuracy(accuracy, ticket.id) ? accuracy : null;
  }, [model]);

  // Spec: "every N rows, a downstream model retrains". Re-checked whenever a
  // fit finishes too, so rows sorted during it are never left untrained.
  const sinceRetrain = useDataDetoxStore((s) => s.sinceRetrain);
  const pipelineVersion = useDataDetoxStore((s) => s.pipelineVersion);
  const retrainRef = useRef(retrain);
  useEffect(() => {
    retrainRef.current = retrain;
  });

  useEffect(() => {
    if (useDataDetoxStore.getState().needsRetrain()) {
      void retrainRef.current();
    }
  }, [sinceRetrain, training, pipelineVersion]);

  /** Retry and next batch: stop the running fit so it stops costing CPU. The
   *  store's ticket bump is what guarantees its result is never written. */
  const retry = useCallback(() => {
    model.stop();
    reset();
  }, [model, reset]);
  const nextBatch = useCallback(() => {
    model.stop();
    newRound();
  }, [model, newRound]);

  // The spec's `timeElapsed`, feeding a small capped penalty. Runs only while
  // the game is on screen — see useAttemptClock.
  useAttemptClock();

  /**
   * The live metric (pedagogy contract #3): held-out accuracy of a real model,
   * retrained on whatever the player's cleaning decisions produced.
   */
  const metric: MetricSpec = {
    label: "Accuracy",
    value: modelAccuracy ?? Number.NaN,
    format: "percent",
    goodDirection: "up",
    caption:
      modelAccuracy === null
        ? training
          ? "training…"
          : `sort ${BATCH_SIZE} rows to train`
        : training
          ? "retraining…"
          : stale && pipeline.undecided === 0
            ? "pipeline changed — retrain to score it"
            : "on held-out rows",
  };

  const secondaryMetrics: MetricSpec[] = [
    {
      label: "Rows kept",
      value: pipeline.kept,
      format: "integer",
      goodDirection: "up",
      caption: `${pipeline.dropped} dropped`,
    },
    {
      label: "Dirt left in",
      value: pipeline.keptWithNaiveFill + pipeline.keptWithOutlier,
      format: "integer",
      goodDirection: "down",
      caption: "blanks + extremes",
    },
    {
      label: "Score",
      value: lastEvaluation?.score ?? Number.NaN,
      format: "percent",
      goodDirection: "up",
      // The seconds shown are the ones the penalty was charged for, not the
      // live clock, so the caption can't drift away from the number beside it.
      caption:
        lastEvaluation === null || lastEvaluation.score === null
          ? `need ${Math.round(WIN_ACCURACY * 100)}%`
          : `−${(lastEvaluation.timePenalty * 100).toFixed(0)}% time (${lastEvaluation.secondsElapsed}s)`,
      state: won ? "good" : undefined,
    },
  ];

  return (
    <GameShell
      slug={SLUG}
      title={meta?.title ?? "Data Detox"}
      metric={metric}
      secondaryMetrics={secondaryMetrics}
      math={{
        equation: MATH_EQUATION,
        code: MATH_CODE,
        codeLanguage: "javascript",
        notes: MATH_NOTES,
      }}
      controls={<Controls retrain={retrain} onNewRound={nextBatch} />}
      visual={<VisualLane />}
      code={<CodeLane retrain={retrain} />}
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
      onRetry={retry}
      onNext={won ? nextBatch : undefined}
      nextLabel="Next batch"
    />
  );
}
