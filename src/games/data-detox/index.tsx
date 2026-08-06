"use client";

import { useCallback, useEffect, useRef } from "react";
import { Sparkles } from "lucide-react";
import { Button, type MetricSpec } from "@/components";
import { GameShell } from "@/engine/GameShell";
import { levelFromXp, useProgression } from "@/engine/progression";
import { useModel } from "@/engine/useModel";
import { getGameMeta } from "@/lib/catalog";
import {
  BATCH_SIZE,
  MATH_CODE,
  MATH_EQUATION,
  MATH_NOTES,
  TRAIN_BATCH,
  TRAIN_EPOCHS,
  WIN_ACCURACY,
  accuracyFromPredictions,
  buildModel,
} from "./ml";
import { SLUG, useDataDetoxStore } from "./store";
import { VisualLane } from "./VisualLane";
import { CodeLane } from "./CodeLane";

const STAR_CRITERIA = [
  "Clean a dataset",
  `Score ${Math.round(WIN_ACCURACY * 100)}% or better`,
  "Clean one from the code lane",
];

function Controls({ retrain }: { retrain: () => Promise<number> }) {
  const pipeline = useDataDetoxStore((s) => s.pipeline);
  const retrains = useDataDetoxStore((s) => s.retrains);
  const training = useDataDetoxStore((s) => s.training);
  const check = useDataDetoxStore((s) => s.check);
  const newRound = useDataDetoxStore((s) => s.newRound);
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
                Math.abs(pipeline.balanceDrift) > 0.12 ? "text-wrong" : undefined
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
          onClick={check}
          disabled={training}
          icon={<Sparkles className="size-4" />}
        >
          Score this pipeline
        </Button>
        {won ? (
          <Button variant="secondary" onClick={newRound}>
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
  const timeElapsed = useDataDetoxStore((s) => s.timeElapsed);
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

  /** Retrain on the current pipeline and push held-out accuracy into the store. */
  const retrain = useCallback(async (): Promise<number> => {
    const state = useDataDetoxStore.getState();
    if (state.pipeline.kept === 0) return 0;

    state.beginTraining();

    // Rebuild every time: this is a fresh fit on a different dataset, not a
    // continuation. Reusing warm weights would let an earlier, dirtier pipeline
    // flatter a later one.
    model.build();
    await model.train({
      xs: state.pipeline.xs,
      ys: state.pipeline.ys,
      epochs: TRAIN_EPOCHS,
      batchSize: TRAIN_BATCH,
      shuffle: false,
    });

    const test = state.testSet();
    const predictions = model.predict(test.xs);
    const accuracy = predictions
      ? accuracyFromPredictions(predictions, test.ys)
      : 0;

    useDataDetoxStore.getState().reportAccuracy(accuracy);
    return accuracy;
  }, [model]);

  // Spec: "every N rows, a downstream model retrains".
  const sinceRetrain = useDataDetoxStore((s) => s.sinceRetrain);
  const retrainRef = useRef(retrain);
  useEffect(() => {
    retrainRef.current = retrain;
  });

  useEffect(() => {
    if (useDataDetoxStore.getState().needsRetrain()) {
      void retrainRef.current();
    }
  }, [sinceRetrain]);

  // The spec's `timeElapsed`, feeding a small capped penalty.
  const startedAt = useDataDetoxStore((s) => s.round);
  useEffect(() => {
    const began = Date.now();
    const timer = setInterval(() => {
      useDataDetoxStore
        .getState()
        .setElapsed(Math.round((Date.now() - began) / 1000));
    }, 1000);
    return () => clearInterval(timer);
  }, [startedAt]);

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
        ? `sort ${BATCH_SIZE} rows to train`
        : training
          ? "retraining…"
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
      caption:
        lastEvaluation === null
          ? `need ${Math.round(WIN_ACCURACY * 100)}%`
          : `−${(lastEvaluation.timePenalty * 100).toFixed(0)}% time (${timeElapsed}s)`,
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
      controls={<Controls retrain={retrain} />}
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
      onRetry={reset}
      onNext={won ? newRound : undefined}
      nextLabel="Next batch"
    />
  );
}
