"use client";

import { useEffect, useMemo } from "react";
import { ChevronRight, RotateCcw, Shield, Square, Zap } from "lucide-react";
import { Button, Slider, type MetricSpec } from "@/components";
import { GameShell } from "@/engine/GameShell";
import {
  HIGH_SCORE_THRESHOLD,
  levelFromXp,
  useProgression,
} from "@/engine/progression";
import { getGameMeta } from "@/lib/catalog";
import {
  CORE_MAX_HP,
  MATH_CODE,
  MATH_EQUATION,
  MATH_NOTES,
  MAX_COMPLEXITY,
  MIN_COMPLEXITY,
  WAVES,
  parameterCount,
  regularizationOf,
} from "./ml";
import {
  SLUG,
  biasOf,
  currentWave,
  gapOf,
  towerCount,
  useTowerDefenseStore,
} from "./store";
import { RegTowerPalette } from "./RegTowerPalette";
import { WaveTracker } from "./WaveTracker";
import { VisualLane } from "./VisualLane";
import { CodeLane } from "./CodeLane";
import { useNeuronDefenseTrainer, type TrainerApi } from "./useTrainer";

const STAR_CRITERIA = [
  "Survive all five waves",
  `Score ${Math.round(HIGH_SCORE_THRESHOLD * 100)}% or better by keeping the core healthy`,
  "Clear a run from the code lane",
];

function Controls({ trainer }: { trainer: TrainerApi }) {
  const complexity = useTowerDefenseStore((s) => s.modelComplexity);
  const phase = useTowerDefenseStore((s) => s.phase);
  const wave = useTowerDefenseStore(currentWave);
  const coreHp = useTowerDefenseStore((s) => s.coreHp);
  const wavesCleared = useTowerDefenseStore((s) => s.wavesCleared);
  const trainPoints = useTowerDefenseStore((s) => s.dataset.train.length);
  const towers = useTowerDefenseStore((s) => s.towers);
  // Memoised, not selected: see the note on regularizationOf in store.ts.
  const reg = useMemo(() => regularizationOf(towers), [towers]);

  const l1 = useTowerDefenseStore((s) => towerCount(s, "l1"));
  const l2 = useTowerDefenseStore((s) => towerCount(s, "l2"));
  const dropout = useTowerDefenseStore((s) => towerCount(s, "dropout"));

  const setComplexity = useTowerDefenseStore((s) => s.setComplexity);
  const addTower = useTowerDefenseStore((s) => s.addTower);
  const removeTower = useTowerDefenseStore((s) => s.removeTower);
  const clearTowers = useTowerDefenseStore((s) => s.clearTowers);
  const nextWave = useTowerDefenseStore((s) => s.nextWave);
  const restart = useTowerDefenseStore((s) => s.restart);

  const training = phase === "training";
  const parameters = parameterCount(complexity);

  return (
    <div className="flex flex-col gap-4">
      <WaveTracker
        wave={wave.index}
        wavesCleared={wavesCleared}
        coreHp={coreHp}
      />

      {/* THE knob (spec: <ComplexitySlider>). */}
      <div className="border-t border-border pt-4">
        <Slider
          label="Model complexity"
          value={complexity}
          min={MIN_COMPLEXITY}
          max={MAX_COMPLEXITY}
          step={1}
          disabled={training}
          onChange={setComplexity}
          format={(value) => `${value} units`}
          hint="Hidden units. This is the raw capacity, before any tower touches it."
        />
        <p className="mt-1 text-xs text-text-muted">
          {parameters} parameters for {trainPoints} training points —{" "}
          <span className="font-mono">
            {(parameters / trainPoints).toFixed(2)}
          </span>{" "}
          per point.
        </p>
      </div>

      <RegTowerPalette
        counts={{ l1, l2, dropout }}
        regularization={reg}
        disabled={training}
        onAdd={addTower}
        onRemove={removeTower}
      />

      <div className="border-t border-border pt-4">
        {training ? (
          <>
            <Button
              variant="secondary"
              className="w-full"
              onClick={trainer.stop}
              icon={<Square className="size-4" />}
            >
              Stop training
            </Button>
            <p className="mt-1.5 text-center font-mono text-xs text-text-muted">
              epoch {trainer.epoch} of {trainer.totalEpochs}
            </p>
          </>
        ) : phase === "resolved" ? (
          <Button
            variant="primary"
            className="w-full"
            onClick={nextWave}
            icon={<ChevronRight className="size-4" />}
          >
            Send wave {wave.index + 1}
          </Button>
        ) : phase === "won" || phase === "lost" ? (
          <Button
            variant="primary"
            className="w-full"
            onClick={restart}
            icon={<RotateCcw className="size-4" />}
          >
            New run
          </Button>
        ) : (
          <>
            <Button
              variant="secondary"
              className="w-full"
              onClick={() => void trainer.trial()}
              icon={<Shield className="size-4" />}
            >
              Trial run (no damage)
            </Button>
            <Button
              variant="primary"
              className="mt-2 w-full"
              onClick={() => void trainer.deploy()}
              icon={<Zap className="size-4" />}
            >
              Deploy against wave {wave.index}
            </Button>
          </>
        )}

        <Button
          variant="ghost"
          className="mt-2 w-full"
          onClick={clearTowers}
          disabled={training || l1 + l2 + dropout === 0}
        >
          Stand down all towers
        </Button>

        {trainer.error ? (
          <p className="mt-2 text-xs text-wrong">{trainer.error}</p>
        ) : null}
      </div>
    </div>
  );
}

export default function OverfitTowerDefense() {
  const meta = getGameMeta(SLUG);
  const trainer = useNeuronDefenseTrainer();

  const gap = useTowerDefenseStore(gapOf);
  const bias = useTowerDefenseStore(biasOf);
  const trainAccuracy = useTowerDefenseStore((s) => s.trainAccuracy);
  const validationAccuracy = useTowerDefenseStore((s) => s.validationAccuracy);
  const coreHp = useTowerDefenseStore((s) => s.coreHp);
  const phase = useTowerDefenseStore((s) => s.phase);
  const wave = useTowerDefenseStore(currentWave);
  const failure = useTowerDefenseStore((s) => s.failure);
  const whyCard = useTowerDefenseStore((s) => s.whyCard);
  const lane = useTowerDefenseStore((s) => s.lane);
  const setLane = useTowerDefenseStore((s) => s.setLane);
  const nextWave = useTowerDefenseStore((s) => s.nextWave);
  const restart = useTowerDefenseStore((s) => s.restart);
  const games = useProgression((s) => s.games);

  const hydrate = useProgression((s) => s.hydrate);
  const hydrated = useProgression((s) => s.hydrated);
  const lastGain = useProgression((s) => s.lastGain);
  const xp = useProgression((s) => s.xp);
  const level = levelFromXp(xp);

  useEffect(() => {
    if (!hydrated) void hydrate();
  }, [hydrated, hydrate]);

  /**
   * The live metric (pedagogy contract #3): the train/validation gap, which is
   * the catalog's stated metric for this game. Sampled every 10 epochs during
   * training from real predictions, so it moves while the model learns — and the
   * moment it starts climbing is overfitting happening in front of you.
   */
  const metric: MetricSpec = {
    label: "Train/val gap",
    value: gap,
    format: "percent",
    precision: 1,
    goodDirection: "down",
    state:
      failure !== null
        ? "bad"
        : phase === "won"
          ? "good"
          : Number.isFinite(gap) && gap > 0.15
            ? "warn"
            : undefined,
    caption: Number.isFinite(gap)
      ? phase === "training"
        ? `epoch ${trainer.epoch} of ${trainer.totalEpochs}`
        : "train accuracy − validation accuracy"
      : "deploy to measure",
  };

  const secondaryMetrics: MetricSpec[] = [
    {
      label: "Bias",
      value: bias,
      format: "percent",
      precision: 1,
      goodDirection: "down",
      caption: "how far short of the ceiling",
      state:
        Number.isFinite(bias) && bias > 0.12 ? "warn" : undefined,
    },
    {
      label: "Validation accuracy",
      value: validationAccuracy ?? Number.NaN,
      format: "percent",
      goodDirection: "up",
      caption:
        trainAccuracy === null
          ? "the number that matters"
          : `train ${Math.round(trainAccuracy * 100)}%`,
    },
    {
      label: "Core",
      value: coreHp,
      format: "integer",
      goodDirection: "up",
      caption: `of ${CORE_MAX_HP} · wave ${wave.index} of ${WAVES.length}`,
      state: coreHp <= CORE_MAX_HP * 0.3 ? "bad" : undefined,
    },
  ];

  return (
    <GameShell
      slug={SLUG}
      title={meta?.title ?? "Overfit Tower Defense"}
      metric={metric}
      secondaryMetrics={secondaryMetrics}
      math={{
        equation: MATH_EQUATION,
        code: MATH_CODE,
        codeLanguage: "javascript",
        notes: MATH_NOTES,
      }}
      controls={<Controls trainer={trainer} />}
      visual={<VisualLane />}
      code={<CodeLane trainer={trainer} />}
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
      onRetry={restart}
      onNext={
        phase === "resolved"
          ? nextWave
          : phase === "won"
            ? restart
            : undefined
      }
      nextLabel={
        phase === "resolved"
          ? `Send wave ${wave.index + 1}`
          : phase === "won"
            ? "Run it again"
            : undefined
      }
    />
  );
}


