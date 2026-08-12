"use client";

import { useEffect, useMemo } from "react";
import { FastForward, Lightbulb, RotateCcw, Unlock } from "lucide-react";
import { Button, type MetricSpec } from "@/components";
import { GameShell } from "@/engine/GameShell";
import {
  HIGH_SCORE_THRESHOLD,
  levelFromXp,
  useProgression,
} from "@/engine/progression";
import { getGameMeta } from "@/lib/catalog";
import {
  BAYESIAN_SEED_TRIALS,
  CRACK_THRESHOLD,
  DIALS,
  LEARNING_RATE,
  MATH_CODE,
  MATH_EQUATION,
  MATH_NOTES,
  describePoint,
  distinctValuesTried,
  gridLevels,
  suggestNext,
} from "./ml";
import {
  SLUG,
  bestObjective,
  lastTemperature,
  triesLeft,
  triesUsed,
  useHeistStore,
} from "./store";
import { BudgetCounter } from "./BudgetCounter";
import { SafeDials } from "./SafeDials";
import { StrategyToggle } from "./StrategyToggle";
import { VisualLane } from "./VisualLane";
import { CodeLane } from "./CodeLane";

const STAR_CRITERIA = [
  "Crack the safe inside the budget",
  `Score ${Math.round(HIGH_SCORE_THRESHOLD * 100)}% or better by cracking it early`,
  "Crack it from the code lane",
];

function Controls() {
  const dials = useHeistStore((s) => s.dials);
  const trials = useHeistStore((s) => s.trials);
  const strategy = useHeistStore((s) => s.strategy);
  const budget = useHeistStore((s) => s.budget);
  const phase = useHeistStore((s) => s.phase);
  const used = useHeistStore(triesUsed);
  const left = useHeistStore(triesLeft);
  const best = useHeistStore(bestObjective);

  const setDial = useHeistStore((s) => s.setDial);
  const setDials = useHeistStore((s) => s.setDials);
  const setStrategy = useHeistStore((s) => s.setStrategy);
  const tryCurrent = useHeistStore((s) => s.tryCurrent);
  const trySuggestion = useHeistStore((s) => s.trySuggestion);
  const runToBudget = useHeistStore((s) => s.runToBudget);
  const reset = useHeistStore((s) => s.reset);

  const cracking = phase === "cracking" && left > 0;

  // Memoised on the trials: fitting the GP and scanning candidates is not free.
  const suggestion = useMemo(
    () => (strategy === "bayesian" ? suggestNext(trials) : null),
    [strategy, trials],
  );

  return (
    <div className="flex flex-col gap-4">
      <BudgetCounter
        used={used}
        budget={budget}
        bestObjective={best}
        crackThreshold={CRACK_THRESHOLD}
      />

      <div className="border-t border-border pt-4">
        <StrategyToggle
          strategy={strategy}
          budget={budget}
          disabled={!cracking}
          onChange={setStrategy}
        />
      </div>

      <SafeDials dials={dials} disabled={!cracking} onChange={setDial} />

      {/* The acquisition function's pick, surfaced as the spec asks. */}
      {strategy === "bayesian" && suggestion && cracking ? (
        <div className="rounded-md border border-warn/50 bg-warn/10 p-2.5">
          <p className="text-xs font-semibold">Acquisition function suggests</p>
          <p className="mt-0.5 font-mono text-[11px] text-text-muted">
            {describePoint(suggestion.point)}
          </p>
          <p className="mt-1 text-[11px] text-text-muted">
            {Number.isFinite(suggestion.expectedImprovement)
              ? `Expected improvement ${suggestion.expectedImprovement.toFixed(
                  4,
                )} · predicts ${(suggestion.predictedMean * 100).toFixed(
                  1,
                )}% ± ${(suggestion.predictedSd * 100).toFixed(1)}`
              : `Still warming up — ${Math.max(
                  0,
                  BAYESIAN_SEED_TRIALS - trials.length,
                )} more spread-out tries before a surrogate is worth fitting.`}
          </p>
          <Button
            variant="secondary"
            size="sm"
            className="mt-2 w-full"
            onClick={() => void trySuggestion()}
            icon={<Lightbulb className="size-4" />}
          >
            Try where it points
          </Button>
        </div>
      ) : null}

      <div className="border-t border-border pt-4">
        {cracking ? (
          <>
            <Button
              variant="primary"
              className="w-full"
              onClick={() => void tryCurrent()}
              icon={<Unlock className="size-4" />}
            >
              Try these dials ({left} left)
            </Button>
            {strategy !== "manual" ? (
              <Button
                variant="secondary"
                className="mt-2 w-full"
                onClick={() => runToBudget(strategy)}
                icon={<FastForward className="size-4" />}
              >
                Spend all {left} on {strategy}
              </Button>
            ) : null}
          </>
        ) : (
          <Button
            variant="primary"
            className="w-full"
            onClick={reset}
            icon={<RotateCcw className="size-4" />}
          >
            New safe
          </Button>
        )}

        <Button
          variant="ghost"
          className="mt-2 w-full"
          onClick={() => setDials(new Array(DIALS.length).fill(0.5))}
          disabled={!cracking}
        >
          Centre the dials
        </Button>
      </div>
    </div>
  );
}

export default function HyperparameterHeist() {
  const meta = getGameMeta(SLUG);

  const trials = useHeistStore((s) => s.trials);
  const budget = useHeistStore((s) => s.budget);
  const phase = useHeistStore((s) => s.phase);
  const evaluation = useHeistStore((s) => s.evaluation);
  const failure = useHeistStore((s) => s.failure);
  const whyCard = useHeistStore((s) => s.whyCard);
  const lane = useHeistStore((s) => s.lane);
  const setLane = useHeistStore((s) => s.setLane);
  const reset = useHeistStore((s) => s.reset);
  const best = useHeistStore(bestObjective);
  const used = useHeistStore(triesUsed);
  const temperature = useHeistStore(lastTemperature);
  const games = useProgression((s) => s.games);

  const hydrate = useProgression((s) => s.hydrate);
  const hydrated = useProgression((s) => s.hydrated);
  const lastGain = useProgression((s) => s.lastGain);
  const xp = useProgression((s) => s.xp);
  const level = levelFromXp(xp);

  useEffect(() => {
    if (!hydrated) void hydrate();
  }, [hydrated, hydrate]);

  const lrSeen = useMemo(
    () => distinctValuesTried(trials, LEARNING_RATE),
    [trials],
  );

  // Deliberately NOT computed here: what the current dials would score. It is one
  // call to objectiveAt away, and showing it would hand the player unlimited free
  // readings — which is the one thing the compute budget exists to prevent. The
  // only way to learn a value in this game is to spend a try on it.

  /**
   * The live metric (pedagogy contract #3): the best objective found so far, which
   * is the catalog's stated metric and the only number that survives a run. It
   * moves on a try and on nothing else, which is the point — under a compute
   * budget, information costs.
   */
  const metric: MetricSpec = {
    label: "Best objective",
    value: best,
    format: "percent",
    precision: 1,
    goodDirection: "up",
    state:
      failure !== null
        ? "bad"
        : phase === "cracked"
          ? "good"
          : Number.isFinite(best) && temperature === "hot"
            ? "warn"
            : undefined,
    caption: Number.isFinite(best)
      ? `opens at ${Math.round(CRACK_THRESHOLD * 100)}% · ${
          temperature ?? "untried"
        }`
      : "nothing tried yet",
  };

  const secondaryMetrics: MetricSpec[] = [
    {
      label: "Tries spent",
      value: used,
      format: "integer",
      goodDirection: "down",
      caption: `of ${budget}`,
      state: budget - used <= 3 && phase === "cracking" ? "warn" : undefined,
    },
    {
      // The coverage number the whole lesson turns on, kept permanently visible.
      label: `Distinct ${DIALS[LEARNING_RATE]!.short} values`,
      value: lrSeen,
      format: "integer",
      goodDirection: "up",
      caption: `a grid would show ${gridLevels(budget)}`,
      state: used >= 6 && lrSeen <= 3 ? "warn" : undefined,
    },
    {
      label: "Score",
      value: evaluation?.score ?? Number.NaN,
      format: "percent",
      goodDirection: "up",
      caption:
        evaluation === null || evaluation.outcome === "searching"
          ? "crack it to score"
          : `${evaluation.outcome.replace("-", " ")}`,
      state: phase === "cracked" ? "good" : undefined,
    },
  ];

  return (
    <GameShell
      slug={SLUG}
      title={meta?.title ?? "Hyperparameter Heist"}
      metric={metric}
      secondaryMetrics={secondaryMetrics}
      math={{
        equation: MATH_EQUATION,
        code: MATH_CODE,
        codeLanguage: "javascript",
        notes: MATH_NOTES,
      }}
      controls={<Controls />}
      visual={<VisualLane />}
      code={<CodeLane />}
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
      onNext={phase === "cracked" ? reset : undefined}
      nextLabel={phase === "cracked" ? "Another safe" : undefined}
    />
  );
}
