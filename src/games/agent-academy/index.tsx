"use client";

import { useEffect } from "react";
import { Brain, Eraser, RotateCcw } from "lucide-react";
import { Button, type MetricSpec } from "@/components";
import { GameShell } from "@/engine/GameShell";
import {
  HIGH_SCORE_THRESHOLD,
  levelFromXp,
  useProgression,
} from "@/engine/progression";
import { getGameMeta } from "@/lib/catalog";
import {
  COMPETENCE_RATE,
  EPISODE_BATCH,
  MATH_CODE,
  MATH_EQUATION,
  MATH_NOTES,
  MAX_EPISODES,
  episodesForScore,
} from "./ml";
import { SLUG, episodesLeft, useAcademyStore } from "./store";
import { EpsilonSlider } from "./EpsilonSlider";
import { RewardEditor } from "./RewardEditor";
import { VisualLane } from "./VisualLane";
import { CodeLane } from "./CodeLane";

/**
 * The engine's rule, stated in this game's terms: ★1 completed, ★2 a best score of
 * at least HIGH_SCORE_THRESHOLD, ★3 that plus a code-lane clear. The episode
 * count in ★2 is the inverse of the scorer (`episodesForScore`), not a second
 * threshold that could drift from it.
 */
const STAR_CRITERIA = [
  `Get the agent reaching the exit in ${Math.round(
    COMPETENCE_RATE * 100,
  )}% of rollouts`,
  `Score ${Math.round(
    HIGH_SCORE_THRESHOLD * 100,
  )}% or better: graduate within ${episodesForScore(
    HIGH_SCORE_THRESHOLD,
  )} episodes`,
  "Graduate an agent by training it from the code lane",
];

function Controls() {
  const rewards = useAcademyStore((s) => s.rewards);
  const optimal = useAcademyStore((s) => s.optimal);
  const epsilon = useAcademyStore((s) => s.epsilon);
  const episodesUsed = useAcademyStore((s) => s.episodesUsed);
  const training = useAcademyStore((s) => s.training);
  const phase = useAcademyStore((s) => s.phase);
  const remaining = useAcademyStore(episodesLeft);

  const setReward = useAcademyStore((s) => s.setReward);
  const setEpsilon = useAcademyStore((s) => s.setEpsilon);
  const train = useAcademyStore((s) => s.train);
  const forget = useAcademyStore((s) => s.forget);
  const reset = useAcademyStore((s) => s.reset);

  return (
    <div className="flex flex-col gap-4">
      <RewardEditor
        rewards={rewards}
        optimal={optimal}
        disabled={training}
        onChange={setReward}
      />

      <div className="border-t border-border pt-4">
        <EpsilonSlider
          epsilon={epsilon}
          disabled={training}
          episodesUsed={episodesUsed}
          onChange={setEpsilon}
        />
      </div>

      <div className="border-t border-border pt-4">
        {phase === "graduated" ? (
          <Button
            variant="primary"
            className="w-full"
            onClick={reset}
            icon={<RotateCcw className="size-4" />}
          >
            New academy
          </Button>
        ) : (
          <Button
            variant="primary"
            className="w-full"
            onClick={() => train()}
            disabled={training || remaining <= 0}
            icon={<Brain className="size-4" />}
          >
            {remaining <= 0
              ? "No episodes left"
              : `Train ${Math.min(EPISODE_BATCH, remaining)} episodes`}
          </Button>
        )}

        <Button
          variant="ghost"
          className="mt-2 w-full"
          onClick={forget}
          disabled={training || episodesUsed === 0}
          icon={<Eraser className="size-4" />}
        >
          Forget everything, keep the rules
        </Button>

        <p className="mt-2 text-xs text-text-muted">
          {episodesUsed} of {MAX_EPISODES} episodes spent. Changing any reward or ε
          starts a fresh agent — one clean experiment at a time.
        </p>
      </div>
    </div>
  );
}

export default function AgentAcademy() {
  const meta = getGameMeta(SLUG);

  const report = useAcademyStore((s) => s.report);
  const optimal = useAcademyStore((s) => s.optimal);
  const episodesUsed = useAcademyStore((s) => s.episodesUsed);
  const epsilon = useAcademyStore((s) => s.epsilon);
  const phase = useAcademyStore((s) => s.phase);
  const failure = useAcademyStore((s) => s.failure);
  const whyCard = useAcademyStore((s) => s.whyCard);
  const lane = useAcademyStore((s) => s.lane);
  const setLane = useAcademyStore((s) => s.setLane);
  const reset = useAcademyStore((s) => s.reset);
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
   * The live metric (pedagogy contract #3): mean episode reward, the catalog's
   * stated live feedback.
   *
   * Reward rather than goal rate on purpose, and it is the sharpest decision in
   * this game's UI. Goal rate would be the number that tracks winning, which is
   * exactly why it is the wrong headline: this game is about the gap between the
   * reward you optimise and the outcome you wanted, and putting the outcome in the
   * big type would quietly close that gap for the player. Mean episode reward is
   * the number an RL practitioner actually watches, and it is the number that goes
   * UP while a reward-hacking agent stops finishing — so the headline metric can
   * climb into the green while the caption says nobody is leaving the maze. That
   * contradiction is the lesson, and it has to be on screen to be felt.
   */
  const metric: MetricSpec = {
    label: "Episode reward",
    value: episodesUsed === 0 ? Number.NaN : report.meanReward,
    format: "decimal",
    precision: 1,
    goodDirection: "up",
    state:
      phase === "graduated"
        ? "good"
        : failure !== null
          ? "bad"
          : undefined,
    caption:
      episodesUsed === 0
        ? "nothing trained yet"
        : `${Math.round(report.goalRate * 100)}% of rollouts reach the exit · need ${Math.round(
            COMPETENCE_RATE * 100,
          )}%`,
  };

  const secondaryMetrics: MetricSpec[] = [
    {
      // The outcome, kept beside the reward rather than instead of it.
      label: "Reaches the exit",
      value: episodesUsed === 0 ? Number.NaN : report.goalRate,
      format: "percent",
      goodDirection: "up",
      caption:
        episodesUsed === 0
          ? "untrained"
          : report.goalRate >= COMPETENCE_RATE
            ? "competent"
            : `${Math.round(report.timeoutRate * 100)}% time out, ${Math.round(
                report.trapRate * 100,
              )}% fall in`,
      state:
        episodesUsed > 0 && report.goalRate >= COMPETENCE_RATE ? "good" : undefined,
    },
    {
      label: "Episodes",
      value: episodesUsed,
      format: "integer",
      goodDirection: "down",
      caption: `of ${MAX_EPISODES} · ε ${epsilon.toFixed(2)}`,
    },
    {
      // What the reward function asks for, next to what it got. This is the pair
      // that makes reward hacking legible rather than mysterious.
      label: "Best possible",
      value: optimal.episodeReward,
      format: "decimal",
      precision: 1,
      goodDirection: "up",
      caption: optimal.reachesGoal
        ? `by walking out in ${optimal.steps} steps`
        : optimal.behaviour === "farm"
          ? "by never leaving the maze"
          : optimal.behaviour === "quit"
            ? "by falling in the pit"
            : "by wandering",
      state: optimal.reachesGoal ? undefined : "bad",
    },
    {
      label: "Cheese per episode",
      value: episodesUsed === 0 ? Number.NaN : report.meanPellets,
      format: "decimal",
      precision: 1,
      goodDirection: "down",
      caption:
        episodesUsed === 0
          ? "untrained"
          : report.meanPellets >= 2
            ? "it is farming, not travelling"
            : "on the way past",
      state: report.meanPellets >= 2 ? "warn" : undefined,
    },
  ];

  return (
    <GameShell
      slug={SLUG}
      title={meta?.title ?? "Agent Academy"}
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
      onNext={phase === "graduated" ? reset : undefined}
      nextLabel={phase === "graduated" ? "Run it again" : undefined}
    />
  );
}
