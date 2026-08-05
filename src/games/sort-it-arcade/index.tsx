"use client";

import { useEffect } from "react";
import { Sparkles, Wand2 } from "lucide-react";
import { Button, DatasetChip, type MetricSpec } from "@/components";
import { GameShell } from "@/engine/GameShell";
import { levelFromXp, useProgression } from "@/engine/progression";
import { getGameMeta } from "@/lib/catalog";
import {
  KNOT_COUNTS,
  MATH_CODE,
  MATH_EQUATION,
  MATH_NOTES,
  TEST_SIZE,
  TRAIN_SIZE,
  WIN_SCORE,
  type BoundaryType,
} from "./ml";
import { SLUG, useSortItStore } from "./store";
import { VisualLane } from "./VisualLane";
import { CodeLane } from "./CodeLane";

const CAPACITY_OPTIONS: ReadonlyArray<{
  type: BoundaryType;
  label: string;
  detail: string;
}> = [
  { type: "line", label: "Line", detail: `${KNOT_COUNTS.line} params` },
  { type: "curve", label: "Curve", detail: `${KNOT_COUNTS.curve} params` },
  { type: "wiggle", label: "Wiggle", detail: `${KNOT_COUNTS.wiggle} params` },
];

const STAR_CRITERIA = [
  "Clear a round",
  `Score ${Math.round(WIN_SCORE * 100)}% or better`,
  "Clear a round from the code lane",
];

/** Right-rail controls: capacity choice, the optimizer, and the verdict. */
function Controls() {
  const boundaryType = useSortItStore((s) => s.boundary.type);
  const setBoundaryType = useSortItStore((s) => s.setBoundaryType);
  const autoFit = useSortItStore((s) => s.autoFit);
  const check = useSortItStore((s) => s.check);
  const newRound = useSortItStore((s) => s.newRound);
  const won = useSortItStore((s) => s.won);
  const round = useSortItStore((s) => s.round);
  const testAccuracy = useSortItStore((s) => s.testAccuracy);

  return (
    <div className="flex flex-col gap-4">
      <fieldset>
        <legend className="mb-2 text-sm font-medium">
          Model capacity
        </legend>
        <p className="mb-2 text-xs text-text-muted">
          How many bends the boundary may have. Each parameter past two costs
          score.
        </p>
        <div className="flex flex-wrap gap-2">
          {CAPACITY_OPTIONS.map((option) => (
            <DatasetChip
              key={option.type}
              name={option.label}
              detail={option.detail}
              selected={boundaryType === option.type}
              onSelect={() => setBoundaryType(option.type)}
            />
          ))}
        </div>
      </fieldset>

      <div className="flex flex-col gap-2">
        <Button
          variant="secondary"
          onClick={autoFit}
          icon={<Wand2 className="size-4" />}
        >
          Fit it for me
        </Button>
        <p className="text-xs text-text-muted">
          Runs the same search your dragging does — it sweeps each handle to
          whatever maximises training accuracy.
        </p>
      </div>

      <div className="flex flex-col gap-2 border-t border-border pt-4">
        <Button
          variant="primary"
          onClick={check}
          icon={<Sparkles className="size-4" />}
        >
          Check generalization
        </Button>
        <p className="text-xs text-text-muted">
          Scores your boundary against {TEST_SIZE} points it has never seen.
        </p>
      </div>

      {won ? (
        <Button variant="secondary" onClick={newRound}>
          New round ({round} cleared)
        </Button>
      ) : testAccuracy !== null ? (
        <Button variant="ghost" onClick={newRound}>
          Try a different round
        </Button>
      ) : null}
    </div>
  );
}

export default function SortItArcade() {
  const meta = getGameMeta(SLUG);

  const accuracy = useSortItStore((s) => s.accuracy);
  const penalty = useSortItStore((s) => s.penalty);
  const score = useSortItStore((s) => s.score);
  const testAccuracy = useSortItStore((s) => s.testAccuracy);
  const complexityCost = useSortItStore((s) => s.boundary.complexityCost);
  const failure = useSortItStore((s) => s.failure);
  const whyCard = useSortItStore((s) => s.whyCard);
  const lane = useSortItStore((s) => s.lane);
  const setLane = useSortItStore((s) => s.setLane);
  const reset = useSortItStore((s) => s.reset);
  const newRound = useSortItStore((s) => s.newRound);
  const won = useSortItStore((s) => s.won);

  const hydrate = useProgression((s) => s.hydrate);
  const hydrated = useProgression((s) => s.hydrated);
  const lastGain = useProgression((s) => s.lastGain);
  const games = useProgression((s) => s.games);
  // Subscribe to xp and derive the level from it, so the XP bar re-renders when
  // it changes. Calling the store's level() would read the right value but
  // wouldn't register a subscription.
  const xp = useProgression((s) => s.xp);
  const level = levelFromXp(xp);

  useEffect(() => {
    if (!hydrated) void hydrate();
  }, [hydrated, hydrate]);

  /**
   * The live metric (pedagogy contract #3): training accuracy, recomputed from
   * real geometry on every boundary change. Never hidden, never hardcoded.
   */
  const metric: MetricSpec = {
    label: "Accuracy",
    value: accuracy,
    format: "percent",
    goodDirection: "up",
    caption: `on the ${TRAIN_SIZE} points you can see`,
  };

  const secondaryMetrics: MetricSpec[] = [
    {
      label: "Complexity",
      value: complexityCost,
      format: "integer",
      goodDirection: "down",
      caption: `−${penalty.toFixed(3)} score`,
    },
    {
      label: "Score",
      value: score,
      format: "percent",
      goodDirection: "up",
      caption: `need ${Math.round(WIN_SCORE * 100)}%`,
      state: won ? "good" : undefined,
    },
    {
      label: "Held-out",
      value: testAccuracy ?? Number.NaN,
      format: "percent",
      goodDirection: "up",
      caption: testAccuracy === null ? "hidden until you check" : "never fitted",
    },
  ];

  return (
    <GameShell
      slug={SLUG}
      title={meta?.title ?? "Sort-It Arcade"}
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
      onNext={won ? newRound : undefined}
      nextLabel="Next round"
    />
  );
}
