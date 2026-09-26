"use client";

import { useEffect } from "react";
import { ClipboardCheck, RotateCcw } from "lucide-react";
import { Button, type MetricSpec } from "@/components";
import { GameShell } from "@/engine/GameShell";
import {
  HIGH_SCORE_THRESHOLD,
  levelFromXp,
  useProgression,
} from "@/engine/progression";
import { getGameMeta } from "@/lib/catalog";
import {
  LEGENDARY_COMBOS,
  MATH_CODE,
  MATH_EQUATION,
  MATH_NOTES,
  TARGET_LIFT,
  legendariesIn,
} from "./ml";
import {
  SLUG,
  hasLeak,
  lift,
  useForgeStore,
} from "./store";
import { ColumnTray } from "./ColumnTray";
import { ForgeSlot } from "./ForgeSlot";
import { TransformPicker } from "./TransformPicker";
import { VisualLane } from "./VisualLane";
import { CodeLane } from "./CodeLane";

/**
 * Exactly the progression engine's rule: one star for a win, two for a best score
 * of at least HIGH_SCORE_THRESHOLD, three for that AND a code-lane clear. The
 * code-lane clear is a winning forge submitted from the Python lane
 * (`api.submit()`), because that is where the action has to come from.
 */
const STAR_CRITERIA = [
  `Lift validation accuracy ${Math.round(TARGET_LIFT * 100)} points over baseline`,
  `Score ${Math.round(HIGH_SCORE_THRESHOLD * 100)}% or better — legendary features add to the score`,
  "Submit a winning forge from the Python lane, with api.submit()",
];

function Controls() {
  const slot = useForgeStore((s) => s.slot);
  const transform = useForgeStore((s) => s.transform);
  const features = useForgeStore((s) => s.features);
  const training = useForgeStore((s) => s.training);
  const ready = useForgeStore((s) => s.ready);
  const phase = useForgeStore((s) => s.phase);
  const submitted = useForgeStore((s) => s.submitted);

  const toggleColumn = useForgeStore((s) => s.toggleColumn);
  const clearSlot = useForgeStore((s) => s.clearSlot);
  const setTransform = useForgeStore((s) => s.setTransform);
  const forge = useForgeStore((s) => s.forge);
  const removeFeature = useForgeStore((s) => s.removeFeature);
  const clearForge = useForgeStore((s) => s.clearForge);
  const submit = useForgeStore((s) => s.submit);
  const reset = useForgeStore((s) => s.reset);

  const busy = training || !ready;

  return (
    <div className="flex flex-col gap-4">
      <TransformPicker
        transform={transform}
        disabled={busy}
        onChange={setTransform}
      />

      <div className="border-t border-border pt-4">
        <ColumnTray
          slot={slot}
          transform={transform}
          disabled={busy}
          onToggle={toggleColumn}
        />
      </div>

      <ForgeSlot
        slot={slot}
        transform={transform}
        features={features}
        training={busy}
        onForge={() => void forge()}
        onClearSlot={clearSlot}
        onRemove={(id) => void removeFeature(id)}
      />

      <div className="border-t border-border pt-4">
        {phase === "forged" ? (
          <Button
            variant="primary"
            className="w-full"
            onClick={() => void reset()}
            icon={<RotateCcw className="size-4" />}
          >
            New forge
          </Button>
        ) : (
          <Button
            variant="primary"
            className="w-full"
            onClick={() => submit()}
            disabled={busy || features.length === 0}
            icon={<ClipboardCheck className="size-4" />}
          >
            {submitted ? "Score it again" : "Score this forge"}
          </Button>
        )}

        <Button
          variant="ghost"
          className="mt-2 w-full"
          onClick={() => void clearForge()}
          disabled={busy || features.length === 0}
        >
          Empty the forge
        </Button>
      </div>
    </div>
  );
}

export default function FeatureForge() {
  const meta = getGameMeta(SLUG);

  const features = useForgeStore((s) => s.features);
  const baselineScore = useForgeStore((s) => s.baselineScore);
  const currentScore = useForgeStore((s) => s.currentScore);
  const trainScore = useForgeStore((s) => s.trainScore);
  const columnNames = useForgeStore((s) => s.columnNames);
  const ready = useForgeStore((s) => s.ready);
  const fitError = useForgeStore((s) => s.fitError);
  const phase = useForgeStore((s) => s.phase);
  const failure = useForgeStore((s) => s.failure);
  const whyCard = useForgeStore((s) => s.whyCard);
  const lane = useForgeStore((s) => s.lane);
  const setLane = useForgeStore((s) => s.setLane);
  const initialise = useForgeStore((s) => s.initialise);
  const reset = useForgeStore((s) => s.reset);
  const currentLift = useForgeStore(lift);
  const leaked = useForgeStore(hasLeak);
  const games = useProgression((s) => s.games);

  const hydrate = useProgression((s) => s.hydrate);
  const hydrated = useProgression((s) => s.hydrated);
  const lastGain = useProgression((s) => s.lastGain);
  const xp = useProgression((s) => s.xp);
  const level = levelFromXp(xp);

  useEffect(() => {
    if (!hydrated) void hydrate();
  }, [hydrated, hydrate]);

  // The baseline has to exist before any lift can mean anything, so it is fitted
  // on mount rather than waiting for a player action.
  useEffect(() => {
    void initialise();
  }, [initialise]);

  const legendary = legendariesIn(features);

  /**
   * The live metric (pedagogy contract #3): lift over baseline, the catalog's
   * stated metric. It is a DIFFERENCE on purpose — an absolute accuracy would let
   * a player feel good about 63% without noticing the baseline was already 63%.
   *
   * It goes red when a leaky column is in the forge, even though the number is at
   * its highest. That inversion is the point: in this game a rising gauge is not
   * automatically good news.
   */
  const metric: MetricSpec = {
    label: "Metric lift",
    value: currentLift,
    format: "percent",
    precision: 1,
    goodDirection: "up",
    state: leaked
      ? "bad"
      : failure !== null
        ? "bad"
        : phase === "forged"
          ? "good"
          : Number.isFinite(currentLift) && currentLift >= TARGET_LIFT
            ? "good"
            : undefined,
    caption: !ready
      ? fitError !== null
        ? "the baseline fit failed — press Retry"
        : "fitting the baseline…"
      : leaked
        ? "not real — a leaky column is in the forge"
        : `need +${Math.round(TARGET_LIFT * 100)} points over ${(
            baselineScore * 100
          ).toFixed(1)}%`,
  };

  const secondaryMetrics: MetricSpec[] = [
    {
      label: "Validation accuracy",
      value: ready ? currentScore : Number.NaN,
      format: "percent",
      precision: 1,
      goodDirection: "up",
      caption: ready
        ? `baseline ${(baselineScore * 100).toFixed(1)}%`
        : "training",
    },
    {
      // Beside it so memorisation is visible: this is the number that rises when
      // extra width buys nothing.
      label: "Training accuracy",
      value: ready ? trainScore : Number.NaN,
      format: "percent",
      precision: 1,
      goodDirection: "up",
      caption:
        ready && trainScore - currentScore > 0.001
          ? `${((trainScore - currentScore) * 100).toFixed(1)} point gap`
          : "no gap",
      state: trainScore - currentScore > 0.06 ? "warn" : undefined,
    },
    {
      label: "Legendary",
      value: legendary.length,
      format: "integer",
      goodDirection: "up",
      caption: `of ${LEGENDARY_COMBOS.length} · ${columnNames.length} matrix columns`,
      state: phase === "forged" ? "good" : undefined,
    },
  ];

  return (
    <GameShell
      slug={SLUG}
      title={meta?.title ?? "Feature Forge"}
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
      onRetry={() => void reset()}
      onNext={phase === "forged" ? () => void reset() : undefined}
      nextLabel={phase === "forged" ? "Forge again" : undefined}
    />
  );
}
