"use client";

import { useEffect, useMemo } from "react";
import {
  ChevronRight,
  ClipboardCheck,
  RotateCcw,
  Wand2,
} from "lucide-react";
import { Button, type MetricSpec } from "@/components";
import { GameShell } from "@/engine/GameShell";
import {
  HIGH_SCORE_THRESHOLD,
  levelFromXp,
  useProgression,
} from "@/engine/progression";
import { getGameMeta } from "@/lib/catalog";
import {
  MATH_CODE,
  MATH_EQUATION,
  MATH_NOTES,
  MIN_HONEST_LEAF,
  ROUNDS,
  describeSplit,
  leavesOf,
  nodeStats,
} from "./ml";
import {
  SLUG,
  currentRound,
  samplesInNode,
  useArchitectStore,
  validationMetricState,
} from "./store";
import { SplitGate } from "./SplitGate";
import { VisualLane } from "./VisualLane";
import { CodeLane } from "./CodeLane";

/**
 * The engine's rule, stated per game: ★1 finish, ★2 a best score of at least
 * HIGH_SCORE_THRESHOLD, ★3 that plus a code-lane clear. A plot's score is its
 * validation accuracy less up to 20% for depth used, averaged over the plots.
 */
const STAR_CRITERIA = [
  "Sign off all three plots",
  `Score ${Math.round(HIGH_SCORE_THRESHOLD * 100)}% or better by keeping trees shallow`,
  "Sign off a plot with api.signOff() from the code lane",
];

/**
 * Picking which leaf to work on.
 *
 * A radio group rather than clicking the diagram. The canvas is read-only, so
 * this is the only way in — and it happens to be the better one: it lists every
 * leaf with its plot count, which is exactly the information needed to choose
 * (work on the biggest impure leaf), and it is reachable by keyboard.
 */
function NodePicker() {
  const tree = useArchitectStore((s) => s.tree);
  const train = useArchitectStore((s) => s.dataset.train);
  const selectedNodeId = useArchitectStore((s) => s.selectedNodeId);
  const selectNode = useArchitectStore((s) => s.selectNode);

  const stats = useMemo(() => nodeStats(tree, train), [tree, train]);
  const leaves = useMemo(
    () =>
      leavesOf(tree).sort(
        (a, b) =>
          (stats[b.id]?.counts.total ?? 0) - (stats[a.id]?.counts.total ?? 0),
      ),
    [tree, stats],
  );

  return (
    <fieldset>
      <legend className="text-sm font-medium">Work on a leaf</legend>
      <p className="mt-1 mb-2 text-xs text-text-muted">
        Sorted by size. The biggest impure leaf is usually where a gate buys the
        most.
      </p>
      <div className="flex flex-col gap-1">
        {leaves.map((leaf) => {
          const stat = stats[leaf.id];
          const total = stat?.counts.total ?? 0;
          const starved = total > 0 && total < MIN_HONEST_LEAF;
          return (
            <label
              key={leaf.id}
              // min-h-11: DESIGN.md's 44px target. These rows ARE the leaf
              // picker, the primary way into the tree on a phone.
              className={`flex min-h-11 cursor-pointer items-center gap-2 rounded border px-2 py-1.5 text-xs ${
                leaf.id === selectedNodeId
                  ? "border-primary bg-primary/10"
                  : "border-border bg-surface-2 hover:bg-surface"
              }`}
            >
              <input
                type="radio"
                name="selected-leaf"
                value={leaf.id}
                checked={leaf.id === selectedNodeId}
                onChange={() => selectNode(leaf.id)}
                className="accent-[var(--primary)]"
              />
              <span className="font-mono">{leaf.id}</span>
              <span className="text-text-muted">depth {leaf.depth}</span>
              <span className="ml-auto font-mono">
                {total} plot{total === 1 ? "" : "s"}
              </span>
              <span className="font-mono text-text-muted">
                {(stat?.gini ?? 0).toFixed(2)}
              </span>
              {starved ? (
                <span className="text-warn" aria-hidden="true">
                  !
                </span>
              ) : null}
              <span className="sr-only">
                impurity {(stat?.gini ?? 0).toFixed(2)}
                {starved
                  ? `, fewer than ${MIN_HONEST_LEAF} plots — too few to decide from`
                  : ""}
              </span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}

function Controls() {
  const round = useArchitectStore(currentRound);
  const tree = useArchitectStore((s) => s.tree);
  const selectedNodeId = useArchitectStore((s) => s.selectedNodeId);
  const depth = useArchitectStore((s) => s.depth);
  const splits = useArchitectStore((s) => s.splits);
  const phase = useArchitectStore((s) => s.phase);
  const attempts = useArchitectStore((s) => s.attempts);

  const splitAt = useArchitectStore((s) => s.splitAt);
  const prune = useArchitectStore((s) => s.prune);
  const takeGreedySplit = useArchitectStore((s) => s.takeGreedySplit);
  const clearTree = useArchitectStore((s) => s.clearTree);
  const signOff = useArchitectStore((s) => s.signOff);
  const nextRound = useArchitectStore((s) => s.nextRound);
  const restart = useArchitectStore((s) => s.restart);

  // Memoised, not selected. `samplesAt` builds a new array, and a zustand
  // selector returning a fresh reference breaks React's cached-snapshot contract.
  const train = useArchitectStore((s) => s.dataset.train);
  const bucket = useMemo(
    () => samplesInNode(tree, train, selectedNodeId),
    [tree, train, selectedNodeId],
  );
  const node = tree[selectedNodeId];
  const isLeaf = node?.split === null;
  const atDepthLimit = (node?.depth ?? 0) >= round.maxDepth;

  return (
    <div className="flex flex-col gap-4">
      <section aria-labelledby="brief-heading">
        <h2
          id="brief-heading"
          className="flex flex-wrap items-baseline justify-between gap-x-2 gap-y-1"
        >
          <span className="font-display text-sm font-semibold">
            Plot {round.index} of {ROUNDS.length}: {round.name}
          </span>
          <span className="text-xs font-normal text-text-muted">
            depth {depth}/{round.maxDepth} · {splits} gate
            {splits === 1 ? "" : "s"}
          </span>
        </h2>
        <blockquote className="mt-1.5 border-l-2 border-primary/60 pl-2.5 text-xs">
          {round.brief}
        </blockquote>
      </section>

      <div className="border-t border-border pt-4">
        <NodePicker />
      </div>

      <div className="border-t border-border pt-4">
        <SplitGate
          nodeId={selectedNodeId}
          bucket={bucket}
          isLeaf={isLeaf}
          atDepthLimit={atDepthLimit}
          onSplit={(split) => splitAt(selectedNodeId, split)}
          onPrune={() => prune(selectedNodeId)}
        />
        {isLeaf && !atDepthLimit && bucket.length >= 2 ? (
          <Button
            variant="ghost"
            size="sm"
            className="mt-2 w-full"
            onClick={() => takeGreedySplit(selectedNodeId)}
            icon={<Wand2 className="size-4" />}
          >
            Take the greedy gate
          </Button>
        ) : null}
      </div>

      <div className="border-t border-border pt-4">
        {phase === "cleared" ? (
          <Button
            variant="primary"
            className="w-full"
            onClick={nextRound}
            icon={<ChevronRight className="size-4" />}
          >
            Next plot
          </Button>
        ) : phase === "complete" ? (
          <Button
            variant="primary"
            className="w-full"
            onClick={restart}
            icon={<RotateCcw className="size-4" />}
          >
            Survey again
          </Button>
        ) : (
          <Button
            variant="primary"
            className="w-full"
            // Arrow, not `onClick={signOff}`: signOff's argument is the lane
            // it came from, and a click would pass the event in its place.
            onClick={() => signOff("visual")}
            icon={<ClipboardCheck className="size-4" />}
          >
            Sign off this tree
          </Button>
        )}

        <Button
          variant="ghost"
          className="mt-2 w-full"
          onClick={clearTree}
          disabled={splits === 0}
        >
          Start the tree over
        </Button>
        {/* The shell's Retry now retries THIS plot, so the full restart it
            used to be needs a button that says what it does — the same one
            Confusion Matrix Chef has. */}
        <Button
          variant="ghost"
          className="mt-2 w-full"
          onClick={restart}
          disabled={round.index === 1 && splits === 0 && attempts === 0}
        >
          Back to plot one
        </Button>
        {attempts > 0 ? (
          <p className="mt-1.5 text-center text-xs text-text-muted">
            {attempts} tree{attempts === 1 ? "" : "s"} signed off on this plot
          </p>
        ) : null}
      </div>
    </div>
  );
}

export default function DecisionTreeArchitect() {
  const meta = getGameMeta(SLUG);

  const round = useArchitectStore(currentRound);
  const tree = useArchitectStore((s) => s.tree);
  const trainAccuracy = useArchitectStore((s) => s.trainAccuracy);
  const validationAccuracy = useArchitectStore((s) => s.validationAccuracy);
  const depth = useArchitectStore((s) => s.depth);
  const splits = useArchitectStore((s) => s.splits);
  const starved = useArchitectStore((s) => s.starved);
  const peakValidation = useArchitectStore((s) => s.peakValidation);
  const peakDepth = useArchitectStore((s) => s.peakDepth);
  const metricState = useArchitectStore(validationMetricState);
  const clearedScores = useArchitectStore((s) => s.clearedScores);
  const phase = useArchitectStore((s) => s.phase);
  const failure = useArchitectStore((s) => s.failure);
  const whyCard = useArchitectStore((s) => s.whyCard);
  const lane = useArchitectStore((s) => s.lane);
  const setLane = useArchitectStore((s) => s.setLane);
  const nextRound = useArchitectStore((s) => s.nextRound);
  const retryRound = useArchitectStore((s) => s.retryRound);
  const restart = useArchitectStore((s) => s.restart);
  const games = useProgression((s) => s.games);

  const hydrate = useProgression((s) => s.hydrate);
  const hydrated = useProgression((s) => s.hydrated);
  const lastGain = useProgression((s) => s.lastGain);
  const xp = useProgression((s) => s.xp);
  const level = levelFromXp(xp);

  useEffect(() => {
    if (!hydrated) void hydrate();
  }, [hydrated, hydrate]);

  const rootGate = tree["n0"]?.split ?? null;

  /**
   * The live metric (pedagogy contract #3): validation accuracy, the catalog's
   * stated metric. It recomputes on every gate, and it is the number that can go
   * DOWN when the tree gets bigger — which is the whole lesson, and the reason
   * training accuracy sits beside it rather than replacing it.
   * Its colour is `validationMetricState`, which the tests pin down.
   */
  const metric: MetricSpec = {
    label: "Validation accuracy",
    value: validationAccuracy,
    format: "percent",
    precision: 1,
    goodDirection: "up",
    state: metricState,
    caption: `needs ${Math.round(round.target * 100)}%${
      peakValidation > 0.01
        ? ` · your best ${Math.round(peakValidation * 100)}% at depth ${peakDepth}`
        : ""
    }`,
  };

  const secondaryMetrics: MetricSpec[] = [
    {
      // Beside the live metric so the divergence is impossible to miss: this one
      // can only ever go up as gates are added.
      label: "Training accuracy",
      value: trainAccuracy,
      format: "percent",
      precision: 1,
      goodDirection: "up",
      caption:
        trainAccuracy - validationAccuracy > 0.001
          ? `${((trainAccuracy - validationAccuracy) * 100).toFixed(1)} point gap`
          : "no gap yet",
      state: trainAccuracy - validationAccuracy > 0.15 ? "warn" : undefined,
    },
    {
      label: "Depth",
      value: depth,
      format: "integer",
      goodDirection: "down",
      caption: `${splits} gate${splits === 1 ? "" : "s"} · limit ${round.maxDepth}${
        starved > 0 ? ` · ${starved} starved` : ""
      }`,
      state: starved > 0 ? "warn" : undefined,
    },
    {
      label: "Plots signed off",
      value: clearedScores.length,
      format: "integer",
      goodDirection: "up",
      caption: `of ${ROUNDS.length}${
        rootGate ? ` · root ${describeSplit(rootGate)}` : ""
      }`,
      state: phase === "complete" ? "good" : undefined,
    },
  ];

  return (
    <GameShell
      slug={SLUG}
      title={meta?.title ?? "Decision Tree Architect"}
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
      // Retry means this plot again. Wiping every plot already signed off is
      // "Back to plot one", which says so.
      onRetry={retryRound}
      onNext={
        phase === "cleared"
          ? nextRound
          : phase === "complete"
            ? restart
            : undefined
      }
      nextLabel={
        phase === "cleared"
          ? `Plot ${round.index + 1}`
          : phase === "complete"
            ? "Survey again"
            : undefined
      }
    />
  );
}
