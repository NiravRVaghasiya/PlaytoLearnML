"use client";

import { useMemo } from "react";
import { Play, RotateCcw } from "lucide-react";
import { Button, CodeEditor } from "@/components";
import { useCodeLane } from "@/engine/useCodeLane";
import {
  FEATURES,
  accuracyOf,
  bestSplit,
  buildGreedyTree,
  candidateSplitsWithLookahead,
  countsOf,
  depthCurve,
  describeSplit,
  gainOf,
  giniOf,
  leavesOf,
  nodeStats,
  roundAt,
} from "./ml";
import { samplesAt, useArchitectStore } from "./store";

/**
 * Decision Tree Architect — the code lane.
 *
 * `api.split` and `api.prune` are the same store actions the picker calls, and
 * `api.signOff` is the same judgement the button calls (CLAUDE.md two-lane rule).
 *
 * The starter snippet is the experiment the visual lane can only gesture at:
 * build greedy CART at every depth and print both accuracies, so the crossover is
 * a table rather than a claim. Then it prunes back to the peak and signs off,
 * which is the actual professional move — grow, measure, prune.
 */

const STARTER_CODE = `// Where does depth stop paying? Build the tree at every depth
// and read both columns.

log('plot', api.round().name, '|', api.trainPoints(), 'plots | ceiling',
    pct(api.achievable()), '| target', pct(api.round().target));
log('');
log('depth  gates   train      val     gap   starved');

let peak = { depth: 0, val: -1 };

for (const point of api.depthCurve(9)) {
  const gap = point.trainAccuracy - point.validationAccuracy;
  log(String(point.depth).padStart(4), String(point.splits).padStart(6),
      pct(point.trainAccuracy), pct(point.validationAccuracy),
      pct(gap).padStart(7), String(api.starvedAtDepth(point.depth)).padStart(6));
  if (point.validationAccuracy > peak.val) {
    peak = { depth: point.depth, val: point.validationAccuracy };
  }
}

log('');
log('validation peaks at depth ' + peak.depth, '->', pct(peak.val));
log('deeper than that, training keeps rising and validation does not.');

// Grow to the peak, then sign it off.
api.clear();
api.growGreedy(peak.depth);
log('built depth', api.depth(), 'with', api.splits(), 'gates ->',
    'train', pct(api.trainAccuracy()), 'val', pct(api.validationAccuracy()));
const result = api.signOff();
log('verdict:', result.outcome);

function pct(x) { return (x * 100).toFixed(1).padStart(6) + '%'; }

// Try api.growGreedy(9) instead and sign that off. Same data, same
// learner, more gates - and the verdict changes.
//
// On the ridge plot, try api.gainTable() at the root: every reading looks
// worthless. Then compare the lookahead column.`;

export function CodeLane() {
  const store = useArchitectStore;

  const api = useMemo(
    () => ({
      /** Build a gate at a node. Same action as the picker. */
      split: (nodeId: string, feature: number, threshold: number) => {
        if (typeof nodeId !== "string" || !store.getState().tree[nodeId]) {
          throw new Error(`no node "${nodeId}" in the tree`);
        }
        if (!Number.isInteger(feature) || feature < 0 || feature >= FEATURES.length) {
          throw new Error(`feature must be 0..${FEATURES.length - 1}`);
        }
        if (!Number.isFinite(threshold)) {
          throw new Error("threshold must be a finite number");
        }
        store.getState().splitAt(nodeId, { feature, threshold });
      },
      /** Take whatever greedy CART would take at a node. */
      greedyAt: (nodeId: string) => store.getState().takeGreedySplit(nodeId),
      prune: (nodeId: string) => store.getState().prune(nodeId),
      clear: () => store.getState().clearTree(),

      /** Grow the whole tree greedily to a depth, replacing what is there. */
      growGreedy: (maxDepth: number) => {
        const state = store.getState();
        state.clearTree();
        // Walk breadth-first, taking the best gain at every leaf, exactly as
        // buildGreedyTree does — but through the store so both lanes see it.
        for (let depth = 0; depth < maxDepth; depth += 1) {
          const current = store.getState();
          for (const leaf of leavesOf(current.tree)) {
            if (leaf.depth !== depth) continue;
            const bucket = samplesAt(store.getState(), leaf.id);
            const candidate = bestSplit(bucket);
            if (candidate === null || candidate.gain <= 0) continue;
            store.getState().splitAt(leaf.id, {
              feature: candidate.feature,
              threshold: candidate.threshold,
            });
          }
        }
      },

      signOff: () => store.getState().signOff(),
      nextRound: () => store.getState().nextRound(),
      restart: () => store.getState().restart(),

      /** The gain table at a node, lookahead column included. */
      gainTable: (nodeId = "n0") => {
        const bucket = samplesAt(store.getState(), nodeId);
        return candidateSplitsWithLookahead(bucket).map((candidate, index) => ({
          feature: index,
          name: FEATURES[index]!.name,
          threshold: candidate?.threshold ?? null,
          gain: candidate?.gain ?? 0,
          lookaheadGain: candidate?.lookaheadGain ?? 0,
        }));
      },
      /** Gain of any gate you like, without building it. */
      gainOf: (nodeId: string, feature: number, threshold: number) =>
        gainOf(samplesAt(store.getState(), nodeId), { feature, threshold }),
      impurityAt: (nodeId: string) =>
        giniOf(countsOf(samplesAt(store.getState(), nodeId))),
      countsAt: (nodeId: string) =>
        countsOf(samplesAt(store.getState(), nodeId)),

      leaves: () => leavesOf(store.getState().tree).map((leaf) => leaf.id),
      gate: (nodeId: string) => {
        const node = store.getState().tree[nodeId];
        return node?.split === null || !node
          ? null
          : { ...node.split, describe: describeSplit(node.split) };
      },

      /** Greedy CART's accuracies at each depth, without touching your tree. */
      depthCurve: (maxDepth = 9) =>
        depthCurve(store.getState().dataset, maxDepth),
      /** How many starved leaves greedy CART would have at a depth. */
      starvedAtDepth: (maxDepth: number) => {
        const { dataset } = store.getState();
        const tree = buildGreedyTree(dataset.train, maxDepth);
        const stats = nodeStats(tree, dataset.train);
        return leavesOf(tree).filter((leaf) => {
          const total = stats[leaf.id]?.counts.total ?? 0;
          return total > 0 && total < 5;
        }).length;
      },
      /** Score any depth of greedy CART without committing to it. */
      tryGreedy: (maxDepth: number) => {
        const { dataset } = store.getState();
        const tree = buildGreedyTree(dataset.train, maxDepth);
        const stats = nodeStats(tree, dataset.train);
        return {
          trainAccuracy: accuracyOf(tree, stats, dataset.train),
          validationAccuracy: accuracyOf(tree, stats, dataset.validation),
        };
      },

      trainAccuracy: () => store.getState().trainAccuracy,
      validationAccuracy: () => store.getState().validationAccuracy,
      depth: () => store.getState().depth,
      splits: () => store.getState().splits,
      starved: () => store.getState().starved,
      peak: () => ({
        validation: store.getState().peakValidation,
        depth: store.getState().peakDepth,
      }),

      round: () => {
        const round = roundAt(store.getState().roundIndex);
        return {
          index: round.index,
          name: round.name,
          boundary: round.boundary,
          target: round.target,
          maxDepth: round.maxDepth,
          trainPoints: round.trainPoints,
        };
      },
      trainPoints: () => store.getState().dataset.train.length,
      achievable: () => store.getState().dataset.achievable,
      features: () => FEATURES.map((meta) => ({ ...meta })),
      phase: () => store.getState().phase,
      cleared: () => store.getState().clearedScores.length,
      lastEvaluation: () => store.getState().evaluation,
    }),
    [store],
  );

  const lane = useCodeLane({ initialCode: STARTER_CODE, api, maxRunMs: 30000 });

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="primary"
          size="sm"
          onClick={() => void lane.run()}
          disabled={lane.running}
          icon={<Play className="size-4" />}
        >
          {lane.running ? "Running…" : "Run"}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={lane.reset}
          icon={<RotateCcw className="size-4" />}
        >
          Restore snippet
        </Button>
        {lane.dirty ? (
          <span className="font-mono text-xs text-text-muted">edited</span>
        ) : null}
      </div>

      <CodeEditor
        className="min-h-0 flex-1"
        label="Tree script"
        value={lane.code}
        onChange={lane.setCode}
        language="javascript"
        error={lane.error}
        rows={18}
        hint="api.split · api.greedyAt · api.growGreedy · api.prune · api.clear · api.signOff · api.gainTable · api.gainOf · api.impurityAt · api.countsAt · api.leaves · api.depthCurve · api.tryGreedy · api.depth · api.peak · api.round"
      />

      <section aria-label="Script output" className="min-h-24">
        <h2 className="mb-1 text-xs font-semibold tracking-wide text-text-muted uppercase">
          Output
        </h2>
        {lane.logs.length === 0 ? (
          <p className="font-mono text-xs text-text-muted">
            Run the script to see output here. Nothing trains — a tree is built by
            counting, so this is instant.
          </p>
        ) : (
          <ul className="max-h-40 overflow-auto rounded-md border border-border bg-bg p-2 font-mono text-xs whitespace-pre">
            {lane.logs.map((entry, index) => (
              <li
                key={index}
                className={
                  entry.level === "error" ? "text-wrong" : "text-text-muted"
                }
              >
                {entry.message}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
