"use client";

import { useMemo } from "react";
import { Play, RotateCcw } from "lucide-react";
import { Button, CodeEditor } from "@/components";
import { useCodeLane } from "@/engine/useCodeLane";
import { createCodeApi } from "./store";

/**
 * Decision Tree Architect — the code lane.
 *
 * `api.split` and `api.prune` are the same store actions the picker calls, and
 * `api.signOff` is the same judgement the button calls (CLAUDE.md two-lane rule).
 * The api lives in `store.ts` as `createCodeApi`, so its argument checks and its
 * code-lane attribution are unit-tested without rendering this component.
 *
 * The starter snippet is the experiment the visual lane can only gesture at:
 * build greedy CART at every depth and print both accuracies, so the crossover is
 * a table rather than a claim. Then it prunes back to the peak and signs off,
 * which is the actual professional move — grow, measure, prune.
 */

export const STARTER_CODE = `// Where does depth stop paying? Build the tree at every depth
// and read both columns.

log('plot', api.round().name, '|', api.trainPoints(), 'plots | ceiling',
    pct(api.achievable()), '| target', pct(api.round().target));
log('');
log('depth  gates   train      val     gap   starved');

let peak = { depth: 0, val: -1 };

// Only as deep as this plot allows: a peak past the limit cannot be signed off.
for (const point of api.depthCurve(api.round().maxDepth)) {
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
if (peak.depth < api.round().maxDepth) {
  log('deeper than that, training keeps rising and validation does not.');
} else {
  // The sweep stops at the limit, so it cannot see where validation turns.
  log('that is the depth limit: validation was still at its best there.');
}
if (peak.val < api.round().target) {
  log('no depth up to the limit reaches', pct(api.round().target).trim(),
      '- greedy CART alone cannot sign this plot off.');
}

// Grow to the peak in one step, then sign it off.
api.growGreedy(peak.depth);
log('built depth', api.depth(), 'with', api.splits(), 'gates ->',
    'train', pct(api.trainAccuracy()), 'val', pct(api.validationAccuracy()));
const result = api.signOff();
log('verdict:', result.outcome);

function pct(x) { return (x * 100).toFixed(1).padStart(6) + '%'; }

// Then try the full depth instead. A signed-off plot stays signed off, so
// start it over first - api.retryRound() - then api.growGreedy(api.round().maxDepth)
// and sign that off. Same data, same learner, more gates - and on the first two
// plots the verdict changes.
//
// On the ridge plot, try api.gainTable() at the root: every reading looks
// worthless. Then compare api.lookaheadOf('n0', 0, 0.5) - the threshold the
// table never tries.`;

export function CodeLane() {
  // One api object per mount: `createCodeApi` reads the store on every call,
  // so it never goes stale.
  const api = useMemo(() => createCodeApi(), []);

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
        hint="api.split · api.greedyAt · api.growGreedy · api.prune · api.clear · api.signOff · api.retryRound · api.gainTable · api.gainOf · api.lookaheadOf · api.impurityAt · api.countsAt · api.leaves · api.depthCurve · api.tryGreedy · api.depth · api.peak · api.round"
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
