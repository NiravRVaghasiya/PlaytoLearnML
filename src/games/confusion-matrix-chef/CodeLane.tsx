"use client";

import { useMemo } from "react";
import { Play, RotateCcw } from "lucide-react";
import { Button, CodeEditor } from "@/components";
import { useCodeLane } from "@/engine/useCodeLane";
import { createCodeApi } from "./store";

/**
 * Confusion Matrix Chef — the code lane.
 *
 * `api.setThreshold` writes the same value the slider writes and `api.serve`
 * calls the same judgement the Serve button calls (CLAUDE.md two-lane rule). The
 * api itself lives in `store.ts` as `createCodeApi`, so its argument checks and
 * its code-lane attribution are unit-tested without rendering this component.
 *
 * The starter snippet is threshold selection done properly: sweep every cutoff,
 * keep the ones that satisfy the brief, and report the band. That is the actual
 * professional procedure — you do not eyeball a threshold, you search for the set
 * that meets your constraints and then choose within it, on the metric the brief
 * is about. Sliding by hand is the intuition; this is the practice.
 */

export const STARTER_CODE = `// Threshold selection, done the way you would actually do it:
// sweep every cutoff and keep the ones that satisfy the brief.

log("shift:", api.scenario().name);
log("brief:", api.constraints().map(c => c.metric + " >= " + c.floor).join(", "));
log("guessing one class would score", pct(api.majorityBaseline()), "accuracy");
log("");

const winners = [];

for (let t = 0; t <= 100; t++) {
  const threshold = t / 100;
  const m = api.metricsAt(threshold);
  const ok = api.constraints().every(c => m[c.metric] >= c.floor);
  if (ok) winners.push({ threshold, ...m });
}

if (winners.length === 0) {
  log("no cutoff satisfies this brief");
} else {
  const lo = winners[0].threshold, hi = winners[winners.length - 1].threshold;
  log("satisfying band:", lo.toFixed(2), "to", hi.toFixed(2),
      "(" + winners.length + " of 101 cutoffs)");

  // Within the band, push the metric this shift is judged on as far as the
  // other floors allow. Which metric that is depends on the brief.
  const primary = api.scenario().primary;
  const best = winners.reduce((a, b) => (b[primary] > a[primary] ? b : a));
  log("best " + primary + " inside the band:", best.threshold.toFixed(2),
      "prec", pct(best.precision), "rec", pct(best.recall));

  // What would chasing accuracy alone have picked?
  let byAccuracy = { threshold: 0, accuracy: -1 };
  for (let t = 0; t <= 100; t++) {
    const m = api.metricsAt(t / 100);
    if (m.accuracy > byAccuracy.accuracy) byAccuracy = { threshold: t / 100, ...m };
  }
  log("most ACCURATE cutoff:", byAccuracy.threshold.toFixed(2),
      "acc", pct(byAccuracy.accuracy), "rec", pct(byAccuracy.recall),
      "-> in band?", winners.some(w => w.threshold === byAccuracy.threshold));

  api.setThreshold(best.threshold);
  const result = await api.serve();
  log("verdict:", result.outcome);
}

function pct(x) { return (x * 100).toFixed(1) + "%"; }

// The last two lines are the argument. On the imbalanced shifts the most
// accurate cutoff is nowhere near the band that satisfies the brief - it
// wins on accuracy by ignoring almost every case that mattered.`;

export function CodeLane() {
  // One api object per mount: `createCodeApi` reads the store on every call,
  // so it never goes stale.
  const api = useMemo(() => createCodeApi(), []);

  const lane = useCodeLane({ initialCode: STARTER_CODE, api, maxRunMs: 20000 });

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
        label="Threshold script"
        value={lane.code}
        onChange={lane.setCode}
        language="javascript"
        error={lane.error}
        rows={18}
        hint="api.setThreshold · api.serve · api.nextShift · api.retryShift · api.metricsAt · api.matrixAt · api.constraints · api.scenario · api.majorityBaseline · api.roc · api.auc · api.samples · api.shift · api.lastResult"
      />

      <section aria-label="Script output" className="min-h-24">
        <h2 className="mb-1 text-xs font-semibold tracking-wide text-text-muted uppercase">
          Output
        </h2>
        {lane.logs.length === 0 ? (
          <p className="font-mono text-xs text-text-muted">
            Run the script to see output here. No model is retrained — the sweep is
            pure arithmetic over fixed scores.
          </p>
        ) : (
          <ul className="max-h-40 overflow-auto rounded-md border border-border bg-bg p-2 font-mono text-xs">
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
