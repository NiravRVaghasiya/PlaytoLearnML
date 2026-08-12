"use client";

import { useMemo } from "react";
import { Play, RotateCcw } from "lucide-react";
import { Button, CodeEditor } from "@/components";
import { useCodeLane } from "@/engine/useCodeLane";
import {
  auc,
  confusionAt,
  majorityBaseline,
  metricsOf,
  rocCurve,
  scenarioAt,
} from "./ml";
import { useChefStore } from "./store";

/**
 * Confusion Matrix Chef — the code lane.
 *
 * `api.setThreshold` writes the same value the slider writes and `api.serve`
 * calls the same judgement the Serve button calls (CLAUDE.md two-lane rule).
 *
 * The starter snippet is threshold selection done properly: sweep every cutoff,
 * keep the ones that satisfy the brief, and report the band. That is the actual
 * professional procedure — you do not eyeball a threshold, you search for the set
 * that meets your constraints and then choose within it. Sliding by hand is the
 * intuition; this is the practice.
 */

const STARTER_CODE = `// Threshold selection, done the way you would actually do it:
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

  // Within the band, take the most balanced one.
  const best = winners.reduce((a, b) => (b.f1 > a.f1 ? b : a));
  log("best F1 inside the band:", best.threshold.toFixed(2),
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
  const store = useChefStore;

  const api = useMemo(
    () => ({
      /** The decision cutoff. Same action as the slider. */
      setThreshold: (threshold: number) => {
        if (!Number.isFinite(threshold)) {
          throw new Error("setThreshold needs a finite number");
        }
        if (threshold < 0 || threshold > 1) {
          throw new Error("threshold must be between 0 and 1");
        }
        store.getState().setThreshold(threshold);
      },
      /** Commit the current cutoff for judgement. Same as the Serve button. */
      serve: () => store.getState().serve(),
      nextShift: () => store.getState().nextShift(),
      restart: () => store.getState().restart(),

      threshold: () => store.getState().threshold,

      /** The matrix at any cutoff, without committing to it. */
      matrixAt: (threshold: number) =>
        confusionAt(store.getState().samples, threshold),
      /** Every metric at any cutoff. */
      metricsAt: (threshold: number) =>
        metricsOf(confusionAt(store.getState().samples, threshold)),

      /** The critic's brief, as data. */
      scenario: () => {
        const scenario = scenarioAt(store.getState().scenarioIndex);
        return {
          index: scenario.index,
          id: scenario.id,
          name: scenario.name,
          prevalence: scenario.prevalence,
          positiveLabel: scenario.positiveLabel,
          primary: scenario.primary,
          falsePositiveCost: scenario.falsePositiveCost,
          falseNegativeCost: scenario.falseNegativeCost,
        };
      },
      constraints: () =>
        scenarioAt(store.getState().scenarioIndex).constraints.map(
          (constraint) => ({ ...constraint }),
        ),
      /** Accuracy obtainable by ignoring the scores entirely. */
      majorityBaseline: () =>
        majorityBaseline(scenarioAt(store.getState().scenarioIndex).prevalence),

      /** The full ROC curve, and its area. */
      roc: () => rocCurve(store.getState().samples),
      auc: () => auc(store.getState().samples),

      /** The scores themselves, read-only. */
      samples: () =>
        store.getState().samples.map((sample) => ({ ...sample })),
      sampleCount: () => store.getState().samples.length,

      shift: () => store.getState().scenarioIndex,
      cleared: () => store.getState().clearedScores.length,
      lastResult: () => store.getState().servedResult,
      phase: () => store.getState().phase,
    }),
    [store],
  );

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
        hint="api.setThreshold · api.serve · api.nextShift · api.metricsAt · api.matrixAt · api.constraints · api.scenario · api.majorityBaseline · api.roc · api.auc · api.samples · api.shift · api.lastResult"
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
