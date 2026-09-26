"use client";

import { useMemo } from "react";
import { Play, RotateCcw } from "lucide-react";
import { Button, CodeEditor } from "@/components";
import { useCodeLane } from "@/engine/useCodeLane";
import { createCodeApi } from "./store";

/**
 * Gradient Descent Skier — the code lane.
 *
 * `api.step()` is the same store action the Step button calls, so the loop written
 * here and the loop clicked out by hand are the same computation on the same state
 * (CLAUDE.md two-lane rule).
 *
 * The starter snippet writes the descent loop out longhand instead of calling a
 * `runToEnd()` helper, because the whole point is that gradient descent is five
 * lines: read the slope, scale it, move, check, repeat.
 */

const STARTER_CODE = `// Gradient descent, written out. This is the entire algorithm.

api.reset();
api.setLearningRate(0.05);
api.setMomentum(0);        // no momentum: plain gradient descent

while (api.stepsRemaining() > 0 && !api.settled() && !api.diverged()) {
  api.step();              // v <- Bv - a*grad ;  theta <- theta + v
  checkBudget();
}

log('steps  ', api.steps());
log('loss   ', api.loss().toFixed(3));
log('at x   ', api.position().x.toFixed(3));
log('deepest', api.globalMinimum().loss.toFixed(3), 'at x', api.globalMinimum().x.toFixed(3));
log('verdict', api.check().outcome);

// You land in the SHALLOW valley: the gradient there is zero, so
// plain descent has nowhere left to go.
//
// Now add momentum - api.setMomentum(0.85) - and run it again.
// Same learning rate, same surface, different answer.
//
// Then try setLearningRate(1.0) and watch it leave the mountain.`;

export function CodeLane() {
  // Built in the store module so its argument checking is unit-tested; the
  // verbs are the same store actions the rail's buttons call (two-lane rule).
  const api = useMemo(() => createCodeApi(), []);

  const lane = useCodeLane({ initialCode: STARTER_CODE, api, maxRunMs: 8000 });

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
        label="Descent script"
        value={lane.code}
        onChange={lane.setCode}
        language="javascript"
        error={lane.error}
        rows={18}
        hint="api.step · api.runToEnd · api.setLearningRate · api.setMomentum · api.position · api.loss · api.gradient · api.slope · api.lossAt · api.gradientAt · api.globalMinimum · api.steps · api.stepsRemaining · api.settled · api.diverged · api.check · api.reset"
      />

      <section aria-label="Script output" className="min-h-24">
        {/* h2: directly inside the lane, so it must not skip a level. */}
        <h2 className="mb-1 text-xs font-semibold tracking-wide text-text-muted uppercase">
          Output
        </h2>
        {lane.logs.length === 0 ? (
          <p className="font-mono text-xs text-text-muted">
            Run the script to see output here.
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
