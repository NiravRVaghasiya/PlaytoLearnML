"use client";

import { useMemo } from "react";
import { Play, RotateCcw } from "lucide-react";
import { Button, CodeEditor } from "@/components";
import { useCodeLane } from "@/engine/useCodeLane";
import { createCodeApi } from "./store";

/**
 * Backprop Blitz — the code lane.
 *
 * `api.choose` writes exactly the state the radio buttons write (CLAUDE.md two-lane
 * rule), so a snippet and a click are the same operation.
 *
 * The starter snippet does the thing the visual lane cannot: it applies one wrong
 * rule at a time across the whole graph and measures what each costs. That turns the
 * game's central claim into a table the player generates themselves — and the table
 * has a genuinely surprising column in it, because several wrong rules come out at
 * 0° from the true gradient and differ only in length.
 */

export const STARTER_CODE = `// What does each broken rule actually cost?

const steps = api.steps();
log('scenario:', api.scenario().title);
log('steps:', steps.length);
log('');

// The options are not labelled correct, so find the right routing the
// honest way: walk forward through the steps and at each one keep the
// option that leaves the fewest node gradients disagreeing with the
// reference trace. Greedy works because a backward pass is a chain -
// a step can only be judged once everything after it is settled.
api.reset();
for (const step of steps) {
  let best = null, bestWrong = Infinity;
  for (const option of step.options) {
    api.choose(step.index, option.id);
    const wrong = api.correctness().wrong.length;
    if (wrong < bestWrong) { bestWrong = wrong; best = option.id; }
  }
  api.choose(step.index, best);
}

const clean = api.correctness();
log('brute-forced routing:', pct(clean.fraction),
    '| angle', clean.angle.toFixed(1) + 'deg',
    '| coincidences', clean.coincidences.length);
const good = api.train({ useTruth: true });
log('correct gradients, 40 steps:', good.final.toFixed(6));
log('');

// Now break exactly one rule at a time and measure the damage.
log('node  rule            correct  angle   length   40 steps');
for (const step of steps) {
  const keep = api.steps()[step.index].chosen;
  for (const option of step.options) {
    if (option.id === keep) continue;
    checkBudget();
    api.choose(step.index, option.id);
    const c = api.correctness();
    const run = api.train();
    log(step.node.padEnd(5), option.id.padEnd(15),
        (c.matched + '/' + c.total).padStart(6),
        (c.angle.toFixed(1) + 'deg').padStart(8),
        c.ratio.toFixed(2).padStart(8),
        (run.diverged ? 'DIVERGED' : run.final.toFixed(5)).padStart(11));
  }
  api.choose(step.index, keep);
}

log('');
log('The rows at 0deg are the interesting ones: the direction is');
log('perfect and only the LENGTH is wrong. That is a bug that hides');
log('inside the learning rate and can survive for years.');

function pct(x) { return (x * 100).toFixed(0) + '%'; }`;

export function CodeLane() {
  const api = useMemo(() => createCodeApi(), []);
  const lane = useCodeLane({ initialCode: STARTER_CODE, api, maxRunMs: 60000 });

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
        label="Routing script"
        value={lane.code}
        onChange={lane.setCode}
        language="javascript"
        error={lane.error}
        rows={18}
        hint="api.steps · api.choose · api.chooseAll · api.correctness · api.mine · api.autograd · api.train · api.scenario · api.nextScenario · api.reset"
      />

      <section aria-label="Script output" className="min-h-24">
        <h2 className="mb-1 text-xs font-semibold tracking-wide text-text-muted uppercase">
          Output
        </h2>
        {lane.logs.length === 0 ? (
          <p className="font-mono text-xs text-text-muted">
            Run the script to see output here. `api.autograd` hands over the real
            trace — in here that is not cheating, it is the reference you would
            check a real implementation against.
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
