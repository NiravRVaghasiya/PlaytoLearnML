"use client";

import { useMemo } from "react";
import { Play, RotateCcw } from "lucide-react";
import { Button, CodeEditor } from "@/components";
import { useCodeLane } from "@/engine/useCodeLane";
import { createCodeApi } from "./store";

/**
 * Convolution Kitchen — the code lane.
 *
 * `api.applyPreset`, `api.setWeights`, `api.addLayer` and the rest write exactly
 * the state the steppers and radio buttons write (CLAUDE.md two-lane rule).
 *
 * The starter snippet settles the claim the visual lane can only assert. "No single
 * layer passes this menu" is a statement about a search over kernel sets, and
 * nobody is going to check it one stepper click at a time — so `api.trial` scores a
 * stack without adopting it, and `api.learn` trains one end to end. Sixteen
 * one-layer stacks in a loop is a proof a player can run themselves, which is worth
 * considerably more than a paragraph from me saying it is so.
 *
 * The setters return a promise that settles when the kitchen has been rescored,
 * and they throw by name on anything the steppers could not have produced — an
 * unknown preset, a slot that does not exist, a spent budget — rather than doing
 * nothing and letting the snippet report success.
 */

const STARTER_CODE = `// The game claims one layer cannot pass this menu. Check it.

log('four dishes, chance is 25%. target is 90%.');
log('');
log('ONE LAYER, hand-designed kernel sets');
log('kernels                                pool   score  per-class');

const singles = [
  ['Vertical edge'],
  ['Horizontal edge'],
  ['Blur'],
  ['Vertical edge', 'Horizontal edge'],
  ['Vertical edge', 'Horizontal edge', 'Diagonal edge'],
  ['Vertical edge', 'Horizontal edge', 'Diagonal edge', 'Centre spot'],
];

let bestOne = 0;
for (const kernels of singles) {
  for (const pool of ['avg', 'max']) {
    checkBudget();
    const r = await api.trial([{ pool, kernels }]);
    bestOne = Math.max(bestOne, r.accuracy);
    log(kernels.join('+').padEnd(38), pool.padEnd(6),
        pct(r.accuracy).padStart(6), ' [' +
        r.perClass.map(pct).join(' ') + ']');
  }
}

log('');
log('best one-layer score:', pct(bestOne));
log('Notice WHICH dishes fail: the two arrangement dishes, always');
log('together. One layer ends in a global average, so it reports how');
log('much of each pattern is present and never where it is.');

log('');
log('Let gradient descent try the same shape - no hand design at all:');
for (const count of [1, 2, 4]) {
  checkBudget();
  const r = await api.learn([{ pool: 'avg', kernels: count }]);
  log('  one layer,', count, 'learned kernels:', pct(r.accuracy));
}

log('');
log('TWO LAYERS, same filter budget');
const stacks = [
  [{ pool: 'avg', kernels: ['Vertical edge'] },
   { pool: 'avg', kernels: ['Pass through', 'Horizontal edge'] }],
  [{ pool: 'avg', kernels: ['Vertical edge', 'Horizontal edge'] },
   { pool: 'avg', kernels: ['Pass through', 'Horizontal edge'] }],
];
for (const spec of stacks) {
  checkBudget();
  const r = await api.trial(spec);
  log('  ' + r.filters + ' filters, ' + r.channels + ' channels:',
      pct(r.accuracy), '[' + r.perClass.map(pct).join(' ') + ']');
}

log('');
log('Three filters and a second layer beats six in one. Depth is not');
log('more capacity here - it is the only way to keep a position.');
log('');
log('Build the winner in your own kitchen (the setters wait for the rescore):');
log('  await api.applyPreset(0, 0, "Vertical edge"); await api.addLayer();');
log('  log(api.score().accuracy)');

function pct(x) { return (x * 100).toFixed(1) + '%'; }`;

export function CodeLane() {
  const api = useMemo(() => createCodeApi(), []);
  const lane = useCodeLane({ initialCode: STARTER_CODE, api, maxRunMs: 120000 });

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
        label="Kitchen script"
        value={lane.code}
        onChange={lane.setCode}
        language="javascript"
        error={lane.error}
        rows={18}
        hint="api.trial · api.learn · api.applyPreset · api.setWeights · api.addKernel · api.removeKernel · api.setPool · api.addLayer · api.removeLayer · api.score · api.inspect · api.maps · api.layers · api.baseline"
      />

      <section aria-label="Script output" className="min-h-24">
        <h2 className="mb-1 text-xs font-semibold tracking-wide text-text-muted uppercase">
          Output
        </h2>
        {lane.logs.length === 0 ? (
          <p className="font-mono text-xs text-text-muted">
            Run the script to see output here. `api.trial` and `api.learn` score
            stacks without adopting them — only the setters change your kitchen.
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
