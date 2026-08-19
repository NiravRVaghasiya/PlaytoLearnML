"use client";

import { useMemo } from "react";
import { Play, RotateCcw } from "lucide-react";
import { Button, CodeEditor } from "@/components";
import { useCodeLane } from "@/engine/useCodeLane";
import { createCodeApi } from "./store";

/**
 * Dimension Diver — the code lane.
 *
 * `api.setAngles` and `api.submit` write exactly the state the sliders and the
 * commit button write (CLAUDE.md two-lane rule).
 *
 * The starter snippet does the two things the sliders cannot. It verifies the claim
 * the whole game rests on — that retained variance depends only on the discarded
 * axis, so roll cannot matter — by sweeping roll and watching the number refuse to
 * move. And it searches the sphere of orientations for both objectives at once,
 * which is what turns "variance is not separation" from an assertion into a table.
 */

const STARTER_CODE = `// Two claims to check, both by brute force.

const pca = api.pca();
log('cloud:', api.cloud().title);
log('eigenvalues:', pca.eigenvalues.map(v => v.toFixed(3)).join('  '));
log('ceiling:', pct(pca.bestRetained),
    '| PCA at yaw', pca.angles.yaw.toFixed(1),
    'pitch', pca.angles.pitch.toFixed(1));
log('');

// 1. Roll cannot change what a projection retains.
log('roll   retained');
for (const roll of [0, 45, 90, 137, 180]) {
  log(String(roll).padEnd(6),
      pct(api.retainedAt({ yaw: 30, pitch: -20, roll })));
}
log('...because retained = 1 - variance(discarded axis) / total,');
log('and rolling spins the shadow inside a plane already chosen.');
log('');

// 2. Sweep the sphere for BOTH objectives and compare.
let bestVar = { v: -1 }, bestSep = { s: -1 }, bestBoth = { s: -1 };
for (let yaw = -90; yaw < 90; yaw += 3) {
  for (let pitch = -90; pitch <= 90; pitch += 3) {
    checkBudget();
    const v = api.retainedAt({ yaw, pitch, roll: 0 });
    const s = api.separationAt({ yaw, pitch, roll: 0 });
    if (v > bestVar.v) bestVar = { v, s, yaw, pitch };
    if (s > bestSep.s) bestSep = { v, s, yaw, pitch };
    // Best separation among planes that still pass the variance target.
    if (v >= pca.bestRetained * 0.98 && s > bestBoth.s) {
      bestBoth = { v, s, yaw, pitch };
    }
  }
}

log('objective              retained  separation  yaw    pitch');
row('max variance', bestVar);
row('max separation', bestSep);
row('max sep within target', bestBoth);
log('');
log('If the first and last rows point somewhere different, this cloud');
log('is one where PCA answers a question nobody asked.');
log('');
log('Commit the plane that shows the groups:');
log('  api.setAngles({ yaw: ' + bestBoth.yaw + ', pitch: ' +
    bestBoth.pitch + ', roll: 0 }); api.submit()');

function row(name, r) {
  log(name.padEnd(22),
      pct(r.v).padStart(8),
      r.s.toFixed(3).padStart(12),
      String(r.yaw).padStart(6),
      String(r.pitch).padStart(7));
}
function pct(x) { return (x * 100).toFixed(2) + '%'; }`;

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
        label="Projection script"
        value={lane.code}
        onChange={lane.setCode}
        language="javascript"
        error={lane.error}
        rows={18}
        hint="api.pca · api.retainedAt · api.separationAt · api.setAngles · api.submit · api.score · api.shadow · api.points · api.cloud · api.nextCloud · api.reset"
      />

      <section aria-label="Script output" className="min-h-24">
        <h2 className="mb-1 text-xs font-semibold tracking-wide text-text-muted uppercase">
          Output
        </h2>
        {lane.logs.length === 0 ? (
          <p className="font-mono text-xs text-text-muted">
            Run the script to see output here. `api.retainedAt` and
            `api.separationAt` measure any orientation without turning the cloud.
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
