"use client";

import { useMemo } from "react";
import { Play, RotateCcw } from "lucide-react";
import { Button, CodeEditor } from "@/components";
import { useCodeLane } from "@/engine/useCodeLane";
import { MAX_K, MIN_K, useKMeansStore } from "./store";

/**
 * K-Means Territory Wars — the code lane.
 *
 * Every function here delegates to the same store action the visual lane's
 * buttons call. `api.assign()` and `api.update()` are the two halves of the
 * algorithm, exposed under their real names, so writing the loop by hand and
 * clicking the buttons are the same operation on the same state (CLAUDE.md
 * two-lane rule).
 *
 * The starter snippet writes the k-means loop out longhand rather than calling a
 * `settle()` helper, because seeing the `while` and the convergence test is the
 * lesson. `checkBudget()` is in there for real: it's what stops a mistyped loop
 * condition from hanging the tab.
 *
 * Every verb checks its arguments and throws a named error the output panel can
 * show. `api.setFlag(0, 0.5)` used to store a flag at y = NaN, which quietly
 * dropped out of every nearest-flag test and left the metric reading "—".
 */

function shown(value: unknown): string {
  if (typeof value === "string") return `"${value}"`;
  if (typeof value === "number") return String(value);
  return value === undefined ? "nothing" : typeof value;
}

/** A coordinate on the 0–1 map, or a named error saying what was wrong. */
function coordinate(verb: string, name: string, value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(
      `${verb}: ${name} must be a number from 0 to 1, got ${shown(value)}.`,
    );
  }
  if (value < 0 || value > 1) {
    throw new RangeError(
      `${verb}: ${name} must be from 0 to 1 (the edges of the map), got ${value}.`,
    );
  }
  return value;
}

export const STARTER_CODE = `// k-means, written out longhand. Every call drives the same
// board the visual lane does — flip the toggle and watch.

api.setK(3);              // how many flags? try 3, then try 7
api.scatterFlags();       // spread them out to start

let previous = Infinity;
for (let iteration = 0; iteration < 50; iteration++) {
  api.assign();           // villages join their nearest flag
  api.update();           // flags move to their villages' mean

  const inertia = api.inertia();
  log('iteration', iteration + 1, 'inertia', inertia.toFixed(3));

  // Converged when a full step no longer improves anything.
  if (previous - inertia < 1e-6) break;
  previous = inertia;

  checkBudget();          // stops a runaway loop from freezing the page
}

const result = api.check();
log('---');
log('k       ', result.k);
log('inertia ', result.inertia.toFixed(3));
log('best at k', result.bestInertiaAtK.toFixed(3));
log('verdict ', result.outcome);

// Now change setK(3) to setK(7). Inertia gets LOWER and the
// verdict gets worse. Inertia alone cannot choose k.`;

/**
 * The code lane's verbs, bound to a store. A plain function rather than inline
 * in the component so the argument checks can be unit-tested without React.
 */
export function createKMeansApi(store: typeof useKMeansStore = useKMeansStore) {
  return {
    /** ASSIGN step. Same action as the Assign button. */
    assign: () => store.getState().assign(),
    /** UPDATE step. Same action as the Update button. */
    update: () => store.getState().update(),
    /** Both halves once. */
    step: () => store.getState().step(),
    /** Iterate to convergence. */
    settle: () => store.getState().settle(),

    /** Add or remove flags until there are exactly `k`. */
    setK: (k: number) => {
      if (typeof k !== "number" || !Number.isInteger(k)) {
        throw new TypeError(
          `setK(k): k must be a whole number of flags, got ${shown(k)}.`,
        );
      }
      if (k < MIN_K || k > MAX_K) {
        throw new RangeError(
          `setK(k): k must be from ${MIN_K} to ${MAX_K}, got ${k}.`,
        );
      }
      // A counted loop, not `while (length !== k)`: if a store guard ever
      // refused a step, a while-loop on the player's number would spin forever.
      const difference = k - store.getState().centroids.length;
      for (let step = 0; step < Math.abs(difference); step += 1) {
        if (difference > 0) store.getState().addFlag();
        else store.getState().removeFlag(store.getState().centroids.length - 1);
      }
    },

    /** Spread the flags evenly over the map — a neutral starting layout. */
    scatterFlags: () => {
      const { centroids, moveFlag } = store.getState();
      const count = centroids.length;
      centroids.forEach((_, index) => {
        const angle = (index / Math.max(1, count)) * Math.PI * 2;
        moveFlag(
          index,
          0.5 + 0.28 * Math.cos(angle),
          0.5 + 0.28 * Math.sin(angle),
        );
      });
    },

    /** Place one flag exactly. This is the initialization choice. */
    setFlag: (index: number, x: number, y: number) => {
      const count = store.getState().centroids.length;
      if (!Number.isInteger(index) || index < 0 || index >= count) {
        throw new RangeError(
          `setFlag(index, x, y): index must be a whole number from 0 to ${
            count - 1
          } (there are ${count} flags), got ${shown(index)}.`,
        );
      }
      store
        .getState()
        .moveFlag(
          index,
          coordinate("setFlag(index, x, y)", "x", x),
          coordinate("setFlag(index, x, y)", "y", y),
        );
    },

    /** Current flag positions. */
    flags: () =>
      store.getState().centroids.map(({ id, x, y }) => ({ id, x, y })),

    /** Live inertia — the same number the metric shows. */
    inertia: () => store.getState().inertia,
    /** Villages per flag. */
    territorySizes: () => [...store.getState().sizes],
    k: () => store.getState().centroids.length,
    converged: () => store.getState().converged,
    /** Best achievable inertia at each k — the elbow data. */
    elbow: () => store.getState().elbowPoints.map((p) => ({ ...p })),

    /** Score the board and name any failure. Credited to the code lane. */
    check: () => store.getState().check("code"),
    /** Back to two flags. */
    reset: () => store.getState().reset(),
  };
}

export type KMeansApi = ReturnType<typeof createKMeansApi>;

export function CodeLane() {
  // Memoised on nothing: the store is a module singleton, and the verbs read
  // current state at call time, so one instance serves the whole mount.
  const api = useMemo(() => createKMeansApi(), []);

  const lane = useCodeLane({ initialCode: STARTER_CODE, api, maxRunMs: 6000 });

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
        label="Clustering script"
        value={lane.code}
        onChange={lane.setCode}
        language="javascript"
        error={lane.error}
        rows={18}
        hint="api.assign · api.update · api.step · api.settle · api.setK · api.scatterFlags · api.setFlag · api.flags · api.inertia · api.territorySizes · api.elbow · api.check · api.reset"
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
