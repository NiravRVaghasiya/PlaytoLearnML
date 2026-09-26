"use client";

import { useMemo } from "react";
import { Play, RotateCcw } from "lucide-react";
import { Button, CodeEditor } from "@/components";
import { useCodeLane } from "@/engine/useCodeLane";
import { useSortItStore } from "./store";
import { KNOT_COUNTS, type BoundaryType } from "./ml";

/**
 * Sort-It Arcade — the code lane.
 *
 * This is not a second implementation of the game. Every function in `api`
 * below is the *same store action* the visual lane's drag handlers call, so a
 * player who types `api.setKnot(2, 0.7)` and a player who drags handle 3 upward
 * perform an identical state transition and get an identical metric (CLAUDE.md
 * two-lane rule).
 *
 * `api.autoFit()` is the one addition, and it earns its place: it runs the real
 * coordinate-descent optimizer from `ml.ts`. That makes the pedagogy contract's
 * second point explicit — the player can watch the algorithm perform the same
 * search their hands were doing.
 *
 * Every verb checks its arguments and throws a named error the output panel can
 * show. The store would quietly ignore a bad value, which is safe but silent —
 * and a player who typed `api.setKnot(1)` deserves to be told the height is
 * missing, not left wondering why nothing moved.
 */

function shown(value: unknown): string {
  if (typeof value === "string") return `"${value}"`;
  if (typeof value === "number") return String(value);
  return value === undefined ? "nothing" : typeof value;
}

export const STARTER_CODE = `// Fit a decision boundary. Every call below drives the same
// game state the visual lane does — flip the toggle and look.

// 1. Choose your model's capacity: 'line' (2), 'curve' (5), 'wiggle' (25).
api.setBoundaryType('curve');

// 2. Let the optimizer place the knots. This sweeps each knot to the
//    height that maximises TRAINING accuracy — no regularization,
//    exactly like an unregularised learner.
api.autoFit();

log('train accuracy', api.trainAccuracy().toFixed(3));
log('parameters    ', api.complexity());
log('penalty       ', api.penalty().toFixed(3));
log('score         ', api.score().toFixed(3));

// 3. Now score it against points the model has never seen.
const result = api.check();
log('---');
log('held-out accuracy', result.testAccuracy.toFixed(3));
log('generalization gap', result.generalizationGap.toFixed(3));
log('verdict', result.outcome);

// Try it with 'wiggle'. Training accuracy goes UP, the generalization
// gap widens, and the verdict gets worse: 20 more parameters cost more
// score than they buy. Add api.reset() before api.autoFit() to fit the
// wiggle from a flat line, and it memorises enough noise to be named
// Overfitting.`;

/**
 * The code lane's verbs, bound to a store. A plain function rather than inline
 * in the component so the argument checks can be unit-tested without React.
 */
export function createSortItApi(store: typeof useSortItStore = useSortItStore) {
  return {
    /** Swap model capacity. Same action as the capacity buttons. */
    setBoundaryType: (type: BoundaryType) => {
      // `Object.hasOwn`, not `in`: `'toString' in KNOT_COUNTS` is true.
      if (typeof type !== "string" || !Object.hasOwn(KNOT_COUNTS, type)) {
        throw new TypeError(
          `setBoundaryType(type): unknown boundary type ${shown(type)}. Use 'line', 'curve' or 'wiggle'.`,
        );
      }
      store.getState().setBoundaryType(type);
    },
    /** Move one knot. Same action as dragging handle `index`. */
    setKnot: (index: number, height: number) => {
      const count = store.getState().boundary.params.length;
      if (!Number.isInteger(index) || index < 0 || index >= count) {
        throw new RangeError(
          `setKnot(index, height): index must be a whole number from 0 to ${
            count - 1
          }, got ${shown(index)}.`,
        );
      }
      if (typeof height !== "number" || !Number.isFinite(height)) {
        throw new TypeError(
          `setKnot(index, height): height must be a number from 0 to 1, got ${shown(height)}.`,
        );
      }
      if (height < 0 || height > 1) {
        throw new RangeError(
          `setKnot(index, height): height must be from 0 (bottom) to 1 (top), got ${height}.`,
        );
      }
      store.getState().setKnot(index, height);
    },
    /** Current knot heights. */
    knots: () => [...store.getState().boundary.params],
    /** Run the real optimizer from ml.ts. */
    autoFit: () => {
      store.getState().autoFit();
    },
    /** Live training accuracy — the same number the metric shows. */
    trainAccuracy: () => store.getState().accuracy,
    /** Complexity penalty currently being charged. */
    penalty: () => store.getState().penalty,
    /** Parameter count. */
    complexity: () => store.getState().boundary.complexityCost,
    /** accuracy − penalty. */
    score: () => store.getState().score,
    /** Score against the held-out set and name any failure. */
    check: () => store.getState().check("code"),
    /** Flatten the boundary and clear the verdict. */
    reset: () => {
      store.getState().reset();
    },
  };
}

export type SortItApi = ReturnType<typeof createSortItApi>;

export function CodeLane() {
  // Memoised on nothing: the store is a module singleton, and the verbs read
  // current state at call time, so one instance serves the whole mount.
  const api = useMemo(() => createSortItApi(), []);

  const lane = useCodeLane({ initialCode: STARTER_CODE, api });

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
        label="Boundary-fitting script"
        value={lane.code}
        onChange={lane.setCode}
        language="javascript"
        error={lane.error}
        rows={16}
        hint="api.setBoundaryType · api.setKnot · api.knots · api.autoFit · api.trainAccuracy · api.penalty · api.complexity · api.score · api.check · api.reset"
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
