"use client";

import { useMemo, useState } from "react";
import { Play, RotateCcw } from "lucide-react";
import { Button, CodeEditor } from "@/components";
import { createPyodideExecutor, useCodeLane } from "@/engine/useCodeLane";
import { PRELUDE, STARTER_CODE, createCodeApi } from "./code-api";

/**
 * Feature Forge — the Python lane's UI. The api, the prelude that wraps it in
 * Python and the starter snippet live in `code-api.ts`, where the tests can
 * reach them without loading Pyodide.
 */
export function CodeLane() {
  const [pythonStatus, setPythonStatus] = useState<string | null>(null);

  // One api object per mount: every verb reads the store when it is called.
  const api = useMemo(() => createCodeApi(), []);

  /**
   * pandas pulls numpy with it — about 7 MB of wheels on top of the 13 MB
   * runtime. Fetched once per visit, from our own /public, and only when someone
   * actually runs Python. `scripts/setup-pyodide.mjs` vendors exactly what this
   * list needs (its `LANE_PACKAGES`), so a package added here must be added
   * there too or the build will ship a lane that cannot import it.
   */
  const executor = useMemo(
    () =>
      createPyodideExecutor<typeof api>({
        packages: ["pandas"],
        prelude: PRELUDE,
        onStatus: setPythonStatus,
      }),
    [],
  );

  const lane = useCodeLane({
    initialCode: STARTER_CODE,
    api,
    language: "python",
    executor,
    // NOT a hard stop for this lane. `checkBudget` is cooperative and a Python
    // run executes inside wasm without calling back into it, so a Python snippet
    // cannot be interrupted (see createPyodideExecutor). Kept generous so that
    // if the engine ever does enforce it, the first run's download and the
    // starter's fits sit comfortably inside it.
    maxRunMs: 240000,
  });

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
          {lane.running ? "Running…" : "Run Python"}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={lane.reset}
          icon={<RotateCcw className="size-4" />}
        >
          Restore snippet
        </Button>
        {lane.running && pythonStatus ? (
          <span
            className="font-mono text-xs text-text-muted"
            data-testid="python-status"
          >
            {pythonStatus}
          </span>
        ) : lane.dirty ? (
          <span className="font-mono text-xs text-text-muted">edited</span>
        ) : null}
      </div>

      {!lane.running && pythonStatus === null ? (
        <p className="rounded-md border border-border bg-surface-2 p-2 text-[11px] text-text-muted">
          This lane runs real CPython with pandas, compiled to WebAssembly. The
          first run downloads about 20 MB, and the page pauses for a few seconds
          while pandas starts up; nothing is fetched until you press Run.
        </p>
      ) : null}

      <CodeEditor
        className="min-h-0 flex-1"
        label="Feature script"
        value={lane.code}
        onChange={lane.setCode}
        language="python"
        error={lane.error}
        rows={18}
        hint="table() · forge(transform, *cols) · score() · baseline() · features() · api.remove_feature · api.clear_forge · api.submit · api.importances_json · api.legendary_count"
      />

      <section aria-label="Script output" className="min-h-24">
        <h2 className="mb-1 text-xs font-semibold tracking-wide text-text-muted uppercase">
          Output
        </h2>
        {lane.logs.length === 0 ? (
          <p className="font-mono text-xs text-text-muted">
            Run the script to see output here. `print()` goes straight to this
            panel.
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
