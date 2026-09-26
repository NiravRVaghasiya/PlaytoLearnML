"use client";

import { useMemo } from "react";
import { Play, RotateCcw } from "lucide-react";
import { Button, CodeEditor } from "@/components";
import { useCodeLane } from "@/engine/useCodeLane";
import { STARTER_CODE, createCodeApi } from "./code-api";

/**
 * Hyperparameter Heist — the code lane's UI.
 *
 * The api the snippet drives, and the starter snippet itself, live in
 * `code-api.ts`: they are plain functions over the same store the sliders and
 * the Try button write (CLAUDE.md two-lane rule), and keeping them out of the
 * component is what lets the tests exercise them directly.
 */
export function CodeLane() {
  // One api object per mount: every verb reads the store when it is called.
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
        label="Cracking script"
        value={lane.code}
        onChange={lane.setCode}
        language="javascript"
        error={lane.error}
        rows={18}
        hint="api.simulate · api.setDials · api.setDialValue · api.try · api.trySuggestion · api.runStrategy · api.objectiveAt · api.influence · api.suggestion · api.surrogateAt · api.gridPoints · api.randomPoints · api.randomSeed · api.dials · api.best · api.triesLeft"
      />

      <section aria-label="Script output" className="min-h-24">
        <h2 className="mb-1 text-xs font-semibold tracking-wide text-text-muted uppercase">
          Output
        </h2>
        {lane.logs.length === 0 ? (
          <p className="font-mono text-xs text-text-muted">
            Run the script to see output here. `api.simulate` costs you nothing —
            only `api.try` and `api.runStrategy` spend the budget.
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
