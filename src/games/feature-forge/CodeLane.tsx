"use client";

import { useMemo, useState } from "react";
import { Play, RotateCcw } from "lucide-react";
import { Button, CodeEditor } from "@/components";
import { createPyodideExecutor, useCodeLane } from "@/engine/useCodeLane";
import { COLUMNS, TARGET_LIFT, TRANSFORMS, type Transform } from "./ml";
import { featureSummaries, useForgeStore } from "./store";

/**
 * Feature Forge — the code lane, in Python.
 *
 * The first Python lane in the project, and the reason the Pyodide seam existed
 * from Phase 1. Real CPython with real pandas, served from /public and loaded on
 * the first run — never on page load, so a player who stays in the visual lane
 * downloads none of it.
 *
 * What pandas is for here is EXPLORATION, not transformation. The transforms
 * themselves stay in the fixed pipeline, because that is what keeps the model
 * honest: if Python could hand over an arbitrary matrix, nothing would stop it
 * fitting statistics on the validation rows and reporting a lift that does not
 * exist. So Python gets the training table to interrogate — group by weekday, look
 * at churn per age band, check a ratio — and then calls `api.forge(...)` with what
 * it found. Which is the real workflow: look at the data, then build the feature.
 */

/** Helpers injected before the player's code. Not shown in the editor. */
const PRELUDE = `
import json
import pandas as pd

def table():
    """The TRAINING rows as a DataFrame, churn included."""
    return pd.DataFrame(json.loads(api.table_json()))

async def forge(transform, *columns):
    """Add a feature and retrain the fixed model. Returns the new score.

    A coroutine because retraining genuinely is asynchronous - it fits a model.
    Call it with 'await forge(...)'; top-level await works here.
    """
    return await api.forge_sync(transform, json.dumps(list(columns)))

def score():
    """Validation accuracy of the current feature set."""
    return api.current_score()

def baseline():
    return api.baseline_score()

def features():
    return json.loads(api.features_json())
`;

const STARTER_CODE = `# Real pandas, running in your browser. Look before you forge.

df = table()
print("rows", len(df), "| churn rate", round(df.churned.mean(), 3))
print("baseline", round(baseline(), 4), "| need +", ${JSON.stringify(TARGET_LIFT)})
print()

# 1. signup_ts is a meaningless number. What is inside it?
df["weekday"] = (df.signup_ts // 86400) % 7
print("churn by weekday:")
print(df.groupby("weekday").churned.mean().round(3).to_string())
print()

# 2. Is the age effect a straight line?
df["age_band"] = pd.cut(df.age, bins=[17, 30, 45, 60, 68, 81])
print("churn by age band:")
print(df.groupby("age_band", observed=True).churned.mean().round(3).to_string())
print()

# 3. city_code is stored 0-5. Does the order mean anything?
print("churn by city:")
print(df.groupby("city_code").churned.mean().round(3).to_string())
print()

# 4. Does income matter on its own, or only per head?
df["per_head"] = df.income / df.household
print("churn by income quartile   ",
      df.groupby(pd.qcut(df.income, 4), observed=True).churned.mean().round(3).values)
print("churn by per-head quartile ",
      df.groupby(pd.qcut(df.per_head, 4), observed=True).churned.mean().round(3).values)
print()

# Now forge what the data actually showed. Each one refits the model,
# so forge is a coroutine and needs awaiting.
for spec in [("day_of_week", "signup_ts"), ("bin", "age"),
             ("one_hot", "city_code"), ("ratio", "income", "household")]:
    after = await forge(*spec)
    print("forged", spec[0], "->", round(after, 4))

print()
print("lift", round(score() - baseline(), 4))

# Notice what the weekday table shows: days 0 and 6 churn far more than the
# rest. A single weight on signup_ts cannot express that - the number only
# ever goes up. Extracting the weekday is not adding information, it is
# making information reachable.
#
# Try forging ("standardise", "refund_issued") and read the score. Then ask
# yourself when that column gets filled in.`;

export function CodeLane() {
  const store = useForgeStore;
  const [pythonStatus, setPythonStatus] = useState<string | null>(null);

  const api = useMemo(
    () => ({
      /** The training table as JSON, for pandas to read. */
      table_json: () => {
        const { dataset } = store.getState();
        const columns: Record<string, number[]> = { churned: [] };
        for (const column of COLUMNS) columns[column.name] = [];
        for (const row of dataset.train) {
          COLUMNS.forEach((column, index) => {
            columns[column.name]!.push(row.values[index] ?? 0);
          });
          columns.churned!.push(row.churned);
        }
        return JSON.stringify(columns);
      },

      /**
       * Forge a feature and retrain, returning the new score.
       *
       * Awaits the retrain internally so Python can call it without dealing in
       * promises — `runPythonAsync` resolves the returned promise for us, and a
       * lane full of `await` noise would obscure the point.
       */
      forge_sync: async (transform: string, columnsJson: string) => {
        const names = TRANSFORMS.map((spec) => spec.id as string);
        if (!names.includes(transform)) {
          throw new Error(
            `unknown transform "${transform}" — try ${names.join(", ")}`,
          );
        }
        const columns = JSON.parse(columnsJson) as string[];
        await store.getState().addFeature(transform as Transform, columns);
        return store.getState().currentScore;
      },

      remove_feature: async (id: string) => {
        await store.getState().removeFeature(id);
        return store.getState().currentScore;
      },
      clear_forge: async () => {
        await store.getState().clearForge();
        return store.getState().currentScore;
      },
      submit: () => store.getState().submit().outcome,

      current_score: () => store.getState().currentScore,
      baseline_score: () => store.getState().baselineScore,
      train_score: () => store.getState().trainScore,
      features_json: () =>
        JSON.stringify(featureSummaries(store.getState().features)),
      importances_json: () => {
        const { columnNames, importances } = store.getState();
        return JSON.stringify(
          columnNames.map((name, index) => ({
            column: name,
            weight: importances[index] ?? 0,
          })),
        );
      },
      legendary_count: () => store.getState().features.length,
      target_lift: () => TARGET_LIFT,
      columns_json: () =>
        JSON.stringify(
          COLUMNS.map((column) => ({
            name: column.name,
            dtype: column.dtype,
            leaky: column.leaky === true,
          })),
        ),
    }),
    [store],
  );

  /**
   * pandas pulls numpy with it — about 7 MB of wheels on top of the 13 MB
   * runtime. Fetched once per visit, from our own /public, and only when someone
   * actually runs Python.
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
    // Generous: the first run downloads a Python runtime and fits five models.
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
          first run downloads about 20 MB and takes a few seconds; nothing is
          fetched until you press Run.
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
        hint="table() · forge(transform, *cols) · score() · baseline() · features() · api.remove_feature · api.clear_forge · api.submit · api.importances_json"
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
