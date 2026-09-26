import {
  AGE_OLD,
  AGE_YOUNG,
  COLUMNS,
  TARGET_LIFT,
  featureProblem,
  legendariesIn,
  type Transform,
} from "./ml";
import { featureSummaries, useForgeStore, type ForgeOutcome } from "./store";

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
export const PRELUDE = `
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

export const STARTER_CODE = `# Real pandas, running in your browser. Look before you forge.

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
df["age_band"] = pd.cut(df.age, bins=[17, ${AGE_YOUNG - 1}, 45, 60, ${AGE_OLD}, 81])
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

# Per head clearly matters, and it looks like a cut-off rather than a slope:
# the bottom quartiles churn alike, then it drops. But a ratio column gets one
# weight, so the model can only answer it with a straight line - dividing
# changes the axis, not the shape - and the baseline's income and household
# weights can already slope a line across the same boundary, because
# "income / household < k" is the same rows as "income - k * household < 0".
# So a ratio is not on the list below. Forge ("ratio", "income", "household")
# afterwards and see what it adds.

# Now forge what the data actually showed. Each one refits the model,
# so forge is a coroutine and needs awaiting. Forging the same feature
# twice is refused, so empty the forge before running this again.
for spec in [("day_of_week", "signup_ts"), ("bin", "age"),
             ("one_hot", "city_code")]:
    after = await forge(*spec)
    print("forged", spec[0], "->", round(after, 4))

print()
print("lift", round(score() - baseline(), 4))

# Notice what the weekday table shows: days 0 and 6 churn far more than the
# rest. A single weight on signup_ts cannot express that - the number only
# ever goes up. Extracting the weekday is not adding information, it is
# making information reachable.
#
# When the lift clears the target, score it from here with api.submit().
#
# Try forging ("standardise", "refund_issued") and read the score. Then ask
# yourself when that column gets filled in.`;

/**
 * A forge that did not happen, as a sentence the player can act on.
 *
 * Thrown from the api so Pyodide raises it as a Python exception at the line
 * that asked for it — which is the difference between "my feature did nothing"
 * and "my feature was never built".
 */
function forgeError(outcome: ForgeOutcome, fitError: string | null): Error | null {
  switch (outcome) {
    case "applied":
      return null;
    case "busy":
      return new Error(
        "the model is still retraining from another forge — await each forge(...) before starting the next",
      );
    case "stale":
      return new Error(
        "the forge was reset while this feature was retraining, so it was dropped",
      );
    case "failed":
      return new Error(
        `retraining failed (${fitError ?? "unknown error"}); the forge is unchanged`,
      );
    case "invalid":
      return new Error("that feature could not be forged");
  }
}

/**
 * The api the Python prelude calls, as a plain object over the store.
 *
 * Outside the component so the tests can drive it exactly as Pyodide does —
 * JSON strings in, promises out — without loading Python. Every verb reads the
 * store when called, so one object per mount is enough.
 */
export function createCodeApi() {
  return {
    /** The training table as JSON, for pandas to read. */
    table_json: () => {
      const { dataset } = useForgeStore.getState();
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
    forge_sync: async (transform: unknown, columnsJson: unknown) => {
      let columns: unknown;
      try {
        columns = JSON.parse(String(columnsJson));
      } catch {
        columns = null;
      }
      if (!Array.isArray(columns)) {
        throw new Error("forge needs column names, e.g. forge(\"bin\", \"age\")");
      }
      if (!useForgeStore.getState().ready) {
        throw new Error(
          "the baseline is still being fitted (or failed to fit) — there is nothing to measure a lift against yet",
        );
      }
      // Every reason a feature can be refused, named before anything trains.
      const problem = featureProblem(
        String(transform),
        columns,
        useForgeStore.getState().features,
      );
      if (problem !== null) throw new Error(problem);

      const outcome = await useForgeStore
        .getState()
        .addFeature(transform as Transform, columns as string[]);
      const error = forgeError(outcome, useForgeStore.getState().fitError);
      if (error) throw error;
      return useForgeStore.getState().currentScore;
    },

    remove_feature: async (id: unknown) => {
      const { features } = useForgeStore.getState();
      if (!features.some((feature) => feature.id === id)) {
        throw new Error(
          `no forged feature with id ${JSON.stringify(id)} — features() lists them: ${
            features.map((feature) => feature.id).join(", ") || "(none)"
          }`,
        );
      }
      const error = forgeError(
        await useForgeStore.getState().removeFeature(String(id)),
        useForgeStore.getState().fitError,
      );
      if (error) throw error;
      return useForgeStore.getState().currentScore;
    },
    clear_forge: async () => {
      const error = forgeError(
        await useForgeStore.getState().clearForge(),
        useForgeStore.getState().fitError,
      );
      if (error) throw error;
      return useForgeStore.getState().currentScore;
    },
    /** Score the forge. Submitted from here, it is a code-lane clear. */
    submit: () => {
      const { training, ready } = useForgeStore.getState();
      if (!ready || training) {
        throw new Error(
          "the model is still retraining — await your last forge(...) before submitting",
        );
      }
      return useForgeStore.getState().submit("code").outcome;
    },

    current_score: () => useForgeStore.getState().currentScore,
    baseline_score: () => useForgeStore.getState().baselineScore,
    train_score: () => useForgeStore.getState().trainScore,
    features_json: () =>
      JSON.stringify(featureSummaries(useForgeStore.getState().features)),
    importances_json: () => {
      const { columnNames, importances } = useForgeStore.getState();
      return JSON.stringify(
        columnNames.map((name, index) => ({
          column: name,
          weight: importances[index] ?? 0,
        })),
      );
    },
    legendary_count: () => legendariesIn(useForgeStore.getState().features).length,
    target_lift: () => TARGET_LIFT,
    columns_json: () =>
      JSON.stringify(
        COLUMNS.map((column) => ({
          name: column.name,
          dtype: column.dtype,
          leaky: column.leaky === true,
        })),
      ),
  };
}
