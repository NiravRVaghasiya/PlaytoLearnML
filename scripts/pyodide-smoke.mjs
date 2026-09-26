/**
 * Feasibility check for the Pyodide code lane, before anything is built on it.
 *
 * Answers three questions in order, and stops at the first "no":
 *   1. Does the runtime load at all in this environment?
 *   2. Does plain Python run?
 *   3. Is pandas reachable — and how big is the download?
 *
 * Exits non-zero if any answer is "no", so it can be scripted.
 *
 * This runs Pyodide in Node, which fetches the wheels from the CDN into
 * node_modules/pyodide. The build does not depend on that any more:
 * setup-pyodide.mjs downloads and checksums the wheels itself.
 *
 * Run with:  node scripts/pyodide-smoke.mjs
 */

const started = Date.now();
const since = () => `${((Date.now() - started) / 1000).toFixed(1)}s`;

let pyodide;
try {
  const { loadPyodide } = await import("pyodide");
  console.log(`[${since()}] module imported`);
  pyodide = await loadPyodide();
  console.log(`[${since()}] runtime loaded — version ${pyodide.version}`);
} catch (error) {
  console.log(`FAIL loading runtime: ${error?.message ?? error}`);
  process.exit(1);
}

try {
  const value = pyodide.runPython("sum(range(10))");
  console.log(`[${since()}] plain python works: sum(range(10)) = ${value}`);
} catch (error) {
  console.log(`FAIL running python: ${error?.message ?? error}`);
  process.exit(1);
}

// Stdlib only — this is what a Python lane could rely on with no extra download.
try {
  const stdlib = pyodide.runPython(`
import json, statistics, math
json.dumps({"median": statistics.median([1,2,3,4]), "sqrt": math.sqrt(9)})
`);
  console.log(`[${since()}] stdlib works: ${stdlib}`);
} catch (error) {
  console.log(`FAIL stdlib: ${error?.message ?? error}`);
  process.exitCode = 1;
}

// The expensive question.
try {
  console.log(`[${since()}] attempting loadPackage("pandas")...`);
  await pyodide.loadPackage("pandas");
  const result = pyodide.runPython(`
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
str(pd.get_dummies(df, columns=["b"]).to_dict("list"))
`);
  console.log(`[${since()}] PANDAS WORKS — ${result}`);
  console.log(`[${since()}] pandas version ${pyodide.runPython("import pandas; pandas.__version__")}`);
} catch (error) {
  console.log(`[${since()}] PANDAS UNAVAILABLE: ${error?.message ?? error}`);
  process.exitCode = 1;
}

console.log(`\ntotal ${since()}`);
