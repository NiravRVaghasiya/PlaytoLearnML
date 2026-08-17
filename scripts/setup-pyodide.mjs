/**
 * Copy the Pyodide runtime out of node_modules and into `public/pyodide`.
 *
 * Why this exists rather than loading from a CDN: the Python lane would otherwise
 * make a runtime request to jsdelivr every time a player opened it. Self-hosting
 * keeps the game working offline, keeps the browser playthrough hermetic, and
 * avoids a third-party dependency in the request path of a core feature.
 *
 * Why this exists rather than committing the files: they are ~20 MB of wasm and
 * wheels that are already pinned by the lockfile. `public/pyodide` is gitignored
 * and rebuilt by this script, which runs before dev and before build.
 *
 * The wheels are NOT in the npm package — Pyodide downloads them on first use and
 * caches them into node_modules. So if they are missing here, this script says so
 * and tells you how to fetch them rather than silently shipping a Python lane that
 * cannot import pandas.
 *
 * Run:  node scripts/setup-pyodide.mjs
 */

import { copyFile, mkdir, readdir, stat } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join } from "node:path";

const SOURCE = join("node_modules", "pyodide");
const TARGET = join("public", "pyodide");

/** The runtime itself. Without all of these, nothing loads. */
const CORE = [
  "pyodide.asm.wasm",
  "pyodide.asm.mjs",
  "pyodide.mjs",
  "pyodide-lock.json",
  "python_stdlib.zip",
];

/** What `import pandas` actually needs, transitively. */
const REQUIRED_PACKAGES = ["numpy", "pandas", "python_dateutil", "pytz", "six"];

if (!existsSync(SOURCE)) {
  console.error(
    `pyodide is not installed. Run your package manager's install first.`,
  );
  process.exit(1);
}

await mkdir(TARGET, { recursive: true });

let copied = 0;
let bytes = 0;

async function copy(name) {
  const from = join(SOURCE, name);
  if (!existsSync(from)) return false;
  await copyFile(from, join(TARGET, name));
  bytes += (await stat(from)).size;
  copied += 1;
  return true;
}

for (const name of CORE) {
  if (!(await copy(name))) {
    console.error(`missing required runtime file: ${name}`);
    process.exit(1);
  }
}

const wheels = (await readdir(SOURCE)).filter((name) => name.endsWith(".whl"));
for (const wheel of wheels) await copy(wheel);

const missing = REQUIRED_PACKAGES.filter(
  (pkg) => !wheels.some((wheel) => wheel.startsWith(`${pkg}-`)),
);

const mb = (value) => `${(value / 1024 / 1024).toFixed(1)} MB`;
console.log(
  `pyodide: copied ${copied} files (${mb(bytes)}) to ${TARGET}, ${wheels.length} wheels`,
);

if (missing.length > 0) {
  console.warn(
    `\nWARNING: no wheel found for ${missing.join(", ")}.\n` +
      `The Python lane will load but 'import pandas' will fail.\n` +
      `Fetch them once with:  node scripts/pyodide-smoke.mjs\n` +
      `(that runs loadPackage("pandas"), which caches the wheels into node_modules)`,
  );
  process.exitCode = 0;
}
