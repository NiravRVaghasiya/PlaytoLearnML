/**
 * Copy the Pyodide runtime out of node_modules and into `public/pyodide`, and
 * make sure the wheels the Python lane imports are there too.
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
 * ── The wheels ──────────────────────────────────────────────────────────────
 * The npm package ships the runtime but NOT the package wheels (its `files` list
 * stops at the stdlib). A clean install — which is exactly what Vercel and CI do
 * — therefore has no numpy or pandas, and a build that only copies what is in
 * node_modules ships a Python lane that fails on `import pandas`.
 *
 * So wheels come from one of three places, in order:
 *   1. already staged with the right checksum (a warm dev box),
 *   2. cached in node_modules/pyodide (pyodide-smoke.mjs puts them there),
 *   3. downloaded from Pyodide's own CDN, at the exact version the lockfile
 *      pinned.
 * Every wheel, whichever route it came by, is checked against the sha256 in the
 * pinned `pyodide-lock.json`, so a download can never be swapped or truncated.
 * (The browser checks it again: Pyodide fetches each wheel with that sha256 as
 * its subresource-integrity hash.)
 *
 * If a required wheel cannot be obtained, a CI or Vercel build FAILS rather than
 * deploying a Python lane that is broken for every player. A local run only
 * warns, so working offline on anything else is still possible.
 * `PYODIDE_ALLOW_MISSING=1` downgrades the CI/Vercel failure to that same
 * warning — for an offline or firewalled build that knowingly ships without
 * the Python lane. Never set it on the production deployment.
 *
 * ── Where ───────────────────────────────────────────────────────────────────
 * `public/pyodide/`, served at /pyodide/ — unless NEXT_PUBLIC_PYODIDE_INDEX_URL
 * moves it, which src/engine/useCodeLane.ts and next.config.ts read too. It is
 * read from the shell and from the .env files Next loads, in Next's order (see
 * `loadNextEnv`), so all three always agree on the path. A path
 * such as /pyodide/v314.0.5/ is staged at public/pyodide/v314.0.5/ (next.config
 * then caches it as immutable); the path must stay under /pyodide/, and a
 * version segment must name the installed version, exactly as next.config's
 * `pyodideLocation` enforces. An absolute URL means a CDN serves the runtime;
 * the default /pyodide/ copy is still staged (unused, but it keeps unsetting the
 * variable a one-step change).
 *
 * Anything else under public/pyodide (the previous version's runtime or wheels
 * after an upgrade) is removed, so what a dev server serves is exactly what a
 * clean deploy would ship.
 *
 * Run:  node scripts/setup-pyodide.mjs
 */

import { copyFile, mkdir, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { basename, join, sep } from "node:path";

const SOURCE = join("node_modules", "pyodide");
/** Everything this script writes lives under here, and nothing else does. */
const ROOT = join("public", "pyodide");

/** The runtime itself. Without all of these, nothing loads. */
const CORE = [
  "pyodide.asm.wasm",
  "pyodide.asm.mjs",
  "pyodide.mjs",
  "pyodide-lock.json",
  "python_stdlib.zip",
];

/**
 * What the Python lanes `loadPackage()`. Keep in sync with every
 * `createPyodideExecutor({ packages })` call; dependencies are resolved from the
 * lockfile below, so only list top-level imports here.
 */
const LANE_PACKAGES = ["pandas"];

/** Per-attempt download timeout, and how many attempts before giving up. */
const DOWNLOAD_TIMEOUT_MS = 60_000;
const DOWNLOAD_ATTEMPTS = 3;

/** An env flag is on unless unset, empty, or an explicit "0"/"false"/"no". */
const flag = (value) =>
  value !== undefined && value !== "" && !/^(0|false|no)$/i.test(value);

/**
 * A build is strict when a broken Python lane would reach real players.
 *
 * Read from the real environment, BEFORE the .env files below are loaded, so a
 * committed .env can't switch a deploy's strictness off.
 */
const STRICT =
  (flag(process.env.CI) || flag(process.env.VERCEL)) &&
  !flag(process.env.PYODIDE_ALLOW_MISSING);

/**
 * Load .env files exactly as Next will, so NEXT_PUBLIC_PYODIDE_INDEX_URL means
 * the same thing here as in the app.
 *
 * This script runs as plain `node` in predev/prebuild, before Next starts, so
 * it doesn't see what Next loads from .env, .env.local and .env.production. A
 * public, non-secret value like this one naturally goes in .env.production, and
 * then the engine requests /pyodide/v314.0.5/pyodide.mjs (and next.config
 * points the cache rule and CSP there) while this script staged the default
 * /pyodide/ and pruned everything else: a Python lane that 404s for every
 * player, from a green build.
 *
 * So this uses Next's own loader, @next/env (resolved through `next`, which
 * depends on it), with the same mode `next dev` or `next build` is about to
 * use. The shell environment still wins over any file, as it does in Next.
 * Returns the file the variable came from, when it came from one.
 */
function loadNextEnv() {
  const name = "NEXT_PUBLIC_PYODIDE_INDEX_URL";
  const lifecycle = process.env.npm_lifecycle_event;
  // predev → `next dev` (development files); prebuild → `next build`
  // (production files). Run by hand, NODE_ENV=production selects the latter.
  const dev =
    lifecycle === "predev"
      ? true
      : lifecycle === "prebuild"
        ? false
        : process.env.NODE_ENV !== "production";

  let loadEnvConfig;
  try {
    const fromHere = createRequire(import.meta.url);
    ({ loadEnvConfig } = createRequire(fromHere.resolve("next/package.json"))(
      "@next/env",
    ));
  } catch (error) {
    console.warn(
      `WARNING: couldn't load @next/env (${error?.message ?? error}), so .env ` +
        `files were not read. ${name} is taken from the shell environment only.`,
    );
    return null;
  }

  const fromShell = process.env[name] !== undefined;
  const { loadedEnvFiles } = loadEnvConfig(process.cwd(), dev, {
    info: () => {},
    error: (...args) => console.error(...args),
  });
  if (fromShell || process.env[name] === undefined) return null;
  return loadedEnvFiles.find((file) => file.env[name] !== undefined)?.path ?? null;
}

const envFile = loadNextEnv();

if (!existsSync(SOURCE)) {
  console.error(
    `pyodide is not installed. Run your package manager's install first.`,
  );
  process.exit(1);
}

const { version } = JSON.parse(
  await readFile(join(SOURCE, "package.json"), "utf8"),
);
const lock = JSON.parse(await readFile(join(SOURCE, "pyodide-lock.json"), "utf8"));
const CDN = `https://cdn.jsdelivr.net/pyodide/v${version}/full/`;

/**
 * The directory to stage into, from NEXT_PUBLIC_PYODIDE_INDEX_URL. The same
 * rules as `pyodideLocation` in next.config.ts (which can't be imported from a
 * plain-Node script); a value either would reject fails here first, before a
 * build has been spent on it.
 */
function stagingDirectory(raw) {
  const name = "NEXT_PUBLIC_PYODIDE_INDEX_URL";
  const value = raw || "/pyodide/";
  const fail = (why) => {
    console.error(`${name} ${why} (got "${value}").`);
    process.exit(1);
  };
  if (!value.endsWith("/")) fail(`must end in "/"`);
  if (!value.startsWith("/") || value.startsWith("//")) {
    // A CDN URL: not ours to stage. Keep the default copy for local fallback.
    let url;
    try {
      url = new URL(value);
    } catch {
      fail("must be a path like /pyodide/ or an absolute http(s) URL");
    }
    if (url.protocol !== "https:" && url.protocol !== "http:") {
      fail("must be an http(s) URL");
    }
    return ROOT;
  }
  if (!/^\/pyodide\/(?:[A-Za-z0-9_.-]+\/)*$/.test(value)) {
    fail("must be a path under /pyodide/");
  }
  const segments = value.split("/").filter(Boolean);
  if (segments.some((segment) => segment === "." || segment === "..")) {
    fail(`must not contain "." or ".."`);
  }
  const named = segments.filter((segment) => /^v\d+(?:\.\d+)*$/.test(segment));
  if (named.some((segment) => segment !== `v${version}`)) {
    fail(`names ${named.join(", ")} but pyodide ${version} is installed`);
  }
  return join("public", ...segments);
}

const TARGET = stagingDirectory(process.env.NEXT_PUBLIC_PYODIDE_INDEX_URL);

await mkdir(TARGET, { recursive: true });

const mb = (value) => `${(value / 1024 / 1024).toFixed(1)} MB`;
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

let bytes = 0;

for (const name of CORE) {
  const from = join(SOURCE, name);
  if (!existsSync(from)) {
    console.error(`missing required runtime file: ${name}`);
    process.exit(1);
  }
  await copyFile(from, join(TARGET, name));
  bytes += (await stat(from)).size;
}

/** Pyodide's own package-name normalisation (lock keys are in this form). */
const canonical = (name) => name.replace(/[-_.]+/g, "-").toLowerCase();

/** Every lockfile entry a lane needs, dependencies included. */
function closure(names) {
  const seen = new Map();
  const visit = (name) => {
    const key = canonical(name);
    if (seen.has(key)) return;
    const entry = lock.packages[key];
    if (!entry) throw new Error(`"${name}" is not in pyodide-lock.json`);
    // file_name becomes a path under public/ and a URL on the CDN. It is a bare
    // file name in every lockfile Pyodide has shipped; refuse anything else
    // rather than write outside TARGET.
    if (basename(entry.file_name) !== entry.file_name) {
      throw new Error(`unexpected file_name for "${name}": ${entry.file_name}`);
    }
    seen.set(key, entry);
    for (const dependency of entry.depends ?? []) visit(dependency);
  };
  names.forEach(visit);
  return [...seen.values()];
}

/** Returns the wheel's bytes if `path` exists and matches the lockfile. */
async function verified(path, entry) {
  if (!existsSync(path)) return null;
  const data = await readFile(path);
  return sha256(data) === entry.sha256 ? data : null;
}

async function downloadOnce(entry) {
  // Without a timeout a stalled connection would hang the build until the
  // platform's own limit, with no hint as to why.
  const response = await fetch(`${CDN}${entry.file_name}`, {
    signal: AbortSignal.timeout(DOWNLOAD_TIMEOUT_MS),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = Buffer.from(await response.arrayBuffer());
  const actual = sha256(data);
  if (actual !== entry.sha256) {
    throw new Error(`checksum mismatch (expected ${entry.sha256}, got ${actual})`);
  }
  return data;
}

/** A few attempts, so one dropped connection doesn't fail a deploy. */
async function download(entry) {
  let lastError;
  for (let attempt = 1; attempt <= DOWNLOAD_ATTEMPTS; attempt += 1) {
    try {
      return await downloadOnce(entry);
    } catch (error) {
      lastError = error;
      if (attempt < DOWNLOAD_ATTEMPTS) {
        await new Promise((resolve) => setTimeout(resolve, 1000 * attempt));
      }
    }
  }
  throw lastError;
}

const wheels = closure(LANE_PACKAGES);
const missing = [];
let fetched = 0;

for (const entry of wheels) {
  const destination = join(TARGET, entry.file_name);
  let data = await verified(destination, entry);

  // Local copies first: public/pyodide itself when staging into a versioned
  // subdirectory (so switching paths doesn't re-download), then node_modules.
  for (const directory of TARGET === ROOT ? [SOURCE] : [ROOT, SOURCE]) {
    if (data) break;
    data = await verified(join(directory, entry.file_name), entry);
    if (data) await writeFile(destination, data);
  }

  if (!data) {
    try {
      data = await download(entry);
      await writeFile(destination, data);
      fetched += 1;
    } catch (error) {
      // Don't leave a wrong-checksum file behind looking like a wheel.
      await rm(destination, { force: true });
      missing.push(`${entry.name} (${error?.message ?? error})`);
      continue;
    }
  }

  bytes += data.length;
}

// Drop whatever an earlier version or an earlier staging path left behind.
// public/pyodide is this script's alone (gitignored, rebuilt every run), so
// anything in it that is neither the current runtime nor on the way to it goes.
const expected = new Set([...CORE, ...wheels.map((entry) => entry.file_name)]);
let pruned = 0;
async function prune(directory) {
  for (const item of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, item.name);
    const keep = item.isDirectory()
      ? path === TARGET || TARGET.startsWith(path + sep)
      : directory === TARGET && expected.has(item.name);
    if (keep) {
      if (item.isDirectory()) await prune(path);
      continue;
    }
    await rm(path, { recursive: true, force: true });
    pruned += 1;
  }
}
await prune(ROOT);

console.log(
  `pyodide ${version}: ${mb(bytes)} in ${TARGET}` +
    (envFile
      ? ` (NEXT_PUBLIC_PYODIDE_INDEX_URL=${process.env.NEXT_PUBLIC_PYODIDE_INDEX_URL} from ${envFile})`
      : "") +
    (fetched > 0 ? ` (${fetched} wheel(s) downloaded from ${CDN})` : "") +
    (pruned > 0 ? ` (${pruned} stale file(s) removed)` : ""),
);

if (missing.length > 0) {
  const overridden =
    (flag(process.env.CI) || flag(process.env.VERCEL)) && !STRICT;
  console[STRICT ? "error" : "warn"](
    `\n${STRICT ? "ERROR" : "WARNING"}: could not obtain ${missing.join(", ")}.\n` +
      `The Python lane will load but 'import pandas' will fail.\n` +
      `The wheels are fetched from ${CDN}; check network access and re-run\n` +
      `  node scripts/setup-pyodide.mjs\n` +
      `(Behind an HTTP proxy, Node's fetch ignores HTTPS_PROXY unless run as\n` +
      `  node --use-env-proxy scripts/setup-pyodide.mjs)` +
      (overridden
        ? `\nPYODIDE_ALLOW_MISSING is set, so this build continues WITHOUT a working Python lane.`
        : ""),
  );
  if (STRICT) process.exit(1);
}
