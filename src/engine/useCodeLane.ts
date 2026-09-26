"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { HighlightLanguage } from "@/lib/highlight";
import { yieldToPaint } from "@/lib/utils";

/**
 * The code lane (CLAUDE.md two-lane rule).
 *
 * A game's code lane is not a second implementation — it is a second *interface*
 * to the same Zustand store. The player edits a snippet, the snippet calls the
 * `api` object this hook injects, and that api mutates exactly the state the
 * visual lane's sliders and drags mutate. That is what keeps the two lanes
 * honest: there is one source of truth, so a code-lane player and a visual-lane
 * player are demonstrably running the same algorithm.
 *
 * ── Security, stated plainly ────────────────────────────────────────────────
 * The JavaScript executor uses `new AsyncFunction(...)`. That runs the player's
 * own code in their own tab, which is the same trust model as a browser console
 * or any code playground. It is NOT a security sandbox: the snippet can still
 * reach globals. Two rules follow, and they matter:
 *
 *   1. NEVER feed this hook code that came from another user, a URL parameter,
 *      or a shared-snippet feature. That turns a playground into stored XSS.
 *      Such a feature needs a Web Worker (or iframe) isolate first.
 *   2. Runaway loops are only *cooperatively* interruptible. `checkBudget()`
 *      throws once the wall-clock budget is spent, but it can only throw when
 *      the snippet calls back into the api. A bare `while (true) {}` will still
 *      hang the tab. A Worker-based executor is the real fix; see
 *      `createPyodideExecutor` for where that seam lives.
 */

export type CodeLanguage = HighlightLanguage;

export interface CodeRunLog {
  level: "log" | "error";
  message: string;
}

export class CodeLaneTimeoutError extends Error {
  constructor(budgetMs: number) {
    // Worded for both causes. The usual one is a loop that never ends, but a
    // genuinely heavy run on a slow device (Convolution Kitchen's learn step on
    // TF.js's CPU backend) hits the same budget, and telling that player their
    // loop is broken would send them hunting for a bug that isn't there.
    super(
      `Your code ran longer than ${budgetMs}ms and was stopped. Look for a loop that never ends, or do less work per run.`,
    );
    this.name = "CodeLaneTimeoutError";
  }
}

export interface CodeRunContext<TApi extends object> {
  /** The game's store, exposed as callable functions. */
  api: TApi;
  /** `log(...)` inside the snippet; surfaces in the lane's output panel. */
  log: (...args: unknown[]) => void;
  /** Throws `CodeLaneTimeoutError` once the run budget is spent. */
  checkBudget: () => void;
}

export type CodeExecutor<TApi extends object> = (
  code: string,
  context: CodeRunContext<TApi>,
) => Promise<unknown>;

/** Async function constructor — `new Function` can't host `await`. */
const AsyncFunction = Object.getPrototypeOf(async function () {})
  .constructor as new (...args: string[]) => (
  ...args: unknown[]
) => Promise<unknown>;

/**
 * JavaScript/TF.js executor. The snippet body may use `await`, and receives
 * `api`, `log`, and `checkBudget` as parameters.
 */
export function createJsExecutor<TApi extends object>(): CodeExecutor<TApi> {
  return async (code, { api, log, checkBudget }) => {
    const fn = new AsyncFunction("api", "log", "checkBudget", code);
    return fn(api, log, checkBudget);
  };
}

/** Minimal shape of the bits of Pyodide this engine touches. */
export interface PyodideRuntime {
  runPythonAsync: (
    code: string,
    options?: { filename?: string },
  ) => Promise<unknown>;
  loadPackage: (
    names: string | string[],
    options?: {
      messageCallback?: (message: string) => void;
      errorCallback?: (message: string) => void;
    },
  ) => Promise<unknown>;
  /**
   * Packages that actually installed, keyed by name. Optional in the type only
   * so a test double can omit it; Pyodide 0.2x+ always provides it.
   */
  loadedPackages?: Record<string, unknown>;
  setStdout: (options: { batched: (text: string) => void }) => void;
  setStderr: (options: { batched: (text: string) => void }) => void;
  globals: { set: (name: string, value: unknown) => void };
  version: string;
}

/** What `pyodide.mjs` exports, as far as this engine is concerned. */
export interface PyodideModule {
  loadPyodide: (options: { indexURL: string }) => Promise<unknown>;
}

/**
 * The default runtime location. A deploy can move it (for example to a
 * versioned, immutably-cached path) by setting `NEXT_PUBLIC_PYODIDE_INDEX_URL`
 * at build time; Next inlines `NEXT_PUBLIC_*` into the client bundle. It must
 * end in a slash and match wherever scripts/setup-pyodide.mjs copies the files.
 */
const DEFAULT_INDEX_URL =
  process.env.NEXT_PUBLIC_PYODIDE_INDEX_URL || "/pyodide/";

/**
 * How long one download step may take before the lane gives up and says so.
 *
 * Generous on purpose: the core is about 13 MB, which is roughly 105 s on a
 * 1 Mbps link. A tighter bound would fail real, merely slow, connections. What
 * it guards against is the case that used to spin forever: a failed
 * `pyodide.asm.wasm` fetch leaves `loadPyodide()` pending with no rejection.
 */
const DEFAULT_LOAD_TIMEOUT_MS = 180_000;

export interface PyodideExecutorOptions {
  /**
   * Python packages to load before the first run, e.g. `["pandas"]`.
   *
   * Each one is a real wheel fetched from `indexURL`. pandas pulls numpy with it
   * and costs about 7 MB, so ask for what the lane actually imports and nothing
   * more.
   */
  packages?: readonly string[];
  /**
   * Python run once per execution, before the player's code.
   *
   * This is where a game puts its own helpers — turning the injected `api` into
   * something idiomatic to use from Python. Kept as a game-supplied string rather
   * than baked in here, because the engine has no business knowing what any
   * particular game's api looks like.
   */
  prelude?: string;
  /** Where the self-hosted runtime lives. See scripts/setup-pyodide.mjs. */
  indexURL?: string;
  /**
   * Progress while the runtime downloads.
   *
   * The hook batches logs and only commits them when a run finishes, which is
   * fine for a 5 ms JavaScript snippet and useless for a 20 MB download. Games
   * wire this to their own state so the lane can say what it is doing.
   */
  onStatus?: (status: string) => void;
  /**
   * Upper bound, per download step (the core runtime; then the packages), before
   * the run fails with a "check your connection" error the player can retry.
   * Defaults to 180 s. `maxRunMs` does NOT apply to Python — see below.
   */
  loadTimeoutMs?: number;
  /**
   * How to fetch `pyodide.mjs`. Games never pass this; it exists so tests can
   * hand in a fake runtime and exercise the failure paths (a wheel that never
   * arrives, a wasm fetch that never settles) without a 13 MB download.
   */
  loadModule?: (indexURL: string) => Promise<PyodideModule>;
}

/** A download step failed or stalled. The message is written for the player. */
export class PyodideLoadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PyodideLoadError";
  }
}

/**
 * A Python exception from the player's code, reduced to the line that matters.
 * `name` is the Python exception type, so the lane shows "NameError: …" rather
 * than "PythonError: Traceback (most recent call last): …".
 */
export class PythonRunError extends Error {
  constructor(type: string, message: string) {
    super(message);
    this.name = type;
  }
}

/**
 * One runtime per index URL, shared across runs and across games.
 *
 * Keyed by index URL only, NOT by package set: the core interpreter is the 13 MB
 * part, and packages are loaded onto it separately on every run (a no-op once
 * they are in). That split is what makes a failed wheel recoverable — the next
 * Run retries just the missing packages on the runtime that already works.
 *
 * The promise is cached rather than the instance so that two lanes mounting at
 * once await the same load instead of racing two of them. A rejected load is
 * evicted, so pressing Run again starts over.
 */
const runtimeCache = new Map<string, Promise<PyodideRuntime>>();

/** Runtimes whose prelude has already paid the first-import cost, per package set. */
const warmedPreludes = new WeakMap<PyodideRuntime, Set<string>>();

function defaultLoadModule(indexURL: string): Promise<PyodideModule> {
  /*
   * Loaded from the self-hosted copy at runtime, NOT bundled.
   *
   * `import("pyodide")` compiles but throws "Cannot find module as expression is
   * too dynamic" in the browser: Pyodide's loader contains conditional
   * `import("node:fs")` calls for its Node build, which no bundler can resolve.
   * The ignore comments keep the bundler out of it entirely, so the browser
   * fetches the module from /public and never evaluates the Node branches.
   *
   * This also keeps ~1 MB of loader JavaScript out of the client bundle, and
   * makes the `pyodide` npm package a build-time asset source rather than a
   * runtime dependency.
   */
  return import(
    /* webpackIgnore: true */ /* turbopackIgnore: true */
    `${indexURL}pyodide.mjs`
  ) as Promise<PyodideModule>;
}

/** Reject with `message` if `promise` hasn't settled within `ms`. */
function withTimeout<T>(promise: Promise<T>, ms: number, message: string): Promise<T> {
  if (!Number.isFinite(ms) || ms <= 0) return promise;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new PyodideLoadError(message)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

async function getRuntime(
  indexURL: string,
  loadModule: (indexURL: string) => Promise<PyodideModule>,
  timeoutMs: number,
  onStatus?: (status: string) => void,
): Promise<PyodideRuntime> {
  const cached = runtimeCache.get(indexURL);
  if (cached) return cached;

  const loading = withTimeout(
    (async () => {
      onStatus?.("Downloading the Python runtime (about 13 MB, once per visit)…");
      let runtimeModule: PyodideModule;
      try {
        // Not named `module`: Next forbids assigning that identifier.
        runtimeModule = await loadModule(indexURL);
      } catch (cause) {
        const detail = cause instanceof Error ? cause.message : String(cause);
        // Keeps the URL it tried: "which file" is the first thing anyone
        // debugging a deploy needs, and it proves a real fetch happened.
        throw new PyodideLoadError(
          `Couldn't load the Python runtime from ${indexURL}pyodide.mjs (${detail}). Check your connection and press Run again.`,
        );
      }
      return (await runtimeModule.loadPyodide({ indexURL })) as PyodideRuntime;
    })(),
    timeoutMs,
    `Downloading the Python runtime took longer than ${Math.round(timeoutMs / 1000)} s. Check your connection and press Run again.`,
  );

  runtimeCache.set(indexURL, loading);

  try {
    return await loading;
  } catch (cause) {
    // Don't cache a failed load — a retry should be allowed to work. A timed-out
    // load can't be aborted (the wasm fetch is the browser's), but dropping it
    // frees the lane and lets the next Run start a fresh one.
    if (runtimeCache.get(indexURL) === loading) runtimeCache.delete(indexURL);
    throw cause;
  }
}

/** Pyodide's lockfile names use dashes; callers sometimes write underscores. */
function packageKey(name: string): string {
  return name.toLowerCase().replace(/_/g, "-");
}

function missingPackages(
  runtime: PyodideRuntime,
  packages: readonly string[],
): string[] {
  const loaded = runtime.loadedPackages;
  // No registry to check against (only a test double would lack one): trust it.
  if (!loaded) return [];
  const have = new Set(Object.keys(loaded).map(packageKey));
  return packages.filter((name) => !have.has(packageKey(name)));
}

/**
 * Make sure every requested package is installed on `runtime`.
 *
 * `loadPackage` does NOT reject when a wheel fails to download: Pyodide catches
 * the error, logs "Failed to load …" and resolves with whatever did install. So
 * the only trustworthy check is to look at `loadedPackages` afterwards. Throwing
 * here — rather than caching a runtime that says "ready" but has no pandas — is
 * what lets the next Run retry just the missing wheels.
 */
async function ensurePackages(
  runtime: PyodideRuntime,
  packages: readonly string[],
  timeoutMs: number,
  onStatus?: (status: string) => void,
): Promise<void> {
  const missing = missingPackages(runtime, packages);
  if (packages.length === 0 || (runtime.loadedPackages && missing.length === 0)) {
    return;
  }

  const wanted = runtime.loadedPackages ? missing : [...packages];
  onStatus?.(`Loading ${wanted.join(", ")}…`);

  const failures: string[] = [];
  await withTimeout(
    runtime.loadPackage([...wanted], {
      errorCallback: (message) => failures.push(message),
    }),
    timeoutMs,
    `Downloading ${wanted.join(", ")} took longer than ${Math.round(timeoutMs / 1000)} s. Check your connection and press Run again.`,
  );

  const stillMissing = missingPackages(runtime, packages);
  if (stillMissing.length > 0) {
    // Pyodide logs a lead-in first ("The following error occurred while
    // loading numpy:") and the actual error after it. The lead-in is not a
    // reason, so skip anything that ends in a colon.
    const detail = failures
      .map((message) => message.trim())
      .find((message) => message.length > 0 && !message.endsWith(":"));
    const reason = detail ? ` (${detail.replace(/\.+$/, "")})` : "";
    throw new PyodideLoadError(
      `Couldn't download ${stillMissing.join(", ")}${reason}. Check your connection and press Run again.`,
    );
  }
}

/** Line number of the last traceback frame that is in the player's own code. */
function playerLine(traceback: string, filename: string): number | null {
  const escaped = filename.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = new RegExp(`File "${escaped}", line (\\d+)`, "g");
  let line: number | null = null;
  for (const match of traceback.matchAll(pattern)) line = Number(match[1]);
  return line;
}

/**
 * Turn a Pyodide `PythonError` into one readable line (plus the full traceback
 * in the Output panel for anyone who wants it).
 */
function toPythonRunError(
  cause: unknown,
  filename: string,
  log: (...args: unknown[]) => void,
  where: string,
): unknown {
  if (!(cause instanceof Error)) return cause;
  // Only Python exceptions carry a traceback worth reducing. A JavaScript error
  // thrown by the game's api surfaces inside Python as a JsException traceback,
  // so it takes this path too and reads as "JsException: Error: …".
  if (!/Traceback|File "/.test(cause.message)) return cause;

  log(cause.message.trimEnd());

  const summary = formatPythonError(cause.message);
  const colon = summary.indexOf(": ");
  const type =
    colon > 0 ? summary.slice(0, colon) : summary || "PythonError";
  const detail = colon > 0 ? summary.slice(colon + 2) : "";
  const line = playerLine(cause.message, filename);
  const at = line === null ? where : `line ${line}${where ? ` ${where}` : ""}`;

  return new PythonRunError(
    type.split(".").pop() || type,
    [detail, at ? `(${at})` : ""].filter(Boolean).join(" "),
  );
}

/**
 * Filenames Python reports in tracebacks. Distinct, so a frame in the game's
 * helper code is never mistaken for a line of the player's own.
 */
const PLAYER_FILENAME = "<your code>";
const PRELUDE_FILENAME = "<lane setup>";

/**
 * Pyodide executor: real CPython in the tab (spec: Feature Forge runs real pandas).
 *
 * The runtime is served from `public/pyodide` rather than a CDN, and it is loaded
 * lazily on the first Python run — never on page load. That is what keeps the
 * default JavaScript lanes fast: a player who never opens a Python lane never
 * downloads any of it.
 *
 * Failure handling, because a 20 MB download on a phone WILL fail sometimes:
 * a runtime that fails or stalls (see `loadTimeoutMs`) is not cached, a package
 * that didn't install is detected rather than trusted, and either way the run
 * rejects with a message saying to press Run again — which then works.
 *
 * ── Three honest limitations ────────────────────────────────────────────────
 * 1. A Python run is NOT interruptible, and `maxRunMs`/`checkBudget()` do not
 *    apply to it. They work for JavaScript because the snippet keeps calling back
 *    into the api; Python executes inside wasm and does not yield. Interrupting it
 *    properly needs `setInterruptBuffer` with a SharedArrayBuffer, which needs
 *    COOP/COEP headers on every response. Until that is in place, a `while True:`
 *    in the Python lane hangs the tab exactly as it would in a JavaScript one.
 * 2. It runs on the main thread. The first `import pandas` blocks the page for a
 *    few seconds; the executor says so in `onStatus` and yields a paint first, so
 *    the pause is announced rather than looking like a crash. A Worker-hosted
 *    runtime is the real fix and a larger change (every api call becomes async).
 * 3. It is not a security sandbox, for the same reason the JavaScript executor
 *    is not. See the note at the top of this file — the rule about never running
 *    code that came from another user applies here identically.
 */
export function createPyodideExecutor<TApi extends object>({
  packages = [],
  prelude,
  indexURL = DEFAULT_INDEX_URL,
  onStatus,
  loadTimeoutMs = DEFAULT_LOAD_TIMEOUT_MS,
  loadModule = defaultLoadModule,
}: PyodideExecutorOptions = {}): CodeExecutor<TApi> {
  const packageSetKey = [...packages].map(packageKey).sort().join(",");

  return async (code, { api, log }) => {
    const runtime = await getRuntime(indexURL, loadModule, loadTimeoutMs, onStatus);
    await ensurePackages(runtime, packages, loadTimeoutMs, onStatus);

    // print() and stderr land in the lane's output panel, so Python players use
    // the language's own idiom rather than learning a bespoke log function.
    runtime.setStdout({
      batched: (text) => {
        if (text.length > 0) log(text);
      },
    });
    runtime.setStderr({
      batched: (text) => {
        if (text.length > 0) log(text);
      },
    });

    runtime.globals.set("api", api);

    const warmed = warmedPreludes.get(runtime) ?? new Set<string>();
    const firstRun = !warmed.has(packageSetKey);

    if (prelude) {
      if (firstRun && packages.length > 0) {
        // The prelude's first `import pandas` is several seconds of synchronous
        // wasm. Say so, and let that sentence paint, before the page freezes.
        onStatus?.(
          `Starting ${packages.join(", ")} — the page may pause for a few seconds…`,
        );
        await yieldToPaint();
      }
      try {
        await runtime.runPythonAsync(prelude, { filename: PRELUDE_FILENAME });
      } catch (cause) {
        throw toPythonRunError(cause, PRELUDE_FILENAME, log, "in the lane's setup code");
      }
    }

    if (firstRun) {
      warmed.add(packageSetKey);
      warmedPreludes.set(runtime, warmed);
      onStatus?.(`Python ${runtime.version} ready.`);
    }

    try {
      return await runtime.runPythonAsync(code, { filename: PLAYER_FILENAME });
    } catch (cause) {
      throw toPythonRunError(cause, PLAYER_FILENAME, log, "");
    }
  };
}

/**
 * Pull the useful line out of a Pyodide traceback.
 *
 * A `PythonError` message is the whole traceback, which is many lines of wasm
 * frames ending in the one line that matters. Showing the lot buries the error.
 */
export function formatPythonError(message: string): string {
  const lines = message
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  const lastMeaningful = [...lines]
    .reverse()
    .find((line) => /^[A-Za-z_][\w.]*(Error|Exception|Warning)\b/.test(line));

  return lastMeaningful ?? lines[lines.length - 1] ?? message;
}

export interface UseCodeLaneOptions<TApi extends object> {
  /** Starter snippet. `reset()` restores exactly this. */
  initialCode: string;
  /**
   * The game's store surface. Keep it small and verb-shaped
   * (`step()`, `setK(3)`, `assign()`) so the snippet reads like the algorithm.
   */
  api: TApi;
  language?: CodeLanguage;
  /**
   * Defaults to the JavaScript executor, or a bare Pyodide one when the language
   * is python. Games that need packages or a prelude build their own with
   * `createPyodideExecutor({ packages, prelude, onStatus })`.
   */
  executor?: CodeExecutor<TApi>;
  /**
   * Cooperative wall-clock budget per run, for JavaScript lanes. Enforced
   * wherever `checkBudget()` is called — by the snippet itself, or on every api
   * call when `budgetApiCalls` is set. It never applies to Python (see
   * `createPyodideExecutor`).
   */
  maxRunMs?: number;
  /**
   * Check the budget on every call into `api`, not only where the snippet calls
   * `checkBudget()` itself. Off by default because it changes when a slow run
   * stops; a game turns it on when its api is where the time goes, so that
   * `for (;;) await api.train()` times out instead of spinning forever.
   */
  budgetApiCalls?: boolean;
  /** Run once on mount so the lane isn't inert on arrival. */
  autoRun?: boolean;
  /** Cap on retained log lines. */
  maxLogs?: number;
  /**
   * Where the player's draft is kept while the lane is unmounted. `GameShell`
   * mounts only the active lane, so without this a switch to the visual lane
   * and back threw the player's edits (and their last output) away.
   *
   * Defaults to `initialCode`, which is already unique per game. Pass a key of
   * your own if two lanes share a starter, or `null` to opt out. The draft
   * lives in memory for the page's lifetime, never in storage; `reset()`
   * forgets it.
   */
  persistKey?: string | null;
}

export interface UseCodeLaneApi {
  code: string;
  setCode: (code: string) => void;
  /** True once the player has edited away from `initialCode`. */
  dirty: boolean;
  running: boolean;
  /** Last run's error, formatted for display. Shown, never thrown. */
  error: string | null;
  logs: readonly CodeRunLog[];
  run: () => Promise<void>;
  reset: () => void;
  clearLogs: () => void;
  language: CodeLanguage;
  /**
   * Whether this language has a real executor. Always true for "javascript"
   * and "python" now that Pyodide is wired in; kept so a future stub language
   * can say "not yet" honestly instead of failing on Run.
   */
  available: boolean;
}

/**
 * What JSON can't show, as something it can. Used as the `JSON.stringify`
 * replacer, so it applies at every depth, not only to the top-level value.
 */
function toLoggable(value: unknown): unknown {
  if (value instanceof Error) return `${value.name}: ${value.message}`;
  if (typeof value === "bigint") return `${value}n`;
  if (value instanceof Map) {
    // An object when every key can be one; otherwise entries, so an object
    // key isn't flattened to "[object Object]".
    const plainKeys = [...value.keys()].every(
      (key) => typeof key === "string" || typeof key === "number",
    );
    return plainKeys ? Object.fromEntries(value) : [...value.entries()];
  }
  if (value instanceof Set) return [...value];
  // Typed arrays; their elements (bigints included) come back through here.
  if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
    return Array.from(value as unknown as ArrayLike<unknown>);
  }
  return value;
}

/**
 * How `log(x)` renders a value in the Output panel.
 *
 * JSON is the right default for plain data, but it is wrong for exactly the
 * things a learner reaches for when debugging: `JSON.stringify(new Error("x"))`
 * is `"{}"`, and a function or symbol stringifies to `undefined`, which joined
 * into an empty line. `try { … } catch (e) { log(e) }` has to show the error.
 */
export function formatLogArg(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (typeof value === "bigint") return `${value}n`;
  if (value === null) return "null";
  if (value === undefined) return "undefined";
  if (typeof value === "symbol") return value.toString();
  if (typeof value === "function") {
    return `[function ${value.name || "anonymous"}]`;
  }
  if (value instanceof Error) return `${value.name}: ${value.message}`;
  try {
    const json = JSON.stringify(value, (_key, inner: unknown) => toLoggable(inner));
    if (value instanceof Map) return `Map ${json}`;
    if (value instanceof Set) return `Set ${json}`;
    return json ?? String(value);
  } catch {
    // Circular structures land here.
    return String(value);
  }
}

/**
 * Wrap every function on `api` so it checks the run budget before it runs.
 *
 * A Proxy rather than a copied object, so getters and non-function values keep
 * reading live store state. Non-configurable, read-only properties are passed
 * through untouched: the Proxy invariants forbid returning anything else for
 * them (a frozen api object would otherwise throw on every access).
 */
function withBudget<TApi extends object>(api: TApi, checkBudget: () => void): TApi {
  const wrappers = new WeakMap<(...args: unknown[]) => unknown, unknown>();
  const proxy: TApi = new Proxy(api, {
    get(target, property) {
      const value: unknown = Reflect.get(target, property, target);
      if (typeof value !== "function") return value;
      const descriptor = Reflect.getOwnPropertyDescriptor(target, property);
      if (descriptor && !descriptor.configurable && !descriptor.writable) {
        return value;
      }
      const fn = value as (...args: unknown[]) => unknown;
      let wrapper = wrappers.get(fn);
      if (!wrapper) {
        wrapper = function (this: unknown, ...args: unknown[]) {
          checkBudget();
          return fn.apply(this === proxy || this === undefined ? target : this, args);
        };
        wrappers.set(fn, wrapper);
      }
      return wrapper;
    },
  });
  return proxy;
}

/** What a lane leaves behind when it unmounts. */
interface CodeLaneDraft {
  code: string;
  error: string | null;
  logs: readonly CodeRunLog[];
}

/**
 * Drafts by `persistKey`, for the life of the page. Module-level because the
 * lane component is unmounted whenever the player switches lanes, and its
 * state goes with it; the game's Zustand store is the wrong home for a text
 * buffer every game would have to re-plumb.
 */
const drafts = new Map<string, CodeLaneDraft>();

/** Test seam: forget every remembered draft. Never needed by app code. */
export function forgetCodeLaneDraftsForTests(): void {
  drafts.clear();
}

export function useCodeLane<TApi extends object>({
  initialCode,
  api,
  language = "javascript",
  executor,
  maxRunMs = 4000,
  budgetApiCalls = false,
  autoRun = false,
  maxLogs = 100,
  persistKey,
}: UseCodeLaneOptions<TApi>): UseCodeLaneApi {
  const draftKey = persistKey === undefined ? initialCode : persistKey;
  // Read once, on mount: a lane coming back picks up where the player left it.
  const [restored] = useState(() =>
    draftKey === null ? undefined : drafts.get(draftKey),
  );
  const [code, setCodeState] = useState(restored?.code ?? initialCode);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(restored?.error ?? null);
  const [logs, setLogs] = useState<readonly CodeRunLog[]>(restored?.logs ?? []);

  /**
   * The snippet text, mirrored into a ref so `run()` is stable and always sees
   * the newest code — including when a component calls `setCode(...)` and
   * `run()` in the same handler, where reading state would be a render behind.
   * Written from event handlers only, never during render.
   */
  const codeRef = useRef(restored?.code ?? initialCode);

  /** Write part of the draft through to the cache. Event handlers only. */
  const remember = useCallback(
    (patch: Partial<CodeLaneDraft>) => {
      if (draftKey === null) return;
      const previous = drafts.get(draftKey) ?? {
        code: codeRef.current,
        error: null,
        logs: [],
      };
      drafts.set(draftKey, { ...previous, ...patch });
    },
    [draftKey],
  );

  const setCode = useCallback(
    (next: string) => {
      codeRef.current = next;
      setCodeState(next);
      remember({ code: next });
    },
    [remember],
  );

  // Props mirrored in an effect: writing a ref during render is unsafe under
  // concurrent React, and `run()` only ever fires from an event or an effect.
  const apiRef = useRef(api);
  const executorRef = useRef<CodeExecutor<TApi> | undefined>(executor);
  useEffect(() => {
    apiRef.current = api;
    executorRef.current = executor;
  });

  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  /**
   * Both languages now have real executors, so this is always true.
   *
   * Kept rather than removed: it existed to let a lane say "Python isn't wired up
   * yet", and the honest thing when that stops being the case is to report it
   * rather than to leave a flag that is quietly always false. If a third language
   * is ever added as a stub, this is where it says so.
   */
  const available =
    language === "javascript" || language === "python" || executor !== undefined;

  /**
   * One run at a time. A second `run()` while one is in flight (a double click
   * in the frame before the button disables, or autoRun racing a click) would
   * interleave two snippets' api calls against one store.
   */
  const inFlightRef = useRef(false);

  const run = useCallback(async () => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;

    const activeExecutor =
      executorRef.current ??
      (language === "python"
        ? createPyodideExecutor<TApi>()
        : createJsExecutor<TApi>());

    const collected: CodeRunLog[] = [];
    let runError: string | null = null;

    setRunning(true);
    setError(null);

    try {
      // Let "Running…" paint before the snippet starts. Without this, a
      // synchronous snippet (Agent Academy's and Hyperparameter Heist's
      // starters are seconds of Monte-Carlo) runs inside the click handler, so
      // React never commits `running` and the tab just looks frozen.
      await yieldToPaint();

      // The budget starts after the yield, so the frame spent painting is
      // never charged to the player's code.
      const startedAt = Date.now();
      const checkBudget = () => {
        if (Date.now() - startedAt > maxRunMs) {
          throw new CodeLaneTimeoutError(maxRunMs);
        }
      };
      const api =
        budgetApiCalls && language !== "python"
          ? withBudget(apiRef.current, checkBudget)
          : apiRef.current;

      await activeExecutor(codeRef.current, {
        api,
        log: (...args: unknown[]) => {
          if (collected.length < maxLogs) {
            collected.push({
              level: "log",
              message: args.map(formatLogArg).join(" "),
            });
          }
        },
        checkBudget,
      });
    } catch (cause) {
      // The engine's own errors are sentences written for the player; a class
      // name in front ("PyodideLoadError: …") is only jargon. The player's
      // errors keep theirs, because "TypeError" or "NameError" is the clue.
      runError =
        cause instanceof PyodideLoadError || cause instanceof CodeLaneTimeoutError
          ? cause.message
          : cause instanceof Error
            ? `${cause.name}: ${cause.message}`
            : String(cause);
      collected.push({ level: "error", message: runError });
      if (mountedRef.current) setError(runError);
    } finally {
      inFlightRef.current = false;
      const kept = collected.slice(-maxLogs);
      // Remembered even if the lane unmounted mid-run (the player switched
      // lanes to watch it), so the output is there when they come back.
      remember({ logs: kept, error: runError });
      if (mountedRef.current) {
        setLogs(kept);
        setRunning(false);
      }
    }
  }, [language, maxRunMs, maxLogs, budgetApiCalls, remember]);

  const reset = useCallback(() => {
    setCode(initialCode);
    setError(null);
    setLogs([]);
    if (draftKey !== null) drafts.delete(draftKey);
  }, [initialCode, setCode, draftKey]);

  const clearLogs = useCallback(() => {
    setLogs([]);
    remember({ logs: [] });
  }, [remember]);

  // Run-on-mount is opt-in and fires once.
  const autoRunDoneRef = useRef(false);
  useEffect(() => {
    if (!autoRun || autoRunDoneRef.current) return;
    autoRunDoneRef.current = true;
    void run();
  }, [autoRun, run]);

  return {
    code,
    setCode,
    dirty: code !== initialCode,
    running,
    error,
    logs,
    run,
    reset,
    clearLogs,
    language,
    available,
  };
}
