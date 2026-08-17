"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { HighlightLanguage } from "@/lib/highlight";

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
    super(
      `Your code ran longer than ${budgetMs}ms and was stopped. Look for a loop that never ends.`,
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
interface PyodideRuntime {
  runPythonAsync: (code: string) => Promise<unknown>;
  loadPackage: (names: string | string[]) => Promise<unknown>;
  setStdout: (options: { batched: (text: string) => void }) => void;
  setStderr: (options: { batched: (text: string) => void }) => void;
  globals: { set: (name: string, value: unknown) => void };
  version: string;
}

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
}

/**
 * One runtime per page, shared across runs and across games.
 *
 * Keyed by index URL and package set: loading Python twice would cost the
 * download twice. The promise is cached rather than the instance so that two
 * lanes mounting at once await the same load instead of racing two of them.
 */
const runtimeCache = new Map<string, Promise<PyodideRuntime>>();

async function getRuntime(
  indexURL: string,
  packages: readonly string[],
  onStatus?: (status: string) => void,
): Promise<PyodideRuntime> {
  const key = `${indexURL}|${[...packages].sort().join(",")}`;
  const cached = runtimeCache.get(key);
  if (cached) return cached;

  const loading = (async () => {
    onStatus?.("Downloading the Python runtime (about 13 MB, once per visit)…");

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
    // Not named `module`: Next forbids assigning that identifier.
    const runtimeModule = (await import(
      /* webpackIgnore: true */ /* turbopackIgnore: true */
      `${indexURL}pyodide.mjs`
    )) as { loadPyodide: (options: { indexURL: string }) => Promise<unknown> };

    const runtime = (await runtimeModule.loadPyodide({
      indexURL,
    })) as PyodideRuntime;

    if (packages.length > 0) {
      onStatus?.(`Loading ${packages.join(", ")}…`);
      await runtime.loadPackage([...packages]);
    }

    onStatus?.(`Python ${runtime.version} ready.`);
    return runtime;
  })();

  runtimeCache.set(key, loading);

  try {
    return await loading;
  } catch (cause) {
    // Don't cache a failed load — a retry should be allowed to work.
    runtimeCache.delete(key);
    throw cause;
  }
}

/**
 * Pyodide executor: real CPython in the tab (spec: Feature Forge runs real pandas).
 *
 * The runtime is served from `public/pyodide` rather than a CDN, and it is loaded
 * lazily on the first Python run — never on page load. That is what keeps the
 * default JavaScript lanes fast: a player who never opens a Python lane never
 * downloads any of it.
 *
 * ── Two honest limitations ──────────────────────────────────────────────────
 * 1. A Python run is NOT interruptible. `checkBudget()` works for JavaScript
 *    because the snippet keeps calling back into the api; Python executes inside
 *    wasm and does not yield. Interrupting it properly needs
 *    `setInterruptBuffer` with a SharedArrayBuffer, which needs COOP/COEP headers
 *    on every response. Until that is in place, a `while True:` in the Python lane
 *    hangs the tab exactly as it would in a JavaScript one.
 * 2. It is not a security sandbox, for the same reason the JavaScript executor
 *    is not. See the note at the top of this file — the rule about never running
 *    code that came from another user applies here identically.
 */
export function createPyodideExecutor<TApi extends object>({
  packages = [],
  prelude,
  indexURL = "/pyodide/",
  onStatus,
}: PyodideExecutorOptions = {}): CodeExecutor<TApi> {
  return async (code, { api, log }) => {
    const runtime = await getRuntime(indexURL, packages, onStatus);

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

    if (prelude) await runtime.runPythonAsync(prelude);
    return runtime.runPythonAsync(code);
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
  /** Cooperative wall-clock budget per run. */
  maxRunMs?: number;
  /** Run once on mount so the lane isn't inert on arrival. */
  autoRun?: boolean;
  /** Cap on retained log lines. */
  maxLogs?: number;
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
  /** False when the language's executor is a stub (Python, for now). */
  available: boolean;
}

function formatLogArg(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (value === null) return "null";
  if (value === undefined) return "undefined";
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function useCodeLane<TApi extends object>({
  initialCode,
  api,
  language = "javascript",
  executor,
  maxRunMs = 4000,
  autoRun = false,
  maxLogs = 100,
}: UseCodeLaneOptions<TApi>): UseCodeLaneApi {
  const [code, setCodeState] = useState(initialCode);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<readonly CodeRunLog[]>([]);

  /**
   * The snippet text, mirrored into a ref so `run()` is stable and always sees
   * the newest code — including when a component calls `setCode(...)` and
   * `run()` in the same handler, where reading state would be a render behind.
   * Written from event handlers only, never during render.
   */
  const codeRef = useRef(initialCode);

  const setCode = useCallback((next: string) => {
    codeRef.current = next;
    setCodeState(next);
  }, []);

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

  const run = useCallback(async () => {
    const activeExecutor =
      executorRef.current ??
      (language === "python"
        ? createPyodideExecutor<TApi>()
        : createJsExecutor<TApi>());

    const collected: CodeRunLog[] = [];
    const startedAt = Date.now();

    setRunning(true);
    setError(null);

    try {
      await activeExecutor(codeRef.current, {
        api: apiRef.current,
        log: (...args: unknown[]) => {
          if (collected.length < maxLogs) {
            collected.push({
              level: "log",
              message: args.map(formatLogArg).join(" "),
            });
          }
        },
        checkBudget: () => {
          if (Date.now() - startedAt > maxRunMs) {
            throw new CodeLaneTimeoutError(maxRunMs);
          }
        },
      });
    } catch (cause) {
      const message =
        cause instanceof Error
          ? `${cause.name}: ${cause.message}`
          : String(cause);
      collected.push({ level: "error", message });
      if (mountedRef.current) setError(message);
    } finally {
      if (mountedRef.current) {
        setLogs(collected.slice(-maxLogs));
        setRunning(false);
      }
    }
  }, [language, maxRunMs, maxLogs]);

  const reset = useCallback(() => {
    setCode(initialCode);
    setError(null);
    setLogs([]);
  }, [initialCode, setCode]);

  const clearLogs = useCallback(() => setLogs([]), []);

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
