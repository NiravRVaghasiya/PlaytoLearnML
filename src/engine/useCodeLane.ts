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

/**
 * Pyodide executor — the seam for Python code lanes (spec: Feature Forge, Phase
 * 3, runs real pandas). Deliberately not implemented yet: Phase 1 ships
 * JavaScript lanes, and loading a ~10 MB Python runtime for games that don't
 * need it would break the "fetches fast on a slow connection" constraint.
 *
 * Throws a clear, player-facing message rather than failing silently, so a game
 * that wires this up before it exists is obvious immediately.
 */
export function createPyodideExecutor<TApi extends object>(): CodeExecutor<TApi> {
  return async () => {
    throw new Error(
      "The Python (Pyodide) code lane isn't wired up yet — it lands with Feature Forge in Phase 3. Switch to the JavaScript lane for now.",
    );
  };
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
  /** Defaults to the JS executor, or the Pyodide stub when language is python. */
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

  const available = language === "javascript" || executor !== undefined;

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
