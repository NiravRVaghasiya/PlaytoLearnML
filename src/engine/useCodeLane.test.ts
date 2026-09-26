import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  CodeLaneTimeoutError,
  createJsExecutor,
  createPyodideExecutor,
  formatLogArg,
  useCodeLane,
  type CodeExecutor,
  type PyodideModule,
  type PyodideRuntime,
} from "./useCodeLane";

/**
 * The point of these tests is the two-lane rule: a code-lane snippet must move
 * the SAME state the visual lane moves. The `api` object below stands in for a
 * game's Zustand store, and the assertions check that running code actually
 * mutates it.
 */
function makeFakeStore() {
  const state = { k: 1, steps: 0 };
  return {
    state,
    api: {
      setK: (k: number) => {
        state.k = k;
      },
      step: () => {
        state.steps += 1;
        return state.steps;
      },
      getK: () => state.k,
    },
  };
}

describe("useCodeLane", () => {
  it("starts with the initial snippet and is not dirty", () => {
    const { api } = makeFakeStore();
    const { result } = renderHook(() =>
      useCodeLane({ initialCode: "log('hi')", api }),
    );

    expect(result.current.code).toBe("log('hi')");
    expect(result.current.dirty).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.logs).toEqual([]);
    expect(result.current.available).toBe(true);
  });

  it("tracks edits and restores on reset", () => {
    const { api } = makeFakeStore();
    const { result } = renderHook(() =>
      useCodeLane({ initialCode: "api.step()", api }),
    );

    act(() => result.current.setCode("api.setK(5)"));
    expect(result.current.dirty).toBe(true);

    act(() => result.current.reset());
    expect(result.current.code).toBe("api.step()");
    expect(result.current.dirty).toBe(false);
  });

  it("drives the shared store — the whole reason the lane exists", async () => {
    const { state, api } = makeFakeStore();
    const { result } = renderHook(() =>
      useCodeLane({ initialCode: "api.setK(4); api.step(); api.step();", api }),
    );

    await act(async () => {
      await result.current.run();
    });

    expect(state.k).toBe(4);
    expect(state.steps).toBe(2);
    expect(result.current.error).toBeNull();
  });

  it("supports await in the snippet body", async () => {
    const { state, api } = makeFakeStore();
    const { result } = renderHook(() =>
      useCodeLane({
        initialCode: "await Promise.resolve(); api.setK(7);",
        api,
      }),
    );

    await act(async () => {
      await result.current.run();
    });

    expect(state.k).toBe(7);
  });

  it("captures log output", async () => {
    const { api } = makeFakeStore();
    const { result } = renderHook(() =>
      useCodeLane({
        initialCode: "log('k is', api.getK()); log({ a: 1 });",
        api,
      }),
    );

    await act(async () => {
      await result.current.run();
    });

    expect(result.current.logs.map((l) => l.message)).toEqual([
      "k is 1",
      '{"a":1}',
    ]);
  });

  it("shows runtime errors instead of throwing them", async () => {
    const { api } = makeFakeStore();
    const { result } = renderHook(() =>
      useCodeLane({ initialCode: "throw new Error('boom')", api }),
    );

    await act(async () => {
      // Must resolve, not reject — a broken snippet is a teaching moment, not
      // an unhandled rejection.
      await expect(result.current.run()).resolves.toBeUndefined();
    });

    expect(result.current.error).toContain("boom");
    expect(result.current.logs.at(-1)?.level).toBe("error");
    expect(result.current.running).toBe(false);
  });

  it("reports syntax errors from the snippet", async () => {
    const { api } = makeFakeStore();
    const { result } = renderHook(() =>
      useCodeLane({ initialCode: "this is not javascript(((", api }),
    );

    await act(async () => {
      await result.current.run();
    });

    expect(result.current.error).toBeTruthy();
  });

  it("clears the previous error on a successful rerun", async () => {
    const { api } = makeFakeStore();
    const { result } = renderHook(() =>
      useCodeLane({ initialCode: "throw new Error('first')", api }),
    );

    await act(async () => {
      await result.current.run();
    });
    expect(result.current.error).toContain("first");

    act(() => result.current.setCode("api.step()"));
    await act(async () => {
      await result.current.run();
    });
    expect(result.current.error).toBeNull();
  });

  it("stops a runaway loop that cooperates via checkBudget", async () => {
    const { api } = makeFakeStore();
    const { result } = renderHook(() =>
      useCodeLane({
        initialCode: "while (true) { api.step(); checkBudget(); }",
        api,
        maxRunMs: 40,
      }),
    );

    await act(async () => {
      await result.current.run();
    });

    expect(result.current.error).toContain("longer than 40ms");
  });

  it("reports the Python lane as available now that Pyodide is real", () => {
    // This test previously asserted the opposite: that Python was a stub and said
    // so. It landed with Feature Forge in Phase 3, so the assertion flipped rather
    // than being deleted — the change in behaviour is the thing worth pinning.
    const { api } = makeFakeStore();
    const { result } = renderHook(() =>
      useCodeLane({ initialCode: "print('hi')", api, language: "python" }),
    );

    expect(result.current.available).toBe(true);
    expect(result.current.language).toBe("python");
  });

  it("accepts a custom executor", async () => {
    const { api } = makeFakeStore();
    const executor: CodeExecutor<typeof api> = vi.fn(async () => undefined);

    const { result } = renderHook(() =>
      useCodeLane({ initialCode: "whatever", api, executor }),
    );

    await act(async () => {
      await result.current.run();
    });

    expect(executor).toHaveBeenCalledOnce();
    expect(vi.mocked(executor).mock.calls[0]?.[0]).toBe("whatever");
  });

  it("runs once on mount when autoRun is set", async () => {
    const { state, api } = makeFakeStore();
    renderHook(() =>
      useCodeLane({ initialCode: "api.step()", api, autoRun: true }),
    );

    await waitFor(() => expect(state.steps).toBe(1));
  });

  it("says a timeout in the player's words, without the class name", async () => {
    const { api } = makeFakeStore();
    const { result } = renderHook(() =>
      useCodeLane({ initialCode: "for (;;) checkBudget();", api, maxRunMs: 20 }),
    );
    await act(async () => {
      await result.current.run();
    });
    expect(result.current.error).toMatch(/^Your code ran longer than 20ms/);
  });

  it("keeps the player's own error names, which are the clue", async () => {
    const { api } = makeFakeStore();
    const { result } = renderHook(() =>
      useCodeLane({ initialCode: "api.nope()", api }),
    );
    await act(async () => {
      await result.current.run();
    });
    expect(result.current.error).toMatch(/^TypeError: /);
  });

  it("clears logs on request", async () => {
    const { api } = makeFakeStore();
    const { result } = renderHook(() =>
      useCodeLane({ initialCode: "log('noise')", api }),
    );

    await act(async () => {
      await result.current.run();
    });
    expect(result.current.logs).toHaveLength(1);

    act(() => result.current.clearLogs());
    expect(result.current.logs).toHaveLength(0);
  });
});

describe("useCodeLane — drafts survive the lane unmounting", () => {
  // GameShell mounts only the active lane, so every switch to the visual lane
  // unmounts this hook. The player's edits must be there when they come back.
  it("restores an edit after an unmount and remount", () => {
    const { api } = makeFakeStore();
    const options = { initialCode: "api.step()", api };
    const first = renderHook(() => useCodeLane(options));
    act(() => first.result.current.setCode("api.setK(9)"));
    first.unmount();

    const second = renderHook(() => useCodeLane(options));
    expect(second.result.current.code).toBe("api.setK(9)");
    expect(second.result.current.dirty).toBe(true);
  });

  it("runs the restored draft, not the starter", async () => {
    const { state, api } = makeFakeStore();
    const options = { initialCode: "api.step()", api };
    const first = renderHook(() => useCodeLane(options));
    act(() => first.result.current.setCode("api.setK(9)"));
    first.unmount();

    const second = renderHook(() => useCodeLane(options));
    await act(async () => {
      await second.result.current.run();
    });
    expect(state.k).toBe(9);
    expect(state.steps).toBe(0);
  });

  it("restores the last output and error with it", async () => {
    const { api } = makeFakeStore();
    const options = { initialCode: "log('before'); throw new Error('boom')", api };
    const first = renderHook(() => useCodeLane(options));
    await act(async () => {
      await first.result.current.run();
    });
    first.unmount();

    const second = renderHook(() => useCodeLane(options));
    expect(second.result.current.logs.map((l) => l.message)).toEqual([
      "before",
      "Error: boom",
    ]);
    expect(second.result.current.error).toBe("Error: boom");
  });

  it("keeps the result of a run that finished while the lane was away", async () => {
    const { api } = makeFakeStore();
    let finish!: () => void;
    const executor: CodeExecutor<typeof api> = async (_code, { log }) => {
      await new Promise<void>((resolve) => {
        finish = resolve;
      });
      log("done");
    };
    const options = { initialCode: "slow()", api, executor };
    const first = renderHook(() => useCodeLane(options));
    let running!: Promise<void>;
    act(() => {
      running = first.result.current.run();
    });
    await waitFor(() => expect(finish).toBeTypeOf("function"));
    first.unmount();
    await act(async () => {
      finish();
      await running;
    });

    const second = renderHook(() => useCodeLane(options));
    expect(second.result.current.logs.map((l) => l.message)).toEqual(["done"]);
  });

  it("restores the starter on reset, and forgets the draft", () => {
    const { api } = makeFakeStore();
    const options = { initialCode: "api.step()", api };
    const first = renderHook(() => useCodeLane(options));
    act(() => first.result.current.setCode("api.setK(9)"));
    act(() => first.result.current.reset());
    expect(first.result.current.code).toBe("api.step()");
    first.unmount();

    const second = renderHook(() => useCodeLane(options));
    expect(second.result.current.code).toBe("api.step()");
    expect(second.result.current.dirty).toBe(false);
    expect(second.result.current.logs).toEqual([]);
  });

  it("keeps each game's draft to itself", () => {
    const { api } = makeFakeStore();
    const a = renderHook(() => useCodeLane({ initialCode: "// game A", api }));
    act(() => a.result.current.setCode("// A, edited"));
    a.unmount();

    const b = renderHook(() => useCodeLane({ initialCode: "// game B", api }));
    expect(b.result.current.code).toBe("// game B");
    const again = renderHook(() => useCodeLane({ initialCode: "// game A", api }));
    expect(again.result.current.code).toBe("// A, edited");
  });

  it("can opt out with persistKey: null", () => {
    const { api } = makeFakeStore();
    const options = { initialCode: "api.step()", api, persistKey: null };
    const first = renderHook(() => useCodeLane(options));
    act(() => first.result.current.setCode("api.setK(9)"));
    first.unmount();

    const second = renderHook(() => useCodeLane(options));
    expect(second.result.current.code).toBe("api.step()");
  });
});

describe("executors", () => {
  it("createJsExecutor passes api, log and checkBudget through", async () => {
    const executor = createJsExecutor<{ ping: () => string }>();
    const messages: string[] = [];

    await executor("log(api.ping()); checkBudget();", {
      api: { ping: () => "pong" },
      log: (...args) => messages.push(String(args[0])),
      checkBudget: () => {},
    });

    expect(messages).toEqual(["pong"]);
  });

  it("createPyodideExecutor loads the runtime from the configured indexURL", async () => {
    // Under vitest there is no server serving /public, so the load fails — and the
    // failure is the evidence: it names the path it tried, which proves it is
    // actually fetching a runtime rather than throwing a canned stub message.
    // Browser-side behaviour is covered by the Feature Forge playthrough.
    const executor = createPyodideExecutor<object>({
      indexURL: "/pyodide-test-path/",
    });

    await expect(
      executor("print(1)", {
        api: {},
        log: () => {},
        checkBudget: () => {},
      }),
    ).rejects.toThrow(/pyodide-test-path/);
  });

  it("does not cache a failed runtime load, so a retry can succeed", async () => {
    const executor = createPyodideExecutor<object>({
      indexURL: "/pyodide-retry-path/",
    });
    const attempt = () =>
      executor("print(1)", { api: {}, log: () => {}, checkBudget: () => {} });

    await expect(attempt()).rejects.toThrow();
    // A cached rejected promise would make every later attempt fail with the same
    // stale error even after the cause was fixed.
    await expect(attempt()).rejects.toThrow(/pyodide-retry-path/);
  });
});

describe("CodeLaneTimeoutError", () => {
  it("names the budget it exceeded", () => {
    const error = new CodeLaneTimeoutError(250);
    expect(error.name).toBe("CodeLaneTimeoutError");
    expect(error.message).toContain("250ms");
  });
});

describe("useCodeLane — responsiveness and re-entrancy", () => {
  it("commits the running state BEFORE the snippet executes", async () => {
    // Regression: a synchronous snippet used to run inside the click handler,
    // so React never committed `running` and "Running…" never painted while
    // the tab froze (Agent Academy's and the Heist's starters take seconds).
    const { api } = makeFakeStore();
    let runningWhenExecuted: boolean | null = null;
    const { result } = renderHook(() =>
      useCodeLane({
        initialCode: "whatever",
        api,
        executor: async () => {
          runningWhenExecuted = result.current.running;
        },
      }),
    );

    let pending!: Promise<void>;
    act(() => {
      pending = result.current.run();
    });

    // The synchronous part of run() must not reach the executor...
    expect(runningWhenExecuted).toBeNull();
    // ...but it must already have rendered the running state.
    expect(result.current.running).toBe(true);

    await act(async () => {
      await pending;
    });
    expect(runningWhenExecuted).toBe(true);
    expect(result.current.running).toBe(false);
  });

  it("ignores a second run while one is still in flight", async () => {
    const { state, api } = makeFakeStore();
    const { result } = renderHook(() =>
      useCodeLane({ initialCode: "api.step()", api }),
    );

    await act(async () => {
      // A double click lands both before the button can disable.
      await Promise.all([result.current.run(), result.current.run()]);
    });

    expect(state.steps).toBe(1);

    // And the lane is usable again afterwards.
    await act(async () => {
      await result.current.run();
    });
    expect(state.steps).toBe(2);
  });

  it("enforces the budget on api calls when budgetApiCalls is set", async () => {
    // Without it, a loop that only ever awaits the api (and never calls
    // checkBudget itself) can't be stopped: maxRunMs would be a no-op.
    const api = {
      tick: async () => {
        await new Promise((resolve) => setTimeout(resolve, 5));
      },
    };
    const { result } = renderHook(() =>
      useCodeLane({
        initialCode: "for (;;) { await api.tick(); }",
        api,
        maxRunMs: 40,
        budgetApiCalls: true,
      }),
    );

    await act(async () => {
      await result.current.run();
    });

    expect(result.current.error).toContain("longer than 40ms");
  });

  it("leaves the api untouched when budgetApiCalls is off", async () => {
    const api = { same: () => api };
    const { result } = renderHook(() =>
      useCodeLane({
        initialCode: "log(String(api.same() === api))",
        api,
      }),
    );

    await act(async () => {
      await result.current.run();
    });
    expect(result.current.logs.map((l) => l.message)).toEqual(["true"]);
  });
});

describe("formatLogArg", () => {
  it("renders errors by name and message, not as {}", () => {
    expect(formatLogArg(new Error("boom"))).toBe("Error: boom");
    expect(formatLogArg(new TypeError("bad"))).toBe("TypeError: bad");
    expect(formatLogArg({ cause: new RangeError("far") })).toBe(
      '{"cause":"RangeError: far"}',
    );
  });

  it("renders functions and symbols instead of empty lines", () => {
    function namedHelper() {}
    expect(formatLogArg(namedHelper)).toBe("[function namedHelper]");
    expect(formatLogArg(Symbol("tag"))).toBe("Symbol(tag)");
    expect(formatLogArg(10n)).toBe("10n");
  });

  it("renders collections by their contents", () => {
    expect(formatLogArg(new Map([["a", 1]]))).toBe('Map {"a":1}');
    expect(formatLogArg(new Set([1, 2]))).toBe("Set [1,2]");
    expect(formatLogArg(new Float32Array([0.5, 1]))).toBe("[0.5,1]");
  });

  it("still survives circular structures", () => {
    const loop: Record<string, unknown> = {};
    loop.self = loop;
    expect(formatLogArg(loop)).toBe("[object Object]");
  });

  it("renders collections by their contents at any depth, not as {}", () => {
    expect(formatLogArg({ m: new Map([["a", 1]]), s: new Set(["x"]) })).toBe(
      '{"m":{"a":1},"s":["x"]}',
    );
    expect(formatLogArg({ w: new Float32Array([0.5]) })).toBe('{"w":[0.5]}');
    // An object key would flatten to "[object Object]"; entries keep it.
    expect(formatLogArg(new Map([[{ a: 1 }, 2]]))).toBe('Map [[{"a":1},2]]');
  });

  it("renders a BigInt typed array instead of making log() itself throw", () => {
    expect(() => formatLogArg(new BigInt64Array([1n, 2n]))).not.toThrow();
    expect(formatLogArg(new BigInt64Array([1n, 2n]))).toBe('["1n","2n"]');
  });
});

// ── Pyodide failure paths, against a fake runtime ─────────────────────────

let urlCounter = 0;
/** A fresh index URL per test, so the module-level runtime cache can't leak between them. */
const uniqueIndexURL = () => `/fake-pyodide-${++urlCounter}/`;

interface FakeRuntimeOptions {
  /** Whether loadPackage actually installs what it was asked for. */
  deliver?: () => boolean;
  run?: (code: string, filename: string | undefined) => Promise<unknown>;
}

function fakeRuntime({ deliver = () => true, run }: FakeRuntimeOptions = {}) {
  const loadedPackages: Record<string, string> = {};
  const runtime: PyodideRuntime & {
    loadPackage: ReturnType<typeof vi.fn>;
    runPythonAsync: ReturnType<typeof vi.fn>;
  } = {
    version: "3.14.0",
    loadedPackages,
    // Pyodide's real behaviour: a failed wheel is logged and the promise still
    // RESOLVES, with only what succeeded.
    loadPackage: vi.fn(async (names: string | string[]) => {
      if (!deliver()) return [];
      for (const name of [names].flat()) loadedPackages[name] = "default channel";
      return [];
    }),
    runPythonAsync: vi.fn(async (code: string, options?: { filename?: string }) =>
      run ? run(code, options?.filename) : undefined,
    ),
    setStdout: () => {},
    setStderr: () => {},
    globals: { set: () => {} },
  };
  return runtime;
}

const context = () => ({ api: {}, log: vi.fn(), checkBudget: () => {} });

describe("createPyodideExecutor failure recovery", () => {
  it("gives Pyodide's real reason for a failed wheel, not its lead-in line", async () => {
    const runtime = fakeRuntime();
    // Pyodide's own order: a colon-terminated header, then the actual error,
    // and the promise still resolves.
    runtime.loadPackage.mockImplementation(
      async (_names: unknown, options?: { errorCallback?: (message: string) => void }) => {
        options?.errorCallback?.("The following error occurred while loading numpy:");
        options?.errorCallback?.("Failed to load '/pyodide/numpy.whl': request failed.");
        return [];
      },
    );
    const executor = createPyodideExecutor<object>({
      indexURL: uniqueIndexURL(),
      packages: ["pandas"],
      loadModule: async () => ({ loadPyodide: async () => runtime }),
    });

    const { result } = renderHook(() =>
      useCodeLane({ initialCode: "print(1)", api: {}, language: "python", executor }),
    );
    await act(async () => {
      await result.current.run();
    });

    // Written for the player: the reason that matters, and no class name.
    expect(result.current.error).toBe(
      "Couldn't download pandas (Failed to load '/pyodide/numpy.whl': request failed). Check your connection and press Run again.",
    );
  });

  it("rejects when a package silently failed to install, and a retry recovers it", async () => {
    let online = false;
    const runtime = fakeRuntime({ deliver: () => online });
    const loadPyodide = vi.fn(async () => runtime);
    const loadModule = vi.fn(async (): Promise<PyodideModule> => ({ loadPyodide }));

    const executor = createPyodideExecutor<object>({
      indexURL: uniqueIndexURL(),
      packages: ["pandas"],
      loadModule,
    });

    // loadPackage resolved, but pandas isn't there: that must NOT read as ready.
    await expect(executor("print(1)", context())).rejects.toThrow(
      /Couldn't download pandas.*press Run again/,
    );
    expect(runtime.runPythonAsync).not.toHaveBeenCalled();

    online = true;
    await expect(executor("print(1)", context())).resolves.toBeUndefined();

    // The retry fetched only the missing wheel onto the runtime that already
    // worked, instead of downloading the 13 MB core a second time.
    expect(loadPyodide).toHaveBeenCalledOnce();
    expect(runtime.loadPackage).toHaveBeenCalledTimes(2);
    expect(runtime.runPythonAsync).toHaveBeenCalledOnce();
  });

  it("matches package names the way Pyodide's lockfile spells them", async () => {
    const runtime = fakeRuntime();
    // The lockfile says python-dateutil; a caller may write python_dateutil.
    runtime.loadPackage.mockImplementation(async () => {
      runtime.loadedPackages!["python-dateutil"] = "default channel";
      return [];
    });
    const executor = createPyodideExecutor<object>({
      indexURL: uniqueIndexURL(),
      packages: ["python_dateutil"],
      loadModule: async () => ({ loadPyodide: async () => runtime }),
    });

    await expect(executor("print(1)", context())).resolves.toBeUndefined();
  });

  it("gives up on a download that never settles, then lets the next run start fresh", async () => {
    let hang = true;
    const runtime = fakeRuntime();
    const loadPyodide = vi.fn(() =>
      hang ? new Promise<never>(() => {}) : Promise.resolve(runtime),
    );
    const executor = createPyodideExecutor<object>({
      indexURL: uniqueIndexURL(),
      loadTimeoutMs: 30,
      loadModule: async () => ({ loadPyodide }),
    });

    // A failed wasm fetch leaves loadPyodide pending forever. It used to spin
    // "Downloading…" for good, with the hung promise cached for the session.
    await expect(executor("print(1)", context())).rejects.toThrow(
      /took longer than.*press Run again/,
    );

    hang = false;
    await expect(executor("print(1)", context())).resolves.toBeUndefined();
    expect(loadPyodide).toHaveBeenCalledTimes(2);
  });

  it("does not cache a runtime whose module failed to load", async () => {
    let attempts = 0;
    const runtime = fakeRuntime();
    const executor = createPyodideExecutor<object>({
      indexURL: uniqueIndexURL(),
      loadModule: async () => {
        attempts += 1;
        if (attempts === 1) throw new Error("network down");
        return { loadPyodide: async () => runtime };
      },
    });

    await expect(executor("print(1)", context())).rejects.toThrow(/network down/);
    await expect(executor("print(1)", context())).resolves.toBeUndefined();
  });

  it("announces the first-import pause before it happens, and 'ready' after", async () => {
    const statuses: string[] = [];
    const runtime = fakeRuntime({
      run: async (_code, filename) => {
        statuses.push(`ran ${filename}`);
      },
    });
    const executor = createPyodideExecutor<object>({
      indexURL: uniqueIndexURL(),
      packages: ["pandas"],
      prelude: "import pandas as pd",
      onStatus: (status) => statuses.push(status),
      loadModule: async () => ({ loadPyodide: async () => runtime }),
    });

    await executor("print(1)", context());

    const starting = statuses.findIndex((s) => /Starting pandas.*pause/.test(s));
    const prelude = statuses.indexOf("ran <lane setup>");
    const ready = statuses.findIndex((s) => /Python 3\.14\.0 ready/.test(s));
    expect(starting).toBeGreaterThanOrEqual(0);
    expect(prelude).toBeGreaterThan(starting);
    expect(ready).toBeGreaterThan(prelude);
    expect(statuses.at(-1)).toBe("ran <your code>");

    // Only the first run pays (and warns about) the import.
    statuses.length = 0;
    await executor("print(2)", context());
    expect(statuses.some((s) => /Starting/.test(s))).toBe(false);
  });
});

describe("Python errors in the lane", () => {
  const TRACEBACK = [
    "Traceback (most recent call last):",
    '  File "/lib/python314.zip/_pyodide/_base.py", line 597, in eval_code_async',
    "    await CodeRunner(",
    '  File "<lane setup>", line 9, in helper',
    '  File "<your code>", line 2, in <module>',
    "NameError: name 'undefined_name' is not defined",
  ].join("\n");

  it("shows the exception and the player's line, not the wasm traceback", async () => {
    const runtime = fakeRuntime({
      run: async (_code, filename) => {
        if (filename === "<your code>") {
          const error = new Error(TRACEBACK);
          error.name = "PythonError";
          throw error;
        }
      },
    });
    const executor = createPyodideExecutor<object>({
      indexURL: uniqueIndexURL(),
      prelude: "def helper(): pass",
      loadModule: async () => ({ loadPyodide: async () => runtime }),
    });

    const { result } = renderHook(() =>
      useCodeLane({
        initialCode: "x = 1\nundefined_name",
        api: {},
        language: "python",
        executor,
      }),
    );

    await act(async () => {
      await result.current.run();
    });

    expect(result.current.error).toBe(
      "NameError: name 'undefined_name' is not defined (line 2)",
    );
    // The full traceback is still there for anyone who wants it.
    expect(result.current.logs.some((l) => l.message.includes("eval_code_async"))).toBe(
      true,
    );
  });
});
