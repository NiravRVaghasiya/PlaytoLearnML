import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  CodeLaneTimeoutError,
  createJsExecutor,
  createPyodideExecutor,
  useCodeLane,
  type CodeExecutor,
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

  it("reports the Python lane as unavailable with a useful message", async () => {
    const { api } = makeFakeStore();
    const { result } = renderHook(() =>
      useCodeLane({ initialCode: "print('hi')", api, language: "python" }),
    );

    expect(result.current.available).toBe(false);

    await act(async () => {
      await result.current.run();
    });

    expect(result.current.error).toContain("Pyodide");
    expect(result.current.error).toContain("Phase 3");
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

  it("createPyodideExecutor rejects with an explanation", async () => {
    const executor = createPyodideExecutor<object>();
    await expect(
      executor("print(1)", {
        api: {},
        log: () => {},
        checkBudget: () => {},
      }),
    ).rejects.toThrow(/Pyodide/);
  });
});

describe("CodeLaneTimeoutError", () => {
  it("names the budget it exceeded", () => {
    const error = new CodeLaneTimeoutError(250);
    expect(error.name).toBe("CodeLaneTimeoutError");
    expect(error.message).toContain("250ms");
  });
});
