import { act, renderHook, waitFor } from "@testing-library/react";
import * as tf from "@tensorflow/tfjs";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { useModel, type EpochMetrics } from "./useModel";

/**
 * These tests exist because of one line in the `tfjs-model-lifecycle` skill:
 * "Check `tf.memory().numTensors` in tests to assert no leaks after a round."
 *
 * Games rebuild models constantly, so a per-round leak is the difference
 * between a lesson and a crashed tab. The leak assertions below are the reason
 * `useModel` disposes the optimizer as well as the model.
 */

/** XOR — small, real, and not linearly separable, so loss actually moves. */
const XS: number[][] = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];
const YS: number[] = [0, 1, 1, 0];

function buildXorModel(): tf.LayersModel {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({ units: 8, activation: "relu", inputShape: [2] }),
  );
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  model.compile({
    optimizer: tf.train.adam(0.05),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

let baseline = 0;

beforeAll(async () => {
  await tf.ready();
  // Warm the backend so its own allocations aren't counted as a leak.
  tf.tidy(() => tf.tensor2d(XS).square());
  baseline = tf.memory().numTensors;
});

describe("useModel", () => {
  it("starts idle with nothing allocated", () => {
    const { result, unmount } = renderHook(() =>
      useModel({ build: buildXorModel }),
    );

    expect(result.current.status).toBe("idle");
    expect(result.current.latest).toBeNull();
    expect(result.current.predict(XS)).toBeNull();

    unmount();
  });

  it("builds a model and reports ready", () => {
    const { result, unmount } = renderHook(() =>
      useModel({ build: buildXorModel }),
    );

    act(() => result.current.build());
    expect(result.current.status).toBe("ready");
    expect(result.current.error).toBeNull();

    unmount();
  });

  it("surfaces a factory failure instead of throwing", () => {
    const { result, unmount } = renderHook(() =>
      useModel({
        build: () => {
          throw new Error("bad architecture");
        },
      }),
    );

    act(() => result.current.build());
    expect(result.current.status).toBe("error");
    expect(result.current.error).toContain("bad architecture");

    unmount();
  });

  it("fires onEpoch with real logs every epoch", async () => {
    const seen: EpochMetrics[] = [];
    const { result, unmount } = renderHook(() =>
      useModel({
        build: buildXorModel,
        onEpoch: (metrics) => seen.push(metrics),
      }),
    );

    act(() => result.current.build());
    await act(async () => {
      await result.current.train({ xs: XS, ys: YS, epochs: 4, batchSize: 4 });
    });

    expect(seen).toHaveLength(4);
    expect(seen.map((m) => m.epoch)).toEqual([0, 1, 2, 3]);
    // Real numbers, not placeholders.
    for (const metrics of seen) {
      expect(Number.isFinite(metrics.loss)).toBe(true);
      expect(metrics.loss).toBeGreaterThan(0);
      expect(metrics.accuracy).not.toBeNull();
    }
    expect(result.current.status).toBe("ready");
    expect(result.current.epoch).toBe(4);
    expect(result.current.totalEpochs).toBe(4);

    unmount();
  });

  it("actually learns — loss falls over a longer run", async () => {
    const seen: EpochMetrics[] = [];
    const { result, unmount } = renderHook(() =>
      useModel({
        build: buildXorModel,
        onEpoch: (metrics) => seen.push(metrics),
      }),
    );

    act(() => result.current.build());
    await act(async () => {
      await result.current.train({
        xs: XS,
        ys: YS,
        epochs: 60,
        batchSize: 4,
        shuffle: false,
      });
    });

    const first = seen[0]!.loss;
    const last = seen[seen.length - 1]!.loss;
    expect(last).toBeLessThan(first);

    unmount();
  });

  it("reports validation metrics when a split is requested", async () => {
    // 8 rows so a 0.25 split leaves a usable validation set.
    const xs = [...XS, ...XS];
    const ys = [...YS, ...YS];
    const seen: EpochMetrics[] = [];

    const { result, unmount } = renderHook(() =>
      useModel({ build: buildXorModel, onEpoch: (m) => seen.push(m) }),
    );

    act(() => result.current.build());
    await act(async () => {
      await result.current.train({
        xs,
        ys,
        epochs: 2,
        batchSize: 2,
        validationSplit: 0.25,
      });
    });

    expect(seen.at(-1)?.valLoss).not.toBeNull();

    unmount();
  });

  it("rejects mismatched xs/ys lengths without touching the model", async () => {
    const { result, unmount } = renderHook(() =>
      useModel({ build: buildXorModel }),
    );

    act(() => result.current.build());
    await act(async () => {
      const outcome = await result.current.train({
        xs: XS,
        ys: [0, 1],
        epochs: 1,
      });
      expect(outcome).toBeNull();
    });

    expect(result.current.status).toBe("error");
    expect(result.current.error).toContain("4 rows");

    unmount();
  });

  it("predicts one value per input row", async () => {
    const { result, unmount } = renderHook(() =>
      useModel({ build: buildXorModel }),
    );

    act(() => result.current.build());
    await act(async () => {
      await result.current.train({ xs: XS, ys: YS, epochs: 2, batchSize: 4 });
    });

    const predictions = result.current.predict(XS);
    expect(predictions).not.toBeNull();
    expect(predictions).toHaveLength(4);
    for (const p of predictions!) {
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }

    unmount();
  });

  it("serialises overlapping trainings instead of racing them", async () => {
    const onDone = vi.fn();
    const { result, unmount } = renderHook(() =>
      useModel({ build: buildXorModel, onDone }),
    );

    act(() => result.current.build());

    // Fire a second train() while the first is still in flight. The hook stops
    // the first fit and awaits it, so both settle and neither corrupts the other.
    let outcomes: Array<EpochMetrics | null> = [];
    await act(async () => {
      const a = result.current.train({ xs: XS, ys: YS, epochs: 30, batchSize: 4 });
      const b = result.current.train({ xs: XS, ys: YS, epochs: 3, batchSize: 4 });
      outcomes = await Promise.all([a, b]);
    });

    // The stopped fit was taken over, so it reports "not trained" and no end;
    // only the fit that actually finished does.
    expect(outcomes[0]).toBeNull();
    expect(outcomes[1]?.epoch).toBe(2);
    expect(onDone).toHaveBeenCalledOnce();
    expect(onDone).toHaveBeenCalledWith(outcomes[1]);
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current.error).toBeNull();

    unmount();
  });

  it("serialises THREE overlapping trainings: the newest wins, nothing races", async () => {
    // Regression: two waiters both awaited the first fit, both woke when it
    // settled, and the second one into model.fit() failed with "Cannot start
    // training because another fit() call is ongoing" — leaving the hook stuck
    // in "error" and dropping the other fit's epochs as stale.
    const onDone = vi.fn();
    const { result, unmount } = renderHook(() =>
      useModel({ build: buildXorModel, onDone }),
    );

    act(() => result.current.build());

    let outcomes: Array<EpochMetrics | null> = [];
    await act(async () => {
      const a = result.current.train({ xs: XS, ys: YS, epochs: 30, batchSize: 4 });
      const b = result.current.train({ xs: XS, ys: YS, epochs: 3, batchSize: 4 });
      const c = result.current.train({ xs: XS, ys: YS, epochs: 3, batchSize: 4 });
      outcomes = await Promise.all([a, b, c]);
    });

    // A ran and was stopped, B was superseded before it started, C ran. Only
    // C finished, so only C resolves with metrics and fires onDone: A's partial
    // epochs describe weights that C has since trained on.
    expect(outcomes[0]).toBeNull();
    expect(outcomes[1]).toBeNull();
    expect(outcomes[2]).not.toBeNull();
    expect(outcomes[2]?.epoch).toBe(2);
    expect(onDone).toHaveBeenCalledOnce();
    expect(onDone).toHaveBeenCalledWith(outcomes[2]);

    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current.error).toBeNull();
    expect(result.current.totalEpochs).toBe(3);

    unmount();
  });

  it("survives build() while a fit is still running (fresh weights, then train)", async () => {
    // Three games do `model.build(); await model.train(…)` on every retrain. If
    // a retrain lands while the previous fit is running, build() used to
    // dispose the weights that fit was still reading.
    const before = tf.memory().numTensors;
    /** For each disposed model: was it still inside fit() when freed? */
    const disposedMidFit: boolean[] = [];
    /** How each model's own fit() settled: freed weights make it reject. */
    const fitSettled: string[] = [];
    const built: tf.LayersModel[] = [];
    // A Sequential delegates fit() to an inner LayersModel, which is where
    // tfjs-layers keeps its `isTraining` flag.
    type Trainable = { isTraining?: boolean; model?: { isTraining?: boolean } };
    const isTraining = (model: tf.LayersModel | undefined) => {
      const m = model as unknown as Trainable | undefined;
      return (m?.model?.isTraining ?? m?.isTraining) === true;
    };
    const buildWatched = () => {
      const model = buildXorModel();
      built.push(model);
      const dispose = model.dispose.bind(model);
      model.dispose = () => {
        disposedMidFit.push(isTraining(model));
        return dispose();
      };
      const fit = model.fit.bind(model);
      model.fit = (...args: Parameters<typeof fit>) =>
        fit(...args).then(
          (history) => {
            fitSettled.push("resolved");
            return history;
          },
          (cause: unknown) => {
            fitSettled.push("rejected");
            throw cause;
          },
        );
      return model;
    };
    const { result, unmount } = renderHook(() =>
      useModel({ build: buildWatched }),
    );

    act(() => result.current.build());

    let second: EpochMetrics | null = null;
    let firstOutcome: EpochMetrics | null = null;
    let midFit = false;
    // One act scope: React's async act drains state updates until the queue is
    // empty, so splitting this across two acts would let the first fit finish
    // before build() ever ran — and the test would prove nothing.
    await act(async () => {
      // Effectively unbounded: XOR is so small that thousands of epochs finish
      // in a few milliseconds. The rebuild below is what stops it.
      const first = result.current.train({
        xs: XS,
        ys: YS,
        epochs: 1_000_000,
        batchSize: 4,
      });
      // Wait until the first fit is genuinely inside model.fit().
      for (let i = 0; i < 500 && !isTraining(built[0]); i += 1) {
        await new Promise((resolve) => setTimeout(resolve, 1));
      }
      midFit = isTraining(built[0]);

      result.current.build();
      second = await result.current.train({ xs: XS, ys: YS, epochs: 3, batchSize: 4 });
      firstOutcome = await first;
    });

    // Guard: the scenario really was "rebuild while a fit is running".
    expect(midFit).toBe(true);

    // The old fit was STOPPED at an epoch boundary, rather than crashing on
    // weights freed underneath it: both fits resolved, neither rejected.
    expect(fitSettled).toEqual(["resolved", "resolved"]);
    // The rebuild took the old fit over, so it reports "not trained" instead of
    // metrics for weights that no longer exist.
    expect(firstOutcome).toBeNull();
    expect(second).not.toBeNull();
    expect(result.current.status).toBe("ready");
    expect(result.current.error).toBeNull();
    // The new model is real and usable.
    expect(result.current.predict(XS)).toHaveLength(4);

    act(() => result.current.reset());
    unmount();
    // Both models were freed, and neither while fit() was still reading it.
    expect(disposedMidFit).toEqual([false, false]);
    // The old model was freed once its fit stopped — not leaked, not double-freed.
    expect(tf.memory().numTensors).toBe(before);
  });

  it("rebuilds from fresh weights on request, after the queue clears", async () => {
    let builds = 0;
    const { result, unmount } = renderHook(() =>
      useModel({
        build: () => {
          builds += 1;
          return buildXorModel();
        },
      }),
    );

    act(() => result.current.build());
    await act(async () => {
      await result.current.train({ xs: XS, ys: YS, epochs: 2, batchSize: 4 });
    });
    expect(builds).toBe(1);

    await act(async () => {
      await result.current.train({
        xs: XS,
        ys: YS,
        epochs: 2,
        batchSize: 4,
        rebuild: true,
      });
    });
    expect(builds).toBe(2);
    expect(result.current.status).toBe("ready");

    unmount();
  });

  it("cancels a train() still queued when reset() lands", async () => {
    const onDone = vi.fn();
    const { result, unmount } = renderHook(() =>
      useModel({ build: buildXorModel, onDone }),
    );

    act(() => result.current.build());

    let queued: EpochMetrics | null | undefined;
    await act(async () => {
      const running = result.current.train({ xs: XS, ys: YS, epochs: 50, batchSize: 4 });
      const waiting = result.current.train({ xs: XS, ys: YS, epochs: 50, batchSize: 4 });
      result.current.reset();
      await running;
      queued = await waiting;
    });

    // The queued request must not resurrect a model the player just reset.
    expect(queued).toBeNull();
    expect(result.current.status).toBe("idle");
    expect(result.current.predict(XS)).toBeNull();

    unmount();
  });

  it("stops an in-flight fit early", async () => {
    const seen: EpochMetrics[] = [];
    const { result, unmount } = renderHook(() =>
      useModel({ build: buildXorModel, onEpoch: (m) => seen.push(m) }),
    );

    act(() => result.current.build());
    await act(async () => {
      const training = result.current.train({
        xs: XS,
        ys: YS,
        epochs: 500,
        batchSize: 4,
      });
      result.current.stop();
      await training;
    });

    // stopTraining lands at an epoch boundary, so a few epochs may complete —
    // the point is that it did not run all 500.
    expect(seen.length).toBeLessThan(500);

    unmount();
  });

  it("reports how far a stopped fit got, but nothing for one reset() took over", async () => {
    const onDone = vi.fn();
    // Counted from onEpoch, not React state: state only flushes when act ends.
    let epochs = 0;
    const { result, unmount } = renderHook(() =>
      useModel({ build: buildXorModel, onDone, onEpoch: () => (epochs += 1) }),
    );
    act(() => result.current.build());

    // Wait for a few real epochs, so "partial" is a real partial result.
    const midFit = async (fit: Promise<EpochMetrics | null>, action: () => void) => {
      const start = epochs;
      for (let i = 0; i < 500 && epochs - start < 3; i += 1) {
        await new Promise((resolve) => setTimeout(resolve, 1));
      }
      action();
      return fit;
    };

    // stop(): the player's own call, and nothing replaced the model. It is
    // partial (check the epoch count), and it is this fit's real end.
    let stopped: EpochMetrics | null = null;
    await act(async () => {
      stopped = await midFit(
        result.current.train({ xs: XS, ys: YS, epochs: 1_000_000, batchSize: 1 }),
        () => result.current.stop(),
      );
    });
    expect(stopped).not.toBeNull();
    expect(stopped!.epoch).toBeGreaterThanOrEqual(2);
    expect(stopped!.epoch).toBeLessThan(1_000_000 - 1);
    expect(onDone).toHaveBeenCalledOnce();
    expect(onDone).toHaveBeenCalledWith(stopped);

    // reset(): the model is gone, so its epochs describe nothing — and a game
    // listening to onDone must not hear about a fit the player threw away.
    let taken: EpochMetrics | null | undefined;
    await act(async () => {
      taken = await midFit(
        result.current.train({ xs: XS, ys: YS, epochs: 1_000_000, batchSize: 1 }),
        () => result.current.reset(),
      );
    });
    expect(taken).toBeNull();
    expect(onDone).toHaveBeenCalledOnce();
    expect(result.current.status).toBe("idle");

    unmount();
  });

  it("clears state on reset", async () => {
    const { result, unmount } = renderHook(() =>
      useModel({ build: buildXorModel }),
    );

    act(() => result.current.build());
    await act(async () => {
      await result.current.train({ xs: XS, ys: YS, epochs: 2, batchSize: 4 });
    });
    expect(result.current.latest).not.toBeNull();

    act(() => result.current.reset());

    expect(result.current.status).toBe("idle");
    expect(result.current.latest).toBeNull();
    expect(result.current.epoch).toBe(0);
    expect(result.current.predict(XS)).toBeNull();

    unmount();
  });

  // ── the leak assertions ────────────────────────────────────────────────

  it("leaks nothing across a full build → train → predict → reset round", async () => {
    const before = tf.memory().numTensors;

    const { result, unmount } = renderHook(() =>
      useModel({ build: buildXorModel }),
    );

    act(() => result.current.build());
    await act(async () => {
      await result.current.train({ xs: XS, ys: YS, epochs: 5, batchSize: 4 });
    });
    result.current.predict(XS);
    act(() => result.current.reset());
    unmount();

    expect(tf.memory().numTensors).toBe(before);
  });

  it("leaks nothing when unmounted mid-training", async () => {
    const before = tf.memory().numTensors;

    const { result, unmount } = renderHook(() =>
      useModel({ build: buildXorModel }),
    );

    act(() => result.current.build());

    let training: Promise<unknown> = Promise.resolve();
    await act(async () => {
      training = result.current.train({
        xs: XS,
        ys: YS,
        epochs: 200,
        batchSize: 4,
      });
      // Give fit a tick to allocate before pulling the rug out.
      await Promise.resolve();
    });

    unmount();
    await training;

    expect(tf.memory().numTensors).toBe(before);
  });

  it("leaks nothing across repeated rebuilds", async () => {
    const before = tf.memory().numTensors;

    const { result, unmount } = renderHook(() =>
      useModel({ build: buildXorModel }),
    );

    // A player mashing "retry": each build must free the previous model.
    for (let i = 0; i < 5; i++) {
      act(() => result.current.build());
      await act(async () => {
        await result.current.train({ xs: XS, ys: YS, epochs: 2, batchSize: 4 });
      });
    }

    act(() => result.current.reset());
    unmount();

    expect(tf.memory().numTensors).toBe(before);
  });

  it("keeps the baseline stable for the whole suite", () => {
    // If this fails, an earlier test leaked and the per-test assertions above
    // were measuring a moving target.
    expect(tf.memory().numTensors).toBe(baseline);
  });
});
