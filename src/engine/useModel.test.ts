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
    await act(async () => {
      const a = result.current.train({ xs: XS, ys: YS, epochs: 30, batchSize: 4 });
      const b = result.current.train({ xs: XS, ys: YS, epochs: 3, batchSize: 4 });
      await Promise.all([a, b]);
    });

    expect(onDone).toHaveBeenCalledTimes(2);
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current.error).toBeNull();

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
