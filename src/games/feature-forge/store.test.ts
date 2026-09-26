import * as tf from "@tensorflow/tfjs";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { useProgression } from "@/engine/progression";
import type * as ForgeMl from "./ml";

/**
 * The forge's async state machine: retries, failed fits, and the code lane's
 * entry points.
 *
 * `fitAndScore` is the real one — these tests train real models — wrapped so a
 * test can make the next fit throw, the way a lost WebGL context or an
 * out-of-memory phone does. Kept in its own file because that wrapper is a
 * module mock, and the ML measurements in feature-forge.test.ts must not run
 * through it.
 */
const control = vi.hoisted(() => ({ failNext: null as string | null }));

vi.mock("./ml", async (importOriginal) => {
  const actual = await importOriginal<typeof ForgeMl>();
  return {
    ...actual,
    fitAndScore: async (...args: Parameters<typeof actual.fitAndScore>) => {
      if (control.failNext !== null) {
        const message = control.failNext;
        control.failNext = null;
        throw new Error(message);
      }
      return actual.fitAndScore(...args);
    },
  };
});

import { LEGENDARY_COMBOS } from "./ml";
import { SLUG, useForgeStore } from "./store";
import { createCodeApi } from "./code-api";

const store = useForgeStore;
const recorded: Array<{ lane: string; codeLaneCleared: boolean }> = [];
const realRecord = useProgression.getState().recordResult;

beforeAll(async () => {
  await tf.ready();
});

beforeEach(async () => {
  control.failNext = null;
  recorded.length = 0;
  useProgression.setState({
    recordResult: vi.fn((result: Parameters<typeof realRecord>[0]) => {
      recorded.push({
        lane: result.lane,
        codeLaneCleared: result.codeLaneCleared === true,
      });
      return realRecord(result);
    }),
  });
  await store.getState().reset();
}, 120000);

afterEach(() => {
  useProgression.setState({ recordResult: realRecord });
});

describe("Retry during a retrain", () => {
  it("drops the stale fit instead of letting it land in the fresh run", async () => {
    // Used to end with a Leakage failure, a red metric and a disabled Submit on
    // an empty forge: the in-flight leaky fit resolved after the reset.
    const leaky = store.getState().addFeature("standardise", ["refund_issued"]);
    expect(store.getState().training).toBe(true);
    const resetting = store.getState().reset();

    expect(await leaky).toBe("stale");
    await resetting;

    const state = store.getState();
    expect(state.features).toEqual([]);
    expect(state.ready).toBe(true);
    expect(state.training).toBe(false);
    expect(state.phase).toBe("forging");
    expect(state.failure).toBeNull();
    expect(state.currentScore).toBe(state.baselineScore);
  }, 120000);
});

describe("a fit that throws", () => {
  it("recovers: training ends, the forge is as it was, and the card says why", async () => {
    const before = store.getState();
    control.failNext = "WebGL context lost";

    const outcome = await store.getState().addFeature("bin", ["age"]);

    const state = store.getState();
    expect(outcome).toBe("failed");
    expect(state.training).toBe(false);
    expect(state.features).toEqual(before.features);
    expect(state.currentScore).toBe(before.currentScore);
    expect(state.fitError).toBe("WebGL context lost");
    expect(state.whyCard?.body).toMatch(/Training failed: WebGL context lost/);

    // And the next forge works.
    expect(await store.getState().addFeature("bin", ["age"])).toBe("applied");
    expect(store.getState().fitError).toBeNull();
  }, 120000);

  it("keeps a named failure standing when the next fit fails", async () => {
    // Rolling back the features must roll back their verdict too: a leaky forge
    // whose next retrain failed is still a leaky forge.
    await store.getState().addFeature("standardise", ["refund_issued"]);
    expect(store.getState().failure?.name).toBe("Leakage");

    control.failNext = "WebGL context lost";
    expect(await store.getState().addFeature("bin", ["age"])).toBe("failed");
    expect(store.getState().failure?.name).toBe("Leakage");
    expect(store.getState().phase).toBe("ruined");
    expect(store.getState().features.map((feature) => feature.id)).toEqual([
      "standardise:refund_issued",
    ]);
  }, 120000);

  it("leaves a failed baseline retryable rather than spinning forever", async () => {
    control.failNext = "out of memory";
    await store.getState().reset();
    expect(store.getState().ready).toBe(false);
    expect(store.getState().training).toBe(false);
    expect(store.getState().fitError).toBe("out of memory");

    await store.getState().reset();
    expect(store.getState().ready).toBe(true);
  }, 120000);
});

describe("re-entrancy", () => {
  it("refuses a second forge while one is retraining, and says so", async () => {
    const first = store.getState().addFeature("bin", ["age"]);
    expect(await store.getState().addFeature("one_hot", ["city_code"])).toBe("busy");
    expect(await first).toBe("applied");
    expect(store.getState().features.map((feature) => feature.id)).toEqual([
      "bin:age",
    ]);
  }, 120000);

  it("records a winning submission once, however often it is repeated", async () => {
    for (const [transform, column] of [
      ["day_of_week", "signup_ts"],
      ["bin", "age"],
      ["one_hot", "city_code"],
    ] as const) {
      expect(await store.getState().addFeature(transform, [column])).toBe("applied");
    }
    expect(store.getState().submit().outcome).toBe("forged");
    expect(store.getState().submit().outcome).toBe("forged");
    expect(recorded).toHaveLength(1);
  }, 120000);
});

describe("the Python lane's api", () => {
  it("raises a named error for every forge it cannot do", async () => {
    const api = createCodeApi();
    await expect(api.forge_sync("bin", JSON.stringify(["city"]))).rejects.toThrow(
      /no column called "city"/,
    );
    await expect(api.forge_sync("bin", JSON.stringify(["city_code"]))).rejects.toThrow(
      /bin needs a numeric column/,
    );
    await expect(api.forge_sync("ratio", JSON.stringify(["income"]))).rejects.toThrow(
      /takes 2 columns/,
    );
    await expect(api.forge_sync("binn", JSON.stringify(["age"]))).rejects.toThrow(
      /unknown transform/,
    );
    await expect(api.forge_sync("bin", "not json")).rejects.toThrow(/column names/);
    expect(store.getState().features).toEqual([]);

    await api.forge_sync("bin", JSON.stringify(["age"]));
    await expect(api.forge_sync("bin", JSON.stringify(["age"]))).rejects.toThrow(
      /already in the forge/,
    );
  }, 120000);

  it("raises rather than reporting success for a forge that was dropped", async () => {
    const api = createCodeApi();
    const first = api.forge_sync("bin", JSON.stringify(["age"]));
    // Used to resolve with the unchanged score as if it had worked.
    await expect(
      api.forge_sync("one_hot", JSON.stringify(["city_code"])),
    ).rejects.toThrow(/still retraining/);
    await first;
  }, 120000);

  it("counts legendary features, not all of them", async () => {
    const api = createCodeApi();
    await api.forge_sync("standardise", JSON.stringify(["support_tickets"]));
    await api.forge_sync("bin", JSON.stringify(["age"]));
    expect(api.legendary_count()).toBe(1);
    expect(LEGENDARY_COMBOS.length).toBeGreaterThan(1);
  }, 120000);

  it("earns the code-lane star from api.submit(), not from the open tab", async () => {
    const api = createCodeApi();
    for (const [transform, column] of [
      ["day_of_week", "signup_ts"],
      ["bin", "age"],
      ["one_hot", "city_code"],
    ] as const) {
      await api.forge_sync(transform, JSON.stringify([column]));
    }

    // The rail's button with the code tab showing is still a visual-lane action.
    store.getState().setLane("code");
    expect(store.getState().submit().outcome).toBe("forged");
    expect(recorded).toEqual([{ lane: "visual", codeLaneCleared: false }]);

    await store.getState().reset();
    for (const [transform, column] of [
      ["day_of_week", "signup_ts"],
      ["bin", "age"],
      ["one_hot", "city_code"],
    ] as const) {
      await api.forge_sync(transform, JSON.stringify([column]));
    }
    expect(api.submit()).toBe("forged");
    expect(recorded[1]).toEqual({ lane: "code", codeLaneCleared: true });
  }, 240000);

  it("still earns the code-lane star after the same forge was scored from the rail", async () => {
    // The once-only guard used to return before recording whatever the source,
    // so api.submit() printed "forged" and silently awarded nothing.
    await useProgression.getState().clear();
    const cleared = () =>
      useProgression.getState().games[SLUG]?.codeLaneCleared ?? false;
    const api = createCodeApi();
    for (const [transform, column] of [
      ["day_of_week", "signup_ts"],
      ["bin", "age"],
      ["one_hot", "city_code"],
    ] as const) {
      await api.forge_sync(transform, JSON.stringify([column]));
    }
    expect(store.getState().submit().outcome).toBe("forged");
    expect(cleared()).toBe(false);
    expect(api.submit()).toBe("forged");
    expect(cleared()).toBe(true);
    expect(recorded).toEqual([
      { lane: "visual", codeLaneCleared: false },
      { lane: "code", codeLaneCleared: true },
    ]);

    // And each lane still records it only once.
    expect(api.submit()).toBe("forged");
    expect(store.getState().submit().outcome).toBe("forged");
    expect(recorded).toHaveLength(2);

    // A changed forge is a new result, for either lane.
    await api.remove_feature("one_hot:city_code");
    await api.forge_sync("one_hot", JSON.stringify(["city_code"]));
    expect(api.submit()).toBe("forged");
    expect(recorded).toHaveLength(3);
  }, 240000);

  it("will not submit a score that describes a different matrix", async () => {
    const api = createCodeApi();
    const pending = api.forge_sync("bin", JSON.stringify(["age"]));
    expect(() => api.submit()).toThrow(/still retraining/);
    await pending;
  }, 120000);

  it("names an unknown feature id when removing", async () => {
    const api = createCodeApi();
    await expect(api.remove_feature("bin:agee")).rejects.toThrow(/no forged feature/);
  });
});
