import { afterEach, describe, expect, it, vi } from "vitest";
import {
  clamp,
  cx,
  formatPercent,
  isWebGLAvailable,
  resetWebGLProbeForTests,
  seededRandom,
  yieldToPaint,
} from "./utils";

describe("cx", () => {
  it("drops falsy parts", () => {
    expect(cx("a", false, null, undefined, "b")).toBe("a b");
  });
});

describe("clamp", () => {
  it("bounds the value", () => {
    expect(clamp(5, 0, 3)).toBe(3);
    expect(clamp(-1, 0, 3)).toBe(0);
    expect(clamp(2, 0, 3)).toBe(2);
  });
});

describe("formatPercent", () => {
  it("renders whole percents by default", () => {
    expect(formatPercent(0.8421)).toBe("84%");
    expect(formatPercent(0.8421, 1)).toBe("84.2%");
  });
});

describe("seededRandom", () => {
  it("is deterministic for a given seed", () => {
    const a = seededRandom(42);
    const b = seededRandom(42);
    const seqA = [a(), a(), a()];
    const seqB = [b(), b(), b()];
    expect(seqA).toEqual(seqB);
  });

  it("stays in [0, 1)", () => {
    const r = seededRandom(7);
    for (let i = 0; i < 200; i++) {
      const v = r();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it("differs across seeds", () => {
    expect(seededRandom(1)()).not.toBe(seededRandom(2)());
  });
});

describe("yieldToPaint", () => {
  it("does not resolve synchronously or in the same microtask burst", async () => {
    // The whole point: "Running…" set before this must get a paint before the
    // heavy job after it starts. A helper that resolved on a microtask would
    // run the job before the browser ever rendered.
    let resolved = false;
    const pending = yieldToPaint().then(() => {
      resolved = true;
    });
    await Promise.resolve();
    await Promise.resolve();
    expect(resolved).toBe(false);
    await pending;
    expect(resolved).toBe(true);
  });

  it("still resolves when requestAnimationFrame never fires (hidden tab)", async () => {
    vi.useFakeTimers();
    const raf = vi
      .spyOn(globalThis, "requestAnimationFrame")
      .mockImplementation(() => 0);
    try {
      let resolved = false;
      const pending = yieldToPaint(50).then(() => {
        resolved = true;
      });
      await vi.advanceTimersByTimeAsync(49);
      expect(resolved).toBe(false);
      await vi.advanceTimersByTimeAsync(1);
      await pending;
      expect(resolved).toBe(true);
    } finally {
      raf.mockRestore();
      vi.useRealTimers();
    }
  });
});

describe("isWebGLAvailable", () => {
  afterEach(() => {
    resetWebGLProbeForTests();
    vi.restoreAllMocks();
  });

  it("releases the probe's GL context instead of leaving it for GC", () => {
    // Browsers cap live contexts and force-lose the oldest one. A probe that
    // keeps its context alive can evict TF.js's WebGL backend mid-lesson.
    const loseContext = vi.fn();
    const fakeGl = {
      getExtension: (name: string) =>
        name === "WEBGL_lose_context" ? { loseContext } : null,
    };
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(
      ((kind: string) =>
        kind === "webgl2" ? fakeGl : null) as unknown as HTMLCanvasElement["getContext"],
    );

    expect(isWebGLAvailable()).toBe(true);
    expect(loseContext).toHaveBeenCalledOnce();
  });

  it("probes once per page, then answers from the cache", () => {
    const getContext = vi
      .spyOn(HTMLCanvasElement.prototype, "getContext")
      .mockImplementation((() => null) as unknown as HTMLCanvasElement["getContext"]);

    expect(isWebGLAvailable()).toBe(false);
    const probesAfterFirst = getContext.mock.calls.length;
    expect(isWebGLAvailable()).toBe(false);
    expect(isWebGLAvailable()).toBe(false);
    expect(getContext.mock.calls.length).toBe(probesAfterFirst);
  });

  it("treats a throwing getContext as no WebGL", () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => {
      throw new Error("blocked");
    });
    expect(isWebGLAvailable()).toBe(false);
  });
});
