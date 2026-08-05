import { describe, expect, it } from "vitest";
import { clamp, cx, formatPercent, seededRandom } from "./utils";

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
