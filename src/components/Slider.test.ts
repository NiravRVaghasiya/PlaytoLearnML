import { describe, expect, it } from "vitest";
import { fromPosition, toPosition } from "./Slider";

/**
 * The log-scale maths matters more than it looks. Learning rate is the knob the
 * Gradient Descent Skier is built around, and it spans 0.001 → 2. On a linear
 * track, 95% of the useful range would sit in the first 2% of the slider, which
 * would quietly destroy the lesson the game exists to teach.
 */
describe("linear scale", () => {
  it("maps endpoints and midpoint", () => {
    expect(fromPosition(0, 0, 10, "linear")).toBe(0);
    expect(fromPosition(0.5, 0, 10, "linear")).toBe(5);
    expect(fromPosition(1, 0, 10, "linear")).toBe(10);
  });

  it("round-trips", () => {
    for (const value of [0, 2.5, 7, 10]) {
      const position = toPosition(value, 0, 10, "linear");
      expect(fromPosition(position, 0, 10, "linear")).toBeCloseTo(value, 10);
    }
  });

  it("clamps out-of-range positions and values", () => {
    expect(fromPosition(-1, 0, 10, "linear")).toBe(0);
    expect(fromPosition(2, 0, 10, "linear")).toBe(10);
    expect(toPosition(-5, 0, 10, "linear")).toBe(0);
    expect(toPosition(50, 0, 10, "linear")).toBe(1);
  });

  it("survives a zero-width range", () => {
    expect(toPosition(5, 5, 5, "linear")).toBe(0);
  });
});

describe("log scale", () => {
  const MIN = 0.001;
  const MAX = 2;

  it("maps endpoints", () => {
    expect(fromPosition(0, MIN, MAX, "log")).toBeCloseTo(MIN, 10);
    expect(fromPosition(1, MIN, MAX, "log")).toBeCloseTo(MAX, 10);
  });

  it("puts the geometric mean at the midpoint", () => {
    expect(fromPosition(0.5, MIN, MAX, "log")).toBeCloseTo(
      Math.sqrt(MIN * MAX),
      10,
    );
  });

  it("round-trips across orders of magnitude", () => {
    for (const value of [0.001, 0.01, 0.03, 0.1, 0.5, 1, 2]) {
      const position = toPosition(value, MIN, MAX, "log");
      expect(position).toBeGreaterThanOrEqual(0);
      expect(position).toBeLessThanOrEqual(1);
      expect(fromPosition(position, MIN, MAX, "log")).toBeCloseTo(value, 8);
    }
  });

  it("spreads each decade evenly — the whole point of a log knob", () => {
    // 0.001 → 2 spans ~3.3 decades. Each decade should get a similar slice of
    // the track, unlike a linear scale where the small end is unreachable.
    const a = toPosition(0.01, MIN, MAX, "log") - toPosition(0.001, MIN, MAX, "log");
    const b = toPosition(0.1, MIN, MAX, "log") - toPosition(0.01, MIN, MAX, "log");
    const c = toPosition(1, MIN, MAX, "log") - toPosition(0.1, MIN, MAX, "log");
    expect(a).toBeCloseTo(b, 6);
    expect(b).toBeCloseTo(c, 6);
  });

  it("falls back to linear when min is not positive", () => {
    // log of a non-positive minimum is undefined; degrade rather than emit NaN.
    expect(fromPosition(0.5, 0, 10, "log")).toBe(5);
    expect(toPosition(5, 0, 10, "log")).toBeCloseTo(0.5, 10);
    expect(Number.isNaN(fromPosition(0.5, -1, 10, "log"))).toBe(false);
  });

  it("handles a non-positive value on a log track", () => {
    expect(Number.isNaN(toPosition(0, MIN, MAX, "log"))).toBe(false);
  });
});
