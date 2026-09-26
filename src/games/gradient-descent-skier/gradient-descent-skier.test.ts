import { beforeEach, describe, expect, it } from "vitest";
import {
  EMPTY_PROGRESSION,
  HIGH_SCORE_THRESHOLD,
  createMemoryAdapter,
  useProgression,
} from "@/engine/progression";
import {
  A,
  B,
  C,
  DEFAULT_LEARNING_RATE,
  DIVERGENCE_LOSS,
  DOMAIN,
  GLOBAL_MINIMUM,
  LOCAL_MINIMA,
  MAX_LEARNING_RATE,
  MAX_MOMENTUM,
  MINIMA,
  MIN_LEARNING_RATE,
  OVERSHOOT_DELTA,
  REACH_TOLERANCE,
  RIDGE_X,
  STABLE_RATE,
  START,
  START_LOSS,
  STEP_BUDGET,
  UPHILL_DELTA,
  WIN_SCORE,
  curvatureX,
  descend,
  distanceTo,
  evaluate,
  gradientAt,
  gradientNorm,
  lossAt,
  makeSkier,
  nearestMinimum,
  nextStepLength,
  step,
  type Outcome,
} from "./ml";
import { createCodeApi, useGradientSkierStore } from "./store";
import { whyCardFor } from "./why-cards";

/** Central difference, for checking the analytic gradient. */
function numericGradient(x: number, y: number, h = 1e-6) {
  return {
    x: (lossAt(x + h, y) - lossAt(x - h, y)) / (2 * h),
    y: (lossAt(x, y + h) - lossAt(x, y - h)) / (2 * h),
  };
}

const run = (learningRate: number, momentum = 0) => {
  const result = descend(learningRate, momentum);
  return {
    ...result,
    evaluation: evaluate({
      pos: result.final.pos,
      finalLoss: result.finalLoss,
      steps: result.steps,
      diverged: result.diverged,
      settled: result.settled,
      learningRate,
      momentum,
      overshoots: result.overshoots,
      rises: result.rises,
      minX: result.minX,
    }),
  };
};

const reachesBottom = (outcome: Outcome) =>
  outcome === "win" || outcome === "near-miss";

// ═══════════════════════════════════════════════════════════════════════════
// THE LESSON
//
// "Optimization is a step-size balancing act; learning rate is the single most
// consequential hyperparameter, and momentum helps escape traps."
//
// Every clause of that has to be true of the actual surface, not just asserted in
// the copy. These are the assertions that hold it to that.
// ═══════════════════════════════════════════════════════════════════════════

describe("the lesson", () => {
  it("a rate too small runs out of budget without diverging or sticking", () => {
    const tiny = run(0.001);
    expect(tiny.evaluation.outcome).toBe("slow-convergence");
    expect(tiny.diverged).toBe(false);
    expect(tiny.steps).toBe(STEP_BUDGET);
    // The distinguishing evidence: every step went downhill, it just didn't
    // travel far. No overshoots at all.
    expect(tiny.overshoots).toBe(0);
    expect(tiny.finalLoss).toBeLessThan(START_LOSS);
  });

  it("a workable rate WITHOUT momentum gets trapped in the shallow valley", () => {
    // This is the heart of the game: descent does exactly what it's told and
    // still lands in the wrong valley.
    for (const rate of [0.05, 0.1, 0.2]) {
      const trapped = run(rate);
      expect(trapped.evaluation.outcome, `rate ${rate}`).toBe("local-minimum");
      expect(trapped.settled).toBe(true);
      expect(trapped.diverged).toBe(false);

      const nearest = nearestMinimum(trapped.final.pos);
      expect(nearest.minimum.kind).toBe("local");
      // Genuinely settled: the slope there really is flat.
      expect(gradientNorm(gradientAt(trapped.final.pos.x, trapped.final.pos.y)))
        .toBeLessThan(0.05);
    }
  });

  it("THE SAME rate WITH momentum escapes the trap and wins", () => {
    // Same α, same surface, different outcome — which is the only honest way to
    // show that momentum is what did it.
    for (const rate of [0.05, 0.1, 0.2]) {
      const trapped = run(rate, 0);
      const freed = run(rate, 0.85);

      expect(trapped.evaluation.outcome, `rate ${rate} no momentum`).toBe(
        "local-minimum",
      );
      expect(freed.evaluation.outcome, `rate ${rate} with momentum`).toBe("win");
      expect(freed.evaluation.reachedGlobal).toBe(true);
      expect(freed.finalLoss).toBeLessThan(trapped.finalLoss);
      expect(freed.evaluation.score).toBeGreaterThanOrEqual(WIN_SCORE);
    }
  });

  it("a rate too large diverges off the mountain", () => {
    const blown = run(1.0);
    expect(blown.evaluation.outcome).toBe("diverged");
    expect(blown.diverged).toBe(true);
    expect(blown.evaluation.failure!.name).toBe("Divergence");
    // Fast: divergence is not a slow drift.
    expect(blown.steps).toBeLessThan(10);
  });

  it("the band just below divergence oscillates rather than settling", () => {
    // A distinct failure from both "too small" and "too big", and the one people
    // most often misdiagnose.
    const bouncing = run(0.6);
    expect(bouncing.evaluation.outcome).toBe("oscillating");
    expect(bouncing.diverged).toBe(false);
    expect(bouncing.settled).toBe(false);
    expect(bouncing.overshoots).toBeGreaterThan(0);
  });

  it("orders the failures by learning rate, low to high", () => {
    // The whole point of "step-size balancing act": failure at both ends, success
    // in the middle.
    const outcomes = [0.001, 0.01, 0.05, 1.0, 1.5].map(
      (rate) => run(rate).evaluation.outcome,
    );
    expect(outcomes[0]).toBe("slow-convergence");
    expect(outcomes[1]).toBe("slow-convergence");
    expect(outcomes[2]).toBe("local-minimum");
    expect(outcomes[3]).toBe("diverged");
    expect(outcomes[4]).toBe("diverged");
  });

  it("names each failure with numbers that back it up", () => {
    for (const [rate, momentum, name] of [
      [1.0, 0, "Divergence"],
      [0.05, 0, "Local minimum"],
      [0.6, 0, "Oscillation"],
      [0.001, 0, "Slow convergence"],
    ] as const) {
      const result = run(rate, momentum);
      expect(result.evaluation.failure, `rate ${rate}`).not.toBeNull();
      expect(result.evaluation.failure!.name).toBe(name);
      // Every claim carries a figure.
      expect(result.evaluation.failure!.detail).toMatch(/\d/);
      expect(result.evaluation.failure!.detail).not.toMatch(/game over/i);
    }
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// The surface and the gradient
// ═══════════════════════════════════════════════════════════════════════════

describe("the surface", () => {
  it("matches the stated formula", () => {
    for (const [x, y] of [
      [0, 0],
      [1, 0],
      [-1.5, 0.7],
      [2, -1.2],
    ] as const) {
      const well = x * x - 1;
      expect(lossAt(x, y)).toBeCloseTo(A * well * well + C * x + B * y * y, 12);
    }
  });

  it("has exactly two valleys, one deeper than the other", () => {
    expect(MINIMA).toHaveLength(2);
    expect(LOCAL_MINIMA).toHaveLength(1);
    expect(GLOBAL_MINIMUM.loss).toBeLessThan(LOCAL_MINIMA[0]!.loss);
    // On opposite sides of the ridge, which is what makes the trap possible.
    expect(GLOBAL_MINIMUM.x).toBeLessThan(0);
    expect(LOCAL_MINIMA[0]!.x).toBeGreaterThan(0);
  });

  it("finds minima that really are minima", () => {
    for (const minimum of MINIMA) {
      // Flat, and lower than its neighbours.
      expect(gradientNorm(gradientAt(minimum.x, minimum.y))).toBeLessThan(1e-6);
      expect(lossAt(minimum.x - 0.05, 0)).toBeGreaterThan(minimum.loss);
      expect(lossAt(minimum.x + 0.05, 0)).toBeGreaterThan(minimum.loss);
    }
  });

  it("starts the skier on the wrong side of the ridge", () => {
    // If the start were on the global side, plain descent would win and the
    // local-minimum lesson would never fire.
    expect(START.x).toBeGreaterThan(0);
    expect(nearestMinimum(START).minimum.kind).toBe("local");
    expect(START_LOSS).toBeGreaterThan(LOCAL_MINIMA[0]!.loss);
  });

  it("has an analytic gradient that matches a numeric one", () => {
    // The claim is that the gradient is exact, not estimated. This is the check.
    for (const [x, y] of [
      [0.3, 0.4],
      [-1.2, -0.6],
      [1.55, 0.95],
      [2.1, 1.4],
      [-0.05, 0.02],
    ] as const) {
      const analytic = gradientAt(x, y);
      const numeric = numericGradient(x, y);
      expect(analytic.x).toBeCloseTo(numeric.x, 5);
      expect(analytic.y).toBeCloseTo(numeric.y, 5);
    }
  });

  it("has zero gradient in y along the valley floor", () => {
    expect(gradientAt(0.7, 0).y).toBe(0);
    expect(gradientAt(-1.3, 0).y).toBe(0);
  });

  it("grows steeper than linearly, which is why big steps explode", () => {
    const near = Math.abs(gradientAt(1.2, 0).x);
    const far = Math.abs(gradientAt(2.4, 0).x);
    // Doubling x more than doubles the slope.
    expect(far / near).toBeGreaterThan(2);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// The update rule
// ═══════════════════════════════════════════════════════════════════════════

describe("step", () => {
  it("implements v ← βv − α∇f and θ ← θ + v exactly", () => {
    const skier = {
      pos: { x: 1.2, y: 0.5 },
      velocity: { x: 0.1, y: -0.2 },
      learningRate: 0.1,
      momentum: 0.9,
    };
    const gradient = gradientAt(1.2, 0.5);
    const result = step(skier);

    const expectedVx = 0.9 * 0.1 - 0.1 * gradient.x;
    const expectedVy = 0.9 * -0.2 - 0.1 * gradient.y;

    expect(result.skier.velocity.x).toBeCloseTo(expectedVx, 12);
    expect(result.skier.velocity.y).toBeCloseTo(expectedVy, 12);
    expect(result.skier.pos.x).toBeCloseTo(1.2 + expectedVx, 12);
    expect(result.skier.pos.y).toBeCloseTo(0.5 + expectedVy, 12);
  });

  it("is plain gradient descent at momentum 0", () => {
    const skier = makeSkier(0.1, 0);
    const gradient = gradientAt(START.x, START.y);
    const result = step(skier);
    expect(result.skier.pos.x).toBeCloseTo(START.x - 0.1 * gradient.x, 12);
    expect(result.skier.pos.y).toBeCloseTo(START.y - 0.1 * gradient.y, 12);
  });

  it("goes downhill for a small enough rate", () => {
    const before = lossAt(START.x, START.y);
    expect(step(makeSkier(0.01, 0)).loss).toBeLessThan(before);
  });

  it("reports the loss of the position it moved to", () => {
    const result = step(makeSkier(0.05, 0));
    expect(result.loss).toBeCloseTo(
      lossAt(result.skier.pos.x, result.skier.pos.y),
      12,
    );
  });

  it("flags leaving the domain as divergence", () => {
    const escaping = {
      pos: { x: DOMAIN.maxX - 0.01, y: 0 },
      velocity: { x: 5, y: 0 },
      learningRate: 0,
      momentum: 1,
    };
    expect(step(escaping).diverged).toBe(true);
  });

  it("flags an exploded loss as divergence even inside the domain", () => {
    const result = step({
      pos: { x: 2.3, y: 1.7 },
      velocity: { x: 0, y: 0 },
      learningRate: 0,
      momentum: 0,
    });
    // Corner of the domain is high but legal.
    expect(result.diverged).toBe(lossAt(2.3, 1.7) > DIVERGENCE_LOSS);
  });

  it("never mutates the skier it was given", () => {
    const skier = makeSkier(0.1, 0.5);
    const snapshot = JSON.stringify(skier);
    step(skier);
    expect(JSON.stringify(skier)).toBe(snapshot);
  });
});

describe("descend", () => {
  it("is deterministic — the same dials give the same path", () => {
    expect(descend(0.05, 0.5).path).toEqual(descend(0.05, 0.5).path);
  });

  it("records one loss per position", () => {
    const result = descend(0.05, 0);
    expect(result.losses).toHaveLength(result.path.length);
    expect(result.path[0]).toEqual(START);
    expect(result.losses[0]).toBeCloseTo(START_LOSS, 12);
  });

  it("never exceeds the budget", () => {
    for (const rate of [0.001, 0.05, 0.6, 1.5]) {
      expect(descend(rate, 0).steps).toBeLessThanOrEqual(STEP_BUDGET);
    }
  });

  it("stops the moment it diverges", () => {
    const result = descend(1.5, 0);
    expect(result.diverged).toBe(true);
    // No steps taken after leaving the mountain.
    expect(result.path).toHaveLength(result.steps + 1);
  });

  it("counts an overshoot only when the loss actually rises", () => {
    const smooth = descend(0.01, 0);
    expect(smooth.overshoots).toBe(0);
    for (let index = 1; index < smooth.losses.length; index += 1) {
      expect(smooth.losses[index]! - smooth.losses[index - 1]!).toBeLessThanOrEqual(
        OVERSHOOT_DELTA,
      );
    }
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Evaluation
// ═══════════════════════════════════════════════════════════════════════════

describe("evaluate", () => {
  const base = {
    steps: 20,
    diverged: false,
    settled: true,
    learningRate: 0.1,
    momentum: 0.85,
    overshoots: 0,
  };

  it("scores zero progress for a divergence", () => {
    const result = evaluate({
      ...base,
      pos: { x: 99, y: 99 },
      finalLoss: Number.POSITIVE_INFINITY,
      diverged: true,
      settled: false,
    });
    expect(result.progress).toBe(0);
    expect(result.score).toBe(0);
  });

  it("scores full progress at the global minimum", () => {
    const result = evaluate({
      ...base,
      pos: { x: GLOBAL_MINIMUM.x, y: GLOBAL_MINIMUM.y },
      finalLoss: GLOBAL_MINIMUM.loss,
    });
    expect(result.progress).toBeCloseTo(1, 6);
    expect(result.reachedGlobal).toBe(true);
    expect(result.outcome).toBe("win");
  });

  it("penalises using more of the budget", () => {
    const quick = evaluate({
      ...base,
      steps: 10,
      pos: { ...GLOBAL_MINIMUM },
      finalLoss: GLOBAL_MINIMUM.loss,
    });
    const slow = evaluate({
      ...base,
      steps: STEP_BUDGET,
      pos: { ...GLOBAL_MINIMUM },
      finalLoss: GLOBAL_MINIMUM.loss,
    });
    expect(quick.efficiency).toBeGreaterThan(slow.efficiency);
    expect(quick.score).toBeGreaterThan(slow.score);
  });

  it("calls arriving slowly a near-miss, not a failure", () => {
    // Reaching the bottom is never a failure, however long it took.
    const result = evaluate({
      ...base,
      steps: STEP_BUDGET,
      pos: { ...GLOBAL_MINIMUM },
      finalLoss: GLOBAL_MINIMUM.loss,
    });
    expect(result.reachedGlobal).toBe(true);
    expect(result.outcome).toBe("near-miss");
    expect(result.failure).toBeNull();
  });

  it("will not call a still-moving skier trapped", () => {
    // Passing through the shallow valley is not the same as being stuck in it,
    // and naming it "local minimum" would teach the wrong word.
    const result = evaluate({
      ...base,
      settled: false,
      steps: 5,
      pos: { x: LOCAL_MINIMA[0]!.x, y: 0 },
      finalLoss: LOCAL_MINIMA[0]!.loss,
    });
    expect(result.outcome).toBe("running");
    expect(result.failure).toBeNull();
  });

  it("treats sitting at the bottom without settling as oscillation", () => {
    const result = evaluate({
      ...base,
      settled: false,
      steps: STEP_BUDGET,
      pos: { ...GLOBAL_MINIMUM },
      finalLoss: GLOBAL_MINIMUM.loss,
      overshoots: 0,
    });
    // Reached it but won't stop: that's a step size too large, not too small.
    expect(result.outcome).toBe("oscillating");
    expect(result.failure!.name).toBe("Oscillation");
  });

  it("uses REACH_TOLERANCE to decide arrival", () => {
    const justInside = evaluate({
      ...base,
      pos: { x: GLOBAL_MINIMUM.x + REACH_TOLERANCE * 0.9, y: 0 },
      finalLoss: GLOBAL_MINIMUM.loss + 0.01,
    });
    const justOutside = evaluate({
      ...base,
      pos: { x: GLOBAL_MINIMUM.x + REACH_TOLERANCE * 1.5, y: 0 },
      finalLoss: GLOBAL_MINIMUM.loss + 0.05,
    });
    expect(justInside.reachedGlobal).toBe(true);
    expect(justOutside.reachedGlobal).toBe(false);
  });
});

describe("distanceTo / nearestMinimum", () => {
  it("measures euclidean distance", () => {
    expect(distanceTo({ x: 0, y: 0 }, { ...GLOBAL_MINIMUM, x: 3, y: 4 })).toBe(5);
  });

  it("picks the closer valley", () => {
    expect(nearestMinimum({ x: 1.2, y: 0 }).minimum.kind).toBe("local");
    expect(nearestMinimum({ x: -1.2, y: 0 }).minimum.kind).toBe("global");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Store
// ═══════════════════════════════════════════════════════════════════════════

describe("store", () => {
  beforeEach(() => {
    useProgression.setState({
      ...EMPTY_PROGRESSION,
      hydrated: true,
      syncError: null,
      lastGain: null,
    });
    useProgression.getState().setAdapter(createMemoryAdapter());
    useGradientSkierStore.getState().fullReset();
  });

  it("starts at the top with the full budget", () => {
    const state = useGradientSkierStore.getState();
    expect(state.skier.pos).toEqual(START);
    expect(state.currentLoss).toBeCloseTo(START_LOSS, 12);
    expect(state.stepsRemaining).toBe(STEP_BUDGET);
    expect(state.stepsTaken).toBe(0);
    expect(state.trail).toEqual([START]);
    expect(state.skier.learningRate).toBe(DEFAULT_LEARNING_RATE);
    expect(state.skier.momentum).toBe(0);
  });

  it("derives the metric from the real surface, not a stored number", () => {
    useGradientSkierStore.getState().step();
    const state = useGradientSkierStore.getState();
    expect(state.currentLoss).toBeCloseTo(
      lossAt(state.skier.pos.x, state.skier.pos.y),
      12,
    );
    // The slope at the skier's feet — where the NEXT step starts — not the
    // one the last step used, which is where the skier used to be.
    expect(state.gradient).toEqual(gradientAt(state.skier.pos.x, state.skier.pos.y));
    expect(state.gradient).not.toEqual(
      gradientAt(state.trail.at(-2)!.x, state.trail.at(-2)!.y),
    );
  });

  it("moves the metric on every step", () => {
    const before = useGradientSkierStore.getState().currentLoss;
    useGradientSkierStore.getState().step();
    expect(useGradientSkierStore.getState().currentLoss).not.toBe(before);
  });

  it("clamps the learning rate to the dial's range", () => {
    useGradientSkierStore.getState().setLearningRate(99);
    expect(useGradientSkierStore.getState().skier.learningRate).toBe(
      MAX_LEARNING_RATE,
    );
    useGradientSkierStore.getState().setLearningRate(-5);
    expect(useGradientSkierStore.getState().skier.learningRate).toBe(
      MIN_LEARNING_RATE,
    );
  });

  it("clamps momentum to the dial's range", () => {
    useGradientSkierStore.getState().setMomentum(5);
    expect(useGradientSkierStore.getState().skier.momentum).toBe(MAX_MOMENTUM);
    useGradientSkierStore.getState().setMomentum(-1);
    expect(useGradientSkierStore.getState().skier.momentum).toBe(0);
  });

  it("grows the trail one point per step", () => {
    for (let index = 0; index < 5; index += 1) {
      useGradientSkierStore.getState().step();
    }
    expect(useGradientSkierStore.getState().trail).toHaveLength(6);
    expect(useGradientSkierStore.getState().stepsTaken).toBe(5);
    expect(useGradientSkierStore.getState().stepsRemaining).toBe(
      STEP_BUDGET - 5,
    );
  });

  it("stops stepping once it diverges", () => {
    useGradientSkierStore.getState().setLearningRate(1.5);
    useGradientSkierStore.getState().runToEnd();

    const afterRun = useGradientSkierStore.getState();
    expect(afterRun.diverged).toBe(true);
    const steps = afterRun.stepsTaken;

    useGradientSkierStore.getState().step();
    expect(useGradientSkierStore.getState().stepsTaken).toBe(steps);
  });

  it("stops stepping once it settles", () => {
    useGradientSkierStore.getState().setLearningRate(0.1);
    useGradientSkierStore.getState().runToEnd();
    expect(useGradientSkierStore.getState().settled).toBe(true);

    const steps = useGradientSkierStore.getState().stepsTaken;
    useGradientSkierStore.getState().step();
    expect(useGradientSkierStore.getState().stepsTaken).toBe(steps);
  });

  it("runToEnd always terminates", () => {
    for (const rate of [0.001, 0.05, 0.6, 1.5]) {
      useGradientSkierStore.getState().fullReset();
      useGradientSkierStore.getState().setLearningRate(rate);
      useGradientSkierStore.getState().runToEnd();
      const state = useGradientSkierStore.getState();
      expect(state.diverged || state.settled || state.stepsRemaining === 0).toBe(
        true,
      );
      expect(state.running).toBe(false);
    }
  });

  it("flags an overshoot on the step that caused it", () => {
    useGradientSkierStore.getState().setLearningRate(0.6);
    useGradientSkierStore.getState().runToEnd();
    expect(useGradientSkierStore.getState().overshoots).toBeGreaterThan(0);
  });

  it("wins with momentum and awards XP once", () => {
    useGradientSkierStore.getState().setLearningRate(0.1);
    useGradientSkierStore.getState().setMomentum(0.85);
    useGradientSkierStore.getState().runToEnd();

    const evaluation = useGradientSkierStore.getState().check();
    expect(evaluation.outcome).toBe("win");
    expect(useGradientSkierStore.getState().won).toBe(true);

    const xp = useProgression.getState().xp;
    expect(xp).toBeGreaterThan(0);
    useGradientSkierStore.getState().check();
    expect(useProgression.getState().xp).toBe(xp);
  });

  it("names Divergence and awards nothing", () => {
    useGradientSkierStore.getState().setLearningRate(1.5);
    useGradientSkierStore.getState().runToEnd();

    const evaluation = useGradientSkierStore.getState().check();
    expect(evaluation.outcome).toBe("diverged");
    expect(useGradientSkierStore.getState().failure?.name).toBe("Divergence");
    expect(useProgression.getState().xp).toBe(0);
  });

  it("names Local minimum without momentum", () => {
    useGradientSkierStore.getState().setLearningRate(0.1);
    useGradientSkierStore.getState().setMomentum(0);
    useGradientSkierStore.getState().runToEnd();

    expect(useGradientSkierStore.getState().check().outcome).toBe(
      "local-minimum",
    );
    expect(useGradientSkierStore.getState().failure?.name).toBe(
      "Local minimum",
    );
  });

  it("credits the code lane for the third star when the check comes from api.check()", () => {
    useGradientSkierStore.getState().setLearningRate(0.1);
    useGradientSkierStore.getState().setMomentum(0.85);
    useGradientSkierStore.getState().runToEnd();
    useGradientSkierStore.getState().check("code");

    const record = useProgression.getState().games["gradient-descent-skier"];
    expect(record?.codeLaneCleared).toBe(true);
    expect(record?.stars).toBe(3);
  });

  it("keeps the dials on reset but restores them on fullReset", () => {
    useGradientSkierStore.getState().setLearningRate(0.3);
    useGradientSkierStore.getState().setMomentum(0.5);
    useGradientSkierStore.getState().step();

    useGradientSkierStore.getState().reset();
    let state = useGradientSkierStore.getState();
    expect(state.skier.learningRate).toBeCloseTo(0.3, 10);
    expect(state.skier.momentum).toBeCloseTo(0.5, 10);
    expect(state.stepsTaken).toBe(0);
    expect(state.trail).toEqual([START]);

    useGradientSkierStore.getState().fullReset();
    state = useGradientSkierStore.getState();
    expect(state.skier.learningRate).toBe(DEFAULT_LEARNING_RATE);
    expect(state.skier.momentum).toBe(0);
  });

  it("attaches a distinct why-card to every action", () => {
    const keys = new Set<string>();
    const record = () => {
      const card = useGradientSkierStore.getState().whyCard;
      expect(card).not.toBeNull();
      keys.add(card!.key);
    };

    record();
    useGradientSkierStore.getState().setLearningRate(0.2);
    record();
    useGradientSkierStore.getState().setMomentum(0.6);
    record();
    useGradientSkierStore.getState().step();
    record();
    useGradientSkierStore.getState().check();
    record();

    expect(keys.size).toBe(5);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Why-cards
// ═══════════════════════════════════════════════════════════════════════════

describe("why-cards", () => {
  it("describes the learning rate as a multiplier, never as speed", () => {
    // The commonest misconception about this hyperparameter.
    const card = whyCardFor({
      kind: "rate-changed",
      learningRate: 0.2,
      previous: 0.05,
      gradient: gradientAt(START.x, START.y),
      skier: makeSkier(0.2, 0),
      finished: false,
    });
    expect(card.body).toMatch(/multiplier on the slope/i);
    expect(card.body).toMatch(/not a speed/i);
  });

  it("names divergence and blames the step size, not the surface", () => {
    const card = whyCardFor({
      kind: "checked",
      evaluation: run(1.0).evaluation,
    });
    expect(card.title).toMatch(/divergence/i);
    expect(card.tone).toBe("bad");
    expect(card.body).toMatch(/step size/i);
  });

  it("names the local minimum and points at momentum", () => {
    const card = whyCardFor({
      kind: "checked",
      evaluation: run(0.1, 0).evaluation,
    });
    expect(card.title).toMatch(/local minimum/i);
    expect(card.body).toMatch(/momentum/i);
    expect(card.tone).toBe("bad");
  });

  it("distinguishes oscillation from slow convergence in the copy", () => {
    const oscillating = whyCardFor({
      kind: "checked",
      evaluation: run(0.6, 0).evaluation,
    });
    const slow = whyCardFor({
      kind: "checked",
      evaluation: run(0.001, 0).evaluation,
    });

    expect(oscillating.title).toMatch(/oscillation/i);
    expect(oscillating.body).toMatch(/uphill/i);
    expect(slow.title).toMatch(/slow convergence/i);
    expect(slow.body).toMatch(/downhill/i);
    expect(slow.body).toMatch(/too small/i);
  });

  it("celebrates a win by naming what made it work", () => {
    const card = whyCardFor({
      kind: "checked",
      evaluation: run(0.1, 0.85).evaluation,
    });
    expect(card.tone).toBe("good");
    expect(card.title).toMatch(/global minimum/i);
  });

  it("explains an overshoot on the step that caused it", () => {
    const card = whyCardFor({
      kind: "stepped",
      loss: 2.4,
      previousLoss: 1.1,
      gradient: gradientAt(1.4, 0.3),
      overshot: true,
      diverged: false,
      settled: false,
      pos: { x: 1.4, y: 0.3 },
      learningRate: 0.5,
      momentum: 0,
      minX: 1.4,
    });
    expect(card.title).toMatch(/overshot/i);
    expect(card.body).toMatch(/past the valley floor/i);
    expect(card.tone).toBe("warn");
  });

  it("tells the two valleys apart when the skier settles", () => {
    const inGlobal = whyCardFor({
      kind: "stepped",
      loss: GLOBAL_MINIMUM.loss,
      previousLoss: GLOBAL_MINIMUM.loss + 0.01,
      gradient: { x: 0.001, y: 0 },
      overshot: false,
      diverged: false,
      settled: true,
      pos: { x: GLOBAL_MINIMUM.x, y: 0 },
      learningRate: 0.1,
      momentum: 0.85,
      minX: GLOBAL_MINIMUM.x - 0.4,
    });
    const inLocal = whyCardFor({
      kind: "stepped",
      loss: LOCAL_MINIMA[0]!.loss,
      previousLoss: LOCAL_MINIMA[0]!.loss + 0.01,
      gradient: { x: 0.001, y: 0 },
      overshot: false,
      diverged: false,
      settled: true,
      pos: { x: LOCAL_MINIMA[0]!.x, y: 0 },
      learningRate: 0.1,
      momentum: 0,
      minX: LOCAL_MINIMA[0]!.x,
    });

    expect(inGlobal.tone).toBe("good");
    expect(inLocal.tone).toBe("warn");
    expect(inLocal.body).toMatch(/shallow/i);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Honest diagnosis when momentum is involved
//
// The first version of the verdict copy was written as if β were always 0. It
// blamed the learning rate for divergence that momentum caused, told a skier
// who had overshot the deep valley and rolled back that it lacked momentum,
// and promised "halve it and the same descent settles" for oscillation where
// halving doesn't help. These pin the replacement to what actually happens.
// ═══════════════════════════════════════════════════════════════════════════

describe("the surface's landmarks", () => {
  it("puts the ridge between the valleys, at the top of the slope", () => {
    expect(RIDGE_X).toBeGreaterThan(GLOBAL_MINIMUM.x);
    expect(RIDGE_X).toBeLessThan(LOCAL_MINIMA[0]!.x);
    expect(Math.abs(gradientAt(RIDGE_X, 0).x)).toBeLessThan(1e-9);
    expect(lossAt(RIDGE_X - 0.05, 0)).toBeLessThan(lossAt(RIDGE_X, 0));
    expect(lossAt(RIDGE_X + 0.05, 0)).toBeLessThan(lossAt(RIDGE_X, 0));
  });

  it("derives the stability bound from the deep valley's curvature, and it holds", () => {
    expect(STABLE_RATE).toBeCloseTo(2 / curvatureX(GLOBAL_MINIMUM.x), 12);
    const settleFrom = (rate: number) => {
      let skier = {
        pos: { x: GLOBAL_MINIMUM.x + 0.01, y: 0 },
        velocity: { x: 0, y: 0 },
        learningRate: rate,
        momentum: 0,
      };
      for (let index = 0; index < 500; index += 1) skier = step(skier).skier;
      return Math.abs(skier.pos.x - GLOBAL_MINIMUM.x);
    };
    expect(settleFrom(STABLE_RATE * 0.95)).toBeLessThan(1e-6);
    expect(settleFrom(STABLE_RATE * 1.05)).toBeGreaterThan(0.01);
  });
});

describe("nextStepLength", () => {
  it("is exactly the length of the step the update rule takes next, momentum included", () => {
    for (const [rate, momentum] of [
      [0.05, 0],
      [0.05, 0.85],
      [0.2, 0.6],
    ] as const) {
      let skier = makeSkier(rate, momentum);
      for (let index = 0; index < 3; index += 1) skier = step(skier).skier;
      expect(nextStepLength(skier)).toBeCloseTo(step(skier).stepLength, 12);
    }
  });

  it("differs from slope × rate once momentum carries speed", () => {
    let skier = makeSkier(0.05, 0.85);
    skier = step(skier).skier;
    const naive = gradientNorm(gradientAt(skier.pos.x, skier.pos.y)) * 0.05;
    expect(Math.abs(nextStepLength(skier) - naive)).toBeGreaterThan(0.05);
  });
});

describe("verdicts with momentum", () => {
  const landsAt = (remedy: { learningRate: number; momentum: number }) =>
    run(remedy.learningRate, remedy.momentum).evaluation.outcome;

  it("(0.05, 0.95): calls high-momentum bouncing oscillation, and offers a fix that works", () => {
    const { evaluation } = run(0.05, 0.95);
    expect(evaluation.outcome).toBe("oscillating");
    expect(evaluation.failure!.detail).not.toMatch(
      /Halve it and the same descent settles/,
    );
    expect(evaluation.remedy).not.toBeNull();
    expect(reachesBottom(landsAt(evaluation.remedy!))).toBe(true);
    expect(evaluation.remedy!.momentum).toBeLessThan(0.95);
    expect(evaluation.failure!.detail).toContain(evaluation.remedy!.sentence);
  });

  it("(0.1, 0.95): no longer calls this rate 'just below the one that would diverge'", () => {
    // Divergence with no momentum starts near 0.675; 0.1 is nowhere near it.
    const { evaluation } = run(0.1, 0.95);
    expect(evaluation.outcome).toBe("oscillating");
    expect(evaluation.failure!.detail).not.toMatch(
      /just below the one that would diverge/,
    );
    expect(evaluation.failure!.detail).toMatch(/momentum 0\.95/);
  });

  it("(0.2, 0.9): a skier that overshot the deep valley and rolled back had too MUCH momentum", () => {
    const result = run(0.2, 0.9);
    const { evaluation } = result;
    expect(evaluation.outcome).toBe("local-minimum");
    expect(result.minX).toBeLessThan(GLOBAL_MINIMUM.x);
    expect(evaluation.crossedRidge).toBe(true);
    expect(evaluation.passedDeepValley).toBe(true);
    expect(evaluation.failure!.detail).toMatch(/too much momentum, not too little/);
    expect(evaluation.failure!.detail).not.toMatch(/wasn't enough/);
    // And the fix it names eases momentum, and genuinely lands.
    expect(evaluation.remedy!.momentum).toBeLessThan(0.9);
    expect(reachesBottom(landsAt(evaluation.remedy!))).toBe(true);

    const card = whyCardFor({ kind: "checked", evaluation });
    expect(card.body).toMatch(/Too much momentum, not too little/);
  });

  it("(0.05, 0.3): a skier that never reached the ridge is told momentum wasn't enough", () => {
    const { evaluation, minX } = run(0.05, 0.3);
    expect(evaluation.outcome).toBe("local-minimum");
    expect(minX).toBeGreaterThan(RIDGE_X);
    expect(evaluation.crossedRidge).toBe(false);
    expect(evaluation.failure!.detail).toMatch(
      /wasn't enough to carry you over the ridge/,
    );
    expect(evaluation.remedy!.momentum).toBeGreaterThan(0.3);
  });

  it("(0.3, 0.85): divergence under momentum names the effective step, not a 0.4 rule", () => {
    const { evaluation } = run(0.3, 0.85);
    expect(evaluation.outcome).toBe("diverged");
    // α/(1−β) = 0.3 / 0.15 = 2.
    expect(evaluation.failure!.detail).toMatch(/α\/\(1−β\) ≈ 2\.00/);
    const card = whyCardFor({ kind: "checked", evaluation });
    expect(card.body).not.toMatch(/Below roughly 0\.4/);
    expect(card.body).toMatch(/step size/i);
  });

  it("(0.005, 0.95): a tiny rate at high momentum is told to RAISE the rate, not lower it", () => {
    // Easing β shrinks the effective step α/(1−β), so the rate has to rise.
    // The fallback used to list only lowering and then say "change both" —
    // read as "lower both", the wrong way on this corner of the dials.
    const { evaluation } = run(0.005, 0.95);
    expect(evaluation.outcome).toBe("oscillating");
    expect(evaluation.remedy).not.toBeNull();
    expect(evaluation.remedy!.learningRate).toBeGreaterThan(0.005);
    expect(evaluation.remedy!.momentum).toBeLessThan(0.95);
    expect(reachesBottom(landsAt(evaluation.remedy!))).toBe(true);
    expect(evaluation.failure!.detail).toContain(evaluation.remedy!.sentence);
    expect(evaluation.failure!.detail).not.toMatch(/change both/);
  });

  it("(0.0081, 0.92): when raising the rate alone settles it, that is the fix it names", () => {
    const { evaluation } = run(0.0081, 0.92);
    expect(evaluation.outcome).toBe("oscillating");
    expect(evaluation.remedy!.momentum).toBe(0.92);
    expect(evaluation.remedy!.learningRate).toBeGreaterThan(0.0081);
    expect(evaluation.remedy!.sentence).toMatch(/^Raising the rate to /);
    expect(reachesBottom(landsAt(evaluation.remedy!))).toBe(true);
  });

  it("runs every rate it names at exactly the number it prints", () => {
    // Near the edge of the winning band 0.61717… settles and 0.617 bounces,
    // so a remedy run at the unrounded value could name a rate that fails.
    let named = 0;
    for (const momentum of [0, 0.01, 0.3, 0.9, 0.95]) {
      for (let rate = MIN_LEARNING_RATE; rate <= MAX_LEARNING_RATE; rate *= 1.3) {
        const { remedy } = run(rate, momentum).evaluation;
        const printed = remedy ? /rate to ([0-9.]+)/.exec(remedy.sentence) : null;
        if (!remedy || !printed) continue;
        named += 1;
        expect(remedy.learningRate, remedy.sentence).toBe(Number(printed[1]));
        expect(reachesBottom(landsAt(remedy)), remedy.sentence).toBe(true);
      }
    }
    expect(named).toBeGreaterThan(20);
  });

  it("never points a fallback the wrong way: every change it says it tried was run, and failed", () => {
    // When nothing tried works, the advice lists what was tried and names no
    // direction for either dial. Re-run each single-dial change it claims.
    const shown = (value: number) => Number(value.toPrecision(3));
    const fails = (learningRate: number, momentum: number) =>
      learningRate < MIN_LEARNING_RATE ||
      learningRate > MAX_LEARNING_RATE ||
      !reachesBottom(landsAt({ learningRate, momentum }));
    const lower = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
    const raise = [1.5, 2, 3, 5, 10, 20];

    let fallbacks = 0;
    // The corners where fallbacks live: huge rates at (almost) no momentum,
    // tiny rates at the top of the momentum slider.
    for (const momentum of [0, 0.01, 0.02, 0.5, 0.9, 0.93, 0.94, 0.95]) {
      for (let rate = MIN_LEARNING_RATE; rate <= MAX_LEARNING_RATE; rate *= 1.05) {
        const { advice, remedy } = run(rate, momentum).evaluation;
        if (remedy || !advice) continue;
        fallbacks += 1;
        const at = `(${rate}, ${momentum}): ${advice}`;

        expect(advice, at).not.toMatch(/change both|add momentum as well/);
        expect(advice, at).toMatch(
          /try other combinations of both dials\.$|change the rate too\.$/,
        );
        if (/lower(ing)? (the )?rate|No lower rate|No other rate|Lowering or raising/i.test(advice)) {
          for (const factor of lower) {
            expect(fails(shown(rate * factor), momentum), `${at} ×${factor}`).toBe(true);
          }
        }
        if (/higher rate|No other rate|Lowering or raising/.test(advice)) {
          for (const factor of raise) {
            expect(fails(shown(rate * factor), momentum), `${at} ×${factor}`).toBe(true);
          }
        }
        const eased = /easing momentum by up to (\d\.\d\d)/.exec(advice);
        if (eased) {
          for (const drop of [0.1, 0.2, 0.3].filter((d) => d <= Number(eased[1]) + 1e-9)) {
            const value = Math.round((momentum - drop) * 100) / 100;
            expect(fails(rate, value), `${at} β ${value}`).toBe(true);
          }
        }
        const between = /between (\d\.\d\d) and (\d\.\d\d)/.exec(advice);
        if (between) {
          for (let step = Math.round(Number(between[1]) * 100) + 1; step <= Math.round(Number(between[2]) * 100); step += 1) {
            expect(fails(rate, step / 100), `${at} β ${step / 100}`).toBe(true);
          }
        }
        if (/No momentum setting on the dial/.test(advice)) {
          for (let step = 1; step <= Math.round(MAX_MOMENTUM * 100); step += 1) {
            expect(fails(rate, step / 100), `${at} β ${step / 100}`).toBe(true);
          }
        }
        expect(advice, at).not.toMatch(/between (\d\.\d\d) and \1/);
        if (/already at the top of the dial/.test(advice)) {
          expect(momentum).toBe(MAX_MOMENTUM);
        }
      }
    }
    expect(fallbacks).toBeGreaterThan(0);
  });

  it("(0.0029, 0.95): an oscillation that never left the right-hand basin says which valley it rocked in", () => {
    const { evaluation } = run(0.0029, 0.95);
    expect(evaluation.outcome).toBe("oscillating");
    expect(evaluation.crossedRidge).toBe(false);
    expect(evaluation.failure!.detail).toMatch(/past the shallow valley's floor and back/);
    const card = whyCardFor({ kind: "checked", evaluation });
    expect(card.body).toMatch(/past the shallow valley's floor and back/);
  });

  it("at the top of the momentum slider, says so instead of 'between 0.95 and 0.95'", () => {
    // α = 0.00279 at β = 0.95 stops short of the ridge, and no higher rate at
    // that momentum lands either (the property test above re-runs them).
    const { evaluation } = run(0.00279, 0.95);
    expect(evaluation.outcome).toBe("local-minimum");
    expect(evaluation.crossedRidge).toBe(false);
    expect(evaluation.remedy).toBeNull();
    expect(evaluation.failure!.detail).not.toMatch(/between 0\.95 and 0\.95/);
    expect(evaluation.advice).toMatch(/^Momentum is already at the top of the dial/);
    expect(evaluation.advice).toMatch(/try other combinations of both dials\.$/);
  });

  it("only ever offers a remedy that was run and reaches the deepest valley", () => {
    // The property behind every "… reaches the deepest valley" sentence.
    for (const momentum of [0, 0.3, 0.6, 0.85, 0.9, 0.95]) {
      for (const rate of [0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.44, 0.6, 1.0, 1.5]) {
        const { evaluation } = run(rate, momentum);
        if (!evaluation.remedy) continue;
        expect(
          reachesBottom(landsAt(evaluation.remedy)),
          `remedy for (${rate}, ${momentum})`,
        ).toBe(true);
      }
    }
  });

  it("never tells a bouncing descent that every step went downhill", () => {
    // α = 0.44 with no momentum settles into a two-step bounce whose uphill
    // steps (0.041) sit under the red-spike threshold; it used to be called
    // slow convergence — "the rate is too small" — which is backwards.
    const bounce = run(0.44, 0);
    expect(bounce.rises).toBeGreaterThan(0);
    expect(bounce.evaluation.outcome).toBe("oscillating");

    for (const momentum of [0, 0.5, 0.9, 0.95]) {
      for (const rate of [0.001, 0.002, 0.005, 0.01, 0.02, 0.43, 0.44, 0.45]) {
        const result = run(rate, momentum);
        if (result.evaluation.outcome !== "slow-convergence") continue;
        for (let index = 1; index < result.losses.length; index += 1) {
          expect(
            result.losses[index]! - result.losses[index - 1]!,
            `(${rate}, ${momentum}) step ${index}`,
          ).toBeLessThanOrEqual(UPHILL_DELTA);
        }
      }
    }
  });

  it("gives no what-if advice for a run whose dials moved part-way", () => {
    const result = descend(0.2, 0.9);
    const evaluation = evaluate({
      pos: result.final.pos,
      finalLoss: result.finalLoss,
      steps: result.steps,
      diverged: result.diverged,
      settled: result.settled,
      learningRate: 0.2,
      momentum: 0.9,
      overshoots: result.overshoots,
      rises: result.rises,
      minX: result.minX,
      constantDials: false,
    });
    expect(evaluation.outcome).toBe("local-minimum");
    expect(evaluation.advice).toBeNull();
    expect(evaluation.remedy).toBeNull();
  });
});

describe("verdicts without momentum", () => {
  it("a plain step big enough to hop the shallow basin wins, and the copy doesn't credit momentum", () => {
    const { evaluation } = run(0.3, 0);
    expect(evaluation.outcome).toBe("win");
    const card = whyCardFor({ kind: "checked", evaluation });
    expect(card.body).not.toMatch(/enough momentum/i);
    expect(card.body).toMatch(/No momentum at all/);
  });

  it("the plain local minimum says a bigger step would also get out, and names a momentum that does", () => {
    const { evaluation } = run(0.1, 0);
    expect(evaluation.failure!.detail).toMatch(
      /momentum, or a step large enough to hop it, can/,
    );
    expect(evaluation.remedy).toEqual(
      expect.objectContaining({ learningRate: 0.1, momentum: 0.85 }),
    );
  });

  it("a plain step that hopped the basin and bounced back was too BIG, and the copy says so", () => {
    // α ≈ 0.63–0.67 at β = 0 clears the ridge, overshoots the deep valley
    // and rolls back. "Small steps can't leave this basin" was backwards.
    const result = run(0.6421, 0);
    const { evaluation } = result;
    expect(evaluation.outcome).toBe("local-minimum");
    expect(evaluation.crossedRidge).toBe(true);
    expect(evaluation.passedDeepValley).toBe(true);
    expect(evaluation.failure!.detail).toMatch(/Too big a step, not too small/);
    expect(evaluation.failure!.detail).not.toMatch(/Small steps with no momentum/);
    // The fix it names is a smaller step, and it lands.
    expect(evaluation.remedy!.learningRate).toBeLessThan(0.6421);
    expect(evaluation.remedy!.momentum).toBe(0);
    expect(reachesBottom(run(evaluation.remedy!.learningRate, 0).evaluation.outcome)).toBe(true);

    const checked = whyCardFor({ kind: "checked", evaluation });
    expect(checked.body).toMatch(/Too big a step, not too small/);
    expect(checked.body).not.toMatch(/this is the trap momentum exists for/);

    const settled = whyCardFor({
      kind: "stepped",
      loss: result.finalLoss,
      previousLoss: result.losses[result.losses.length - 2]!,
      gradient: gradientAt(result.final.pos.x, result.final.pos.y),
      overshot: false,
      diverged: false,
      settled: true,
      pos: result.final.pos,
      learningRate: 0.6421,
      momentum: 0,
      minX: result.minX,
    });
    expect(settled.body).toMatch(/did hop the ridge/);
    expect(settled.body).toMatch(/too big to stop in the deep valley/);
  });

  it("credits momentum in a win only when the same rate without it gets stuck", () => {
    const { evaluation } = run(0.1, 0.85);
    expect(evaluation.outcome).toBe("win");
    expect(evaluation.withoutMomentum).toBe("local-minimum");
    const card = whyCardFor({ kind: "checked", evaluation });
    expect(card.body).toMatch(/with none, this same rate stops in the shallow valley/);
  });

  it("names a lower rate that settles an oscillating plain descent", () => {
    const { evaluation } = run(0.6, 0);
    expect(evaluation.outcome).toBe("oscillating");
    expect(evaluation.remedy!.learningRate).toBeLessThan(0.6);
    expect(evaluation.remedy!.momentum).toBe(0);
  });
});

describe("store: what the readouts and the verdict describe", () => {
  const store = () => useGradientSkierStore.getState();

  beforeEach(() => {
    useProgression.setState({
      ...EMPTY_PROGRESSION,
      hydrated: true,
      syncError: null,
      lastGain: null,
    });
    useProgression.getState().setAdapter(createMemoryAdapter());
    store().fullReset();
    store().setLane("visual");
  });

  it("previews the next step the rule will actually take", () => {
    store().setMomentum(0.85);
    store().step();
    const before = store().skier;
    const preview = nextStepLength(before);
    store().step();
    const after = store().skier;
    expect(
      Math.hypot(after.pos.x - before.pos.x, after.pos.y - before.pos.y),
    ).toBeCloseTo(preview, 12);
  });

  it("tracks how far left the path got", () => {
    store().setLearningRate(0.2);
    store().setMomentum(0.9);
    store().runToEnd();
    expect(store().minX).toBeCloseTo(descend(0.2, 0.9).minX, 12);
    expect(store().minX).toBeLessThan(RIDGE_X);
  });

  it("quotes the rate that ran, not the dial turned after the run ended", () => {
    store().setLearningRate(1.0);
    store().runToEnd();
    expect(store().diverged).toBe(true);
    store().setLearningRate(0.05);
    const evaluation = store().check();
    expect(evaluation.learningRate).toBe(1.0);
    expect(evaluation.failure!.detail).toMatch(/learning rate of 1\.00/);
    expect(evaluation.failure!.detail).not.toMatch(/0\.0500/);
  });

  it("tells the player a dial turned after the run applies from the top", () => {
    store().setLearningRate(1.0);
    store().runToEnd();
    store().setLearningRate(0.2);
    expect(store().whyCard!.body).toMatch(/applies from the top/);
  });

  it("withholds advice when a dial moved mid-run", () => {
    store().setLearningRate(0.2);
    store().setMomentum(0.9);
    for (let index = 0; index < 5; index += 1) store().step();
    store().setMomentum(0.91);
    store().runToEnd();
    expect(store().dialsChangedMidRun).toBe(true);
    expect(store().check().advice).toBeNull();
  });

  it("records a slow arrival at the bottom as a clear with one star", () => {
    // Spec: "Reach the global minimum before the timer." Taking most of the
    // budget costs score (so the second star), not the clear itself.
    store().setLearningRate(0.05);
    store().setMomentum(0.9);
    store().runToEnd();
    const evaluation = store().check();
    expect(evaluation.outcome).toBe("near-miss");
    expect(evaluation.score).toBeLessThan(HIGH_SCORE_THRESHOLD);
    const record = useProgression.getState().games["gradient-descent-skier"];
    expect(record?.completed).toBe(true);
    expect(record?.stars).toBe(1);
    expect(store().won).toBe(false);
  });

  it("counts a run once however many times it is scored", () => {
    store().setLearningRate(0.1);
    store().setMomentum(0.85);
    store().runToEnd();
    store().check();
    store().check();
    store().check();
    expect(
      useProgression.getState().games["gradient-descent-skier"]?.playCount,
    ).toBe(1);
  });

  it("credits the lane the check came from, not the tab that happens to be open", () => {
    store().setLane("code");
    store().setLearningRate(0.1);
    store().setMomentum(0.85);
    store().runToEnd();
    store().check("visual");
    const record = useProgression.getState().games["gradient-descent-skier"];
    expect(record?.codeLaneCleared).toBe(false);
    expect(record?.stars).toBe(2);
  });
});

describe("code lane api", () => {
  const store = () => useGradientSkierStore.getState();

  beforeEach(() => {
    useProgression.setState({
      ...EMPTY_PROGRESSION,
      hydrated: true,
      syncError: null,
      lastGain: null,
    });
    useProgression.getState().setAdapter(createMemoryAdapter());
    store().fullReset();
  });

  it("drives the same store as the rail", () => {
    const api = createCodeApi();
    api.setLearningRate(0.1);
    api.setMomentum(0.85);
    api.runToEnd();
    expect(api.check().outcome).toBe("win");
    expect(
      useProgression.getState().games["gradient-descent-skier"]?.codeLaneCleared,
    ).toBe(true);
    expect(api.gradient()).toEqual(
      gradientAt(api.position().x, api.position().y),
    );
  });

  it("rejects a rate or momentum that isn't a finite number, by name", () => {
    const api = createCodeApi();
    expect(() => api.setLearningRate(Number.NaN)).toThrow(
      /setLearningRate: the rate must be a finite number/,
    );
    expect(() => api.setLearningRate("0.1" as unknown as number)).toThrow(
      TypeError,
    );
    expect(() => api.setMomentum(Number.POSITIVE_INFINITY)).toThrow(/setMomentum/);
    expect(store().skier.learningRate).toBe(DEFAULT_LEARNING_RATE);
  });

  it("rejects values off the dial instead of silently clamping them", () => {
    const api = createCodeApi();
    expect(() => api.setLearningRate(2)).toThrow(RangeError);
    expect(() => api.setLearningRate(MIN_LEARNING_RATE / 2)).toThrow(
      /outside the dial's range/,
    );
    expect(() => api.setMomentum(1)).toThrow(RangeError);
    expect(() => api.setMomentum(-0.1)).toThrow(RangeError);
    api.setLearningRate(MAX_LEARNING_RATE);
    api.setMomentum(MAX_MOMENTUM);
    expect(store().skier.learningRate).toBe(MAX_LEARNING_RATE);
    expect(store().skier.momentum).toBe(MAX_MOMENTUM);
  });

  it("rejects non-numeric coordinates in the surface probes", () => {
    const api = createCodeApi();
    expect(() => api.lossAt(Number.NaN, 0)).toThrow(
      /lossAt: x must be a finite number/,
    );
    expect(() => api.gradientAt(0, "1" as unknown as number)).toThrow(
      /gradientAt: y/,
    );
    expect(api.lossAt(1, 0)).toBeCloseTo(lossAt(1, 0), 12);
  });
});
