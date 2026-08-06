import { beforeEach, describe, expect, it } from "vitest";
import {
  EMPTY_PROGRESSION,
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
  START,
  START_LOSS,
  STEP_BUDGET,
  WIN_SCORE,
  descend,
  distanceTo,
  evaluate,
  gradientAt,
  gradientNorm,
  lossAt,
  makeSkier,
  nearestMinimum,
  step,
} from "./ml";
import { useGradientSkierStore } from "./store";
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
    }),
  };
};

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
    expect(state.gradient).toEqual(
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

  it("credits the code lane for the third star", () => {
    useGradientSkierStore.getState().setLane("code");
    useGradientSkierStore.getState().setLearningRate(0.1);
    useGradientSkierStore.getState().setMomentum(0.85);
    useGradientSkierStore.getState().runToEnd();
    useGradientSkierStore.getState().check();

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
    });

    expect(inGlobal.tone).toBe("good");
    expect(inLocal.tone).toBe("warn");
    expect(inLocal.body).toMatch(/shallow/i);
  });
});
