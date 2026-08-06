import type { NamedFailure } from "@/engine/types";
import { clamp } from "@/lib/utils";

/**
 * Gradient Descent Skier — the real machine learning.
 *
 * The gradient is analytic, not estimated: `gradientAt` returns the exact partial
 * derivatives of the surface below. Each step the skier takes is a textbook
 * heavy-ball update, which makes the pedagogy contract's second point literal —
 * the dial the player turns IS the α in the update rule, and nothing else.
 *
 * ── The surface ─────────────────────────────────────────────────────────────
 *
 *   f(x, y) = A(x² − 1)² + Cx + By²
 *
 * A tilted double well. The (x² − 1)² term makes two valleys; the Cx tilt makes
 * one of them deeper. So the surface has exactly what the lesson needs:
 *
 *   - a global minimum (the deep valley, on the left)
 *   - a local minimum (the shallow valley, on the right, where you start)
 *   - a ridge between them that only momentum gets you over
 *   - curvature that grows like x³, so an over-large step really does explode
 *
 * Every failure mode in this game is a property of that surface plus the update
 * rule, not a scripted event.
 */

/** Depth of the wells. Lower A means a lower ridge between them. */
export const A = 0.5;
/** Tilt. Makes the left well deeper than the right one. */
export const C = 0.35;
/** Curvature across the slope. */
export const B = 0.6;

/** Playable domain. Leaving it counts as flying off the mountain. */
export const DOMAIN = { minX: -2.4, maxX: 2.4, minY: -1.8, maxY: 1.8 };

/** Where the skier starts: on the right slope, above the shallow valley. */
export const START = { x: 1.55, y: 0.95 };

/** Gradient steps available. See the docblock on `STEP_BUDGET` below. */
export const STEP_BUDGET = 120;

export const MIN_LEARNING_RATE = 0.001;
export const MAX_LEARNING_RATE = 1.5;
export const DEFAULT_LEARNING_RATE = 0.05;
export const MAX_MOMENTUM = 0.95;

/** Loss above this, or leaving the domain, is divergence. */
export const DIVERGENCE_LOSS = 40;
/**
 * A step that raises the loss by more than this is an overshoot.
 *
 * Counting overshoots is what separates two failures that look identical from the
 * outside — both run out of budget without settling — but have opposite causes:
 * a step size too SMALL descends smoothly and simply runs out of iterations,
 * while a step size too LARGE bounces across the valley, going uphill on every
 * other step. Diagnosing those the same way would teach the player to fix the
 * wrong dial.
 */
export const OVERSHOOT_DELTA = 0.05;
/** Gradient norm below this counts as settled. */
export const SETTLED_GRADIENT = 0.02;
/** How close to a minimum counts as reaching it. */
export const REACH_TOLERANCE = 0.18;

export const WIN_SCORE = 0.8;
/** Fraction of the score that using the whole budget can cost. */
export const EFFICIENCY_WEIGHT = 0.3;

export interface Vec2 {
  x: number;
  y: number;
}

/** Loss at a point. The height of the mountain. */
export function lossAt(x: number, y: number): number {
  const well = x * x - 1;
  return A * well * well + C * x + B * y * y;
}

/**
 * The exact gradient — ∂f/∂x and ∂f/∂y, differentiated by hand.
 *
 *   ∂f/∂x = 4Ax(x² − 1) + C
 *   ∂f/∂y = 2By
 *
 * Analytic rather than a finite difference, so the number the game shows is the
 * true slope and not an approximation of one.
 */
export function gradientAt(x: number, y: number): Vec2 {
  return {
    x: 4 * A * x * (x * x - 1) + C,
    y: 2 * B * y,
  };
}

export function gradientNorm(gradient: Vec2): number {
  return Math.hypot(gradient.x, gradient.y);
}

// ── The minima ─────────────────────────────────────────────────────────────

export interface Minimum {
  x: number;
  y: number;
  loss: number;
  kind: "global" | "local";
}

/**
 * Find the valleys by scanning ∂f/∂x for sign changes along y = 0, then
 * bisecting. Numeric rather than solving the cubic in closed form: it stays
 * correct if the constants above are retuned, which a hand-derived root would
 * not.
 */
function findMinima(): Minimum[] {
  const found: Array<{ x: number }> = [];
  const steps = 4000;

  let previous = gradientAt(DOMAIN.minX, 0).x;
  for (let index = 1; index <= steps; index += 1) {
    const x = DOMAIN.minX + ((DOMAIN.maxX - DOMAIN.minX) * index) / steps;
    const current = gradientAt(x, 0).x;

    // A minimum is where the derivative crosses from negative to positive.
    if (previous < 0 && current >= 0) {
      let low = x - (DOMAIN.maxX - DOMAIN.minX) / steps;
      let high = x;
      for (let iteration = 0; iteration < 80; iteration += 1) {
        const mid = (low + high) / 2;
        if (gradientAt(mid, 0).x < 0) low = mid;
        else high = mid;
      }
      found.push({ x: (low + high) / 2 });
    }
    previous = current;
  }

  const withLoss = found.map((point) => ({
    x: point.x,
    y: 0,
    loss: lossAt(point.x, 0),
  }));
  const deepest = Math.min(...withLoss.map((point) => point.loss));

  return withLoss.map((point) => ({
    ...point,
    kind: point.loss === deepest ? ("global" as const) : ("local" as const),
  }));
}

/** The spec's `Surface.minima[]`. Computed once — the surface never changes. */
export const MINIMA: readonly Minimum[] = findMinima();

export const GLOBAL_MINIMUM: Minimum =
  MINIMA.find((minimum) => minimum.kind === "global") ??
  ({ x: 0, y: 0, loss: lossAt(0, 0), kind: "global" } as Minimum);

export const LOCAL_MINIMA: readonly Minimum[] = MINIMA.filter(
  (minimum) => minimum.kind === "local",
);

export const START_LOSS = lossAt(START.x, START.y);

// ── The update rule ────────────────────────────────────────────────────────

export interface Skier {
  pos: Vec2;
  velocity: Vec2;
  learningRate: number;
  momentum: number;
}

export interface StepResult {
  skier: Skier;
  loss: number;
  gradient: Vec2;
  /** True when this step left the playable surface or blew the loss up. */
  diverged: boolean;
  /** Distance travelled by this step. */
  stepLength: number;
}

/**
 * One gradient-descent step, with heavy-ball momentum:
 *
 *   v ← βv − α∇f(θ)
 *   θ ← θ + v
 *
 * At β = 0 this is plain gradient descent. That is the whole algorithm; the dials
 * are α and β and there is nothing else in the loop.
 */
export function step(skier: Skier): StepResult {
  const gradient = gradientAt(skier.pos.x, skier.pos.y);

  const velocity = {
    x: skier.momentum * skier.velocity.x - skier.learningRate * gradient.x,
    y: skier.momentum * skier.velocity.y - skier.learningRate * gradient.y,
  };

  const pos = { x: skier.pos.x + velocity.x, y: skier.pos.y + velocity.y };
  const loss = lossAt(pos.x, pos.y);

  const outOfBounds =
    pos.x < DOMAIN.minX ||
    pos.x > DOMAIN.maxX ||
    pos.y < DOMAIN.minY ||
    pos.y > DOMAIN.maxY;

  const diverged =
    outOfBounds ||
    !Number.isFinite(loss) ||
    loss > DIVERGENCE_LOSS ||
    !Number.isFinite(pos.x) ||
    !Number.isFinite(pos.y);

  return {
    skier: { ...skier, pos, velocity },
    loss,
    gradient,
    diverged,
    stepLength: Math.hypot(velocity.x, velocity.y),
  };
}

export function makeSkier(
  learningRate = DEFAULT_LEARNING_RATE,
  momentum = 0,
): Skier {
  return {
    pos: { ...START },
    velocity: { x: 0, y: 0 },
    learningRate,
    momentum,
  };
}

/** Distance from a point to a minimum. */
export function distanceTo(pos: Vec2, minimum: Minimum): number {
  return Math.hypot(pos.x - minimum.x, pos.y - minimum.y);
}

export function nearestMinimum(pos: Vec2): { minimum: Minimum; distance: number } {
  let best = MINIMA[0]!;
  let bestDistance = distanceTo(pos, best);
  for (const minimum of MINIMA) {
    const distance = distanceTo(pos, minimum);
    if (distance < bestDistance) {
      best = minimum;
      bestDistance = distance;
    }
  }
  return { minimum: best, distance: bestDistance };
}

/**
 * Run the whole descent from a fresh skier. Used by the code lane, and by tests
 * to assert what each learning rate actually does.
 */
export function descend(
  learningRate: number,
  momentum = 0,
  budget = STEP_BUDGET,
): {
  path: Vec2[];
  losses: number[];
  steps: number;
  diverged: boolean;
  settled: boolean;
  overshoots: number;
  final: Skier;
  finalLoss: number;
} {
  let skier = makeSkier(learningRate, momentum);
  const path: Vec2[] = [{ ...skier.pos }];
  const losses: number[] = [lossAt(skier.pos.x, skier.pos.y)];

  let diverged = false;
  let settled = false;
  let steps = 0;
  let overshoots = 0;

  for (let index = 0; index < budget; index += 1) {
    const previousLoss = losses[losses.length - 1]!;
    const result = step(skier);
    steps += 1;
    skier = result.skier;
    path.push({ ...skier.pos });
    losses.push(result.loss);

    if (result.loss - previousLoss > OVERSHOOT_DELTA) overshoots += 1;

    if (result.diverged) {
      diverged = true;
      break;
    }

    // Settled means the slope is flat AND we've stopped moving. Checking the
    // gradient alone would call the top of the ridge "settled".
    if (
      gradientNorm(result.gradient) < SETTLED_GRADIENT &&
      result.stepLength < SETTLED_GRADIENT
    ) {
      settled = true;
      break;
    }
  }

  return {
    path,
    losses,
    steps,
    diverged,
    settled,
    overshoots,
    final: skier,
    finalLoss: losses[losses.length - 1]!,
  };
}

// ── Evaluation ─────────────────────────────────────────────────────────────

export type Outcome =
  | "win"
  | "diverged"
  | "local-minimum"
  | "oscillating"
  | "slow-convergence"
  | "near-miss"
  | "running";

export interface Evaluation {
  finalLoss: number;
  globalLoss: number;
  /** Fraction of the available descent actually achieved, 0–1. */
  progress: number;
  steps: number;
  /** Penalty factor for using the budget up. */
  efficiency: number;
  score: number;
  distanceToGlobal: number;
  reachedGlobal: boolean;
  overshoots: number;
  outcome: Outcome;
  failure: NamedFailure | null;
}

export interface EvaluateInput {
  pos: Vec2;
  finalLoss: number;
  steps: number;
  diverged: boolean;
  settled: boolean;
  learningRate: number;
  momentum: number;
  /** Steps that went uphill by more than OVERSHOOT_DELTA. */
  overshoots: number;
  budget?: number;
}

/**
 * Score the run and name the failure — honestly.
 *
 * Divergence is checked first because once the loss has exploded every other
 * number is meaningless. "Local minimum" requires the skier to have actually
 * SETTLED in one: a skier still moving through the shallow valley hasn't been
 * trapped yet, it's just passing through, and calling that a local minimum would
 * teach the wrong word.
 */
export function evaluate({
  pos,
  finalLoss,
  steps,
  diverged,
  settled,
  learningRate,
  momentum,
  overshoots,
  budget = STEP_BUDGET,
}: EvaluateInput): Evaluation {
  const globalLoss = GLOBAL_MINIMUM.loss;
  const available = START_LOSS - globalLoss;
  const progress = diverged
    ? 0
    : clamp(available > 0 ? (START_LOSS - finalLoss) / available : 0, 0, 1);

  const efficiency = clamp(
    1 - EFFICIENCY_WEIGHT * (steps / Math.max(1, budget)),
    1 - EFFICIENCY_WEIGHT,
    1,
  );
  const score = clamp(progress * efficiency, 0, 1);

  const distanceToGlobal = distanceTo(pos, GLOBAL_MINIMUM);
  const reachedGlobal = !diverged && distanceToGlobal <= REACH_TOLERANCE;

  const round2 = (value: number) => value.toFixed(2);
  const trapped = settled && !reachedGlobal;

  let outcome: Outcome;
  let failure: NamedFailure | null = null;

  if (diverged) {
    outcome = "diverged";
    failure = {
      name: "Divergence",
      detail: `A learning rate of ${learningRate.toPrecision(
        3,
      )} made each step longer than the slope it was measured on, so every step overshot further than the last. The loss went to ${
        Number.isFinite(finalLoss) ? round2(finalLoss) : "infinity"
      } in ${steps} steps and the skier left the mountain. Nothing about the surface is wrong — the step size is.`,
    };
  } else if (trapped) {
    const nearest = nearestMinimum(pos);
    outcome = "local-minimum";
    failure = {
      name: "Local minimum",
      detail: `Settled at loss ${round2(finalLoss)} after ${steps} steps, but the deepest valley is at ${round2(
        globalLoss,
      )} — you're in the shallow one at x ${round2(nearest.minimum.x)}. The gradient here really is zero, so plain descent has no reason to move${
        momentum > 0
          ? `, and momentum ${momentum.toFixed(2)} wasn't enough to carry you over the ridge.`
          : `. Momentum carries speed across a ridge that the gradient alone won't.`
      }`,
    };
  } else if (reachedGlobal && settled) {
    // Arrived. Whether it counts depends on how many steps it took.
    outcome = score >= WIN_SCORE ? "win" : "near-miss";
  } else if (steps >= budget) {
    // Out of budget without settling. Two opposite causes, told apart by whether
    // the descent went uphill on the way — or, failing that, by whether it is
    // already sitting at the bottom and simply refusing to stop, which is mild
    // oscillation and emphatically not slow convergence.
    if (overshoots > 0 || reachedGlobal) {
      outcome = "oscillating";
      failure = {
        name: "Oscillation",
        detail: reachedGlobal
          ? `You're at the bottom — ${round2(
              distanceToGlobal,
            )} from it — after all ${steps} steps, and still moving. A learning rate of ${learningRate.toPrecision(
              3,
            )} steps past the valley floor and back again instead of coming to rest. Halve it and the same descent settles.`
          : `${steps} steps and never settled: ${overshoots} of them went UPHILL. A learning rate of ${learningRate.toPrecision(
              3,
            )} overshoots the valley floor and lands on the opposite slope, then overshoots back. The loss is ${round2(
              finalLoss,
            )} and bouncing rather than falling — this is a step size just below the one that would diverge outright.`,
      };
    } else {
      outcome = "slow-convergence";
      failure = {
        name: "Slow convergence",
        detail: `${steps} steps used, loss down to ${round2(
          finalLoss,
        )} and still ${round2(
          distanceToGlobal,
        )} from the bottom — every step went downhill, just not far. A learning rate of ${learningRate.toPrecision(
          3,
        )} is too small to cover the distance in the budget. Nothing diverged and nothing is stuck; you ran out of iterations.`,
      };
    }
  } else {
    outcome = "running";
  }

  return {
    finalLoss,
    globalLoss,
    progress,
    steps,
    efficiency,
    score,
    distanceToGlobal,
    reachedGlobal,
    overshoots,
    outcome,
    failure,
  };
}

// ── Reveal the math ────────────────────────────────────────────────────────

export const MATH_EQUATION = String.raw`f(x, y) = A\left(x^2 - 1\right)^2 + Cx + By^2
\qquad
\nabla f = \begin{bmatrix} 4Ax\left(x^2-1\right) + C \\ 2By \end{bmatrix}
\\[1.2em]
\textbf{update:}\quad v \leftarrow \beta v - \alpha \nabla f(\theta)
\qquad
\theta \leftarrow \theta + v`;

export const MATH_CODE = `// The exact gradient, differentiated by hand — not estimated.
export function gradientAt(x, y) {
  return {
    x: 4 * A * x * (x * x - 1) + C,
    y: 2 * B * y,
  };
}

// One step. This is the entire algorithm.
export function step(skier) {
  const gradient = gradientAt(skier.pos.x, skier.pos.y);

  // v <- Bv - a*grad     (momentum, then the gradient step)
  const velocity = {
    x: skier.momentum * skier.velocity.x - skier.learningRate * gradient.x,
    y: skier.momentum * skier.velocity.y - skier.learningRate * gradient.y,
  };

  // theta <- theta + v
  const pos = { x: skier.pos.x + velocity.x, y: skier.pos.y + velocity.y };

  return { pos, velocity, loss: lossAt(pos.x, pos.y) };
}`;

export const MATH_NOTES = `α is the learning-rate dial and β is the momentum dial — there is nothing else in the loop. The surface is a tilted double well: (x²−1)² makes two valleys and Cx tips one deeper, so the bottom you can see from the start is not the bottom of the mountain. Because ∂f/∂x grows like x³, an α that is comfortable near a valley is explosive on a steep slope — which is why divergence is a property of step size and curvature together, not of a bad surface. With A=${A}, C=${C}, B=${B} the deep valley sits at loss ${GLOBAL_MINIMUM.loss.toFixed(
  2,
)} and the shallow one at ${(LOCAL_MINIMA[0]?.loss ?? 0).toFixed(2)}.`;
