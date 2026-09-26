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
 *   - a ridge between them that small plain steps can't get over — momentum
 *     can, and so can a plain step large enough to hop the shallow basin
 *     without overshooting the deep one (α ≈ 0.3–0.39 at β = 0, measured)
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
/**
 * A step that raises the loss by more than this went uphill, full stop.
 *
 * Smaller than OVERSHOOT_DELTA, which only decides when the readout spikes red.
 * Using that threshold for the verdict as well let a period-2 bounce whose
 * uphill steps were each 0.041 (α = 0.44, β = 0) be told "every step went
 * downhill — the rate is too small", the opposite of the truth. 0.001 is the
 * smallest rise the three-decimal loss readout can show.
 */
export const UPHILL_DELTA = 0.001;
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

/**
 * The top of the ridge between the two valleys: where ∂f/∂x crosses from
 * positive back to negative along y = 0, between the minima. Found the same
 * numeric way as the minima, so it follows the constants if they are retuned.
 * A path whose x ever went below this crossed the ridge.
 */
export const RIDGE_X: number = (() => {
  const low = Math.min(...MINIMA.map((minimum) => minimum.x));
  const high = Math.max(...MINIMA.map((minimum) => minimum.x));
  const steps = 4000;
  let previous = gradientAt(low, 0).x;
  for (let index = 1; index <= steps; index += 1) {
    const x = low + ((high - low) * index) / steps;
    const current = gradientAt(x, 0).x;
    if (previous > 0 && current <= 0) {
      let left = x - (high - low) / steps;
      let right = x;
      for (let iteration = 0; iteration < 80; iteration += 1) {
        const mid = (left + right) / 2;
        if (gradientAt(mid, 0).x > 0) left = mid;
        else right = mid;
      }
      return (left + right) / 2;
    }
    previous = current;
  }
  return (low + high) / 2;
})();

/** ∂²f/∂x² — how sharply the surface curves along x. */
export function curvatureX(x: number): number {
  return 4 * A * (3 * x * x - 1);
}

/**
 * The largest plain-descent rate the deep valley can hold: 2 ÷ its curvature,
 * the textbook stability bound for a bowl (the Concept Library's "steps below
 * 2/L converge"). Across the slope the curvature is 2B, which is gentler, so x
 * is the binding direction. About 0.40 with the shipped constants.
 */
export const STABLE_RATE =
  2 / Math.max(curvatureX(GLOBAL_MINIMUM.x), 2 * B);

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

/**
 * How far the skier's NEXT step will actually travel: |βv − α∇f(θ)|, the
 * velocity the update rule is about to produce, with the gradient taken where
 * the skier stands now. "Slope × rate" is only this when momentum is off — with
 * β = 0.85 it was out by a factor of seven.
 */
export function nextStepLength(skier: Skier): number {
  const gradient = gradientAt(skier.pos.x, skier.pos.y);
  return Math.hypot(
    skier.momentum * skier.velocity.x - skier.learningRate * gradient.x,
    skier.momentum * skier.velocity.y - skier.learningRate * gradient.y,
  );
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
  /** Steps that went uphill by more than UPHILL_DELTA. */
  rises: number;
  /** The furthest left the path reached — below RIDGE_X means it crossed. */
  minX: number;
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
  let rises = 0;
  let minX = skier.pos.x;

  for (let index = 0; index < budget; index += 1) {
    const previousLoss = losses[losses.length - 1]!;
    const result = step(skier);
    steps += 1;
    skier = result.skier;
    path.push({ ...skier.pos });
    losses.push(result.loss);
    minX = Math.min(minX, skier.pos.x);

    if (result.loss - previousLoss > OVERSHOOT_DELTA) overshoots += 1;
    if (result.loss - previousLoss > UPHILL_DELTA) rises += 1;

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
    rises,
    minX,
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

/**
 * A change to the dials that was actually run from the top and actually
 * reaches the deepest valley and stops there. Never asserted, always measured.
 */
export interface Remedy {
  learningRate: number;
  momentum: number;
  /** One sentence naming the change. */
  sentence: string;
}

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
  /** Steps that went uphill by more than UPHILL_DELTA. */
  rises: number;
  /** The dials the run was scored with. */
  learningRate: number;
  momentum: number;
  /** Furthest left the path got, and what that says about the ridge. */
  minX: number;
  crossedRidge: boolean;
  /** Went past the deepest valley's floor (and, if trapped, came back). */
  passedDeepValley: boolean;
  /**
   * What would have worked instead, if a small change does — or an honest
   * sentence saying none of the small changes does. Null when the dials moved
   * mid-run, since a what-if for a run that never happened proves nothing.
   */
  advice: string | null;
  remedy: Remedy | null;
  /**
   * For a momentum run that reached the bottom: what the same rate does with
   * no momentum at all. Null otherwise. Lets the win copy say whether momentum
   * was actually what cleared the ridge.
   */
  withoutMomentum: Outcome | null;
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
  /** Steps that went uphill by more than UPHILL_DELTA. Defaults to `overshoots`. */
  rises?: number;
  /** Furthest left the path reached. Defaults to the final position. */
  minX?: number;
  /**
   * False when the dials were changed part-way through the run. The verdict
   * still stands, but no what-if advice is computed for a run that never
   * happened.
   */
  constantDials?: boolean;
  budget?: number;
}

/**
 * The verdict alone, with no copy. Split out of `evaluate` so the remedy search
 * can classify candidate runs without recursing into more remedy searches.
 */
function classify({
  pos,
  steps,
  diverged,
  settled,
  overshoots,
  rises,
  finalLoss,
  budget,
}: {
  pos: Vec2;
  steps: number;
  diverged: boolean;
  settled: boolean;
  overshoots: number;
  rises: number;
  finalLoss: number;
  budget: number;
}): { outcome: Outcome; progress: number; efficiency: number; score: number } {
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

  const reachedGlobal =
    !diverged && distanceTo(pos, GLOBAL_MINIMUM) <= REACH_TOLERANCE;

  let outcome: Outcome;
  if (diverged) outcome = "diverged";
  else if (settled && !reachedGlobal) outcome = "local-minimum";
  else if (reachedGlobal && settled)
    outcome = score >= WIN_SCORE ? "win" : "near-miss";
  else if (steps >= budget)
    // Out of budget without settling. Two opposite causes, told apart by
    // whether the descent ever went uphill — or, failing that, by whether it
    // is already sitting at the bottom and simply refusing to stop, which is
    // mild oscillation and emphatically not slow convergence.
    outcome =
      overshoots > 0 || rises > 0 || reachedGlobal
        ? "oscillating"
        : "slow-convergence";
  else outcome = "running";

  return { outcome, progress, efficiency, score };
}

/** Run a whole descent from the top and classify it. */
function outcomeFor(learningRate: number, momentum: number): Outcome {
  const run = descend(learningRate, momentum);
  return classify({
    pos: run.final.pos,
    steps: run.steps,
    diverged: run.diverged,
    settled: run.settled,
    overshoots: run.overshoots,
    rises: run.rises,
    finalLoss: run.finalLoss,
    budget: STEP_BUDGET,
  }).outcome;
}

const reachesBottom = (outcome: Outcome) =>
  outcome === "win" || outcome === "near-miss";

/** How momentum values are written everywhere in the copy. */
const beta = (value: number) => value.toFixed(2);
const rate = (value: number) => value.toPrecision(3);
/**
 * A candidate rate, rounded to exactly what its sentence will print. Near the
 * edge of the winning band the descent is sensitive enough that 0.61717…
 * settles and 0.617 bounces, so a remedy has to be run at the number it names.
 */
const shown = (value: number) => Number(rate(value));

/**
 * Try a fixed list of changes to the dials, run each one from the top, and
 * return the first that reaches the deepest valley and stops there. Each run
 * is at most STEP_BUDGET steps of arithmetic, so even the longest list (every
 * momentum on the slider) costs well under a millisecond per candidate.
 *
 * This is what replaced "Halve it and the same descent settles": that sentence
 * was printed for every oscillation, and on this surface halving the rate as
 * advised led to a win in only 5 of 58 oscillating settings — high-momentum
 * oscillation is under-damping, and the lever for it is momentum, not rate.
 */
function findRemedy(
  learningRate: number,
  momentum: number,
  candidates: Array<{ learningRate: number; momentum: number; sentence: string }>,
): Remedy | null {
  for (const candidate of candidates) {
    // Only settings the dials can actually be turned to — a clamped candidate
    // would run one value while the sentence named another.
    if (
      candidate.learningRate < MIN_LEARNING_RATE ||
      candidate.learningRate > MAX_LEARNING_RATE ||
      candidate.momentum < 0 ||
      candidate.momentum > MAX_MOMENTUM
    ) {
      continue;
    }
    if (
      candidate.learningRate === learningRate &&
      Math.abs(candidate.momentum - momentum) < 1e-9
    ) {
      continue;
    }
    if (reachesBottom(outcomeFor(candidate.learningRate, candidate.momentum))) {
      return {
        learningRate: candidate.learningRate,
        momentum: candidate.momentum,
        sentence: candidate.sentence,
      };
    }
  }
  return null;
}

/** Momentum candidates below the current one, nearest first. */
function easedMomenta(momentum: number): number[] {
  return [0.1, 0.2, 0.3]
    .map((drop) => Math.round((momentum - drop) * 100) / 100)
    .filter((value) => value >= 0);
}

/** Every momentum the slider can set (steps of 0.01), strictly above `from`. */
function momentaAbove(from: number): number[] {
  const values: number[] = [];
  const top = Math.round(MAX_MOMENTUM * 100);
  for (let step = Math.floor(from * 100 + 1e-6) + 1; step <= top; step += 1) {
    values.push(step / 100);
  }
  return values;
}

/** Lower rates at the same momentum: a half first, then 90% of it down to 10%. */
function lowerRates(learningRate: number, momentum: number) {
  return [0.5, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1].map((factor) => ({
    learningRate: shown(learningRate * factor),
    momentum,
    sentence:
      factor === 0.5
        ? `Halving the rate to ${rate(learningRate * factor)}`
        : `Lowering the rate to ${rate(learningRate * factor)}`,
  }));
}

const LOWER_FACTORS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
const RAISE_FACTORS = [1.5, 2, 3, 5, 10, 20];

/** Higher rates at the same momentum, the smallest raise first. */
function higherRates(learningRate: number, momentum: number) {
  return RAISE_FACTORS.map((factor) => ({
    learningRate: shown(learningRate * factor),
    momentum,
    sentence: `Raising the rate to ${rate(learningRate * factor)}`,
  }));
}

/**
 * Both dials at once: each rate factor, against momentum eased by 0.05 to 0.3.
 *
 * Lowering both is the obvious pair, and the right one for divergence. Raising
 * the rate while easing momentum is the one nobody guesses, and it is the only
 * fix for a tiny rate at high momentum (0.005 at 0.95, say): easing β shrinks
 * the effective step α/(1−β), so the rate has to rise to make up for it. A
 * sentence that only ever mentioned lowering sent those players the wrong way.
 */
function bothDials(learningRate: number, momentum: number, factors: number[]) {
  return factors.flatMap((factor) =>
    [0.05, 0.1, 0.15, 0.2, 0.3]
      .map((drop) => Math.round((momentum - drop) * 100) / 100)
      .filter((value) => value >= 0)
      .map((value) => ({
        learningRate: shown(learningRate * factor),
        momentum: value,
        sentence: `${factor > 1 ? "Raising" : "Lowering"} the rate to ${rate(
          learningRate * factor,
        )} and easing momentum to ${beta(value)}`,
      })),
  );
}

/**
 * What a fallback says when nothing tried worked. It names no direction for
 * either dial: the changes tried above all failed, and the one that works may
 * lie either way.
 */
const TRY_OTHERS = "try other combinations of both dials.";

const plural = (count: number, noun: string) =>
  `${count} ${noun}${count === 1 ? "" : "s"}`;

/**
 * Score the run and name the failure — honestly.
 *
 * Divergence is checked first because once the loss has exploded every other
 * number is meaningless. "Local minimum" requires the skier to have actually
 * SETTLED in one: a skier still moving through the shallow valley hasn't been
 * trapped yet, it's just passing through, and calling that a local minimum would
 * teach the wrong word.
 *
 * The copy knows what momentum did. The first version was written as if β were
 * always 0, so it blamed the learning rate for divergence that momentum caused,
 * told a skier who had overshot the deep valley and rolled back that it lacked
 * momentum, and advised halving the rate for under-damped oscillation where
 * halving doesn't help. Now the path's own facts (how far left it got, whether
 * it crossed the ridge) pick the sentence, and any advice is a change that was
 * actually run from the top and actually works.
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
  rises = overshoots,
  minX = pos.x,
  constantDials = true,
  budget = STEP_BUDGET,
}: EvaluateInput): Evaluation {
  const globalLoss = GLOBAL_MINIMUM.loss;
  const { outcome, progress, efficiency, score } = classify({
    pos,
    steps,
    diverged,
    settled,
    overshoots,
    rises,
    finalLoss,
    budget,
  });

  const distanceToGlobal = distanceTo(pos, GLOBAL_MINIMUM);
  const reachedGlobal = !diverged && distanceToGlobal <= REACH_TOLERANCE;
  const crossedRidge = minX < RIDGE_X;
  const passedDeepValley = minX < GLOBAL_MINIMUM.x - REACH_TOLERANCE;

  const round2 = (value: number) => value.toFixed(2);
  const effectiveStep = momentum < 1 ? learningRate / (1 - momentum) : Infinity;
  const settles = "reaches the deepest valley and stops there.";

  // Candidate changes, in the order a practitioner would try them.
  const [halve, ...otherRates] = lowerRates(learningRate, momentum);
  const eased = easedMomenta(momentum).map((value) => ({
    learningRate,
    momentum: value,
    sentence: `Easing momentum to ${beta(value)} at the same rate`,
  }));
  // How far momentum was actually eased — under 0.3 when β is small — so a
  // fallback never claims a try that the dial's floor ruled out.
  const easedBy =
    eased.length > 0 ? beta(momentum - eased[eased.length - 1]!.momentum) : null;

  let failure: NamedFailure | null = null;
  let remedy: Remedy | null = null;
  let advice: string | null = null;
  let withoutMomentum: Outcome | null = null;

  const resolve = (found: Remedy | null, none: string) => ({
    remedy: found,
    advice: found ? `${found.sentence} ${settles}` : none,
  });

  if (outcome === "diverged") {
    if (constantDials) {
      ({ remedy, advice } = resolve(
        findRemedy(learningRate, momentum, [
          halve!,
          ...eased,
          ...otherRates,
          ...bothDials(learningRate, momentum, LOWER_FACTORS),
        ]),
        easedBy
          ? `Lowering the rate (to anywhere from 90% down to 10% of it, as far as the dial goes), easing momentum by up to ${easedBy}, and both together were each run from the top, and none lands it — ${TRY_OTHERS}`
          : `No lower rate on its own (tried from 90% down to 10% of this one, as far as the dial goes) reaches the deepest valley from here — ${TRY_OTHERS}`,
      ));
    }
    const lost = `The loss went to ${
      Number.isFinite(finalLoss) ? round2(finalLoss) : "infinity"
    } in ${plural(steps, "step")} and the skier left the mountain.`;
    failure = {
      name: "Divergence",
      detail:
        momentum > 0
          ? `A learning rate of ${rate(learningRate)} with momentum ${beta(
              momentum,
            )}: under a steady slope the momentum builds each step toward α/(1−β) ≈ ${rate(
              effectiveStep,
            )} times the slope, and that — not ${rate(
              learningRate,
            )} alone — is the step size the curvature had to tolerate. It didn't. ${lost}${
              advice ? ` ${advice}` : ""
            }`
          : `A learning rate of ${rate(
              learningRate,
            )} made each step too long for the slope it was measured on, so every step overshot further than the last. ${lost} With no momentum, even the deep valley only holds a rate below 2 ÷ its curvature ≈ ${round2(
              STABLE_RATE,
            )}. Nothing about the surface is wrong — the step size is.${
              advice ? ` ${advice}` : ""
            }`,
    };
  } else if (outcome === "local-minimum") {
    const nearest = nearestMinimum(pos);
    const where = `Settled at loss ${round2(finalLoss)} after ${steps} steps, but the deepest valley is at ${round2(
      globalLoss,
    )} — you're in the shallow one at x ${round2(nearest.minimum.x)}.`;

    let cause: string;
    if (momentum === 0 && crossedRidge) {
      // A plain step that DID hop the basin, and was too big to stop in the
      // deep valley (α ≈ 0.63–0.67): the opposite of "small steps can't leave".
      if (constantDials) {
        ({ remedy, advice } = resolve(
          findRemedy(learningRate, momentum, [halve!, ...otherRates]),
          `No lower rate on its own (tried from 90% down to 10% of this one) stops in the deepest valley from here — ${TRY_OTHERS}`,
        ));
      }
      cause = ` The gradient here really is zero, but this step was not too small: the path hopped the ridge and reached x ${round2(
        minX,
      )}${
        passedDeepValley
          ? `, right past the deep valley's floor at x ${round2(GLOBAL_MINIMUM.x)},`
          : ""
      } and a step that long bounced it back into the shallow one. Too big a step, not too small.`;
    } else if (momentum === 0) {
      if (constantDials) {
        ({ remedy, advice } = resolve(
          findRemedy(
            learningRate,
            momentum,
            // The lesson's own setting first, then every slider value.
            [0.85, ...momentaAbove(0)].map((value) => ({
              learningRate,
              momentum: value,
              sentence: `At this same rate, momentum ${beta(value)}`,
            })),
          ),
          "No momentum setting on the dial gets this rate over the ridge on its own — change the rate too.",
        ));
      }
      cause = ` The gradient here really is zero, so plain descent has no reason to move. Small steps with no momentum can't leave this basin; momentum, or a step large enough to hop it, can.`;
    } else if (crossedRidge) {
      if (constantDials) {
        ({ remedy, advice } = resolve(
          findRemedy(learningRate, momentum, [
            ...eased,
            halve!,
            ...otherRates,
            ...bothDials(learningRate, momentum, LOWER_FACTORS),
          ]),
          easedBy
            ? `Easing momentum by up to ${easedBy}, lowering the rate (to anywhere from 90% down to 10% of it, as far as the dial goes), and both together were each run from the top, and none lands it — ${TRY_OTHERS}`
            : `No lower rate on its own (tried from 90% down to 10% of this one, as far as the dial goes) lands it from here — ${TRY_OTHERS}`,
        ));
      }
      cause = passedDeepValley
        ? ` You did get over the ridge: the path reached x ${round2(
            minX,
          )}, right past the deep valley's floor at x ${round2(
            GLOBAL_MINIMUM.x,
          )}, and momentum ${beta(
            momentum,
          )} carried you back over the ridge into the shallow one. That's too much momentum, not too little.`
        : ` You did get over the ridge — the path reached x ${round2(
            minX,
          )} — but momentum ${beta(
            momentum,
          )} carried too much speed to stop in the deep valley, and rolled you back. That's too much momentum, not too little.`;
    } else {
      if (constantDials) {
        const more = momentaAbove(momentum).map((value) => ({
            learningRate,
            momentum: value,
            sentence: `Momentum ${beta(value)} at the same rate`,
          }));
        // More momentum first — it's what the cause below says was missing —
        // then a bigger rate at this momentum, which builds speed the same way.
        const tried =
          more.length > 0
            ? `No momentum setting between ${beta(momentum)} and ${beta(
                MAX_MOMENTUM,
              )} at this rate, and no higher rate (up to 20× this one, as far as the dial goes) at this momentum,`
            : `Momentum is already at the top of the dial, and no higher rate (up to 20× this one, as far as the dial goes) at ${beta(
                momentum,
              )}`;
        ({ remedy, advice } = resolve(
          findRemedy(learningRate, momentum, [
            ...more,
            ...higherRates(learningRate, momentum),
          ]),
          `${tried} reaches the deepest valley and stops there — ${TRY_OTHERS}`,
        ));
      }
      cause = ` The gradient here really is zero, and momentum ${beta(
        momentum,
      )} wasn't enough to carry you over the ridge at x ${round2(
        RIDGE_X,
      )}: the furthest you got was x ${round2(minX)}.`;
    }

    failure = {
      name: "Local minimum",
      detail: `${where}${cause}${advice ? ` ${advice}` : ""}`,
    };
  } else if (outcome === "oscillating") {
    if (constantDials) {
      // The lowering candidates first, as before; then the raises. A tiny rate
      // at high momentum oscillates too, and only a bigger rate — usually with
      // less momentum — settles it.
      ({ remedy, advice } = resolve(
        findRemedy(learningRate, momentum, [
          halve!,
          ...eased,
          ...otherRates,
          ...higherRates(learningRate, momentum),
          ...bothDials(learningRate, momentum, RAISE_FACTORS),
          ...bothDials(learningRate, momentum, LOWER_FACTORS),
        ]),
        easedBy
          ? `Lowering or raising the rate (from a tenth of it to 20×, as far as the dial goes), easing momentum by up to ${easedBy}, and both together were each run from the top, and none settles it — ${TRY_OTHERS}`
          : `No other rate on its own (tried from a tenth of this one to 20×, as far as the dial goes) settles it here — ${TRY_OTHERS}`,
      ));
    }
    // A path that never left the right-hand basin was rocking in the shallow
    // valley, not the deep one; say which floor it kept crossing.
    const floor =
      !reachedGlobal && !crossedRidge ? "the shallow valley's floor" : "the valley floor";
    const why =
      momentum > 0
        ? `A rate of ${rate(learningRate)} with momentum ${beta(
            momentum,
          )} carries each step past ${floor} and back instead of coming to rest — with momentum, too much speed is its own kind of overshoot.`
        : `A learning rate of ${rate(
            learningRate,
          )} steps past ${floor} and back again instead of coming to rest.`;

    failure = {
      name: "Oscillation",
      detail: reachedGlobal
        ? `You're at the bottom — ${round2(
            distanceToGlobal,
          )} from it — after all ${plural(steps, "step")}, and still moving. ${why}${
            advice ? ` ${advice}` : ""
          }`
        : `${plural(steps, "step")} and never settled: ${rises} of them went UPHILL. ${why} The loss is ${round2(
            finalLoss,
          )} and bouncing rather than falling.${advice ? ` ${advice}` : ""}`,
    };
  } else if (outcome === "slow-convergence") {
    failure = {
      name: "Slow convergence",
      detail: `${steps} steps used, loss down to ${round2(
        finalLoss,
      )} and still ${round2(
        distanceToGlobal,
      )} from the bottom — every step went downhill, just not far. A learning rate of ${rate(
        learningRate,
      )} is too small to cover the distance in the budget. Nothing diverged and nothing is stuck; you ran out of iterations.`,
    };
  } else if (
    (outcome === "win" || outcome === "near-miss") &&
    momentum > 0 &&
    constantDials
  ) {
    withoutMomentum = outcomeFor(learningRate, 0);
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
    rises,
    learningRate,
    momentum,
    minX,
    crossedRidge,
    passedDeepValley,
    advice,
    remedy,
    withoutMomentum,
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
