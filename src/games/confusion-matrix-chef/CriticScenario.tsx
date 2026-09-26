"use client";

import { Check, X } from "lucide-react";
import {
  METRIC_LABELS,
  SCENARIOS,
  majorityBaseline,
  type ConstraintResult,
  type Scenario,
} from "./ml";

export interface CriticScenarioProps {
  scenario: Scenario;
  /** Live constraint status at the current cutoff. */
  constraints: ConstraintResult[];
  attempts: number;
}

/**
 * The critic's brief (spec: `<CriticScenario>`).
 *
 * Shows the floors live rather than only after serving, which is deliberate: the
 * player should be able to see themselves cross a floor while sliding. Hiding it
 * until the round is scored would turn a lesson about a tradeoff into a guessing
 * game about a hidden number.
 *
 * What it does NOT show is the verdict — whether the whole brief passes is only
 * settled when the cutoff is committed, because deciding you are done is part of
 * the skill.
 */
export function CriticScenario({
  scenario,
  constraints,
  attempts,
}: CriticScenarioProps) {
  const baseline = majorityBaseline(scenario.prevalence);

  return (
    <section aria-labelledby="critic-heading">
      <h2
        id="critic-heading"
        className="flex flex-wrap items-baseline justify-between gap-x-2 gap-y-1"
      >
        <span className="font-display text-sm font-semibold">
          Shift {scenario.index} of {SCENARIOS.length}: {scenario.name}
        </span>
        <span className="text-xs font-normal text-text-muted">
          {Math.round(scenario.prevalence * 100)}% are{" "}
          {scenario.positiveLabelPlural}
        </span>
      </h2>

      <blockquote className="mt-1.5 border-l-2 border-primary/60 pl-2.5 text-xs text-text">
        {scenario.brief}
      </blockquote>

      <dl className="mt-2 grid grid-cols-1 gap-1 text-[11px] sm:grid-cols-2">
        <div className="rounded border border-border bg-surface-2 p-1.5">
          <dt className="font-medium">A false alarm costs</dt>
          <dd className="text-text-muted">{scenario.falsePositiveCost}</dd>
        </div>
        <div className="rounded border border-border bg-surface-2 p-1.5">
          <dt className="font-medium">A miss costs</dt>
          <dd className="text-text-muted">{scenario.falseNegativeCost}</dd>
        </div>
      </dl>

      <ul className="mt-2 flex flex-col gap-1">
        {constraints.map((constraint) => (
          <li
            key={constraint.metric}
            className="flex items-center gap-2 text-xs"
          >
            {constraint.met ? (
              <Check aria-hidden="true" className="size-3.5 shrink-0 text-correct" />
            ) : (
              <X aria-hidden="true" className="size-3.5 shrink-0 text-wrong" />
            )}
            <span>
              {METRIC_LABELS[constraint.metric]} ≥{" "}
              {Math.round(constraint.floor * 100)}%
            </span>
            <span className="ml-auto font-mono">
              {(constraint.achieved * 100).toFixed(1)}%
            </span>
            {/* Colour and icon both carry the state; this carries it in words. */}
            <span className="sr-only">
              {constraint.met ? "met" : "not met yet"}
            </span>
          </li>
        ))}
      </ul>

      <p className="mt-2 text-[11px] text-text-muted">
        {baseline >= 0.8
          ? `Ignoring the scores entirely would score ${Math.round(
              baseline * 100,
            )}% accuracy on this shift. Accuracy is not the scoreboard here.`
          : "Classes are near balanced, so accuracy is roughly honest — it still cannot say which mistake you are making."}
        {attempts > 0
          ? ` ${attempts} cutoff${attempts === 1 ? "" : "s"} served so far.`
          : ""}
      </p>
    </section>
  );
}
