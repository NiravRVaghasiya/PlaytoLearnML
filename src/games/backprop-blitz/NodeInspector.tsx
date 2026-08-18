"use client";

import { Check, X } from "lucide-react";
import { Button } from "@/components";
import { cx } from "@/lib/utils";
import {
  STEPS,
  STEP_COUNT,
  consumersOf,
  nodeById,
  type Rule,
  type Step,
  type Values,
} from "./ml";

export interface NodeInspectorProps {
  step: Step;
  stepIndex: number;
  /** The rule chosen at this step, if any. */
  chosen: string | null;
  /** Forward values, for showing the local derivative concretely. */
  values: Values;
  /** The gradient arriving at this node, if it has settled. */
  incoming: number | null;
  /** Contributions waiting to be combined, for an accumulate step. */
  pending: number[];
  /** What the player's current choice sends onward. */
  outgoing: Array<{ to: string; mine: number | null; truth: number; correct: boolean }>;
  answeredCount: number;
  onChoose: (ruleId: string) => void;
  onClear: () => void;
  onStep: (index: number) => void;
}

const signed = (value: number) =>
  `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;

/**
 * The node being routed (spec: `<NodeInspector>`).
 *
 * This is where the player actually plays. It shows what arrived, what the node's
 * inputs were during the forward pass, and the candidate rules — and then, once a
 * rule is picked, the numbers it sends onward with a tick or a cross against each.
 *
 * The options are deliberately phrased as RULES rather than as numbers. "Each input
 * gets g times the other input" is a thing you can carry to a graph you have never
 * seen; "4.5" is not. Getting the arithmetic right by picking the plausible-looking
 * figure would teach nothing, so the arithmetic is done for the player and the
 * choice is the part that requires understanding.
 */
export function NodeInspector({
  step,
  stepIndex,
  chosen,
  values,
  incoming,
  pending,
  outgoing,
  answeredCount,
  onChoose,
  onClear,
  onStep,
}: NodeInspectorProps) {
  const node = nodeById(step.nodeId);
  const chosenRule = step.options.find((option) => option.id === chosen);

  return (
    <section aria-labelledby="inspector-heading" className="flex flex-col gap-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 id="inspector-heading" className="text-sm font-semibold">
          Step {stepIndex + 1} of {STEP_COUNT} · node {node.label}
        </h3>
        <span className="text-xs text-text-muted">
          {answeredCount} routed
        </span>
      </div>

      <div className="flex gap-1">
        <Button
          variant="ghost"
          size="sm"
          disabled={stepIndex === 0}
          onClick={() => onStep(stepIndex - 1)}
        >
          Back
        </Button>
        {/* Promoted once this step is answered: the player has seen the result and
            the next node is what they want. Before that it stays quiet so the
            obvious move is to read the numbers rather than to skip ahead. */}
        <Button
          variant={chosen !== null && stepIndex < STEP_COUNT - 1 ? "primary" : "ghost"}
          size="sm"
          disabled={stepIndex >= STEP_COUNT - 1}
          onClick={() => onStep(stepIndex + 1)}
        >
          {chosen !== null && stepIndex < STEP_COUNT - 1
            ? `Next: ${nodeById(STEPS[stepIndex + 1]!.nodeId).label}`
            : "Forward"}
        </Button>
        {chosen !== null ? (
          <Button variant="ghost" size="sm" onClick={onClear}>
            Un-route
          </Button>
        ) : null}
      </div>

      <p className="text-xs text-text">{step.prompt}</p>

      {/* The blurb sits outside the <dl>: a definition list may only contain
          dt/dd groups (plus div/script/template), and a stray <p> in there is a
          real markup error Lighthouse is right to flag. */}
      <div className="rounded-md border border-border bg-surface-2 p-3 text-xs">
      <dl>
        <div className="flex items-baseline justify-between gap-2">
          <dt className="text-text-muted">
            {step.kind === "accumulate" ? "Arriving from" : "Gradient arriving"}
          </dt>
          <dd className="font-mono tabular-nums">
            {step.kind === "accumulate"
              ? consumersOf(step.nodeId)
                  .map(
                    (consumer, index) =>
                      `${nodeById(consumer).label} ${signed(
                        pending[index] ?? 0,
                      )}`,
                  )
                  .join(" · ")
              : incoming === null
                ? "not settled yet"
                : signed(incoming)}
          </dd>
        </div>
        {step.kind === "route" ? (
          <div className="mt-1.5 flex items-baseline justify-between gap-2">
            <dt className="text-text-muted">
              Forward values it was given
            </dt>
            <dd className="font-mono tabular-nums">
              {node.inputs
                .map(
                  (input) =>
                    `${nodeById(input).label} = ${(values[input] ?? 0).toFixed(
                      2,
                    )}`,
                )
                .join(", ")}
            </dd>
          </div>
        ) : null}
      </dl>
        <p className="mt-2 border-t border-border pt-2 text-text-muted">
          {node.blurb}
        </p>
      </div>

      <fieldset className="flex flex-col gap-1.5">
        <legend className="mb-1 text-xs font-semibold">
          {step.kind === "accumulate"
            ? "How do the arriving gradients combine?"
            : "How does the gradient transform here?"}
        </legend>
        {step.options.map((option) => (
          <RuleOption
            key={option.id}
            option={option}
            stepIndex={stepIndex}
            selected={chosen === option.id}
            /* Whether the choice was right is only revealed once it is made — the
               tick would otherwise be the answer key. */
            verdict={chosen === option.id ? option.correct : null}
            onChoose={() => onChoose(option.id)}
          />
        ))}
      </fieldset>

      {chosenRule !== undefined && outgoing.length > 0 ? (
        <div className="rounded-md border border-border p-3">
          <h4 className="text-xs font-semibold">What that sends onward</h4>
          <table className="mt-1.5 w-full text-xs">
            <thead className="text-text-muted">
              <tr>
                <th scope="col" className="text-left font-medium">
                  To
                </th>
                <th scope="col" className="text-right font-medium">
                  Yours
                </th>
                <th scope="col" className="text-right font-medium">
                  Autograd
                </th>
              </tr>
            </thead>
            <tbody>
              {outgoing.map((entry) => (
                <tr key={entry.to} className="border-t border-border">
                  <th scope="row" className="py-1 text-left font-normal">
                    {nodeById(entry.to).label}
                  </th>
                  <td
                    className={cx(
                      "py-1 text-right font-mono tabular-nums",
                      entry.correct ? "text-correct" : "text-wrong",
                    )}
                  >
                    {entry.mine === null ? "—" : signed(entry.mine)}
                  </td>
                  <td className="py-1 text-right font-mono tabular-nums text-text-muted">
                    {signed(entry.truth)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}

function RuleOption({
  option,
  stepIndex,
  selected,
  verdict,
  onChoose,
}: {
  option: Rule;
  stepIndex: number;
  selected: boolean;
  verdict: boolean | null;
  onChoose: () => void;
}) {
  const id = `rule-${stepIndex}-${option.id}`;
  return (
    <label
      htmlFor={id}
      className={cx(
        "flex cursor-pointer gap-2 rounded-md border p-2 text-xs",
        selected
          ? verdict
            ? "border-correct bg-correct/10"
            : "border-wrong bg-wrong/10"
          : "border-border hover:border-primary/60",
      )}
    >
      <input
        id={id}
        type="radio"
        name={`step-${stepIndex}`}
        checked={selected}
        onChange={onChoose}
        className="mt-0.5 size-3.5 shrink-0"
      />
      <span className="min-w-0">
        <span className="flex items-center gap-1.5 font-medium">
          {option.label}
          {selected ? (
            verdict ? (
              <Check aria-hidden="true" className="size-3.5 text-correct" />
            ) : (
              <X aria-hidden="true" className="size-3.5 text-wrong" />
            )
          ) : null}
          {selected ? (
            <span className="sr-only-live">
              {verdict ? "correct" : "not the rule here"}
            </span>
          ) : null}
        </span>
        {selected ? (
          <span className="mt-0.5 block text-text-muted">{option.detail}</span>
        ) : null}
      </span>
    </label>
  );
}
