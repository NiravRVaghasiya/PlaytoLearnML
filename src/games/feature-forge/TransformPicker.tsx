"use client";

import { TRANSFORMS, type Transform } from "./ml";

export interface TransformPickerProps {
  transform: Transform;
  disabled: boolean;
  onChange: (transform: Transform) => void;
}

/**
 * Which transform to apply (spec: `<TransformPicker>`).
 *
 * A radio group: mutually exclusive, arrow-key navigable, one tab stop.
 *
 * Each option carries its mechanism rather than just its name, because the whole
 * distinction the game turns on — changing units versus changing meaning — is
 * invisible from the labels alone. "Standardise" and "Bin" both sound like
 * housekeeping; only one of them widens what a linear model can express.
 */
export function TransformPicker({
  transform,
  disabled,
  onChange,
}: TransformPickerProps) {
  return (
    <fieldset className="border-t border-border pt-4">
      <legend className="text-sm font-medium">Transform</legend>
      <div className="mt-2 flex flex-col gap-1">
        {TRANSFORMS.map((spec) => (
          // htmlFor rather than relying on nesting: the label's text sits two
          // spans deep, which the a11y linter cannot see through.
          <label
            key={spec.id}
            htmlFor={`transform-${spec.id}`}
            className={`flex cursor-pointer flex-wrap items-baseline gap-x-1.5 gap-y-0.5 rounded border px-2 py-1.5 text-xs ${
              spec.id === transform
                ? "border-primary bg-primary/10"
                : "border-border bg-surface-2 hover:bg-surface"
            } ${disabled ? "cursor-not-allowed opacity-60" : ""}`}
          >
            <input
              id={`transform-${spec.id}`}
              type="radio"
              name="forge-transform"
              value={spec.id}
              checked={spec.id === transform}
              disabled={disabled}
              onChange={() => onChange(spec.id)}
              className="mt-0.5 accent-[var(--primary)]"
            />
            {/* Direct children, so the label's accessible text is one level deep
                rather than buried in wrapper spans. */}
            <span className="font-medium">{spec.label}</span>
            <span className="text-[10px] text-text-muted">
              {spec.arity === 1 ? "1 column" : "2 columns"}
            </span>
            <span className="basis-full text-[11px] text-text-muted">
              {spec.blurb}
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}
