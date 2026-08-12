"use client";

import { Slider } from "@/components";
import { DIALS, toRealValue, type UnitPoint } from "./ml";

export interface SafeDialsProps {
  dials: UnitPoint;
  disabled: boolean;
  onChange: (index: number, unit: number) => void;
}

/**
 * The safe's dials (spec: `<SafeDials>`).
 *
 * Each slider runs over the unit interval and displays the real value, which is
 * how a log-scaled hyperparameter has to be presented: on a linear track from
 * 0.0001 to 1, everything interesting sits in the first 3% of travel. Moving the
 * learning rate dial by a tenth of its range moves it by four tenths of a decade,
 * evenly, wherever you are.
 *
 * That is also why every strategy searches in unit coordinates. A grid laid out in
 * raw learning-rate units would put both its samples inside the top decade and
 * never look at the four below.
 */
export function SafeDials({ dials, disabled, onChange }: SafeDialsProps) {
  return (
    <fieldset className="border-t border-border pt-4">
      <legend className="text-sm font-medium">Dials</legend>
      <div className="mt-1 flex flex-col gap-1">
        {DIALS.map((dial, index) => {
          const unit = dials[index] ?? 0.5;
          return (
            <div key={dial.name}>
              <Slider
                label={dial.name}
                value={unit}
                min={0}
                max={1}
                step={0.005}
                disabled={disabled}
                onChange={(value) => onChange(index, value)}
                format={() => dial.format(toRealValue(dial, unit))}
                hint={dial.description}
              />
            </div>
          );
        })}
      </div>
    </fieldset>
  );
}
