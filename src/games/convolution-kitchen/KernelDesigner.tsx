"use client";

import { Minus, Plus, Trash2 } from "lucide-react";
import { Button } from "@/components";
import { cx } from "@/lib/utils";
import {
  KERNEL_PRESETS,
  KERNEL_SIZE,
  WEIGHT_MAX,
  WEIGHT_MIN,
  isBlank,
  isSingleSigned,
  kernelSum,
  type FilterHealth,
  type Kernel,
} from "./ml";

export interface KernelDesignerProps {
  kernel: Kernel;
  layerIndex: number;
  kernelIndex: number;
  health: FilterHealth | undefined;
  disabled: boolean;
  removable: boolean;
  selected: boolean;
  onWeight: (cell: number, value: number) => void;
  onPreset: (preset: string) => void;
  onRemove: () => void;
  onSelect: () => void;
}

/**
 * One 3x3 kernel, editable (spec: `<KernelDesigner>`).
 *
 * ── Steppers, not a canvas you paint on ─────────────────────────────────────
 * The obvious design is a grid you drag or scribble on. That is mouse-only, and
 * this project's accessibility floor does not allow a control that a keyboard
 * cannot reach — the same reason the React Flow graphs in Neuron Forge and Decision
 * Tree Architect are read-only. Each cell is a pair of buttons around a number, so
 * the whole kernel is reachable by Tab and Enter, and the number is legible rather
 * than implied by a shade. The tint is added on top of the number, never instead of
 * it, and it is a blue-to-orange pair rather than red-to-green so that a
 * red-green colour deficiency does not hide the sign.
 *
 * The weight sum is printed under the grid because it is the single most
 * diagnostic number about a kernel: near zero means a detector, far from zero
 * means a brightness meter, well below zero means it will be silenced by the ReLU.
 */
export function KernelDesigner({
  kernel,
  layerIndex,
  kernelIndex,
  health,
  disabled,
  removable,
  selected,
  onWeight,
  onPreset,
  onRemove,
  onSelect,
}: KernelDesignerProps) {
  const sum = kernelSum(kernel);
  const flat = isBlank(kernel);
  const oneSided = isSingleSigned(kernel) && !flat;
  const groupId = `kernel-${layerIndex}-${kernelIndex}`;

  const tint = (weight: number): string => {
    if (weight === 0) return "var(--surface-2)";
    const strength = (Math.abs(weight) / WEIGHT_MAX) * 62 + 14;
    return weight > 0
      ? `color-mix(in srgb, var(--class-a) ${strength}%, var(--surface-2))`
      : `color-mix(in srgb, var(--class-b) ${strength}%, var(--surface-2))`;
  };

  return (
    <div
      className={cx(
        "rounded-md border p-3",
        selected ? "border-[var(--primary)] bg-surface-2" : "border-border",
      )}
    >
      <div className="mb-2 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h4 className="truncate text-xs font-semibold" id={`${groupId}-label`}>
            {kernel.label}
          </h4>
          <p className="text-[11px] text-text-muted">
            sum {sum > 0 ? "+" : ""}
            {sum}
            {health ? ` · mean output ${health.meanActivation.toFixed(3)}` : ""}
          </p>
        </div>
        <div className="flex shrink-0 gap-1">
          <Button
            variant={selected ? "primary" : "ghost"}
            size="sm"
            aria-pressed={selected}
            onClick={onSelect}
            disabled={layerIndex !== 0}
            title={
              layerIndex === 0
                ? "Show this filter in the sliding window"
                : "The sliding window demonstrates layer 1, which reads pixels"
            }
          >
            Slide
          </Button>
          {removable ? (
            <Button
              variant="ghost"
              size="sm"
              onClick={onRemove}
              disabled={disabled}
              icon={<Trash2 className="size-3.5" />}
            >
              <span className="sr-only-live">
                Remove {kernel.label} from layer {layerIndex + 1}
              </span>
            </Button>
          ) : null}
        </div>
      </div>

      <div
        role="group"
        aria-labelledby={`${groupId}-label`}
        className="grid grid-cols-3 gap-1"
      >
        {/*
         * Each stepper is at least 24px tall.
         *
         * Not a style preference: at py-0.5 they were about 14px and Lighthouse
         * failed the route on target-size, which is WCAG 2.2's 24x24 minimum. Nine
         * cells times two buttons is eighteen tap targets in a small space, so this
         * is exactly the control where undersized hit areas hurt most.
         */}
        {kernel.weights.map((weight, cell) => (
          <div
            key={cell}
            className="flex flex-col items-center rounded-sm border border-border"
            style={{ background: tint(weight) }}
          >
            <button
              type="button"
              className="flex h-6 w-full items-center justify-center text-text-muted hover:text-text disabled:opacity-40"
              disabled={disabled || weight >= WEIGHT_MAX}
              onClick={() => onWeight(cell, weight + 1)}
              aria-label={`Increase row ${
                Math.floor(cell / KERNEL_SIZE) + 1
              } column ${(cell % KERNEL_SIZE) + 1} of ${kernel.label}, currently ${weight}`}
            >
              <Plus className="size-3.5" />
            </button>
            <span className="font-mono text-xs font-semibold leading-4 tabular-nums">
              {weight > 0 ? "+" : ""}
              {weight}
            </span>
            <button
              type="button"
              className="flex h-6 w-full items-center justify-center text-text-muted hover:text-text disabled:opacity-40"
              disabled={disabled || weight <= WEIGHT_MIN}
              onClick={() => onWeight(cell, weight - 1)}
              aria-label={`Decrease row ${
                Math.floor(cell / KERNEL_SIZE) + 1
              } column ${(cell % KERNEL_SIZE) + 1} of ${kernel.label}, currently ${weight}`}
            >
              <Minus className="size-3.5" />
            </button>
          </div>
        ))}
      </div>

      {health?.dead ? (
        <p className="mt-2 text-[11px] text-[var(--wrong)]">
          Outputs nothing: mean activation {health.meanActivation.toFixed(4)}.
        </p>
      ) : oneSided && layerIndex === 0 ? (
        <p className="mt-2 text-[11px] text-[var(--warn)]">
          No sign change — this is a brightness meter, not a detector.
        </p>
      ) : null}

      <label className="mt-2 block">
        <span className="sr-only-live">
          Preset for {kernel.label} in layer {layerIndex + 1}
        </span>
        <select
          className="w-full rounded-sm border border-border bg-bg px-2 py-1 text-xs"
          value=""
          disabled={disabled}
          onChange={(event) => {
            if (event.target.value) onPreset(event.target.value);
          }}
          aria-label={`Load a preset into ${kernel.label}, layer ${
            layerIndex + 1
          } slot ${kernelIndex + 1}`}
        >
          <option value="">Load a preset…</option>
          {KERNEL_PRESETS.map((preset) => (
            <option key={preset.label} value={preset.label}>
              {preset.label}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
