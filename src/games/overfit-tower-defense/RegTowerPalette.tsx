"use client";

import { Minus, Plus } from "lucide-react";
import { MAX_TOWERS_PER_TYPE, type Regularization, type TowerType } from "./ml";

export interface RegTowerPaletteProps {
  counts: Record<TowerType, number>;
  regularization: Regularization;
  disabled: boolean;
  onAdd: (type: TowerType) => void;
  onRemove: (type: TowerType) => void;
}

interface TowerMeta {
  type: TowerType;
  name: string;
  /** What it does, in one clause. */
  effect: string;
}

const TOWERS: readonly TowerMeta[] = [
  {
    type: "l1",
    name: "L1",
    effect: "drives useless weights to exactly zero",
  },
  {
    type: "l2",
    name: "L2",
    effect: "shrinks every weight, none to zero",
  },
  {
    type: "dropout",
    name: "Dropout",
    effect: "switches off random units each step",
  },
] as const;

/**
 * The regularization palette (spec: `<RegTowerPalette>`).
 *
 * Deliberately steppers rather than drag-and-drop placement. The spec describes
 * "dropping towers", but what actually matters is *how many* of each, and a
 * count is something a keyboard and a screen reader can both operate. Each row
 * shows the λ it produces, so the connection between the game object and the
 * number in the loss function is never hidden.
 */
export function RegTowerPalette({
  counts,
  regularization,
  disabled,
  onAdd,
  onRemove,
}: RegTowerPaletteProps) {
  const lambdaFor = (type: TowerType) =>
    type === "l1"
      ? `λ₁ ${regularization.l1.toFixed(3)}`
      : type === "l2"
        ? `λ₂ ${regularization.l2.toFixed(3)}`
        : `rate ${regularization.dropout.toFixed(2)}`;

  return (
    <fieldset className="border-t border-border pt-4">
      <legend className="text-sm font-medium">Regularization towers</legend>

      <ul className="mt-2 flex flex-col gap-2">
        {TOWERS.map((tower) => {
          const count = counts[tower.type];
          return (
            <li
              key={tower.type}
              className="rounded-md border border-border bg-surface-2 p-2"
            >
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium">{tower.name}</span>
                <span className="ml-auto font-mono text-[10px] text-text-muted">
                  {lambdaFor(tower.type)}
                </span>
              </div>

              <div className="mt-1.5 flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => onRemove(tower.type)}
                  disabled={disabled || count <= 0}
                  aria-label={`Remove a ${tower.name} tower, currently ${count}`}
                  className="rounded border border-border p-1 hover:bg-surface focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none disabled:opacity-40"
                >
                  <Minus aria-hidden="true" className="size-3.5" />
                </button>

                {/* Dots read as a quantity at a glance; the count is the text. */}
                <span aria-hidden="true" className="flex gap-1">
                  {Array.from({ length: MAX_TOWERS_PER_TYPE }, (_, index) => (
                    <span
                      key={index}
                      className={`inline-block size-2 rounded-sm ${
                        index < count ? "bg-primary" : "bg-border"
                      }`}
                    />
                  ))}
                </span>

                <button
                  type="button"
                  onClick={() => onAdd(tower.type)}
                  disabled={disabled || count >= MAX_TOWERS_PER_TYPE}
                  aria-label={`Add a ${tower.name} tower, currently ${count}`}
                  className="ml-auto rounded border border-border p-1 hover:bg-surface focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none disabled:opacity-40"
                >
                  <Plus aria-hidden="true" className="size-3.5" />
                </button>
              </div>

              <p className="mt-1 text-[11px] text-text-muted">{tower.effect}</p>
            </li>
          );
        })}
      </ul>
    </fieldset>
  );
}
