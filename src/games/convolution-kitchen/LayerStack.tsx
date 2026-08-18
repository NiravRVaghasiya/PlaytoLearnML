"use client";

import { Layers, Plus, Trash2 } from "lucide-react";
import { Button } from "@/components";
import {
  FILTER_BUDGET,
  MAX_KERNELS_PER_LAYER,
  MAX_LAYERS,
  POOL_TYPES,
  filtersUsed,
  stackGeometry,
  type FilterHealth,
  type Layer,
  type PoolType,
} from "./ml";
import { KernelDesigner } from "./KernelDesigner";

export interface LayerStackProps {
  layers: Layer[];
  health: FilterHealth[];
  disabled: boolean;
  selectedKernel: number;
  onWeight: (layerIndex: number, kernelIndex: number, cell: number, value: number) => void;
  onPreset: (layerIndex: number, kernelIndex: number, preset: string) => void;
  onAddKernel: (layerIndex: number) => void;
  onRemoveKernel: (layerIndex: number, kernelIndex: number) => void;
  onPool: (layerIndex: number, pool: PoolType) => void;
  onAddLayer: () => void;
  onRemoveLayer: (layerIndex: number) => void;
  onSelectKernel: (index: number) => void;
}

/**
 * The stack the player is building (spec: `<LayerStack>`).
 *
 * Layers in order, each with its filters and its pooling choice, and the geometry
 * that results printed underneath. The geometry line is not decoration: "one output
 * cell sees 10 pixels" is the fact that explains why depth works at all, and it is
 * the only place a player can watch the receptive field grow as they add a layer or
 * turn pooling on.
 */
export function LayerStack({
  layers,
  health,
  disabled,
  selectedKernel,
  onWeight,
  onPreset,
  onAddKernel,
  onRemoveKernel,
  onPool,
  onAddLayer,
  onRemoveLayer,
  onSelectKernel,
}: LayerStackProps) {
  const used = filtersUsed(layers);
  const geometry = stackGeometry(layers);

  return (
    <section aria-labelledby="stack-heading" className="flex flex-col gap-3">
      <div className="flex items-baseline justify-between gap-2">
        <h3 id="stack-heading" className="text-sm font-semibold">
          Layer stack
        </h3>
        <span className="text-xs text-text-muted">
          {used} of {FILTER_BUDGET} filters
        </span>
      </div>

      {layers.map((layer, layerIndex) => {
        const layerHealth = health.filter(
          (filter) => filter.layerIndex === layerIndex,
        );
        return (
          <div
            key={layer.id}
            className="rounded-md border border-border bg-surface p-3"
          >
            <div className="mb-2 flex items-center justify-between gap-2">
              <h4 className="text-xs font-semibold">
                Layer {layerIndex + 1}
                <span className="ml-1.5 font-normal text-text-muted">
                  {layerIndex === 0 ? "reads pixels" : "reads layer 1's maps"}
                </span>
              </h4>
              {layers.length > 1 ? (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onRemoveLayer(layerIndex)}
                  disabled={disabled}
                  icon={<Trash2 className="size-3.5" />}
                >
                  <span className="sr-only-live">
                    Remove layer {layerIndex + 1}
                  </span>
                </Button>
              ) : null}
            </div>

            <div className="grid gap-2 sm:grid-cols-2">
              {layer.kernels.map((kernel, kernelIndex) => (
                <KernelDesigner
                  key={kernel.id}
                  kernel={kernel}
                  layerIndex={layerIndex}
                  kernelIndex={kernelIndex}
                  health={layerHealth[kernelIndex]}
                  disabled={disabled}
                  removable={layer.kernels.length > 1}
                  selected={layerIndex === 0 && selectedKernel === kernelIndex}
                  onWeight={(cell, value) =>
                    onWeight(layerIndex, kernelIndex, cell, value)
                  }
                  onPreset={(preset) => onPreset(layerIndex, kernelIndex, preset)}
                  onRemove={() => onRemoveKernel(layerIndex, kernelIndex)}
                  onSelect={() => onSelectKernel(kernelIndex)}
                />
              ))}
            </div>

            <Button
              variant="ghost"
              size="sm"
              className="mt-2 w-full"
              onClick={() => onAddKernel(layerIndex)}
              disabled={
                disabled ||
                layer.kernels.length >= MAX_KERNELS_PER_LAYER ||
                used >= FILTER_BUDGET
              }
              icon={<Plus className="size-3.5" />}
            >
              {layer.kernels.length >= MAX_KERNELS_PER_LAYER
                ? `Layer full (${MAX_KERNELS_PER_LAYER} max)`
                : used >= FILTER_BUDGET
                  ? "Budget spent"
                  : "Add a filter"}
            </Button>

            <fieldset className="mt-3">
              <legend className="mb-1 text-[11px] font-medium text-text-muted">
                Pooling after layer {layerIndex + 1}
              </legend>
              <div className="flex flex-wrap gap-1">
                {POOL_TYPES.map((pool) => (
                  <label
                    key={pool.id}
                    className="flex cursor-pointer items-center gap-1 rounded-sm border border-border px-2 py-1 text-[11px] has-checked:border-[var(--primary)] has-checked:bg-surface-2"
                  >
                    <input
                      type="radio"
                      name={`pool-${layer.id}`}
                      value={pool.id}
                      checked={layer.pool === pool.id}
                      disabled={disabled}
                      onChange={() => onPool(layerIndex, pool.id)}
                      className="size-3"
                    />
                    {pool.label}
                  </label>
                ))}
              </div>
              <p className="mt-1 text-[11px] text-text-muted">
                {POOL_TYPES.find((pool) => pool.id === layer.pool)?.hint}
              </p>
            </fieldset>
          </div>
        );
      })}

      {layers.length < MAX_LAYERS ? (
        <Button
          variant="secondary"
          size="sm"
          onClick={onAddLayer}
          disabled={disabled || used >= FILTER_BUDGET}
          icon={<Layers className="size-4" />}
        >
          Add layer 2
        </Button>
      ) : null}

      <p className="text-xs text-text-muted">
        Map sizes {geometry.sizes.join(" → ") || "—"} · one output cell sees{" "}
        <strong className="font-semibold text-text">
          {geometry.receptiveField}px
        </strong>{" "}
        of the picture. Everything then goes through a global average, one number
        per channel, into a classifier you cannot change.
      </p>
    </section>
  );
}
