"use client";

import { useMemo } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components";
import { formatPercent } from "@/lib/utils";
import {
  CHANCE_RATE,
  DISHES,
  featureMapsFor,
  stackGeometry,
} from "./ml";
import { previewSample, useKitchenStore } from "./store";
import { FeatureMapGallery } from "./FeatureMapGallery";
import { ImageCanvas } from "./ImageCanvas";

/**
 * The visual lane.
 *
 * Selectors take stable slices only — never a freshly built object, which breaks
 * `useSyncExternalStore`'s cached-snapshot check and re-renders forever. The
 * feature maps are derived here in `useMemo` rather than stored, because they are a
 * pure function of one image and the stack and a stale one would be the single most
 * misleading thing this game could show.
 */
export function VisualLane() {
  const layers = useKitchenStore((s) => s.layers);
  const previewIndex = useKitchenStore((s) => s.previewIndex);
  const windowRow = useKitchenStore((s) => s.windowRow);
  const windowCol = useKitchenStore((s) => s.windowCol);
  const windowKernel = useKitchenStore((s) => s.windowKernel);
  const score = useKitchenStore((s) => s.score);
  const baseline = useKitchenStore((s) => s.baseline);
  const sample = useKitchenStore(previewSample);
  const setPreview = useKitchenStore((s) => s.setPreview);
  const moveWindow = useKitchenStore((s) => s.moveWindow);

  const maps = useMemo(() => featureMapsFor(sample, layers), [sample, layers]);
  const geometry = useMemo(() => stackGeometry(layers), [layers]);
  const receptiveFields = useMemo(() => {
    // Receptive field after each layer, for the gallery's per-layer caption.
    const fields: number[] = [];
    for (let count = 1; count <= layers.length; count += 1) {
      fields.push(stackGeometry(layers.slice(0, count)).receptiveField);
    }
    return fields;
  }, [layers]);

  const kernel = layers[0]?.kernels[windowKernel] ?? null;

  return (
    <div className="flex flex-col gap-5">
      <section aria-labelledby="input-heading" className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 id="input-heading" className="text-sm font-semibold">
            Slide the kernel
          </h2>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setPreview(previewIndex - 1)}
              disabled={previewIndex === 0}
              icon={<ChevronLeft className="size-4" />}
            >
              <span className="sr-only-live">Previous plate</span>
            </Button>
            <span className="min-w-24 text-center text-xs text-text-muted">
              plate {previewIndex + 1}
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setPreview(previewIndex + 1)}
              icon={<ChevronRight className="size-4" />}
            >
              <span className="sr-only-live">Next plate</span>
            </Button>
          </div>
        </div>

        <ImageCanvas
          sample={sample}
          kernel={kernel}
          row={windowRow}
          col={windowCol}
          onMove={moveWindow}
        />
      </section>

      <div className="border-t border-border pt-4">
        <FeatureMapGallery
          sample={sample}
          maps={maps}
          receptiveFields={receptiveFields}
        />
      </div>

      <section
        aria-labelledby="perclass-heading"
        className="border-t border-border pt-4"
      >
        <h2 id="perclass-heading" className="mb-2 text-sm font-semibold">
          Which dishes it can read
        </h2>
        <table className="w-full text-xs">
          <caption className="sr-only-live">
            Detection accuracy for each of the four dishes, on plates the
            classifier has not seen
          </caption>
          <thead>
            <tr className="text-text-muted">
              <th scope="col" className="py-1 text-left font-medium">
                Dish
              </th>
              <th scope="col" className="py-1 text-right font-medium">
                Read correctly
              </th>
              <th scope="col" className="py-1 pl-3 text-left font-medium">
                <span className="sr-only-live">Bar</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {DISHES.map((dish) => {
              const value = score.perClass[dish.id] ?? 0;
              return (
                <tr key={dish.id} className="border-t border-border">
                  <th scope="row" className="py-1.5 text-left font-normal">
                    {dish.label}
                    <span className="block text-[11px] text-text-muted">
                      {dish.tell}
                    </span>
                  </th>
                  <td className="py-1.5 text-right font-mono tabular-nums">
                    {formatPercent(value)}
                  </td>
                  <td className="w-1/3 py-1.5 pl-3">
                    <span className="flex h-2 w-full items-center rounded-sm bg-surface-2">
                      <span
                        aria-hidden="true"
                        className="block h-2 rounded-sm"
                        style={{
                          width: `${Math.round(value * 100)}%`,
                          background:
                            value >= 0.9
                              ? "var(--correct)"
                              : value >= 0.5
                                ? "var(--warn)"
                                : "var(--wrong)",
                        }}
                      />
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>

        <p className="mt-2 text-xs text-text-muted">
          The same classifier on raw pixels manages{" "}
          {baseline === null ? "…" : formatPercent(baseline)} against{" "}
          {formatPercent(CHANCE_RATE)} for guessing — random stripe phase means no
          fixed pixel template fits, so everything above that came from your
          filters. Your stack hands it {geometry.finalSize}x{geometry.finalSize}{" "}
          maps, averaged to one number per channel.
        </p>
      </section>
    </div>
  );
}
