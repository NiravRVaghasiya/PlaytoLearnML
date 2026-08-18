"use client";

import { useEffect, useRef } from "react";
import { DISHES, type FeatureMap, type Sample } from "./ml";

export interface FeatureMapGalleryProps {
  sample: Sample;
  /** One entry per layer. */
  maps: FeatureMap[][];
  receptiveFields: number[];
}

/**
 * What each filter sees (spec: `<FeatureMapGallery>`).
 *
 * The whole point of this panel is the moment a player notices that a working
 * feature map does NOT look like the photograph. A blur's map is a softened copy of
 * the input; an edge detector's map is mostly black with bright lines where the
 * pattern was found. That difference is the difference between smoothing an image
 * and detecting something in it, and it is far more convincing seen than described.
 *
 * Each tile is normalised to its own peak so a weak filter is still visible, and
 * the peak is printed underneath — otherwise two tiles that look equally bright
 * could be an order of magnitude apart, which would be a lie told by autoscaling.
 * A dead filter is called out in words, because a black tile and a tile of very
 * small numbers look identical and mean different things.
 */
export function FeatureMapGallery({
  sample,
  maps,
  receptiveFields,
}: FeatureMapGalleryProps) {
  const dish = DISHES[sample.label]!;

  return (
    <section aria-labelledby="gallery-heading" className="flex flex-col gap-3">
      <div className="flex items-baseline justify-between gap-2">
        <h2 id="gallery-heading" className="text-sm font-semibold">
          What each filter sees
        </h2>
        <span className="text-xs text-text-muted">{dish.label}</span>
      </div>

      {maps.length === 0 ? (
        <p className="text-xs text-text-muted">
          No filters yet. Add one to layer 1 and its feature map appears here.
        </p>
      ) : (
        maps.map((layerMaps, layerIndex) => (
          <div key={layerIndex}>
            <h3 className="mb-1.5 text-xs font-medium text-text-muted">
              Layer {layerIndex + 1} · {layerMaps.length} map
              {layerMaps.length === 1 ? "" : "s"} ·{" "}
              {layerMaps[0]?.size ?? 0}x{layerMaps[0]?.size ?? 0} · one cell sees{" "}
              {receptiveFields[layerIndex] ?? 3}px of the picture
            </h3>
            <ul className="flex flex-wrap gap-2">
              {layerMaps.map((map, index) => (
                <li key={`${layerIndex}-${index}`}>
                  <MapTile map={map} />
                </li>
              ))}
            </ul>
          </div>
        ))
      )}
    </section>
  );
}

const TILE = 72;

function MapTile({ map }: { map: FeatureMap }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dead = map.max <= 1e-6;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const image = context.createImageData(map.size, map.size);
    const peak = Math.max(map.max, 1e-9);
    for (let index = 0; index < map.size * map.size; index += 1) {
      // Normalised to this map's own peak, so a quiet filter is still readable.
      // The printed peak below is what stops that from misleading anybody.
      const value = Math.round(((map.values[index] ?? 0) / peak) * 255);
      image.data[index * 4] = value;
      image.data[index * 4 + 1] = value;
      image.data[index * 4 + 2] = value;
      image.data[index * 4 + 3] = 255;
    }

    const buffer = document.createElement("canvas");
    buffer.width = map.size;
    buffer.height = map.size;
    buffer.getContext("2d")?.putImageData(image, 0, 0);

    context.imageSmoothingEnabled = false;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(buffer, 0, 0, canvas.width, canvas.height);
  }, [map]);

  return (
    <figure className="m-0 w-[76px]">
      <canvas
        ref={canvasRef}
        width={TILE}
        height={TILE}
        className="block rounded-sm border border-border"
        role="img"
        aria-label={
          dead
            ? `${map.label}: feature map is entirely zero — this filter fires nowhere on this picture.`
            : `${map.label}: feature map ${map.size} by ${
                map.size
              }, brightest response ${map.max.toFixed(
                2,
              )}, mean ${map.mean.toFixed(2)}.`
        }
      />
      <figcaption className="mt-1 text-[10px] leading-tight text-text-muted">
        <span className="block truncate font-medium text-text" title={map.label}>
          {map.label}
        </span>
        {dead ? (
          <span className="text-[var(--wrong)]">all zero</span>
        ) : (
          <span className="tabular-nums">
            peak {map.max.toFixed(2)} · mean {map.mean.toFixed(2)}
          </span>
        )}
      </figcaption>
    </figure>
  );
}
