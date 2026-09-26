"use client";

import { KERNEL_SIZE, type LearnedResult } from "./ml";
import { weightTint } from "./tint";

export interface LearnedKernelsProps {
  learned: LearnedResult;
}

/** One decimal, signed, and a plain 0.0 for anything that rounds to it. */
const signed = (value: number) =>
  Math.abs(value) < 0.05 ? "0.0" : `${value > 0 ? "+" : ""}${value.toFixed(1)}`;

/**
 * The kernels "Let it learn" settled on, read-only.
 *
 * The why-card, the button's caption and the math notes all tell the player to
 * look at what gradient descent picked, and for a while nothing drew it — the
 * kernels came back from `learnStack` and went straight into state unread. This is
 * that view: one 3x3 grid per learned kernel, numbers first and tint second, the
 * same way `<KernelDesigner>` shows the player's own, so the two can be compared
 * cell for cell.
 *
 * Each grid is a real table, not a picture, so the nine numbers are reachable by
 * a screen reader row by row. The weights are scaled so each grid's largest reads
 * ±2 — the stepper range — which changes nothing about what the kernel detects:
 * a positive scale passes straight through the ReLU and only scales the map.
 *
 * Layer 2 is depthwise and learns a separate grid for every layer-1 map; drawing
 * all of them would be K1 x K2 grids for one idea, so it shows the grids applied to
 * the first map and says so.
 */
export function LearnedKernels({ learned }: LearnedKernelsProps) {
  if (learned.kernels.length === 0) return null;

  return (
    <section
      aria-labelledby="learned-kernels-heading"
      className="mt-3 flex flex-col gap-2"
    >
      <h3 id="learned-kernels-heading" className="text-xs font-semibold">
        What gradient descent picked
      </h3>

      {learned.kernels.map((layer, layerIndex) => (
        <div key={layerIndex}>
          <p className="mb-1 text-[11px] text-text-muted">
            Layer {layerIndex + 1}
            {layerIndex > 0 ? " · the grids it applied to layer 1's first map" : ""}
          </p>
          <ul className="flex flex-wrap gap-2">
            {layer.map((weights, kernelIndex) => (
              <li key={kernelIndex}>
                <table className="border-collapse text-[10px]">
                  <caption className="sr-only-live">
                    Learned kernel {kernelIndex + 1} in layer {layerIndex + 1}, row
                    by row, scaled so its largest weight is 2
                  </caption>
                  <tbody>
                    {Array.from({ length: KERNEL_SIZE }, (_, row) => (
                      <tr key={row}>
                        {Array.from({ length: KERNEL_SIZE }, (_, col) => {
                          const weight = weights[row * KERNEL_SIZE + col] ?? 0;
                          return (
                            <td
                              key={col}
                              className="h-7 w-9 border border-border text-center font-mono tabular-nums"
                              style={{ background: weightTint(weight) }}
                            >
                              {signed(weight)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </li>
            ))}
          </ul>
        </div>
      ))}

      <p className="text-[11px] text-text-muted">
        Scaled so each grid&apos;s largest weight reads ±2, the range your steppers
        use. Scaling a kernel only scales its map, so compare the pattern, not the
        size.
      </p>
    </section>
  );
}
