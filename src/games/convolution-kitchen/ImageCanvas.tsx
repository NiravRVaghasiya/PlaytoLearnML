"use client";

import { useCallback, useEffect, useRef } from "react";
import {
  DISHES,
  IMAGE_SIZE,
  KERNEL_SIZE,
  convolveOne,
  type Kernel,
  type Sample,
} from "./ml";
import { windowCaption } from "./why-cards";

export interface ImageCanvasProps {
  sample: Sample;
  kernel: Kernel | null;
  row: number;
  col: number;
  onMove: (row: number, col: number) => void;
}

const SCALE = 11;

/**
 * The input picture with the kernel's window on it (spec: `<ImageCanvas>`).
 *
 * This component is where pedagogy contract #2 lands. The spec says "sliding the
 * kernel = the convolution operation itself", so the window is a real control: move
 * it with the arrow keys and the nine multiplies and their sum are recomputed and
 * printed. The arithmetic comes from `convolveOne`, which the tests check against
 * `tf.conv2d` on the same patch — the explanation and the implementation are held
 * to the same number.
 *
 * ── Canvas, and what that costs ─────────────────────────────────────────────
 * The spec names canvas and canvas is right for pixel data: 576 `<rect>` elements
 * per image, times a gallery of feature maps, is a lot of DOM for something that is
 * a bitmap. The cost is that a canvas is invisible to assistive technology, so the
 * two things that carry MEANING are not left inside it: the window's nine numbers
 * are a real table, and the picture itself gets a text description. The bitmap is
 * the illustration; the table is the content.
 */
export function ImageCanvas({
  sample,
  kernel,
  row,
  col,
  onMove,
}: ImageCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const image = context.createImageData(IMAGE_SIZE, IMAGE_SIZE);
    for (let index = 0; index < IMAGE_SIZE * IMAGE_SIZE; index += 1) {
      const value = Math.round((sample.pixels[index] ?? 0) * 255);
      image.data[index * 4] = value;
      image.data[index * 4 + 1] = value;
      image.data[index * 4 + 2] = value;
      image.data[index * 4 + 3] = 255;
    }

    // Draw at 1:1 into an offscreen buffer, then blit it up with smoothing off so
    // the pixels stay square and countable.
    const buffer = document.createElement("canvas");
    buffer.width = IMAGE_SIZE;
    buffer.height = IMAGE_SIZE;
    buffer.getContext("2d")?.putImageData(image, 0, 0);

    context.imageSmoothingEnabled = false;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(buffer, 0, 0, canvas.width, canvas.height);
  }, [sample]);

  const handleKey = useCallback(
    (event: React.KeyboardEvent<HTMLButtonElement>) => {
      const step = event.shiftKey ? 3 : 1;
      let handled = true;
      if (event.key === "ArrowUp") onMove(row - step, col);
      else if (event.key === "ArrowDown") onMove(row + step, col);
      else if (event.key === "ArrowLeft") onMove(row, col - step);
      else if (event.key === "ArrowRight") onMove(row, col + step);
      else if (event.key === "Home") onMove(0, 0);
      else if (event.key === "End") onMove(IMAGE_SIZE - 3, IMAGE_SIZE - 3);
      else handled = false;
      if (handled) event.preventDefault();
    },
    [onMove, row, col],
  );

  /** Click to place the window's centre where the pointer is. */
  const handleClick = useCallback(
    (event: React.MouseEvent<HTMLButtonElement>) => {
      const bounds = event.currentTarget.getBoundingClientRect();
      const x = Math.floor((event.clientX - bounds.left) / SCALE);
      const y = Math.floor((event.clientY - bounds.top) / SCALE);
      onMove(y - 1, x - 1);
    },
    [onMove],
  );

  const dish = DISHES[sample.label]!;
  const result = kernel ? convolveOne(sample, kernel, row, col) : null;

  return (
    <div className="flex flex-col gap-3 sm:flex-row">
      <div className="flex flex-col gap-1">
        {/*
         * A real button, not a div with a role.
         *
         * This needs to be operable by keyboard AND by pointer, and the honest way
         * to get both is an interactive element: arrow keys nudge the window, a
         * click drops it where you point. Wrapping it as `role="application"` would
         * have worked for screen readers and left mouse users with no way to jump
         * across the picture.
         */}
        <button
          type="button"
          aria-label={`Kernel window on the input picture, at row ${
            row + 1
          }, column ${
            col + 1
          }. Arrow keys move it one pixel, shift and arrow keys move it three, or click to place it.`}
          className="relative block w-fit rounded-md border border-border focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--focus)]"
          onKeyDown={handleKey}
          onClick={handleClick}
        >
          <canvas
            ref={canvasRef}
            width={IMAGE_SIZE * SCALE}
            height={IMAGE_SIZE * SCALE}
            className="block rounded-md"
            // The bitmap is illustration. Everything that matters is in the table
            // below and in the description beside it.
            aria-hidden="true"
          />
          <span
            aria-hidden="true"
            className="pointer-events-none absolute border-2 border-[var(--primary)] shadow-[0_0_0_2px_var(--bg)]"
            style={{
              left: col * SCALE,
              top: row * SCALE,
              width: KERNEL_SIZE * SCALE,
              height: KERNEL_SIZE * SCALE,
            }}
          />
        </button>
        <p className="text-xs text-text-muted">
          {dish.label} · {IMAGE_SIZE}x{IMAGE_SIZE} · window at row {row + 1},
          column {col + 1}
        </p>
      </div>

      <div className="min-w-0 flex-1">
        <table className="w-full text-xs">
          <caption className="mb-1 text-left text-xs font-semibold">
            The nine multiplies under the window
          </caption>
          <thead>
            <tr className="text-text-muted">
              <th scope="col" className="py-1 text-left font-medium">
                Pixel
              </th>
              <th scope="col" className="py-1 text-right font-medium">
                Weight
              </th>
              <th scope="col" className="py-1 text-right font-medium">
                Product
              </th>
            </tr>
          </thead>
          <tbody>
            {result === null ? (
              <tr>
                <td colSpan={3} className="py-2 text-text-muted">
                  Add a filter to layer 1 to see the arithmetic.
                </td>
              </tr>
            ) : (
              result.patch.map((pixel, index) => (
                <tr key={index} className="border-t border-border">
                  <td className="py-0.5 font-mono tabular-nums">
                    {pixel.toFixed(2)}
                  </td>
                  <td className="py-0.5 text-right font-mono tabular-nums">
                    {kernel!.weights[index]! > 0 ? "+" : ""}
                    {kernel!.weights[index]}
                  </td>
                  <td className="py-0.5 text-right font-mono tabular-nums">
                    {result.products[index]!.toFixed(2)}
                  </td>
                </tr>
              ))
            )}
          </tbody>
          {result !== null ? (
            <tfoot>
              <tr className="border-t-2 border-border font-semibold">
                <th scope="row" colSpan={2} className="py-1 text-left">
                  Sum
                </th>
                <td className="py-1 text-right font-mono tabular-nums">
                  {result.sum.toFixed(2)}
                </td>
              </tr>
              <tr>
                <th scope="row" colSpan={2} className="py-1 text-left font-normal">
                  After ReLU
                </th>
                <td className="py-1 text-right font-mono tabular-nums">
                  {result.activated.toFixed(2)}
                </td>
              </tr>
            </tfoot>
          ) : null}
        </table>

        {result !== null ? (
          <p className="mt-2 text-xs text-text-muted">
            {windowCaption(result.sum, result.activated)}
          </p>
        ) : null}
      </div>
    </div>
  );
}
