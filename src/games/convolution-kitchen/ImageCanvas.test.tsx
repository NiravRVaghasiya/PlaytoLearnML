import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { IMAGE_SIZE, generateDataset, makeKernel } from "./ml";
import { ImageCanvas } from "./ImageCanvas";
import { LearnedKernels } from "./LearnedKernels";

/**
 * The sliding window's pointer and keyboard contract, and the learned-kernel view.
 *
 * jsdom lays nothing out, so the canvas's rendered box is stubbed — which is the
 * point: the click mapping has to follow the size the canvas is DRAWN at, now
 * that it shrinks to fit a phone, not the 11 px per pixel it is painted at.
 */

const sample = generateDataset(4711).validation[0]!;
const kernel = makeKernel("Vertical edge");

function stubCanvasBox(width: number) {
  const canvas = document.querySelector("canvas")!;
  vi.spyOn(canvas, "getBoundingClientRect").mockReturnValue({
    left: 0,
    top: 0,
    right: width,
    bottom: width,
    width,
    height: width,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  });
}

const windowButton = () =>
  screen.getByRole("button", { name: /Kernel window on the input picture/ });

describe("the sliding window", () => {
  it("maps a click through the canvas's drawn size, so a shrunken canvas still lands true", () => {
    const onMove = vi.fn();
    render(<ImageCanvas sample={sample} kernel={kernel} row={8} col={8} onMove={onMove} />);
    // Drawn at half size: 132 px for 24 image pixels, 5.5 px each.
    stubCanvasBox(132);
    // Pixel (row 10, col 20) sits at y = 10 * 5.5 + 1, x = 20 * 5.5 + 1.
    fireEvent.click(windowButton(), { clientX: 20 * 5.5 + 1, clientY: 10 * 5.5 + 1, detail: 1 });
    // The window is centred on the pixel under the pointer.
    expect(onMove).toHaveBeenLastCalledWith(9, 19);
  });

  it("ignores the click that Enter and Space fire, instead of jumping to the corner", () => {
    const onMove = vi.fn();
    render(<ImageCanvas sample={sample} kernel={kernel} row={8} col={8} onMove={onMove} />);
    stubCanvasBox(264);
    fireEvent.click(windowButton(), { clientX: 0, clientY: 0, detail: 0 });
    expect(onMove).not.toHaveBeenCalled();
  });

  it("keeps the window inside the picture from the keyboard", () => {
    const onMove = vi.fn();
    render(
      <ImageCanvas
        sample={sample}
        kernel={kernel}
        row={IMAGE_SIZE - 3}
        col={0}
        onMove={onMove}
      />,
    );
    fireEvent.keyDown(windowButton(), { key: "ArrowDown" });
    expect(onMove).toHaveBeenLastCalledWith(IMAGE_SIZE - 3, 0);
    fireEvent.keyDown(windowButton(), { key: "ArrowRight", shiftKey: true });
    expect(onMove).toHaveBeenLastCalledWith(IMAGE_SIZE - 3, 3);
  });

  it("announces what the window found after a move, and not before", () => {
    const onMove = vi.fn();
    const { rerender } = render(
      <ImageCanvas sample={sample} kernel={kernel} row={8} col={8} onMove={onMove} />,
    );
    const live = document.querySelector('[aria-live="polite"]')!;
    expect(live.textContent).toBe("");

    fireEvent.keyDown(windowButton(), { key: "ArrowRight" });
    // The store moves the window; the parent re-renders with the new position.
    rerender(<ImageCanvas sample={sample} kernel={kernel} row={8} col={9} onMove={onMove} />);
    expect(live.textContent).toMatch(/^Window at row 9, column 10: the nine products sum to -?\d+\.\d{2}/);

    // A weight edit is not a move, so it is not read out as one.
    rerender(
      <ImageCanvas sample={sample} kernel={makeKernel("Blur")} row={8} col={9} onMove={onMove} />,
    );
    expect(live.textContent).toBe("");
  });

  it("positions the window overlay in proportion to the picture, not in fixed pixels", () => {
    render(<ImageCanvas sample={sample} kernel={kernel} row={12} col={6} onMove={() => {}} />);
    const overlay = windowButton().querySelector("span")!;
    expect(overlay.style.top).toBe(`${(12 / IMAGE_SIZE) * 100}%`);
    expect(overlay.style.left).toBe(`${(6 / IMAGE_SIZE) * 100}%`);
    expect(overlay.style.width).toBe(`${(3 / IMAGE_SIZE) * 100}%`);
  });
});

describe("the learned kernels", () => {
  it("draws every learned grid as a table of nine readable numbers", () => {
    render(
      <LearnedKernels
        learned={{
          accuracy: 0.8,
          kernels: [
            [
              [2, 0, -2, 1.26, 0.01, -1.26, 2, 0, -2],
              [-0.5, -1, -0.5, 0, 0, 0, 0.5, 1, 0.5],
            ],
          ],
        }}
      />,
    );
    const tables = screen.getAllByRole("table");
    expect(tables).toHaveLength(2);
    const cells = tables[0]!.querySelectorAll("td");
    expect(cells).toHaveLength(9);
    expect([...cells].map((cell) => cell.textContent)).toEqual([
      "+2.0",
      "0.0",
      "-2.0",
      "+1.3",
      "0.0",
      "-1.3",
      "+2.0",
      "0.0",
      "-2.0",
    ]);
    expect(screen.getByRole("heading", { name: "What gradient descent picked" })).toBeInTheDocument();
  });

  it("says which grids a depthwise layer 2 is showing", () => {
    render(
      <LearnedKernels
        learned={{
          accuracy: 0.95,
          kernels: [[[0, 0, 0, 0, 2, 0, 0, 0, 0]], [[1, 2, 1, 0, 0, 0, -1, -2, -1]]],
        }}
      />,
    );
    expect(screen.getByText(/Layer 2 · the grids it applied to layer 1's first map/)).toBeInTheDocument();
  });

  it("renders nothing for an empty result", () => {
    const { container } = render(<LearnedKernels learned={{ accuracy: 0, kernels: [] }} />);
    expect(container).toBeEmptyDOMElement();
  });
});
