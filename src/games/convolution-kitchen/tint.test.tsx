import { readFileSync } from "node:fs";
import { join } from "node:path";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { KernelDesigner } from "./KernelDesigner";
import { LearnedKernels } from "./LearnedKernels";
import { WEIGHT_MAX, WEIGHT_MIN, makeKernel } from "./ml";
import { TINT_CEILING, TINT_FLOOR, weightTint } from "./tint";

/**
 * The kernel grids' tint, checked against WCAG AA with the real tokens.
 *
 * Lighthouse cannot guard this: it audits the first render, and the grids with
 * the strongest orange only appear after "Add a filter" (the Vertical edge preset
 * has two -2 cells) or after "Let it learn" (every learned grid has a cell at
 * exactly ±2). So the ratio is computed here, from the colours in globals.css
 * rather than from copies of them, and a retuned palette or tint is checked too.
 */

type Rgb = [number, number, number];

const css = readFileSync(join(process.cwd(), "src", "app", "globals.css"), "utf8");

function token(name: string): Rgb {
  const hex = css.match(new RegExp(`--${name}:\\s*#([0-9a-fA-F]{6})\\b`))?.[1];
  if (hex === undefined) throw new Error(`--${name} is not a hex token in globals.css`);
  return [0, 2, 4].map((at) => parseInt(hex.slice(at, at + 2), 16)) as Rgb;
}

/** What the browser paints for a `weightTint` value: `in srgb` mixes the encoded channels. */
function resolve(background: string): Rgb {
  const plain = background.match(/^var\(--([\w-]+)\)$/);
  if (plain) return token(plain[1]!);
  const mix = background.match(
    /^color-mix\(in srgb, var\(--([\w-]+)\) ([\d.]+)%, var\(--([\w-]+)\)\)$/,
  );
  if (!mix) throw new Error(`Unrecognised tint: ${background}`);
  const share = Number(mix[2]) / 100;
  const [a, b] = [token(mix[1]!), token(mix[3]!)];
  return a.map((channel, index) => channel * share + b[index]! * (1 - share)) as Rgb;
}

function luminance([r, g, b]: Rgb): number {
  const linear = (channel: number) => {
    const s = channel / 255;
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b);
}

function contrast(foreground: Rgb, background: Rgb): number {
  const [light, dark] = [luminance(foreground), luminance(background)].sort(
    (x, y) => y - x,
  );
  return (light! + 0.05) / (dark! + 0.05);
}

const strengthOf = (weight: number) =>
  Number(weightTint(weight).match(/ ([\d.]+)%/)?.[1] ?? Number.NaN);

describe("the kernel tint", () => {
  it("keeps every cell's number at AA contrast, the strongest orange included", () => {
    const text = token("text");
    // Every weight a grid can show: the steppers' integers and a learned grid's
    // floats, both signs, at 0.05 steps.
    let worst = { weight: 0, ratio: Number.POSITIVE_INFINITY };
    for (let step = WEIGHT_MIN * 20; step <= WEIGHT_MAX * 20; step += 1) {
      const weight = step / 20;
      const ratio = contrast(text, resolve(weightTint(weight)));
      if (ratio < worst.ratio) worst = { weight, ratio };
    }
    expect(worst.ratio, `worst cell is ${worst.weight}`).toBeGreaterThanOrEqual(4.5);
    // The failing cell this cap was set for: a -2 used to measure 2.8:1.
    expect(contrast(text, resolve(weightTint(WEIGHT_MIN)))).toBeGreaterThanOrEqual(4.5);
    expect(contrast(text, resolve(weightTint(WEIGHT_MAX)))).toBeGreaterThanOrEqual(4.5);
  });

  it("gives equal magnitudes equal strength, from the floor to the ceiling", () => {
    expect(strengthOf(WEIGHT_MAX)).toBe(TINT_CEILING);
    expect(strengthOf(WEIGHT_MIN)).toBe(TINT_CEILING);
    expect(strengthOf(1)).toBe(strengthOf(-1));
    expect(strengthOf(0.5)).toBeGreaterThan(TINT_FLOOR);
    expect(strengthOf(0.5)).toBeLessThan(strengthOf(1));
    // Out of range reads as the extreme rather than overshooting the cap.
    expect(strengthOf(7)).toBe(TINT_CEILING);
    expect(weightTint(1)).toContain("--class-a");
    expect(weightTint(-1)).toContain("--class-b");
    // Anything that prints as 0.0 is not tinted either.
    expect(weightTint(0)).toBe("var(--surface-2)");
    expect(weightTint(-0.04)).toBe("var(--surface-2)");
  });

  it("is the tint both grids paint, under text that is never muted", () => {
    render(
      <>
        <KernelDesigner
          kernel={makeKernel("Vertical edge")}
          layerIndex={0}
          kernelIndex={0}
          health={undefined}
          disabled={false}
          removable={false}
          selected={false}
          onWeight={() => {}}
          onPreset={() => {}}
          onRemove={() => {}}
          onSelect={() => {}}
        />
        <LearnedKernels
          learned={{ accuracy: 0.8, kernels: [[[1, 0, -1, 2, 0, -2, 1, 0, -1]]] }}
        />
      </>,
    );

    // A drawn -2 cell and a learned -2.0 cell carry the same background.
    const drawn = screen
      .getByRole("button", { name: /^Increase row 2 column 3 of Vertical edge/ })
      .closest("div")!;
    const learned = screen.getAllByText("-2.0")[0]!;
    expect(drawn.getAttribute("style")).toContain(weightTint(-2));
    expect(learned.getAttribute("style")).toContain(weightTint(-2));

    // The plus and minus glyphs sit on the tint too, so they are drawn in --text,
    // the colour checked above: --text-muted on the strongest orange is 2.4:1,
    // under the 3:1 an icon needs.
    for (const stepper of screen.getAllByRole("button", {
      name: /^(Increase|Decrease) row/,
    })) {
      expect(stepper.className).not.toMatch(/text-text-muted/);
    }
  });
});
