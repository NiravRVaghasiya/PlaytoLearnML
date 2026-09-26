import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { buildCloud, project, DEFAULT_ANGLES } from "./ml";
import {
  GROUP_COLOUR_TOKENS,
  ShadowPlane2D,
  UNREVEALED_COLOUR_TOKEN,
} from "./ShadowPlane2D";
import { tokenColour } from "./PointCloud3D";

/**
 * A group is one colour in both views (DESIGN.md: class colours are fixed). The
 * 3D cloud used to carry three hexes of its own — 0x4c78d0, 0xe08b3c, 0x59a14f —
 * beside a shadow drawn in the class tokens, so "blue" in one view was a
 * different blue in the other. Both now read one list of tokens.
 */

const css = readFileSync(resolve(process.cwd(), "src/app/globals.css"), "utf8");
const declared = (token: string) =>
  css.match(new RegExp(`${token}:\\s*(#[0-9a-fA-F]{6})`))?.[1]?.toLowerCase();

describe("Dimension Diver's group colours", () => {
  afterEach(() => {
    for (const { token } of [...GROUP_COLOUR_TOKENS, UNREVEALED_COLOUR_TOKEN]) {
      document.documentElement.style.removeProperty(token);
    }
  });

  it("are the class tokens, and the fallbacks are those tokens' values", () => {
    expect(GROUP_COLOUR_TOKENS.map(({ token }) => token)).toEqual([
      "--class-a",
      "--class-b",
      "--class-c",
    ]);
    for (const { token, fallback } of [...GROUP_COLOUR_TOKENS, UNREVEALED_COLOUR_TOKEN]) {
      expect(declared(token), token).toBe(fallback);
    }
  });

  it("fill the shadow from those tokens", () => {
    const shadow = project(buildCloud("pancake", 2029), DEFAULT_ANGLES);
    const { container } = render(
      <ShadowPlane2D shadow={shadow} showGroups retained={0.5} />,
    );
    const fills = new Set(
      [...container.querySelectorAll("circle, rect, polygon")].map((glyph) =>
        glyph.getAttribute("fill"),
      ),
    );
    expect(fills).toEqual(
      new Set(GROUP_COLOUR_TOKENS.map(({ token }) => `var(${token})`)),
    );
  });

  it("give the 3D view the colour the stylesheet declares, not one of its own", () => {
    for (const entry of GROUP_COLOUR_TOKENS) {
      const value = declared(entry.token)!;
      document.documentElement.style.setProperty(entry.token, value);
      expect(`#${tokenColour(entry).getHexString()}`, entry.token).toBe(value);
    }
  });

  it("fall back to the token's value where no stylesheet is loaded", () => {
    expect(`#${tokenColour(GROUP_COLOUR_TOKENS[0]).getHexString()}`).toBe(
      GROUP_COLOUR_TOKENS[0].fallback,
    );
  });
});
