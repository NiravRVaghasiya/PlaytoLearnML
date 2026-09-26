"use client";

import { useMemo } from "react";
import { GROUP_LABELS, type Shadow } from "./ml";

export interface ShadowPlane2DProps {
  shadow: readonly Shadow[];
  showGroups: boolean;
  retained: number;
}

const SIZE = 320;
const PAD = 14;

/**
 * The design tokens each group is drawn in, by group id, and the one an
 * unrevealed point is drawn in. `<PointCloud3D>` reads the same list, so a group
 * is the same colour in both views (DESIGN.md: class colours are fixed). The hex
 * values are DESIGN.md's, used only where the stylesheet has not loaded.
 */
export const GROUP_COLOUR_TOKENS = [
  { token: "--class-a", fallback: "#0072b2" },
  { token: "--class-b", fallback: "#e69f00" },
  { token: "--class-c", fallback: "#009e73" },
] as const;
export const UNREVEALED_COLOUR_TOKEN = {
  token: "--text-muted",
  fallback: "#9aa4b2",
};

/**
 * The 2D shadow (spec: `<ShadowPlane2D>`).
 *
 * SVG, not canvas, and this is the deliberate half of the split with
 * `<PointCloud3D>`: the 3D view is the thing you turn, and this is the thing you
 * READ. Every judgement the player makes — are there groups, how many, do they
 * overlap — is made here, so this view is the one that has to be in the
 * accessibility tree, with the group counts and their spread available as text.
 *
 * Colour is never the only channel. Groups are drawn with distinct SHAPES as well
 * as distinct hues from the Okabe–Ito set, so three overlapping clusters remain
 * countable without colour vision.
 *
 * And each glyph has to be visible against what it is drawn on (WCAG 1.4.11,
 * 3:1). The plot sits on --bg, not --surface-2: Okabe–Ito blue is 3.65:1 on
 * --bg and only 2.88:1 on --surface-2, and the 0.82 opacity every glyph used
 * to carry took group A down to 2.4:1. Group A is drawn solid for that reason;
 * the other two clear 3:1 comfortably with the opacity kept.
 */
export function ShadowPlane2D({
  shadow,
  showGroups,
  retained,
}: ShadowPlane2DProps) {
  const { scaled, extent } = useMemo(() => {
    if (shadow.length === 0) return { scaled: [], extent: 1 };
    const span = shadow.reduce(
      (max, point) => Math.max(max, Math.abs(point.u), Math.abs(point.v)),
      1e-6,
    );
    const scale = (SIZE / 2 - PAD) / span;
    return {
      scaled: shadow.map((point) => ({
        x: SIZE / 2 + point.u * scale,
        y: SIZE / 2 - point.v * scale,
        groupId: point.groupId,
        depth: point.depth,
      })),
      extent: span,
    };
  }, [shadow]);

  const perGroup = useMemo(() => {
    const counts = GROUP_LABELS.map(() => 0);
    for (const point of shadow) {
      counts[point.groupId] = (counts[point.groupId] ?? 0) + 1;
    }
    return counts;
  }, [shadow]);

  const description = showGroups
    ? `Shadow of the cloud, ${shadow.length} points in ${
        GROUP_LABELS.length
      } groups: ${GROUP_LABELS.map(
        (label, index) => `${label} ${perGroup[index] ?? 0} points`,
      ).join(", ")}. This projection keeps ${(retained * 100).toFixed(
        1,
      )} percent of the cloud's spread, and the points span ${extent.toFixed(
        1,
      )} units from the centre.`
    : `Shadow of the cloud, ${shadow.length} points, groups not yet revealed. This projection keeps ${(
        retained * 100
      ).toFixed(1)} percent of the cloud's spread.`;

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        className="w-full max-w-[340px] rounded-md border border-border bg-bg"
        role="img"
        aria-label={description}
      >
        <line
          x1={PAD}
          y1={SIZE / 2}
          x2={SIZE - PAD}
          y2={SIZE / 2}
          stroke="var(--border)"
          strokeDasharray="3 3"
        />
        <line
          x1={SIZE / 2}
          y1={PAD}
          x2={SIZE / 2}
          y2={SIZE - PAD}
          stroke="var(--border)"
          strokeDasharray="3 3"
        />

        {scaled.map((point, index) => {
          const colour = showGroups
            ? (GROUP_COLOUR_TOKENS[point.groupId] ?? UNREVEALED_COLOUR_TOKEN)
            : UNREVEALED_COLOUR_TOKEN;
          const fill = `var(${colour.token})`;

          // Shape carries the group as well as colour.
          if (!showGroups || point.groupId === 0) {
            return (
              <circle
                key={index}
                cx={point.x}
                cy={point.y}
                r={2.6}
                fill={fill}
                // Solid for group A: at 0.82 it measured 2.8:1 on --bg.
                opacity={showGroups ? 1 : 0.82}
              />
            );
          }
          if (point.groupId === 1) {
            return (
              <rect
                key={index}
                x={point.x - 2.3}
                y={point.y - 2.3}
                width={4.6}
                height={4.6}
                fill={fill}
                opacity={0.82}
              />
            );
          }
          return (
            <polygon
              key={index}
              points={`${point.x},${point.y - 3} ${point.x + 2.8},${
                point.y + 2.2
              } ${point.x - 2.8},${point.y + 2.2}`}
              fill={fill}
              opacity={0.82}
            />
          );
        })}
      </svg>

      <figcaption className="mt-1.5 text-xs text-text-muted">
        {showGroups ? (
          <span className="flex flex-wrap gap-x-3 gap-y-1">
            {GROUP_LABELS.map((label, index) => (
              <span key={label} className="inline-flex items-center gap-1">
                <span aria-hidden="true">
                  {index === 0 ? "●" : index === 1 ? "■" : "▲"}
                </span>
                {label} · {perGroup[index] ?? 0}
              </span>
            ))}
          </span>
        ) : (
          <>Groups are hidden until you commit to a projection.</>
        )}
      </figcaption>
    </figure>
  );
}
