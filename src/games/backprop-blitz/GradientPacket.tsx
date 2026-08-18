"use client";

export interface GradientPacketProps {
  from: { x: number; y: number };
  to: { x: number; y: number };
  value: number;
  correct: boolean;
}

/**
 * The gradient in transit (spec: `<GradientPacket>`).
 *
 * A labelled token that slides from a node to one of its producers, carrying the
 * number it delivers. The spec asks for a pinball, chain-reaction feel and this is
 * the honest version of it: the packet moves BACKWARD along a forward edge, which is
 * the one visual fact about backpropagation worth animating. Everything else about
 * the algorithm is arithmetic, and arithmetic is better read than watched.
 *
 * Rendered only when the player has motion enabled — the caller checks
 * `useReducedMotion` and omits this component entirely rather than animating to a
 * zero duration, so nothing moves and nothing is lost: the number it carries is
 * already printed on the edge.
 */
export function GradientPacket({
  from,
  to,
  value,
  correct,
}: GradientPacketProps) {
  const path = `M${from.x},${from.y} L${to.x},${to.y}`;
  const stroke = correct ? "var(--correct)" : "var(--wrong)";

  return (
    <g aria-hidden="true">
      <path id="packet-path" d={path} fill="none" stroke="none" />
      <g>
        <animateMotion dur="0.85s" repeatCount="1" fill="freeze" path={path} />
        <circle r={13} fill="var(--bg)" stroke={stroke} strokeWidth={2} />
        <text
          textAnchor="middle"
          y={3}
          className="fill-[var(--text)] text-[8px] font-semibold tabular-nums"
        >
          {value.toFixed(1)}
        </text>
      </g>
    </g>
  );
}
