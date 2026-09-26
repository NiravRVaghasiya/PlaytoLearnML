"use client";

import { useEffect, useRef } from "react";

export interface GradientPacketProps {
  from: { x: number; y: number };
  to: { x: number; y: number };
  value: number;
  correct: boolean;
}

/** Duration of one flight. Short: the number is already printed on the edge. */
const FLIGHT = "0.85s";

/** Start a SMIL animation now. A no-op where SMIL is absent (jsdom, say). */
function begin(element: SVGElement | null) {
  (element as SVGAnimationElement | null)?.beginElement?.();
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
 * ── Why the animation is started by hand ────────────────────────────────────
 * The first version declared `<animateMotion dur="0.85s">` and never moved.
 * SMIL timing is anchored to the SVG document's timeline, not to when an element
 * is inserted, so a packet added after the first choice was already "finished"
 * when it appeared — frozen at its end point, and painted under the node it had
 * arrived at, so it was invisible too. Now both animations wait (`indefinite`) and
 * are begun on mount; the caller keys the packet on the choice, so every choice is
 * a fresh mount and a fresh flight. The group rests at opacity 0 and the fade
 * lifts it only while it travels, so a packet that is not flying is never a stray
 * disc in the corner of the graph.
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
  const motionRef = useRef<SVGElement | null>(null);
  const fadeRef = useRef<SVGElement | null>(null);
  const path = `M${from.x},${from.y} L${to.x},${to.y}`;
  const stroke = correct ? "var(--correct)" : "var(--wrong)";

  useEffect(() => {
    begin(motionRef.current);
    begin(fadeRef.current);
  }, []);

  return (
    <g aria-hidden="true" pointerEvents="none" opacity={0}>
      <animate
        ref={fadeRef}
        attributeName="opacity"
        values="1;1;0"
        keyTimes="0;0.8;1"
        dur={FLIGHT}
        begin="indefinite"
        fill="remove"
      />
      <g>
        <animateMotion
          ref={motionRef}
          dur={FLIGHT}
          begin="indefinite"
          fill="freeze"
          path={path}
        />
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
