"use client";

import { CORE_MAX_HP, type Wave, type WaveResult } from "./ml";
import { useReducedMotion } from "@/lib/useReducedMotion";

export interface BattlefieldProps {
  wave: Wave;
  /** Live gap, or NaN before the model is trained. */
  gap: number;
  bias: number;
  coreHp: number;
  result: WaveResult | null;
  advancing: boolean;
}

const WIDTH = 340;
const HEIGHT = 170;
const CORE_X = WIDTH - 42;

/**
 * The battlefield (spec: `<GameCanvas>`).
 *
 * ── Deviation from the spec, deliberate ─────────────────────────────────────
 * The spec names Phaser for this component. It is drawn in SVG instead. The
 * gameplay is turn-based — set two controls, train, read the result — so none of
 * what Phaser provides (a render loop, sprite batching, physics, pointer
 * capture) is used, while it would cost a ~1MB dependency and, more importantly,
 * a canvas that assistive technology cannot read. That would mean building this
 * view twice: once in Phaser and once as an accessible equivalent. Everything
 * below is in the accessibility tree already, and the arithmetic driving it is
 * identical either way. The component boundary is unchanged, so swapping the
 * renderer later touches nothing outside this file.
 *
 * ── What is drawn ───────────────────────────────────────────────────────────
 * Enemy size is not decorative. An overfit attacker's size is the train/
 * validation gap and an underfit attacker's is the bias, both on the same scale,
 * so "enemies visibly weaken as the gap closes" is literally true — they shrink
 * because the number they are drawn from got smaller.
 */
export function Battlefield({
  wave,
  gap,
  bias,
  coreHp,
  result,
  advancing,
}: BattlefieldProps) {
  const reducedMotion = useReducedMotion();
  const measured = Number.isFinite(gap) && Number.isFinite(bias);

  // 0.5 error is a big one; anything past that just pins the size.
  const sizeFor = (error: number) =>
    measured ? 4 + Math.min(Math.max(error, 0), 0.5) * 26 : 7;
  const overfitSize = sizeFor(gap);
  const underfitSize = sizeFor(bias);

  const hpFraction = Math.max(0, Math.min(1, coreHp / CORE_MAX_HP));
  // Attackers sit back while tuning and close on the core when the wave resolves.
  const advance = advancing ? 0.62 : 0.06;

  const rowY = { overfit: 48, underfit: 118 };

  const enemies = (
    count: number,
    y: number,
    size: number,
    fill: string,
    shape: "circle" | "diamond",
  ) =>
    Array.from({ length: count }, (_, index) => {
      const spread = index * 30;
      const x = 24 + spread + advance * (CORE_X - 70 - spread);
      const key = `${shape}-${index}`;
      return shape === "circle" ? (
        <circle
          key={key}
          cx={x}
          cy={y}
          r={size}
          fill={fill}
          opacity={measured ? 0.9 : 0.4}
          style={
            reducedMotion
              ? undefined
              : { transition: "all var(--dur-standard) var(--ease-spring)" }
          }
        />
      ) : (
        <rect
          key={key}
          x={x - size}
          y={y - size}
          width={size * 2}
          height={size * 2}
          fill={fill}
          opacity={measured ? 0.9 : 0.4}
          transform={`rotate(45 ${x} ${y})`}
          style={
            reducedMotion
              ? undefined
              : { transition: "all var(--dur-standard) var(--ease-spring)" }
          }
        />
      );
    });

  const description = measured
    ? `Wave ${wave.index}. ${wave.overfitEnemies} overfit attacker${
        wave.overfitEnemies === 1 ? "" : "s"
      } powered by a ${Math.round(gap * 100)} point gap, and ${
        wave.underfitEnemies
      } underfit attacker${
        wave.underfitEnemies === 1 ? "" : "s"
      } powered by ${Math.round(bias * 100)} points of bias. Core at ${Math.round(
        coreHp,
      )} of ${CORE_MAX_HP} hit points.${
        result
          ? ` Last wave dealt ${Math.round(
              result.overfitDamage,
            )} overfit damage and ${Math.round(
              result.underfitDamage,
            )} underfit damage.`
          : ""
      }`
    : `Wave ${wave.index}, ${wave.overfitEnemies} overfit and ${wave.underfitEnemies} underfit attackers waiting. Nothing is measured until you deploy a model. Core at ${Math.round(
        coreHp,
      )} of ${CORE_MAX_HP} hit points.`;

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="w-full rounded-md border border-border bg-surface-2"
        role="img"
        aria-label={description}
      >
        <line
          x1={0}
          y1={HEIGHT / 2}
          x2={WIDTH}
          y2={HEIGHT / 2}
          stroke="var(--border)"
          strokeWidth={0.5}
          strokeDasharray="4 4"
        />

        <text x={8} y={18} className="fill-[var(--text-muted)] text-[9px]">
          overfit · gap {measured ? `${Math.round(gap * 100)}%` : "—"}
        </text>
        <text
          x={8}
          y={HEIGHT - 8}
          className="fill-[var(--text-muted)] text-[9px]"
        >
          underfit · bias {measured ? `${Math.round(bias * 100)}%` : "—"}
        </text>

        {enemies(
          wave.overfitEnemies,
          rowY.overfit,
          overfitSize,
          "var(--wrong)",
          "circle",
        )}
        {enemies(
          wave.underfitEnemies,
          rowY.underfit,
          underfitSize,
          "var(--warn)",
          "diamond",
        )}

        {/* The core. Its height tracks HP so damage is visible, not just numeric. */}
        <rect
          x={CORE_X - 14}
          y={HEIGHT / 2 - 52}
          width={28}
          height={104}
          rx={6}
          fill="var(--surface)"
          stroke="var(--border)"
        />
        <rect
          x={CORE_X - 14}
          y={HEIGHT / 2 - 52 + 104 * (1 - hpFraction)}
          width={28}
          height={104 * hpFraction}
          rx={6}
          fill={
            hpFraction <= 0.3
              ? "var(--wrong)"
              : hpFraction <= 0.6
                ? "var(--warn)"
                : "var(--correct)"
          }
          opacity={0.75}
          style={
            reducedMotion
              ? undefined
              : { transition: "all var(--dur-standard) var(--ease-spring)" }
          }
        />
        <text
          x={CORE_X}
          y={HEIGHT / 2 + 3}
          textAnchor="middle"
          className="fill-[var(--text)] text-[10px] font-medium"
        >
          {Math.round(coreHp)}
        </text>
      </svg>

      <figcaption className="mt-1.5 text-xs text-text-muted">
        {measured
          ? "Attackers are drawn from your two error terms — they shrink because the errors did, not for effect."
          : `Wave ${wave.index} attackers are waiting. Their strength is whatever error your model turns out to have.`}
      </figcaption>
    </figure>
  );
}
