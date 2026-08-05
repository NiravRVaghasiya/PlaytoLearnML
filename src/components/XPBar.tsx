"use client";

import { cx, clamp } from "@/lib/utils";
import { useAnimatedNumber } from "./useAnimatedNumber";

export interface XPBarProps {
  /** Player level, derived by `progression.levelFromXp`. */
  level: number;
  /** XP earned inside the current level. */
  xpIntoLevel: number;
  /** XP needed to clear the current level. */
  xpForNextLevel: number;
  /** Set briefly after a gain to flash the delta ("+120 XP"). */
  recentGain?: number | null;
  className?: string;
}

/**
 * XP + level bar (DESIGN.md §6): gold fill, spring animation on gain.
 *
 * Exposed as a real `progressbar` with `aria-valuetext`, so a screen reader
 * hears "Level 3, 140 of 300 XP" rather than a bare percentage.
 */
export function XPBar({
  level,
  xpIntoLevel,
  xpForNextLevel,
  recentGain,
  className,
}: XPBarProps) {
  const safeTarget = Math.max(1, xpForNextLevel);
  const ratio = clamp(xpIntoLevel / safeTarget, 0, 1);
  const animatedRatio = useAnimatedNumber(ratio, { durationMs: 360 });

  return (
    <div className={cx("flex items-center gap-3", className)}>
      <span className="font-mono text-sm whitespace-nowrap text-text-muted">
        Lv {level}
      </span>

      <div
        role="progressbar"
        // A name as well as a value: without this, a screen reader reads
        // "140 of 400" with no clue what is being measured.
        aria-label="Experience points"
        aria-valuemin={0}
        aria-valuemax={safeTarget}
        aria-valuenow={Math.round(xpIntoLevel)}
        aria-valuetext={`Level ${level}, ${Math.round(xpIntoLevel)} of ${safeTarget} XP`}
        className="h-2 min-w-24 flex-1 overflow-hidden rounded-full bg-surface-2"
      >
        <div
          className="h-full rounded-full bg-xp transition-[width] dur-entrance"
          style={{ width: `${animatedRatio * 100}%` }}
        />
      </div>

      <span className="font-mono text-xs whitespace-nowrap text-text-muted">
        {Math.round(xpIntoLevel)}/{safeTarget} XP
      </span>

      {/* Gain is announced politely; the visual flash is secondary.
          Re-keying on `recentGain` replays the CSS animation, so the flash needs
          no timer and no effect. */}
      <span aria-live="polite" className="min-w-16">
        {recentGain ? (
          <span
            key={recentGain}
            className="xp-gain-flash inline-block font-mono text-xs whitespace-nowrap text-xp"
          >
            +{Math.round(recentGain)} XP
          </span>
        ) : null}
      </span>
    </div>
  );
}
