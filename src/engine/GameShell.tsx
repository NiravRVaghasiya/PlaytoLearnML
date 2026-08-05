"use client";

import { useState, type ReactNode } from "react";
import Link from "next/link";
import { ArrowLeft, ArrowRight, RotateCcw, Sigma } from "lucide-react";
import {
  Button,
  LaneToggle,
  MathDrawer,
  MetricReadout,
  StarRating,
  WhyCard,
  XPBar,
  type MetricSpec,
  type WhyCardContent,
} from "@/components";
import { cx } from "@/lib/utils";
import type { Lane, MathReveal, NamedFailure, StarCount } from "./types";

export interface GameShellProgress {
  level: number;
  xpIntoLevel: number;
  xpForNextLevel: number;
  stars: StarCount;
  /** Set briefly after an award so the XP bar can flash the delta. */
  recentGain?: number | null;
  /** What each mastery star requires, for the tooltip. */
  starCriteria?: readonly string[];
}

export interface GameShellProps {
  /** Catalog slug — used for progression keys and analytics. */
  slug: string;
  title: string;
  backHref?: string;

  /**
   * The always-visible live metric (pedagogy contract #3). Required, because a
   * game without one cannot satisfy the contract.
   */
  metric: MetricSpec;
  /** Extra readouts (complexity penalty, train/val pair, iteration count). */
  secondaryMetrics?: readonly MetricSpec[];

  /** Reveal-the-math payload. Must match the game's real `ml.ts`. */
  math: MathReveal;

  /** Right-rail controls: sliders, dials, palettes. */
  controls: ReactNode;
  /** The no-code lane. */
  visual: ReactNode;
  /** The code lane, driving the same store. */
  code: ReactNode;

  /** Explanation of the player's last action (pedagogy contract #1 on screen). */
  whyCard?: WhyCardContent | null;

  /**
   * The named ML failure (pedagogy contract #4). When set, the metric turns red
   * and a named-failure strip appears with a one-click retry. There is
   * deliberately no blocking "Game Over" screen (DESIGN.md §8).
   */
  failure?: NamedFailure | null;

  progress?: GameShellProgress;

  /** Controlled lane. Omit both to let the shell manage it internally. */
  lane?: Lane;
  onLaneChange?: (lane: Lane) => void;
  codeLaneDisabled?: boolean;
  codeLaneDisabledReason?: string;

  onRetry?: () => void;
  onNext?: () => void;
  nextLabel?: string;
}

/**
 * The fixed frame every GameML game renders inside (DESIGN.md §5).
 *
 * The anatomy is deliberately identical across all 14 games so players build
 * muscle memory: back + title + lane toggle + ƒ Math on top, canvas taking the
 * real estate, controls and the live metric in the right rail (metric sticky to
 * the top on mobile so it is never scrolled away), the WhyCard docked below, and
 * XP + stars + retry/next in the footer.
 *
 * Two things worth knowing before you use it:
 *
 * 1. **Only the active lane is mounted.** Game state persists across the toggle
 *    because it lives in the game's Zustand store, not in the lane components.
 *    That keeps a Three.js or Phaser canvas from sitting in memory while the
 *    player reads code, and it is why the two-lane rule requires a shared store.
 *
 * 2. **The shell is presentational.** It computes nothing about the game. If the
 *    metric moves, it is because the game recomputed it from real client-side
 *    ML and passed a new value down.
 */
export function GameShell({
  slug,
  title,
  backHref = "/",
  metric,
  secondaryMetrics,
  math,
  controls,
  visual,
  code,
  whyCard,
  failure,
  progress,
  lane: controlledLane,
  onLaneChange,
  codeLaneDisabled,
  codeLaneDisabledReason,
  onRetry,
  onNext,
  nextLabel = "Next",
}: GameShellProps) {
  const [internalLane, setInternalLane] = useState<Lane>("visual");
  const [mathOpen, setMathOpen] = useState(false);

  const lane = controlledLane ?? internalLane;
  const setLane = onLaneChange ?? setInternalLane;

  // A named failure always paints the metric red, even if the game forgot to.
  const metricSpec: MetricSpec = failure
    ? { ...metric, state: metric.state ?? "bad" }
    : metric;

  return (
    <div className="flex min-h-dvh flex-col bg-bg" data-game={slug}>
      {/* ---- Top bar ---- */}
      <header className="sticky top-0 z-30 border-b border-border bg-bg/95 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-3 px-4 py-3">
          <Link
            href={backHref}
            className="inline-flex min-h-11 items-center gap-2 rounded-md px-2 text-sm text-text-muted hover:text-text"
          >
            <ArrowLeft aria-hidden="true" className="size-4" />
            Back
          </Link>

          <h1 className="mr-auto font-display text-lg font-semibold">{title}</h1>

          <LaneToggle
            value={lane}
            onChange={setLane}
            codeDisabled={codeLaneDisabled}
            codeDisabledReason={codeLaneDisabledReason}
          />

          <Button
            variant="secondary"
            size="sm"
            onClick={() => setMathOpen(true)}
            aria-haspopup="dialog"
            aria-expanded={mathOpen}
            icon={<Sigma className="size-4" />}
          >
            Math
          </Button>
        </div>
      </header>

      {/* ---- Canvas + rail ----
          Mobile order: metric (sticky) → canvas → controls.
          Desktop: canvas left; controls above metric in the right rail. */}
      <div className="mx-auto w-full max-w-7xl flex-1 px-4 py-4">
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px] lg:grid-rows-[auto_auto]">
          <div
            className={cx(
              "order-1 lg:order-none lg:col-start-2 lg:row-start-2",
              // DESIGN.md §5: "the metric readout is never hidden".
              "sticky top-16 z-20 lg:static",
            )}
          >
            <MetricReadout {...metricSpec} />

            {secondaryMetrics && secondaryMetrics.length > 0 ? (
              <div className="mt-2 grid grid-cols-2 gap-2 lg:grid-cols-1">
                {secondaryMetrics.map((spec) => (
                  <MetricReadout
                    key={spec.label}
                    {...spec}
                    size="md"
                    // Only the primary metric announces, or screen readers get
                    // three sentences per action.
                    announce={false}
                  />
                ))}
              </div>
            ) : null}
          </div>

          <main
            id="game-canvas"
            tabIndex={-1}
            aria-label={`${title} — ${lane === "visual" ? "visual" : "code"} lane`}
            className="order-2 min-h-[420px] min-w-0 rounded-md border border-border bg-surface p-4 lg:order-none lg:col-start-1 lg:row-start-1 lg:row-span-2"
          >
            {lane === "visual" ? visual : code}
          </main>

          <div className="order-3 rounded-md border border-border bg-surface p-4 lg:order-none lg:col-start-2 lg:row-start-1">
            <h2 className="mb-3 text-sm font-semibold tracking-wide text-text-muted uppercase">
              Controls
            </h2>
            {controls}
          </div>
        </div>

        {/* ---- Named failure strip (pedagogy contract #4) ----
            Inline, not modal: retry is one click and the game stays on screen. */}
        {failure ? (
          <div
            role="alert"
            data-testid="named-failure"
            className="mt-4 flex flex-wrap items-center gap-3 rounded-md border border-wrong bg-wrong/10 p-4"
          >
            <div className="min-w-0 flex-1">
              <p className="font-display font-semibold text-wrong">
                {failure.name}
              </p>
              <p className="mt-1 font-mono text-sm text-text">
                {failure.detail}
              </p>
            </div>
            {onRetry ? (
              <Button
                variant="primary"
                onClick={onRetry}
                icon={<RotateCcw className="size-4" />}
              >
                Retry
              </Button>
            ) : null}
          </div>
        ) : null}

        {/* ---- WhyCard dock ---- */}
        <section aria-label="Why did that happen?" className="mt-4">
          <WhyCard content={whyCard ?? null} />
        </section>
      </div>

      {/* ---- Footer: XP, stars, retry/next ---- */}
      <footer className="border-t border-border bg-surface/80">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-4 px-4 py-3">
          {progress ? (
            <>
              <XPBar
                className="min-w-56 flex-1"
                level={progress.level}
                xpIntoLevel={progress.xpIntoLevel}
                xpForNextLevel={progress.xpForNextLevel}
                recentGain={progress.recentGain}
              />
              <StarRating
                earned={progress.stars}
                criteria={progress.starCriteria}
              />
            </>
          ) : (
            <div className="flex-1" />
          )}

          {onRetry ? (
            <Button
              variant="ghost"
              size="sm"
              onClick={onRetry}
              icon={<RotateCcw className="size-4" />}
            >
              Retry
            </Button>
          ) : null}

          {onNext ? (
            <Button variant="primary" size="sm" onClick={onNext}>
              {nextLabel}
              <ArrowRight aria-hidden="true" className="size-4" />
            </Button>
          ) : null}
        </div>
      </footer>

      <MathDrawer
        open={mathOpen}
        onClose={() => setMathOpen(false)}
        math={math}
        title={title}
      />
    </div>
  );
}
