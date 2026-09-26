"use client";

import { useEffect, useId, useState, type ReactNode } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  ArrowRight,
  Award,
  CloudOff,
  RotateCcw,
  Sigma,
} from "lucide-react";
import {
  Button,
  LaneToggle,
  MathDrawer,
  MetricReadout,
  StarRating,
  WhyCard,
  XPBar,
  whyCardHeadline,
  type MetricSpec,
  type WhyCardContent,
} from "@/components";
import { cx } from "@/lib/utils";
import { LaneErrorBoundary } from "./LaneErrorBoundary";
import { badgeLabel, useProgression } from "./progression";
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
 * How long a cleared failure's announcement lingers before the live region
 * empties. A game that clears its failure optimistically and re-derives the
 * same one a moment later (Convolution Kitchen rescores ~1–2 s after every
 * edit) must not make a screen reader re-read a 150-word paragraph each time.
 */
const FAILURE_CLEAR_GRACE_MS = 3000;

/**
 * Delay before a new WhyCard headline is announced. Just after the metric's
 * own 600 ms debounced sentence, so the two queue in a sensible order ("Accuracy
 * rose to 84 percent" … "Good news: that split was worth it") instead of
 * colliding, and so a burst of cards (a drag) announces only the last.
 */
const WHY_ANNOUNCE_DELAY_MS = 700;

/**
 * The fixed frame every GameML game renders inside (DESIGN.md §5).
 *
 * The anatomy is deliberately identical across all 14 games so players build
 * muscle memory: back + title + lane toggle + ƒ Math on top, canvas taking the
 * real estate, controls and the live metric in the right rail, the WhyCard
 * docked below, and XP + stars + retry/next in the footer.
 *
 * On phones (below `lg`) the header scrolls away and ONLY the primary metric
 * pins to the top. Pinning both used to stack a wrapping, two-row header on top
 * of the metric column: the header hid the number it was meant to keep in view,
 * and header + secondary metrics took up to two thirds of the screen. The
 * primary card alone is about 125 px — under a fifth of a 740 px-tall phone,
 * about a third of a landscape one — and only on zoom-short viewports (320 px
 * tall or less, e.g. 400% zoom) does nothing pin.
 *
 * Things worth knowing before you use it:
 *
 * 1. **Only the active lane is mounted.** Game state persists across the toggle
 *    because it lives in the game's Zustand store, not in the lane components
 *    (and the player's code-lane draft in `useCodeLane`'s draft cache).
 *    That keeps a Three.js canvas and its WebGL context from sitting in memory
 *    while the player reads code, and it is why the two-lane rule requires a
 *    shared store.
 *
 * 2. **The shell is presentational.** It computes nothing about the game. If the
 *    metric moves, it is because the game recomputed it from real client-side
 *    ML and passed a new value down. (It does read the progression store, for
 *    the "not saved" hint and the concept badge — never for game logic.)
 *
 * 3. **It owns the announcements.** One persistent assertive region for the
 *    named failure (announced when its text changes or it recurs after a
 *    Retry, not every time a game re-renders the strip) and one polite region
 *    for WhyCard headlines. Both
 *    live outside the regions that go `inert` under the Math dialog.
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
  const controlsHeadingId = useId();

  const lane = controlledLane ?? internalLane;
  const setLane = onLaneChange ?? setInternalLane;

  // A named failure always paints the metric red, even if the game forgot to.
  const metricSpec: MetricSpec = failure
    ? { ...metric, state: metric.state ?? "bad" }
    : metric;

  // ---- Named-failure announcement (pedagogy contract #4, WCAG 4.1.3) ------
  // The visible strip mounts and unmounts with `failure`; the live region never
  // does. It is filled only when the failure's NAME changes, so a game that
  // re-sets the same failure (or clears it and brings it straight back) is
  // not re-read aloud on every edit.
  //
  // Keyed on the name rather than the whole text because some details carry
  // live numbers: Convolution Kitchen's quotes per-class accuracy, which moves
  // on nearly every edit while the diagnosis stays the same. Re-reading that
  // paragraph assertively after each stepper click buries everything else. The
  // strip always shows the current detail; the region says the diagnosis once.
  const failureText = failure ? `${failure.name}. ${failure.detail}` : "";
  const [failureAnnouncement, setFailureAnnouncement] = useState("");
  const [announcedName, setAnnouncedName] = useState("");
  // Adjusting state while rendering (React's "storing information from
  // previous renders" pattern): a NEW failure is announced in this very
  // commit. The same one again changes nothing, so nothing is re-read.
  if (failure && (failure.name !== announcedName || !failureAnnouncement)) {
    setAnnouncedName(failure.name);
    setFailureAnnouncement(failureText);
  }
  // Once the failure clears, empty the region after a grace period — so the
  // same failure coming straight back is not a "change", but a later one is.
  useEffect(() => {
    if (failureText || !failureAnnouncement) return;
    const timer = setTimeout(
      () => setFailureAnnouncement(""),
      FAILURE_CLEAR_GRACE_MS,
    );
    return () => clearTimeout(timer);
  }, [failureText, failureAnnouncement]);
  // Retry is a fresh attempt, so the grace does not apply to it: the same
  // failure straight after a Retry is a second loss, and is announced as one
  // (a synchronous "Run to end" can re-derive identical words within 3 s).
  const retry = onRetry
    ? () => {
        setFailureAnnouncement("");
        onRetry();
      }
    : undefined;

  // ---- WhyCard headline announcement (WCAG 4.1.3 status messages) ---------
  // The card itself re-keys on every action (that restarts its fade-in), and a
  // live region that is remounted with its content already inside is often not
  // announced. So the headline goes through this persistent region instead.
  const whyKey = whyCard?.key ?? null;
  const whyHeadline = whyCard ? whyCardHeadline(whyCard) : "";
  // A "bad" card during a failure is the failure itself — the assertive region
  // already says it, so don't say it twice.
  const whyDuplicatesFailure = Boolean(failure) && whyCard?.tone === "bad";
  const [whyAnnouncement, setWhyAnnouncement] = useState("");
  useEffect(() => {
    if (!whyKey || whyDuplicatesFailure) return;
    const timer = setTimeout(
      () => setWhyAnnouncement(whyHeadline),
      WHY_ANNOUNCE_DELAY_MS,
    );
    return () => clearTimeout(timer);
    // Strings compare by value, so a re-render with the same card doesn't
    // restart the delay; a new key (a new action) does, even with the same
    // title — which then leaves the region's text unchanged, so a repeated
    // identical headline is not re-read.
  }, [whyKey, whyHeadline, whyDuplicatesFailure]);

  // ---- Progression extras -------------------------------------------------
  const syncError = useProgression((state) => state.syncError);
  const badgeEarned = useProgression((state) => state.badges.includes(slug));
  const badge = badgeEarned ? badgeLabel(slug) : null;

  // A "+N XP" belongs to the game that earned it. Leaving the game drops it, so
  // it can't replay as a phantom award on arrival in the next one.
  useEffect(() => () => useProgression.getState().dismissGain(), []);

  const laneLabel = lane === "visual" ? "visual lane" : "code lane";

  return (
    <div className="flex min-h-dvh flex-col bg-bg" data-game={slug}>
      {/* ---- Top bar ----
          Sticky only at lg. Below that it wraps to two or three rows, and a
          sticky header that tall covered the pinned metric beneath it. */}
      <header
        inert={mathOpen}
        className="z-30 border-b border-border bg-bg/95 backdrop-blur lg:sticky lg:top-0"
      >
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
            // Browser checks open the drawer through this (see MathDrawer).
            data-testid="math-open"
            icon={<Sigma className="size-4" />}
          >
            Math
          </Button>
        </div>
      </header>

      {/* ---- Canvas + rail ----
          Mobile order: metric (primary pinned) → canvas → controls.
          Desktop: canvas left; controls above metric in the right rail. */}
      <div inert={mathOpen} className="mx-auto w-full max-w-7xl flex-1 px-4 py-4">
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px] lg:grid-rows-[auto_auto]">
          {/* `contents` below lg: the primary and the secondaries become grid
              items of their own, so the primary can stick for the whole height
              of the grid while the secondaries scroll with the page. At lg it is
              the ordinary right-rail cell it always was. */}
          <div className="contents lg:col-start-2 lg:row-start-2 lg:block">
            <section
              aria-label="Live metric"
              className={cx(
                // DESIGN.md §5: "the metric readout is never hidden".
                "order-1 sticky top-0 z-20 bg-bg lg:static lg:order-none",
                // Zoom/reflow only. At 400% zoom the viewport is 256–270 px
                // tall (1280×1024 or 1920×1080 screens), and a pinned card
                // would leave no room for the game. Landscape phones are
                // 340–412 px tall with the browser's toolbars, so they keep
                // the pin: about 125 px, leaving 215+ px of game below it.
                // globals.css drops its scroll-padding at the same height (a
                // test fails if the two drift apart).
                "[@media(max-height:320px)]:static",
              )}
            >
              <MetricReadout {...metricSpec} />
            </section>

            {secondaryMetrics && secondaryMetrics.length > 0 ? (
              <section
                aria-label="More metrics"
                className="order-1 grid grid-cols-2 gap-2 lg:order-none lg:mt-2 lg:grid-cols-1"
              >
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
              </section>
            ) : null}
          </div>

          <main
            id="game-canvas"
            tabIndex={-1}
            aria-label={`${title} — ${lane === "visual" ? "visual" : "code"} lane`}
            className="order-2 min-h-[420px] min-w-0 rounded-md border border-border bg-surface p-4 lg:order-none lg:col-start-1 lg:row-start-1 lg:row-span-2"
          >
            <LaneErrorBoundary key={lane} laneLabel={laneLabel}>
              {lane === "visual" ? visual : code}
            </LaneErrorBoundary>
          </main>

          {/* Stays a <div> immediately after <main>: the browser playthroughs
              find the rail as `#game-canvas + div`. role=region makes it a
              landmark, so landmark navigation reaches the controls. */}
          <div
            role="region"
            aria-labelledby={controlsHeadingId}
            className="order-3 rounded-md border border-border bg-surface p-4 lg:order-none lg:col-start-2 lg:row-start-1"
          >
            <h2
              id={controlsHeadingId}
              className="mb-3 text-sm font-semibold tracking-wide text-text-muted uppercase"
            >
              Controls
            </h2>
            {controls}
          </div>
        </div>

        {/* ---- Named failure strip (pedagogy contract #4) ----
            Inline, not modal: retry is one click and the game stays on screen.
            Announced through the persistent region at the end of the shell. */}
        {failure ? (
          <div
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
            {retry ? (
              <Button
                variant="primary"
                onClick={retry}
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
      <footer inert={mathOpen} className="border-t border-border bg-surface/80">
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
              <div className="inline-flex flex-wrap items-center">
                <StarRating
                  earned={progress.stars}
                  criteria={progress.starCriteria}
                />
                {/* Persistent polite region, so earning the concept badge (spec
                    §4) — or losing the ability to save — is announced when it
                    happens. Spacing lives on the children, so the empty region
                    adds no gap. */}
                <span role="status" className="inline-flex flex-wrap items-center">
                  {badge ? (
                    <span className="ml-3 inline-flex items-center gap-1.5 text-xs text-xp">
                      <Award aria-hidden="true" className="size-4" />
                      {badge}
                    </span>
                  ) : null}
                  {syncError ? (
                    <span className="ml-3 inline-flex items-center gap-1.5 text-xs text-text-muted">
                      <CloudOff aria-hidden="true" className="size-4 text-warn" />
                      Progress isn&apos;t being saved on this device.
                    </span>
                  ) : null}
                </span>
              </div>
            </>
          ) : (
            <div className="flex-1" />
          )}

          {retry ? (
            <Button
              variant="ghost"
              size="sm"
              onClick={retry}
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

      {/* ---- Announcements ----
          Outside the header/content/footer so they keep working (and are not
          hidden from AT) while those are inert under the Math dialog. */}
      <div role="alert" className="sr-only-live">
        {failureAnnouncement}
      </div>
      <p className="sr-only-live" aria-live="polite" aria-atomic="true">
        {whyAnnouncement}
      </p>

      <MathDrawer
        open={mathOpen}
        onClose={() => setMathOpen(false)}
        math={math}
        title={title}
      />
    </div>
  );
}
