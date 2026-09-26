"use client";

import { useCallback, useId } from "react";
import { Ban, Scissors, Sigma, ThumbsUp } from "lucide-react";
import { Button } from "@/components";
import { cx } from "@/lib/utils";
import {
  CLEANING_ACTIONS,
  FEATURES,
  isOutOfRange,
  type CleaningAction,
  type Row,
} from "./ml";
import { useDataDetoxStore } from "./store";

/**
 * Data Detox — the no-code lane (spec: `<ConveyorBelt>`, `<DirtyRow>`,
 * `<CleaningBin>` ×4).
 *
 * ── The belt advances on decision, not on a clock ────────────────────────────
 * The spec describes rows flowing down a conveyor. This implements the belt as a
 * queue that moves when the player sorts a row, rather than on a timer. That is a
 * deliberate departure: a timed belt turns a lesson about preprocessing tradeoffs
 * into a reaction test, locks out anyone using a screen reader or switch access,
 * and makes the whole thing untestable. The queue still reads as a belt — you can
 * see what's coming — without penalising thinking.
 *
 * Sorting is available three ways: click a bin, Tab to a bin and press Enter, or
 * press 1–4 while focus is on the belt or a bin. Dirt is signalled by an icon
 * and a text label as well as colour and size (DESIGN.md §9).
 *
 * ── Why the number keys only work inside the lane ────────────────────────────
 * They used to listen on the whole document, which is a single-character
 * shortcut with no way to turn it off (WCAG 2.1.4): a 2 pressed on the Math
 * dialog's Close button, or on the Back link, silently sorted the row behind it,
 * and holding 4 dropped rows at the key-repeat rate. Scoping them to focus
 * within the lane is the criterion's "active only on focus" option, and loses
 * nothing — the belt itself is focusable, and a bin keeps focus after a click.
 */

const BIN_META: Record<
  CleaningAction,
  { label: string; hint: string; key: string; Icon: typeof ThumbsUp }
> = {
  keep: {
    label: "Keep",
    hint: "Use as-is. Blanks become 0.",
    key: "1",
    Icon: ThumbsUp,
  },
  impute: {
    label: "Impute",
    hint: "Fill blanks with the median.",
    key: "2",
    Icon: Sigma,
  },
  cap: {
    label: "Cap",
    hint: "Clamp extremes into range.",
    key: "3",
    Icon: Scissors,
  },
  drop: {
    label: "Drop",
    hint: "Discard the row entirely.",
    key: "4",
    Icon: Ban,
  },
};

/** One-line spoken summary of a row, for the belt's live region. */
function describeRow(row: Row, position: number): string {
  const cells = FEATURES.map((feature) => {
    const value = row.features[feature];
    if (value === null) return `${feature} blank`;
    return `${feature} ${value.toFixed(2)}${isOutOfRange(value) ? ", out of range" : ""}`;
  });
  return `Now sorting row ${position}: ${cells.join("; ")}.`;
}

function RowCard({
  row,
  position,
  isHead,
}: {
  row: Row;
  position: number;
  isHead: boolean;
}) {
  return (
    <li
      className={cx(
        "rounded-md border p-3 transition-colors dur-standard",
        isHead
          ? "border-primary bg-surface-2"
          : "border-border bg-surface/60 opacity-90",
      )}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className="font-mono text-xs text-text-muted">
          {isHead ? "now sorting" : `#${position}`}
        </span>
        <span className="flex gap-2 text-xs">
          {/* Text in --text, not --wrong: red text on a red tint measures
              3.4–3.9:1 on these cards. The red lives in the border and tint. */}
          {row.isNull ? (
            <span className="rounded-full border border-wrong bg-wrong/15 px-2 font-medium text-text">
              blank cell
            </span>
          ) : null}
          {row.isOutlier ? (
            <span className="rounded-full bg-warn/15 px-2 font-medium text-warn">
              out of range
            </span>
          ) : null}
          {!row.isNull && !row.isOutlier ? (
            <span className="rounded-full bg-correct/15 px-2 font-medium text-correct">
              clean
            </span>
          ) : null}
          {row.playerAction ? (
            <span className="rounded-full bg-surface-2 px-2 font-mono text-text-muted">
              {BIN_META[row.playerAction].label}
            </span>
          ) : null}
        </span>
      </div>

      <dl className="mt-2 grid grid-cols-3 gap-2">
        {FEATURES.map((feature) => {
          const value = row.features[feature];
          const missing = value === null;
          // "Oversized" (spec) means physically impossible, not merely in a
          // tail: the percentile band this used to use missed a third of the
          // real outliers and flagged clean readings.
          const extreme = isOutOfRange(value);

          return (
            <div key={feature}>
              <dt className="text-xs text-text-muted">{feature}</dt>
              <dd
                className={cx(
                  "font-mono tabular-nums",
                  missing
                    ? // The spec's red glow, as a static outline and tint with
                      // readable text — no blink, which dipped this word to
                      // 1.8:1 at the bottom of every pulse.
                      "w-fit rounded-sm bg-wrong/15 px-1 text-text shadow-[0_0_6px_var(--wrong)] ring-1 ring-wrong"
                    : extreme
                      ? "text-lg font-bold text-warn"
                      : "text-text",
                )}
              >
                {missing ? "blank" : value.toFixed(2)}
              </dd>
            </div>
          );
        })}
      </dl>
    </li>
  );
}

export function VisualLane() {
  const rows = useDataDetoxStore((s) => s.rows);
  const cursor = useDataDetoxStore((s) => s.cursor);
  const pipeline = useDataDetoxStore((s) => s.pipeline);
  const sortRow = useDataDetoxStore((s) => s.sortRow);
  const skip = useDataDetoxStore((s) => s.skip);
  const hintId = useId();

  const head = rows[cursor];
  const upcoming = rows.slice(cursor + 1, cursor + 4);

  // Number keys sort the head of the belt — but only while focus is inside
  // this lane (see the docblock).
  const onKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      // One key press, one decision. Holding a key must not empty the belt.
      if (event.repeat) return;
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable ||
          target.closest('[role="dialog"], [aria-modal="true"]'))
      ) {
        return;
      }

      const index = ["1", "2", "3", "4"].indexOf(event.key);
      if (index >= 0) {
        event.preventDefault();
        sortRow(CLEANING_ACTIONS[index]!);
      }
    },
    [sortRow],
  );

  const decided = rows.length - pipeline.undecided;
  const position = decided + (head?.playerAction === null ? 1 : 0);

  return (
    // Not interactive itself: keydown bubbles here from the focusable belt and
    // the bin buttons inside it, which is exactly the scope we want.
    <div onKeyDown={onKeyDown} className="flex h-full min-h-0 flex-col gap-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p id={hintId} className="text-sm text-text-muted">
          Row {position} of {rows.length}. Click a bin, or press 1–4 while the
          belt or a bin has focus.
        </p>
        <p className="font-mono text-xs text-text-muted">
          {pipeline.undecided} left on the belt
        </p>
      </div>

      {/* The belt. Focusable so keyboard users have somewhere for 1–4 to land. */}
      <section
        aria-label="Conveyor belt"
        aria-describedby={hintId}
        tabIndex={0}
        className="min-h-0 rounded-md focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
      >
        <ol className="flex flex-col gap-2">
          {head ? <RowCard row={head} position={cursor + 1} isHead /> : null}
          {upcoming.map((row, offset) => (
            <RowCard
              key={row.id}
              row={row}
              position={cursor + offset + 2}
              isHead={false}
            />
          ))}
        </ol>
      </section>

      {/* The head row, spoken. Focus stays on the bin after a sort, so without
          this a screen-reader user hears nothing about what arrived next. */}
      <p className="sr-only" aria-live="polite" aria-atomic="true">
        {head && head.playerAction === null ? describeRow(head, position) : ""}
      </p>

      {/* The four bins. */}
      <section aria-label="Cleaning bins" className="mt-auto">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {CLEANING_ACTIONS.map((action) => {
            const meta = BIN_META[action];
            return (
              <Button
                key={action}
                variant={action === "drop" ? "danger" : "secondary"}
                onClick={() => sortRow(action)}
                disabled={!head}
                className="h-auto flex-col items-start gap-1 py-2 text-left"
                icon={<meta.Icon className="size-4" />}
              >
                {/* No `opacity` on this text. Dimming it to suggest secondary
                    importance drops it below AA contrast, especially the white
                    label on the red Drop bin. Size and weight already do that job
                    without touching the contrast ratio. */}
                <span className="flex w-full items-baseline justify-between gap-2">
                  {meta.label}
                  <kbd className="font-mono text-xs">{meta.key}</kbd>
                </span>
                <span className="text-xs font-normal">{meta.hint}</span>
              </Button>
            );
          })}
        </div>
        <div className="mt-2 flex items-center justify-between gap-2">
          <p className="font-mono text-xs text-text-muted">
            kept {pipeline.kept} · dropped {pipeline.dropped} · blanks left{" "}
            {pipeline.keptWithNaiveFill} · extremes left{" "}
            {pipeline.keptWithOutlier}
          </p>
          <Button variant="ghost" size="sm" onClick={skip} disabled={!head}>
            Come back to this row
          </Button>
        </div>
      </section>
    </div>
  );
}
