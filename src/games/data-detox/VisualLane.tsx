"use client";

import { useCallback, useEffect, useRef } from "react";
import { Ban, Scissors, Sigma, ThumbsUp } from "lucide-react";
import { Button } from "@/components";
import { cx } from "@/lib/utils";
import {
  CLEANING_ACTIONS,
  FEATURES,
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
 * Sorting is available three ways: click a bin, press 1–4, or Tab to a bin and
 * press Enter. Dirt is signalled by an icon and a text label as well as colour and
 * size (DESIGN.md §9).
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

/** Is this observed value far outside the normal band? */
function isExtreme(value: number | null, low: number, high: number): boolean {
  return value !== null && (value > high || value < low);
}

function RowCard({
  row,
  position,
  isHead,
  stats,
}: {
  row: Row;
  position: number;
  isHead: boolean;
  stats: { low: Record<string, number>; high: Record<string, number> };
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
          {row.isNull ? (
            <span className="rounded-full bg-wrong/15 px-2 font-medium text-wrong">
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
          const extreme = isExtreme(
            value,
            stats.low[feature] ?? 0,
            stats.high[feature] ?? 1,
          );

          return (
            <div key={feature}>
              <dt className="text-xs text-text-muted">{feature}</dt>
              <dd
                className={cx(
                  "font-mono tabular-nums",
                  missing
                    ? "flash-wrong text-wrong"
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
  const stats = useDataDetoxStore((s) => s.stats);
  const pipeline = useDataDetoxStore((s) => s.pipeline);
  const sortRow = useDataDetoxStore((s) => s.sortRow);
  const skip = useDataDetoxStore((s) => s.skip);

  const head = rows[cursor];
  const upcoming = rows.slice(cursor + 1, cursor + 4);
  const beltRef = useRef<HTMLDivElement>(null);

  // Number keys sort the head of the belt. Scoped to this lane so it can't
  // hijack typing in the code lane.
  const onKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable)
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

  useEffect(() => {
    const node = beltRef.current;
    if (!node) return;
    const owner = node.ownerDocument;
    owner.addEventListener("keydown", onKeyDown);
    return () => owner.removeEventListener("keydown", onKeyDown);
  }, [onKeyDown]);

  const decided = rows.length - pipeline.undecided;

  return (
    <div ref={beltRef} className="flex h-full min-h-0 flex-col gap-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-sm text-text-muted">
          Row {decided + (head?.playerAction === null ? 1 : 0)} of {rows.length}.
          Click a bin, or press 1–4.
        </p>
        <p className="font-mono text-xs text-text-muted">
          {pipeline.undecided} left on the belt
        </p>
      </div>

      {/* The belt. */}
      <section aria-label="Conveyor belt" className="min-h-0">
        <ol className="flex flex-col gap-2">
          {head ? (
            <RowCard row={head} position={cursor + 1} isHead stats={stats} />
          ) : null}
          {upcoming.map((row, offset) => (
            <RowCard
              key={row.id}
              row={row}
              position={cursor + offset + 2}
              isHead={false}
              stats={stats}
            />
          ))}
        </ol>
      </section>

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
