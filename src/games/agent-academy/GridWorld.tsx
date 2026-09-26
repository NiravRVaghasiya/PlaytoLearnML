"use client";

import { useReducedMotion } from "@/lib/useReducedMotion";
import { cx } from "@/lib/utils";
import {
  ACTION_ARROWS,
  ACTION_LABELS,
  COLS,
  GOAL,
  GRID,
  ROWS,
  START,
  TRAP,
  type Cell,
  type Position,
} from "./ml";
import type { CellValue } from "./store";

export interface GridWorldProps {
  /** The greedy walk to draw. */
  trail: Position[];
  /** The optimal policy's walk, when the overlay is on. */
  optimalTrail: Position[] | null;
  /** Per-cell best Q-value, when the heatmap is on. */
  values: CellValue[] | null;
  episodesUsed: number;
}

const TILE = 44;
const PAD = 6;
const WIDTH = COLS * TILE + PAD * 2;
const HEIGHT = ROWS * TILE + PAD * 2;

const CELL_FILL: Record<Cell, string> = {
  empty: "var(--surface-2)",
  wall: "var(--border)",
  trap: "color-mix(in srgb, var(--wrong) 26%, var(--surface-2))",
  pellet: "color-mix(in srgb, var(--class-b) 26%, var(--surface-2))",
  start: "var(--surface-2)",
  goal: "color-mix(in srgb, var(--correct) 26%, var(--surface-2))",
};

const CELL_WORDS: Record<Cell, string> = {
  empty: "floor",
  wall: "wall",
  trap: "pit",
  pellet: "cheese",
  start: "start",
  goal: "exit",
};

const x = (col: number) => PAD + col * TILE;
const y = (row: number) => PAD + row * TILE;
const cx0 = (col: number) => x(col) + TILE / 2;
const cy0 = (row: number) => y(row) + TILE / 2;

/**
 * The grid world (spec: `<GridWorld>`).
 *
 * ── Deviation from the spec, deliberate ─────────────────────────────────────
 * The spec names Phaser. This is SVG, for the same reasons as Overfit Tower
 * Defense's battlefield and one more specific to this game: the Q-value heatmap
 * has to sit on exactly these 28 cells, and the value in each cell has to be
 * readable as a NUMBER, not just a shade. In SVG that is a `<text>` per tile and
 * it lands in the accessibility tree for free. In a canvas it would be a second,
 * parallel, hand-built accessible view of the same data — two things to keep in
 * step for no gain. There is no render loop here either: the agent's walk is a
 * static polyline through a deterministic grid, drawn once per training batch.
 *
 * ── Colour is never the only channel ────────────────────────────────────────
 * Every special cell carries a glyph and a label as well as a tint, and the
 * heatmap prints its numbers. A player who cannot distinguish the exit's green
 * from the pit's red can still read "EXIT" and "PIT", and can still see +14.0
 * against -20.0.
 */
export function GridWorld({
  trail,
  optimalTrail,
  values,
  episodesUsed,
}: GridWorldProps) {
  const reducedMotion = useReducedMotion();

  const finite = (values ?? []).filter(
    (cell) => Number.isFinite(cell.best) && cell.visited,
  );
  const maxValue = finite.reduce((best, cell) => Math.max(best, cell.best), 0);
  const minValue = finite.reduce((least, cell) => Math.min(least, cell.best), 0);
  const span = Math.max(maxValue - minValue, 1e-6);

  /**
   * Diverging fill for a Q-value.
   *
   * Blue for "the agent expects to do well here", orange for "badly" — the
   * Okabe–Ito pair already used for dataset classes, chosen because it stays
   * distinguishable under every common form of colour vision deficiency. Never
   * red/green, which is the pairing this scale would most obviously reach for and
   * the one a deuteranope cannot read.
   */
  const heatFill = (cell: CellValue): string => {
    if (!cell.visited) return "var(--surface-2)";
    const t = (cell.best - minValue) / span;
    return t >= 0.5
      ? `color-mix(in srgb, var(--class-a) ${Math.round(
          (t - 0.5) * 2 * 78 + 12,
        )}%, var(--surface-2))`
      : `color-mix(in srgb, var(--class-b) ${Math.round(
          (0.5 - t) * 2 * 78 + 12,
        )}%, var(--surface-2))`;
  };

  const trailPoints = trail
    .map((position) => `${cx0(position.col)},${cy0(position.row)}`)
    .join(" ");
  const optimalPoints = (optimalTrail ?? [])
    .map((position) => `${cx0(position.col)},${cy0(position.row)}`)
    .join(" ");

  const last = trail.at(-1) ?? START;
  const ended =
    last.row === GOAL.row && last.col === GOAL.col
      ? "reaches the exit"
      : last.row === TRAP.row && last.col === TRAP.col
        ? "falls in the pit"
        : "loops without finishing";

  const description = `A ${ROWS} by ${COLS} grid. Start at the top-left corner. The cheese is two squares to its right. The pit is one row down and one column right of the start. The exit is the bottom-right corner. ${
    episodesUsed === 0
      ? "The agent has not trained yet."
      : `After ${episodesUsed} episodes the greedy walk is ${
          trail.length - 1
        } steps long and ${ended}.`
  }${
    values
      ? ` Q-value overlay on: highest ${maxValue.toFixed(
          1,
        )}, lowest ${minValue.toFixed(1)}.`
      : ""
  }`;

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="w-full max-w-[420px] rounded-md border border-border bg-surface"
        role="img"
        aria-label={description}
      >
        {GRID.map((cells, row) =>
          cells.map((cell, col) => {
            const state = row * COLS + col;
            const value = values?.[state];
            const fill =
              value && cell !== "wall" ? heatFill(value) : CELL_FILL[cell];
            return (
              <g key={state}>
                <rect
                  x={x(col) + 1}
                  y={y(row) + 1}
                  width={TILE - 2}
                  height={TILE - 2}
                  rx={4}
                  fill={fill}
                  stroke={cell === "wall" ? "none" : "var(--border)"}
                  strokeWidth={0.5}
                />
                {cell === "goal" ? (
                  <text
                    x={cx0(col)}
                    y={cy0(row) + 3}
                    textAnchor="middle"
                    className="fill-[var(--text)] text-[8px] font-bold"
                  >
                    EXIT
                  </text>
                ) : cell === "trap" ? (
                  <text
                    x={cx0(col)}
                    y={cy0(row) + 3}
                    textAnchor="middle"
                    className="fill-[var(--text)] text-[8px] font-bold"
                  >
                    PIT
                  </text>
                ) : cell === "pellet" ? (
                  <text
                    x={cx0(col)}
                    y={cy0(row) + 3}
                    textAnchor="middle"
                    className="fill-[var(--text)] text-[13px]"
                  >
                    ▲
                  </text>
                ) : cell === "start" ? (
                  <text
                    x={cx0(col)}
                    y={cy0(row) + 3}
                    textAnchor="middle"
                    className="fill-[var(--text-muted)] text-[8px] font-bold"
                  >
                    START
                  </text>
                ) : null}

                {/* The learned value, printed. A shade alone is not a reading. */}
                {value && cell !== "wall" && value.visited ? (
                  <text
                    x={x(col) + TILE - 3}
                    y={y(row) + 11}
                    textAnchor="end"
                    className="fill-[var(--text)] text-[7px] font-medium tabular-nums"
                  >
                    {value.best.toFixed(1)}
                  </text>
                ) : null}
                {value && cell !== "wall" && value.visited ? (
                  <text
                    x={x(col) + 4}
                    y={y(row) + TILE - 4}
                    className="fill-[var(--text-muted)] text-[9px]"
                  >
                    {ACTION_ARROWS[value.action]}
                  </text>
                ) : null}
              </g>
            );
          }),
        )}

        {/* What the rewards ask for, when asked. Dashed, behind the agent's walk. */}
        {optimalPoints ? (
          <polyline
            points={optimalPoints}
            fill="none"
            stroke="var(--warn)"
            strokeWidth={2.5}
            strokeDasharray="5 4"
            strokeLinejoin="round"
            opacity={0.85}
          />
        ) : null}

        {trail.length > 1 ? (
          <polyline
            points={trailPoints}
            fill="none"
            stroke="var(--primary)"
            strokeWidth={2.5}
            strokeLinejoin="round"
            opacity={0.9}
            style={
              reducedMotion
                ? undefined
                : { transition: "all var(--dur-standard) var(--ease-spring)" }
            }
          />
        ) : null}

        {/* The agent. */}
        <circle
          cx={cx0(last.col)}
          cy={cy0(last.row)}
          r={8}
          fill="var(--primary)"
          stroke="var(--bg)"
          strokeWidth={2}
          style={
            reducedMotion
              ? undefined
              : { transition: "all var(--dur-standard) var(--ease-spring)" }
          }
        />
      </svg>

      <figcaption className="mt-2 text-xs text-text-muted">
        <span className="inline-flex items-center gap-1.5">
          <span
            aria-hidden="true"
            className="inline-block h-0.5 w-4 bg-[var(--primary)]"
          />
          what the agent learned
        </span>
        {optimalTrail ? (
          <span className="ml-3 inline-flex items-center gap-1.5">
            <span
              aria-hidden="true"
              className="inline-block h-0.5 w-4 border-t-2 border-dashed border-[var(--warn)]"
            />
            what your rewards ask for
          </span>
        ) : null}
      </figcaption>

      {/*
       * The same grid as text, so the maze is legible without reading a picture.
       *
       * The clipping class sits on a wrapping div, not on the table. A table box is
       * never narrower than its min-content width and ignores `overflow: hidden`,
       * so a 1px `sr-only` TABLE still laid out 566px wide and gave the whole page
       * a horizontal scrollbar on phones. A block container clips properly.
       */}
      <div className="sr-only-live">
      <table>
        <caption>Grid contents by row and column</caption>
        <thead>
          <tr>
            <th scope="col">Row</th>
            {Array.from({ length: COLS }, (_, col) => (
              <th key={col} scope="col">
                Column {col + 1}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {GRID.map((cells, row) => (
            <tr key={row}>
              <th scope="row">Row {row + 1}</th>
              {cells.map((cell, col) => {
                const value = values?.[row * COLS + col];
                return (
                  <td key={col}>
                    {CELL_WORDS[cell]}
                    {value?.visited
                      ? `, value ${value.best.toFixed(
                          1,
                        )}, best action ${ACTION_LABELS[value.action]}`
                      : ""}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      </div>
    </figure>
  );
}

/** Legend for the heatmap scale, shown beside the grid when the overlay is on. */
export function HeatLegend({ className }: { className?: string }) {
  return (
    <p className={cx("text-xs text-text-muted", className)}>
      <span
        aria-hidden="true"
        className="mr-1 inline-block size-2.5 rounded-sm bg-[var(--class-a)]"
      />
      expects to do well ·
      <span
        aria-hidden="true"
        className="mx-1 inline-block size-2.5 rounded-sm bg-[var(--class-b)]"
      />
      expects to do badly · untinted cells were never visited, so their values are
      still 0
    </p>
  );
}
