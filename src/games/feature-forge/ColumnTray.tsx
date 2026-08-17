"use client";

import { AlertTriangle } from "lucide-react";
import { COLUMNS, transformById, type Transform } from "./ml";

export interface ColumnTrayProps {
  slot: string[];
  transform: Transform;
  disabled: boolean;
  onToggle: (name: string) => void;
}

const DTYPE_LABEL: Record<string, string> = {
  numeric: "number",
  categorical: "category",
  timestamp: "timestamp",
};

/**
 * The raw columns (spec: `<ColumnTray>`).
 *
 * Checkboxes rather than draggable tiles. The spec describes dragging columns into
 * the forge, but a drag is the one interaction a keyboard cannot perform, and what
 * is actually being expressed is a selection of one or two columns — which is
 * exactly what a checkbox group is for. Same decision as Neuron Forge's editor and
 * the tree canvas, for the same reason.
 *
 * Columns the selected transform cannot accept are disabled rather than hidden, so
 * the player can see WHY a pairing is unavailable — a timestamp is not a number,
 * and that fact is part of the lesson.
 */
export function ColumnTray({
  slot,
  transform,
  disabled,
  onToggle,
}: ColumnTrayProps) {
  const spec = transformById(transform);

  return (
    <fieldset>
      <legend className="text-sm font-medium">
        Raw columns{" "}
        <span className="font-normal text-text-muted">
          — pick {spec.arity === 1 ? "one" : "two"}
        </span>
      </legend>

      <ul className="mt-2 flex flex-col gap-1">
        {COLUMNS.map((column) => {
          const accepted = spec.accepts.includes(column.dtype);
          const selected = slot.includes(column.name);
          const order = slot.indexOf(column.name);

          return (
            <li key={column.name}>
              <label
                className={`flex cursor-pointer items-start gap-2 rounded border px-2 py-1.5 text-xs ${
                  selected
                    ? "border-primary bg-primary/10"
                    : column.leaky
                      ? "border-warn/50 bg-warn/5 hover:bg-warn/10"
                      : "border-border bg-surface-2 hover:bg-surface"
                } ${!accepted || disabled ? "cursor-not-allowed opacity-50" : ""}`}
              >
                <input
                  type="checkbox"
                  checked={selected}
                  disabled={!accepted || disabled}
                  onChange={() => onToggle(column.name)}
                  className="mt-0.5 accent-[var(--primary)]"
                />
                <span className="min-w-0">
                  <span className="flex flex-wrap items-baseline gap-1.5">
                    <span className="font-mono font-medium">{column.name}</span>
                    <span className="text-[10px] text-text-muted">
                      {DTYPE_LABEL[column.dtype]}
                    </span>
                    {column.leaky ? (
                      <span className="inline-flex items-center gap-0.5 text-[10px] text-warn">
                        <AlertTriangle aria-hidden="true" className="size-3" />
                        risky
                      </span>
                    ) : null}
                    {spec.arity > 1 && selected ? (
                      <span className="text-[10px] text-primary">
                        #{order + 1}
                      </span>
                    ) : null}
                  </span>
                  <span className="mt-0.5 block text-[11px] text-text-muted">
                    {column.description}
                  </span>
                  {!accepted ? (
                    <span className="mt-0.5 block text-[10px] text-text-muted">
                      {spec.label} needs a{" "}
                      {spec.accepts.map((d) => DTYPE_LABEL[d]).join(" or ")}.
                    </span>
                  ) : null}
                </span>
              </label>
            </li>
          );
        })}
      </ul>
    </fieldset>
  );
}
