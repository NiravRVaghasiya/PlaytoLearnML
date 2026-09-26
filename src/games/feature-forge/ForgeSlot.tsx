"use client";

import { AlertTriangle, Hammer, Sparkles, Trash2 } from "lucide-react";
import { Button } from "@/components";
import {
  LEGENDARY_COMBOS,
  describeFeature,
  isValidFeature,
  legendariesIn,
  transformById,
  usesLeakyColumn,
  type Feature,
  type Transform,
} from "./ml";

export interface ForgeSlotProps {
  slot: string[];
  transform: Transform;
  features: Feature[];
  training: boolean;
  onForge: () => void;
  onClearSlot: () => void;
  onRemove: (id: string) => void;
}

/**
 * The forge itself (spec: `<ForgeSlot>`).
 *
 * Shows the pending feature, whether it is legal, and everything already forged.
 * Legendary combos are flagged as they land, per the spec's glowing feature — but
 * the flag carries the REASON, because "legendary" on its own is flair and the
 * reason is the lesson.
 */
export function ForgeSlot({
  slot,
  transform,
  features,
  training,
  onForge,
  onClearSlot,
  onRemove,
}: ForgeSlotProps) {
  const spec = transformById(transform);
  const pending: Feature = {
    id: `${transform}:${slot.join("+")}`,
    transform,
    sourceCols: slot,
  };
  const valid = slot.length > 0 && isValidFeature(pending);
  const duplicate = features.some((feature) => feature.id === pending.id);
  const legendary = legendariesIn(features);

  return (
    <div className="border-t border-border pt-4">
      <p className="text-sm font-medium">Forge</p>

      <div
        className={`mt-2 rounded-md border-2 border-dashed p-2.5 ${
          valid && !duplicate
            ? "border-primary/60 bg-primary/5"
            : "border-border bg-surface-2"
        }`}
      >
        {slot.length === 0 ? (
          <p className="text-xs text-text-muted">
            Empty. Pick {spec.arity === 1 ? "a column" : "two columns"} for{" "}
            {spec.label.toLowerCase()}.
          </p>
        ) : (
          <>
            <p className="font-mono text-sm">{describeFeature(pending)}</p>
            <p className="mt-0.5 text-[11px] text-text-muted">
              {duplicate
                ? "Already in the forge — a duplicate column adds width and nothing else."
                : valid
                  ? spec.mechanism
                  : `${spec.label} needs ${spec.arity} ${
                      spec.arity === 1 ? "column" : "columns"
                    } of the right type.`}
            </p>
          </>
        )}
      </div>

      <div className="mt-2 flex gap-2">
        <Button
          variant="primary"
          size="sm"
          className="flex-1"
          onClick={onForge}
          disabled={!valid || duplicate || training}
          icon={<Hammer className="size-4" />}
        >
          {training ? "Retraining…" : "Forge it"}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={onClearSlot}
          disabled={slot.length === 0 || training}
        >
          Clear
        </Button>
      </div>

      <div className="mt-3">
        <p className="text-xs font-medium">
          Forged features{" "}
          <span className="font-normal text-text-muted">
            ({features.length})
          </span>
        </p>
        {features.length === 0 ? (
          <p className="mt-1 text-[11px] text-text-muted">
            None yet. The baseline columns are always in the model — these are on
            top.
          </p>
        ) : (
          <ul className="mt-1 flex flex-col gap-1">
            {features.map((feature) => {
              const combo = LEGENDARY_COMBOS.find((entry) =>
                entry.matches([feature]),
              );
              const leaky = usesLeakyColumn(feature);
              return (
                <li
                  key={feature.id}
                  className={`rounded border px-2 py-1.5 text-xs ${
                    leaky
                      ? "border-wrong/60 bg-wrong/10"
                      : combo
                        ? "border-correct/50 bg-correct/10"
                        : "border-border bg-surface-2"
                  }`}
                >
                  <span className="flex items-center gap-1.5">
                    {combo && !leaky ? (
                      <Sparkles
                        aria-hidden="true"
                        className="size-3.5 shrink-0 text-correct"
                      />
                    ) : null}
                    <span className="font-mono">{describeFeature(feature)}</span>
                    {/* 32px square: past WCAG 2.5.8's 24px minimum on its own,
                        rather than leaning on the spacing exception, without
                        turning a compact list row into a toolbar. */}
                    <button
                      type="button"
                      onClick={() => onRemove(feature.id)}
                      disabled={training}
                      aria-label={`Remove ${describeFeature(feature)} from the forge`}
                      className="-my-1 ml-auto inline-flex size-8 shrink-0 items-center justify-center rounded text-text-muted hover:bg-surface hover:text-wrong focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none disabled:opacity-50"
                    >
                      <Trash2 aria-hidden="true" className="size-4" />
                    </button>
                  </span>
                  {leaky ? (
                    // The red stays on the border and the icon; the words are in
                    // body text colour. Red text on its own red tint measured
                    // 4.1:1, under the 4.5:1 small text needs.
                    <span className="mt-0.5 flex items-start gap-1 text-[10px] text-text">
                      <AlertTriangle
                        aria-hidden="true"
                        className="mt-px size-3 shrink-0 text-wrong"
                      />
                      Leaky — this column is only set after churn has happened.
                    </span>
                  ) : combo ? (
                    <span className="mt-0.5 block text-[10px] text-text-muted">
                      {combo.label}: {combo.why}
                    </span>
                  ) : null}
                </li>
              );
            })}
          </ul>
        )}
        {features.length > 0 ? (
          <p className="mt-1.5 text-[11px] text-text-muted">
            {legendary.length} of {LEGENDARY_COMBOS.length} legendary
            relationships unlocked.
          </p>
        ) : null}
      </div>
    </div>
  );
}
