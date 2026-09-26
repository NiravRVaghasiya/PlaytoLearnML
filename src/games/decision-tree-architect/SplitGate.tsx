"use client";

import { useMemo, useState } from "react";
import { Scissors, Sparkles } from "lucide-react";
import { Button, Slider } from "@/components";
import { PurityPie } from "./PurityPie";
import {
  FEATURES,
  MIN_HONEST_LEAF,
  candidateSplitsWithLookahead,
  countsOf,
  gainOf,
  giniOf,
  lookaheadGainOf,
  partition,
  type Sample,
  type Split,
} from "./ml";

export interface SplitGateProps {
  nodeId: string;
  /** Training samples reaching this node. */
  bucket: Sample[];
  isLeaf: boolean;
  atDepthLimit: boolean;
  onSplit: (split: Split) => void;
  onPrune: () => void;
}

/**
 * The split picker (spec: `<SplitGate>`).
 *
 * This is where "player action = algorithm" lives. The table below is exactly
 * what greedy CART computes before every split it makes: the best threshold on
 * each feature and the gain it would achieve. The player reads the same numbers
 * and makes the same choice by hand.
 *
 * The second column is the one that is not in CART. One-step lookahead — how much
 * gain becomes available in the children if this gate is built — exists because
 * an XOR-shaped boundary gives the RIGHT gate almost no immediate gain. Greedy
 * CART takes the useless wind reading's 0.018 and never recovers.
 *
 * The table cannot show the right gate itself, and says so. Each row is a
 * reading at its best IMMEDIATE threshold — exactly what CART would try — and on
 * the ridge that puts slope at a sliver, "< 0.04", lookahead 0.017. The right
 * gate is slope < 0.50: gain 0.001, lookahead 0.39. It is found with the
 * threshold slider, whose readout computes lookahead live for wherever the
 * player puts it, beside the best gain the table offers for comparison. What
 * the table's column does show there is a hint worth following: bedrock's
 * lookahead (0.125) stands clear of the others (about 0.02), so something real
 * is underneath one of the two readings that looked worthless.
 */
export function SplitGate({
  nodeId,
  bucket,
  isLeaf,
  atDepthLimit,
  onSplit,
  onPrune,
}: SplitGateProps) {
  const [feature, setFeature] = useState(0);
  const [manual, setManual] = useState<number | null>(null);

  const counts = useMemo(() => countsOf(bucket), [bucket]);
  const gini = giniOf(counts);
  // Lookahead runs the split search on both children of every candidate, so it is
  // memoised on the bucket rather than recomputed per render.
  const table = useMemo(() => candidateSplitsWithLookahead(bucket), [bucket]);

  const chosen = table[feature] ?? null;
  const threshold = manual ?? chosen?.threshold ?? 0.5;
  // Memoised so the two derivations below actually cache: a fresh object here
  // would change their dependencies on every render.
  const liveSplit = useMemo<Split>(
    () => ({ feature, threshold }),
    [feature, threshold],
  );
  const liveGain = useMemo(
    () => (bucket.length > 1 ? gainOf(bucket, liveSplit) : 0),
    [bucket, liveSplit],
  );
  // Live lookahead for wherever the slider sits — the only place the right
  // ridge gate (slope < 0.50) ever shows its 0.39. Two partitions and a split
  // search on each child: well under a millisecond per slider step.
  const liveLookahead = useMemo(
    () => (bucket.length > 1 ? lookaheadGainOf(bucket, liveSplit) : 0),
    [bucket, liveSplit],
  );
  // What one gate could already buy here, so lookahead has a yardstick: a gate
  // whose lookahead only matches this has passed the problem down, not solved it.
  const bestGain = table.reduce(
    (best, candidate) => Math.max(best, candidate?.gain ?? 0),
    0,
  );
  const { left, right } = useMemo(
    () => partition(bucket, liveSplit),
    [bucket, liveSplit],
  );

  const canSplit =
    isLeaf &&
    !atDepthLimit &&
    bucket.length >= 2 &&
    left.length > 0 &&
    right.length > 0;

  return (
    <div>
      <div className="flex items-center gap-2">
        <PurityPie counts={counts} gini={gini} size={30} />
        <div className="min-w-0">
          <p className="text-sm font-medium">
            {isLeaf ? "Leaf" : "Gate"} {nodeId}
          </p>
          <p className="text-xs text-text-muted">
            {counts.total} plot{counts.total === 1 ? "" : "s"} · {counts.positive}{" "}
            safe, {counts.negative} unsafe · impurity {gini.toFixed(3)}
            {counts.total > 0 && counts.total < MIN_HONEST_LEAF
              ? " — too few to decide from"
              : ""}
          </p>
        </div>
      </div>

      {!isLeaf ? (
        <div className="mt-3 border-t border-border pt-3">
          <p className="text-xs text-text-muted">
            This node already has a gate. Prune it to rebuild from here, or pick a
            leaf below it.
          </p>
          <Button
            variant="secondary"
            size="sm"
            className="mt-2 w-full"
            onClick={onPrune}
            icon={<Scissors className="size-4" />}
          >
            Prune this gate
          </Button>
        </div>
      ) : (
        <>
          <table className="mt-3 w-full border-collapse text-left text-xs">
            <caption className="mb-1 text-left text-[11px] text-text-muted">
              Each reading at its best immediate threshold. Gain is what it buys
              now; lookahead is the best gate available underneath it. A gate
              that changes little passes the same structure down, so its
              lookahead only matches the best gain here — look for one well
              above it.
            </caption>
            <thead>
              <tr className="border-b border-border">
                <th scope="col" className="py-1 font-medium">
                  Reading
                </th>
                <th scope="col" className="py-1 text-right font-medium">
                  Gain
                </th>
                <th scope="col" className="py-1 text-right font-medium">
                  Lookahead
                </th>
                <th scope="col" className="py-1 text-right font-medium">
                  <span className="sr-only">Build this gate</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {FEATURES.map((meta, index) => {
                const candidate = table[index];
                const isCurrent = index === feature;
                return (
                  <tr
                    key={meta.name}
                    className={`border-b border-border/50 ${
                      isCurrent ? "bg-surface-2" : ""
                    }`}
                  >
                    <th scope="row" className="py-1 font-normal">
                      <span className="font-medium">{meta.name}</span>
                      {candidate ? (
                        <span className="block font-mono text-[10px] text-text-muted">
                          &lt; {candidate.threshold.toFixed(2)}
                        </span>
                      ) : null}
                    </th>
                    <td className="py-1 text-right font-mono">
                      {candidate ? candidate.gain.toFixed(4) : "—"}
                    </td>
                    <td className="py-1 text-right font-mono">
                      {candidate ? candidate.lookaheadGain.toFixed(3) : "—"}
                    </td>
                    <td className="py-1 pl-1 text-right">
                      {/* The shared Button, not a hand-rolled one: its 44px
                          minimum target (DESIGN.md §9) is the point, and this
                          is the main way a gate gets built on a phone. */}
                      <Button
                        variant="secondary"
                        size="sm"
                        disabled={!candidate || !canSplit}
                        onClick={() => {
                          setFeature(index);
                          setManual(null);
                          if (candidate) {
                            onSplit({
                              feature: index,
                              threshold: candidate.threshold,
                            });
                          }
                        }}
                        aria-label={`Build a gate on ${meta.name}${
                          candidate
                            ? ` at ${candidate.threshold.toFixed(
                                2,
                              )}, gain ${candidate.gain.toFixed(
                                4,
                              )}, lookahead ${candidate.lookaheadGain.toFixed(3)}`
                            : ", not available"
                        }`}
                      >
                        Build
                      </Button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <fieldset className="mt-3 border-t border-border pt-3">
            <legend className="text-xs font-medium">
              Or place the threshold yourself
            </legend>
            <label className="sr-only" htmlFor="feature-select">
              Reading to split on
            </label>
            <select
              id="feature-select"
              value={feature}
              onChange={(event) => {
                setFeature(Number(event.target.value));
                setManual(null);
              }}
              className="mt-1.5 min-h-11 w-full rounded-md border border-border bg-surface-2 px-2 py-1.5 text-sm"
            >
              {FEATURES.map((meta, index) => (
                <option key={meta.name} value={index}>
                  {meta.name} — {meta.description}
                </option>
              ))}
            </select>

            <Slider
              label={`Threshold on ${FEATURES[feature]?.name ?? ""}`}
              value={threshold}
              min={0}
              max={1}
              step={0.01}
              onChange={(value) => setManual(value)}
              format={(value) => value.toFixed(2)}
              hint="Plots below the threshold go left. The table only tries each reading's best immediate threshold; here you can try any, with lookahead computed live."
            />

            <p className="mt-1 text-xs text-text-muted">
              gain{" "}
              <span className="font-mono">{liveGain.toFixed(4)}</span> ·
              lookahead{" "}
              <span className="font-mono">{liveLookahead.toFixed(3)}</span>{" "}
              (best gain here{" "}
              <span className="font-mono">{bestGain.toFixed(4)}</span>) · splits{" "}
              {left.length} / {right.length}
              {left.length === 0 || right.length === 0
                ? " — everything goes one way, so this gate does nothing"
                : ""}
            </p>

            <Button
              variant="primary"
              size="sm"
              className="mt-2 w-full"
              onClick={() => onSplit(liveSplit)}
              disabled={!canSplit}
              icon={<Sparkles className="size-4" />}
            >
              Build this gate
            </Button>
            {atDepthLimit ? (
              <p className="mt-1 text-xs text-warn">
                At the depth limit for this plot. Prune something to go deeper.
              </p>
            ) : null}
            {bucket.length < 2 ? (
              <p className="mt-1 text-xs text-text-muted">
                Not enough plots reach this leaf to split it.
              </p>
            ) : null}
          </fieldset>
        </>
      )}
    </div>
  );
}
