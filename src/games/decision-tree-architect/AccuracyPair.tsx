"use client";

export interface AccuracyPairProps {
  trainAccuracy: number;
  validationAccuracy: number;
  achievable: number;
  target: number;
  /** Best validation this player has reached on this plot. */
  peakValidation: number;
  peakDepth: number;
}

const WIDTH = 320;
const HEIGHT = 104;
const LEFT = 6;
const RIGHT = WIDTH - 6;
const TRACK = RIGHT - LEFT;

const at = (value: number) => LEFT + Math.min(Math.max(value, 0), 1) * TRACK;

/**
 * Training and validation accuracy side by side (spec: `<AccuracyPair>`).
 *
 * Two bars rather than one number, and a marker for the best validation accuracy
 * the player has already reached. That marker is the whole point: overfitting is
 * not "validation is low", it is "validation is lower than it was", and a single
 * pair of bars cannot show a player something they used to have.
 */
export function AccuracyPair({
  trainAccuracy,
  validationAccuracy,
  achievable,
  target,
  peakValidation,
  peakDepth,
}: AccuracyPairProps) {
  const gap = Math.max(0, trainAccuracy - validationAccuracy);
  const givingBack = peakValidation - validationAccuracy >= 0.01;

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="w-full rounded-md border border-border bg-surface-2"
        role="img"
        aria-label={`Training accuracy ${Math.round(
          trainAccuracy * 100,
        )} percent, validation accuracy ${Math.round(
          validationAccuracy * 100,
        )} percent, a gap of ${Math.round(gap * 100)} points. Target ${Math.round(
          target * 100,
        )} percent.${
          peakValidation > 0
            ? ` Best validation so far ${Math.round(
                peakValidation * 100,
              )} percent at depth ${peakDepth}.${
                givingBack ? " Validation is currently below that." : ""
              }`
            : ""
        }`}
      >
        {/* Ceiling imposed by wrong verdicts in the data. */}
        <line
          x1={at(achievable)}
          y1={6}
          x2={at(achievable)}
          y2={HEIGHT - 22}
          stroke="var(--text-muted)"
          strokeWidth={1}
          strokeDasharray="3 3"
        />
        {/* The bar to clear. */}
        <line
          x1={at(target)}
          y1={6}
          x2={at(target)}
          y2={HEIGHT - 22}
          stroke="var(--primary)"
          strokeWidth={1}
        />

        {/* The gap, shaded, so it is a distance and not a subtraction. */}
        {gap > 0.005 ? (
          <rect
            x={at(validationAccuracy)}
            y={20}
            width={at(trainAccuracy) - at(validationAccuracy)}
            height={40}
            fill="var(--wrong)"
            opacity={0.18}
          />
        ) : null}

        <rect
          x={LEFT}
          y={22}
          width={at(trainAccuracy) - LEFT}
          height={16}
          rx={3}
          fill="var(--class-b)"
        />
        <rect
          x={LEFT}
          y={42}
          width={at(validationAccuracy) - LEFT}
          height={16}
          rx={3}
          fill={givingBack ? "var(--wrong)" : "var(--class-a)"}
        />

        {/* Where validation used to be. */}
        {peakValidation > 0.01 ? (
          <>
            <line
              x1={at(peakValidation)}
              y1={40}
              x2={at(peakValidation)}
              y2={62}
              stroke="var(--correct)"
              strokeWidth={2}
            />
            <text
              x={at(peakValidation)}
              y={HEIGHT - 8}
              textAnchor="middle"
              className="fill-[var(--correct)] text-[8px]"
            >
              best {Math.round(peakValidation * 100)}% @ d{peakDepth}
            </text>
          </>
        ) : null}

        <text x={at(target)} y={16} textAnchor="middle" className="fill-[var(--primary)] text-[8px]">
          target
        </text>
        <text
          x={at(achievable)}
          y={16}
          textAnchor="middle"
          className="fill-[var(--text-muted)] text-[8px]"
        >
          ceiling
        </text>

        <text x={at(trainAccuracy) + 4} y={34} className="fill-[var(--text)] text-[9px]">
          train {Math.round(trainAccuracy * 100)}%
        </text>
        <text
          x={at(validationAccuracy) + 4}
          y={54}
          className="fill-[var(--text)] text-[9px]"
        >
          validation {Math.round(validationAccuracy * 100)}%
        </text>
      </svg>

      <figcaption className="mt-1.5 text-xs text-text-muted">
        {givingBack
          ? `Validation is ${((peakValidation - validationAccuracy) * 100).toFixed(
              1,
            )} points below your best. Every gate raises the top bar; only some raise the bottom one.`
          : gap > 0.12
            ? "The red band is the gap. It widens as gates start describing individual surveys."
            : "Both bars moving together means the tree is learning the ground, not the sample."}
      </figcaption>
    </figure>
  );
}
