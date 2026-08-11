"use client";

export interface GapMeterProps {
  trainAccuracy: number | null;
  validationAccuracy: number | null;
  /** Best accuracy the label noise allows. */
  achievable: number;
}

const WIDTH = 320;
const HEIGHT = 92;
const LEFT = 6;
const RIGHT = WIDTH - 6;
const TRACK = RIGHT - LEFT;

const at = (value: number) => LEFT + Math.min(Math.max(value, 0), 1) * TRACK;

/**
 * Train vs validation accuracy, with the gap and the bias drawn as regions
 * (spec: `<GapMeter>`, "live train vs. validation accuracy gap bar").
 *
 * Two quantities, drawn as two different distances, because they are the two
 * different ways to be wrong:
 *
 *   gap  = train − validation   the distance between the two bars
 *   bias = ceiling − train      the distance from the train bar to the ceiling
 *
 * Drawing them on one axis is what makes it visible that closing one by
 * overshooting opens the other. A single "how good is my model" number could not
 * show that, which is why this is not a single number.
 */
export function GapMeter({
  trainAccuracy,
  validationAccuracy,
  achievable,
}: GapMeterProps) {
  const ready = trainAccuracy !== null && validationAccuracy !== null;
  const gap = ready ? Math.max(0, trainAccuracy - validationAccuracy) : 0;
  const bias = ready ? Math.max(0, achievable - trainAccuracy) : 0;

  const label = ready
    ? `Training accuracy ${Math.round(trainAccuracy * 100)} percent, validation accuracy ${Math.round(
        validationAccuracy * 100,
      )} percent. Gap ${Math.round(gap * 100)} points, bias ${Math.round(
        bias * 100,
      )} points against a ceiling of ${Math.round(achievable * 100)} percent.`
    : "No model trained yet. Deploy to measure the gap.";

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="w-full rounded-md border border-border bg-surface-2"
        role="img"
        aria-label={label}
      >
        {/* The ceiling: no honest model beats this, because of label noise. */}
        <line
          x1={at(achievable)}
          y1={8}
          x2={at(achievable)}
          y2={HEIGHT - 20}
          stroke="var(--text-muted)"
          strokeWidth={1}
          strokeDasharray="3 3"
        />
        <text
          x={at(achievable)}
          y={HEIGHT - 8}
          textAnchor="middle"
          className="fill-[var(--text-muted)] text-[8px]"
        >
          ceiling {Math.round(achievable * 100)}%
        </text>

        {ready ? (
          <>
            {/* Bias region: how far short of achievable the model fell. */}
            {bias > 0.005 ? (
              <rect
                x={at(trainAccuracy)}
                y={20}
                width={at(achievable) - at(trainAccuracy)}
                height={HEIGHT - 44}
                fill="var(--warn)"
                opacity={0.18}
              />
            ) : null}

            {/* Gap region: how much of the training score didn't transfer. */}
            {gap > 0.005 ? (
              <rect
                x={at(validationAccuracy)}
                y={20}
                width={at(trainAccuracy) - at(validationAccuracy)}
                height={HEIGHT - 44}
                fill="var(--wrong)"
                opacity={0.22}
              />
            ) : null}

            {/* Validation bar: the only number that reflects the real world. */}
            <rect
              x={LEFT}
              y={22}
              width={at(validationAccuracy) - LEFT}
              height={16}
              rx={3}
              fill="var(--class-a)"
            />
            {/* Training bar. */}
            <rect
              x={LEFT}
              y={44}
              width={at(trainAccuracy) - LEFT}
              height={16}
              rx={3}
              fill="var(--class-b)"
            />

            <text
              x={at(validationAccuracy) + 4}
              y={34}
              className="fill-[var(--text)] text-[9px]"
            >
              val {Math.round(validationAccuracy * 100)}%
            </text>
            <text
              x={at(trainAccuracy) + 4}
              y={56}
              className="fill-[var(--text)] text-[9px]"
            >
              train {Math.round(trainAccuracy * 100)}%
            </text>
          </>
        ) : (
          <text
            x={WIDTH / 2}
            y={HEIGHT / 2 - 4}
            textAnchor="middle"
            className="fill-[var(--text-muted)] text-[11px]"
          >
            Deploy the model to measure the gap
          </text>
        )}
      </svg>

      <figcaption className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 text-xs">
        <span className="inline-flex items-center gap-1.5">
          <span
            aria-hidden="true"
            className="inline-block h-2.5 w-4 rounded-sm"
            style={{ background: "var(--class-b)" }}
          />
          <span className="text-text-muted">train</span>
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span
            aria-hidden="true"
            className="inline-block h-2.5 w-4 rounded-sm"
            style={{ background: "var(--class-a)" }}
          />
          <span className="text-text-muted">validation</span>
        </span>
        <span className="text-text-muted">
          {ready
            ? gap > bias
              ? "Red band is the gap — score that didn't transfer."
              : bias > 0.02
                ? "Amber band is the bias — score it never reached."
                : "Both bands are thin. This is what generalising looks like."
            : "Red band will show the gap, amber the bias."}
        </span>
      </figcaption>
    </figure>
  );
}
