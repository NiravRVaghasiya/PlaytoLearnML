"use client";

import { Compass } from "lucide-react";
import { Button } from "@/components";
import type { Analysis } from "./ml";

export interface PCAHintButtonProps {
  analysis: Analysis;
  used: boolean;
  disabled: boolean;
  onUse: () => void;
}

/**
 * Snap to the principal plane (spec: `<PCAHintButton>`).
 *
 * A deliberate escape hatch with a deliberate price. Hunting an orientation by eye
 * is the exercise, but being stuck at 80% with no idea which way to turn teaches
 * nothing either — so the answer is available, it says what it did rather than just
 * moving the sliders, and it costs the top score for that cloud.
 *
 * The cost is stated on the button before it is pressed. A penalty discovered
 * afterwards would be a trap.
 */
export function PCAHintButton({
  analysis,
  used,
  disabled,
  onUse,
}: PCAHintButtonProps) {
  return (
    <div>
      <Button
        variant="secondary"
        className="w-full"
        onClick={onUse}
        disabled={disabled || used}
        icon={<Compass className="size-4" />}
      >
        {used ? "Principal plane used" : "Show me PCA's answer"}
      </Button>
      <p className="mt-1.5 text-xs text-text-muted">
        {used
          ? `Snapped to yaw ${analysis.pcaAngles.yaw.toFixed(
              1,
            )}°, pitch ${analysis.pcaAngles.pitch.toFixed(
              1,
            )}° — the plane spanned by the two largest eigenvectors. Retains ${(
              analysis.best * 100
            ).toFixed(1)}%, which is the ceiling.`
          : `Eigendecomposes the covariance matrix and turns the cloud to the answer. Caps your score for this cloud, so it is worth a few more turns first.`}
      </p>
    </div>
  );
}
