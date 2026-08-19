import type { WhyCardContent } from "@/components";
import {
  CLOUDS,
  GROUP_LABELS,
  SEPARATION_FLOOR,
  VARIANCE_TARGET,
  varianceRetained,
  type Analysis,
  type CloudSpec,
  type Evaluation,
} from "./ml";

/**
 * "Why did that happen?" copy for Dimension Diver.
 *
 * The rule this copy follows: never say "you lost variance" without saying which
 * direction it went into. The gauge already shows the number; what transfers is the
 * fact that the score is decided entirely by the axis being discarded, and that the
 * cloud's three variances tell you in advance what any projection is worth.
 *
 * The other rule is about honesty over tidiness. Two of these clouds are built to
 * embarrass PCA — one where the loudest directions carry no group information, one
 * where no linear projection separates the groups at all — and the copy says so
 * plainly rather than treating a method's limitation as a player's mistake.
 */
export type DiverEvent =
  | { kind: "briefing"; cloud: CloudSpec }
  | { kind: "rolled"; retained: number; roll: number }
  | { kind: "hinted"; analysis: Analysis; retained: number }
  | {
      kind: "submitted";
      cloud: CloudSpec;
      evaluation: Evaluation;
      analysis: Analysis;
      hintUsed: boolean;
    };

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

export function whyCardFor(event: DiverEvent): WhyCardContent {
  switch (event.kind) {
    case "briefing": {
      const { cloud } = event;
      const isFirst = cloud.id === CLOUDS[0]!.id;
      return {
        key: `briefing-${cloud.id}`,
        title: isFirst ? "Turn it until the shadow is widest" : cloud.title,
        body: `${cloud.lesson}${
          isFirst
            ? ` Three groups are hidden in there and you cannot see which point belongs to which — that is revealed when you commit. What you can see is the shadow, and how much of the cloud's spread survives being flattened into it. Worth knowing before you start: the score depends only on the direction pointing away from you, so the roll slider will spin the picture without moving the number at all.`
            : ``
        }`,
        tone: "info",
      };
    }

    case "rolled":
      return {
        key: `rolled-${Math.round(event.roll)}`,
        title: "Rolling cannot change the score",
        body: `The picture turned and the gauge did not move — still ${percent(
          event.retained,
        )}. That is not a bug and it is the most useful piece of geometry in this game: what a projection keeps is trace(C) minus the variance along the axis you discard, and rolling spins the shadow inside a plane it has already chosen. The discarded axis never moved. Which means you have two real degrees of freedom here, not three: where to point the direction you are throwing away.`,
        tone: "info",
      };

    case "hinted": {
      const { analysis, retained } = event;
      return {
        key: `hinted-${Math.round(retained * 1000)}`,
        title: `PCA's answer: ${percent(retained)}`,
        body: `Snapped to the plane spanned by the top two eigenvectors of the covariance matrix. No search was involved — the three variances along the principal axes are ${analysis.eigen.values
          .map((value) => value.toFixed(2))
          .join(
            ", ",
          )}, and the best plane is simply the one that discards the smallest. That is what makes PCA unusual among everything else on this site: it has a closed-form answer, and the eigenvalues tell you what the answer is worth before you look at the picture. Using the hint costs you the top score for this cloud, which seems fair.`,
        tone: "info",
      };
    }

    case "submitted": {
      const { cloud, evaluation, analysis, hintUsed } = event;
      const { outcome } = evaluation;

      if (outcome === "surfaced") {
        if (!evaluation.separable) {
          return {
            key: `surfaced-inseparable-${cloud.id}`,
            title: `${percent(evaluation.retained)} retained — and the groups are still one blob`,
            body: `You found the best plane available and it did not help, because for this cloud there is no plane that would. One group sits inside the other as a shell, and a linear projection can only ever take weighted sums of coordinates — no weighted sum of coordinates distinguishes "near the centre" from "far from it". The best separation anywhere in the whole space of projections is ${analysis.bestSeparation.ratio.toFixed(
              3,
            )}, which is nothing. Notice also how flat the gauge was as you turned: the three variances are ${analysis.eigen.values
              .map((value) => value.toFixed(2))
              .join(
                ", ",
              )}, nearly equal, so this cloud has no preferred plane to find. This is the exact gap t-SNE and UMAP exist to fill — they give up being linear, and being able to project a new point without refitting, in exchange for being able to see structure like this at all.`,
            tone: "good",
          };
        }

        return {
          key: `surfaced-${cloud.id}-${Math.round(evaluation.retained * 1000)}`,
          title: `Surfaced with ${percent(evaluation.retained)} of the spread`,
          body: `${
            evaluation.share >= 0.999
              ? `That is the optimum — the same plane an eigendecomposition would have handed you.`
              : `Within ${percent(
                  1 - evaluation.share,
                )} of the best any flat shadow can do.`
          } And the groups came apart: the best line through your shadow scores ${evaluation.separation.ratio.toFixed(
            2,
          )} on between-group over within-group scatter. The three principal variances are ${analysis.eigen.values
            .map((value) => value.toFixed(2))
            .join(
              ", ",
            )}, so the arithmetic was always visible: you were looking for the direction the cloud is thinnest along, and discarding it costs ${percent(
            1 - analysis.best,
          )}.${
            hintUsed
              ? ` You took the hint on this one, which is why the score is capped.`
              : ` Found by eye, which is more than PCA does — it never looks at the picture.`
          }`,
          tone: "good",
        };
      }

      if (outcome === "mixed") {
        return {
          key: `mixed-${cloud.id}`,
          title: "Maximum variance, and nothing to see",
          body: `This is the most important thing this game has to say, so it is worth being precise. PCA does not look for structure. It looks for spread, and hands back the plane with the most of it — and here the loudest direction in the cloud is ${(
            analysis.eigen.values[0] / analysis.eigen.values[2]
          ).toFixed(
            0,
          )} times louder than the quietest while carrying no group information whatsoever. The groups differ along the quietest axis, which is exactly the one PCA discards. There is a plane retaining ${percent(
            varianceRetained(
              analysis.covariance,
              analysis.reachableSeparation.angles,
            ),
          )} — still comfortably inside the target — where they come apart. Nothing in the algorithm knows the groups exist, so nothing in the algorithm was ever going to warn you.`,
          tone: "bad",
        };
      }

      return {
        key: `lost-${cloud.id}-${Math.round(evaluation.retained * 1000)}`,
        title: `${percent(evaluation.retained)} kept, ${percent(
          analysis.best - evaluation.retained,
        )} thrown away`,
        body: `The axis you are pointing away from has real spread along it, so flattening it destroys structure that was there. The cloud's variances along its own principal axes are ${analysis.eigen.values
          .map((value) => value.toFixed(2))
          .join(
            ", ",
          )} — discard anything but the smallest and you pay for it. Keep turning: you are hunting for the direction the data is thinnest along, and the gauge is a live reading of exactly how well you are doing. ${
          GROUP_LABELS.length > 0
            ? `The groups are showing now, which should tell you which way to go.`
            : ``
        }`,
        tone: "warn",
      };
    }
  }
}

/** Caption for the variance gauge. */
export function gaugeCaption(
  retained: number,
  best: number,
  submitted: boolean,
): string {
  const share = best <= 1e-12 ? 1 : retained / best;
  if (share >= 0.999) {
    return `This is the optimum. No plane keeps more than ${(best * 100).toFixed(
      1,
    )}%.`;
  }
  if (share >= VARIANCE_TARGET) {
    return `Within ${((1 - share) * 100).toFixed(
      1,
    )}% of the best possible ${(best * 100).toFixed(1)}%.`;
  }
  if (submitted) {
    return `${((best - retained) * 100).toFixed(
      1,
    )} points of spread are being flattened away.`;
  }
  return `The ceiling is ${(best * 100).toFixed(
    1,
  )}%, set by the two largest eigenvalues. Only the discarded axis matters.`;
}

/** Caption for the separation readout. */
export function separationCaption(
  ratio: number,
  reachable: number,
  separable: boolean,
): string {
  if (!separable) {
    return `No flat shadow of this cloud separates the groups — the best anywhere is ${reachable.toFixed(
      3,
    )}. That is a limit of linear projection, not of your angle.`;
  }
  if (ratio >= reachable * 0.85) {
    return `The groups come apart cleanly in this shadow.`;
  }
  if (ratio < SEPARATION_FLOOR) {
    return `Nothing is separated here. There is a plane reaching ${reachable.toFixed(
      2,
    )} without giving up the variance target.`;
  }
  return `Partly separated. The best reachable is ${reachable.toFixed(2)}.`;
}
