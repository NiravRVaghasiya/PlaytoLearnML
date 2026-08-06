import type { WhyCardContent } from "@/components";
import { MAX_K, WIN_SCORE, type Evaluation } from "./ml";

/**
 * "Why did that happen?" copy for K-Means Territory Wars.
 *
 * Keyed to the player's last action (spec §4). Every number comes from a real
 * computation in `ml.ts` — nothing here invents a figure.
 *
 * The through-line is the core intuition, delivered a piece at a time: assign
 * and update are two distinct steps; where you put the flags decides which
 * answer you converge to; and inertia alone can't tell you how many clusters
 * there are.
 */

export type KMeansEvent =
  | { kind: "reset" }
  | { kind: "flag-moved"; flag: number; inertia: number }
  | { kind: "k-changed"; k: number; elbowK: number; added: boolean }
  | { kind: "assigned"; inertia: number; sizes: number[] }
  | { kind: "updated"; inertia: number; shift: number; converged: boolean }
  | {
      kind: "stepped";
      inertia: number;
      shift: number;
      iteration: number;
      converged: boolean;
    }
  | { kind: "settled"; inertia: number; iterations: number }
  | { kind: "checked"; evaluation: Evaluation };

const CLUSTERING_HREF = "/concepts/k-means";
const ELBOW_HREF = "/concepts/choosing-k";

const num = (value: number) => value.toFixed(2);
const percent = (value: number) => `${Math.round(value * 100)}%`;

export function whyCardFor(event: KMeansEvent): WhyCardContent {
  switch (event.kind) {
    case "reset":
      return {
        key: "reset",
        title: "Two flags, and probably the wrong number of them",
        body: `Villages join their nearest flag; flags then move to the middle of whoever joined them. Run those two steps until nothing moves. Your real job is deciding how many flags there should be — the elbow chart is how you find out.`,
        tone: "info",
        conceptHref: CLUSTERING_HREF,
        conceptLabel: "How k-means works",
      };

    case "flag-moved":
      return {
        key: `moved-${event.flag}-${event.inertia.toFixed(4)}`,
        title: `Flag ${event.flag} moved — inertia ${num(event.inertia)}`,
        body: `The colouring is now out of date: villages still belong to whoever they joined last time. Press Assign to let them switch. Where you drop flags is the initialization, and it decides which answer you end up in.`,
        tone: "info",
      };

    case "k-changed": {
      const { k, elbowK, added } = event;
      if (added) {
        return {
          key: `k-${k}-up`,
          title: `${k} flags now`,
          body:
            k > elbowK
              ? `Adding flags always lowers inertia — with one flag per village it would hit zero. So a smaller number here is not automatically a better answer; check the elbow chart before you trust it.`
              : `More flags can capture more separate groups. Assign and settle to see whether this one found real structure or just split an existing group.`,
          tone: k > elbowK ? "warn" : "info",
          conceptHref: ELBOW_HREF,
          conceptLabel: "Why inertia can't choose k",
        };
      }
      return {
        key: `k-${k}-down`,
        title: `${k} flags now`,
        body: `Inertia will rise — fewer flags means villages are further from the one they joined. That's expected, and it's why inertia alone can't tell you the right number.`,
        tone: "info",
        conceptHref: ELBOW_HREF,
      };
    }

    case "assigned": {
      const empty = event.sizes.filter((size) => size === 0).length;
      const smallest = Math.min(...event.sizes);
      return {
        key: `assigned-${event.inertia.toFixed(4)}`,
        title: `Assigned — inertia ${num(event.inertia)}`,
        body: empty
          ? `${empty} flag${empty === 1 ? "" : "s"} ended up with no villages at all. A flag with no members has no mean to move to, so it will sit there for ever — move it somewhere useful or take it off the board.`
          : `Every village joined its nearest flag. The smallest territory has ${smallest} villages. Now press Update to move each flag to the middle of its own territory.`,
        tone: empty ? "warn" : "good",
      };
    }

    case "updated":
      return {
        key: `updated-${event.inertia.toFixed(4)}-${event.shift.toFixed(4)}`,
        title: event.converged
          ? "Nothing moved — converged"
          : `Flags moved ${num(event.shift * 100)} units`,
        body: event.converged
          ? `Every flag is already at the centre of its own territory, so further steps change nothing. Inertia ${num(event.inertia)} is where this layout lands.`
          : `Each flag jumped to the mean of its villages, so inertia is now ${num(event.inertia)}. Some villages are probably closer to a different flag — press Assign again.`,
        tone: event.converged ? "good" : "info",
      };

    case "stepped":
      return {
        key: `stepped-${event.iteration}-${event.inertia.toFixed(4)}`,
        title: event.converged
          ? `Converged after ${event.iteration} iteration${event.iteration === 1 ? "" : "s"}`
          : `Iteration ${event.iteration} — inertia ${num(event.inertia)}`,
        body: event.converged
          ? `Assign and update now both leave everything where it is. That's what convergence means: a fixed point of the two-step loop, not a fixed number of rounds.`
          : `Assign then update, once. Inertia never goes up across a full step — that's why the loop is guaranteed to settle, though not necessarily on the best answer.`,
        tone: event.converged ? "good" : "info",
      };

    case "settled":
      return {
        key: `settled-${event.inertia.toFixed(4)}`,
        title: `Settled after ${event.iterations} iteration${event.iterations === 1 ? "" : "s"}`,
        body: `Ran assign/update until no flag moved. Inertia ${num(event.inertia)}. This is the answer your starting layout leads to — a different layout with the same number of flags can land somewhere else entirely.`,
        tone: "good",
        conceptHref: CLUSTERING_HREF,
      };

    case "checked": {
      const {
        outcome,
        k,
        inertia,
        bestInertiaAtK,
        elbowK,
        emptyClusters,
        score,
        convergenceQuality,
      } = event.evaluation;

      const key = `checked-${outcome}-${score.toFixed(4)}`;

      switch (outcome) {
        case "empty-cluster":
          return {
            key,
            title: "A flag with no villages",
            body: `${emptyClusters.length} of your ${k} flags own nothing. The update step moves a flag to the average of its members, and an average of nothing is undefined — so it's frozen. Drop it next to real villages, or remove it.`,
            tone: "bad",
            conceptHref: CLUSTERING_HREF,
          };

        case "bad-k":
          return {
            key,
            title: "Bad k",
            body:
              k > elbowK
                ? `${k} flags got inertia down to ${num(inertia)} — lower than the right answer would. That's the trap: inertia always falls as you add flags, so it can't tell you when to stop. The elbow chart flattens after ${elbowK}, which is where the real groups run out.`
                : `${k} flags can't cover ${elbowK} separate groups. You converged, and ${num(inertia)} really is the best ${k} flags can manage — the limit is the number of flags, not where you put them. One of them is sitting between two clusters, serving both badly.`,
            tone: "bad",
            conceptHref: ELBOW_HREF,
            conceptLabel: "Reading an elbow chart",
          };

        case "local-minimum":
          return {
            key,
            title: "Local minimum",
            body: `Converged at inertia ${num(inertia)}, but ${k} flags can reach ${num(bestInertiaAtK)} from a better starting layout — you're at ${percent(convergenceQuality)} of what this k can do. The algorithm did its job; it just found the nearest good answer rather than the best one. That's initialization deciding the outcome.`,
            tone: "bad",
            conceptHref: CLUSTERING_HREF,
            conceptLabel: "Why initialization matters",
          };

        case "not-converged":
          return {
            key,
            title: "Not settled yet",
            body: `Keep stepping until no flag moves. Convergence is a state you reach, not a number of turns — and you can't judge a layout that's still moving.`,
            tone: "warn",
          };

        case "win":
          return {
            key,
            title: `Territory settled — ${percent(score)}`,
            body: `${k} flags, inertia ${num(inertia)}, matching the best a ${k}-flag layout can do here. You found the number of groups the data actually has, and you converged to the good answer rather than a nearby mediocre one.`,
            tone: "good",
            conceptHref: ELBOW_HREF,
          };

        case "near-miss":
        default:
          return {
            key,
            title: `${percent(score)} — needs ${percent(WIN_SCORE)}`,
            body: `Right number of flags, but inertia ${num(inertia)} against ${num(bestInertiaAtK)} achievable. Try moving a flag into the middle of a territory it's currently sharing, then settle again. Up to ${MAX_K} flags are allowed.`,
            tone: "warn",
            conceptHref: CLUSTERING_HREF,
          };
      }
    }
  }
}
