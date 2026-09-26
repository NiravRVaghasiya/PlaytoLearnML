import type { WhyCardContent } from "@/components";
import {
  CHANCE_RATE,
  DISHES,
  FILTER_BUDGET,
  ONE_LAYER_CEILING,
  TARGET_ACCURACY,
  clearlyAboveChance,
  detectorsAboveLayerOne,
  filtersUsed,
  isBlank,
  isSingleSigned,
  kernelSum,
  outputChannels,
  stackGeometry,
  type Evaluation,
  type Kernel,
  type KernelPreset,
  type Layer,
  type LearnedResult,
  type PoolType,
} from "./ml";

/**
 * "Why did that happen?" copy for Convolution Kitchen.
 *
 * The rule: never describe the feature map, explain it. "The map lit up along the
 * left edges" is something the player can already see. "The map lit up there
 * because your positive weights landed on bright pixels and your negative weights
 * landed on dark ones, and the sum went high" is the sentence that transfers.
 *
 * The other rule is about the ceiling. This game makes an unusually strong claim —
 * that one layer CANNOT do the job — so the copy is careful to attach that claim to
 * the reason (a global average leaves one number per filter) and to offer the
 * player the experiment that checks it, rather than asking to be believed.
 */
export type KitchenEvent =
  | { kind: "briefing" }
  | { kind: "weight-changed"; kernel: Kernel; layerIndex: number }
  | { kind: "preset-applied"; preset: KernelPreset; layerIndex: number }
  | {
      kind: "pool-changed";
      pool: PoolType;
      layerIndex: number;
      layers: readonly Layer[];
      previousAccuracy: number;
    }
  | { kind: "layer-added"; layers: readonly Layer[] }
  | { kind: "budget-spent" }
  | {
      kind: "scored";
      evaluation: Evaluation;
      layers: readonly Layer[];
      baseline: number | null;
    }
  | {
      kind: "learned";
      learned: LearnedResult;
      layers: readonly Layer[];
      mine: number;
    }
  | { kind: "error"; message: string };

const percent = (value: number) => `${Math.round(value * 100)}%`;

/**
 * The one Concept Library page this game can honestly point at. "Let it learn"
 * is gradient descent on the kernels themselves, and that page explains the loop.
 * Nothing in the library covers convolution yet, so no other card links out.
 */
const GD_HREF = "/concepts/gradient-descent";

export function whyCardFor(event: KitchenEvent): WhyCardContent {
  switch (event.kind) {
    case "briefing":
      return {
        key: "briefing",
        title: "Nine numbers, slid over a picture",
        body: `That is all a convolution is, and it is all you get to design. The kitchen opens with a Blur in slot one, which is the filter almost everyone reaches for first and the one thing guaranteed not to work here — every plate was made with randomised brightness, so a filter that averages nine pixels is measuring the one quantity that carries no answer. Four dishes to tell apart, ${FILTER_BUDGET} filters to spend, and a classifier on top that you cannot touch.`,
        tone: "info",
      };

    case "weight-changed": {
      const { kernel, layerIndex } = event;
      const sum = kernelSum(kernel);
      const blank = isBlank(kernel);
      const oneSided = isSingleSigned(kernel) && !blank;

      if (blank) {
        return {
          key: `blank-${layerIndex}`,
          title: "Every weight is zero",
          body: `The sum of nine zeros times anything is zero, so this filter's feature map is a flat black square no matter what you slide it over. Nothing downstream can use a constant.`,
          tone: "warn",
        };
      }

      if (oneSided) {
        return {
          key: `onesided-${layerIndex}-${sum}`,
          title: `Weights sum to ${sum > 0 ? "+" : ""}${sum}, all one sign`,
          body: `With no sign change, the nine multiplies can only ever add up — so this filter reports a weighted average of local brightness and cannot report anything else. It has no opposite: there is no patch that makes it answer "definitely not this". Put at least one negative weight next to a positive one and it becomes a detector of the boundary between them.`,
          tone: "warn",
        };
      }

      return {
        key: `weights-${layerIndex}-${kernel.weights.join(",")}`,
        title: `${kernel.label}, weights sum to ${sum > 0 ? "+" : ""}${sum}`,
        body: `A sum near zero is what you want from a detector: it means the filter ignores flat regions — where every pixel is about the same, the positives and negatives cancel — and only answers where the brightness CHANGES. ${
          sum < 0
            ? `Careful though: a negative total means most patches produce a negative answer, and the ReLU turns every one of those into zero. Watch the mean activation.`
            : `Slide the window over the picture and watch the nine products: the answer is large exactly where the patch under it looks like the kernel.`
        }`,
        tone: "info",
      };
    }

    case "preset-applied": {
      const { preset, layerIndex } = event;
      return {
        key: `preset-${layerIndex}-${preset.label}`,
        title: preset.label,
        body: `${preset.hint} ${
          isSingleSigned({ id: "", label: "", weights: [...preset.weights] })
            ? layerIndex === 0
              ? `Which makes it a brightness meter on raw pixels, and brightness is the one thing these plates randomise. Expect it to score nothing on its own.`
              : `In layer 2 that is legitimate: the input here is a feature map, not a photograph, and averaging a map is a reasonable way to pass its energy on.`
            : `Every classic edge detector has this shape — positive on one side, negative on the other, near-zero total.`
        }`,
        tone: "info",
      };
    }

    case "pool-changed": {
      const { pool, layerIndex, layers } = event;
      const geometry = stackGeometry(layers);
      const twoLayers =
        layers.filter((layer) => layer.kernels.length > 0).length > 1;

      return {
        key: `pool-${layerIndex}-${pool}`,
        title:
          pool === "none"
            ? `Layer ${layerIndex + 1}: pooling off`
            : `Layer ${layerIndex + 1}: ${pool === "max" ? "max" : "average"} 2x2`,
        body: `${
          pool === "max"
            ? `Max pooling keeps the strongest response in each block, which asks "is this pattern present anywhere near here". It saturates: a picture with stripes everywhere and a picture with stripes in half of it can both peak at the same value.`
            : pool === "avg"
              ? `Average pooling keeps the mean response, which asks "how much of this pattern is around here". On this menu that is usually the more useful question, because two of the dishes differ in how much of each texture they contain.`
              : `No pooling keeps the map at full size. Note that with a global average at the end of the stack, average pooling and no pooling produce almost the same numbers — averaging twice is still averaging. Pooling's real job here is the next line.`
        } ${
          twoLayers
            ? `One cell of your output now sees ${geometry.receptiveField} pixels of the original picture. That is what pooling buys: the layer above looks at a 3x3 window of an already-shrunken map, so it covers far more ground than three pixels.`
            : `Add a second layer and this choice starts to matter a lot: pooling is what lets the layer above see ${geometry.receptiveField} pixels at a time instead of three.`
        }`,
        tone: "info",
      };
    }

    case "layer-added": {
      const geometry = stackGeometry(event.layers);
      return {
        key: `layer-added-${event.layers.length}`,
        title: "Layer 2 sees feature maps, not pixels",
        body: `This is the step that matters. A filter here slides over layer 1's OUTPUT, so it responds to arrangements of layer-1 patterns — "vertical texture with less vertical texture below it" is a thing a 3x3 kernel can express up here and cannot express down there. One output cell now covers ${geometry.receptiveField} pixels of the original. The pass-through filter is not padding: it carries layer 1's raw amounts forward, so the classifier gets both how much of each texture there is AND how it is arranged. Take it out and you lose the first half.`,
        tone: "info",
      };
    }

    case "budget-spent":
      return {
        key: "budget-spent",
        title: `All ${FILTER_BUDGET} filters are spent`,
        body: `Remove one before adding another. The budget is the point rather than a limitation: the spec scores this kitchen on the fewest, cleanest filters, and a stack that wins with three beats a stack that wins with six.`,
        tone: "warn",
      };

    case "scored": {
      const { evaluation, layers, baseline } = event;
      const { score, outcome } = evaluation;
      const channels = outputChannels(layers);
      const filters = filtersUsed(layers);

      if (outcome === "served") {
        const deep = layers.filter((layer) => layer.kernels.length > 0).length > 1;
        return {
          key: `served-${filters}-${Math.round(score.accuracy * 1000)}`,
          title: `Served — ${percent(score.accuracy)} on ${filters} filter${
            filters === 1 ? "" : "s"
          }`,
          body: `${
            deep
              ? `The second layer is what did it, and it is worth being precise about why. Layer 1 reports how much of each texture is in the picture; the global average at the end destroys any record of where it was. A filter in layer 2 slides over layer 1's MAP, so it can answer "more of this above than below" — a position, converted into a quantity, before the average gets to it. That is the whole of what people mean by hierarchical features: edges, then textures, then arrangement. `
              : ``
          }Look at what did the work: the classifier on top is a single linear layer that never changed, and the same layer on raw pixels manages ${
            baseline === null ? "barely above chance" : percent(baseline)
          } against ${percent(
            CHANCE_RATE,
          )} for guessing. Every point above that came from ${filters} kernels — ${
            filters * 9
          } numbers in total — reused at every position in the image. That reuse is the entire reason convolution scales: the same ${
            filters * 9
          } numbers would work on a picture a thousand times larger.`,
          tone: "good",
        };
      }

      if (outcome === "blur-only") {
        // With a sign-changing filter in layer 2, "every feature map looks like
        // the photograph" is false — that filter's maps do not — so the card says
        // what is actually true: layer 1 is handing layer 2 a photograph. With
        // only pass-throughs and blurs up there, the photograph claim IS true, and
        // the single-layer card below is the right one.
        const detectors = detectorsAboveLayerOne(layers);
        if (detectors.length > 0) {
          const one = detectors.length === 1;
          const names = `${detectors.map((label) => `"${label}"`).join(" and ")} in layer 2`;
          return {
            key: "blur-only-deep",
            title: "Layer 1 is only passing the photograph on",
            body: `Every filter in layer 1 has no sign change, so its map is a softened copy of the picture — and that copy is all layer 2 gets to read. ${
              clearlyAboveChance(score.accuracy)
                ? `${names} can still find stripes in it, which is why this scores ${percent(
                    score.accuracy,
                  )} rather than ${percent(CHANCE_RATE)}, but ${
                    one ? "it is" : "they are"
                  } doing a first layer's job one level up, and the stack gets none of what depth is for.`
                : `${names} can look for stripes in it, but at ${percent(
                    score.accuracy,
                  )} against ${percent(CHANCE_RATE)} for guessing ${
                    one ? "it is" : "they are"
                  } finding almost nothing there — and even a detector that did would be doing a first layer's job one level up, which gets the stack none of what depth is for.`
            } Make one layer-1 weight negative and its map stops looking like a picture. That is the point of layer 1: a feature map is not an image, it is a map of where a pattern was found — and arrangement is something layer 2 can only read off a map like that.`,
            tone: "bad",
          };
        }
        return {
          key: "blur-only",
          title: "Every feature map still looks like the photograph",
          body: `That is the tell. A filter with no sign change cannot do anything except smooth or brighten, so its map is a softened copy of the input — and a copy of the input is exactly what the classifier already could not use. ${percent(
            score.accuracy,
          )} against ${percent(
            CHANCE_RATE,
          )} for guessing. Make one weight negative and watch the map turn into something that does not look like a picture any more. That moment is the whole idea: a feature map is not an image, it is a map of where a pattern was found.`,
          tone: "bad",
        };
      }

      if (outcome === "dead-filters") {
        const dead = evaluation.health.filter((filter) => filter.dead);
        return {
          key: `dead-${dead.length}-${dead[0]?.label ?? ""}`,
          title: `${dead.length} filter${
            dead.length === 1 ? "" : "s"
          } producing nothing at all`,
          body: `Its feature map is uniformly zero, which the gallery shows as a flat black tile. The reason is the ReLU: it replaces every negative answer with zero, so a kernel whose weights sum well below zero answers negative almost everywhere and is silenced everywhere. This is worth recognising by sight, because in a real network nothing tells you — the loss still falls, the training still finishes, and a slice of the model was never doing anything.`,
          tone: "bad",
        };
      }

      if (outcome === "duplicate-filters") {
        const pair = evaluation.duplicates[0];
        return {
          key: `dup-${pair?.labelA}-${pair?.labelB}`,
          title: "Two filters, one detector",
          body: `"${pair?.labelA}" and "${
            pair?.labelB
          }" respond to the same thing — their maps correlate at ${
            pair?.correlation.toFixed(3) ?? "?"
          }. Note this is measured on the activations, not on the weights: kernels that look different on the grid can be the same filter in practice, and a kernel next to its own negation looks identical but is NOT a duplicate, because after the ReLU they catch opposite edge polarities. You have ${filters} of ${FILTER_BUDGET} filters spent and ${channels} channels reaching the classifier.`,
          tone: "warn",
        };
      }

      if (outcome === "one-layer-ceiling") {
        return {
          key: `ceiling-${Math.round(score.accuracy * 1000)}`,
          title: "This is a wall, not a plateau",
          body: `Your filters are fine. Read the per-class row: ${DISHES.map(
            (dish, index) => `${dish.short} ${percent(score.perClass[index] ?? 0)}`,
          ).join(", ")}. The two arrangement dishes are being mistaken for each other, and with one layer they will keep being mistaken, because your stack ends in a global average — one layer gives the classifier one number per filter, "how much of this pattern is in the picture", and those two dishes contain the same amount of everything. They differ only in where it is. Apart from the few rows where the two textures meet, nothing you type into a 3x3 grid can see that; a second layer can, because it sees WHERE layer 1 responded before the average erases it.`,
          tone: "bad",
        };
      }

      const worst = score.perClass.reduce(
        (lowest, value, index) => (value < (score.perClass[lowest] ?? 1) ? index : lowest),
        0,
      );
      return {
        key: `progress-${Math.round(score.accuracy * 1000)}-${filters}`,
        title: `${percent(score.accuracy)} — ${percent(
          TARGET_ACCURACY,
        )} needed`,
        body: `${channels} channel${
          channels === 1 ? "" : "s"
        } reaching the classifier from ${filters} filter${
          filters === 1 ? "" : "s"
        }. The dish it is worst at is "${DISHES[worst]!.label}" at ${percent(
          score.perClass[worst] ?? 0,
        )} — ${
          DISHES[worst]!.tell
        } Look at that dish in the gallery and find the filter that ought to be lighting up on it. ${
          score.trainAccuracy - score.accuracy > 0.08
            ? `Also worth noting: ${percent(
                score.trainAccuracy,
              )} on the training plates against ${percent(
                score.accuracy,
              )} on unseen ones. The gap means some of those channels are memorising rather than detecting.`
            : ``
        }`,
        tone: "info",
      };
    }

    case "learned": {
      const { learned, layers, mine } = event;
      const single =
        layers.filter((layer) => layer.kernels.length > 0).length < 2;
      const better = learned.accuracy - mine;

      // One layer: learnStack is exactly the player's shape. Two layers: its
      // depthwise layer 2 learns a grid per input map where the player's shares
      // one, so "exact" would overclaim — say which it is.
      return {
        key: `learned-${Math.round(learned.accuracy * 1000)}`,
        title: `Gradient descent got ${percent(learned.accuracy)} out of ${
          single ? "your exact shape" : "the shape of your stack"
        }`,
        body: `${
          single && learned.accuracy < TARGET_ACCURACY
            ? `It had a free hand with all ${
                filtersUsed(layers) * 9
              } weights and it still could not pass, which settles the question: the shape is the limit, not your kernel design. One layer plus a global average has never been measured past ${percent(
                ONE_LAYER_CEILING,
              )} on this menu.`
            : better > 0.05
              ? `That is ${percent(
                  better,
                )} above your hand-designed version on the same architecture, so there is room left in the weights.`
              : `Which is about what you got by hand — you found kernels as good as the ones gradient descent settles on. That is a real result, and it is also the last time it will happen: this is a 3x3 filter over one channel, the largest thing a person can reason about directly.`
        } ${
          single
            ? ``
            : `One difference to keep in mind: its layer 2 learned a separate 3x3 grid for each layer-1 map, where yours shares one grid across all of them, so it had a little more freedom than your version does. `
        }Its kernels are drawn under the Let it learn button — look at what it actually picked. Some of those grids will be recognisable — a positive band against a negative band is an edge detector however you arrive at it. Others will look like nothing in particular and work anyway. A real layer-2 filter mixes every input channel, so it is 3x3xC numbers with no grid to draw it on, and that is where hand-designing stops for good.`,
        tone: "info",
        conceptHref: GD_HREF,
        conceptLabel: "How gradient descent works",
      };
    }

    case "error":
      return {
        key: `error-${event.message.slice(0, 24)}`,
        title: "That did not run",
        body: event.message,
        tone: "bad",
      };
  }
}

/** Caption for the sliding window: what the nine numbers mean. */
export function windowCaption(sum: number, activated: number): string {
  if (activated > 0) {
    return `The nine products add to ${sum.toFixed(
      2,
    )}, and the ReLU passes it through unchanged. This patch looks like the kernel.`;
  }
  if (sum === 0) {
    return `The nine products add to exactly 0. Either the patch is flat, or the positives and negatives cancelled — both mean "nothing to report here".`;
  }
  return `The nine products add to ${sum.toFixed(
    2,
  )}, which is negative, so the ReLU outputs 0. The patch looks like the kernel's opposite, and "definitely not this pattern" and "not this pattern" are the same answer to the next layer.`;
}
