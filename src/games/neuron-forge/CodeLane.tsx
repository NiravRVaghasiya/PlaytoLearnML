"use client";

import { useMemo } from "react";
import { Play, RotateCcw } from "lucide-react";
import { Button, CodeEditor } from "@/components";
import { useCodeLane } from "@/engine/useCodeLane";
import {
  ACTIVATIONS,
  MAX_LAYERS,
  MAX_NEURONS_PER_LAYER,
  architectureOf,
  isEffectivelyLinear,
  patternById,
  type Activation,
  type Layer,
  type PatternId,
} from "./ml";
import { budgetRemaining, useNeuronForgeStore } from "./store";
import type { TrainerApi } from "./useTrainer";

/**
 * Neuron Forge — the code lane.
 *
 * `api.setLayers` writes the same `layers` array the steppers write, and
 * `api.train` is the same trainer the Train button calls, so a design built here
 * and one clicked out by hand are the same model (CLAUDE.md two-lane rule).
 *
 * The starter snippet is the game's argument, run as an experiment: it trains the
 * entire budget as linear layers, then HALF the budget as a single relu layer, and
 * prints both. The smaller network wins. Nothing about that is narrated in the
 * copy — the numbers come out of two real `model.fit` calls.
 */

const STARTER_CODE = `// Two architectures on the same circle. One wins.
// Both are real tf.sequential models; the numbers below are measured.

api.setPattern('circle');
log('budget', api.budget(), 'neurons | target', (api.target() * 100) + '%');

// 1. Spend the WHOLE budget, two layers deep, all linear.
api.setLayers([
  { neurons: 4, activation: 'linear' },
  { neurons: 4, activation: 'linear' },
]);
const flat = await api.train();
log('8 linear neurons, 2 layers ->', pct(flat.accuracy), flat.outcome);

// 2. Half the neurons. One layer. One word different.
api.setLayers([
  { neurons: 4, activation: 'relu' },
]);
const bent = await api.train();
log('4 relu neurons,   1 layer  ->', pct(bent.accuracy), bent.outcome);

function pct(x) { return (x * 100).toFixed(1) + '%'; }

// Half the size, and it solves what the bigger one could not.
//
// Stacked linear layers multiply out to a single matrix, so all eight of
// those neurons describe one straight cut. Capacity is not a neuron count.
//
// Now try:
//   api.setPattern('xor')    - 4 relu still enough?
//   api.setPattern('spiral') - try [{8,'relu'}] then [{8,'relu'},{8,'relu'}]
//                              same width, same activation, only depth differs`;

/** Fifteen minutes. See the note where the lane is created. */
export const MAX_RUN_MS = 900_000;

export interface CodeLaneProps {
  trainer: TrainerApi;
}

/**
 * Validate one `{ neurons, activation }` from a script.
 *
 * Out-of-budget sizes are clamped by the store, on purpose (see `setLayers`).
 * What is rejected here is input that is not a size at all — NaN, a string, zero
 * — which the store would otherwise quietly turn into a 1-neuron layer, so the
 * script's author would be told nothing while training something they never
 * asked for.
 */
function checkLayer(layer: unknown, where: string): Layer {
  const candidate = layer as Partial<Layer> | null;
  if (typeof candidate !== "object" || candidate === null) {
    throw new TypeError(`${where} must be an object like { neurons: 4, activation: 'relu' }`);
  }
  if (
    typeof candidate.neurons !== "number" ||
    !Number.isFinite(candidate.neurons) ||
    candidate.neurons < 1
  ) {
    throw new RangeError(
      `${where}: neurons must be a number of at least 1 — got ${String(candidate.neurons)}`,
    );
  }
  if (!ACTIVATIONS.includes(candidate.activation as Activation)) {
    throw new RangeError(
      `${where}: activation must be one of ${ACTIVATIONS.join(", ")} — got "${String(candidate.activation)}"`,
    );
  }
  return { neurons: candidate.neurons, activation: candidate.activation as Activation };
}

export function CodeLane({ trainer }: CodeLaneProps) {
  const store = useNeuronForgeStore;

  const api = useMemo(
    () => ({
      /** Switch puzzle. Clears the architecture, exactly like the selector. */
      setPattern: (id: PatternId) => {
        if (!patternById(id) || !["linear", "circle", "xor", "spiral"].includes(id)) {
          throw new Error(
            `unknown pattern "${id}" — try linear, circle, xor or spiral`,
          );
        }
        store.getState().setPattern(id);
      },

      /**
       * Replace the architecture. Over-budget or oversized layers are clamped
       * rather than rejected, so the code lane can never reach a state the visual
       * lane could not also express.
       */
      setLayers: (layers: Layer[]) => {
        if (!Array.isArray(layers)) {
          throw new TypeError("setLayers needs an array of { neurons, activation }");
        }
        store
          .getState()
          .applyArchitecture(
            layers.map((layer, index) => checkLayer(layer, `layer ${index + 1}`)),
          );
      },

      addLayer: (neurons: number, activation: Activation = "relu") => {
        const current = store.getState().layers;
        const layer = checkLayer({ neurons, activation }, "addLayer");
        store.getState().applyArchitecture([...current, layer]);
      },

      /**
       * Train the current architecture. Real `model.fit`, so await it.
       *
       * Credited to the code lane (`fromCode`), which is what mastery's third
       * star asks for. Throws rather than resolving null, so a script that forgot
       * an `await` gets a sentence instead of "cannot read accuracy of null".
       */
      train: async () => {
        const result = await trainer.train({ fromCode: true });
        if (result === null) {
          throw new Error(
            "api.train() did not run: another training run is still in progress. Await each api.train() before starting the next.",
          );
        }
        return result;
      },

      layers: () => store.getState().layers.map((layer) => ({ ...layer })),
      neurons: () => architectureOf(store.getState().layers).totalNeurons,
      /** True when the whole stack collapses to one straight cut. */
      isLinear: () => isEffectivelyLinear(architectureOf(store.getState().layers)),

      budget: () => patternById(store.getState().patternId).budget,
      remaining: () => budgetRemaining(store.getState()),
      target: () => patternById(store.getState().patternId).target,
      pattern: () => store.getState().patternId,
      maxLayers: () => MAX_LAYERS,
      maxNeuronsPerLayer: () => MAX_NEURONS_PER_LAYER,

      /** Held-out accuracy from the last run, null before training. */
      accuracy: () => store.getState().accuracy,
      loss: () => store.getState().loss,
      epoch: () => store.getState().epoch,
      lossHistory: () => [...store.getState().lossHistory],
      evaluation: () => store.getState().lastEvaluation,

      reset: () => store.getState().reset(),
    }),
    [store, trainer],
  );

  // Generous: each api.train() is a real fit, and a ladder script runs half a
  // dozen of them — on a CPU backend that is minutes, not seconds. The budget
  // exists to stop a runaway loop, never an honest experiment. It is checked on
  // every api call (`budgetApiCalls`), because the api is where the time goes:
  // without that, `for (;;) await api.train()` never calls checkBudget() and
  // never stops.
  const lane = useCodeLane({
    initialCode: STARTER_CODE,
    api,
    maxRunMs: MAX_RUN_MS,
    budgetApiCalls: true,
  });

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="primary"
          size="sm"
          onClick={() => void lane.run()}
          disabled={lane.running || trainer.training}
          icon={<Play className="size-4" />}
        >
          {lane.running ? "Running…" : "Run"}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={lane.reset}
          icon={<RotateCcw className="size-4" />}
        >
          Restore snippet
        </Button>
        {lane.running && trainer.training ? (
          <span className="font-mono text-xs text-text-muted">
            training epoch {trainer.epoch} of {trainer.totalEpochs}
          </span>
        ) : lane.dirty ? (
          <span className="font-mono text-xs text-text-muted">edited</span>
        ) : null}
      </div>

      <CodeEditor
        className="min-h-0 flex-1"
        label="Architecture script"
        value={lane.code}
        onChange={lane.setCode}
        language="javascript"
        error={lane.error}
        rows={18}
        hint="api.setPattern · api.setLayers · api.addLayer · api.train · api.layers · api.neurons · api.isLinear · api.budget · api.remaining · api.target · api.accuracy · api.loss · api.lossHistory · api.evaluation · api.reset"
      />

      <section aria-label="Script output" className="min-h-24">
        <h2 className="mb-1 text-xs font-semibold tracking-wide text-text-muted uppercase">
          Output
        </h2>
        {lane.logs.length === 0 ? (
          <p className="font-mono text-xs text-text-muted">
            Run the script to see output here. Each train() takes a few seconds —
            it is a real network, not a lookup.
          </p>
        ) : (
          <ul className="max-h-40 overflow-auto rounded-md border border-border bg-bg p-2 font-mono text-xs">
            {lane.logs.map((entry, index) => (
              <li
                key={index}
                className={
                  entry.level === "error" ? "text-wrong" : "text-text-muted"
                }
              >
                {entry.message}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
