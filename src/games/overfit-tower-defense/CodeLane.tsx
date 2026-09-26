"use client";

import { useMemo } from "react";
import { Play, RotateCcw } from "lucide-react";
import { Button, CodeEditor } from "@/components";
import { useCodeLane } from "@/engine/useCodeLane";
import {
  MAX_COMPLEXITY,
  MAX_TOWERS_PER_TYPE,
  MIN_COMPLEXITY,
  TOWER_TYPES,
  parameterCount,
  regularizationOf,
  type TowerType,
} from "./ml";
import { towerCount, useTowerDefenseStore } from "./store";
import type { TrainerApi } from "./useTrainer";

/**
 * Overfit Tower Defense — the code lane.
 *
 * `api.setComplexity` and `api.setTowers` write the same state the slider and the
 * palette write, and `api.trial`/`api.deploy` call the same trainer the Deploy
 * button calls (CLAUDE.md two-lane rule).
 *
 * The starter snippet is a real regularization sweep — the thing a practitioner
 * actually does. It trains five models on the current wave and prints train,
 * validation, gap and bias for each, so the U-shape falls out of measurements
 * rather than being asserted in prose. `trial` deliberately costs no core health,
 * because selecting on validation data before committing is the habit worth
 * building.
 */

const STARTER_CODE = `// A real regularization sweep on the current wave.
// Five models get trained; none of them cost the core any health.

api.setComplexity(32);           // plenty of capacity, on purpose
log('wave', api.wave(), '|', api.trainPoints(), 'points | ceiling',
    pct(api.achievable()));
log('');
log('L2 towers   train     val      gap     bias');

let best = null;

for (const l2 of [0, 1, 2, 4, 6]) {
  api.setTowers({ l1: 0, l2, dropout: 0 });
  const r = await api.trial();

  log('    ' + l2 + '       ' + pct(r.trainAccuracy) + '    ' + pct(r.validationAccuracy) +
      '    ' + pct(r.gap) + '    ' + pct(r.bias));

  if (best === null || r.validationAccuracy > best.val) {
    best = { l2, val: r.validationAccuracy };
  }
}

log('');
log('best validation accuracy at L2 x' + best.l2, '->', pct(best.val));

function pct(x) { return (x * 100).toFixed(1).padStart(5) + '%'; }

// Read the two right-hand columns together. Going down the list, the gap
// shrinks the whole way - but bias starts climbing, and validation
// accuracy peaks somewhere in the middle and then falls.
//
// That peak is what you select on, and it is not at either end. More
// regularization is not "safer". (The wave itself charges the gap and the
// bias separately, so glance at both columns for the one you pick.)
//
// Then commit the winner:  api.setTowers({ l2: best.l2 }); await api.deploy();`;

/** Fifteen minutes. See the note where the lane is created. */
export const MAX_RUN_MS = 900_000;

export interface CodeLaneProps {
  trainer: TrainerApi;
}

export function CodeLane({ trainer }: CodeLaneProps) {
  const store = useTowerDefenseStore;

  const api = useMemo(
    () => ({
      /** The bias-variance knob. Same action as the slider. */
      setComplexity: (complexity: number) => {
        if (!Number.isFinite(complexity)) {
          throw new Error("setComplexity needs a finite number");
        }
        if (complexity < MIN_COMPLEXITY || complexity > MAX_COMPLEXITY) {
          throw new Error(
            `complexity must be between ${MIN_COMPLEXITY} and ${MAX_COMPLEXITY}`,
          );
        }
        store.getState().setComplexity(complexity);
      },

      /**
       * Set tower counts. Same state the palette writes.
       *
       * Each count must be a whole number the palette could show. NaN used to
       * slip through as "no towers", silently standing a defence down.
       */
      setTowers: (counts: Partial<Record<TowerType, number>>) => {
        if (typeof counts !== "object" || counts === null) {
          throw new TypeError("setTowers needs an object like { l2: 3 }");
        }
        for (const [key, value] of Object.entries(counts)) {
          if (!TOWER_TYPES.includes(key as TowerType)) {
            throw new RangeError(
              `unknown tower "${key}" — try ${TOWER_TYPES.join(", ")}`,
            );
          }
          if (
            typeof value !== "number" ||
            !Number.isInteger(value) ||
            value < 0 ||
            value > MAX_TOWERS_PER_TYPE
          ) {
            throw new RangeError(
              `${key} towers must be a whole number from 0 to ${MAX_TOWERS_PER_TYPE} — got ${String(value)}`,
            );
          }
        }
        store.getState().setTowers(counts);
      },

      /**
       * Train and measure. Costs the core nothing, and works in any phase —
       * a trial after a wave resolves leaves it resolved.
       */
      trial: async () => {
        const measured = await trainer.trial();
        if (measured === null) {
          throw new Error(
            store.getState().phase === "training"
              ? "api.trial() did not run: another training run is still in progress. Await each trial() or deploy() before the next."
              : trainer.unscoredReason() === "failed"
                ? "api.trial() did not finish: the fit ended with an error before the model could be measured. The error is shown with the controls."
                : "api.trial() was stopped before the model finished training, so nothing was measured.",
          );
        }
        return measured;
      },
      /**
       * Train and let the wave hit. This one counts — once. Credited to the
       * code lane, which is what mastery's third star asks for.
       */
      deploy: async () => {
        const { phase, wave } = store.getState();
        if (phase === "resolved") {
          throw new Error(
            `wave ${wave} has already been fought — call api.nextWave() before deploying again`,
          );
        }
        if (phase === "won" || phase === "lost") {
          throw new Error(`the run is over (${phase}) — call api.restart() to play again`);
        }
        if (phase === "training") {
          throw new Error(
            "api.deploy() did not run: another training run is still in progress. Await each trial() or deploy() before the next.",
          );
        }
        const result = await trainer.deploy({ fromCode: true });
        if (result === null) {
          throw new Error(
            trainer.unscoredReason() === "failed"
              ? `api.deploy() did not finish: the fit ended with an error before the model could be measured, so wave ${wave} did not attack. The error is shown with the controls.`
              : `api.deploy() was stopped before the model finished training — wave ${wave} did not attack.`,
          );
        }
        return result;
      },
      /** Advance after a resolved wave. */
      nextWave: () => {
        const { phase, wave } = store.getState();
        if (phase !== "resolved") {
          throw new Error(
            phase === "tuning"
              ? `wave ${wave} has not been fought yet — await api.deploy() first`
              : `no next wave from phase "${phase}"`,
          );
        }
        store.getState().nextWave();
      },

      complexity: () => store.getState().modelComplexity,
      parameters: () => parameterCount(store.getState().modelComplexity),
      towers: () => ({
        l1: towerCount(store.getState(), "l1"),
        l2: towerCount(store.getState(), "l2"),
        dropout: towerCount(store.getState(), "dropout"),
      }),
      lambdas: () => regularizationOf(store.getState().towers),
      maxTowers: () => MAX_TOWERS_PER_TYPE,

      wave: () => store.getState().wave,
      trainPoints: () => store.getState().dataset.train.length,
      noiseRate: () => 1 - store.getState().dataset.achievable,
      achievable: () => store.getState().dataset.achievable,

      trainAccuracy: () => store.getState().trainAccuracy,
      validationAccuracy: () => store.getState().validationAccuracy,
      coreHp: () => store.getState().coreHp,
      phase: () => store.getState().phase,
      lastResult: () => store.getState().lastResult,

      /** Start over. A fit still in flight is stopped and its result retired. */
      restart: () => {
        trainer.stop();
        store.getState().restart();
      },
    }),
    [store, trainer],
  );

  // Five real fits in the starter snippet, and a full five-wave run is five
  // more — minutes on a CPU backend. The budget exists to stop a runaway loop,
  // never an honest experiment. It is checked on every api call
  // (`budgetApiCalls`), because the api is where the time goes: without that,
  // `for (;;) await api.trial()` never calls checkBudget() and never stops.
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
        label="Defence script"
        value={lane.code}
        onChange={lane.setCode}
        language="javascript"
        error={lane.error}
        rows={18}
        hint="api.setComplexity · api.setTowers · api.trial · api.deploy · api.nextWave · api.complexity · api.parameters · api.towers · api.lambdas · api.wave · api.trainPoints · api.achievable · api.coreHp · api.lastResult · api.restart"
      />

      <section aria-label="Script output" className="min-h-24">
        <h2 className="mb-1 text-xs font-semibold tracking-wide text-text-muted uppercase">
          Output
        </h2>
        {lane.logs.length === 0 ? (
          <p className="font-mono text-xs text-text-muted">
            Run the script to see output here. The sweep trains five real models,
            so give it a few seconds.
          </p>
        ) : (
          <ul className="max-h-40 overflow-auto rounded-md border border-border bg-bg p-2 font-mono text-xs whitespace-pre">
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
