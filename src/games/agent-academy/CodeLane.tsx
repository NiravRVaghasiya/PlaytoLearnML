"use client";

import { useMemo } from "react";
import { Play, RotateCcw } from "lucide-react";
import { Button, CodeEditor } from "@/components";
import { useCodeLane } from "@/engine/useCodeLane";
import { createCodeApi } from "./store";

/**
 * Agent Academy — the code lane.
 *
 * `api.setReward`, `api.setEpsilon` and `api.train` write exactly the state the
 * sliders and the Train button write (CLAUDE.md two-lane rule), so a snippet and a
 * drag are the same operation on the same store.
 *
 * The starter snippet runs the experiment the visual lane structurally cannot. The
 * grid shows one agent at a time, and every interesting claim in this game is a
 * claim about a DISTRIBUTION: "ε 0 never solves this" is not something one run can
 * establish, and neither is "cheese at 3 cannot be fixed by exploring harder".
 * `api.simulate` trains throwaway agents against the same environment without
 * touching the player's own agent or spending their episode budget, which is what
 * makes those questions askable at all.
 *
 * The sweep is 132 training runs and about a second of main-thread work on a
 * desktop, several on a phone. `api.simulate` stays synchronous on purpose —
 * `r.solved` on a Promise would silently read `undefined` in every snippet a
 * player has written — so the starter yields between rows instead, which keeps
 * each task short enough for the page to repaint. The sample size stays at twelve
 * seeds: the claims the tables settle are about distributions, and fewer seeds
 * would make the printed rates noisy.
 */

const STARTER_CODE = `// One agent is an anecdote. Sweep it.

const SEEDS = 12;
const EPISODES = 400;
// Each row trains twelve agents, which is real work. Pausing between rows
// lets the page repaint instead of freezing until the whole sweep is done.
const breathe = () => new Promise((resolve) => setTimeout(resolve, 0));

log('epsilon   solved   mean goal rate   cheese eaten   reward');
for (const epsilon of [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]) {
  let solved = 0, goal = 0, cheese = 0, reward = 0;

  for (let seed = 1; seed <= SEEDS; seed++) {
    checkBudget();
    const r = api.simulate({ epsilon, episodes: EPISODES, seed: seed * 7919 });
    if (r.solved) solved++;
    goal += r.goalRate;
    cheese += r.meanPellets;
    reward += r.meanReward;
  }

  log(String(epsilon).padEnd(9),
      (solved + '/' + SEEDS).padStart(6),
      (100 * goal / SEEDS).toFixed(0).padStart(15) + '%',
      (cheese / SEEDS).toFixed(1).padStart(14),
      (reward / SEEDS).toFixed(1).padStart(8));
  await breathe();
}

log('');
log('Now the other failure. Raise the cheese and sweep epsilon again:');
log('epsilon   solved   reward   cheese eaten');
for (const epsilon of [0.1, 0.3, 0.5, 0.9]) {
  let solved = 0, reward = 0, cheese = 0;
  for (let seed = 1; seed <= SEEDS; seed++) {
    checkBudget();
    const r = api.simulate({
      rewards: { pellet: 3 }, epsilon, episodes: EPISODES, seed: seed * 7919,
    });
    if (r.solved) solved++;
    reward += r.meanReward;
    cheese += r.meanPellets;
  }
  log(String(epsilon).padEnd(9),
      (solved + '/' + SEEDS).padStart(6),
      (reward / SEEDS).toFixed(1).padStart(8),
      (cheese / SEEDS).toFixed(1).padStart(14));
  await breathe();
}

log('');
log('Exploration fixed the first table and did nothing for the second,');
log('and here is why - no training involved, just arithmetic:');
for (const pellet of [1, 2, 3]) {
  const s = api.solve({ pellet });
  log('  cheese ' + pellet +
      ': round trip ' + s.roundTripValue.toFixed(1) +
      ', pacing worth ' + s.farmValue.toFixed(1) +
      ', best policy ' + (s.reachesGoal ? 'walks out' : 'never leaves') +
      ' for ' + s.episodeReward.toFixed(1));
}

log('');
log('Set what you want and train your own agent:');
log('  api.setEpsilon(0.3); api.train(300); log(api.report().goalRate)');`;

export function CodeLane() {
  const api = useMemo(() => createCodeApi(), []);
  const lane = useCodeLane({ initialCode: STARTER_CODE, api, maxRunMs: 60000 });

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="primary"
          size="sm"
          onClick={() => void lane.run()}
          disabled={lane.running}
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
        {lane.dirty ? (
          <span className="font-mono text-xs text-text-muted">edited</span>
        ) : null}
      </div>

      <CodeEditor
        className="min-h-0 flex-1"
        label="Training script"
        value={lane.code}
        onChange={lane.setCode}
        language="javascript"
        error={lane.error}
        rows={18}
        hint="api.simulate · api.solve · api.setReward · api.setEpsilon · api.train · api.forget · api.report · api.optimal · api.values · api.rewards · api.epsilon · api.episodes"
      />

      <section aria-label="Script output" className="min-h-24">
        <h2 className="mb-1 text-xs font-semibold tracking-wide text-text-muted uppercase">
          Output
        </h2>
        {lane.logs.length === 0 ? (
          <p className="font-mono text-xs text-text-muted">
            Run the script to see output here. `api.simulate` and `api.solve` cost
            you nothing — only `api.train` spends your episode budget.
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
