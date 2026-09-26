import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CodeLane, MAX_RUN_MS } from "./CodeLane";
import { architectureOf, evaluate, patternById } from "./ml";
import type { TrainerApi } from "./useTrainer";

/**
 * The code lane's run budget, against the api it actually runs.
 *
 * Every `api.train()` is a real fit, so a runaway loop spends its time inside
 * the api and never calls `checkBudget()` itself. These tests stand a fake
 * trainer in for TF.js and move the clock by a fixed amount per "fit", so the
 * 15-minute budget can be crossed in milliseconds.
 */
function fakeTrainer(fitMs: number) {
  let elapsed = 0;
  const realNow = Date.now.bind(Date);
  vi.spyOn(Date, "now").mockImplementation(() => realNow() + elapsed);

  const pattern = patternById("circle");
  const architecture = architectureOf([{ neurons: 4, activation: "relu" }]);
  const evaluation = evaluate({ pattern, architecture, accuracy: 0.91, loss: 0.2 });
  const train = vi.fn(async () => {
    elapsed += fitMs;
    return evaluation;
  });
  const trainer: TrainerApi = {
    train,
    stop: () => {},
    stopAndWait: async () => {},
    training: false,
    epoch: 0,
    totalEpochs: 160,
    error: null,
  };
  return { trainer, train };
}

function runScript(code: string | null) {
  if (code !== null) {
    fireEvent.change(screen.getByLabelText("Architecture script"), {
      target: { value: code },
    });
  }
  fireEvent.click(screen.getByRole("button", { name: "Run" }));
}

/** Scoped to the output: the starter's own source mentions the same words. */
const output = () => within(screen.getByRole("region", { name: "Script output" }));

afterEach(() => {
  vi.restoreAllMocks();
});

describe("the code lane's run budget", () => {
  it("stops a script that loops on api.train() forever", async () => {
    // 70 s per fit: twelve fit inside 900 s, the thirteenth ends at 910 s.
    const { trainer, train } = fakeTrainer(70_000);
    render(<CodeLane trainer={trainer} />);
    runScript("for (;;) { await api.train(); }");

    await waitFor(
      () =>
        expect(
          output().getAllByText(new RegExp(`ran longer than ${MAX_RUN_MS}ms`)).length,
        ).toBeGreaterThan(0),
      { timeout: 10_000 },
    );
    // Stopped at the first api call past the budget, not one fit later.
    expect(train).toHaveBeenCalledTimes(Math.floor(MAX_RUN_MS / 70_000) + 1);
  });

  it("lets an honest multi-fit script finish", async () => {
    // The starter's two fits at five minutes each: slow, and still inside.
    const { trainer, train } = fakeTrainer(300_000);
    render(<CodeLane trainer={trainer} />);
    runScript(null);

    await waitFor(
      () =>
        expect(
          output().getByText(/4 relu neurons,\s+1 layer\s+->\s+91\.0% win/),
        ).toBeInTheDocument(),
      { timeout: 10_000 },
    );
    expect(train).toHaveBeenCalledTimes(2);
    expect(output().queryByText(/ran longer than/)).toBeNull();
  });
});
