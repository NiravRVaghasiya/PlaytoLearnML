import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CodeLane, MAX_RUN_MS } from "./CodeLane";
import { useTowerDefenseStore } from "./store";
import type { Measurement, TrainerApi } from "./useTrainer";

/**
 * The code lane against the api it actually runs, with a fake trainer standing
 * in for TF.js. Each "fit" moves the clock by a fixed amount, so the 15-minute
 * run budget can be crossed in milliseconds.
 */
function fakeTrainer(
  fitMs: number,
  { unscored = null }: { unscored?: "stopped" | "failed" | null } = {},
) {
  let elapsed = 0;
  const realNow = Date.now.bind(Date);
  vi.spyOn(Date, "now").mockImplementation(() => realNow() + elapsed);

  const measured: Measurement = {
    trainAccuracy: 0.9,
    validationAccuracy: 0.84,
    gap: 0.06,
    bias: 0.02,
  };
  const trial = vi.fn(async () => {
    elapsed += fitMs;
    return unscored === null ? measured : null;
  });
  const deploy = vi.fn(async () => {
    elapsed += fitMs;
    return null;
  });
  const trainer: TrainerApi = {
    deploy,
    trial,
    stop: () => {},
    stopAndWait: async () => {},
    unscoredReason: () => unscored,
    training: false,
    epoch: 0,
    totalEpochs: 120,
    error: null,
  };
  return { trainer, trial, deploy };
}

function runScript(code: string | null) {
  if (code !== null) {
    fireEvent.change(screen.getByLabelText("Defence script"), {
      target: { value: code },
    });
  }
  fireEvent.click(screen.getByRole("button", { name: "Run" }));
}

/** Scoped to the output: the starter's own source mentions the same words. */
const shown = (pattern: RegExp) =>
  within(screen.getByRole("region", { name: "Script output" })).queryAllByText(
    pattern,
  ).length > 0;

beforeEach(() => {
  useTowerDefenseStore.getState().restart();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("the code lane", () => {
  it("stops a script that loops on api.trial() forever", async () => {
    // 70 s per fit: twelve fit inside 900 s, the thirteenth ends at 910 s.
    const { trainer, trial } = fakeTrainer(70_000);
    render(<CodeLane trainer={trainer} />);
    runScript("for (;;) { await api.trial(); }");

    await waitFor(
      () => expect(shown(new RegExp(`ran longer than ${MAX_RUN_MS}ms`))).toBe(true),
      { timeout: 10_000 },
    );
    expect(trial).toHaveBeenCalledTimes(Math.floor(MAX_RUN_MS / 70_000) + 1);
  });

  it("lets the starter sweep finish", async () => {
    // Five trials at two minutes each: slow, and still inside the budget.
    const { trainer, trial } = fakeTrainer(120_000);
    render(<CodeLane trainer={trainer} />);
    runScript(null);

    await waitFor(
      () => expect(shown(/best validation accuracy at L2 x/)).toBe(true),
      { timeout: 10_000 },
    );
    expect(trial).toHaveBeenCalledTimes(5);
    expect(shown(/ran longer than/)).toBe(false);
  });

  it("says a failed deployment failed, rather than that it was stopped", async () => {
    const { trainer } = fakeTrainer(1_000, { unscored: "failed" });
    render(<CodeLane trainer={trainer} />);
    runScript("await api.deploy();");

    await waitFor(
      () => expect(shown(/api\.deploy\(\) did not finish: the fit ended with an error/)).toBe(true),
      { timeout: 10_000 },
    );
    expect(shown(/was stopped before/)).toBe(false);
  });
});
