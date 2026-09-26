import { act, render, within } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { STEPS } from "./ml";
import { useBlitzStore } from "./store";
import { VisualLane } from "./VisualLane";

/**
 * What the visual lane draws for a choice whose gradient has nowhere to come
 * from yet. Routing out of order is allowed — clicking any node jumps to it —
 * and a rule chosen at a node whose own gradient has not arrived sends nothing.
 * The view used to fly a red "0.0" packet along that edge and colour the
 * inspector's empty cell as wrong: a number nobody computed, marked as a
 * mistake the player did not make.
 */

const routeStep = (nodeId: string) =>
  STEPS.findIndex((step) => step.kind === "route" && step.nodeId === nodeId);
const correctRule = (index: number) =>
  STEPS[index]!.options.find((option) => option.correct)!.id;

/** The packet is the only r=13 disc in the graph; the nodes are larger. */
const packets = (root: HTMLElement) => root.querySelectorAll('circle[r="13"]');

function choose(index: number) {
  act(() => {
    useBlitzStore.getState().focusStep(index);
    useBlitzStore.getState().choose(index, correctRule(index));
  });
}

describe("the visual lane, for a choice that sends nothing", () => {
  beforeEach(() => {
    useBlitzStore.setState({ clearedIds: [], peekedIds: [], scenarioIndex: 0 });
    useBlitzStore.getState().reset();
  });

  it("flies no packet and marks nothing wrong", () => {
    const { container } = render(<VisualLane />);
    const z = routeStep("z");
    choose(z);

    expect(packets(container)).toHaveLength(0);
    const panel = within(container).getByText("What that sends onward").parentElement!;
    // Skip the header row; each body row is a row header, then Yours, then Autograd.
    const rows = within(panel).getAllByRole("row").slice(1);
    expect(rows.length).toBe(2);
    for (const row of rows) {
      const [yours] = within(row).getAllByRole("cell");
      expect(yours).toHaveTextContent("nothing sent yet");
      expect(yours!.className).toContain("text-text-muted");
      expect(yours!.className).not.toContain("text-wrong");
    }
  });

  it("still flies the packet for a gradient that was actually sent", () => {
    const { container } = render(<VisualLane />);
    choose(0);
    expect(packets(container)).toHaveLength(1);
    expect(packets(container)[0]!.getAttribute("stroke")).toBe("var(--correct)");
  });
});
