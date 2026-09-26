import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { Dial } from "./Dial";
import { Slider, toPosition } from "./Slider";

/** A controlled Dial, the way the Skier uses it (log scale, 0.001 → 1.5). */
function LearningRateDial({ initial = 0.05 }: { initial?: number }) {
  const [value, setValue] = useState(initial);
  return (
    <Dial
      label="Learning rate"
      value={value}
      min={0.001}
      max={1.5}
      scale="log"
      onChange={setValue}
      format={(v) => v.toPrecision(2)}
    />
  );
}

describe("<Dial> single-pointer alternative (WCAG 2.2 SC 2.5.7)", () => {
  it("steps the value with a tap on + and −, no drag needed", async () => {
    const user = userEvent.setup();
    render(<LearningRateDial />);
    const dial = screen.getByRole("slider", { name: "Learning rate" });
    const start = Number(dial.getAttribute("aria-valuenow"));

    await user.click(screen.getByRole("button", { name: "Increase Learning rate" }));
    const up = Number(dial.getAttribute("aria-valuenow"));
    expect(up).toBeGreaterThan(start);

    // One tap is exactly one arrow-key step (2% of the log track).
    const moved =
      toPosition(up, 0.001, 1.5, "log") - toPosition(start, 0.001, 1.5, "log");
    expect(moved).toBeCloseTo(0.02, 3);

    await user.click(screen.getByRole("button", { name: "Decrease Learning rate" }));
    await user.click(screen.getByRole("button", { name: "Decrease Learning rate" }));
    expect(Number(dial.getAttribute("aria-valuenow"))).toBeLessThan(start);
  });

  it("gives the step buttons at least a 24px target (they are 44px)", () => {
    render(<LearningRateDial />);
    for (const name of ["Increase Learning rate", "Decrease Learning rate"]) {
      expect(screen.getByRole("button", { name }).className).toContain("size-11");
    }
  });

  it("clamps at the ends instead of wrapping or overshooting", async () => {
    const user = userEvent.setup();
    render(<LearningRateDial initial={1.5} />);
    await user.click(screen.getByRole("button", { name: "Increase Learning rate" }));
    expect(
      Number(screen.getByRole("slider", { name: "Learning rate" }).getAttribute("aria-valuenow")),
    ).toBeCloseTo(1.5, 6);
  });

  it("announces a stepped value, since focus stays on the button", async () => {
    const user = userEvent.setup();
    const { container } = render(<LearningRateDial />);
    const live = () => container.querySelector('[aria-live="polite"]');
    expect(live()).toBeEmptyDOMElement();

    await user.click(screen.getByRole("button", { name: "Increase Learning rate" }));
    expect(live()?.textContent).toBe(
      screen.getByRole("slider", { name: "Learning rate" }).getAttribute("aria-valuetext"),
    );

    // Arrow keys on the dial are announced by the dial itself; the extra
    // region steps aside so nothing is said twice.
    screen.getByRole("slider", { name: "Learning rate" }).focus();
    await user.keyboard("{ArrowUp}");
    expect(live()).toBeEmptyDOMElement();
  });

  it("does not read out a later change that no tap made", async () => {
    // The Skier's code lane calls api.setLearningRate(), and a reset moves the
    // dial too. After one tap, those used to be announced out of context.
    function Harness() {
      const [value, setValue] = useState(0.05);
      return (
        <>
          <Dial
            label="Learning rate"
            value={value}
            min={0.001}
            max={1.5}
            scale="log"
            onChange={setValue}
            format={(v) => v.toPrecision(2)}
          />
          <button type="button" onClick={() => setValue(0.5)}>
            Code lane sets 0.5
          </button>
        </>
      );
    }
    const user = userEvent.setup();
    const { container } = render(<Harness />);
    const live = () => container.querySelector('[aria-live="polite"]');

    await user.click(screen.getByRole("button", { name: "Increase Learning rate" }));
    expect(live()).not.toBeEmptyDOMElement();

    await user.click(screen.getByRole("button", { name: "Code lane sets 0.5" }));
    expect(screen.getByRole("slider", { name: "Learning rate" })).toHaveAttribute(
      "aria-valuetext",
      "0.50",
    );
    expect(live()).toBeEmptyDOMElement();
  });

  it("still announces a tap when the game rounds the value it stores", async () => {
    function Rounding() {
      const [value, setValue] = useState(0.05);
      return (
        <Dial
          label="Learning rate"
          value={value}
          min={0.001}
          max={1.5}
          scale="log"
          onChange={(v) => setValue(Number(v.toFixed(3)))}
          format={(v) => v.toFixed(3)}
        />
      );
    }
    const user = userEvent.setup();
    const { container } = render(<Rounding />);
    await user.click(screen.getByRole("button", { name: "Increase Learning rate" }));
    expect(container.querySelector('[aria-live="polite"]')?.textContent).toBe(
      screen.getByRole("slider", { name: "Learning rate" }).getAttribute("aria-valuetext"),
    );
  });
});

describe("value readouts are not a second live region", () => {
  it("hides the Dial's <output> from assistive tech", () => {
    const { container } = render(<LearningRateDial />);
    expect(container.querySelector("output")).toHaveAttribute("aria-hidden", "true");
  });

  it("hides the Slider's <output> from assistive tech", () => {
    const { container } = render(
      <Slider label="Momentum" value={0.5} min={0} max={0.95} onChange={() => {}} />,
    );
    // <output> is an implicit role="status": exposed, it re-spoke every change
    // that the range input's aria-valuetext had already announced.
    expect(container.querySelector("output")).toHaveAttribute("aria-hidden", "true");
    expect(screen.getByRole("slider", { name: "Momentum" })).toHaveAttribute(
      "aria-valuetext",
      "0.500",
    );
  });
});
