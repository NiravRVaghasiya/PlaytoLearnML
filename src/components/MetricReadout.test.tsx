import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  MetricReadout,
  formatMetric,
  metricAnnouncement,
} from "./MetricReadout";

describe("formatMetric", () => {
  it("formats percentages", () => {
    expect(formatMetric(0.8421, "percent")).toBe("84%");
    expect(formatMetric(0.8421, "percent", 1)).toBe("84.2%");
  });

  it("formats decimals with three places by default", () => {
    expect(formatMetric(1.23456, "decimal")).toBe("1.235");
    expect(formatMetric(1.23456, "decimal", 1)).toBe("1.2");
  });

  it("formats integers with thousands separators", () => {
    expect(formatMetric(12345.6, "integer")).toBe("12,346");
  });

  it("renders a dash for non-finite values rather than NaN", () => {
    expect(formatMetric(Number.NaN, "decimal")).toBe("—");
    expect(formatMetric(Number.POSITIVE_INFINITY, "percent")).toBe("—");
  });
});

describe("metricAnnouncement", () => {
  it("speaks percent as a word so screen readers say it correctly", () => {
    expect(metricAnnouncement("Accuracy", 0.84, null, "percent")).toBe(
      "Accuracy 84 percent",
    );
  });

  it("names the direction of travel", () => {
    expect(metricAnnouncement("Accuracy", 0.84, 0.6, "percent")).toBe(
      "Accuracy increased to 84 percent",
    );
    expect(metricAnnouncement("Loss", 0.2, 0.9, "decimal")).toBe(
      "Loss decreased to 0.200",
    );
  });

  it("omits a direction when nothing changed", () => {
    expect(metricAnnouncement("Inertia", 5, 5, "decimal")).toBe(
      "Inertia 5.000",
    );
  });
});

describe("<MetricReadout>", () => {
  it("shows the label and the formatted value", () => {
    render(<MetricReadout label="Accuracy" value={0.84} format="percent" />);

    expect(screen.getByText("Accuracy")).toBeInTheDocument();
    expect(screen.getByTestId("metric-value")).toHaveTextContent("84%");
  });

  it("hides the tweening number from assistive tech", () => {
    // Otherwise <output>'s implicit role="status" would announce every frame of
    // the count-up animation.
    render(<MetricReadout label="Accuracy" value={0.84} format="percent" />);
    expect(screen.getByTestId("metric-value")).toHaveAttribute(
      "aria-hidden",
      "true",
    );
  });

  it("exposes a polite live region for the metric", async () => {
    render(<MetricReadout label="Accuracy" value={0.84} format="percent" />);

    // The announcement is debounced so a training run doesn't spam the reader.
    await waitFor(
      () =>
        expect(
          document.querySelector('[aria-live="polite"]'),
        ).toHaveTextContent("Accuracy 84 percent"),
      { timeout: 3000 },
    );
  });

  it("announces the direction after a change", async () => {
    const { rerender } = render(
      <MetricReadout label="Accuracy" value={0.6} format="percent" />,
    );

    await waitFor(
      () =>
        expect(
          document.querySelector('[aria-live="polite"]'),
        ).toHaveTextContent("Accuracy 60 percent"),
      { timeout: 3000 },
    );

    rerender(<MetricReadout label="Accuracy" value={0.84} format="percent" />);

    await waitFor(
      () =>
        expect(
          document.querySelector('[aria-live="polite"]'),
        ).toHaveTextContent("Accuracy increased to 84 percent"),
      { timeout: 3000 },
    );
  });

  it("can suppress announcements for secondary metrics", () => {
    render(
      <MetricReadout
        label="Complexity"
        value={3}
        format="integer"
        announce={false}
      />,
    );

    expect(document.querySelector('[aria-live="polite"]')).toBeNull();
  });

  it("pairs direction with a glyph, not colour alone", () => {
    const { rerender, container } = render(
      <MetricReadout label="Loss" value={0.9} goodDirection="down" />,
    );

    rerender(<MetricReadout label="Loss" value={0.4} goodDirection="down" />);

    // DESIGN.md §9: never encode meaning in colour alone. A falling metric shows
    // a down glyph regardless of palette.
    expect(container.textContent).toContain("▼");
  });

  it("colours a forced failure state red", () => {
    render(<MetricReadout label="Accuracy" value={0.61} state="bad" />);
    expect(screen.getByTestId("metric-value").className).toContain(
      "text-wrong",
    );
  });

  it("renders a caption when given one", () => {
    render(
      <MetricReadout label="Accuracy" value={0.7} caption="on held-out set" />,
    );
    expect(screen.getByText("on held-out set")).toBeInTheDocument();
  });
});
