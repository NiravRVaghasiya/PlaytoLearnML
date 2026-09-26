import { act, render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ReactFlowProvider, useStoreApi } from "@xyflow/react";
import type * as XyFlow from "@xyflow/react";
import { RefitOnChange } from "./TreeCanvas";

/**
 * The canvas has pan and zoom switched off, so the fit is the only way to see
 * a gate. React Flow needs a real layout to draw, so this checks the part that
 * decides WHEN to fit: the real provider and store, with only `fitView`
 * replaced by a spy.
 */
const { fitView } = vi.hoisted(() => ({ fitView: vi.fn(async () => true) }));

vi.mock("@xyflow/react", async (importOriginal) => ({
  ...(await importOriginal<typeof XyFlow>()),
  useReactFlow: () => ({ fitView }),
}));

type FlowStore = ReturnType<typeof useStoreApi>;

function mount(signature: string) {
  let store: FlowStore | null = null;
  function Capture() {
    store = useStoreApi();
    return null;
  }
  const view = render(
    <ReactFlowProvider>
      <Capture />
      <RefitOnChange signature={signature} />
    </ReactFlowProvider>,
  );
  const rerender = (next: string) =>
    view.rerender(
      <ReactFlowProvider>
        <Capture />
        <RefitOnChange signature={next} />
      </ReactFlowProvider>,
    );
  return { store: () => store!, rerender };
}

describe("when the tree canvas re-fits", () => {
  beforeEach(() => {
    fitView.mockClear();
  });

  it("fits on mount, and again when the set of nodes changes", () => {
    const { rerender } = mount("n0");
    expect(fitView).toHaveBeenCalledTimes(1);
    expect(fitView).toHaveBeenLastCalledWith(
      expect.objectContaining({ minZoom: 0.02, maxZoom: 1.25, padding: 0.15 }),
    );

    // Same nodes (moving the selection rebuilds node objects, not ids): no refit.
    rerender("n0");
    expect(fitView).toHaveBeenCalledTimes(1);

    rerender("n0,n0L,n0R");
    expect(fitView).toHaveBeenCalledTimes(2);
  });

  it("re-fits when the pane is resized, with the same nodes", () => {
    // A fitted wide tree in a pane that then narrowed used to keep the old
    // transform, and every node past the new right edge stayed clipped.
    const { store } = mount("n0,n0L,n0R");
    act(() => store().setState({ width: 900, height: 300 }));
    const afterFirstMeasure = fitView.mock.calls.length;

    act(() => store().setState({ width: 330 }));
    expect(fitView).toHaveBeenCalledTimes(afterFirstMeasure + 1);

    act(() => store().setState({ height: 420 }));
    expect(fitView).toHaveBeenCalledTimes(afterFirstMeasure + 2);

    // An unrelated store update is not a resize.
    act(() => store().setState({ nodesDraggable: false }));
    expect(fitView).toHaveBeenCalledTimes(afterFirstMeasure + 2);
  });
});
