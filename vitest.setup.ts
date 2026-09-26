import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { forgetCodeLaneDraftsForTests } from "@/engine/useCodeLane";

// Code-lane drafts outlive the lane on purpose (a lane switch must not lose
// the player's edits). Between tests that is a leak: a snippet edited in one
// test would greet the next as its "starter". Same reason RTL unmounts.
afterEach(() => forgetCodeLaneDraftsForTests());

// jsdom does not implement matchMedia, which DESIGN.md §7 requires us to read
// for `prefers-reduced-motion`. Provide a spec-shaped stub so components that
// respect reduced motion can be unit-tested.
if (typeof window !== "undefined" && !window.matchMedia) {
  window.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia;
}

// jsdom lacks ResizeObserver; D3/Three canvases measure their container.
if (typeof globalThis !== "undefined" && !("ResizeObserver" in globalThis)) {
  class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver =
    ResizeObserverStub;
}

// jsdom has no canvas backend. Its `getContext()` logs "Not implemented:
// HTMLCanvasElement's getContext() method" to stderr and then returns nothing,
// and TF.js calls it on every backend init while probing for WebGL — so every
// TF test file printed that line several times.
//
// Returning `null` is exactly what a browser without WebGL does, and what jsdom
// already returned after the warning. So nothing changes about which backend TF
// ends up on (it still falls back to "cpu", as it always did here) or what the
// games' own `if (!context) return;` guards see; the noise just stops hiding
// real warnings. Tests that need a context stub `getContext` themselves.
if (typeof HTMLCanvasElement !== "undefined") {
  Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
    configurable: true,
    writable: true,
    value: function getContext() {
      return null;
    },
  });
}
