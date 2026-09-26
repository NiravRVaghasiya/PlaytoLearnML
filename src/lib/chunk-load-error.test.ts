import { describe, expect, it } from "vitest";
import { isChunkLoadError } from "./chunk-load-error";

describe("isChunkLoadError", () => {
  it("matches every browser's and bundler's wording for a failed chunk", () => {
    const chunkLoad = Object.assign(new Error("Loading chunk 42 failed."), {
      name: "ChunkLoadError",
    });
    expect(isChunkLoadError(chunkLoad)).toBe(true);
    for (const message of [
      "Loading chunk 7 failed.\n(error: https://example.com/_next/static/chunks/7.js)",
      "Failed to load chunk /_next/static/chunks/abc.js",
      "Loading CSS chunk 3 failed.",
      "Failed to fetch dynamically imported module: https://example.com/x.js",
      "Importing a module script failed.",
      "error loading dynamically imported module",
    ]) {
      expect(isChunkLoadError(new Error(message)), message).toBe(true);
    }
  });

  it("does not mistake an ordinary game bug for a download failure", () => {
    expect(isChunkLoadError(new Error("Cannot read properties of undefined"))).toBe(
      false,
    );
    expect(isChunkLoadError(new Error("WebGL is not supported"))).toBe(false);
    expect(isChunkLoadError("Loading chunk 1 failed")).toBe(false); // not an Error
    expect(isChunkLoadError(null)).toBe(false);
  });
});
