import { availableParallelism } from "node:os";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    // Vite 8 resolves the `@/*` alias from tsconfig.json natively.
    tsconfigPaths: true,
  },
  // Vite 8 (oxc) transforms .tsx with the automatic React 19 JSX runtime out of
  // the box, so no @vitejs/plugin-react is needed here — we don't want
  // react-refresh/HMR inside a test runner anyway.
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
    // TF.js training in tests is slow on the CPU backend; give it room.
    testTimeout: 30_000,
    hookTimeout: 30_000,
    // Vitest defaults to one worker per logical core, which is the wrong shape
    // for this suite: TF.js's CPU backend is itself multi-threaded, so a dozen
    // forks each training a model oversubscribe the machine several times over
    // and every heavy test slows down together.
    //
    // Measured on a 12-thread laptop, whole suite: the default took 954s and
    // pushed convolution-kitchen's slowest case past its 300s timeout, while
    // capping workers at 4 took 621s with all 1002 tests passing. Fewer workers
    // is both faster and stable here.
    //
    // Expressed as a ratio rather than a constant so a larger CI box still gets
    // more workers, at the same one-third-of-cores oversubscription budget.
    maxWorkers: Math.max(2, Math.floor(availableParallelism() / 3)),
  },
});
