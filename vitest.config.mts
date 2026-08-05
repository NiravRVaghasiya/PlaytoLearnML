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
  },
});
