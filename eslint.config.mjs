import coreWebVitals from "eslint-config-next/core-web-vitals";
import nextTypescript from "eslint-config-next/typescript";

/**
 * GameML lint rules.
 *
 * Beyond the Next.js defaults we harden two things the project treats as
 * non-negotiable:
 *   1. Accessibility (DESIGN.md §9) — jsx-a11y rules are errors, not warnings.
 *   2. The "no ML server" guardrail (CLAUDE.md) — flagged in review, not lint,
 *      but `no-restricted-syntax` below catches the obvious `fetch('/api/...')`
 *      shape inside game modules.
 *
 * @type {import('eslint').Linter.Config[]}
 */
const config = [
  {
    ignores: [
      ".next/**",
      "out/**",
      "build/**",
      "node_modules/**",
      "coverage/**",
      // Vendored Pyodide runtime, copied in by scripts/setup-pyodide.mjs. It is
      // 20 MB of emscripten glue that we neither wrote nor can fix, and linting it
      // buries our own diagnostics under hundreds of upstream warnings.
      "public/pyodide/**",
      "next-env.d.ts",
    ],
  },

  ...coreWebVitals,
  ...nextTypescript,

  {
    rules: {
      // --- Accessibility is a hard requirement, not a suggestion ------------
      "jsx-a11y/alt-text": "error",
      "jsx-a11y/aria-props": "error",
      "jsx-a11y/aria-proptypes": "error",
      "jsx-a11y/aria-role": "error",
      "jsx-a11y/aria-unsupported-elements": "error",
      "jsx-a11y/role-has-required-aria-props": "error",
      "jsx-a11y/role-supports-aria-props": "error",
      "jsx-a11y/no-redundant-roles": "error",
      "jsx-a11y/label-has-associated-control": "error",
      "jsx-a11y/no-noninteractive-element-interactions": "error",
      "jsx-a11y/interactive-supports-focus": "error",
      "jsx-a11y/click-events-have-key-events": "error",

      // --- General hygiene --------------------------------------------------
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
      "@typescript-eslint/consistent-type-imports": [
        "error",
        { prefer: "type-imports", fixStyle: "inline-type-imports" },
      ],
    },
  },

  {
    // Games must never reach a server for ML. Catch the obvious shapes.
    files: ["src/games/**/*.{ts,tsx}", "src/engine/**/*.{ts,tsx}"],
    rules: {
      "no-restricted-syntax": [
        "error",
        {
          selector:
            "CallExpression[callee.name='fetch'] > Literal.arguments[value=/^\\/api\\//]",
          message:
            "CLAUDE.md guardrail: ML must run client-side. No /api/ inference calls from games or the engine.",
        },
      ],
    },
  },
];

export default config;
