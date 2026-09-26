# 🎮 GameML

An interactive site that teaches Machine Learning by playing it. Fourteen games
across six ML categories, and **every model trains in the browser** — no ML
server, no GPU, nothing to install to play.

All 14 games are built. There is also a Concept Library of eight short
explainers (`/concepts`); 11 of the 14 games link into it from their feedback
cards. Agent Academy, Confusion Matrix Chef and Dimension Diver have no concept
page yet, so their cards don't link anywhere.

## Prerequisites

| Tool | Version | Why |
|---|---|---|
| [bun](https://bun.sh) | 1.3.13 (`packageManager` in `package.json`) | Package manager and script runner. The lockfile is `bun.lock`. |
| Node.js | 22, at least 22.22.2 (`engines`: `>=22.22.2 <23`) | `next`, `tsc`, `eslint`, `vitest` and every `scripts/*.mjs` run under Node. jsdom 30, the test environment, needs ≥ 22.22.2. |
| Chromium for Playwright | installed once with `bunx playwright install chromium` | Only for the browser harnesses (`verify:play`, `verify:mobile`, `audit:a11y`). The `playwright` package does not download a browser on install. |
| Network access to `cdn.jsdelivr.net` | on the first `dev` or `build` | That run downloads the Pyodide wheels. See [The Python lane](#the-python-lane-and-its-wheels). |

## Running it

```bash
bun install
bun run dev          # http://localhost:3000
bun run build        # production build
bun run start        # serve the production build on :3000
```

**No environment variables are needed**, for development or for a deploy. Every
variable the code reads is optional; they're listed [below](#environment-variables-all-optional).

`predev` and `prebuild` both run `scripts/setup-pyodide.mjs`, which stages the
Python runtime into `public/pyodide/`. That directory is gitignored and rebuilt
on every run.

## The Python lane and its wheels

Feature Forge's code lane is real CPython plus pandas, running in the tab through
[Pyodide](https://pyodide.org). The runtime is self-hosted under `/pyodide/`
rather than fetched from a CDN at play time.

The `pyodide` npm package contains the runtime but **not** the package wheels.
`scripts/setup-pyodide.mjs` handles this:

1. It copies the runtime files (about 13 MB) out of `node_modules/pyodide`.
2. It works out the wheels the lane needs from the pinned `pyodide-lock.json`.
   That is pandas plus its dependencies (numpy, python-dateutil, pytz, six):
   five wheels, about 7.8 MB.
3. It uses a copy that is already staged, or cached in `node_modules/pyodide`,
   if that copy's sha256 matches the lockfile. Otherwise it downloads the wheel
   from `https://cdn.jsdelivr.net/pyodide/v<version>/full/`, at the exact
   version installed. Every wheel is checked against the lockfile's sha256, and
   the browser checks it again when it loads the wheel.
4. It deletes anything else under `public/pyodide/` (for example files left by
   an older version), so a dev server serves exactly what a clean deploy ships.

If a wheel can't be obtained:

- **On CI or Vercel** (`CI` or `VERCEL` set to any non-empty value other than
  `0`, `false` or `no`), the build **fails**. This is on purpose: otherwise the
  deploy would ship a Python lane that fails on `import pandas` for every player.
- **Locally**, it prints a warning and carries on, so you can still work offline
  on everything else.
- `PYODIDE_ALLOW_MISSING=1` turns the CI/Vercel failure back into a warning. Use
  it only for an offline or firewalled build that knowingly ships without the
  Python lane. Never set it on the production deployment.
- Behind an HTTP proxy, Node's `fetch` ignores `HTTPS_PROXY` unless you run
  `node --use-env-proxy scripts/setup-pyodide.mjs`.

`bun run verify:pyodide` (`scripts/pyodide-smoke.mjs`) is a standalone check that
loads Pyodide and pandas in Node. The build doesn't need it.

## Environment variables (all optional)

`NEXT_PUBLIC_*` values and the response headers are fixed when the app is
**built**, so changing one needs a rebuild or redeploy, not just a restart. If
you set any locally, put them in `.env.local`, which is gitignored.

| Variable | Read by | Effect |
|---|---|---|
| `NEXT_PUBLIC_SITE_URL` | `src/lib/site.ts`, `next.config.ts` | The absolute origin used for canonical URLs, `og:url`, `/sitemap.xml` and `/robots.txt`. It is resolved in this order: this variable, then `https://$VERCEL_PROJECT_PRODUCTION_URL` (Vercel sets that on every build, previews included, so a preview's canonical points at production), then `http://localhost:3000`. Set it on any host that isn't Vercel, and on Vercel if you serve a custom domain that isn't the project's production domain. A bare `example.com` is accepted. An invalid value fails the build. A production build that falls through to localhost (neither variable set, so not on Vercel) prints a one-time warning saying so (`siteOriginWarning` in `next.config.ts`). That is harmless on a CI build nobody deploys; before a deploy, set the variable and rebuild. |
| `NEXT_PUBLIC_PYODIDE_INDEX_URL` | `useCodeLane.ts`, `next.config.ts`, `setup-pyodide.mjs` | Where the browser loads Pyodide from. The default is `/pyodide/`, and the value must end in `/`. A path must stay under `/pyodide/`. A versioned path such as `/pyodide/v314.0.5/` must name the installed version, or the build fails; that path is served with `immutable`, one-year caching. An absolute `http(s)` URL means a CDN serves the runtime, and its origin is added to the CSP's `script-src` and `connect-src`. `setup-pyodide.mjs` reads it from the shell **and** from the `.env` files Next loads, through Next's own `@next/env`: the development files for `predev`, the production files for `prebuild`, and for a manual run the development files unless `NODE_ENV=production`. The shell wins, as in Next, so `.env.production` is a safe place for it. The staging log line names the file the value came from. |
| `CSP_REPORT_ONLY=1` | `next.config.ts` | Sends the policy as `Content-Security-Policy-Report-Only`, so violations are logged instead of blocked. For trying a CSP change on a preview. |
| `PYODIDE_ALLOW_MISSING=1` | `setup-pyodide.mjs` | See above. |
| `CI`, `VERCEL` | `setup-pyodide.mjs` | Turn on strict wheel staging. Vercel and GitHub Actions set these themselves. These and `PYODIDE_ALLOW_MISSING` are read from the real environment only, before any `.env` file loads, so a `.env` file can't change the script's strictness. |
| `VERCEL_ENV` | `next.config.ts` | Set by Vercel. On `preview`, the CSP also allows Vercel's preview toolbar (`vercel.live` and friends). Production never gets those sources. |
| `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY` | `src/lib/supabase.ts`, `next.config.ts` | **Have no effect on progress today.** Progression always uses localStorage (see [Known limitations](#known-limitations)). The only visible effect: if the URL is set, its origin and its `wss:` twin are added to the CSP's `connect-src`, and a malformed URL fails the build. |
| `AUDIT_BASE` | the three browser harnesses | Server to test against. Default `http://localhost:3000`. |
| `AUDIT_FORM_FACTORS` | `a11y-audit.mjs` | `desktop`, `mobile`, or both (the default, `desktop,mobile`). |

Never set `SUPABASE_SERVICE_ROLE_KEY` on a deploy. Nothing in the app reads it,
and it bypasses Row Level Security.

## Verifying it

There are four layers, because each catches different things. Unit tests prove
the ML is correct. The playthroughs prove it is wired to the screen. The phone
harness proves the layout holds up on a small screen. Lighthouse proves every
page is reachable with assistive technology.

```bash
bun run verify          # tsc --noEmit && eslint . && vitest run
bun run test:run src/games/data-detox    # one folder or file

bun run build && bun run start           # the three below need a running server
bun run verify:play                      # drives all 14 games in a real browser
bun run verify:play sort-it-arcade       # …or just one
bun run verify:mobile                    # every route at 360×740, touch, 2× density
bun run audit:a11y                       # Lighthouse accessibility, desktop and mobile
```

| Layer | What fails it | Time |
|---|---|---|
| `verify` | Type errors, lint errors (jsx-a11y rules are errors), any unit test | The full vitest suite takes **about 5–11 minutes** (326 s and 659 s in two runs on the same laptop; `vitest.config.mts` records 621 s on a 12-thread machine). TF.js training on the CPU backend is the slow part. The config caps workers at a third of the cores on purpose. |
| `verify:play` | Any game's own checks (`scripts/playthroughs/<slug>.mjs`). Also, across every game: `NaN`, `undefined`, `Infinity` or `null` reaching a live region, a console error or uncaught page error, or a bug-shaped warning (KaTeX, React hydration or key warnings). Headless-GPU noise and TF.js's WebGL-to-CPU fallback are ignored; the rules are in `scripts/harness.mjs`. Every game's Math dialog is opened and checked too (`checkMathDialog` in `harness.mjs`): one "Math" button that reports `aria-expanded`, a named dialog, the equation rendered by KaTeX with MathML, focus moved in, Tab and Shift+Tab trapped (also after a click on the dialog's text), and Escape closing it with focus back on the button. | Several minutes; not benchmarked. The TF.js games really train, and the Feature Forge playthrough loads real pandas. |
| `verify:mobile` | Sideways scrolling, a layout viewport forced wider than 360 px, console errors, the wrong HTTP status, a page with no `<main>` or `<h1>`, and a `/play/*` page whose live metric never appears. Targets under 24×24 px are reported as warnings, not failures. | A few minutes for all 26 routes; not benchmarked. |
| `audit:a11y` | A Lighthouse accessibility score under 95 on any route, in either pass. | 26 routes × 2 passes = 52 Lighthouse runs. Set `AUDIT_FORM_FACTORS` to run one pass. |

`verify:mobile` and `audit:a11y` take their routes from `scripts/harness.mjs`,
which reads the slugs out of `src/games/registry.ts` and `src/lib/concepts.ts`. The
routes are the home page, all 14 games, the Concept Library and its 8 pages, and
two URLs that must return 404. So a new game or concept is covered without
editing a list. `verify:play` runs every file in `scripts/playthroughs/`, and
`registry.test.ts` fails if a playable game has none. The playthroughs and the
phone harness share `harness.mjs`'s rules for which console messages count as
bugs.

What a Lighthouse 100 does **not** prove: Lighthouse only sees each page as it
first loads. It never opens the Math drawer, runs a code lane or triggers a named
failure, and it can't check target size, drag alternatives, reflow, or focus
hidden behind pinned chrome. `verify:mobile` and the playthroughs cover some of
that; the playthroughs check the open Math dialog's behaviour, though not its
contrast. The rest is manual.

`verify:play` exists because Sort-It Arcade once shipped two silent pedagogical
bugs that every unit test passed.

## CI

`.github/workflows/ci.yml` runs on every push to `main`, on every pull request,
and on manual dispatch. It uses bun 1.3.13, Node 22, a read-only token, and
`bun install --frozen-lockfile`. There are three jobs, which run in parallel:

| Job | Runs |
|---|---|
| `static` | `bun run typecheck`, `bun run lint` |
| `unit` | `bun run test:run --shard=N/3`, split across 3 shards |
| `browser` | `bun run build` with `CI=1` (strict wheel staging), `bunx playwright install --with-deps chromium`, and `bun run start`. It then checks that the security headers are served on `/`, that `X-Powered-By` is absent, and that `/pyodide/pyodide.asm.wasm` has its `Cache-Control` policy. Finally it runs `verify:play`, `verify:mobile` and `audit:a11y` against the server. |

Make all three required status checks on `main` (branch protection, or Vercel's
deployment checks), so production only ever deploys from a green commit.

## Deploying to Vercel

No `vercel.json` is needed.

1. **Import the repository.** Vercel detects Next.js. It detects bun from
   `bun.lock`, so the install command is `bun install`.
2. **Build command:** leave it at the default, which runs the `build` script
   (`bun run build`). That runs `prebuild` first, which stages Pyodide in strict
   mode because Vercel sets `VERCEL=1`. If you override it, keep it
   `bun run build`. A bare `next build` skips `prebuild`: `public/pyodide/` is
   gitignored, so the deploy would ship with no Python runtime at all.
3. **Node version** comes from `engines` in `package.json`, which resolves to 22.x.
4. **Environment variables:** none are required. If the site is served from a
   domain other than the project's Vercel production domain, set
   `NEXT_PUBLIC_SITE_URL` so canonical URLs, Open Graph and the sitemap point at
   it.

What the deployment serves, and where it comes from:

- **Security headers** are built in `next.config.ts` (`headers()`) and apply to
  every route: the CSP, `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: strict-origin-when-cross-origin`, a `Permissions-Policy` that
  turns off unused device APIs, `X-Frame-Options: DENY`, and
  `Cross-Origin-Opener-Policy: same-origin`. `poweredByHeader` is off.
  - The CSP allows `'unsafe-eval'`, because the JavaScript code lane runs player
    code through `new AsyncFunction`, and `'wasm-unsafe-eval'` for Pyodide.
  - It allows `'unsafe-inline'` scripts, because every route is statically
    generated and can't carry a nonce.
  - Adding any third-party origin (a CDN, an API, analytics) means changing the
    CSP. `src/lib/security-headers.test.ts` fails if app code fetches or imports
    an absolute URL.
  - `Cross-Origin-Embedder-Policy` is deliberately not set.
- **Caching of `/pyodide/*`:** `public, max-age=86400, stale-while-revalidate=604800`
  at the default path, or a year and `immutable` at a versioned
  `NEXT_PUBLIC_PYODIDE_INDEX_URL`. Without a rule, Vercel would serve these
  roughly 21 MB of files with `max-age=0`.
- **Static routes:** every page is prerendered. `/play/[slug]` and
  `/concepts/[slug]` set `dynamicParams = false`, so an unknown slug is a static
  404, not an on-demand render. `/sitemap.xml` (24 URLs) is generated from the
  registry and the Concept Library, and `/robots.txt` allows everything and
  points at it.

After a deploy, check:

1. The build log has a line like `pyodide 314.0.5: … in public/pyodide`, and no
   `ERROR: could not obtain …`. With a missing wheel the build would have failed.
2. `curl -I https://<deployment>/` shows `content-security-policy`,
   `x-frame-options: DENY` and no `x-powered-by`.
3. `curl -I https://<deployment>/pyodide/pyodide.asm.wasm` shows the
   `cache-control` above. A wheel, for example
   `/pyodide/pandas-3.0.2-cp314-cp314-pyemscripten_2026_0_wasm32.whl`, returns 200.
4. `/robots.txt` and `/sitemap.xml` name the origin you expect.
5. Run the harnesses against the deployment:

   ```bash
   AUDIT_BASE=https://<deployment> bun run verify:play
   AUDIT_BASE=https://<deployment> bun run verify:mobile
   AUDIT_BASE=https://<deployment> bun run audit:a11y
   ```

   The Feature Forge playthrough includes a "real pandas is running" check, which
   is the end-to-end proof that the wheels were deployed. If Vercel Deployment
   Protection is on for previews, the harnesses will only see the login page.
   Test a URL they can reach.
6. On the **first preview** deploy, open the browser console once. The Vercel
   toolbar's CSP sources come from Vercel's documented list and have not been
   tested against a live preview.

## Known limitations

These are stated plainly so nobody discovers them in production.

- **Python runs on the main thread.** The first `import pandas` blocks the page
  for a few seconds. The lane says so ("the page may pause…") and lets that
  message paint first, but the page still freezes. The first Python run also
  downloads about 13 MB of runtime plus about 7.8 MB of wheels. A Web Worker
  runtime would fix both, and is not built.
- **Python runs can't be interrupted.** `maxRunMs` and `checkBudget()` don't
  apply to Python, so a `while True:` hangs the tab. Stopping it properly needs
  the runtime in a Worker plus Pyodide's interrupt buffer, a `SharedArrayBuffer`
  that needs cross-origin isolation (COEP). None of that is enabled.
- **JavaScript runs are only cooperatively interruptible.** The time budget can
  only throw when the snippet calls `checkBudget()`, or calls into the api of a
  lane that turns on `budgetApiCalls` (Neuron Forge and Overfit Tower Defense
  do). A bare `while (true) {}` hangs the tab.
- **Code-lane edits last only as long as the page.** A lane switch, or leaving
  a game and coming back, restores the edited snippet and its last output. They
  are kept in memory, never in storage, so a reload or a new tab starts from the
  starter snippet again.
- **The code lane is not a security sandbox.** Player code runs with the page's
  full privileges, the same trust model as a browser console. Never run code
  that came from another user or a URL. A share-a-snippet feature would first
  need a Worker or iframe isolate.
- **Progress is per browser.** It is kept in `localStorage`. Tabs merge each
  other's progress, and a failed save shows "Progress isn't being saved on this
  device" in the game footer. There are no accounts, and nothing syncs across
  devices. `createSupabaseAdapter` exists and is tested, but nothing calls it:
  there is no sign-in flow to supply a user id.
- **No lock/unlock states, leaderboards, My Lab, Classrooms or Sandbox**
  (spec §6). Every game is open. A game shows its own concept badge in its
  footer once earned, but badges can't be shared.
- **Without WebGL**, TF.js falls back to its CPU backend. The games still work,
  but the heavy ones get slow; Convolution Kitchen's learn step can take minutes.
  The two Three.js views fall back as well: Gradient Descent Skier switches to
  its contour map, and Dimension Diver's 3D panel says it's unavailable while the
  2D shadow still works.
- **No social image.** Link previews use Twitter's `summary` card (title and
  description only).
- **Dev-only advisories.** `bun audit` reports three, all reached through
  `lighthouse`, which only `scripts/a11y-audit.mjs` uses and which is never
  bundled: `@opentelemetry/core` < 2.8.0 (moderate) and `extract-zip` ≤ 2.0.1
  (two high).

## The four load-bearing ideas

1. **Pedagogy contract** — every game has a core intuition, player-action =
   algorithm, always-visible live feedback, and a *named* failure mode.
2. **Two-lane rule** — every game ships a visual (no-code) lane and a code lane
   sharing one state store.
3. **Client-side ML only** — TensorFlow.js and Pyodide in-browser. No inference
   endpoint, which is what makes hosting cheap.
4. **Shared engine** — `GameShell`, `useModel`, `useCodeLane` and `progression`
   are built once and reused, never forked per game.

## Layout

```
├── CLAUDE.md                     ← how we build: rules, guardrails, DoD
├── DESIGN.md                     ← how it looks: tokens, shell anatomy, a11y
├── docs/
│   ├── GameML_Build_Spec.md      ← what we build: the 14 games and their models
│   └── ENGINE_API.md             ← shared engine API reference
├── .github/workflows/ci.yml      ← typecheck, lint, sharded tests, browser harnesses
├── src/
│   ├── app/                      ← routes: /, /play/[slug], /concepts, /concepts/[slug],
│   │                               404 + error boundaries, sitemap, robots, icons
│   ├── engine/                   ← SHARED: GameShell, useModel, useCodeLane, progression
│   ├── components/               ← shared primitives (MetricReadout, WhyCard, …)
│   ├── games/<slug>/             ← one self-contained folder per game
│   │                               store · ml · VisualLane · CodeLane · why-cards · tests
│   └── lib/                      ← catalog, concepts, site metadata, utils, supabase seam
├── scripts/                      ← Pyodide staging, playthroughs, phone harness, a11y audit
└── .claude/                      ← agents, skills and slash commands for adding a game
```

## Docs, in reading order

1. **`CLAUDE.md`** — how we build. The pedagogy contract, the golden rules and
   the Definition of Done. Cited throughout the code.
2. **`DESIGN.md`** — how it looks. Colour and type tokens, `GameShell` anatomy,
   and the accessibility requirements. The most-cited document in the codebase;
   code comments reference its sections by number.
3. **`docs/GameML_Build_Spec.md`** — what each game teaches. `src/lib/catalog.ts`
   is a transcription of it, and the spec wins if they disagree. Its §9 lists the
   places where the build deliberately differs.
4. **`docs/ENGINE_API.md`** — the shared engine's API, for building a game or
   extending the engine.

If they conflict: CLAUDE = how, DESIGN = look, SPEC = what. Ask rather than guess.

## Adding a game

The spec has one unbuilt stretch entry, GAN Duel. `.claude/` holds the tooling
for building it: the `new-game-module` skill, the `game-builder`, `ml-verifier`
and `design-reviewer` subagents, and the `/new-game` and `/verify-game` commands.

A game is invisible until it is registered. `CLAUDE.md` lists every place, and
`src/games/registry.test.ts` fails if a `src/games/<slug>/index.tsx` is
unregistered or has no playthrough. A game is not done until all four
pedagogy-contract points are demonstrable in code and the Definition of Done in
`CLAUDE.md` passes.

## Secrets and tooling

- Secrets never get committed. `.env`, `.env.local` and `.env.*.local` are
  gitignored. The app itself needs no secrets.
- MCP servers for agent tooling are configured in `.mcp.json`. The `github` and
  `supabase` servers are gated behind "ask" in `.claude/settings.json`. As of
  this writing, `.mcp.json`'s `git` server points at
  `@modelcontextprotocol/server-git`, which isn't published on npm, so that
  server can't start. The allowlist in `.claude/settings.json` also still names
  `npm`/`npx` rather than `bun`.

## Provenance

The 14-game catalog came out of an ideation prompt, which produced
`docs/GameML_Build_Spec.md`, which in turn produced this kit's `CLAUDE.md`,
`DESIGN.md` and `.claude/` tooling. The original ideation and kickoff prompts
were removed once the build was complete — they described work that is now done,
and are recoverable from git history if the provenance is ever needed.
