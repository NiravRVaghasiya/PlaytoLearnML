import type { NextConfig } from "next";
import pyodidePackage from "pyodide/package.json";

/** What `headers()` returns: one `{ source, headers }` rule per path pattern. */
type HeaderRules = Awaited<ReturnType<NonNullable<NextConfig["headers"]>>>;
type HeaderPair = HeaderRules[number]["headers"][number];

/**
 * The only environment the headers depend on. Taken as a parameter rather than
 * read from `process.env` inside, so the policy is a pure function a unit test
 * can pin down (src/lib/security-headers.test.ts).
 *
 * Note that `headers()` runs at BUILD time: the values are compiled into the
 * routes manifest, and on Vercel into the deployment's routing config. Changing
 * an env var therefore needs a rebuild, not just a restart.
 */
export interface HeaderEnv {
  /** Supabase project URL. Unset today — nothing calls the Supabase adapter yet. */
  NEXT_PUBLIC_SUPABASE_URL?: string;
  /**
   * Where the browser loads Pyodide from (src/engine/useCodeLane.ts reads the
   * same variable; unset means `/pyodide/`). See `pyodideLocation` below.
   */
  NEXT_PUBLIC_PYODIDE_INDEX_URL?: string;
  /** Set by Vercel: "production" | "preview" | "development". */
  VERCEL_ENV?: string;
  /**
   * "1" ships the policy as Content-Security-Policy-Report-Only: violations are
   * logged to the console instead of blocked. An escape hatch for trying a CSP
   * change on a preview deploy without a code change — not the default.
   */
  CSP_REPORT_ONLY?: string;
}

/** The installed Pyodide — the exact version scripts/setup-pyodide.mjs stages. */
export const PYODIDE_VERSION: string = pyodidePackage.version;

/** Where the runtime lives when nothing moves it. Mirrors useCodeLane.ts. */
export const PYODIDE_DEFAULT_PATH = "/pyodide/";

/**
 * Where the Python lane's runtime is served from, validated.
 *
 * - A path (the default, `/pyodide/`) is self-hosted: setup-pyodide.mjs copies
 *   the runtime there, applying these same rules. It must stay under /pyodide/
 *   so .gitignore and the eslint ignore keep covering it, and may name a
 *   version segment (`/pyodide/v314.0.5/`) — which then must be the installed
 *   version, or the build fails. That check is what makes `immutable` caching
 *   of a versioned path safe: an upgrade can't reuse an old version's URL.
 * - An absolute URL (a CDN) is third-party: its origin joins script-src and
 *   connect-src, and no cache rule applies — it isn't our server.
 *
 * Anything else throws. A bad value would otherwise ship a Python lane that 404s
 * or is blocked by this very CSP, visible only in a player's console.
 */
export type PyodideLocation =
  | { kind: "path"; path: string; versioned: boolean }
  | { kind: "origin"; origin: string };

export function pyodideLocation(
  raw: string | undefined,
  version: string = PYODIDE_VERSION,
): PyodideLocation {
  const name = "NEXT_PUBLIC_PYODIDE_INDEX_URL";
  // `||`, like the engine: an empty value means the default there too.
  const value = raw || PYODIDE_DEFAULT_PATH;
  if (!value.endsWith("/")) {
    throw new Error(`${name} must end in "/" (got "${value}").`);
  }

  if (value.startsWith("/") && !value.startsWith("//")) {
    if (!/^\/pyodide\/(?:[A-Za-z0-9_.-]+\/)*$/.test(value)) {
      throw new Error(
        `${name} must be a path under ${PYODIDE_DEFAULT_PATH} (got "${value}").`,
      );
    }
    const segments = value.split("/").filter(Boolean);
    if (segments.some((segment) => segment === "." || segment === "..")) {
      throw new Error(`${name} must not contain "." or ".." (got "${value}").`);
    }
    const named = segments.filter((segment) => /^v\d+(?:\.\d+)*$/.test(segment));
    if (named.some((segment) => segment !== `v${version}`)) {
      throw new Error(
        `${name} names ${named.join(", ")} but pyodide ${version} is installed ` +
          `(got "${value}"). Update it to v${version}.`,
      );
    }
    return { kind: "path", path: value, versioned: named.length > 0 };
  }

  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new Error(
      `${name} must be a path like ${PYODIDE_DEFAULT_PATH} or an absolute ` +
        `http(s) URL (got "${value}").`,
    );
  }
  if (url.protocol !== "https:" && url.protocol !== "http:") {
    throw new Error(`${name} must be an http(s) URL (got "${url.protocol}").`);
  }
  return { kind: "origin", origin: url.origin };
}

/**
 * Supabase origins for connect-src: the REST/auth origin plus its WebSocket twin
 * (Realtime). Origin only — a CSP source with a path would pin every request to
 * that path.
 *
 * A malformed URL throws rather than being skipped. Skipping would produce a
 * policy that silently blocks Supabase the day it is wired up, and the build
 * error names the variable to fix.
 */
function supabaseSources(raw: string | undefined): string[] {
  const value = raw?.trim();
  if (!value) return [];
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new Error(
      `NEXT_PUBLIC_SUPABASE_URL is not a valid URL (got "${value}"). ` +
        `Expected something like https://<project>.supabase.co`,
    );
  }
  if (url.protocol !== "https:" && url.protocol !== "http:") {
    throw new Error(
      `NEXT_PUBLIC_SUPABASE_URL must be an http(s) URL (got "${url.protocol}").`,
    );
  }
  const socket = url.protocol === "https:" ? "wss:" : "ws:";
  return [url.origin, `${socket}//${url.host}`];
}

/**
 * Vercel's preview toolbar (comments, share, feedback) is injected into preview
 * deployments only, from vercel.live. These are the sources Vercel documents for
 * running the toolbar under a CSP. Production never gets them.
 */
const VERCEL_TOOLBAR = {
  script: ["https://vercel.live"],
  style: ["https://vercel.live"],
  img: ["https://vercel.live", "https://vercel.com"],
  font: ["https://vercel.live", "https://assets.vercel.com"],
  connect: ["https://vercel.live", "wss://ws-us3.pusher.com"],
  frame: ["https://vercel.live"],
} as const;

/**
 * The Content-Security-Policy, as one header value.
 *
 * Every relaxation below is load-bearing — each one names what breaks without it:
 *
 * - script-src 'unsafe-inline': every route is statically generated, and Next's
 *   inline flight/hydration scripts on a static page cannot carry a nonce (a
 *   nonce needs a per-request render, which would make every route dynamic).
 * - script-src 'unsafe-eval': the code lane is the product. The JS lane runs the
 *   player's snippet through `new AsyncFunction(...)` (src/engine/useCodeLane.ts),
 *   which without it throws an EvalError before any player code runs. React's
 *   dev build also uses eval for stack traces.
 * - script-src 'wasm-unsafe-eval': Pyodide compiles pyodide.asm.wasm with
 *   `WebAssembly.instantiateStreaming`; with neither eval source it never
 *   finishes booting. 'unsafe-eval' happens to imply it today, but naming it
 *   keeps the Python lane working if the JS lane ever stops needing eval.
 *   (pyodide.asm.mjs does contain `eval` paths — EM_ASM, emscripten_run_script
 *   — but booting it and loading pandas under a policy with only
 *   'wasm-unsafe-eval' was checked in Chromium and works.) Everything else
 *   Pyodide needs is same-origin: it is self-hosted under /pyodide/
 *   (scripts/setup-pyodide.mjs), loaded by a dynamic `import()`, and fetches its
 *   stdlib and wheels from the same directory, so no CDN appears here — unless
 *   NEXT_PUBLIC_PYODIDE_INDEX_URL points at one, in which case exactly that
 *   origin is added to script-src and connect-src.
 * - style-src 'unsafe-inline': KaTeX's rendered HTML, React Flow and every
 *   `style={...}` prop are inline styles.
 * - img-src / font-src data: blob: — SVG/canvas data URLs and blob textures;
 *   next/font self-hosts the fonts, so font-src needs no third party.
 *
 * What stays locked: no wildcard or scheme-wide (https:) script source, no
 * plugins (object-src 'none'), no <base> hijack, forms post only to us, and the
 * site cannot be framed (frame-ancestors 'none'; X-Frame-Options below covers
 * old browsers).
 *
 * Deliberately NOT here: `upgrade-insecure-requests` — Vercel already serves
 * HTTPS with HSTS, and on a plain-http `next start` (LAN testing) it would
 * rewrite every asset URL to https and break the page.
 */
export function contentSecurityPolicy(env: HeaderEnv = {}): string {
  const preview = env.VERCEL_ENV === "preview";
  const toolbar = (sources: readonly string[]) => (preview ? sources : []);
  const pyodide = pyodideLocation(env.NEXT_PUBLIC_PYODIDE_INDEX_URL);
  const pyodideCdn = pyodide.kind === "origin" ? [pyodide.origin] : [];

  const directives: Record<string, readonly string[]> = {
    "default-src": ["'self'"],
    "script-src": [
      "'self'",
      "'unsafe-inline'",
      "'unsafe-eval'",
      "'wasm-unsafe-eval'",
      ...pyodideCdn,
      ...toolbar(VERCEL_TOOLBAR.script),
    ],
    "style-src": ["'self'", "'unsafe-inline'", ...toolbar(VERCEL_TOOLBAR.style)],
    "img-src": ["'self'", "data:", "blob:", ...toolbar(VERCEL_TOOLBAR.img)],
    "font-src": ["'self'", "data:", ...toolbar(VERCEL_TOOLBAR.font)],
    "connect-src": [
      "'self'",
      ...pyodideCdn,
      ...supabaseSources(env.NEXT_PUBLIC_SUPABASE_URL),
      ...toolbar(VERCEL_TOOLBAR.connect),
    ],
    "worker-src": ["'self'", "blob:"],
    "frame-src": ["'self'", ...toolbar(VERCEL_TOOLBAR.frame)],
    "object-src": ["'none'"],
    "base-uri": ["'self'"],
    "form-action": ["'self'"],
    "frame-ancestors": ["'none'"],
  };

  return Object.entries(directives)
    .map(([name, sources]) => `${name} ${sources.join(" ")}`)
    .join("; ");
}

/**
 * Powerful features the app never uses, switched off so an injected script can't
 * use them either. Only names Chromium recognises: an unknown feature name makes
 * it log "Error with Permissions-Policy header: Unrecognized feature" on every
 * page load. `fullscreen` is left alone on purpose.
 */
const PERMISSIONS_POLICY = [
  "camera",
  "microphone",
  "geolocation",
  "payment",
  "usb",
  "serial",
  "hid",
  "midi",
  "display-capture",
  "accelerometer",
  "gyroscope",
  "magnetometer",
]
  .map((feature) => `${feature}=()`)
  .join(", ");

/** Response headers for every route. */
export function securityHeaders(env: HeaderEnv = {}): HeaderPair[] {
  return [
    {
      key:
        env.CSP_REPORT_ONLY === "1"
          ? "Content-Security-Policy-Report-Only"
          : "Content-Security-Policy",
      value: contentSecurityPolicy(env),
    },
    { key: "X-Content-Type-Options", value: "nosniff" },
    { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
    { key: "Permissions-Policy", value: PERMISSIONS_POLICY },
    // Legacy twin of frame-ancestors 'none', for browsers that predate CSP 2.
    { key: "X-Frame-Options", value: "DENY" },
    // Isolates the browsing context group from cross-origin popups/openers.
    //
    // Cross-Origin-Embedder-Policy is deliberately NOT set. COOP + COEP would
    // make the page crossOriginIsolated, which is what Pyodide's interrupt
    // buffer (a SharedArrayBuffer) needs to stop a runaway Python loop. But
    // that only helps once Python runs in a worker (today it runs on the main
    // thread, which a loop blocks anyway), and COEP blocks every cross-origin
    // resource that doesn't opt in via CORP/CORS — a site-wide change to verify
    // on its own. Deferred.
    { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
  ];
}

/**
 * Cache policy for the self-hosted Pyodide runtime (~21 MB: wasm, stdlib,
 * wheels). Without one, Vercel serves public/ files as `max-age=0,
 * must-revalidate`: ten conditional requests before Python can start.
 *
 * At the default, unversioned /pyodide/ path: a day fresh, then a week of
 * stale-while-revalidate — NOT `immutable`. After a Pyodide upgrade an immutable
 * cache could pair an old pyodide.asm.wasm with a new pyodide.mjs at the same
 * URL; a day's staleness bounds that window, and stale-while-revalidate
 * refreshes it in the background on the next visit.
 *
 * At a versioned path (NEXT_PUBLIC_PYODIDE_INDEX_URL=/pyodide/v<version>/, which
 * `pyodideLocation` holds to the installed version) an upgrade changes the URL,
 * so a year and `immutable` are safe.
 */
export const PYODIDE_CACHE_CONTROL =
  "public, max-age=86400, stale-while-revalidate=604800";
export const PYODIDE_CACHE_CONTROL_VERSIONED =
  "public, max-age=31536000, immutable";

/** Everything `headers()` returns. Later rules win on a shared key; none share. */
export function buildHeaders(env: HeaderEnv = {}): HeaderRules {
  const rules: HeaderRules = [
    { source: "/:path*", headers: securityHeaders(env) },
  ];
  const pyodide = pyodideLocation(env.NEXT_PUBLIC_PYODIDE_INDEX_URL);
  if (pyodide.kind === "path") {
    rules.push({
      source: `${pyodide.path}:path*`,
      headers: [
        {
          key: "Cache-Control",
          value: pyodide.versioned
            ? PYODIDE_CACHE_CONTROL_VERSIONED
            : PYODIDE_CACHE_CONTROL,
        },
      ],
    });
  }
  return rules;
}

/** What `siteOriginWarning` reads: src/lib/site.ts's two variables, and the mode. */
export interface SiteOriginEnv {
  NEXT_PUBLIC_SITE_URL?: string;
  VERCEL_PROJECT_PRODUCTION_URL?: string;
  NODE_ENV?: string;
}

/** src/lib/site.ts's fallback origin, for when neither variable is set. */
export const LOCAL_SITE_ORIGIN = "http://localhost:3000";

/**
 * The warning a production build prints when it has no site origin, or null.
 *
 * `siteUrl()` in src/lib/site.ts falls back to http://localhost:3000 when
 * neither NEXT_PUBLIC_SITE_URL nor VERCEL_PROJECT_PRODUCTION_URL is set. That
 * is right for `next dev` and for a CI build nobody deploys, and wrong for a
 * deploy: every page's canonical URL and og:url, the sitemap and robots.txt
 * would all name localhost. Vercel always sets VERCEL_PROJECT_PRODUCTION_URL,
 * so this is for every other host (Netlify, Docker, Cloudflare, a VM).
 *
 * The same test as `siteUrl()`, restated here rather than imported so the
 * config loader never has to transpile app code; security-headers.test.ts
 * checks the two agree. An invalid value is not this function's business:
 * `siteUrl()` fails the build on it with its own message.
 */
export function siteOriginWarning(env: SiteOriginEnv): string | null {
  if (env.NODE_ENV !== "production") return null;
  if (env.NEXT_PUBLIC_SITE_URL?.trim()) return null;
  if (env.VERCEL_PROJECT_PRODUCTION_URL?.trim()) return null;
  return (
    `⚠ NEXT_PUBLIC_SITE_URL is not set, so this build's canonical URLs, og:url, ` +
    `sitemap.xml and robots.txt all point at ${LOCAL_SITE_ORIGIN}.\n` +
    `  That is fine for a local or CI build that won't be deployed. Before ` +
    `deploying, set NEXT_PUBLIC_SITE_URL to the site's origin ` +
    `(e.g. https://gameml.example) and rebuild. On Vercel it isn't needed.`
  );
}

let siteOriginWarned = false;

/**
 * Prints `siteOriginWarning` at most once per process.
 *
 * Called from `headers()` because that is the hook `next build` runs exactly
 * once, in its main process, before compiling (next/dist/build/index.js,
 * "load-custom-routes"). The alternatives print at the wrong time or too often:
 * this module is evaluated again by `next start`, and page metadata is
 * evaluated in every static-generation worker. `next dev` calls `headers()`
 * too, with NODE_ENV=development, which stays quiet.
 */
function warnAboutSiteOrigin(): void {
  if (siteOriginWarned) return;
  const warning = siteOriginWarning({
    NEXT_PUBLIC_SITE_URL: process.env.NEXT_PUBLIC_SITE_URL,
    VERCEL_PROJECT_PRODUCTION_URL: process.env.VERCEL_PROJECT_PRODUCTION_URL,
    NODE_ENV: process.env.NODE_ENV,
  });
  if (!warning) return;
  siteOriginWarned = true;
  console.warn(warning);
}

const nextConfig: NextConfig = {
  reactStrictMode: true,

  // Drop the `X-Powered-By: Next.js` banner; it only helps fingerprinting.
  poweredByHeader: false,

  async headers() {
    warnAboutSiteOrigin();
    return buildHeaders({
      NEXT_PUBLIC_SUPABASE_URL: process.env.NEXT_PUBLIC_SUPABASE_URL,
      NEXT_PUBLIC_PYODIDE_INDEX_URL: process.env.NEXT_PUBLIC_PYODIDE_INDEX_URL,
      VERCEL_ENV: process.env.VERCEL_ENV,
      CSP_REPORT_ONLY: process.env.CSP_REPORT_ONLY,
    });
  },

  // GameML has no ML backend by design (CLAUDE.md: "Never add a server-side ML
  // inference endpoint"). Keeping TF.js / Three.js / Pyodide out of the server
  // graph is NOT done here: every game is loaded through `next/dynamic` with
  // `ssr: false` in src/games/GameMount.tsx, and Pyodide is fetched from
  // /pyodide/ at run time.
  experimental: {
    // Rewrites barrel imports (`import { scaleLinear } from "d3"`) into direct
    // submodule imports, so only the d3 modules actually used are bundled.
    // lucide-react is already on Next's built-in list; naming it is harmless.
    optimizePackageImports: ["lucide-react", "d3"],
  },
};

export default nextConfig;
