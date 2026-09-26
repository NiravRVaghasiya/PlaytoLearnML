import { readFileSync, readdirSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it, vi } from "vitest";
import nextConfig, {
  buildHeaders,
  contentSecurityPolicy,
  PYODIDE_CACHE_CONTROL,
  PYODIDE_CACHE_CONTROL_VERSIONED,
  PYODIDE_DEFAULT_PATH,
  PYODIDE_VERSION,
  pyodideLocation,
  securityHeaders,
  type HeaderEnv,
} from "../../next.config";

/**
 * The response headers every route ships with (next.config.ts).
 *
 * A CSP fails in two directions, and both are silent until a player hits them:
 * too loose and it protects nothing (one `*` in script-src undoes the rest); too
 * tight and a lane dies — the JS lane needs eval, Pyodide needs wasm compilation
 * — with nothing but a console line to say why. These tests pin both edges, and
 * scan the app for third-party origins the policy would block.
 */

/** Directive name → its source list. Throws on a repeated directive. */
function parse(csp: string): Map<string, string[]> {
  const directives = new Map<string, string[]>();
  for (const part of csp.split(";")) {
    const [name, ...sources] = part.trim().split(/\s+/);
    if (!name) throw new Error(`empty directive in: ${csp}`);
    if (directives.has(name)) throw new Error(`"${name}" appears twice`);
    directives.set(name, sources);
  }
  return directives;
}

const sources = (csp: string, directive: string): string[] => {
  const list = parse(csp).get(directive);
  if (!list) throw new Error(`no ${directive} in: ${csp}`);
  return list;
};

const header = (env: HeaderEnv, key: string) =>
  securityHeaders(env).find((pair) => pair.key === key)?.value;

/** Every environment shape the policy is built for. */
const ENVS: Record<string, HeaderEnv> = {
  local: {},
  production: { VERCEL_ENV: "production" },
  preview: { VERCEL_ENV: "preview" },
  supabase: {
    VERCEL_ENV: "preview",
    NEXT_PUBLIC_SUPABASE_URL: "https://abcd.supabase.co",
  },
  cdn: {
    NEXT_PUBLIC_PYODIDE_INDEX_URL: "https://cdn.jsdelivr.net/pyodide/v314.0.5/full/",
  },
};

describe("Content-Security-Policy", () => {
  it("has exactly the directives the app needs, in production", () => {
    const csp = parse(contentSecurityPolicy({ VERCEL_ENV: "production" }));
    expect(Object.fromEntries(csp)).toEqual({
      "default-src": ["'self'"],
      "script-src": [
        "'self'",
        "'unsafe-inline'",
        "'unsafe-eval'",
        "'wasm-unsafe-eval'",
      ],
      "style-src": ["'self'", "'unsafe-inline'"],
      "img-src": ["'self'", "data:", "blob:"],
      "font-src": ["'self'", "data:"],
      "connect-src": ["'self'"],
      "worker-src": ["'self'", "blob:"],
      "frame-src": ["'self'"],
      "object-src": ["'none'"],
      "base-uri": ["'self'"],
      "form-action": ["'self'"],
      "frame-ancestors": ["'none'"],
    });
  });

  it.each(Object.entries(ENVS))(
    "never allows a wildcard or scheme-wide script source (%s)",
    (_, env) => {
      const csp = contentSecurityPolicy(env);
      for (const directive of ["script-src", "default-src"]) {
        for (const source of sources(csp, directive)) {
          expect(source, `${directive} ${source}`).not.toContain("*");
          // A bare scheme ("https:", "data:", "blob:") admits every script
          // served over it — as good as a wildcard.
          expect(source, `${directive} ${source}`).not.toMatch(/^[a-z][a-z0-9+.-]*:$/i);
        }
      }
      // And the lock-down directives never loosen, whatever the environment.
      expect(sources(csp, "object-src")).toEqual(["'none'"]);
      expect(sources(csp, "frame-ancestors")).toEqual(["'none'"]);
      expect(sources(csp, "base-uri")).toEqual(["'self'"]);
      expect(sources(csp, "form-action")).toEqual(["'self'"]);
    },
  );

  it("lets the Vercel toolbar in on preview deployments only", () => {
    const preview = contentSecurityPolicy({ VERCEL_ENV: "preview" });
    for (const directive of ["script-src", "connect-src", "frame-src", "img-src"]) {
      expect(sources(preview, directive), directive).toContain(
        "https://vercel.live",
      );
    }
    for (const env of [ENVS.local, ENVS.production]) {
      expect(contentSecurityPolicy(env)).not.toContain("vercel");
    }
  });

  it("adds the Supabase origin and its WebSocket twin to connect-src only", () => {
    const csp = contentSecurityPolicy({
      NEXT_PUBLIC_SUPABASE_URL: "https://abcd.supabase.co/rest/v1/",
    });
    // Origin only — a path would restrict every request to that path.
    expect(sources(csp, "connect-src")).toEqual([
      "'self'",
      "https://abcd.supabase.co",
      "wss://abcd.supabase.co",
    ]);
    expect(sources(csp, "script-src").join(" ")).not.toContain("supabase");

    // A local Supabase (supabase start) is plain http on a port.
    const local = contentSecurityPolicy({
      NEXT_PUBLIC_SUPABASE_URL: "http://127.0.0.1:54321",
    });
    expect(sources(local, "connect-src")).toEqual([
      "'self'",
      "http://127.0.0.1:54321",
      "ws://127.0.0.1:54321",
    ]);
  });

  it("treats an unset or blank Supabase URL as no Supabase", () => {
    for (const value of [undefined, "", "   "]) {
      expect(
        sources(
          contentSecurityPolicy({ NEXT_PUBLIC_SUPABASE_URL: value }),
          "connect-src",
        ),
      ).toEqual(["'self'"]);
    }
  });

  it("fails the build, by name, on a malformed Supabase URL", () => {
    expect(() =>
      contentSecurityPolicy({ NEXT_PUBLIC_SUPABASE_URL: "abcd.supabase.co" }),
    ).toThrow(/NEXT_PUBLIC_SUPABASE_URL is not a valid URL/);
    expect(() =>
      contentSecurityPolicy({ NEXT_PUBLIC_SUPABASE_URL: "javascript:alert(1)" }),
    ).toThrow(/NEXT_PUBLIC_SUPABASE_URL must be an http\(s\) URL/);
  });
});

describe("security headers", () => {
  it("sends one enforcing CSP by default, report-only only when asked", () => {
    const keys = (env: HeaderEnv) =>
      securityHeaders(env)
        .map((pair) => pair.key)
        .filter((key) => key.startsWith("Content-Security-Policy"));

    expect(keys({})).toEqual(["Content-Security-Policy"]);
    expect(keys({ CSP_REPORT_ONLY: "0" })).toEqual(["Content-Security-Policy"]);
    expect(keys({ CSP_REPORT_ONLY: "1" })).toEqual([
      "Content-Security-Policy-Report-Only",
    ]);
  });

  it("sets the hardening headers", () => {
    expect(header({}, "X-Content-Type-Options")).toBe("nosniff");
    expect(header({}, "Referrer-Policy")).toBe("strict-origin-when-cross-origin");
    expect(header({}, "X-Frame-Options")).toBe("DENY");
    expect(header({}, "Cross-Origin-Opener-Policy")).toBe("same-origin");
  });

  it("denies the powerful features the app never uses", () => {
    const policy = header({}, "Permissions-Policy") ?? "";
    const entries = policy.split(", ");
    for (const entry of entries) expect(entry).toMatch(/^[a-z-]+=\(\)$/);
    const features = entries.map((entry) => entry.replace("=()", ""));
    expect(new Set(features).size).toBe(features.length);
    for (const feature of ["camera", "microphone", "geolocation"]) {
      expect(features).toContain(feature);
    }
    // Games may legitimately go fullscreen.
    expect(features).not.toContain("fullscreen");
  });

  it("does not set COEP, which would block cross-origin resources site-wide", () => {
    const keys = securityHeaders({}).map((pair) => pair.key.toLowerCase());
    expect(keys).not.toContain("cross-origin-embedder-policy");
  });
});

describe("headers() routing", () => {
  it("applies the security headers to every route", () => {
    const [all] = buildHeaders({});
    expect(all?.source).toBe("/:path*");
    expect(all?.headers).toEqual(securityHeaders({}));
  });

  /** The Cache-Control rule for a Pyodide location, if there is one. */
  const pyodideCache = (env: HeaderEnv) =>
    buildHeaders(env)
      .slice(1)
      .map((rule) => ({
        source: rule.source,
        value: rule.headers.find((pair) => pair.key === "Cache-Control")?.value,
      }));

  it("caches the unversioned Pyodide runtime for a day, never as immutable", () => {
    expect(pyodideCache({})).toEqual([
      { source: "/pyodide/:path*", value: PYODIDE_CACHE_CONTROL },
    ]);
    expect(PYODIDE_CACHE_CONTROL).toBe(
      "public, max-age=86400, stale-while-revalidate=604800",
    );
    // The path is unversioned, so an immutable cache could pair an old
    // pyodide.asm.wasm with a new pyodide.mjs after an upgrade.
    expect(PYODIDE_CACHE_CONTROL).not.toContain("immutable");
  });

  it("caches a versioned Pyodide path as immutable, and only that path", () => {
    const path = `/pyodide/v${PYODIDE_VERSION}/`;
    expect(pyodideCache({ NEXT_PUBLIC_PYODIDE_INDEX_URL: path })).toEqual([
      { source: `${path}:path*`, value: PYODIDE_CACHE_CONTROL_VERSIONED },
    ]);
    expect(PYODIDE_CACHE_CONTROL_VERSIONED).toBe(
      "public, max-age=31536000, immutable",
    );
    // A non-version subdirectory is still just a path: no immutability.
    expect(
      pyodideCache({ NEXT_PUBLIC_PYODIDE_INDEX_URL: "/pyodide/runtime/" }),
    ).toEqual([
      { source: "/pyodide/runtime/:path*", value: PYODIDE_CACHE_CONTROL },
    ]);
  });

  it("serves a CDN-hosted runtime through the CSP, with no cache rule of ours", () => {
    const cdn = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
    const env = { NEXT_PUBLIC_PYODIDE_INDEX_URL: cdn };
    expect(pyodideCache(env)).toEqual([]);
    const csp = contentSecurityPolicy(env);
    // pyodide.mjs is import()ed (script-src); wasm, stdlib, wheels are fetched.
    expect(sources(csp, "script-src")).toContain("https://cdn.jsdelivr.net");
    expect(sources(csp, "connect-src")).toContain("https://cdn.jsdelivr.net");
    expect(sources(csp, "img-src").join(" ")).not.toContain("jsdelivr");
    // And the default never names a CDN.
    expect(contentSecurityPolicy({})).not.toContain("jsdelivr");
  });

  it.each([
    ["no trailing slash", "/pyodide", /must end in "\/"/],
    ["outside /pyodide/ (not gitignored)", "/python/", /must be a path under \/pyodide\//],
    ["a traversal", "/pyodide/../etc/", /must not contain "\." or "\.\."/],
    ["a stale version", "/pyodide/v0.0.1/", /names v0\.0\.1 but pyodide .* is installed/],
    ["protocol-relative", "//cdn.example/pyodide/", /must be a path like \/pyodide\/ or an absolute/],
    ["not http(s)", "ftp://cdn.example/pyodide/", /must be an http\(s\) URL/],
  ])("rejects a Pyodide location that is %s", (_, value, message) => {
    expect(() => pyodideLocation(value)).toThrow(message);
    // …and the build fails on it, rather than shipping a lane that 404s.
    expect(() => buildHeaders({ NEXT_PUBLIC_PYODIDE_INDEX_URL: value })).toThrow(
      message,
    );
  });

  it("agrees with the engine on where Pyodide lives by default", () => {
    // useCodeLane falls back to "/pyodide/" when NEXT_PUBLIC_PYODIDE_INDEX_URL
    // is unset or empty. If that default moves, this config's cache rule and
    // setup-pyodide.mjs's staging directory must move with it.
    const engine = readFileSync(
      join(process.cwd(), "src", "engine", "useCodeLane.ts"),
      "utf8",
    );
    expect(engine).toContain(`"${PYODIDE_DEFAULT_PATH}"`);
    expect(pyodideLocation(undefined)).toEqual({
      kind: "path",
      path: PYODIDE_DEFAULT_PATH,
      versioned: false,
    });
    expect(pyodideLocation("")).toEqual(pyodideLocation(undefined));
    const setup = readFileSync(
      join(process.cwd(), "scripts", "setup-pyodide.mjs"),
      "utf8",
    );
    expect(setup).toContain(`raw || "${PYODIDE_DEFAULT_PATH}"`);
  });

  it("is what next.config serves, built from the real build environment", async () => {
    vi.stubEnv("VERCEL_ENV", "preview");
    vi.stubEnv("NEXT_PUBLIC_SUPABASE_URL", "https://abcd.supabase.co");
    vi.stubEnv("NEXT_PUBLIC_PYODIDE_INDEX_URL", "");
    vi.stubEnv("CSP_REPORT_ONLY", "");
    try {
      const served = await nextConfig.headers?.();
      expect(served).toEqual(buildHeaders(ENVS.supabase));
      const csp = served?.[0]?.headers.find(
        (pair) => pair.key === "Content-Security-Policy",
      )?.value;
      expect(sources(csp ?? "", "connect-src")).toContain(
        "https://abcd.supabase.co",
      );
    } finally {
      vi.unstubAllEnvs();
    }
    // And no `X-Powered-By: Next.js` fingerprint.
    expect(nextConfig.poweredByHeader).toBe(false);
  });
});

describe("no third-party origin the CSP would block", () => {
  /** Every non-test .ts/.tsx under src/. */
  function appSources(dir: string): string[] {
    return readdirSync(dir, { withFileTypes: true }).flatMap((item) => {
      const path = join(dir, item.name);
      if (item.isDirectory()) return appSources(path);
      return /\.tsx?$/.test(item.name) && !/\.(test|spec)\.tsx?$/.test(item.name)
        ? [path]
        : [];
    });
  }

  it("never fetches, imports or opens a socket to an absolute URL", () => {
    // connect-src and script-src are 'self' only (plus opt-in Supabase). A
    // hardcoded CDN or API origin in any of these calls would be blocked in
    // production while working fine in every unit test, so catch it here.
    const call =
      /\b(?:fetch|import|importScripts|new\s+(?:Worker|SharedWorker|WebSocket|EventSource))\s*\(\s*["'`](?:https?|wss?):\/\//;
    const indexUrl = /indexURL\s*[:=]\s*["'`](?:https?):\/\//;
    const root = join(process.cwd(), "src");
    const offenders = appSources(root).filter((file) => {
      const source = readFileSync(file, "utf8");
      return call.test(source) || indexUrl.test(source);
    });
    expect(
      offenders.map((file) => relative(root, file)),
      "add the origin to the CSP in next.config.ts, or self-host it",
    ).toEqual([]);
  });
});
