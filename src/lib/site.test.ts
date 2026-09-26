import { afterEach, describe, expect, it, vi } from "vitest";
import {
  LOCAL_SITE_ORIGIN,
  siteOriginWarning,
  type SiteOriginEnv,
} from "../../next.config";
import {
  SITE_NAME,
  SITE_TITLE,
  absoluteUrl,
  pageSocialMetadata,
  siteUrl,
  type SiteEnv,
} from "./site";

describe("siteUrl", () => {
  it("prefers NEXT_PUBLIC_SITE_URL over the Vercel production domain", () => {
    const url = siteUrl({
      NEXT_PUBLIC_SITE_URL: "https://gameml.example",
      VERCEL_PROJECT_PRODUCTION_URL: "gameml.vercel.app",
    });
    expect(url.origin).toBe("https://gameml.example");
  });

  it("falls back to https:// plus the Vercel production domain", () => {
    // Vercel provides the bare host, with no scheme.
    const url = siteUrl({ VERCEL_PROJECT_PRODUCTION_URL: "gameml.vercel.app" });
    expect(url.origin).toBe("https://gameml.vercel.app");
  });

  it("falls back to localhost when neither is set, or both are blank", () => {
    expect(siteUrl({}).origin).toBe("http://localhost:3000");
    expect(
      siteUrl({ NEXT_PUBLIC_SITE_URL: "  ", VERCEL_PROJECT_PRODUCTION_URL: "" })
        .origin,
    ).toBe("http://localhost:3000");
  });

  it("normalises a bare domain, a trailing slash and a stray path to an origin", () => {
    expect(siteUrl({ NEXT_PUBLIC_SITE_URL: "gameml.example" }).href).toBe(
      "https://gameml.example/",
    );
    expect(siteUrl({ NEXT_PUBLIC_SITE_URL: "https://gameml.example/" }).href).toBe(
      "https://gameml.example/",
    );
    // A path would otherwise be prepended to every canonical URL.
    expect(
      siteUrl({ NEXT_PUBLIC_SITE_URL: "https://gameml.example/app/" }).href,
    ).toBe("https://gameml.example/");
  });

  it("fails loudly on a value that is not a URL, instead of shipping localhost canonicals", () => {
    expect(() => siteUrl({ NEXT_PUBLIC_SITE_URL: "https://" })).toThrow(
      /NEXT_PUBLIC_SITE_URL/,
    );
    expect(() => siteUrl({ NEXT_PUBLIC_SITE_URL: "ftp://gameml.example" })).toThrow(
      /http\(s\)/,
    );
  });
});

describe("absoluteUrl", () => {
  it("resolves a site path against the origin", () => {
    const base = new URL("https://gameml.example");
    expect(absoluteUrl("/play/sort-it-arcade", base)).toBe(
      "https://gameml.example/play/sort-it-arcade",
    );
    expect(absoluteUrl("/", base)).toBe("https://gameml.example/");
  });
});

describe("pageSocialMetadata", () => {
  it("gives every page a canonical, a full Open Graph block and a Twitter card", () => {
    const meta = pageSocialMetadata({
      title: "Sort-It Arcade",
      description: "A classifier draws a boundary.",
      path: "/play/sort-it-arcade",
    });
    expect(meta.alternates.canonical).toBe("/play/sort-it-arcade");
    // Next merges metadata shallowly, so each page's openGraph must carry the
    // site-level fields itself or they vanish from that page.
    expect(meta.openGraph).toMatchObject({
      title: `Sort-It Arcade · ${SITE_NAME}`,
      description: "A classifier draws a boundary.",
      url: "/play/sort-it-arcade",
      siteName: SITE_NAME,
      type: "website",
    });
    expect(meta.twitter).toMatchObject({
      card: "summary",
      title: `Sort-It Arcade · ${SITE_NAME}`,
    });
  });

  it("does not double-brand the home page title", () => {
    const meta = pageSocialMetadata({
      title: SITE_TITLE,
      description: "x",
      path: "/",
    });
    expect(meta.openGraph.title).toBe(SITE_TITLE);
  });

  it("marks concept explainers as articles", () => {
    const meta = pageSocialMetadata({
      title: "What is overfitting?",
      description: "x",
      path: "/concepts/overfitting",
      type: "article",
    });
    expect(meta.openGraph.type).toBe("article");
  });
});

describe("the build warning for localhost canonicals", () => {
  const production = (env: SiteEnv): SiteOriginEnv => ({
    ...env,
    NODE_ENV: "production",
  });

  it("warns on a production build with no origin, naming what it breaks and the fix", () => {
    const warning = siteOriginWarning(production({}));
    expect(warning).not.toBeNull();
    for (const part of [
      LOCAL_SITE_ORIGIN,
      "NEXT_PUBLIC_SITE_URL",
      "canonical",
      "og:url",
      "sitemap.xml",
      "robots.txt",
    ]) {
      expect(warning).toContain(part);
    }
  });

  it("stays quiet in dev and test, and whenever an origin is set", () => {
    expect(siteOriginWarning({ NODE_ENV: "development" })).toBeNull();
    expect(siteOriginWarning({ NODE_ENV: "test" })).toBeNull();
    expect(siteOriginWarning({})).toBeNull();
    expect(
      siteOriginWarning(production({ NEXT_PUBLIC_SITE_URL: "https://gameml.example" })),
    ).toBeNull();
    expect(
      siteOriginWarning(production({ VERCEL_PROJECT_PRODUCTION_URL: "gameml.vercel.app" })),
    ).toBeNull();
  });

  it("fires exactly when siteUrl() falls back to localhost", () => {
    // next.config.ts restates siteUrl()'s test rather than importing it, so the
    // two are held together here: same inputs, same answer.
    const envs: SiteEnv[] = [
      {},
      { NEXT_PUBLIC_SITE_URL: "  ", VERCEL_PROJECT_PRODUCTION_URL: "" },
      { NEXT_PUBLIC_SITE_URL: "https://gameml.example" },
      { NEXT_PUBLIC_SITE_URL: "gameml.example" },
      { VERCEL_PROJECT_PRODUCTION_URL: "gameml.vercel.app" },
      { NEXT_PUBLIC_SITE_URL: "", VERCEL_PROJECT_PRODUCTION_URL: "gameml.vercel.app" },
    ];
    for (const env of envs) {
      const fellBack = siteUrl(env).origin === new URL(LOCAL_SITE_ORIGIN).origin;
      expect(siteOriginWarning(production(env)) !== null, JSON.stringify(env)).toBe(
        fellBack,
      );
    }
  });

  describe("printed by next.config's headers() hook", () => {
    afterEach(() => {
      vi.unstubAllEnvs();
      vi.restoreAllMocks();
    });

    /** A freshly evaluated next.config, so its print-once flag starts clear. */
    async function freshConfig() {
      vi.resetModules();
      return (await import("../../next.config")).default;
    }

    function stubBuildEnv(nodeEnv: string) {
      vi.stubEnv("NODE_ENV", nodeEnv);
      vi.stubEnv("NEXT_PUBLIC_SITE_URL", "");
      vi.stubEnv("VERCEL_PROJECT_PRODUCTION_URL", "");
      // headers() also reads these; keep them at their defaults.
      vi.stubEnv("NEXT_PUBLIC_PYODIDE_INDEX_URL", "");
      vi.stubEnv("NEXT_PUBLIC_SUPABASE_URL", "");
      vi.stubEnv("VERCEL_ENV", "");
      vi.stubEnv("CSP_REPORT_ONLY", "");
    }

    it("once per build, however often Next asks for the headers", async () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      stubBuildEnv("production");
      const config = await freshConfig();
      await config.headers?.();
      await config.headers?.();
      expect(warn).toHaveBeenCalledTimes(1);
      expect(warn).toHaveBeenCalledWith(siteOriginWarning(production({})));
    });

    it("not at all under next dev", async () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      stubBuildEnv("development");
      const config = await freshConfig();
      await config.headers?.();
      expect(warn).not.toHaveBeenCalled();
    });
  });
});
