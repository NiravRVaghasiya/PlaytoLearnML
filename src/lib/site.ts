/**
 * Where the site lives, and the metadata every page shares.
 *
 * Canonical URLs, Open Graph `og:url` and the sitemap all need an ABSOLUTE
 * origin, and a statically-built page cannot ask the request for one. So the
 * origin is resolved once, at build time, in this order:
 *
 *   1. `NEXT_PUBLIC_SITE_URL` — set this on any real deploy with a custom domain.
 *   2. `https://$VERCEL_PROJECT_PRODUCTION_URL` — Vercel sets this on every
 *      build, previews included, to the project's production domain. That is
 *      the right answer for a preview too: a preview's canonical URL should
 *      point at production, not at a throwaway `*-git-branch.vercel.app` host
 *      that search engines would otherwise index as a duplicate.
 *   3. `http://localhost:3000` — local development and CI. A production build
 *      that lands here prints a warning once (`siteOriginWarning` in
 *      next.config.ts), because a deploy built this way would ship localhost
 *      canonicals on every page.
 *
 * Server-only in practice (metadata, `sitemap.ts`, `robots.ts`). Nothing here is
 * secret, but nothing on the client needs it either.
 */

export const SITE_NAME = "GameML";

/** The home page's `<title>`, also the social-card title for the home page. */
export const SITE_TITLE = "GameML — learn Machine Learning by playing";

export const SITE_DESCRIPTION =
  "Fourteen games that teach real Machine Learning. Every model trains in your browser — no server, no GPU, no signup to start.";

const LOCAL_ORIGIN = "http://localhost:3000";

/** The two variables `siteUrl` reads, so tests can pass them explicitly. */
export interface SiteEnv {
  NEXT_PUBLIC_SITE_URL?: string | undefined;
  VERCEL_PROJECT_PRODUCTION_URL?: string | undefined;
}

/**
 * Accepts `https://gameml.dev`, `https://gameml.dev/` and bare `gameml.dev`.
 * The bare form is how people actually paste a domain into a dashboard, and it
 * is the form Vercel itself uses for `VERCEL_PROJECT_PRODUCTION_URL`.
 */
function parseOrigin(raw: string, source: keyof SiteEnv): URL {
  const trimmed = raw.trim();
  const withScheme = /^[a-z][a-z0-9+.-]*:\/\//i.test(trimmed)
    ? trimmed
    : `https://${trimmed}`;

  let url: URL;
  try {
    url = new URL(withScheme);
  } catch {
    // Fail the build rather than quietly falling back to localhost. A typo here
    // would otherwise ship canonical URLs pointing at http://localhost:3000 on
    // every page, which is worse for search than having none.
    throw new Error(
      `${source} is set to "${raw}", which is not a valid site URL. Use an absolute origin such as https://gameml.example.`,
    );
  }
  if (url.protocol !== "https:" && url.protocol !== "http:") {
    throw new Error(
      `${source} must be an http(s) origin, got "${raw}".`,
    );
  }
  // An origin, not a page: a path here would be silently prepended to every
  // canonical URL. `new URL(path, base)` resolves against the origin anyway.
  return new URL(url.origin);
}

/**
 * The absolute origin used as Next's `metadataBase` and for the sitemap.
 * Pure over `env` so the resolution order is unit-testable.
 */
export function siteUrl(
  env: SiteEnv = {
    // Spelled out rather than passing `process.env` whole, so each variable is a
    // literal `process.env.X` read — the only form Next inlines at build time.
    NEXT_PUBLIC_SITE_URL: process.env.NEXT_PUBLIC_SITE_URL,
    VERCEL_PROJECT_PRODUCTION_URL: process.env.VERCEL_PROJECT_PRODUCTION_URL,
  },
): URL {
  if (env.NEXT_PUBLIC_SITE_URL?.trim()) {
    return parseOrigin(env.NEXT_PUBLIC_SITE_URL, "NEXT_PUBLIC_SITE_URL");
  }
  if (env.VERCEL_PROJECT_PRODUCTION_URL?.trim()) {
    return parseOrigin(
      env.VERCEL_PROJECT_PRODUCTION_URL,
      "VERCEL_PROJECT_PRODUCTION_URL",
    );
  }
  return new URL(LOCAL_ORIGIN);
}

/** `path` resolved against the site origin, as a string. */
export function absoluteUrl(path: string, base: URL = siteUrl()): string {
  return new URL(path, base).toString();
}

export interface PageSocialInput {
  /** The page's own title, WITHOUT the " · GameML" suffix. */
  title: string;
  description: string;
  /** Site-relative path, e.g. `/play/sort-it-arcade`. Resolved via metadataBase. */
  path: string;
  type?: "website" | "article";
}

/**
 * Canonical + Open Graph + Twitter card for one page.
 *
 * One helper rather than per-page literals because Next merges metadata
 * SHALLOWLY: a page that sets `openGraph` replaces the layout's `openGraph`
 * wholesale, so a page that forgot `siteName` would silently drop it. Every
 * page goes through here, so every page gets the full set.
 *
 * No `og:image`: there is no social image yet, and a generated one would need a
 * font fetched at build time. Twitter's `summary` card is the one that renders
 * correctly without an image.
 */
export function pageSocialMetadata({
  title,
  description,
  path,
  type = "website",
}: PageSocialInput) {
  const socialTitle = title === SITE_TITLE ? title : `${title} · ${SITE_NAME}`;
  return {
    alternates: { canonical: path },
    openGraph: {
      title: socialTitle,
      description,
      url: path,
      siteName: SITE_NAME,
      type,
      locale: "en_US",
    },
    twitter: {
      card: "summary" as const,
      title: socialTitle,
      description,
    },
  };
}
