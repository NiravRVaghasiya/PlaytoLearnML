import type { MetadataRoute } from "next";
import { playableSlugs } from "@/games/registry";
import { conceptSlugs } from "@/lib/concepts";
import { absoluteUrl, siteUrl } from "@/lib/site";

/**
 * `/sitemap.xml`: the home page, every playable game, the Concept Library and
 * each of its pages. Spec §8 names the library as the SEO growth loop, so its
 * pages are listed at a higher priority than the games they funnel into.
 *
 * Built from the registry and the library rather than a list, so a new game or
 * concept is in the sitemap the moment it is on the site. Static: evaluated
 * once at build, like the pages it lists.
 *
 * No `lastModified`. Nothing records when a page last changed, and a build
 * timestamp would tell crawlers every page changed on every deploy.
 */
export default function sitemap(): MetadataRoute.Sitemap {
  const base = siteUrl();
  const entry = (
    path: string,
    priority: number,
  ): MetadataRoute.Sitemap[number] => ({
    url: absoluteUrl(path, base),
    changeFrequency: "monthly",
    priority,
  });

  return [
    entry("/", 1),
    entry("/concepts", 0.8),
    ...conceptSlugs().map((slug) => entry(`/concepts/${slug}`, 0.7)),
    ...playableSlugs().map((slug) => entry(`/play/${slug}`, 0.6)),
  ];
}
