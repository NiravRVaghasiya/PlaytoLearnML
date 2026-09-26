import type { Metadata, Viewport } from "next";
import { Space_Grotesk, Inter, JetBrains_Mono } from "next/font/google";
import { SITE_DESCRIPTION, SITE_NAME, SITE_TITLE, siteUrl } from "@/lib/site";
import { SkipLink } from "./_components/SkipLink";
import "./globals.css";

/* DESIGN.md §3 Typography.
   Each font exposes a CSS variable that globals.css maps into the Tailwind
   theme (`font-display`, `font-body`, `font-mono`). */
const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk",
  display: "swap",
});

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
  display: "swap",
});

/**
 * Site-wide defaults. Pages add their own canonical URL, Open Graph and Twitter
 * card through `pageSocialMetadata()` — deliberately NOT here: Next merges
 * metadata shallowly, so an `alternates.canonical` set in the layout would be
 * inherited by every page that forgot its own, and they would all claim to be
 * the home page.
 *
 * The icons come from the file conventions next to this layout (`icon.svg`,
 * `favicon.ico`), so there is no `icons` key.
 */
export const metadata: Metadata = {
  metadataBase: siteUrl(),
  title: {
    default: SITE_TITLE,
    template: `%s · ${SITE_NAME}`,
  },
  description: SITE_DESCRIPTION,
  applicationName: SITE_NAME,
  openGraph: {
    siteName: SITE_NAME,
    type: "website",
    locale: "en_US",
  },
};

export const viewport: Viewport = {
  themeColor: "#0E1116", // --bg
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body
        className={`${spaceGrotesk.variable} ${inter.variable} ${jetbrainsMono.variable} font-body antialiased`}
      >
        {/* Targets #main-content, which every route — the 404 and error pages
            included — renders on the server. See SkipLink for why game pages
            then hand focus on to the canvas. */}
        <SkipLink />
        {children}
      </body>
    </html>
  );
}
