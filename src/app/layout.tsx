import type { Metadata, Viewport } from "next";
import { Space_Grotesk, Inter, JetBrains_Mono } from "next/font/google";
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

export const metadata: Metadata = {
  title: {
    default: "GameML — learn Machine Learning by playing",
    template: "%s · GameML",
  },
  description:
    "Fourteen games that teach real Machine Learning. Every model trains in your browser — no server, no GPU, no signup to start.",
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
        {/* DESIGN.md §9 — keyboard users get a way past the chrome.
            Two details that are easy to get wrong:

            1. Uses Tailwind's `sr-only` / `focus:not-sr-only` pair, which are
               built to undo each other. The project's own `.sr-only-live` helper
               is NOT interchangeable: `not-sr-only` doesn't reset its
               `clip-path`, so the link would stay clipped while focused.
            2. Targets `#main-content`, which every route renders on the SERVER.
               Pointing it at the game canvas looked right but was broken — games
               load via `dynamic(ssr: false)`, so that element doesn't exist when
               the page loads, and the link went nowhere. */}
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:rounded-md focus:bg-primary focus:px-4 focus:py-2 focus:text-primary-ink"
        >
          Skip to content
        </a>
        {children}
      </body>
    </html>
  );
}
