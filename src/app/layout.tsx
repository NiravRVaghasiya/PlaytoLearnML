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
        {/* DESIGN.md §9 — keyboard users get a way past the chrome. */}
        <a
          href="#game-canvas"
          className="sr-only-live focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:rounded-md focus:bg-primary focus:px-4 focus:py-2 focus:text-primary-ink"
        >
          Skip to game
        </a>
        {children}
      </body>
    </html>
  );
}
