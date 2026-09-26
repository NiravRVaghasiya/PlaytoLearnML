"use client"; // Error boundaries must be Client Components.

import { ErrorPanel } from "./_components/ErrorPanel";
import "./globals.css";

/**
 * Last-resort error UI, for a crash in the root layout itself.
 *
 * It REPLACES the root layout, so it renders its own `<html lang="en">` and
 * `<body>`, imports the global stylesheet itself (Next does not carry the
 * layout's styles over), and cannot export `metadata` — React 19's `<title>`
 * is hoisted into the head instead. The layout's fonts are not loaded here, so
 * text falls back to the system stack the theme already lists after them.
 */
export default function GlobalError({
  error,
  retry,
  reset,
}: {
  error: Error & { digest?: string };
  retry?: () => void;
  reset: () => void;
}) {
  return (
    <html lang="en">
      <body className="font-body antialiased">
        <title>Something went wrong · GameML</title>
        <ErrorPanel error={error} retry={retry ?? reset} />
      </body>
    </html>
  );
}
