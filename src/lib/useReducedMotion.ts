"use client";

import { useSyncExternalStore } from "react";

const QUERY = "(prefers-reduced-motion: reduce)";

function canQuery(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function";
}

function subscribe(onStoreChange: () => void): () => void {
  if (!canQuery()) return () => {};
  const mq = window.matchMedia(QUERY);
  mq.addEventListener("change", onStoreChange);
  return () => mq.removeEventListener("change", onStoreChange);
}

function getSnapshot(): boolean {
  return canQuery() ? window.matchMedia(QUERY).matches : false;
}

/** Server render can't know the preference; assume motion is allowed. */
function getServerSnapshot(): boolean {
  return false;
}

/**
 * Reactive `prefers-reduced-motion` (DESIGN.md §7/§9).
 *
 * Components use this to drop *decorative* motion while keeping *functional*
 * feedback. A metric must still change and still recolour when reduced motion is
 * on — it just snaps instead of tweening.
 *
 * `useSyncExternalStore` rather than `useState` + `useEffect`: matchMedia is
 * exactly the external store this hook is designed for, and it gets SSR
 * correctness and tear-free reads without a cascading render on mount.
 */
export function useReducedMotion(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}
