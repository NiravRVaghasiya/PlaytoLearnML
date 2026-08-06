"use client";

import type { ComponentPropsWithRef, ReactNode } from "react";
import { cx } from "@/lib/utils";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
export type ButtonSize = "sm" | "md";

// `ComponentPropsWithRef` rather than `ButtonHTMLAttributes` so callers can pass
// `ref` directly — React 19 treats ref as a normal prop for function components.
export interface ButtonProps extends ComponentPropsWithRef<"button"> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  /** Decorative leading icon. Must never be the only label (DESIGN.md §9). */
  icon?: ReactNode;
  children: ReactNode;
}

const VARIANTS: Record<ButtonVariant, string> = {
  primary: "bg-primary text-primary-ink hover:brightness-110",
  secondary:
    "bg-surface-2 text-text border border-border hover:border-primary/60",
  ghost: "bg-transparent text-text-muted hover:text-text hover:bg-surface-2",
  // Dark ink on the red, not white. White on --wrong (#E74C3C) is 3.8:1, below
  // the 4.5:1 AA minimum for normal text; --bg on the same red is 4.7:1. This
  // mirrors how `primary` pairs --primary with --primary-ink rather than white.
  danger: "bg-wrong text-bg hover:brightness-110",
};

const SIZES: Record<ButtonSize, string> = {
  // Both keep the 44px minimum touch target from DESIGN.md §9.
  sm: "min-h-11 px-3 text-sm",
  md: "min-h-11 px-4 text-base",
};

/**
 * The shared action button (DESIGN.md §6).
 *
 * Games must not roll their own — Retry/Next/Step/Train all come from here so
 * the 44px target and focus ring behaviour stay consistent across 14 games.
 */
export function Button({
  variant = "secondary",
  size = "md",
  icon,
  children,
  className,
  type = "button",
  ...rest
}: ButtonProps) {
  return (
    <button
      type={type}
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-md font-medium",
        "transition-[filter,background-color,border-color,color] dur-micro",
        "disabled:cursor-not-allowed disabled:opacity-50",
        VARIANTS[variant],
        SIZES[size],
        className,
      )}
      {...rest}
    >
      {icon ? (
        <span aria-hidden="true" className="shrink-0">
          {icon}
        </span>
      ) : null}
      {children}
    </button>
  );
}
