"use client";

import { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertTriangle, RotateCcw } from "lucide-react";
import { Button } from "@/components/Button";

interface LaneErrorBoundaryProps {
  /** Which lane is mounted — shown in the message so the player knows what broke. */
  laneLabel: string;
  children: ReactNode;
}

interface LaneErrorBoundaryState {
  error: Error | null;
}

/**
 * Keeps a crashing lane from taking the whole page with it.
 *
 * Without a boundary, a render-time throw in a lane — a WebGL canvas that can't
 * get a context, a TF.js call on a disposed model — unwinds to Next's root
 * boundary, which replaces everything (header, Back link, metric, controls) with
 * a generic "Application error". Caught here, the rest of the shell stays up:
 * the player can retry the lane or switch to the other one, and game state is
 * untouched because it lives in the game's store, not in the lane.
 *
 * `GameShell` keys this by lane, so switching lanes always starts clean.
 */
export class LaneErrorBoundary extends Component<
  LaneErrorBoundaryProps,
  LaneErrorBoundaryState
> {
  state: LaneErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: unknown): LaneErrorBoundaryState {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Still reported, so it shows up in the console and in any error tooling.
    console.error(
      `GameShell: the ${this.props.laneLabel} crashed`,
      error,
      info.componentStack,
    );
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div className="flex h-full min-h-[240px] flex-col items-start justify-center gap-3 p-4">
        <p className="flex items-center gap-2 font-display font-semibold text-text">
          <AlertTriangle aria-hidden="true" className="size-5 text-warn" />
          The {this.props.laneLabel} stopped working.
        </p>
        <p className="max-w-prose text-sm text-text-muted">
          {error.message || "Something went wrong while drawing it."} Try it
          again, or switch to the other lane — the game itself is still here.
        </p>
        <Button
          variant="secondary"
          size="sm"
          onClick={() => this.setState({ error: null })}
          icon={<RotateCcw className="size-4" />}
        >
          Try the {this.props.laneLabel} again
        </Button>
      </div>
    );
  }
}
