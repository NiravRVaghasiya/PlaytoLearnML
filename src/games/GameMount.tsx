"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import {
  Component,
  Fragment,
  useEffect,
  useRef,
  type ComponentType,
  type ErrorInfo,
  type ReactNode,
} from "react";
import { ArrowLeft, RefreshCw, RotateCcw } from "lucide-react";
import { Button } from "@/components/Button";
import { getGameMeta } from "@/lib/catalog";
import { isChunkLoadError } from "@/lib/chunk-load-error";

/**
 * Mounts a game by slug.
 *
 * A Client Component, because each entry is a `next/dynamic` import with
 * `ssr: false` — which Next only permits on the client. That flag is not
 * incidental: games lean on WebGL, pointer capture, and `requestAnimationFrame`,
 * so there is nothing to gain from server rendering them and plenty to break.
 *
 * The loading fallback matches the shell's canvas box so the layout doesn't jump
 * while a game's chunk arrives.
 *
 * Every game renders inside `GameErrorBoundary`, so a crash in one game that
 * its lane boundary can't contain — in its controls, in the shell, or a game
 * chunk that fails to download — replaces that game with a recovery panel that
 * names it, instead of taking the whole page down to Next's unbranded fallback
 * with no way home.
 */

const GAME_COMPONENTS: Record<string, ComponentType> = {
  "sort-it-arcade": dynamic(() => import("./sort-it-arcade"), {
    ssr: false,
    loading: () => <GameLoading />,
  }),
  "k-means-territory-wars": dynamic(
    () => import("./k-means-territory-wars"),
    { ssr: false, loading: () => <GameLoading /> },
  ),
  "data-detox": dynamic(() => import("./data-detox"), {
    ssr: false,
    loading: () => <GameLoading />,
  }),
  "gradient-descent-skier": dynamic(
    () => import("./gradient-descent-skier"),
    { ssr: false, loading: () => <GameLoading /> },
  ),
  "neuron-forge": dynamic(() => import("./neuron-forge"), {
    ssr: false,
    loading: () => <GameLoading />,
  }),
  "overfit-tower-defense": dynamic(
    () => import("./overfit-tower-defense"),
    { ssr: false, loading: () => <GameLoading /> },
  ),
  "confusion-matrix-chef": dynamic(() => import("./confusion-matrix-chef"), {
    ssr: false,
    loading: () => <GameLoading />,
  }),
  "decision-tree-architect": dynamic(
    () => import("./decision-tree-architect"),
    { ssr: false, loading: () => <GameLoading /> },
  ),
  "hyperparameter-heist": dynamic(() => import("./hyperparameter-heist"), {
    ssr: false,
    loading: () => <GameLoading />,
  }),
  "feature-forge": dynamic(() => import("./feature-forge"), {
    ssr: false,
    loading: () => <GameLoading />,
  }),
  "agent-academy": dynamic(() => import("./agent-academy"), {
    ssr: false,
    loading: () => <GameLoading />,
  }),
  "convolution-kitchen": dynamic(() => import("./convolution-kitchen"), {
    ssr: false,
    loading: () => <GameLoading />,
  }),
  "backprop-blitz": dynamic(() => import("./backprop-blitz"), {
    ssr: false,
    loading: () => <GameLoading />,
  }),
  "dimension-diver": dynamic(() => import("./dimension-diver"), {
    ssr: false,
    loading: () => <GameLoading />,
  }),
};

/** Slugs this module can actually mount. Compared against the registry in tests. */
export const MOUNTED_SLUGS = Object.keys(GAME_COMPONENTS);

/**
 * What stands in for a game while its chunk downloads. It is server-rendered
 * too, so it is marked `data-game-loading` for the play page's <noscript>
 * stylesheet to hide: with scripting off the chunk never comes, and a
 * viewport-tall "Loading game…" status under the page's "this needs
 * JavaScript" explanation would promise a game that can't arrive.
 */
export function GameLoading() {
  return (
    <div
      role="status"
      data-game-loading=""
      className="flex min-h-dvh items-center justify-center bg-bg text-text-muted"
    >
      Loading game…
    </div>
  );
}

// ── Crash recovery ─────────────────────────────────────────────────────────

interface GameErrorBoundaryProps {
  slug: string;
  children: ReactNode;
}

interface GameErrorBoundaryState {
  error: Error | null;
  /** Bumped by "Try again" to remount the game from scratch. */
  attempt: number;
}

/**
 * One boundary per mounted game.
 *
 * A class because React still has no hook for catching render errors. Three
 * layers, innermost first: GameShell's own `LaneErrorBoundary` catches a crash
 * inside the active lane and keeps the shell (metric, controls, lane toggle)
 * up; this one catches everything else in a game — its controls, the shell
 * itself, or a chunk that never arrived, before any shell existed — and it has
 * to live out here because a shell cannot catch its own crash; and route-level
 * `src/app/error.tsx` backs both up for anything outside a game.
 */
export class GameErrorBoundary extends Component<
  GameErrorBoundaryProps,
  GameErrorBoundaryState
> {
  state: GameErrorBoundaryState = { error: null, attempt: 0 };

  static getDerivedStateFromError(
    thrown: unknown,
  ): Partial<GameErrorBoundaryState> {
    return {
      error: thrown instanceof Error ? thrown : new Error(String(thrown)),
    };
  }

  componentDidCatch(error: unknown, info: ErrorInfo): void {
    // Never swallowed: the panel explains it to the player, and this is how it
    // reaches a developer (and any error reporting wired to the console).
    console.error(
      `[GameML] "${this.props.slug}" crashed${isChunkLoadError(error) ? " (a game chunk failed to load)" : ""}:`,
      error,
      info.componentStack ?? "",
    );
  }

  private retry = () => {
    this.setState((state) => ({ error: null, attempt: state.attempt + 1 }));
  };

  render() {
    const { error, attempt } = this.state;
    if (error) {
      return (
        <GameCrashPanel
          slug={this.props.slug}
          chunkFailed={isChunkLoadError(error)}
          onRetry={this.retry}
        />
      );
    }
    // Keyed on the attempt, so "Try again" remounts the whole subtree —
    // effects, canvases and TF.js models start fresh — rather than re-rendering
    // into whatever half-initialised state threw.
    return <Fragment key={attempt}>{this.props.children}</Fragment>;
  }
}

const CHUNK_FAILED_MESSAGE =
  "Part of the game couldn't be fetched. That usually means the connection dropped, or the site was updated while this tab was open. Reloading the page fetches a fresh copy.";

/**
 * Honest about what each button does. A game's state lives in its module-level
 * store, which a remount does not touch, so "Try again" redraws the game where
 * the player left it; only a reload starts the game over. Earned XP and stars
 * survive both, because progression is saved to localStorage.
 */
const CRASHED_MESSAGE =
  "Something in the game broke — that's our bug, not your mistake. Try again to redraw it where you left off. If it breaks again, reloading the page starts the game fresh; your XP and stars are kept.";

function GameCrashPanel({
  slug,
  chunkFailed,
  onRetry,
}: {
  slug: string;
  chunkFailed: boolean;
  onRetry: () => void;
}) {
  const title = getGameMeta(slug)?.title ?? "The game";
  const headingRef = useRef<HTMLHeadingElement>(null);

  // The control the player was using has just vanished along with the game, so
  // focus would otherwise fall back to <body>. Moving it to the heading tells a
  // screen-reader user what happened and starts the next Tab at the recovery
  // actions.
  useEffect(() => {
    headingRef.current?.focus();
  }, []);

  return (
    <div className="flex min-h-dvh items-center justify-center bg-bg px-4 py-12">
      <main
        aria-labelledby="game-crash-heading"
        className="w-full max-w-lg rounded-md border border-border bg-surface p-6"
      >
        {/* The game's name is in the heading, not only in an eyebrow above it,
            because the heading is what receives focus and gets read out. */}
        <h1
          id="game-crash-heading"
          ref={headingRef}
          tabIndex={-1}
          className="font-display text-xl font-semibold"
        >
          {chunkFailed
            ? `${title} didn't finish downloading`
            : `${title} hit an unexpected error`}
        </h1>
        <p className="mt-3 leading-relaxed text-text-muted">
          {chunkFailed ? CHUNK_FAILED_MESSAGE : CRASHED_MESSAGE}
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          {chunkFailed ? null : (
            <Button
              variant="primary"
              onClick={onRetry}
              icon={<RotateCcw className="size-4" />}
            >
              Try again
            </Button>
          )}
          <Button
            variant={chunkFailed ? "primary" : "secondary"}
            onClick={() => window.location.reload()}
            icon={<RefreshCw className="size-4" />}
          >
            Reload the page
          </Button>
          <Link
            href="/"
            className="inline-flex min-h-11 items-center justify-center gap-2 rounded-md border border-border bg-surface-2 px-4 font-medium text-text transition-[border-color] dur-micro hover:border-primary/60"
          >
            <ArrowLeft aria-hidden="true" className="size-4" />
            Back to all games
          </Link>
        </div>
      </main>
    </div>
  );
}

function GameBySlug({ slug }: { slug: string }) {
  const Game = GAME_COMPONENTS[slug];

  // Unreachable by design, and kept only because the index signature is
  // optional: `/play/[slug]` 404s anything outside the registry, and
  // registry.test.ts asserts MOUNTED_SLUGS and playableSlugs() are the same
  // set. Reaching here means those two drifted apart, which is a bug in the
  // wiring rather than a state to render an apology for — so it throws, and
  // the boundary around it reports it like any other crash.
  if (!Game) {
    throw new Error(
      `GameMount: no module registered for "${slug}". MOUNTED_SLUGS and the registry have drifted.`,
    );
  }

  return <Game />;
}

export function GameMount({ slug }: { slug: string }) {
  return (
    // Keyed on the slug so moving between games never carries one game's
    // crash, or its retry count, into the next.
    <GameErrorBoundary key={slug} slug={slug}>
      <GameBySlug slug={slug} />
    </GameErrorBoundary>
  );
}
