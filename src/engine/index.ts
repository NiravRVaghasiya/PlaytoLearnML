/**
 * The shared GameML engine (CLAUDE.md: "built once and reused").
 *
 * Games import from `@/engine` and `@/components`. Nothing in here knows about
 * a specific game, and nothing in `src/games/**` may re-implement it.
 */

export { GameShell, type GameShellProps, type GameShellProgress } from "./GameShell";
export {
  useModel,
  type UseModelApi,
  type UseModelOptions,
  type ModelStatus,
  type EpochMetrics,
  type TrainRequest,
} from "./useModel";
export {
  useCodeLane,
  createJsExecutor,
  createPyodideExecutor,
  CodeLaneTimeoutError,
  type UseCodeLaneApi,
  type UseCodeLaneOptions,
  type CodeExecutor,
  type CodeRunContext,
  type CodeRunLog,
  type CodeLanguage,
} from "./useCodeLane";
export {
  useProgression,
  applyResult,
  xpForScore,
  xpToClearLevel,
  levelFromXp,
  starsFor,
  isUnlocked,
  unlockedSlugs,
  badgeLabel,
  createLocalAdapter,
  createSupabaseAdapter,
  createMemoryAdapter,
  BASE_XP,
  CODE_LANE_MULTIPLIER,
  HIGH_SCORE_THRESHOLD,
  CONCEPT_BADGES,
  EMPTY_PROGRESSION,
  STORAGE_KEY,
  type ProgressionState,
  type ProgressionStore,
  type ProgressionAdapter,
  type GameProgress,
  type GameResult,
  type AppliedResult,
  type LevelInfo,
} from "./progression";
export type { Lane, MathReveal, NamedFailure, StarCount } from "./types";
