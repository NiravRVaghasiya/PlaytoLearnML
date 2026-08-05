/**
 * Shared UI primitives (DESIGN.md §6).
 *
 * Games import from here. Never copy one of these into `src/games/**` and tweak
 * it — that's the forking CLAUDE.md prohibits. If a game needs different
 * behaviour, add a prop here and document it in DESIGN.md.
 */

export { Button, type ButtonProps, type ButtonVariant, type ButtonSize } from "./Button";
export { CodeBlock, type CodeBlockProps } from "./CodeBlock";
export { CodeEditor, type CodeEditorProps } from "./CodeEditor";
export { DatasetChip, type DatasetChipProps } from "./DatasetChip";
export { Dial, type DialProps } from "./Dial";
export { LaneToggle, type LaneToggleProps } from "./LaneToggle";
export { MathDrawer, type MathDrawerProps } from "./MathDrawer";
export {
  MetricReadout,
  formatMetric,
  metricAnnouncement,
  type MetricReadoutProps,
  type MetricSpec,
  type MetricFormat,
  type MetricDirection,
  type MetricState,
} from "./MetricReadout";
export {
  Slider,
  fromPosition,
  toPosition,
  type SliderProps,
  type ControlScale,
} from "./Slider";
export { StarRating, type StarRatingProps } from "./StarRating";
export { WhyCard, type WhyCardProps, type WhyCardContent, type WhyTone } from "./WhyCard";
export { XPBar, type XPBarProps } from "./XPBar";
export { useAnimatedNumber } from "./useAnimatedNumber";
