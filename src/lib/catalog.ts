/**
 * The GameML catalog — transcribed from `docs/GameML_Build_Spec.md` §2/§3.
 *
 * This is metadata only: no gameplay, no ML. It exists so the home map, the
 * progression service (unlock graph), and the games index all read the same
 * roster instead of each hardcoding its own list.
 *
 * The spec is the source of truth. If you change a phase or a concept here,
 * change it in the spec first.
 */

export type Category =
  | "Data & Preprocessing"
  | "Supervised Learning"
  | "Neural Networks"
  | "Optimization & Tuning"
  | "Unsupervised Learning"
  | "Reinforcement Learning";

export type Difficulty = "Beginner" | "Intermediate" | "Advanced";
export type Complexity = "Easy" | "Medium" | "Hard";
export type Phase = 1 | 2 | 3 | 4;

export interface GameMeta {
  /** kebab-case folder name under `src/games/` and the URL segment. */
  slug: string;
  title: string;
  category: Category;
  /** The specific ML idea taught (spec: "Concept"). */
  concept: string;
  /** The one sentence the learner must walk away with (pedagogy contract #1). */
  coreIntuition: string;
  /** Short label for the always-visible live metric (pedagogy contract #3). */
  metricLabel: string;
  /** The NAMED ML failure surfaced on loss (pedagogy contract #4). */
  failureMode: string;
  difficulty: Difficulty;
  complexity: Complexity;
  phase: Phase;
  /** Slugs that must be completed before this unlocks. Empty = always open. */
  requires: string[];
}

export const PHASES: readonly Phase[] = [1, 2, 3, 4] as const;

export const GAME_CATALOG: readonly GameMeta[] = [
  // --- Phase 1 (MVP) -------------------------------------------------------
  {
    slug: "sort-it-arcade",
    title: "Sort-It Arcade",
    category: "Supervised Learning",
    concept: "Binary/multiclass classification & decision boundaries",
    coreIntuition:
      "A classifier draws a boundary; simpler boundaries that misclassify a few points often generalize better than perfect-but-complex ones.",
    metricLabel: "Accuracy",
    failureMode: "Overfit boundary",
    difficulty: "Beginner",
    complexity: "Easy",
    phase: 1,
    requires: [],
  },
  {
    slug: "k-means-territory-wars",
    title: "K-Means Territory Wars",
    category: "Unsupervised Learning",
    concept: "K-means clustering, centroids, convergence, choosing k",
    coreIntuition:
      "Clustering finds structure without labels; results depend heavily on k and initialization, and centroids converge by alternating assign/update steps.",
    metricLabel: "Inertia",
    failureMode: "Bad k / poor initialization",
    difficulty: "Intermediate",
    complexity: "Easy",
    phase: 1,
    requires: ["sort-it-arcade"],
  },
  {
    slug: "data-detox",
    title: "Data Detox",
    category: "Data & Preprocessing",
    concept: "Data cleaning — missing values, outliers, normalization",
    coreIntuition:
      "Garbage in = garbage out; cleaning decisions directly move model quality, and there is no single right answer — only tradeoffs.",
    metricLabel: "Accuracy",
    failureMode: "Data starvation",
    difficulty: "Beginner",
    complexity: "Easy",
    phase: 1,
    requires: [],
  },
  {
    slug: "gradient-descent-skier",
    title: "Gradient Descent Skier",
    category: "Optimization & Tuning",
    concept: "Gradient descent, learning rate, local vs. global minima",
    coreIntuition:
      "Optimization is a step-size balancing act; learning rate is the single most consequential hyperparameter, and momentum helps escape traps.",
    metricLabel: "Loss",
    failureMode: "Divergence",
    difficulty: "Beginner",
    complexity: "Medium",
    phase: 1,
    requires: [],
  },

  // --- Phase 2 -------------------------------------------------------------
  {
    slug: "neuron-forge",
    title: "Neuron Forge",
    category: "Neural Networks",
    concept: "Network architecture — layers, neurons, activations",
    coreIntuition:
      "More capacity isn't free; architecture determines what patterns a network can even represent.",
    metricLabel: "Loss",
    failureMode: "Insufficient capacity",
    difficulty: "Intermediate",
    complexity: "Medium",
    phase: 2,
    requires: ["sort-it-arcade", "gradient-descent-skier"],
  },
  {
    slug: "overfit-tower-defense",
    title: "Overfit Tower Defense",
    category: "Optimization & Tuning",
    concept: "Bias–variance tradeoff, regularization, overfitting",
    coreIntuition:
      "Generalization is a moving target between two failure modes; regularization trades a little training accuracy for a lot of robustness.",
    metricLabel: "Train/val gap",
    failureMode: "Overfitting / underfitting",
    difficulty: "Intermediate",
    complexity: "Medium",
    phase: 2,
    requires: ["sort-it-arcade"],
  },
  {
    slug: "confusion-matrix-chef",
    title: "Confusion Matrix Chef",
    category: "Supervised Learning",
    concept: "Evaluation metrics — precision, recall, F1, threshold tuning",
    coreIntuition:
      "Accuracy lies on imbalanced data; the right metric depends on the cost of each error type.",
    metricLabel: "F1",
    failureMode: "Accuracy paradox",
    difficulty: "Intermediate",
    complexity: "Easy",
    phase: 2,
    requires: ["sort-it-arcade"],
  },
  {
    slug: "decision-tree-architect",
    title: "Decision Tree Architect",
    category: "Supervised Learning",
    concept: "Decision trees, information gain, splitting",
    coreIntuition:
      "Trees split to reduce impurity; depth raises training accuracy but eventually hurts validation.",
    metricLabel: "Validation accuracy",
    failureMode: "Overfit depth",
    difficulty: "Intermediate",
    complexity: "Medium",
    phase: 2,
    requires: ["sort-it-arcade"],
  },

  // --- Phase 3 -------------------------------------------------------------
  {
    slug: "hyperparameter-heist",
    title: "Hyperparameter Heist",
    category: "Optimization & Tuning",
    concept: "Hyperparameter tuning — grid vs. random vs. Bayesian search",
    coreIntuition:
      "Search strategy matters under a compute budget; random and Bayesian search beat brute-force grid search in high dimensions.",
    metricLabel: "Best objective",
    failureMode: "Budget exhausted",
    difficulty: "Advanced",
    complexity: "Medium",
    phase: 3,
    requires: ["gradient-descent-skier"],
  },
  {
    slug: "feature-forge",
    title: "Feature Forge",
    category: "Data & Preprocessing",
    concept: "Feature engineering & encoding (one-hot, binning, scaling)",
    coreIntuition:
      "Good features often beat fancier models; representation matters more than algorithm choice for tabular data.",
    metricLabel: "Metric lift",
    failureMode: "Leakage / no lift",
    difficulty: "Intermediate",
    complexity: "Medium",
    phase: 3,
    requires: ["data-detox"],
  },
  {
    slug: "agent-academy",
    title: "Agent Academy",
    category: "Reinforcement Learning",
    concept: "RL — rewards, exploration vs. exploitation, Q-learning",
    coreIntuition:
      "You shape behavior through rewards, not direct control; poorly-specified rewards produce reward hacking.",
    metricLabel: "Episode reward",
    failureMode: "Reward hacking",
    difficulty: "Advanced",
    complexity: "Hard",
    phase: 3,
    requires: ["gradient-descent-skier"],
  },
  {
    slug: "convolution-kitchen",
    title: "Convolution Kitchen",
    category: "Neural Networks",
    concept: "CNNs — filters, feature maps, pooling",
    coreIntuition:
      "CNNs learn hierarchical features (edges → textures → parts → objects); convolution is a sliding pattern-matcher.",
    metricLabel: "Detection score",
    failureMode: "Dead filters",
    difficulty: "Advanced",
    complexity: "Medium",
    phase: 3,
    requires: ["neuron-forge"],
  },

  // --- Phase 4 -------------------------------------------------------------
  {
    slug: "backprop-blitz",
    title: "Backprop Blitz",
    category: "Neural Networks",
    concept: "Backpropagation & the chain rule",
    coreIntuition:
      "Learning = error flowing backward to assign blame to each weight; gradients are how much you contributed to the mistake.",
    metricLabel: "Gradient correctness",
    failureMode: "Broken chain rule",
    difficulty: "Advanced",
    complexity: "Hard",
    phase: 4,
    requires: ["neuron-forge"],
  },
  {
    slug: "dimension-diver",
    title: "Dimension Diver",
    category: "Unsupervised Learning",
    concept: "Dimensionality reduction — PCA / t-SNE / UMAP intuition",
    coreIntuition:
      "High-dimensional data often compresses to a few meaningful axes; PCA finds the projection preserving the most variance.",
    metricLabel: "Variance retained",
    failureMode: "Lost variance",
    difficulty: "Advanced",
    complexity: "Hard",
    phase: 4,
    requires: ["k-means-territory-wars"],
  },
] as const;

/** Slugs shipped in the Phase-1 MVP, in build order (spec §7). */
export const PHASE_1_BUILD_ORDER = [
  "sort-it-arcade",
  "k-means-territory-wars",
  "data-detox",
  "gradient-descent-skier",
] as const;

export function getGameMeta(slug: string): GameMeta | undefined {
  return GAME_CATALOG.find((g) => g.slug === slug);
}
