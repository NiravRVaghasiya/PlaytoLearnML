# 🎮 GameML — Master Build Specification

**An interactive website that teaches Machine Learning through games and hands-on challenges.**

- **Version:** 1.0 (Build Spec)
- **Scope:** All 14 games across 6 ML topic categories
- **Core principle:** Every game ships in **two lanes** — a *no-code visual lane* (drag/click/slider) and a *code lane* (editable TensorFlow.js / Pyodide snippet driving the same simulation). Explorers play visual; Practitioners toggle to code.
- **Cost stance:** All ML runs **client-side** (TensorFlow.js + Pyodide). MVP infra = static hosting + free-tier auth/DB. No GPUs, no inference servers.

---

## 0. How to Read This Spec

Each game entry is written to be handed directly to an engineer and contains:

| Field | Meaning |
|---|---|
| **Concept** | The specific ML idea taught |
| **Core intuition** | The one sentence the learner must walk away with |
| **Player action = algorithm** | The mechanic-to-algorithm mapping (the pedagogy contract) |
| **Gameplay loop** | Moment-to-moment interaction |
| **Win / lose / scoring** | Success criteria and the named failure mode |
| **Live feedback** | The always-visible signal that creates the "aha" |
| **Data model** | Key state objects (pseudo-schema) |
| **Components** | UI/React component breakdown |
| **ML logic** | What computation runs under the hood |
| **Tech** | Libraries |
| **Difficulty / Complexity / Phase** | Audience level, build effort, roadmap phase |

---

## 1. Target Audience Segments

| Segment | Profile | Design Priority | Kills Engagement |
|---|---|---|---|
| **Explorers** (Complete beginners) | No code, curious, math-averse | Zero-friction, visual, immediate cause→effect | Jargon, formulas up front |
| **Builders** (Intermediate) | Some Python/JS, want intuition + practice | Sandbox tuning, "why did it break?" | Hand-holding, no code access |
| **Practitioners** (Advanced) | Know basics, want depth/edge cases | Real datasets, adversarial scenarios, leaderboards | Oversimplification, no math reveal |

**Two-lane rule** satisfies "both no-code and code-based" inside every module rather than splitting the site.

---

## 2. The Game Catalog

Ordered **beginner → advanced**. Six categories, 14 games (+1 bonus).

---

### 🎯 Category 1 — Data & Preprocessing

---

#### Game 1 — Data Detox
- **Concept:** Data cleaning — missing values, outliers, normalization
- **Core intuition:** Garbage in = garbage out; cleaning decisions directly move model quality, and there's no single "right" answer — only tradeoffs.
- **Player action = algorithm:** Sorting each dirty row into *Impute / Drop / Cap / Keep* = the analyst's preprocessing pipeline; a downstream model retrains on the cleaned data.
- **Gameplay loop:** A messy dataset flows down a conveyor belt as rows (nulls glow red, outliers oversized). Player drags each row into a bin. Every N rows, a downstream model retrains and the accuracy bar animates.
- **Win / lose / scoring:** Score = final model accuracy − time penalty. "Lose" = accuracy collapses (e.g. dropping too many rows starves the model).
- **Live feedback:** Accuracy bar reacts within 1–2s of each batch of cleaning choices.
- **Data model:**
  ```
  Row { id, features:{}, isNull:bool, isOutlier:bool, playerAction:enum }
  GameState { rows[], modelAccuracy, timeElapsed, binCounts{} }
  ```
- **Components:** `<ConveyorBelt>`, `<DirtyRow>`, `<CleaningBin>` (x4), `<AccuracyMeter>`, `<WhyCard>`
- **ML logic:** Small decision tree in TF.js retrained on the cleaned subset; accuracy on a held-out set drives the meter.
- **Tech:** React + TensorFlow.js
- **Difficulty:** Beginner · **Complexity:** Easy · **Phase 1**

---

#### Game 2 — Feature Forge
- **Concept:** Feature engineering & encoding (one-hot, binning, scaling)
- **Core intuition:** Good features often beat fancier models; representation matters more than algorithm choice for tabular data.
- **Player action = algorithm:** Combining/transforming raw columns = feature engineering; the model is fixed so only representation changes the score.
- **Gameplay loop:** Player is a blacksmith. Drag raw columns into a "forge" to create features: `date → day_of_week`, bin `age`, scale `income`. Each new feature triggers a retrain; some combos forge a glowing "legendary feature."
- **Win / lose / scoring:** Score = target metric lift over baseline features. "Legendary" combos give bonus multipliers.
- **Live feedback:** Metric gauge moves per forged feature; feature-importance bars re-rank live.
- **Data model:**
  ```
  Column { name, dtype, values[] }
  Feature { id, sourceCols[], transform:enum, importance }
  GameState { baselineScore, currentScore, features[] }
  ```
- **Components:** `<ColumnTray>`, `<ForgeSlot>`, `<TransformPicker>`, `<MetricGauge>`, `<ImportanceBars>`
- **ML logic:** Fixed model (logistic reg / small tree) in TF.js; Pyodide runs real pandas transforms in the code lane.
- **Tech:** React + Pyodide (pandas) + TensorFlow.js
- **Difficulty:** Intermediate · **Complexity:** Medium · **Phase 3**

---

### 🎯 Category 2 — Supervised Learning (Classification & Regression)

---

#### Game 3 — Sort-It Arcade
- **Concept:** Binary/multiclass classification & decision boundaries
- **Core intuition:** A classifier draws a boundary; simpler boundaries that misclassify a few points often generalize better than perfect-but-complex ones.
- **Player action = algorithm:** Drawing/adjusting the boundary = fitting a classifier; the complexity penalty = regularization pressure.
- **Gameplay loop:** Two-color points rain onto a 2D field. Player drags a boundary — first a line, later a curve, then a freeform wiggle. Score updates live as points land on each side.
- **Win / lose / scoring:** Score = accuracy − complexity penalty. The tempting wiggly line that nails training points quietly loses points → foreshadows overfitting.
- **Live feedback:** Misclassified points flash; accuracy % and complexity cost shown side by side.
- **Data model:**
  ```
  Point { x, y, label, predictedSide }
  Boundary { type:enum, params[], complexityCost }
  GameState { points[], accuracy, penalty, score }
  ```
- **Components:** `<PointCanvas>` (D3/p5), `<BoundaryTool>`, `<ScorePanel>`, `<ComplexityMeter>`
- **ML logic:** Geometric point-vs-boundary classification; complexity = param count or curvature integral.
- **Tech:** D3.js or p5.js
- **Difficulty:** Beginner · **Complexity:** Easy · **Phase 1**

---

#### Game 4 — Decision Tree Architect
- **Concept:** Decision trees, information gain, splitting
- **Core intuition:** Trees split to reduce impurity; depth raises training accuracy but eventually hurts validation — bias/variance made tactile.
- **Player action = algorithm:** Dragging feature-threshold "gates" = choosing splits; the game computes the resulting impurity just like a real tree learner.
- **Gameplay loop:** Player builds a tree node-by-node by dragging feature+threshold gates. Each split shows resulting purity (Gini/entropy) as a color-coded pie. Adding too many splits summons an "overfit ghost" as validation accuracy dips.
- **Win / lose / scoring:** Score rewards high validation accuracy at low depth. "Lose" = validation accuracy drops while training climbs.
- **Live feedback:** Per-node purity pies; dual train/validation accuracy bars.
- **Data model:**
  ```
  TreeNode { id, feature, threshold, gini, left, right, samples[] }
  GameState { tree, trainAcc, valAcc, depth }
  ```
- **Components:** `<TreeCanvas>` (React Flow), `<SplitGate>`, `<PurityPie>`, `<AccuracyPair>`, `<OverfitGhost>`
- **ML logic:** Real impurity computation per split; TF.js decision-forest for reference scoring.
- **Tech:** React Flow + TensorFlow.js
- **Difficulty:** Intermediate · **Complexity:** Medium · **Phase 2**

---

#### Game 5 — Confusion Matrix Chef
- **Concept:** Evaluation metrics — precision, recall, F1, threshold tuning
- **Core intuition:** Accuracy lies on imbalanced data; the right metric depends on the cost of each error type.
- **Player action = algorithm:** Moving the threshold slider = setting the classifier's decision cutoff; the confusion matrix recomputes exactly as it would in production.
- **Gameplay loop:** Player runs a restaurant filtering "spam" orders. A slider sets the decision threshold; the live confusion matrix updates as orders are served/rejected. A "critic" issues scenarios: "high recall for cancer screening" vs. "high precision for spam."
- **Win / lose / scoring:** Each scenario has a target metric; hit the target band to win the round.
- **Live feedback:** Live 2×2 confusion matrix + ROC curve point that slides with the threshold.
- **Data model:**
  ```
  Sample { score, trueLabel }
  GameState { threshold, TP, FP, TN, FN, scenarioTarget }
  ```
- **Components:** `<ThresholdSlider>`, `<ConfusionMatrix>`, `<ROCCurve>`, `<CriticScenario>`, `<MetricReadout>`
- **ML logic:** Pre-scored samples; metrics recomputed on threshold change (no retrain needed).
- **Tech:** React + Highcharts/D3
- **Difficulty:** Intermediate · **Complexity:** Easy · **Phase 2**

---

### 🎯 Category 3 — Neural Networks & Deep Learning

---

#### Game 6 — Neuron Forge
- **Concept:** Network architecture — layers, neurons, activations
- **Core intuition:** More capacity isn't free; architecture (depth, width, activation) determines what patterns a network can even represent.
- **Player action = algorithm:** Adding neurons/layers = defining model architecture; a real network then trains on the chosen puzzle.
- **Gameplay loop:** A TensorFlow-Playground-style remix. Drag neurons and layers to solve escalating pattern puzzles (linear → circle → spiral → XOR). A **compute budget** caps total neurons, forcing efficient designs.
- **Win / lose / scoring:** Solve the pattern (loss below threshold) under the neuron budget. Efficiency bonus for fewer neurons.
- **Live feedback:** Live decision-surface heatmap + loss curve while training animates.
- **Data model:**
  ```
  Layer { neurons:int, activation:enum }
  Architecture { layers[], totalNeurons }
  GameState { arch, loss, epoch, budget, patternId }
  ```
- **Components:** `<NetworkEditor>` (React Flow), `<NeuronNode>`, `<DecisionSurface>`, `<LossCurve>`, `<BudgetBar>`
- **ML logic:** Real TF.js sequential model trained in-browser on 2D toy datasets.
- **Tech:** TensorFlow.js + React Flow *(inspired by TensorFlow Playground)*
- **Difficulty:** Intermediate · **Complexity:** Medium · **Phase 2**

---

#### Game 7 — Backprop Blitz
- **Concept:** Backpropagation & the chain rule
- **Core intuition:** Learning = error flowing backward to assign blame to each weight; gradients are just "how much did you contribute to the mistake."
- **Player action = algorithm:** Routing the error signal backward and splitting gradients at branches = executing the chain rule by hand.
- **Gameplay loop:** Error appears at the output and must be pushed backward through the network. The player routes the error through each node, choosing how gradients split at branches (pinball/chain-reaction feel). Correct routing lights the path green and the weights visibly nudge.
- **Win / lose / scoring:** Score = correctness of gradient routing per node vs. the true autograd trace.
- **Live feedback:** Path lights green/red; weight deltas animate; a running "gradient correctness" %.
- **Data model:**
  ```
  Node { id, value, localGrad, incomingGrad, outgoingGrad }
  Edge { from, to, weight }
  GameState { graph, playerRouting[], trueGrads[], correctness }
  ```
- **Components:** `<ComputeGraph>`, `<GradientPacket>`, `<NodeInspector>`, `<CorrectnessMeter>`
- **ML logic:** Real forward pass + autograd trace as ground truth; player routing diffed against it.
- **Tech:** p5.js/WebGL over a real autograd trace
- **Difficulty:** Advanced · **Complexity:** Hard · **Phase 4**

---

#### Game 8 — Convolution Kitchen
- **Concept:** CNNs — filters, feature maps, pooling
- **Core intuition:** CNNs learn hierarchical features (edges → textures → parts → objects); convolution is a sliding pattern-matcher.
- **Player action = algorithm:** Designing kernels and stacking them = defining conv layers; sliding the kernel = the convolution operation itself.
- **Gameplay loop:** Player designs image "stamps" (kernels) and slides them over pictures to hunt edges, textures, shapes. Stack filters to detect increasingly complex objects; a live feature-map gallery shows what each layer "sees."
- **Win / lose / scoring:** Complete detection challenges (find edges → find the cat) with the fewest/cleanest filters.
- **Live feedback:** Real-time feature-map gallery per layer; activation heatmaps.
- **Data model:**
  ```
  Kernel { weights[3x3], stride, padding }
  Layer { kernels[], poolType }
  GameState { layers[], inputImage, featureMaps[][] }
  ```
- **Components:** `<KernelDesigner>`, `<ImageCanvas>`, `<FeatureMapGallery>`, `<LayerStack>`
- **ML logic:** TF.js conv ops on canvas image data; optional webcam input.
- **Tech:** TensorFlow.js + canvas *(webcam optional, à la Teachable Machine)*
- **Difficulty:** Advanced · **Complexity:** Medium · **Phase 3**

---

### 🎯 Category 4 — Optimization & Model Tuning

---

#### Game 9 — Gradient Descent Skier
- **Concept:** Gradient descent, learning rate, local vs. global minima
- **Core intuition:** Optimization is a step-size balancing act; learning rate is the single most consequential hyperparameter, and momentum helps escape traps.
- **Player action = algorithm:** Setting the learning-rate dial = choosing step size; each frame the skier takes a gradient step down the loss surface.
- **Gameplay loop:** Ski down a 3D loss landscape. One dial controls learning rate: too small = never reach bottom before the timer; too big = launch off the mountain (divergence). Momentum power-ups blast through local-minima dips.
- **Win / lose / scoring:** Reach the global minimum before the timer. "Lose" = divergence (fly off) or stuck in a local minimum.
- **Live feedback:** Loss number drops as you descend, spikes red on overshoot; trail shows the optimization path.
- **Data model:**
  ```
  Surface { fn(x,y) → loss, minima[] }
  Skier { pos{x,y}, velocity, learningRate, momentum }
  GameState { surface, skier, currentLoss, timer }
  ```
- **Components:** `<LossTerrain>` (Three.js), `<LearningRateDial>`, `<MomentumPowerup>`, `<LossReadout>`, `<PathTrail>`
- **ML logic:** Real gradient of a chosen 2D loss surface; skier position updated by GD step each frame.
- **Tech:** Three.js + real gradient computation
- **Difficulty:** Beginner→Intermediate · **Complexity:** Medium · **Phase 1**

---

#### Game 10 — Overfit Tower Defense
- **Concept:** Bias–variance tradeoff, regularization, overfitting/underfitting
- **Core intuition:** Generalization is a moving target between two failure modes; regularization trades a little training accuracy for a lot of robustness.
- **Player action = algorithm:** Placing regularization towers (L1/L2, dropout) and setting model complexity = tuning the bias-variance knob in real time.
- **Gameplay loop:** Waves of "underfit" enemies (too simple) and "overfit" enemies (memorizers) attack your generalization core. Defend by tuning model complexity + dropping regularization towers. Over-defend and you starve the model into underfitting.
- **Win / lose / scoring:** Survive waves = maintain a small train/val gap. "Lose" = core destroyed by whichever failure mode you neglected.
- **Live feedback:** Live train vs. validation accuracy gap bar; enemies visibly weaken as the gap closes.
- **Data model:**
  ```
  Enemy { type:'underfit'|'overfit', strength }
  Tower { type:'L1'|'L2'|'dropout', strength }
  GameState { modelComplexity, towers[], trainAcc, valAcc, wave, coreHP }
  ```
- **Components:** `<GameCanvas>` (Phaser), `<ComplexitySlider>`, `<RegTowerPalette>`, `<GapMeter>`, `<WaveTracker>`
- **ML logic:** TF.js model whose complexity/regularization the player controls; train/val gap scores the wave.
- **Tech:** Phaser.js + TensorFlow.js
- **Difficulty:** Intermediate · **Complexity:** Medium · **Phase 2**

---

#### Game 11 — Hyperparameter Heist
- **Concept:** Hyperparameter tuning — grid vs. random vs. Bayesian search
- **Core intuition:** Search strategy matters under a compute budget; random and Bayesian search beat brute-force grid search in high dimensions.
- **Player action = algorithm:** Choosing which dial combos to "try" under a limited budget = executing a search strategy; the Bayesian hint mode surfaces the acquisition-function suggestion.
- **Gameplay loop:** A safecracking game — each dial is a hyperparameter, with limited "tries" (compute budget). Grid = mechanically trying every combo (slow); random finds the safe faster; Bayesian "hint" mode learns from prior cracks to suggest the next dial.
- **Win / lose / scoring:** Crack the safe (find the optimum region) within the try budget; fewer tries = higher score.
- **Live feedback:** Search-trajectory viz over the objective surface; "temperature" hot/cold proximity readout.
- **Data model:**
  ```
  HyperParam { name, range, currentValue }
  Trial { params{}, objectiveValue }
  GameState { params[], trials[], budget, mode, best }
  ```
- **Components:** `<SafeDials>`, `<ObjectiveSurface>` (D3), `<TrajectoryTrail>`, `<BudgetCounter>`, `<StrategyToggle>`
- **ML logic:** Real objective surface in TF.js; grid/random/Bayesian (GP or TPE-lite) sampling.
- **Tech:** React + TensorFlow.js + D3
- **Difficulty:** Intermediate→Advanced · **Complexity:** Medium · **Phase 3**

---

### 🎯 Category 5 — Unsupervised Learning

---

#### Game 12 — K-Means Territory Wars
- **Concept:** K-means clustering, centroids, convergence, choosing *k*
- **Core intuition:** Clustering finds structure without labels; results depend heavily on *k* and initialization, and centroids converge by alternating assign/update steps.
- **Player action = algorithm:** Placing *k* flags (centroids) and stepping the simulation = running the assign→update loop of k-means yourself.
- **Gameplay loop:** Player drops *k* flags on a map of scattered villages. Villages join the nearest flag; flags then auto-migrate to their village center. Player watches the iterative dance and can re-place flags. An "elbow meter" scores whether *k* is sensible.
- **Win / lose / scoring:** Minimize within-cluster distance (inertia) with a reasonable *k*; elbow meter rewards the sweet spot.
- **Live feedback:** Villages recolor by assignment each step; inertia number drops toward convergence.
- **Data model:**
  ```
  Point { x, y, clusterId }
  Centroid { x, y, id }
  GameState { points[], centroids[], k, inertia, iteration }
  ```
- **Components:** `<ClusterMap>` (D3/p5), `<CentroidFlag>`, `<StepButton>`, `<ElbowMeter>`, `<InertiaReadout>`
- **ML logic:** Pure-JS k-means (assign/update); no ML library required.
- **Tech:** D3.js / p5.js
- **Difficulty:** Intermediate · **Complexity:** Easy · **Phase 1**

---

#### Game 13 — Dimension Diver
- **Concept:** Dimensionality reduction — PCA / t-SNE / UMAP intuition
- **Core intuition:** High-dimensional data often compresses to a few meaningful axes; PCA finds the projection preserving the most variance.
- **Player action = algorithm:** Rotating the cloud to find the most separating "shadow" = discovering principal components by eye.
- **Gameplay loop:** A tangled 3D point cloud floats in space. The player rotates it to find the 2D "shadow" (projection) that best separates hidden groups. A "variance retained" gauge rewards informative angles.
- **Win / lose / scoring:** Match or beat the PCA-optimal projection's separation/variance; higher retained variance = higher score.
- **Live feedback:** Live 2D shadow updates as you rotate; variance-retained gauge; group-separation score.
- **Data model:**
  ```
  Point3D { x, y, z, groupId }
  Projection { basisVectors[2], varianceRetained }
  GameState { cloud[], projection, bestVariance }
  ```
- **Components:** `<PointCloud3D>` (Three.js), `<ShadowPlane2D>`, `<VarianceGauge>`, `<PCAHintButton>`
- **ML logic:** Real PCA (eigendecomposition/SVD) as reference; player projection scored against it.
- **Tech:** Three.js + real PCA in the code lane
- **Difficulty:** Advanced · **Complexity:** Hard · **Phase 4**

---

### 🎯 Category 6 — Reinforcement Learning

---

#### Game 14 — Agent Academy
- **Concept:** RL — rewards, exploration vs. exploitation, Q-learning
- **Core intuition:** You shape behavior through rewards, not direct control; poorly-specified rewards produce "reward hacking," and exploration is essential to avoid getting stuck.
- **Player action = algorithm:** Designing the reward function and setting exploration (ε) = defining the RL problem; the agent then learns the policy over episodes.
- **Gameplay loop:** The player does NOT control the character — they design its **reward function** and watch an agent learn a maze/grid over episodes. Set the cheese reward too high near a trap → greedy suicidal behavior; tune ε to escape bad habits. Speed-run the agent to competence.
- **Win / lose / scoring:** Agent reaches reliable competence in fewest episodes; bonus for avoiding reward-hacking pitfalls.
- **Live feedback:** Episode-by-episode reward curve; heatmap of the learned Q-values / visited states.
- **Data model:**
  ```
  Grid { cells[][], rewards{} }
  Agent { qTable{}, epsilon, policy }
  GameState { grid, agent, episode, cumulativeReward }
  ```
- **Components:** `<GridWorld>` (Phaser), `<RewardEditor>`, `<EpsilonSlider>`, `<RewardCurve>`, `<QValueHeatmap>`
- **ML logic:** Tabular Q-learning (or DQN) loop in TF.js; runs many episodes fast client-side.
- **Tech:** Phaser.js + TensorFlow.js
- **Difficulty:** Advanced · **Complexity:** Hard · **Phase 3**

---

#### Bonus — GAN Duel *(stretch)*
- **Concept:** Generative adversarial networks — generator vs. discriminator
- **Core intuition:** Two networks improve by competing; the generator learns to fool an ever-sharpening detective.
- **Gameplay loop:** Two-slider adversarial game — player alternately plays Forger (generator) and Detective (discriminator), watching fake images improve as the two compete.
- **Tech:** TensorFlow.js *(inspired by GAN Lab)* · **Advanced · Hard · Phase 4+**

---

## 3. Prioritization Matrix

Build the top-right quadrant first (high impact, low effort).

| Game | Category | Edu. Impact | Complexity | Engagement | Phase |
|---|---|---|---|---|---|
| Gradient Descent Skier | Optimization | High | Medium | 🔥🔥🔥 | **1** |
| Sort-It Arcade | Supervised | High | Easy | 🔥🔥 | **1** |
| K-Means Territory Wars | Unsupervised | High | Easy | 🔥🔥 | **1** |
| Data Detox | Preprocessing | High | Easy | 🔥🔥 | **1** |
| Neuron Forge | Neural Nets | High | Medium | 🔥🔥🔥 | **2** |
| Overfit Tower Defense | Optimization | High | Medium | 🔥🔥🔥 | **2** |
| Confusion Matrix Chef | Supervised | Medium | Easy | 🔥🔥 | **2** |
| Decision Tree Architect | Supervised | High | Medium | 🔥🔥 | **2** |
| Hyperparameter Heist | Optimization | Medium | Medium | 🔥🔥 | **3** |
| Feature Forge | Preprocessing | Medium | Medium | 🔥 | **3** |
| Agent Academy | Reinforcement | High | Hard | 🔥🔥🔥 | **3** |
| Convolution Kitchen | Neural Nets | High | Medium | 🔥🔥 | **3** |
| Backprop Blitz | Neural Nets | High | Hard | 🔥🔥 | **4** |
| Dimension Diver | Unsupervised | Medium | Hard | 🔥🔥 | **4** |

**MVP = Phase 1's four games** — they span preprocessing, classification, optimization, and clustering, cover all three audiences, and are each Easy/Medium with no expensive infra.

---

## 4. Learning Journey & Progression

**Skill-tree map, not a linear course.** Games are nodes on a visual "ML Continent"; completing one unlocks adjacent territories.

```
  [Data Island] ──▶ [Classification Coast] ──▶ [Neural Peaks]
        │                    │                      │
        ▼                    ▼                      ▼
  [Cluster Caves]     [Tuning Foundry]       [Deep Jungle: CNN/RL]
```

**Progression systems**
- **XP & Levels** — each game grants XP scaled by score and lane (code lane pays more).
- **Mastery Stars (1–3)** — ⭐ complete · ⭐⭐ high score · ⭐⭐⭐ clear the code-lane challenge.
- **Unlockables** — real datasets (Titanic, MNIST, Iris), harder scenarios, cosmetic lab themes.
- **Concept Badges** — "I understand overfitting" — shareable to LinkedIn (growth loop).

**Feedback loops**
- **Live metric on every action** — the accuracy/loss number never hides.
- **"Why did that happen?" cards** — 2-line explanation tied to the player's last action.
- **Failure is instructional** — losing shows the *named* failure mode ("You overfit — 99% train, 61% test").
- **Reveal-the-math toggle** — expand any game to see the equation + real code driving the sim.

---

## 5. Technical Architecture

| Layer | Choice | Why |
|---|---|---|
| **Framework** | React (Next.js for SEO/blog) | Component reuse across 14 games |
| **In-browser ML** | TensorFlow.js | Real training/inference client-side → zero server ML cost |
| **Python-in-browser** | Pyodide (code lane) | Real pandas/sklearn-style code for Builders/Practitioners |
| **Data viz** | D3.js + Highcharts | Boundaries, ROC, confusion matrix, search trajectories |
| **Creative canvas** | p5.js | Fast 2D interactive games |
| **3D** | Three.js | Loss landscapes, PCA point clouds |
| **Game engine** | Phaser.js | Tower defense, RL grid worlds |
| **Node editors** | React Flow | Neuron Forge, Decision Tree Architect |
| **Backend (minimal)** | Supabase/Firebase | Auth, XP, leaderboards — no ML server |

**Shared engine layer (build once, reuse everywhere):**
- `<GameShell>` — wraps every game: lane toggle (visual/code), "Reveal the Math," WhyCard host, XP hook.
- `useModel()` — TF.js model lifecycle (build/train/predict) shared across games.
- `useCodeLane()` — Pyodide/TF.js editable snippet bound to the same game state.
- `<MetricReadout>` / `<WhyCard>` — the always-visible feedback primitives.
- `progression` service — XP, stars, badges, unlocks (Supabase).

**Cost note:** All ML runs client-side → MVP infra = static hosting + free-tier auth/DB. No GPUs, no inference servers.

---

## 6. Website Structure & Navigation

```
🏠 Home / "The ML Continent" (visual skill-tree map)
│
├── 🎮 Play  → games grouped by category, with lock/unlock states
│     └── Each game: [Visual Lane] ⇄ [Code Lane] toggle + "Reveal the Math"
│
├── 📈 My Lab  → XP, badges, mastery stars, streaks, next-recommended game
│
├── 🏆 Leaderboards  → per-game high scores + weekly challenges
│
├── 📚 Concept Library  → short explainer per concept (SEO growth engine)
│
├── 👥 Classrooms  → teacher dashboards, assign games, track cohort progress
│
└── ⚙️ Sandbox  → free-play: bring your own CSV, run any game's engine
```

**First-run flow (Explorer):** Land on map → auto-start Gradient Descent Skier (30-sec no-signup demo) → hit an "aha" → prompt to save progress (signup) → skill-tree opens.

---

## 7. Build Roadmap

| Phase | Games | Goal | Est. |
|---|---|---|---|
| **1 — MVP** | Skier, Sort-It, K-Means, Data Detox | Demoable product; validate two-lane + progression | ~3–4 weeks |
| **2** | Neuron Forge, Overfit TD, Confusion Chef, Tree Architect | Depth in supervised + NN intuition | ~4–6 weeks |
| **3** | Hyperparam Heist, Feature Forge, Agent Academy, Conv Kitchen | Advanced tuning, RL, CNNs | ~6–8 weeks |
| **4** | Backprop Blitz, Dimension Diver, GAN Duel | Hard/expert modules | ~4–6 weeks |

**Phase-1 sequencing:** Build `<GameShell>` + `useModel()` + progression service first → then Sort-It Arcade (simplest, proves the loop) → K-Means → Data Detox → Gradient Descent Skier (the hero demo).

---

## 8. Monetization & Growth

**Growth loops (free, do first)**
- Shareable badge cards for LinkedIn/X.
- Concept Library as SEO magnet (ranks for "what is overfitting" → funnels into the game).
- Weekly challenge + leaderboard (recurring return).
- Embeddable single-game widgets with attribution back-links.

**Monetization (freemium, layered)**
1. **Free** — all Phase-1 games, limited datasets, capped XP.
2. **Learner Pro (~$8–12/mo)** — code lane, real datasets, certificates, full skill-tree, no ads.
3. **Educator/Classroom (per-seat)** — dashboards, assignments, cohort analytics, LMS export (highest-value B2B).
4. **Certifications** — paid "ML Intuition" certificate, LinkedIn-verified.
5. **Content partnerships** — co-branded modules; affiliate hand-off to deeper courses.

**Inspiration benchmarks:** Teachable Machine (onboarding), TensorFlow Playground (NN intuition), GAN Lab (adversarial viz), Brilliant.org (interactive-first + freemium), Kaggle Learn (dataset-driven progression).

---

*End of spec — GameML v1.0. Ready to hand to engineering.*
