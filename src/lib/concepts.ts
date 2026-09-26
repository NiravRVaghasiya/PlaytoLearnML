/**
 * The Concept Library — spec §6 ("short explainer per concept") and §8, where
 * it is named as the SEO growth loop that should rank for questions like "what
 * is overfitting" and hand the reader into the game that teaches it.
 *
 * This module is the whole library. The routes under `src/app/concepts/` are
 * thin renderers over it, exactly as the home map is a thin renderer over
 * `catalog.ts`.
 *
 * Two rules hold this together, both enforced by `concepts.test.ts`:
 *
 *   1. Every `/concepts/...` link in any game's `why-cards.ts` resolves to an
 *      entry here. Those links shipped before the library did, so all four
 *      Phase-1 games were pointing their "read the concept" affordance at a
 *      404. The test makes that class of bug impossible to reintroduce.
 *   2. A link's promise is kept. A WhyCard that says "What momentum does" must
 *      land on a page with a section headed exactly that. Otherwise the link is
 *      technically alive and pedagogically a dead end.
 *
 * House style matches the WhyCards: plain language, name the mechanism rather
 * than the outcome, and never invent a number. Every figure quoted below is one
 * the shipped games actually produce.
 */

export interface ConceptSection {
  /**
   * Rendered as an `<h2>`. When a WhyCard links here with a `conceptLabel`,
   * one section heading must match that label — see rule 2 above.
   */
  heading: string;
  /** Paragraphs. Plain prose; no markdown parsing on the render side. */
  body: string[];
}

export interface ConceptFailureMode {
  /** The same name the game puts on screen, so the two reinforce each other. */
  name: string;
  gloss: string;
}

export interface ConceptMeta {
  /** URL segment: `/concepts/<slug>`. */
  slug: string;
  /** Short human name. Used as the `<h1>` and in navigation. */
  title: string;
  /** The phrase people actually search for. Drives the `<title>` tag. */
  question: string;
  /** A direct answer to `question`, shown as the lede. */
  answer: string;
  /** `<meta name="description">`. One sentence, under ~155 characters. */
  summary: string;
  sections: ConceptSection[];
  /** Game slugs that teach this, most relevant first. Must exist in the catalog. */
  games: string[];
  failureModes: ConceptFailureMode[];
  /** Other concept slugs worth reading next. */
  related: string[];
}

export const CONCEPT_LIBRARY: readonly ConceptMeta[] = [
  {
    slug: "decision-boundaries",
    title: "Decision boundaries",
    question: "What is a decision boundary?",
    answer:
      "A decision boundary is the surface a classifier draws through the space of features: on one side it answers one class, on the other side another. For a classifier, the boundary is not a picture of the model — it IS the model. Everything else is machinery for deciding where to put it.",
    summary:
      "A decision boundary is where a classifier switches its answer. Learn what it is, and why a more flexible boundary is not a better one.",
    sections: [
      {
        heading: "What is a decision boundary?",
        body: [
          "Put every example on a chart, one axis per measurement. A classifier's job is to divide that chart into regions and label each one. The dividing surface is the decision boundary. In two dimensions it looks like a line or a curve; in three it is a sheet; above that it stops being drawable but behaves exactly the same way.",
          "Predicting is then trivial: find which side of the boundary a new point falls on and answer with that region's label. All the difficulty in supervised learning is in the placing, not the looking-up.",
          "This is why two very different-sounding algorithms can be near-identical in practice. Logistic regression, a linear support vector machine and a single-layer network with no hidden units all draw a straight boundary. They disagree about which straight boundary is best, and about nothing else.",
        ],
      },
      {
        heading: "Bias and capacity",
        body: [
          "Capacity is how much a model is allowed to bend its boundary. A straight line has very little; a high-degree curve has a lot. Bias is the error you are stuck with because your capacity is too low to express the real pattern — a straight line cannot separate a ring from its centre no matter how well you fit it.",
          "The tempting conclusion is to always take more capacity, since it can only reduce bias. It does. It also lets the boundary chase individual points, and points include mistakes. Every extra bend is a bend that can be spent on signal or wasted on noise, and the model cannot tell which it is doing.",
          "So capacity is a budget rather than a free upgrade. The right amount is the least that can express the shape you actually believe is there. In Sort-It Arcade the deliberate labelling noise makes this concrete: on every round, the wiggliest boundary scores best on the points you can see, and worse than the five-parameter curve on the ones you cannot.",
        ],
      },
      {
        heading: "Reading a boundary you cannot draw",
        body: [
          "Real problems have dozens or thousands of features, so nobody looks at the boundary directly. Two habits replace looking. First, watch accuracy on data the model was not fitted to — a boundary that generalises and a boundary that memorises are indistinguishable on the training set and obvious on a held-out set.",
          "Second, count parameters against examples. A boundary with 25 free parameters fitted to 200 points has enough freedom to route around individual points; the same boundary fitted to 200,000 points does not. The ratio, not the parameter count alone, is what tells you whether the boundary is describing the data or reciting it.",
        ],
      },
    ],
    games: ["sort-it-arcade", "neuron-forge", "decision-tree-architect"],
    failureModes: [
      {
        name: "Overfit boundary",
        gloss:
          "The boundary bends to catch mislabelled points, so it fits the sample you have and misses the population you care about.",
      },
      {
        name: "Insufficient capacity",
        gloss:
          "The boundary cannot bend enough to express the real shape, so accuracy stalls well below what the data allows.",
      },
    ],
    related: ["overfitting", "gradient-descent"],
  },

  {
    slug: "overfitting",
    title: "Overfitting",
    question: "What is overfitting?",
    answer:
      "Overfitting is when a model learns the particular data it was shown instead of the pattern behind it. You can see it in a single comparison: high accuracy on the training data, noticeably lower accuracy on data the model has never seen. The gap between those two numbers is the symptom.",
    summary:
      "Overfitting is learning your sample instead of the pattern. See how to spot the train/validation gap and why simpler models often generalise better.",
    sections: [
      {
        heading: "What is overfitting?",
        body: [
          "Any real dataset is signal plus noise: a genuine pattern, plus measurement error, mislabelled rows and the accidents of which examples happened to be collected. A model with enough freedom will fit both, because nothing in the training objective distinguishes them. Fitting the noise is what we call overfitting.",
          "It is diagnosed by comparison, never from one number. Hold some data back, fit on the rest, then score both. On Sort-It Arcade's first round, switch to the 25-parameter Wiggle and press Fit it for me, and the overfit verdict reads 96% on the 200 points you fitted and 77% on 400 points it never saw — the 96% on its own looks like success, and it is the 19-point gap that tells the truth.",
          "The opposite failure exists and is often mistaken for this one. Underfitting also produces mediocre validation accuracy, but with no gap: the model is bad on data it has seen too. The distinction matters because the fixes are opposites. A gap means take capacity away; no gap and low accuracy means add some.",
        ],
      },
      {
        heading: "Why complexity is a cost",
        body: [
          "Extra parameters are usually described as capability, which makes them sound free. They are not. Each one is another degree of freedom the fitting process can spend on a coincidence in your particular sample, and the fitting process has no way to know that is what it is doing — a coincidence reduces training error exactly as convincingly as real structure does.",
          "So complexity is priced in generalisation. You pay for it whether or not you needed it, which is why the question is never \"can this model represent the pattern\" but \"is this the least model that can\". Decision Tree Architect makes the price visible: on its first plot, growing the tree from depth 3 to depth 6 takes training accuracy from 95% to 100% while validation falls from 93% to 88%. The deeper tree is better at the only thing it was optimised for, and worse at the thing that matters.",
          // Not "L1 pushes them to exactly zero": that needs a solver built to
          // land on zero, and the Adam-trained model in Overfit Tower Defense
          // (which links here) leaves its quieted weights near zero, not at it.
          "Regularization is how you make the cost explicit inside the objective. L2 penalises large weights, L1 pushes the unhelpful ones close to zero (exactly zero only with a solver built to land there), and dropout removes units at random during training. All three say the same thing to the optimiser: fit the data, but only spend flexibility that pays for itself.",
        ],
      },
      {
        heading: "Why simpler generalises",
        body: [
          "A simpler model has fewer ways to be wrong. There are enormously many wiggly boundaries that separate your training points and only a handful of straight ones, so if a straight one works, that is weak evidence it found something real — the wiggly one had so many chances to fit by luck that its success tells you much less.",
          "That intuition has a formal shape. Generalisation error decomposes into bias, variance and irreducible noise. Adding capacity lowers bias and raises variance, and total error is a U: it falls, bottoms out, then climbs. Overfitting is the right-hand side of that U, and the optimum is almost never at maximum capacity.",
          "Practically: fit the simplest model that clears your bar, add capacity only while held-out accuracy improves, and stop when the gap starts widening faster than validation accuracy rises. \"Simple\" here means constrained, not crude — a heavily regularized large network is simple in the sense that matters.",
        ],
      },
    ],
    games: [
      "overfit-tower-defense",
      "sort-it-arcade",
      "decision-tree-architect",
    ],
    failureModes: [
      {
        name: "Overfitting",
        gloss:
          "Training accuracy far above validation accuracy. The model memorised the sample; remove capacity or add regularization.",
      },
      {
        name: "Underfitting",
        gloss:
          "Both scores low, no gap. The model cannot express the pattern; add capacity or take regularization away.",
      },
      {
        name: "Overfit depth",
        gloss:
          "A tree grown past its best depth, with leaves holding too few examples to mean anything. Prune back.",
      },
    ],
    related: ["decision-boundaries", "learning-rate"],
  },

  {
    slug: "k-means",
    title: "K-means clustering",
    question: "How does k-means clustering work?",
    answer:
      "K-means splits unlabelled data into k groups by alternating two steps: assign every point to its nearest centre, then move each centre to the average of the points that chose it. Repeat until nothing moves. That is the entire algorithm.",
    summary:
      "K-means alternates assign and update until centres stop moving. Learn the loop, what inertia measures, and why the starting positions change the answer.",
    sections: [
      {
        heading: "How k-means works",
        body: [
          "Start with k centres placed somewhere. Then loop two steps. Assign: give every point to whichever centre is closest. Update: move each centre to the mean position of the points now assigned to it. Moving a centre can change which points are closest to it, so you assign again, and keep going until an assignment step changes nothing.",
          "The quantity being minimised is inertia — the total squared distance from each point to its own centre. Both steps reduce it and neither can increase it, which is why the loop always terminates. K-Means Territory Wars separates the two steps into two buttons for exactly this reason: run them by hand and the convergence stops being magic.",
          "What k-means is doing is fitting k spherical blobs of roughly equal size. When that assumption matches the data it is fast and hard to beat. When the real groups are elongated, nested or very different in size, it will still return an answer, confidently, and the answer will cut straight through them.",
        ],
      },
      {
        heading: "Why initialization matters",
        body: [
          "The loop only guarantees it will stop, not that it will stop anywhere good. Inertia has many local minima, and which one you land in is fixed entirely by where the centres started. Two runs on identical data with different starting positions can produce visibly different clusterings, both of them stable.",
          "The classic failure is two centres sharing one dense group while a real group goes unclaimed. Every assign step confirms the split and every update step keeps both centres inside it; nothing in the algorithm can notice the group it missed, because moving a centre out there would raise inertia on the way.",
          "The standard defences are cheap. Run it several times from different random starts and keep the lowest-inertia result — that is what scikit-learn's `n_init` does. Better, seed with k-means++, which picks each new centre with probability proportional to its squared distance from the nearest existing one, so centres start spread out instead of clumped.",
        ],
      },
    ],
    // Only the game that actually runs k-means. Dimension Diver used to be
    // listed too, apparently copied from its catalog unlock edge
    // (`requires: ["k-means-territory-wars"]`), but it is a PCA game with no
    // clustering in it, so "Play it instead of reading about it" sent readers
    // to a different algorithm. An unlock edge says what to play NEXT; this
    // list says what teaches THIS.
    games: ["k-means-territory-wars"],
    failureModes: [
      {
        name: "Bad k",
        gloss:
          "Too few centres to cover the real groups, or so many that they subdivide groups that were already correct.",
      },
      {
        name: "Poor initialization",
        gloss:
          "Centres started clumped, converged to a stable-but-wrong split, and no further iteration can escape it.",
      },
    ],
    related: ["choosing-k", "decision-boundaries"],
  },

  {
    slug: "choosing-k",
    title: "Choosing k",
    question: "How do you choose the number of clusters?",
    answer:
      "You cannot choose k by minimising inertia, because inertia always falls as k rises. Instead you look for the k where it stops falling steeply — the elbow — and then check that the groups you get mean something outside the maths.",
    summary:
      "Inertia always improves with more clusters, so it cannot pick k. How to read an elbow chart, and what to do when there is no elbow.",
    sections: [
      {
        heading: "Why inertia can't choose k",
        body: [
          "Inertia is the total squared distance from each point to its assigned centre, and adding a centre can only reduce it. In the limit where k equals the number of points, every point is its own centre and inertia is exactly zero — a perfect score for a clustering that has learned nothing.",
          "So inertia is a fine objective for a fixed k and useless for comparing across k. K-Means Territory Wars demonstrates this rather than asserting it. On its first map, which has three real groups, the elbow table lists the best inertia each number of flags can reach: 1.01 for three flags and 0.44 for eight. The eight-flag answer is numerically more than twice as good, and settling eight flags there still earns the Bad k verdict.",
          "This is a general shape, not a quirk of clustering. Any score that improves monotonically with model size cannot be used to select model size. Training accuracy has the same problem in supervised learning, which is why held-out data exists.",
        ],
      },
      {
        heading: "Reading an elbow chart",
        body: [
          "Plot inertia against k, from 1 up to more clusters than you think you need. The curve falls steeply while each new centre is claiming a genuine group, then flattens once the real groups are covered and further centres are only subdividing them. The corner between those two regimes is the elbow, and the k at the corner is your candidate.",
          "Read the slope, not the height. The useful question at each step is how much inertia the previous centre bought: a large drop means it found structure, a small one means it split something that was already one group. When the drops become uniformly small, you have passed the elbow.",
          "Two honest caveats. Elbows are often ambiguous, and on genuinely smooth data there is no corner at all — which is itself the answer, that the data has no particular number of groups. And an elbow is evidence about geometry only. Silhouette score, gap statistic and the Calinski-Harabasz index give you a second opinion, but none of them knows what the clusters are for.",
        ],
      },
      {
        heading: "When the number comes from outside the data",
        body: [
          "Often k is not really a modelling question. If you are cutting a customer base into segments that each need a written strategy, the constraint is how many strategies anyone will maintain. If the clusters map to physical bins, the number of bins decides it. In those cases the elbow chart is a sanity check on a number you already have, not a source for it.",
          "Where you do have freedom, the strongest test is stability. Re-run the clustering on bootstrap samples and see whether the same k keeps producing the same groups. A k that survives resampling is describing the data; a k that reshuffles every run is describing the sample.",
        ],
      },
    ],
    games: ["k-means-territory-wars"],
    failureModes: [
      {
        name: "Bad k",
        gloss:
          "Chosen by minimising inertia, which rewards splitting real groups apart. Read the elbow instead.",
      },
    ],
    related: ["k-means", "overfitting"],
  },

  {
    slug: "data-cleaning",
    title: "Data cleaning",
    question: "What does data cleaning actually do?",
    answer:
      "Data cleaning is the set of decisions you make about rows and values that are missing, wrong or on the wrong scale. None of those decisions is neutral — each one trades some fidelity for some usability, and the trade shows up directly in model quality.",
    summary:
      "Cleaning is a series of tradeoffs, not a correctness pass. What imputation, outlier handling and scaling really cost you.",
    sections: [
      {
        heading: "What data cleaning does",
        body: [
          "It is tempting to think of cleaning as removing errors so the real data can show through. It is closer to the opposite: you are choosing what to assume where the data does not say. Fill a blank with the column mean and you have asserted that the missing value was average. Drop the row and you have asserted the row was expendable. Both are claims, and both can be wrong.",
          "Three families of decision cover most of it. Missing values, where the options are drop, impute or model the missingness explicitly. Outliers, where a value is either a data-entry error to remove or a rare real event that is often the most informative thing in the table. And scaling, which changes units so that distance-based and gradient-based methods treat features comparably.",
          "Scaling is worth separating from the other two because it is the only one that is close to lossless. Standardising a column preserves every ordering and every relationship; it just stops a feature measured in thousands from dominating one measured in fractions. Feature Forge leans on this: standardising alone produces no lift, because changing units does not change meaning.",
        ],
      },
      {
        heading: "Outliers are not automatically errors",
        body: [
          "A blood-pressure reading of 400 is a typo. A transaction 200 times larger than any other might be the fraud you are trying to detect. The same statistical test flags both, so the test cannot be the decision — you need to know which mechanism produced the value.",
          "The practical rule is to separate impossible from improbable. Values outside physical limits, negative durations, dates before the system existed: those are errors and removing them loses nothing. Extreme but possible values should be kept, and if they destabilise the model, handled by a method that tolerates them — a robust loss, a rank transform, or a bin that says \"very large\" — rather than deleted.",
        ],
      },
      {
        heading: "Why order matters, and why the split comes first",
        body: [
          "Cleaning steps compose, and some pairs do not commute. Remove outliers before computing the mean you will impute with and you get a different mean than the other way round. Neither is wrong, but you have to know which you did, which is why pipelines exist rather than a sequence of ad-hoc edits.",
          "One ordering rule is not a preference. Split your data before you fit anything a cleaning step needs to learn. A mean, a standard deviation, a category vocabulary and a bin edge are all fitted quantities; compute them over the whole dataset and your validation set has leaked into your training set through the scaler. Fit on train, apply to both.",
        ],
      },
    ],
    games: ["data-detox", "feature-forge"],
    failureModes: [
      {
        name: "Data starvation",
        gloss:
          "So many rows removed in the name of cleanliness that nothing is left to learn a pattern from.",
      },
      {
        name: "Leakage",
        gloss:
          "A cleaning step fitted on all the data, or a column that is only known after the outcome, so validation scores look excellent and mean nothing.",
      },
    ],
    related: ["missing-data", "overfitting"],
  },

  {
    slug: "missing-data",
    title: "Missing data",
    question: "What should you do about missing values?",
    answer:
      "It depends entirely on why the values are missing. If they went missing for reasons unrelated to what you are predicting, dropping the rows is merely wasteful. If the reason is related, dropping them changes what your data represents — and the model will confidently learn the changed version.",
    summary:
      "Dropping rows with blanks is not a neutral act. The difference between MCAR, MAR and MNAR, and what each one costs you.",
    sections: [
      {
        heading: "Missing not at random",
        body: [
          "Statisticians split missingness into three cases, and the names are worth learning because they carry the whole decision. Missing completely at random (MCAR): the blanks are unrelated to anything, so your surviving rows are a smaller but fair sample. Missing at random (MAR): the blanks depend on other columns you do have, so they can be modelled. Missing not at random (MNAR): the blanks depend on the missing value itself, or on the thing you are predicting.",
          "MNAR is the dangerous one and it is extremely common, because the reason a value is absent is usually connected to the situation that produced it. A test not ordered because the patient looked well. Income left blank more often at the top and bottom of the range. A sensor that fails in the conditions you most want to detect. In each case the blank is evidence, and discarding it discards the evidence.",
          "You cannot tell these apart from the blanks alone; it requires knowing how the data was collected. What you can do is test for the consequence — compare the distribution of your outcome and key features between rows that are complete and rows that are not. If those distributions differ, you are not in the MCAR case and dropping is not safe.",
        ],
      },
      {
        heading: "The cost of dropping rows",
        body: [
          "Dropping incomplete rows, or complete-case analysis, has two costs. The obvious one is sample size, and with blanks scattered across many columns it compounds fast: ten columns each 5% missing can eliminate roughly 40% of rows even though 95% of every column is present.",
          "The subtler cost is that the survivors are a biased sample. Data Detox measures this directly — drop the rows with blanks and the 39 remaining rows are 33% healthy, where the data you started from was 53% healthy. The model then trains on a population that is sicker than the real one and calibrates to it. Its accuracy on that skewed validation set can even look fine, which is what makes selection bias hard to catch.",
          "So the honest summary is that dropping is only free under the assumption you are least likely to be able to check. It is a reasonable default when very few rows are affected and you have reason to believe the blanks are incidental. It is a bad default at scale.",
        ],
      },
      {
        heading: "Why dropping isn't neutral",
        body: [
          "The alternatives are not obviously better; they are differently wrong, which is the point. Mean or median imputation keeps every row and shrinks the column's variance, weakening whatever relationship it had. Model-based imputation preserves relationships and risks inventing ones that were never there. Multiple imputation is the statistically correct answer and costs you a single tidy dataset to work with.",
          "The technique that is usually undervalued is to stop treating the blank as an absence. Add a boolean column recording that the value was missing, then impute whatever you like. If the missingness carried information, the model can now use it explicitly instead of having it smuggled in through a biased sample. If it did not, the extra column is close to free and the model will ignore it.",
          "Whichever route you take, fit the imputer on the training split only and apply it to validation. An imputer is a fitted model, and fitting it on everything is leakage in the same way any other pre-processing step would be.",
        ],
      },
    ],
    games: ["data-detox"],
    failureModes: [
      {
        name: "Selection bias",
        gloss:
          "The surviving rows have a different class balance from the data you started with, so the model learns a population that does not exist.",
      },
      {
        name: "Data starvation",
        gloss:
          "Enough rows dropped that the remaining sample cannot support the pattern you are trying to learn.",
      },
    ],
    related: ["data-cleaning", "overfitting"],
  },

  {
    slug: "gradient-descent",
    title: "Gradient descent",
    question: "How does gradient descent work?",
    answer:
      "Gradient descent finds parameters that make a loss small by repeatedly measuring which way the loss increases fastest and taking a step in the opposite direction. It only ever uses local information, which is what makes it cheap and what makes it fallible.",
    summary:
      "Gradient descent walks downhill using only the local slope. How the loop works, why it gets stuck, and what momentum does about it.",
    sections: [
      {
        heading: "How gradient descent works",
        body: [
          "Write your error as a function of the parameters — that surface is the loss landscape, with parameters as the position and loss as the height. The gradient at a point is the direction of steepest increase. Step a small distance the other way, recompute, repeat. Each step lowers the loss provided the step is small enough that the slope has not changed much along the way.",
          "That proviso is the whole practical difficulty. The gradient tells you a direction and nothing about distance, so the step size, the learning rate, is a separate choice and the most consequential one you will make. Too small and you never arrive; too large and each step overshoots into terrain that is steeper still, and the loss climbs.",
          "It scales because it is local. Computing the gradient of a loss with a hundred million parameters costs about as much as evaluating the loss once, via backpropagation, and needs no knowledge of the surface beyond the current point. Stochastic gradient descent goes further and estimates the gradient from a small batch, trading exactness for far more steps per second.",
        ],
      },
      {
        heading: "Local versus global minima",
        body: [
          "A point where the gradient is zero and the surface curves upward in every direction is a local minimum. Gradient descent stops there and reports success, because from where it stands every direction is uphill and it has no way to see over the ridge. Gradient Descent Skier makes this a level: plain descent settles at loss 0.33, in the shallow valley at x 0.90, while the deepest valley is at x -1.08 with a loss of -0.36.",
          "In low dimensions this is a serious problem. In the very high dimensions of a neural network it is less common than it sounds, because a point is only a local minimum if the surface curves up along every one of millions of axes — usually at least one curves down, making it a saddle point rather than a trap. Saddles slow training badly, since the gradient is nearly zero, but they can be escaped.",
          "The defences are all forms of not trusting a single trajectory: restart from several initialisations and keep the best, keep the learning rate large early so shallow basins cannot hold you, or use stochastic gradients whose noise is itself enough to shake a shallow minimum apart.",
        ],
      },
      {
        heading: "What momentum does",
        body: [
          "Momentum keeps a running average of past gradients and steps along that instead of along the current gradient alone. Formally, a velocity vector accumulating each new gradient with a decay factor, typically around 0.9. Physically, it is the difference between a walker who re-decides at every step and a ball that has been rolling for a while.",
          "This buys two things. Gradients that keep pointing the same way compound, so long shallow slopes get crossed quickly. Gradients that flip back and forth — the ping-ponging you get across a narrow valley — cancel in the average, so oscillation damps. Both are more valuable than escaping traps, and both are why momentum is a default rather than a trick.",
          "Escaping traps is the visible bonus. Keep the identical learning rate that got stuck at 0.33 and set momentum to 0.9, and the skier is carried through the shallow valley and settles at a loss of -0.36, at x -1.08 — the global minimum. Nothing changed but the accumulated history. Adam adds per-parameter step scaling on top of the same idea, which is why it is usually the first optimiser people reach for.",
        ],
      },
    ],
    // Convolution Kitchen last: its "Let it learn" button hands the kernels the
    // player designed by hand to gradient descent, and its WhyCard links here.
    games: [
      "gradient-descent-skier",
      "backprop-blitz",
      "neuron-forge",
      "convolution-kitchen",
    ],
    failureModes: [
      {
        name: "Local minimum",
        gloss:
          "Converged to a valley that is not the deepest one. Every local direction is uphill, so the loop cannot tell.",
      },
      {
        name: "Divergence",
        gloss:
          "Steps too large for the curvature, so each one overshoots and the loss grows until it stops being a number.",
      },
      {
        // Neuron Forge names this one, from a measured all-zero relu layer.
        name: "Dying ReLU",
        gloss:
          "Every relu unit in a layer outputs zero for every input. A relu at zero passes back zero gradient, so no signal flows back through that layer and nothing can revive it: the optimiser failed, not the model's capacity.",
      },
    ],
    related: ["learning-rate", "overfitting"],
  },

  {
    slug: "learning-rate",
    title: "Learning rate",
    question: "Why does the learning rate matter so much?",
    answer:
      "The learning rate sets how far you move per step, and it is the only hyperparameter that can make training fail completely in both directions. Too large and the loss explodes within a handful of steps; too small and it converges so slowly you conclude the model cannot learn.",
    summary:
      "The learning rate is the hyperparameter that dominates training. Why too-large rates explode, and how to find a workable one quickly.",
    sections: [
      {
        heading: "Why learning rate dominates",
        body: [
          "Every other hyperparameter changes how well a training run ends up doing. The learning rate decides whether the run means anything at all. Get it wrong upward and the loss is infinite after a few steps; wrong downward and you are watching a curve that will eventually work, given more time than you have. Neither failure is subtle, and both are frequently misread as a problem with the architecture.",
          "It also interacts with almost everything else. Batch size changes the gradient's noise and so the rate you can tolerate. Normalisation layers change the scale of the gradients reaching each weight. Momentum effectively multiplies your step length by roughly 1/(1-β), so switching momentum on without lowering the rate can turn a stable run unstable. Tune the learning rate first, then the rest, then the learning rate again.",
          "Because it spans orders of magnitude, search it on a log scale. Hyperparameter Heist is built around this: a grid that spends its whole sixteen-try budget sees only two distinct learning rates and never cracks the safe, while random search on the same budget sees around twelve and usually does. Resolution on the dial that matters beats coverage of the dials that do not.",
        ],
      },
      {
        heading: "Why too-large rates explode",
        body: [
          "The gradient is a good description of the surface only very near where you measured it. A step takes you somewhere the gradient was not measured, and if the surface curves, the slope there is different. On the outside of a bowl the slope steepens as you move away from the bottom, so an overshooting step lands somewhere steeper, producing a larger gradient, a larger next step, and a positive feedback loop.",
          "There is a threshold, not a gradual decline. For a quadratic bowl of curvature L, steps below 2/L converge and steps above 2/L diverge, with no smooth region between. Gradient Descent Skier's deep valley puts that line at a plain-descent rate of about 0.40, and shows where a real surface parts company with the bowl: just past the line the skier does not explode, it bounces from wall to wall and never comes to rest, which the game names Oscillation, because its surface is two valleys rather than one bowl. The loss runs away from a rate of about 0.68, within three steps, and from 0.85 on the very first one, when that first move has already left the region the gradient described.",
          "The classic symptoms are worth memorising. Loss rising monotonically from step one, loss reaching NaN or infinity, or weights growing without bound all mean the rate is too high — nothing else produces that shape. Loss falling then bouncing around a floor usually means the rate was fine to start with and is now too high for where you are, which is exactly what a decay schedule fixes.",
        ],
      },
      {
        heading: "Finding one without guessing",
        body: [
          "The range test is cheap and almost always enough. Train for a few hundred steps while increasing the rate exponentially, and plot loss against rate. You get a curve that is flat while the rate is too small to matter, falls steeply through the useful band, and turns sharply upward at divergence. Pick a value in the steep part, roughly an order of magnitude below where it turned up.",
          "Then decay it. Almost every good schedule starts high, to cross the landscape and skip shallow basins, and ends low, to settle precisely. Cosine decay and step decay both work; the choice between them matters far less than having a schedule at all. A short warmup from near zero is worth adding when the model is large or the batch is big, because the first few gradients are the least trustworthy ones you will see.",
          "Adaptive optimisers like Adam scale the step per parameter and genuinely reduce how carefully you have to tune. They do not remove the need: they replace one sensitive number with one less-sensitive number, and a badly chosen Adam rate still diverges the same way.",
        ],
      },
    ],
    games: ["gradient-descent-skier", "hyperparameter-heist"],
    failureModes: [
      {
        name: "Divergence",
        gloss:
          "Rate far enough past the stability threshold that every step overshoots further than the last. The loss climbs from the first step and never recovers.",
      },
      {
        name: "Oscillation",
        gloss:
          "Steps, or momentum, too large to come to rest: each one carries past the valley floor and back, so the loss bounces instead of settling.",
      },
      {
        name: "Grid trap",
        gloss:
          "A grid search spending its budget on dials that do not matter, leaving too few distinct learning rates to find a working one.",
      },
    ],
    related: ["gradient-descent", "overfitting"],
  },
] as const;

export function getConcept(slug: string): ConceptMeta | undefined {
  return CONCEPT_LIBRARY.find((c) => c.slug === slug);
}

export function conceptSlugs(): string[] {
  return CONCEPT_LIBRARY.map((c) => c.slug);
}

/** `/concepts/<slug>` — the shape the WhyCards link to. */
export function conceptPath(slug: string): string {
  return `/concepts/${slug}`;
}
