import { defaultStarterDSL } from "@/engine/dsl";
import type { Chapter, ExplainConceptGroup } from "./types";

/** Helper to keep chapter data readable. */
function concepts(
  ...groups: Array<[id: string, label: string, ...synonyms: string[]]>
): ExplainConceptGroup[] {
  return groups.map(([id, label, ...synonyms]) => ({ id, label, synonyms }));
}

export const CHAPTERS: Chapter[] = [
  {
    id: "ch1",
    number: 1,
    title: "Single Neuron / Perceptron",
    subtitle: "One unit, two weights, and the birth of a decision",
    goals: [
      "Build a single dense unit with sigmoid activation",
      "Train on XOR or AND and watch weights move",
      "Predict outputs before training, then compare",
    ],
    theory: `A **perceptron** (here: one dense neuron) computes

y = σ(w₁x₁ + w₂x₂ + b)

where σ is the sigmoid. It draws a **linear** decision boundary.
XOR is *not* linearly separable — a single neuron will struggle.
AND and OR are linearly separable — the neuron can succeed.

**Immediate Mode tip:** change \`lr\` or \`epochs\` and train again.
Watch the loss curve and the glowing neuron activations.`,
    starterDSL: defaultStarterDSL("ch1"),
    challenges: [
      {
        id: "ch1-c1",
        title: "Predict AND",
        description:
          "Before training: what should a perfect AND neuron output for (1,1) and (0,1)?",
        steps: [
          {
            kind: "predict",
            prompt:
              "For inputs (1,1) and (0,1), ideal AND outputs are closest to:",
            choices: [
              "(1,1)→1 and (0,1)→0",
              "(1,1)→0 and (0,1)→1",
              "both → 0.5",
              "both → 1",
            ],
            correctIndex: 0,
          },
          {
            kind: "experiment",
            prompt:
              "Set dataset=and, train a single dense neuron, reach accuracy ≥ 0.99.",
            experimentCheck: {
              minAccuracy: 0.99,
              dataset: "and",
              dslIncludes: ["dense"],
            },
          },
          {
            kind: "explain",
            prompt:
              "In your own words: why can one neuron solve AND but struggle with XOR?",
            explainConcepts: concepts(
              [
                "linear_boundary",
                "linear boundary",
                "linear",
                "line",
                "straight line",
                "linearly",
              ],
              [
                "separability",
                "separability",
                "separable",
                "separability",
                "separate",
                "separation",
              ],
              [
                "xor_vs_and",
                "AND vs XOR",
                "xor",
                "and",
                "dataset",
              ]
            ),
          },
        ],
      },
      {
        id: "ch1-c2",
        title: "Learning rate feel",
        description: "Feel how learning rate changes training dynamics.",
        steps: [
          {
            kind: "predict",
            prompt: "If learning rate is extremely large (e.g. 50), loss usually:",
            choices: [
              "Decreases smoothly to zero",
              "Oscillates or diverges",
              "Stays exactly constant forever",
              "Becomes negative",
            ],
            correctIndex: 1,
          },
          {
            kind: "experiment",
            prompt:
              "Train on dataset=or with lr between 0.1 and 2. Reach accuracy ≥ 0.99.",
            experimentCheck: {
              minAccuracy: 0.99,
              dataset: "or",
            },
          },
          {
            kind: "explain",
            prompt: "What did you observe when you tried a very high learning rate?",
            explainConcepts: concepts(
              [
                "instability",
                "unstable updates",
                "oscillate",
                "oscillation",
                "diverge",
                "divergence",
                "unstable",
                "explode",
                "jump",
              ],
              [
                "high_lr",
                "high learning rate",
                "high",
                "large",
                "learning rate",
                "lr",
                "step size",
              ],
              [
                "loss_behavior",
                "loss curve",
                "loss",
                "curve",
                "metrics",
                "accuracy",
              ]
            ),
          },
        ],
      },
      {
        id: "ch1-c3",
        title: "XOR wall",
        description: "Experience the limits of a single neuron on XOR.",
        steps: [
          {
            kind: "predict",
            prompt: "A single sigmoid neuron on XOR can reliably reach 100% accuracy?",
            choices: [
              "Yes, always",
              "No — XOR is not linearly separable",
              "Only if lr=0",
              "Only with softmax",
            ],
            correctIndex: 1,
          },
          {
            kind: "experiment",
            prompt:
              "Train dataset=xor with a single dense layer for ≥100 epochs. Observe accuracy plateau.",
            experimentCheck: {
              dataset: "xor",
              dslIncludes: ["dense"],
            },
          },
          {
            kind: "explain",
            prompt: "Explain why accuracy gets stuck near 50–75% for XOR with one neuron.",
            explainConcepts: concepts(
              [
                "not_linearly_separable",
                "not linearly separable",
                "linear",
                "linearly",
                "line",
                "separable",
                "separability",
              ],
              [
                "single_unit_limit",
                "one neuron limit",
                "single",
                "one neuron",
                "one unit",
                "cannot",
                "can't",
                "impossible",
                "stuck",
                "plateau",
              ],
              [
                "xor",
                "XOR problem",
                "xor",
                "exclusive",
              ]
            ),
          },
        ],
      },
    ],
  },
  {
    id: "ch2",
    number: 2,
    title: "Multi-layer Perceptron + Activations",
    subtitle: "Depth, nonlinearity, and solving XOR",
    unlockAfter: "ch1",
    goals: [
      "Stack dense layers with ReLU/sigmoid",
      "Solve XOR with an MLP",
      "Compare activations (relu vs tanh vs sigmoid)",
    ],
    theory: `An **MLP** composes linear maps with nonlinear activations:

h = ReLU(W₁x + b₁)
y = σ(W₂h + b₂)

Hidden units create **piecewise** decision regions. XOR becomes solvable.

**Try:** change hidden size (4 vs 16) and activation. Watch the decision boundary morph in Immediate Mode.`,
    starterDSL: defaultStarterDSL("ch2"),
    challenges: [
      {
        id: "ch2-c1",
        title: "Crack XOR",
        description: "Build an MLP that masters XOR.",
        steps: [
          {
            kind: "predict",
            prompt: "Minimum hidden nonlinearity needed to solve XOR is roughly:",
            choices: [
              "Zero hidden layers (single neuron)",
              "At least one hidden layer with nonlinear activations",
              "Softmax only",
              "Learning rate = 0",
            ],
            correctIndex: 1,
          },
          {
            kind: "experiment",
            prompt: "Train an MLP on xor to accuracy ≥ 0.99.",
            experimentCheck: {
              minAccuracy: 0.99,
              dataset: "xor",
            },
          },
          {
            kind: "explain",
            prompt: "How do hidden units help separate the XOR classes?",
            explainConcepts: concepts(
              [
                "hidden_units",
                "hidden layer",
                "hidden",
                "layer",
                "units",
                "neurons",
              ],
              [
                "nonlinearity",
                "nonlinearity",
                "nonlinear",
                "non-linear",
                "relu",
                "activation",
                "compose",
                "composition",
              ],
              [
                "regions",
                "decision regions",
                "region",
                "space",
                "feature space",
                "transform",
                "boundary",
              ]
            ),
          },
        ],
      },
      {
        id: "ch2-c2",
        title: "Activation swap",
        description: "Feel the difference between activations.",
        steps: [
          {
            kind: "predict",
            prompt: "ReLU(x) for x < 0 is:",
            choices: ["x", "0", "1", "sigmoid(x)"],
            correctIndex: 1,
          },
          {
            kind: "experiment",
            prompt:
              "Use at least one relu hidden layer and train on moons (accuracy ≥ 0.85).",
            experimentCheck: {
              minAccuracy: 0.85,
              dataset: "moons",
              dslIncludes: ["relu"],
            },
          },
          {
            kind: "explain",
            prompt: "When might sigmoid in hidden layers train more poorly than ReLU?",
            explainConcepts: concepts(
              [
                "vanishing_gradients",
                "vanishing gradients",
                "gradient",
                "gradients",
                "vanish",
                "vanishing",
              ],
              [
                "saturation",
                "saturation",
                "saturate",
                "saturated",
                "flat",
                "plateau",
              ],
              [
                "relu_contrast",
                "ReLU vs sigmoid",
                "relu",
                "sigmoid",
                "slow",
                "training",
              ]
            ),
          },
        ],
      },
      {
        id: "ch2-c3",
        title: "Circles",
        description: "Non-linear boundary on concentric circles.",
        steps: [
          {
            kind: "predict",
            prompt: "Concentric circles need a nonlinear model because:",
            choices: [
              "Classes are linearly separable",
              "A straight line cannot separate ring from core",
              "Sigmoid is undefined",
              "We only have one feature",
            ],
            correctIndex: 1,
          },
          {
            kind: "experiment",
            prompt: "Train on dataset=circles with an MLP, accuracy ≥ 0.85.",
            experimentCheck: {
              minAccuracy: 0.85,
              dataset: "circles",
            },
          },
          {
            kind: "explain",
            prompt: "Describe the decision boundary shape you observed.",
            explainConcepts: concepts(
              [
                "curved_shape",
                "curved boundary",
                "circle",
                "circular",
                "ring",
                "curved",
                "round",
                "radial",
              ],
              [
                "nonlinear_boundary",
                "nonlinear boundary",
                "nonlinear",
                "non-linear",
                "boundary",
                "decision",
              ],
              [
                "regions",
                "class regions",
                "region",
                "inside",
                "outside",
                "core",
                "annulus",
              ]
            ),
          },
        ],
      },
    ],
  },
  {
    id: "ch3",
    number: 3,
    title: "Overfitting, Regularization & Failure",
    subtitle: "When memorization is not understanding",
    unlockAfter: "ch2",
    goals: [
      "Build a high-capacity net that memorizes noise",
      "Apply L2 regularization",
      "Read loss curves as a diagnostic",
    ],
    theory: `**Overfitting:** train loss ↓ but the model fails to generalize.
High capacity (wide layers) + small data → memorize.

**L2 regularization** adds λ‖w‖² to the loss, preferring smaller weights.

In NeuralBASIC set \`l2=0.01\` (or similar) above the train line.
Compare decision boundaries with and without L2 on moons.`,
    starterDSL: defaultStarterDSL("ch3"),
    challenges: [
      {
        id: "ch3-c1",
        title: "Capacity explosion",
        description: "Train a wide net and inspect the boundary.",
        steps: [
          {
            kind: "predict",
            prompt: "Very wide nets on small data tend to:",
            choices: [
              "Always underfit",
              "Memorize quirks / overfit",
              "Ignore learning rate",
              "Remove the need for data",
            ],
            correctIndex: 1,
          },
          {
            kind: "experiment",
            prompt:
              "Train a multi-layer net on moons (any accuracy). Include dense layers.",
            experimentCheck: {
              dataset: "moons",
              dslIncludes: ["dense"],
            },
          },
          {
            kind: "explain",
            prompt: "What signs of overfitting might you see in the boundary?",
            explainConcepts: concepts(
              [
                "overfit",
                "overfitting",
                "overfit",
                "overfitting",
                "memorize",
                "memorization",
                "memorising",
                "memorizing",
              ],
              [
                "complex_boundary",
                "complex boundary",
                "wiggle",
                "wiggly",
                "jagged",
                "noise",
                "complex",
                "irregular",
              ],
              [
                "capacity",
                "high capacity",
                "wide",
                "capacity",
                "parameters",
                "quirk",
              ]
            ),
          },
        ],
      },
      {
        id: "ch3-c2",
        title: "Add L2",
        description: "Tame weights with weight decay.",
        steps: [
          {
            kind: "predict",
            prompt: "L2 regularization encourages weights to be:",
            choices: [
              "Larger",
              "Smaller / sparse magnitude",
              "Exactly zero always",
              "One-hot",
            ],
            correctIndex: 1,
          },
          {
            kind: "experiment",
            prompt: "Set l2 to a positive value (e.g. 0.01) and train on moons.",
            experimentCheck: {
              dataset: "moons",
              dslIncludes: ["l2"],
            },
          },
          {
            kind: "explain",
            prompt: "How did the decision boundary change after adding L2?",
            explainConcepts: concepts(
              [
                "smoother",
                "smoother boundary",
                "smooth",
                "smoother",
                "simple",
                "simpler",
              ],
              [
                "smaller_weights",
                "smaller weights",
                "smaller",
                "weight",
                "weights",
                "decay",
                "penalty",
              ],
              [
                "regularization",
                "regularization",
                "regular",
                "l2",
                "regularization",
                "constraint",
              ]
            ),
          },
        ],
      },
      {
        id: "ch3-c3",
        title: "Why models fail",
        description: "Connect failure modes to causes.",
        steps: [
          {
            kind: "predict",
            prompt: "Loss stuck high with tiny lr often means:",
            choices: [
              "Optimization too slow / stuck",
              "Perfect convergence",
              "Too much data",
              "Softmax is broken",
            ],
            correctIndex: 0,
          },
          {
            kind: "experiment",
            prompt: "Reach accuracy ≥ 0.9 on moons with any regularized or MLP setup.",
            experimentCheck: {
              minAccuracy: 0.9,
              dataset: "moons",
            },
          },
          {
            kind: "explain",
            prompt:
              "List two reasons a model fails: one optimization, one generalization.",
            explainConcepts: concepts(
              [
                "optimization",
                "optimization failure",
                "learning rate",
                "lr",
                "gradient",
                "stuck",
                "slow",
                "optimization",
                "converge",
              ],
              [
                "generalization",
                "generalization failure",
                "overfit",
                "underfit",
                "generalize",
                "generalization",
                "data",
              ],
              [
                "regularization_or_capacity",
                "capacity / regularization",
                "regular",
                "capacity",
                "l2",
                "wide",
              ]
            ),
          },
        ],
      },
    ],
  },
  {
    id: "ch4",
    number: 4,
    title: "Convolutional Basics",
    subtitle: "Local filters that slide across space",
    unlockAfter: "ch3",
    goals: [
      "Define a conv2d layer on tiny 4×4 images",
      "See filters respond to bars/edges",
      "Flatten + dense for classification",
    ],
    theory: `A **convolution** slides a small kernel across the image:

output(y,x) = Σ_c Σ_{ky,kx} K_{c,ky,kx} · input(c, y+ky, x+kx) + b

Weight **sharing** detects the same pattern everywhere.
Our toy set: vertical vs horizontal bars on 4×4 grids.`,
    starterDSL: defaultStarterDSL("ch4"),
    challenges: [
      {
        id: "ch4-c1",
        title: "Why conv?",
        description: "Motivation for local filters.",
        steps: [
          {
            kind: "predict",
            prompt: "Compared to a dense layer on pixels, conv layers:",
            choices: [
              "Use separate weights per pixel location (no sharing)",
              "Share a kernel across positions",
              "Cannot use ReLU",
              "Only work on text",
            ],
            correctIndex: 1,
          },
          {
            kind: "experiment",
            prompt: "Train TinyCNN-style net on tiny_images (accuracy ≥ 0.75).",
            experimentCheck: {
              minAccuracy: 0.75,
              dataset: "tiny_images",
              dslIncludes: ["conv2d"],
            },
          },
          {
            kind: "explain",
            prompt: "Why is weight sharing useful for detecting an edge anywhere?",
            explainConcepts: concepts(
              [
                "weight_sharing",
                "weight sharing",
                "share",
                "shared",
                "sharing",
                "same weights",
              ],
              [
                "kernel_filter",
                "kernel / filter",
                "kernel",
                "filter",
                "detector",
              ],
              [
                "position_invariance",
                "position invariance",
                "position",
                "anywhere",
                "location",
                "spatial",
                "edge",
                "local",
              ]
            ),
          },
        ],
      },
      {
        id: "ch4-c2",
        title: "Kernel size",
        description: "Experiment with kernel hyperparameters.",
        steps: [
          {
            kind: "predict",
            prompt: "A larger kernel generally sees:",
            choices: [
              "A smaller local region",
              "A wider local region",
              "No spatial structure",
              "Only the bias",
            ],
            correctIndex: 1,
          },
          {
            kind: "experiment",
            prompt: "Train with conv2d on tiny_images (any kernel), reach accuracy ≥ 0.7.",
            experimentCheck: {
              minAccuracy: 0.7,
              dataset: "tiny_images",
              dslIncludes: ["conv2d"],
            },
          },
          {
            kind: "explain",
            prompt: "What happens if the kernel is larger than the image?",
            explainConcepts: concepts(
              [
                "kernel_vs_image",
                "kernel vs image size",
                "kernel",
                "larger",
                "bigger",
                "size",
              ],
              [
                "image_size",
                "image dimensions",
                "image",
                "input",
                "spatial",
              ],
              [
                "invalid_or_pad",
                "invalid / padding",
                "invalid",
                "pad",
                "padding",
                "cannot",
                "doesn't fit",
                "too big",
                "error",
              ]
            ),
          },
        ],
      },
      {
        id: "ch4-c3",
        title: "Flatten bridge",
        description: "From feature maps to class scores.",
        steps: [
          {
            kind: "predict",
            prompt: "Flatten is used to:",
            choices: [
              "Increase spatial resolution",
              "Turn feature maps into a vector for dense layers",
              "Apply attention",
              "Shuffle labels",
            ],
            correctIndex: 1,
          },
          {
            kind: "experiment",
            prompt: "Include flatten in a conv model trained on tiny_images.",
            experimentCheck: {
              dataset: "tiny_images",
              dslIncludes: ["flatten", "conv2d"],
            },
          },
          {
            kind: "explain",
            prompt: "Describe the data flow: image → conv → flatten → dense.",
            explainConcepts: concepts(
              [
                "conv_stage",
                "convolution stage",
                "conv",
                "convolution",
                "filter",
                "feature map",
                "feature",
              ],
              [
                "flatten_stage",
                "flatten stage",
                "flatten",
                "vector",
                "reshape",
              ],
              [
                "dense_classify",
                "dense classifier",
                "dense",
                "class",
                "classification",
                "score",
                "output",
              ]
            ),
          },
        ],
      },
    ],
  },
  {
    id: "ch5",
    number: 5,
    title: "Attention + Tiny Transformer",
    subtitle: "Soft search over tokens",
    unlockAfter: "ch4",
    goals: [
      "Run a tiny transformer block on bag-of-features text",
      "Inspect attention heatmaps",
      "Contrast with pure dense baselines",
    ],
    theory: `**Attention** lets each token gather information from others:

Attention(Q,K,V) = softmax(QKᵀ / √d) V

Multi-head attention runs this in parallel subspaces.
A **transformer block** = attention + residual + feed-forward.

Here tokens are feature dimensions of tiny sentiment vectors.
Watch the attention heatmap after training.`,
    starterDSL: defaultStarterDSL("ch5"),
    challenges: [
      {
        id: "ch5-c1",
        title: "Softmax weights",
        description: "Attention as a weighted average.",
        steps: [
          {
            kind: "predict",
            prompt: "Rows of the attention matrix (after softmax) typically:",
            choices: [
              "Sum to ~1 (distribution over keys)",
              "Are always all zeros",
              "Contain only ±1",
              "Ignore the queries",
            ],
            correctIndex: 0,
          },
          {
            kind: "experiment",
            prompt:
              "Train a transformer (or attention) model on tiny_text, accuracy ≥ 0.7.",
            experimentCheck: {
              minAccuracy: 0.7,
              dataset: "tiny_text",
              dslIncludes: ["transformer"],
            },
          },
          {
            kind: "explain",
            prompt: "What does a bright cell in the attention heatmap mean?",
            explainConcepts: concepts(
              [
                "attention_weight",
                "attention weight",
                "attend",
                "attention",
                "weight",
                "weights",
              ],
              [
                "token_focus",
                "token focus",
                "token",
                "tokens",
                "focus",
                "pair",
              ],
              [
                "high_relevance",
                "high relevance",
                "high",
                "bright",
                "strong",
                "relevance",
                "relevant",
                "important",
              ]
            ),
          },
        ],
      },
      {
        id: "ch5-c2",
        title: "Heads",
        description: "Why multi-head?",
        steps: [
          {
            kind: "predict",
            prompt: "Multiple heads allow the model to:",
            choices: [
              "Attend only to the first token",
              "Capture different relation patterns in parallel",
              "Avoid using V",
              "Remove residuals",
            ],
            correctIndex: 1,
          },
          {
            kind: "experiment",
            prompt: "Train with heads=2 (transformer) on tiny_text.",
            experimentCheck: {
              dataset: "tiny_text",
              dslIncludes: ["transformer", "heads"],
            },
          },
          {
            kind: "explain",
            prompt: "Give one example of two different patterns heads might capture.",
            explainConcepts: concepts(
              [
                "multi_head",
                "multiple heads",
                "head",
                "heads",
                "multi-head",
                "parallel",
              ],
              [
                "different_patterns",
                "different patterns",
                "different",
                "pattern",
                "patterns",
                "relation",
                "relations",
              ],
              [
                "example_kinds",
                "kinds of relations",
                "syntax",
                "position",
                "semantic",
                "sentiment",
                "negation",
              ]
            ),
          },
        ],
      },
      {
        id: "ch5-c3",
        title: "Vs dense",
        description: "When attention helps.",
        steps: [
          {
            kind: "predict",
            prompt: "Attention is especially useful when:",
            choices: [
              "Inputs are fixed-size and independent always",
              "Relationships between tokens/positions matter",
              "We want no gradients",
              "Dataset is empty",
            ],
            correctIndex: 1,
          },
          {
            kind: "experiment",
            prompt: "Train either transformer or dense on tiny_text; accuracy ≥ 0.8.",
            experimentCheck: {
              minAccuracy: 0.8,
              dataset: "tiny_text",
            },
          },
          {
            kind: "explain",
            prompt:
              "Compare what a dense bag-of-features model vs attention can express.",
            explainConcepts: concepts(
              [
                "dense_bag",
                "dense bag-of-features",
                "dense",
                "bag",
                "independent",
                "linear",
              ],
              [
                "token_interaction",
                "token interactions",
                "interaction",
                "interactions",
                "pair",
                "token",
                "tokens",
                "context",
              ],
              [
                "attention_expressivity",
                "attention expressivity",
                "attention",
                "attend",
                "relationship",
                "relationships",
                "depend",
              ]
            ),
          },
        ],
      },
    ],
  },
];

export function getChapter(id: string): Chapter | undefined {
  return CHAPTERS.find((c) => c.id === id);
}

export function getChapterByNumber(n: number): Chapter | undefined {
  return CHAPTERS.find((c) => c.number === n);
}

export function isChapterUnlocked(
  chapterId: string,
  completedChapters: string[]
): boolean {
  const ch = getChapter(chapterId);
  if (!ch) return false;
  if (!ch.unlockAfter) return true;
  return completedChapters.includes(ch.unlockAfter);
}
