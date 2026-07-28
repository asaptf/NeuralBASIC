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
    lesson: [
      {
        heading: "One neuron is a scorer with a threshold",
        body: `A neuron does two things, and it helps to keep them separate in your head.

First it computes a **score** — one number — by weighting each input and adding an offset:

score = w₁x₁ + w₂x₂ + b

Then it squashes that score into a confidence between 0 and 1 with the **sigmoid**, σ. A large positive score becomes close to 1, a large negative score close to 0, and a score of exactly 0 becomes 0.5 — dead undecided.

So the weights decide *how much each input matters and in which direction*, and the bias \`b\` decides *how eager the neuron is to fire at all*. The sigmoid only converts a score to a confidence; it does no deciding of its own.`,
      },
      {
        heading: "Why the boundary is a straight line",
        body: `The neuron answers "class 1" when its confidence passes 0.5 — which happens exactly when the score passes 0. So the frontier between its two answers is the set of points where

w₁x₁ + w₂x₂ + b = 0

That is the equation of a **straight line**. Not a metaphor: it is literally a line, and it's the line you see in the Data Lab.

Two consequences worth holding on to:

- The weights set the line's **orientation**; the bias slides it back and forth. Without a bias, the line would be nailed to the origin.
- One neuron gets **one** line. Everything it can ever learn is "which side of this line are you on?"`,
        example: {
          label: "Run AND on one neuron",
          dsl: `network Perceptron {
  dense 2 -> 1 activation=sigmoid
}
train dataset=and lr=0.8 epochs=200`,
          expect:
            "AND reaches 100% and the summary says all 4 points are on the correct side. One line is enough: only (1,1) needs separating from the other three.",
        },
      },
      {
        heading: "What the loss number actually is",
        body: `The **Loss** card is not "how many did it get wrong". It is **binary cross-entropy**: the average penalty for being confidently wrong.

Cross-entropy punishes confidence, not just error. Predicting 0.51 for a class-0 point is a small mistake. Predicting 0.99 for the same point is a big one, even though both are "wrong" by the same accuracy count. That's deliberate — it's what pushes the neuron to become *decisive* rather than merely correct-by-a-hair.

**Accuracy**, next to it, is the blunt count: what fraction of points land on the right side of 0.5. The two can move in opposite directions, and when they do it's informative — loss falling while accuracy sits still usually means the neuron is getting more confident about points it already had right.`,
      },
      {
        heading: "How it learns, and when the curve gets jagged",
        body: `Training here is **stochastic gradient descent**, one sample at a time. For each example the engine asks "which way should I nudge each weight to make *this* point's loss smaller?", takes a small step that way, and moves on. One **epoch** is one pass over the data.

The loss is then re-measured over all the data at the end of each epoch — which explains a detail you'll notice immediately. On \`or\`, the curve slides down smoothly. On \`moons\` or \`xor\`, it's visibly jagged.

The difference isn't difficulty as such, it's **agreement**. On \`or\` every sample wants the weights moved the same way, so the steps compound and the curve is clean. On \`moons\` the samples disagree: a step that helps one point hurts another, so the weights get tugged back and forth within a single epoch and the end-of-epoch measurement lands somewhere slightly different each time. Jaggedness is a picture of disagreement between your data points, not of the neuron being broken.`,
      },
      {
        heading: "What a large learning rate really does",
        body: `The **learning rate** is the size of each step. The usual warning is "too large and the loss will oscillate wildly" — so it's worth seeing what actually happens here, because it isn't quite that.

Turn the learning rate up on \`moons\` and the curve does *not* get noticeably more jagged. What changes is the **height** of the loss: it multiplies several times over, while accuracy barely budges.

Here's the mechanism. Big steps drive the weights to large magnitudes. Large weights mean large scores, and the sigmoid squashes large scores to almost exactly 0 or 1 — so the neuron stops hedging and becomes absolutely certain about everything. Cross-entropy charges enormously for confident mistakes, so the handful of points it still gets wrong now cost a fortune each.

That's the real damage from too high a rate, and it's why you should watch **loss and accuracy together**. A model that is 82% right and screaming is in worse shape than one that is 85% right and hedging, even though the accuracy gap looks trivial.`,
        example: {
          label: "Compare lr=20 against lr=0.8 on moons",
          dsl: `network Perceptron {
  dense 2 -> 1 activation=sigmoid
}
train dataset=moons lr=20 epochs=200`,
          expect:
            "Accuracy lands near 82% — much like lr=0.8 — but the loss balloons to several times its old value, instead of 0.3. How far it balloons varies a lot between runs, which is itself the point: the high-rate model is unstable. Edit lr back to 0.8, train again, and watch the accuracy hold while the loss collapses.",
        },
      },
      {
        heading: "The XOR wall",
        body: `Now the wall this chapter is really about. Plot XOR's four points and try to draw a single straight line with class 1 on one side and class 0 on the other:

- (0,0) → 0 and (1,1) → 0 sit on opposite corners
- (0,1) → 1 and (1,0) → 1 sit on the *other* opposite corners

Each class occupies two diagonally opposite corners. No straight line separates a diagonal pair from the other diagonal pair. XOR is **not linearly separable**, and one neuron only ever gets one line.

So the neuron doesn't fail because it's badly tuned, or because you were impatient with the epochs. It fails because you asked a question its shape cannot answer. Turning up \`lr\` or \`epochs\` cannot fix a geometry problem — a genuinely useful thing to recognise, because a lot of real debugging is telling this apart from bad tuning.`,
        example: {
          label: "Hit the wall on XOR",
          dsl: `network Perceptron {
  dense 2 -> 1 activation=sigmoid
}
train dataset=xor lr=0.8 epochs=400`,
          expect:
            "Accuracy lands anywhere from 25% to 75% depending on the random starting weights, and the Data Lab keeps ringing points no matter how long you train. Press Reset and Train a few times: the number moves, the wall doesn't.",
        },
      },
      {
        heading: "Common traps",
        body: `- **"More epochs will fix it."** On AND, more epochs sharpens confidence. On XOR, more epochs changes nothing that matters — the ceiling is structural.
- **"50% accuracy means it half-learned."** On a balanced two-class set, 50% is exactly what coin-flipping scores. That's the dashed *chance* line under the accuracy curve; below it, you're worse than guessing.
- **"The line is the neuron."** The line is a *consequence* of the weights and bias. Change the numbers and the line moves; the neuron is the numbers.
- **Reading a run as final.** Weights start random, so XOR can land at 25%, 50% or 75% on different runs. Press Reset and Train a few times before concluding anything from one number.`,
      },
      {
        heading: "Where this goes next",
        body: `You've found the exact limitation that motivated the next fifty years of the field. If one line is not enough, you need a way to **bend** the boundary.

The fix turns out to be almost embarrassingly simple: put a layer of neurons *between* the input and the output, so the network can carve the space with several lines and then combine them. That combining step needs a nonlinearity — without one, stacked layers collapse back into a single line, which is a trap Chapter 2 makes you walk into on purpose.

Finish this chapter's challenges and Chapter 2 opens.`,
      },
    ],
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
        description: "Find out what a large learning rate actually costs you.",
        steps: [
          {
            kind: "predict",
            prompt:
              "On moons, compared with lr=0.8, training at lr=20 usually ends up:",
            choices: [
              "About as accurate, but with a much higher loss",
              "Much less accurate, with a much higher loss",
              "More accurate, because bigger steps learn faster",
              "Identical — the learning rate only changes how long it takes",
            ],
            correctIndex: 0,
          },
          {
            kind: "experiment",
            prompt:
              "Train dataset=moons with lr between 0.1 and 2, and reach accuracy \u2265 0.80. Then try lr=20 and compare the loss.",
            experimentCheck: {
              minAccuracy: 0.8,
              dataset: "moons",
            },
          },
          {
            kind: "explain",
            prompt:
              "A high learning rate barely moved accuracy but multiplied the loss. Why does the loss react so much more strongly?",
            explainConcepts: concepts(
              [
                "confidence",
                "overconfidence",
                "confident",
                "confidence",
                "certain",
                "certainty",
                "saturate",
                "saturated",
                "extreme",
                "0 or 1",
              ],
              [
                "loss_penalty",
                "cross-entropy punishes confident errors",
                "cross-entropy",
                "cross entropy",
                "penalty",
                "punish",
                "penalise",
                "penalize",
                "cost",
              ],
              [
                "accuracy_vs_loss",
                "accuracy only counts sides, loss measures confidence",
                "accuracy",
                "threshold",
                "0.5",
                "side",
                "count",
              ],
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
    lesson: [
      {
        heading: "Combining lines instead of drawing a better one",
        body: `Chapter 1 ended at a wall: one neuron gets one straight line, and XOR needs more than that. The fix is not a cleverer line. It's **several** lines plus something that combines them.

Put a layer of neurons between the input and the output. Each hidden neuron does exactly what the Chapter 1 neuron did — computes a score, squashes it — so each one carves the plane with its own line. The output neuron then looks at *their* answers rather than at the raw inputs, and asks a question about the combination: "which side of line A *and* which side of line B?"

That composition is what buys you a bent boundary. Regions, not halves.`,
      },
      {
        heading: "The trap: depth is worthless without nonlinearity",
        body: `Before adding a hidden layer, it's worth understanding what makes it work — because it is easy to add one that does nothing at all.

Stack two layers with **no** activation between them and you have applied one matrix, then another. But a matrix times a matrix is just... another matrix. Two linear maps compose into a single linear map, which is a single line. The network has more numbers in it and exactly the same power.

This isn't a subtlety you have to take on trust. Run the example: a hidden layer with \`activation=linear\` parks at **50% on XOR** — chance — and it stays at 50% whether the hidden layer has 4 units or 16. Width cannot rescue a model that has no nonlinearity, because the problem isn't capacity, it's shape.

The activation function is not a performance tweak. It is the thing that makes depth mean anything.`,
        example: {
          label: "Watch a linear hidden layer fail",
          dsl: `network LinearTrap {
  dense 2 -> 16 activation=linear
  dense 16 -> 1 activation=sigmoid
}
train dataset=xor lr=0.35 epochs=500`,
          expect:
            "Accuracy stays around 50% — chance, no better than the single neuron, despite 16 hidden units. Change activation=linear to activation=relu and the same network suddenly solves it.",
        },
      },
      {
        heading: "Representable is not the same as findable",
        body: `Textbooks will tell you two hidden units are enough to represent XOR. That's true, and it's also misleading, because you don't get to place the weights by hand — gradient descent has to *find* them from a random start.

Measured on this engine over **forty runs per width**, single ReLU hidden layer, counting a run as solved at \u2265 99% accuracy:

- **2 hidden units** — solved 15% of runs
- **3 units** — 20%
- **4 units** — 20%
- **8 units** — 75%
- **16 units** — 100%

The theoretical minimum works about one time in seven. Extra units give gradient descent more chances to stumble into a workable arrangement: with sixteen, some pair of them lands somewhere useful early and the rest follows.

Note also that 3 and 4 units are indistinguishable here. Capacity doesn't buy reliability smoothly — there's a threshold, and below it you are mostly buying lottery tickets.

This gap between "a solution exists in there" and "training reliably reaches it" is one of the most practically important things in the subject, and it never shows up in the equations.`,
        example: {
          label: "Give XOR sixteen hidden units",
          dsl: `network MLP {
  dense 2 -> 16 activation=relu
  dense 16 -> 1 activation=sigmoid
}
train dataset=xor lr=0.35 epochs=500`,
          expect:
            "Solves XOR in about 98 runs out of 100, landing from 50% to 100% across attempts, and the Data Lab finally shows a bent boundary with no points ringed. Edit 16 down to 2 and train repeatedly \u2014 it will mostly fail, which is the point.",
        },
      },
      {
        heading: "Which activation, honestly",
        body: `The folklore is "use ReLU". Here's what actually happened on XOR with eight hidden units, **sixty runs each**:

- **sigmoid** — solved 97% of runs
- **tanh** — 93%
- **relu** — 78%

ReLU came last, by a clear margin. That should make you suspicious of received wisdom applied out of context.

Worth noting how this was found: an earlier eight-run sample had ReLU and tanh tied at 88% and sigmoid at a flawless 100%. All three numbers were wrong, and the ordering of ReLU against tanh was wrong too. Small samples of a random process are how confident false beliefs get made — including in published work. Sixty runs cost seconds here; take them.

ReLU's real advantages are about **deep** networks: sigmoid and tanh squash their input into a narrow range, so gradients shrink as they pass back through many layers until early layers barely learn — the vanishing-gradient problem. ReLU passes positive values through untouched, so the signal survives depth. On a two-layer network solving four points, there is no depth for gradients to vanish through, and that advantage simply doesn't apply.

ReLU has a cost too: it outputs exactly zero for any negative input, so a unit pushed firmly negative receives no gradient and can stop learning altogether — a "dead" unit. With few units that's expensive, which is part of why narrow ReLU nets fail as often as they do above.

Use ReLU by default in deep networks because the reason for it holds there. Don't carry the habit into places where you can just measure.`,
        example: {
          label: "Try sigmoid hidden units on XOR",
          dsl: `network SigmoidMLP {
  dense 2 -> 8 activation=sigmoid
  dense 8 -> 1 activation=sigmoid
}
train dataset=xor lr=0.35 epochs=500`,
          expect:
            "Solves it in about 99 runs out of 100, landing from 50% to 100% across attempts \u2014 so train it more than once before judging. Swap the hidden activation to relu and repeat: it fails noticeably more often.",
        },
      },
      {
        heading: "Depth is not free either",
        body: `If one hidden layer helps, two should help more. Measured, at eight units per layer on XOR:

- **one ReLU hidden layer** — 7/8
- **two ReLU hidden layers** — 5/8

It got *worse*. More layers mean more ways for the random start to go wrong, more dead units, and a longer path for the gradient to travel — and XOR simply doesn't need the extra expressiveness, so all you've added is difficulty.

Depth pays when the problem genuinely has hierarchical structure: edges into shapes into objects. Four points in a plane do not. Reach for capacity when your model is failing because it's too simple, and check by measuring rather than assuming.`,
      },
      {
        heading: "Common traps",
        body: `- **"Adding layers adds power."** Only with a nonlinearity between them. Without one you've added parameters and nothing else — the 50% result above.
- **"It failed, so the architecture is wrong."** At narrow widths the same architecture succeeds and fails across runs on identical settings. Train three times before concluding anything.
- **"ReLU is always the right default."** It is a good default *in deep networks*, for a specific reason that doesn't apply to a two-layer toy.
- **Reading the hidden layer as human features.** Hidden units aren't "the vertical detector" and "the diagonal detector". They're whatever arrangement of lines gradient descent happened to find, usually redundant and rarely interpretable.`,
      },
      {
        heading: "Where this goes next",
        body: `You now have a model that can bend its boundary as much as you like — and that turns out to be a new problem rather than the end of them.

A network with enough capacity can carve a region around *every single training point*, scoring perfectly on the data you gave it while learning nothing transferable. Chapter 3 hands you a deliberately oversized network and lets you watch it do exactly that, then introduces the tools for reining it back in.`,
      },
    ],
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
    theory: `**Overfitting:** the training score improves while the model gets *worse* at data it hasn't seen.

This chapter uses \`noisy_moons\`, where some labels are deliberately flipped, and \`val=0.3\` to hold data back. Watch the two accuracy cards diverge.

**L2 regularization** adds λ‖w‖² to the loss, preferring smaller weights.

**L2** adds λ‖w‖² to the loss, preferring smaller weights and smoother boundaries.

Set \`l2=0.005\` above the train line and compare **held-out** accuracy — not the loss, which the penalty inflates by construction.`,
    lesson: [
      {
        heading: "A score of 94% that means nothing",
        body: `Train the chapter's starter network — three layers, 64 units wide — on \`noisy_moons\` and it reaches about **94% training accuracy**. Averaged over forty runs it lands between 84% and 100%.

If that were the whole story you'd ship it.

It isn't, and the reason is the number next to it. This chapter's dataset is \`noisy_moons\`: the same two crescents as before, except a substantial fraction of the labels have been **deliberately flipped**. Some points genuinely sit in the wrong crescent. No function can get those right and still describe the underlying shape — so a model scoring 94% on this data is not a model that understands crescents. It's a model that has memorised which specific points were mislabelled.`,
      },
      {
        heading: "The held-out split, and the only number that matters",
        body: `The \`val=0.3\` on the train line holds back 30% of the data. The network never trains on it; it is only ever scored on it. That gives you two accuracy cards instead of one:

- **Accuracy · train** — how well it does on data it has already seen
- **Accuracy · held-out** — how well it does on data it hasn't

For the starter network, measured over forty runs: training **94.3%**, held-out **63.3%**. A gap of **31 percentage points**, and the smallest gap in forty runs was still 13pp. This is not occasional bad luck — it happens every time.

The metrics panel fills in the band between the two accuracy curves for exactly this reason. That shaded area *is* the overfitting. Watch it open up as training proceeds.`,
        example: {
          label: "Watch it memorise the noise",
          dsl: `network OverfitDemo {
  dense 2 -> 64 activation=relu
  dense 64 -> 64 activation=relu
  dense 64 -> 1 activation=sigmoid
}
l2=0.0
train dataset=noisy_moons lr=0.08 epochs=400 val=0.3`,
          expect:
            "Training accuracy climbs to somewhere from 65% to 100% while held-out accuracy stalls at 63%, and the band between the two curves opens wide. The training number keeps improving long after the useful learning has stopped.",
        },
      },
      {
        heading: "Why capacity is what lets this happen",
        body: `A model can only memorise individual points if it has enough freedom to bend around them. Sixty-four units across two hidden layers is thousands of weights for eighty points — more than enough to carve a private pocket around every mislabelled one.

So the first and bluntest fix is to take that freedom away. Six hidden units cannot make thousands of independent bends; the best it can do is approximate the general shape.

Measured over forty runs, six hidden units gives training **79.1%** and held-out **68.8%**. Read that carefully: training accuracy fell by fifteen points, and held-out accuracy *rose* by five. You made the visible number worse on purpose and got a better model.

That trade is the whole subject. If you only ever look at training accuracy you will reject every fix that works.`,
        example: {
          label: "Shrink the network instead",
          dsl: `network SmallNet {
  dense 2 -> 6 activation=relu
  dense 6 -> 1 activation=sigmoid
}
train dataset=noisy_moons lr=0.3 epochs=250 val=0.3`,
          expect:
            "Training accuracy drops to somewhere from 50% to 95% — clearly worse than the wide net — while held-out accuracy rises to 69%. The gap between the two cards shrinks from around 31pp to roughly 10pp.",
        },
      },
      {
        heading: "L2: keeping the capacity, spending it more carefully",
        body: `Throwing away capacity is crude. Often you want a large model — you just don't want it using its size to memorise. **L2 regularisation** adds a penalty proportional to the sum of the squared weights:

loss = cross-entropy + λ · Σw²

Large weights are what let a network turn sharply. Penalising them biases training toward smoother functions, which can follow a broad trend but can't spike around one stray point. Note the shape of the incentive: λ doesn't forbid a large weight, it makes one *expensive*, so the model will only pay if the data really justifies it.

Same 64×64 network, \`l2=0.005\`: training **81.0%**, held-out **74.8%**. Held-out accuracy went from 63.3% to 74.8% — **eleven and a half points better** than the unregularised model, and better than shrinking the network achieved. The gap fell from 31pp to 6pp.

Keeping the capacity and constraining how it's used beat removing the capacity. That is usually, though not always, the way it goes.`,
        example: {
          label: "Regularise instead of shrinking",
          dsl: `network Regularised {
  dense 2 -> 64 activation=relu
  dense 64 -> 64 activation=relu
  dense 64 -> 1 activation=sigmoid
}
l2=0.005
train dataset=noisy_moons lr=0.12 epochs=300 val=0.3`,
          expect:
            "Training accuracy lands from 50% to 95%, held-out accuracy rises to 75% — the best held-out score of the three examples — and the accuracy band nearly closes. Compare the held-out card against the unregularised run: that difference is the entire point of the technique.",
        },
      },
      {
        heading: "A trap in the loss number itself",
        body: `Here is something that will confuse you if nobody warns you: **turning on L2 makes the displayed loss go up.**

That's not the model getting worse. Look again at the formula — the penalty term λ·Σw² is *part of the loss being reported*. Switch on L2 and you have added a quantity to the number on screen, by construction. The unregularised run shows a loss near zero partly because it has no penalty to pay and partly because it has memorised the answers.

So during regularisation work the loss cards are close to useless for comparing configurations. **Compare held-out accuracy instead.** It's the only figure here that means the same thing across different values of λ.`,
      },
      {
        heading: "Common traps",
        body: `- **"Training accuracy went up, so it's learning."** On noisy data past a certain point, rising training accuracy is the model absorbing noise. The held-out card is what tells you which is happening.
- **"More regularisation is safer."** Push λ far enough and the model gives up entirely — held-out accuracy collapses back toward chance because you have forbidden it from fitting anything at all. There is a middle, and you find it by measuring.
- **"The best run is the result."** Held-out accuracy on this dataset varies a lot between runs: the regularised setup ranges from about 62% to 85% depending on the random start. Comparing single runs of two configurations tells you almost nothing. Run each a few times.
- **"L2 raised my loss, so I'll revert it."** The penalty is inside the reported loss. Judge by held-out accuracy.
- **Tuning against the held-out set.** Once you've picked λ by watching held-out accuracy, that set has quietly become part of your training process. Real practice needs a third, untouched set for the final honest number — a distinction worth knowing exists even though this app has only two.`,
      },
      {
        heading: "Where this goes next",
        body: `You now have the discipline that separates working practice from tinkering: hold data back, judge on the part the model hasn't seen, and be willing to accept a worse visible score for a better real one.

Everything so far has treated inputs as a flat list of numbers — two coordinates, order irrelevant. Chapter 4 introduces data where **position matters**: a small image, where a pattern means the same thing wherever it appears. Fully connected layers handle that badly, and the fix is to stop giving every input its own private weight.`,
      },
    ],
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
              dataset: "noisy_moons",
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
              dataset: "noisy_moons",
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
              dataset: "noisy_moons",
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
