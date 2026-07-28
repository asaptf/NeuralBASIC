# NeuralBASIC (NeuronPad / AI Tablet)

**An interactive educational IDE for learning neural networks and modern AI** — the spiritual successor to classic QuickBASIC and [TabletBasic](https://github.com/asaptf/TabletBasic).

Users should feel the same joy of immediate experimentation that people felt with BASIC in the 1980s–90s, applied to neural networks. The app forces **active construction**, **productive struggle**, and **deep intuition**. It is **not** a passive video course and **not** a ChatGPT wrapper that writes code for you.

---

## Vision

- **Immediate Mode** — change architecture, learning rate, or data → instant visual + numerical feedback. No notebook-style “run cell”.
- **Live visualization is the primary teacher** — the Data Lab (dataset scatter + decision boundary) is the largest panel on screen; glowing neurons, weights, loss curves and attention heatmaps sit alongside it.
- **You watch it learn** — training advances epoch-by-epoch on an animation loop, with pause, `Step +1` and an epoch progress readout. Not a jump from “before” to “after”.
- **Progress gated by demonstrated understanding** — challenges + Socratic tutor checks. Explanations are graded across *distinct concepts*, so keyword stuffing does not unlock a chapter.
- **Tablet-first** IDE with keyboard shortcuts (Ctrl/⌘+Enter to train).

### Pedagogical rules (non-negotiable)

1. The built-in **AI Tutor is strictly Socratic**: it never pastes complete working solutions first.
2. The loop is always: **predict → experiment → observe → explain**.
3. Everything is Immediate Mode; metrics and the visualizer update from the same train path.
4. Curriculum progress unlocks when challenges are completed, not when videos are watched.

---

## Quick start

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

```bash
npm test          # engine, tutor graders, persistence, curriculum completeness
npm run build     # production build
npm run typecheck
```

**Definition of Done path (Chapter 1):**

1. Open the app (Modern Dark or Retro Blue theme).
2. Chapter 1 starter is a single dense perceptron on XOR/AND.
3. Press **Train ▶** (or Ctrl/⌘+Enter) — the boundary sweeps across the Data Lab as the loss curve fills in. Use **Pause** / **Step +1** to inspect a single epoch.
4. Switch `dataset` between `and` and `xor`: AND ends at 0/4 misclassified, XOR sticks at 2/4 with the failing points ringed. That contrast *is* the lesson.
5. Chat with the **Socratic Tutor** (ask for a full solution — it will refuse and redirect).
6. Complete a challenge: predict → experiment → explain. A wrong prediction gets a nudge, not the answer; a hand-wavy explanation gets sent back.
7. **Export Model** → JSON weights + PyTorch-equivalent snippet.

---

## Tech stack (MVP)

| Area | Choice |
|------|--------|
| App | Next.js 15, React, TypeScript, Tailwind CSS |
| State | Zustand |
| Editor | Monaco |
| Neural runtime | **Pure TypeScript educational engine** (`src/engine`) — fully client-side train/infer, CPU path, no WebGPU required |
| Persistence | `localStorage` + JSON import/export |
| Tutor | Deterministic Socratic mock + hard system prompt (optional LLM later) |
| Tests | Vitest |

### Why a pure TS engine (not TF.js in MVP)?

Transparency for teaching: every weight, activation, and attention map is a plain array you can visualize each tick. Unit tests run in Node without WebGPU. The architecture keeps a clear model graph so a TensorFlow.js / WebGPU backend can be swapped behind the same `createAndTrain` / `predict` surface later.

---

## Architecture

```
src/
  engine/           # DSL parser, datasets, layers, train, export
  curriculum/       # 5 chapters + challenges (data-driven, concept-tagged)
  tutor/            # Socratic system prompt, mock replies, offline graders
  store/            # Zustand Immediate Mode store + animated training loop
  components/       # IDE shell, editor, lab, network, metrics, tutor, curriculum
  lib/              # persistence, utils
  app/              # Next.js App Router entry
```

### Data flow (Immediate Mode)

1. User edits **DSL** or Immediate controls (LR / epochs / dataset).
2. **Train** parses DSL → builds model → opens a `TrainingSession` (`createTrainingSession`). The store drives `runEpoch()` from a `requestAnimationFrame` loop paced against the wall clock, so a run animates over ~2.5s whatever the epoch count — and collapses to a single synchronous run under `prefers-reduced-motion`.
3. Each epoch's `TrainStepResult` feeds **Metrics** (autoscaled loss + accuracy vs chance), the **Data Lab** (boundary grid + scatter + misclassified points), the **Network** panel (activations/weights/attention), and challenge **experiment gates**.
4. Tutor receives chapter id + DSL + last metrics; replies under Socratic policy. Predict and explain steps are graded offline by `@/tutor` — deterministic, no network call.

### NeuralBASIC DSL (minimal)

```text
network Perceptron {
  dense 2 -> 1 activation=sigmoid
}
train dataset=xor lr=0.8 epochs=200
```

Supported layers: `dense`, `conv2d`, `flatten`, `attention`, `transformer`.  
Datasets: `xor`, `and`, `or`, `linear`, `moons`, `circles`, `spiral`, `tiny_images`, `tiny_text`.  
Regularizer: `l2=0.01`.

**The parser rejects what it doesn't understand**, and says where:

```text
Line 2: Unknown activation `banana`. Valid activations: linear, sigmoid, relu, tanh, softmax.
```

This is a pedagogical requirement, not just hygiene. The parser used to accept anything and silently
fall back to a default `dense 2 -> 1` on `xor` — so a learner who typo'd `activaton=relu` got a
plausible result from a network they never wrote, and drew a confident wrong conclusion. Under
**predict → experiment → observe**, a silent default is worse than a crash: it corrupts the
observation. Errors surface in a strip under the editor and name the valid alternatives.

---

## Curriculum

| # | Chapter | Focus |
|---|---------|--------|
| 1 | Single Neuron / Perceptron | Linear boundaries, AND/OR vs XOR |
| 2 | MLP + activations | Hidden layers, solve XOR, moons/circles |
| 3 | Overfitting & regularization | Capacity, L2, failure modes |
| 4 | Convolutional basics | Kernels on 4×4 bar patterns |
| 5 | Attention + tiny Transformer | Softmax attention, heatmaps, tiny text |

Each chapter has **2–3 challenges** with **predict → experiment → explain** steps. Later chapters unlock when the previous chapter’s challenges are completed.

Because a locked chapter hides its own bugs until a learner has already earned their way in, the
curriculum is covered by tests rather than by clicking: `src/curriculum/curriculum.completeness.test.ts`
proves — for all five chapters — that every starter program trains, every experiment gate is
actually winnable, every `explain` step accepts a genuine answer and rejects keyword stuffing, and
that the unlock chain can reach Chapter 5. Gate evaluation lives in `src/curriculum/gates.ts` and is
shared by the store and those tests, so progression can't drift from what the tests assert.

---

## Themes

- **Modern Dark** — glowing neurons, cyber aesthetic (`data-theme="modern"`).
- **Retro Blue** — QuickBASIC / DOS blue-screen nostalgia (`data-theme="retro"`).

---

## Export

- **Save / Load** — browser `localStorage`.
- **Export JSON** — full experiment (DSL + config + weights + history).
- **Export Model** — `neuralbasic-model-v1` JSON + a simple **PyTorch-equivalent** training sketch.

---

## Inspiration

- [TabletBasic](https://github.com/asaptf/TabletBasic)
- TensorFlow Playground
- Transformer Explainer
- Andrej Karpathy — Neural Networks: Zero to Hero
- fast.ai pedagogical style

---

## License

[Apache License 2.0](LICENSE) — © 2026 Andrey Sapunov. Use, modify and redistribute freely,
including commercially; the license also grants patent rights and asks that you state your changes.
