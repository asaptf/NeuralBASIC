/**
 * Socratic Tutor — hard pedagogical policy.
 * Never pastes complete working solutions first.
 * Forces predict → experiment → observe → explain.
 */

export const TUTOR_SYSTEM_PROMPT = `You are the NeuralBASIC Socratic Tutor — a strict educational coach for learning neural networks by building them.

NON-NEGOTIABLE RULES:
1. NEVER paste a complete working solution (full DSL program, full weight values, or end-to-end fixed code) as your FIRST response to a request for answers.
2. ALWAYS force the loop: predict → experiment → observe → explain.
3. Ask short, sharp questions. Prefer one question at a time.
4. If the user asks "just give me the code" or "full answer", refuse the complete solution first. Offer a hint, a prediction question, or a minimal knob to turn (e.g. "what happens if you change lr?").
5. You may confirm correctness AFTER the user has attempted an experiment or explanation.
6. Use the user's current chapter, DSL, and metrics as context.
7. Celebrate productive struggle. Never shame mistakes.
8. Keep replies concise (2–5 short sentences or bullets).

You are not a ChatGPT code-writer. You are a lab partner from the 80s BASIC era: curious, demanding, kind.`;

export interface TutorContext {
  chapterId: string;
  chapterTitle: string;
  theorySnippet?: string;
  dsl: string;
  metrics?: { loss: number; accuracy: number; epoch?: number } | null;
  challengeId?: string | null;
  challengeStep?: string | null;
}

export interface TutorMessage {
  id: string;
  role: "user" | "tutor" | "system";
  content: string;
  ts: number;
}

const SOLUTION_REQUEST =
  /(give me (the )?(full |complete |entire )?(answer|solution|code|dsl|program)|just (give|tell|write|paste)|solve it for me|what is the (full )?code|write (the )?(whole|entire|complete)|paste (the )?(solution|code))/i;

const COMPLETE_SOLUTION_MARKERS =
  /network\s+\w+\s*\{[\s\S]*dense[\s\S]*train\s+dataset=/i;

export function isSolutionRequest(userText: string): boolean {
  return SOLUTION_REQUEST.test(userText);
}

export function looksLikeCompleteSolution(text: string): boolean {
  return COMPLETE_SOLUTION_MARKERS.test(text);
}

/** Deterministic mock tutor that obeys Socratic rules. */
export function mockTutorReply(
  userText: string,
  ctx: TutorContext,
  // history reserved for future multi-turn LLM adapters
  history: TutorMessage[] = []
): string {
  void history;
  const text = userText.trim();
  const lower = text.toLowerCase();

  if (isSolutionRequest(text)) {
    return socraticRefusal(ctx);
  }

  // If user pasted their own full program asking "is this right?"
  if (looksLikeCompleteSolution(text) && /right|correct|work|good|ok\??/i.test(text)) {
    return (
      `I see a full network definition — good that *you* wrote it. ` +
      `Before I comment: what accuracy do you observe after training, and does that match your prediction? ` +
      `(Chapter: ${ctx.chapterTitle})`
    );
  }

  if (/hello|hi\b|hey|start/i.test(lower)) {
    return (
      `Welcome to ${ctx.chapterTitle}. ` +
      `Before we train: what do you *expect* the loss to do in the first 20 epochs? ` +
      `Make a prediction, then hit Train.`
    );
  }

  if (/why.*(xor|fail|stuck)/i.test(lower)) {
    return (
      `Strong question. Draw the four XOR points in your head: can a single straight line separate the classes? ` +
      `Try dataset=and with the same single neuron, then dataset=xor. What changes?`
    );
  }

  if (/learning rate|lr\b/i.test(lower)) {
    return (
      `Learning rate scales each weight update. ` +
      `Predict: if you multiply lr by 10, will loss fall faster, oscillate, or explode? ` +
      `Change only \`lr\`, train, and tell me what the curve did.`
    );
  }

  if (/overfit|regular|l2/i.test(lower)) {
    return (
      `Overfitting is memorizing quirks instead of structure. ` +
      `What do you expect the decision boundary to look like with a huge hidden layer and l2=0? ` +
      `Then set l2=0.01 and compare. Describe the difference.`
    );
  }

  if (/conv|kernel|cnn|filter/i.test(lower)) {
    return (
      `A kernel is a tiny shared detector sliding over the image. ` +
      `If your vertical-bar detector is correct, where should its activation light up? ` +
      `Train on tiny_images and watch the visualizer — then explain one filter's job.`
    );
  }

  if (/attention|transformer|softmax/i.test(lower)) {
    return (
      `Attention redistributes focus: softmax turns scores into weights that sum to ~1. ` +
      `After one train run, open the attention heatmap. Which token pairs are bright? Why might that help sentiment?`
    );
  }

  if (/accuracy|loss|metric/i.test(lower)) {
    const m = ctx.metrics;
    if (m) {
      return (
        `You currently have loss=${m.loss.toFixed(4)}, accuracy=${(m.accuracy * 100).toFixed(1)}%` +
        (m.epoch != null ? ` at epoch ${m.epoch}` : "") +
        `. Is that better or worse than you predicted before training? What will you change next — architecture, lr, or data?`
      );
    }
    return `I don't have metrics yet. Train once in Immediate Mode, then ask me again about the numbers.`;
  }

  if (/help|stuck|hint/i.test(lower)) {
    return hintForChapter(ctx);
  }

  if (/explain|because|i think|my answer/i.test(lower)) {
    return (
      `I'm listening. Can you connect that explanation to something you *observed* in the visualizer or metrics? ` +
      `One concrete observation + one sentence of theory is a solid tutor check.`
    );
  }

  // Default Socratic nudge with context
  const dslLine = ctx.dsl.split("\n").find((l) => /dense|conv|attention|transformer/i.test(l));
  return (
    `You're in **${ctx.chapterTitle}**. ` +
    (dslLine ? `Your network includes: \`${dslLine.trim()}\`. ` : "") +
    `What's your next experiment — and what result would *falsify* your current hypothesis?`
  );
}

function socraticRefusal(ctx: TutorContext): string {
  return (
    `I won't paste a complete working solution first — that would rob you of the "aha". ` +
    `Socratic rule: predict → experiment → observe → explain.\n\n` +
    `Try this instead: (1) write down the output you expect for one input, ` +
    `(2) change a single knob (lr, a layer width, or dataset), ` +
    `(3) train and report loss/accuracy. ` +
    `Then I'll help you interpret.\n\n` +
    `Context: chapter **${ctx.chapterTitle}**.`
  );
}

function hintForChapter(ctx: TutorContext): string {
  switch (ctx.chapterId) {
    case "ch1":
      return `Hint (not a solution): a single neuron draws one line. Which datasets are linearly separable — AND, OR, or XOR? Predict, then switch \`dataset=\` and train.`;
    case "ch2":
      return `Hint: XOR needs a nonlinear hidden layer. Try one hidden dense with relu, then a sigmoid output. What accuracy do you predict before training?`;
    case "ch3":
      return `Hint: train the wide net on \`noisy_moons\` with l2=0 and compare the two accuracy cards — training against held-out. Then set l2=0.005 and look again. Which number got better, and which got worse?`;
    case "ch4":
      return `Hint: the path that works is conv2d → \`pool global=true\` → dense. Try replacing the pool with \`flatten\` and compare held-out accuracy — then explain what the pooling was doing for you.`;
    case "ch5":
      return `Hint: use \`transformer d_model=8 heads=2\` then a dense output on tiny_text. After training, read the attention heatmap before asking for architecture changes.`;
    default:
      return `Hint: change one variable only, train, observe metrics + visualizer, then explain.`;
  }
}

/** Ensure a tutor reply never starts with a complete solution dump. */
export function enforceSocratic(reply: string): string {
  if (looksLikeCompleteSolution(reply)) {
    return (
      `I almost handed you a full program — catching myself. ` +
      `Let's stay Socratic: what is the *smallest* change you can try next, and what do you expect to see?`
    );
  }
  return reply;
}

export function buildTutorContextPrompt(ctx: TutorContext): string {
  return [
    TUTOR_SYSTEM_PROMPT,
    "",
    "## Current context",
    `Chapter: ${ctx.chapterId} — ${ctx.chapterTitle}`,
    ctx.challengeId ? `Challenge: ${ctx.challengeId} (${ctx.challengeStep ?? ""})` : "",
    "DSL:",
    "```",
    ctx.dsl.slice(0, 2000),
    "```",
    ctx.metrics
      ? `Metrics: loss=${ctx.metrics.loss}, accuracy=${ctx.metrics.accuracy}, epoch=${ctx.metrics.epoch ?? "?"}`
      : "Metrics: none yet",
  ]
    .filter(Boolean)
    .join("\n");
}
