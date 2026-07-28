import { describe, expect, it } from "vitest";
import {
  TUTOR_SYSTEM_PROMPT,
  buildTutorContextPrompt,
  enforceSocratic,
  isSolutionRequest,
  looksLikeCompleteSolution,
  mockTutorReply,
  type TutorContext,
} from "./socratic";
import { CHAPTERS } from "@/curriculum/chapters";

const ctx: TutorContext = {
  chapterId: "ch1",
  chapterTitle: "Single Neuron / Perceptron",
  dsl: `network Perceptron {\n  dense 2 -> 1 activation=sigmoid\n}\ntrain dataset=xor lr=0.5 epochs=100\n`,
  metrics: { loss: 0.5, accuracy: 0.5, epoch: 10 },
};

describe("Socratic tutor policy", () => {
  it("system prompt forbids pasting complete solutions first", () => {
    expect(TUTOR_SYSTEM_PROMPT).toMatch(/NEVER paste a complete working solution/i);
    expect(TUTOR_SYSTEM_PROMPT).toMatch(/predict/i);
  });

  it("detects solution-seeking user messages", () => {
    expect(isSolutionRequest("just give me the full solution")).toBe(true);
    expect(isSolutionRequest("paste the code")).toBe(true);
    expect(isSolutionRequest("what is learning rate?")).toBe(false);
  });

  it("refuses complete solutions when user asks for the full answer", () => {
    const reply = mockTutorReply("give me the full solution / complete code now", ctx);
    expect(looksLikeCompleteSolution(reply)).toBe(false);
    expect(reply.toLowerCase()).toMatch(/won't paste|socratic|predict|experiment/);
    expect(reply).not.toMatch(/network\s+\w+\s*\{[\s\S]*train\s+dataset=/i);
  });

  it("injects chapter + code context into prompt builder", () => {
    const prompt = buildTutorContextPrompt(ctx);
    expect(prompt).toContain("ch1");
    expect(prompt).toContain("Single Neuron");
    expect(prompt).toContain("dense 2 -> 1");
    expect(prompt).toContain(TUTOR_SYSTEM_PROMPT.slice(0, 40));
  });

  it("enforceSocratic strips accidental full program dumps", () => {
    const bad =
      "Sure!\nnetwork Foo {\n  dense 2 -> 1 activation=sigmoid\n}\ntrain dataset=xor lr=0.5 epochs=100\n";
    const fixed = enforceSocratic(bad);
    expect(looksLikeCompleteSolution(fixed)).toBe(false);
  });
});

describe("curriculum structure", () => {
  it("has five chapters each with ≥2 challenges and predict/experiment/explain", () => {
    expect(CHAPTERS).toHaveLength(5);
    for (const ch of CHAPTERS) {
      expect(ch.challenges.length).toBeGreaterThanOrEqual(2);
      for (const c of ch.challenges) {
        const kinds = c.steps.map((s) => s.kind);
        expect(kinds).toContain("predict");
        expect(kinds).toContain("experiment");
        expect(kinds).toContain("explain");
      }
      expect(ch.starterDSL.length).toBeGreaterThan(10);
    }
  });
});
