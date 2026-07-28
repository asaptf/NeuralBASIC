import { describe, expect, it } from "vitest";
import { CHAPTERS } from "@/curriculum/chapters";
import type { ExplainConceptGroup } from "@/curriculum/types";
import {
  gradeExplanation,
  gradePredictAnswer,
  type ExplainGradeResult,
} from "./explain";

const xorAndConcepts: ExplainConceptGroup[] = [
  {
    id: "linear_boundary",
    label: "linear boundary",
    synonyms: ["linear", "line", "straight line", "linearly"],
  },
  {
    id: "separability",
    label: "separability",
    synonyms: ["separable", "separability", "separate", "separation"],
  },
  {
    id: "xor_vs_and",
    label: "AND vs XOR",
    synonyms: ["xor", "and"],
  },
];

describe("gradeExplanation", () => {
  it("rejects keyword stuffing", () => {
    const stuffed = "linear linear linear linear linear linear linear linear";
    const r = gradeExplanation(stuffed, xorAndConcepts);
    expect(r.passed).toBe(false);
    expect(r.feedback.length).toBeGreaterThan(10);
    // Socratic — should not dump a full answer
    expect(r.feedback.toLowerCase()).not.toMatch(/because and is linearly separable/);
  });

  it("accepts a genuine one-sentence answer with synonyms/inflections", () => {
    const good =
      "AND is linearly separable so one line works, but XOR is not — a single line can't split those classes.";
    const r = gradeExplanation(good, xorAndConcepts);
    expect(r.passed).toBe(true);
    expect(r.score).toBeGreaterThan(0.5);
    expect(r.matchedConcepts.length).toBeGreaterThanOrEqual(2);
    expect(r.matchedConcepts).toEqual(
      expect.arrayContaining(["linear_boundary", "separability"])
    );
  });

  it("fails a single-concept answer on a multi-concept challenge", () => {
    const onlyOne =
      "I think the important idea here is something about a line somehow.";
    const r = gradeExplanation(onlyOne, xorAndConcepts);
    expect(r.passed).toBe(false);
    expect(r.matchedConcepts.length).toBeLessThan(2);
    expect(r.missingConcepts.length).toBeGreaterThan(0);
  });

  it("rejects empty padding with no concept content", () => {
    const pad =
      "well I guess it is what it is and things are things when they are things";
    const r = gradeExplanation(pad, xorAndConcepts);
    expect(r.passed).toBe(false);
  });

  it("returns structured fields the UI can render", () => {
    const r: ExplainGradeResult = gradeExplanation(
      "linearly separable boundary for and but not xor",
      xorAndConcepts
    );
    expect(typeof r.passed).toBe("boolean");
    expect(typeof r.score).toBe("number");
    expect(Array.isArray(r.matchedConcepts)).toBe(true);
    expect(Array.isArray(r.missingConcepts)).toBe(true);
    expect(typeof r.feedback).toBe("string");
  });
});

describe("gradePredictAnswer", () => {
  it("marks correct choice without leaking index in the nudge", () => {
    const r = gradePredictAnswer(1, 1, ["a", "b", "c"]);
    expect(r.correct).toBe(true);
    expect(r.nudge).not.toMatch(/\b1\b.*correct|correct.*\b1\b|index\s*1/i);
  });

  it("gives a Socratic nudge on wrong answers without revealing the answer", () => {
    const r = gradePredictAnswer(0, 2, ["a", "b", "c"]);
    expect(r.correct).toBe(false);
    expect(r.nudge.length).toBeGreaterThan(15);
    expect(r.nudge.toLowerCase()).not.toContain("option 3");
    expect(r.nudge.toLowerCase()).not.toContain("correct index");
    expect(r.nudge).toMatch(/observe|guess|true|pick|option/i);
  });
});

/** Plausible good answers per challenge explain step (for curriculum coverage). */
const PLAUSIBLE_ANSWERS: Record<string, string> = {
  "ch1-c1":
    "AND is linearly separable with a straight line, but XOR is not — one neuron can't split XOR.",
  "ch1-c2":
    "A very high learning rate saturates the sigmoid so the neuron becomes overconfident, and cross-entropy penalises those confident mistakes heavily, while accuracy only counts which side of the threshold each point falls on.",
  "ch1-c3":
    "XOR is not linearly separable, so a single neuron gets stuck and accuracy plateaus below perfect.",
  "ch2-c1":
    "Hidden layers with nonlinear activations reshape the space so XOR classes fall into separable regions.",
  "ch2-c2":
    "Sigmoid hidden units saturate and gradients vanish, so training is slower than with ReLU.",
  "ch2-c3":
    "I saw a curved, roughly circular decision boundary separating the ring region from the core.",
  "ch3-c1":
    "Overfitting shows up as a jagged, wiggly boundary that memorizes noise with high capacity.",
  "ch3-c2":
    "After L2 the boundary looked smoother and simpler because weights were pushed smaller by regularization.",
  "ch3-c3":
    "Optimization can fail when the learning rate is too small and the model gets stuck; generalization fails when we overfit the data.",
  "ch4-c1":
    "Weight sharing lets one kernel detect an edge at any position instead of learning a new filter per location.",
  "ch4-c2":
    "If the kernel is larger than the image it can't slide validly without padding and the spatial size becomes invalid.",
  "ch4-c3":
    "The image goes through conv feature maps, flatten turns them into a vector, then dense scores the class.",
  "ch5-c1":
    "A bright cell means a high attention weight — that token pair is strongly focused as relevant.",
  "ch5-c2":
    "Different heads can capture different relation patterns in parallel, like position vs sentiment syntax.",
  "ch5-c3":
    "A dense bag-of-features model mixes features independently, while attention models token interactions and context.",
};

describe("curriculum explain steps have gradable concept data", () => {
  it("every explain step in all 5 chapters has explainConcepts", () => {
    expect(CHAPTERS).toHaveLength(5);
    let explainCount = 0;
    for (const ch of CHAPTERS) {
      for (const c of ch.challenges) {
        for (const step of c.steps) {
          if (step.kind !== "explain") continue;
          explainCount++;
          expect(step.explainConcepts).toBeDefined();
          expect(step.explainConcepts!.length).toBeGreaterThanOrEqual(2);
          for (const g of step.explainConcepts!) {
            expect(g.id.length).toBeGreaterThan(0);
            expect(g.synonyms.length).toBeGreaterThan(0);
          }
        }
      }
    }
    expect(explainCount).toBeGreaterThanOrEqual(15);
  });

  it("grader accepts a plausible good answer for every explain step", () => {
    for (const ch of CHAPTERS) {
      for (const c of ch.challenges) {
        const step = c.steps.find((s) => s.kind === "explain");
        expect(step, c.id).toBeDefined();
        const answer = PLAUSIBLE_ANSWERS[c.id];
        expect(answer, `missing plausible answer for ${c.id}`).toBeDefined();
        const r = gradeExplanation(answer!, step!.explainConcepts ?? []);
        expect(r.passed, `${c.id}: ${r.feedback} (matched=${r.matchedConcepts})`).toBe(
          true
        );
      }
    }
  });

  it("keyword-stuffed answers fail every multi-concept explain step", () => {
    for (const ch of CHAPTERS) {
      for (const c of ch.challenges) {
        const step = c.steps.find((s) => s.kind === "explain");
        if (!step?.explainConcepts?.length) continue;
        const firstSyn = step.explainConcepts[0]!.synonyms[0]!;
        const stuffed = Array(12).fill(firstSyn).join(" ");
        const r = gradeExplanation(stuffed, step.explainConcepts);
        expect(r.passed, c.id).toBe(false);
      }
    }
  });
});
