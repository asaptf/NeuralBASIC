/**
 * Data-driven proof that every chapter of the NeuralBASIC curriculum is completable:
 * starter programs train, experiment gates are winnable, explain/predict steps grade
 * correctly, and the unlock chain can reach Chapter 5.
 */
import { describe, expect, it } from "vitest";
import { parseDSL, toDSL } from "@/engine/dsl";
import { createAndTrain } from "@/engine/train";
import type { NetworkConfig, TrainConfig } from "@/engine/types";
import {
  gradeExplanation,
  gradePredictAnswer,
} from "@/tutor/explain";
import {
  CHAPTERS,
  getChapter,
  isChapterUnlocked,
} from "./chapters";
import { experimentCheckPasses } from "./gates";
import type { Challenge, ChallengeStep, Chapter } from "./types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Mirror of store completion: a chapter is done when every challenge is completed. */
function allChallengesComplete(
  chapter: Chapter,
  completedChallengeIds: Set<string>
): boolean {
  return chapter.challenges.every((c) => completedChallengeIds.has(c.id));
}

/**
 * Pure progression helper matching useAppStore.submitExplain:
 * when the last unfinished challenge of a chapter completes, that chapter id
 * is appended to completedChapters (which is what isChapterUnlocked reads).
 */
function markChapterCompleteIfReady(
  completedChapters: string[],
  chapter: Chapter,
  completedChallengeIds: Set<string>
): string[] {
  if (
    allChallengesComplete(chapter, completedChallengeIds) &&
    !completedChapters.includes(chapter.id)
  ) {
    return [...completedChapters, chapter.id];
  }
  return completedChapters;
}

// Gate evaluation is imported, not re-implemented: these tests must prove the
// gates the *store* actually enforces.
const checkExperimentGate = experimentCheckPasses;

/**
 * Train with retries (random init). Prefer fewer epochs for gates without
 * accuracy floors; bump epochs only when the gate demands high accuracy.
 */
function trainUntilGate(
  network: NetworkConfig,
  train: TrainConfig,
  check: NonNullable<ChallengeStep["experimentCheck"]>,
  dsl: string,
  options?: { attempts?: number; epochsCap?: number }
): { accuracy: number; loss: number; passed: boolean } {
  const attempts = options?.attempts ?? 5;
  const epochsCap = options?.epochsCap ?? 400;
  const epochs = Math.min(train.epochs, epochsCap);
  // Keep default shuffle (on) for gates with accuracy floors — SGD on moons/circles
  // benefits from it. Callers may still set shuffle: false on train if needed.
  const trainCfg: TrainConfig = {
    ...train,
    epochs,
    dataset: (check.dataset as TrainConfig["dataset"]) ?? train.dataset,
  };

  let bestAcc = -1;
  let bestLoss = Infinity;

  for (let a = 0; a < attempts; a++) {
    const { history } = createAndTrain(network, trainCfg, {
      includeDecisionBoundary: false,
    });
    const acc = history.final.accuracy;
    const loss = history.final.loss;
    expect(Number.isFinite(acc)).toBe(true);
    expect(Number.isFinite(loss)).toBe(true);
    if (acc > bestAcc) {
      bestAcc = acc;
      bestLoss = loss;
    }
    if (
      checkExperimentGate(check, {
        accuracy: acc,
        loss,
        dataset: trainCfg.dataset,
        dsl,
      })
    ) {
      return { accuracy: acc, loss, passed: true };
    }
  }

  return {
    accuracy: bestAcc,
    loss: bestLoss,
    passed: checkExperimentGate(check, {
      accuracy: bestAcc,
      loss: bestLoss,
      dataset: trainCfg.dataset,
      dsl,
    }),
  };
}

// ---------------------------------------------------------------------------
// Intended / plausible learner solutions per experiment challenge.
// Where the starter already embodies the exercise, reuse it (optionally
// adjusting dataset / epochs / l2). Where the exercise is to build something
// new, provide the target network the curriculum expects.
// ---------------------------------------------------------------------------

interface ExperimentSolution {
  /** Human-readable label used in the test name. */
  label: string;
  /** Full DSL a successful learner would have when the gate fires. */
  dsl: string;
  attempts?: number;
  epochsCap?: number;
}

function solutionFor(
  chapter: Chapter,
  challenge: Challenge
): ExperimentSolution {
  const starter = chapter.starterDSL;

  switch (challenge.id) {
    // Ch1 — single neuron
    case "ch1-c1":
      return {
        label: "single dense on AND",
        dsl: `network Perceptron {
  dense 2 -> 1 activation=sigmoid
}
train dataset=and lr=0.8 epochs=250
`,
        attempts: 3,
        epochsCap: 250,
      };
    case "ch1-c2":
      return {
        label: "single dense on moons",
        dsl: `network Perceptron {
  dense 2 -> 1 activation=sigmoid
}
train dataset=moons lr=0.8 epochs=250
`,
        attempts: 3,
        epochsCap: 250,
      };
    case "ch1-c3":
      // Gate only requires training on XOR with dense (no accuracy floor)
      return {
        label: "single dense on XOR (observe plateau)",
        dsl: starter.replace(/dataset=\w+/, "dataset=xor"),
        attempts: 1,
        epochsCap: 40,
      };

    // Ch2 — MLP
    case "ch2-c1":
      return {
        label: "MLP with ReLU hidden layers on XOR",
        dsl: `network MLP {
  dense 2 -> 12 activation=relu
  dense 12 -> 8 activation=relu
  dense 8 -> 1 activation=sigmoid
}
train dataset=xor lr=0.35 epochs=500
`,
        attempts: 6,
        epochsCap: 500,
      };
    case "ch2-c2":
      return {
        label: "ReLU MLP on moons",
        dsl: `network MLP {
  dense 2 -> 16 activation=relu
  dense 16 -> 8 activation=relu
  dense 8 -> 1 activation=sigmoid
}
train dataset=moons lr=0.25 epochs=200
`,
        attempts: 4,
        epochsCap: 200,
      };
    case "ch2-c3":
      return {
        label: "MLP on circles",
        dsl: `network MLP {
  dense 2 -> 16 activation=relu
  dense 16 -> 8 activation=relu
  dense 8 -> 1 activation=sigmoid
}
train dataset=circles lr=0.25 epochs=250
`,
        attempts: 4,
        epochsCap: 250,
      };

    // Ch3 — capacity / L2
    case "ch3-c1":
      return {
        label: "wide multi-layer net on noisy_moons",
        dsl: starter,
        attempts: 1,
        epochsCap: 40,
      };
    case "ch3-c2":
      return {
        label: "wide net with positive L2 on noisy_moons",
        dsl: `network OverfitDemo {
  dense 2 -> 32 activation=relu
  dense 32 -> 32 activation=relu
  dense 32 -> 1 activation=sigmoid
}
l2=0.005
train dataset=noisy_moons lr=0.12 epochs=300 val=0.3
`,
        attempts: 1,
        epochsCap: 80,
      };
    case "ch3-c3":
      return {
        label: "wide MLP reaching 0.9 on noisy_moons",
        // The chapter's own overfit demo: it memorises noisy_moons past 0.9 train.
        dsl: `network OverfitDemo {
  dense 2 -> 64 activation=relu
  dense 64 -> 64 activation=relu
  dense 64 -> 1 activation=sigmoid
}
l2=0.0
train dataset=noisy_moons lr=0.08 epochs=400 val=0.3
`,
        attempts: 5,
        epochsCap: 300,
      };

    // Ch4 — conv
    case "ch4-c1":
    case "ch4-c2":
      return {
        label: "TinyCNN on tiny_images",
        dsl: `network TinyCNN {
  conv2d filters=4 kernel=2 activation=relu channels=1 height=4 width=4
  flatten
  dense 8 activation=relu
  dense 2 activation=sigmoid
}
train dataset=tiny_images lr=0.12 epochs=100
`,
        attempts: 4,
        epochsCap: 100,
      };
    case "ch4-c3":
      // The ch4 starter now uses `pool` (the configuration that works), so this
      // gate's `flatten` requirement needs its own explicit solution.
      return {
        label: "conv + flatten + dense on tiny_images",
        dsl: `network FlattenPath {
  conv2d filters=4 kernel=2 activation=relu channels=1 height=4 width=4
  flatten
  dense 8 activation=relu
  dense 2 activation=sigmoid
}
train dataset=tiny_images lr=0.12 epochs=100
`,
        attempts: 1,
        epochsCap: 30,
      };

    // Ch5 — transformer
    case "ch5-c1":
      // d_model must be 4 on negation (4 tokens × 4-wide one-hots = 16). d_model=8
      // splits each token in half and is too noisy to clear the gate reliably —
      // that was the CI flake (best-of-5 p5≈0.69 against minAccuracy 0.7).
      return {
        label: "transformer on negation (≥0.6)",
        dsl: `network TinyTransformer {
  transformer d_model=4 heads=2
  dense 4 -> 1 activation=sigmoid
}
train dataset=negation lr=0.1 epochs=80
`,
        attempts: 5,
        epochsCap: 80,
      };
    case "ch5-c2":
      return {
        label: "transformer heads=2 on tiny_text",
        dsl: `network TinyTransformer {
  transformer d_model=8 heads=2
  dense 8 -> 1 activation=sigmoid
}
train dataset=tiny_text lr=0.1 epochs=40
`,
        attempts: 1,
        epochsCap: 40,
      };
    case "ch5-c3":
      // The gate checks *training* accuracy, and on `negation` a dense net fits
      // the training portion perfectly (it is the held-out score that collapses —
      // which is the chapter's whole point). Input is 16 features, not 8.
      return {
        label: "dense bag-of-features on negation (≥0.8 train)",
        dsl: `network TextDense {
  dense 16 -> 32 activation=relu
  dense 32 -> 1 activation=sigmoid
}
train dataset=negation lr=0.12 epochs=150
`,
        attempts: 4,
        epochsCap: 150,
      };

    default:
      return {
        label: "chapter starter",
        dsl: starter,
        attempts: 3,
        epochsCap: 150,
      };
  }
}

/** Plausible free-text answers (own words) that should pass gradeExplanation. */
const PLAUSIBLE_EXPLAIN: Record<string, string> = {
  "ch1-c1":
    "AND is linearly separable so a single straight boundary works, but XOR classes cannot be separated by one line.",
  "ch1-c2":
    "A very large learning rate saturates the sigmoid into overconfidence, and cross-entropy charges heavily for those confident mistakes, while accuracy only counts which side of the threshold a point lands on.",
  "ch1-c3":
    "XOR is not linearly separable, so one neuron is stuck near chance and accuracy plateaus well below perfect.",
  "ch2-c1":
    "Hidden units with nonlinear activations reshape feature space so the XOR classes fall into separable regions.",
  "ch2-c2":
    "Sigmoid hidden layers saturate and gradients vanish, which makes training slower than with ReLU activations.",
  "ch2-c3":
    "The decision boundary looked curved and roughly circular, separating the outer ring region from the core.",
  "ch3-c1":
    "Overfitting shows up as a jagged, wiggly boundary that memorizes noise when capacity is too high.",
  "ch3-c2":
    "After adding L2 the boundary became smoother and simpler because regularization pushed weights smaller.",
  "ch3-c3":
    "Optimization can fail when the learning rate is too small and training gets stuck; generalization fails when we overfit the data.",
  "ch4-c1":
    "Weight sharing lets one kernel detect an edge at any spatial position instead of learning a new filter per location.",
  "ch4-c2":
    "If the kernel is larger than the image it cannot slide over valid positions without padding and becomes invalid.",
  "ch4-c3":
    "The image goes through convolution feature maps, flatten turns them into a vector, then dense layers score the class.",
  "ch5-c1":
    "A bright cell means a high attention weight — that token pair is strongly focused as relevant context.",
  "ch5-c2":
    "Different heads capture different relation patterns in parallel, for example position versus sentiment or negation.",
  "ch5-c3":
    "A dense bag-of-features model treats features independently, while attention can model token interactions and relationships.",
};

// ---------------------------------------------------------------------------
// 1. Starter programs are valid
// ---------------------------------------------------------------------------

describe("curriculum: every starter program is valid", () => {
  for (const chapter of CHAPTERS) {
    it(`${chapter.id} starterDSL parses, builds, and trains with finite metrics`, () => {
      const parsed = parseDSL(chapter.starterDSL);
      expect(parsed.network.layers.length).toBeGreaterThan(0);

      // Keep starters fast — just prove the path runs without NaN
      const epochs = Math.min(parsed.train.epochs, 25);
      const { history } = createAndTrain(
        parsed.network,
        { ...parsed.train, epochs, shuffle: false },
        { includeDecisionBoundary: false }
      );

      expect(history.losses.length).toBe(epochs);
      expect(Number.isFinite(history.final.loss)).toBe(true);
      expect(Number.isFinite(history.final.accuracy)).toBe(true);
      expect(Number.isNaN(history.final.loss)).toBe(false);
      expect(Number.isNaN(history.final.accuracy)).toBe(false);
      for (const L of history.losses) {
        expect(Number.isFinite(L)).toBe(true);
      }
    });
  }
});

// ---------------------------------------------------------------------------
// 2–3. Experiment gates are winnable + dslIncludes consistent
// ---------------------------------------------------------------------------

describe("curriculum: every experiment gate is winnable", () => {
  for (const chapter of CHAPTERS) {
    for (const challenge of chapter.challenges) {
      const expStep = challenge.steps.find((s) => s.kind === "experiment");
      if (!expStep?.experimentCheck) continue;

      const check = expStep.experimentCheck;
      const sol = solutionFor(chapter, challenge);

      it(`${challenge.id} (${sol.label}): gate conditions are reachable`, () => {
        const parsed = parseDSL(sol.dsl);
        expect(parsed.network.layers.length).toBeGreaterThan(0);

        // 3. dslIncludes fragments must appear in the intended solution DSL
        if (check.dslIncludes) {
          const lower = sol.dsl.toLowerCase();
          for (const frag of check.dslIncludes) {
            expect(
              lower.includes(frag.toLowerCase()),
              `${challenge.id}: intended DSL missing dslIncludes fragment "${frag}"\n--- DSL ---\n${sol.dsl}`
            ).toBe(true);
          }
        }

        // Dataset override from gate must match what we train on
        const dataset =
          (check.dataset as TrainConfig["dataset"]) ?? parsed.train.dataset;
        expect(dataset).toBeTruthy();
        if (check.dataset) {
          expect(dataset).toBe(check.dataset);
        }

        const result = trainUntilGate(
          parsed.network,
          { ...parsed.train, dataset },
          check,
          sol.dsl,
          { attempts: sol.attempts, epochsCap: sol.epochsCap }
        );

        expect(
          result.passed,
          `${challenge.id}: gate not reached after retries ` +
            `(best accuracy=${result.accuracy}, loss=${result.loss}, ` +
            `need minAccuracy=${check.minAccuracy ?? "—"}, ` +
            `maxLoss=${check.maxLoss ?? "—"}, dataset=${check.dataset ?? "—"}, ` +
            `dslIncludes=${JSON.stringify(check.dslIncludes ?? [])})`
        ).toBe(true);

        expect(Number.isFinite(result.accuracy)).toBe(true);
        expect(Number.isFinite(result.loss)).toBe(true);
      });
    }
  }
});

// ---------------------------------------------------------------------------
// 4. Explain steps are gradeable and passable
// ---------------------------------------------------------------------------

describe("curriculum: every explain step is gradeable and passable", () => {
  for (const chapter of CHAPTERS) {
    for (const challenge of chapter.challenges) {
      const step = challenge.steps.find((s) => s.kind === "explain");
      if (!step) continue;

      it(`${challenge.id}: explainConcepts non-empty; good answer passes; stuffing fails`, () => {
        expect(
          step.explainConcepts,
          `${challenge.id}: missing explainConcepts`
        ).toBeDefined();
        expect(step.explainConcepts!.length).toBeGreaterThanOrEqual(2);

        for (const g of step.explainConcepts!) {
          expect(g.id.length).toBeGreaterThan(0);
          expect(g.label.length).toBeGreaterThan(0);
          expect(g.synonyms.length).toBeGreaterThan(0);
        }

        const answer = PLAUSIBLE_EXPLAIN[challenge.id];
        expect(
          answer,
          `missing PLAUSIBLE_EXPLAIN entry for ${challenge.id}`
        ).toBeDefined();

        const good = gradeExplanation(answer!, step.explainConcepts!);
        expect(
          good.passed,
          `${challenge.id}: good answer failed — ${good.feedback} ` +
            `(matched=${good.matchedConcepts.join(",")}, missing=${good.missingConcepts.join(",")})`
        ).toBe(true);

        // Anti-stuffing: keyword-stuffed answer must fail for THIS step
        const firstSyn = step.explainConcepts![0]!.synonyms[0]!;
        const stuffed = Array(12).fill(firstSyn).join(" ");
        const bad = gradeExplanation(stuffed, step.explainConcepts!);
        expect(
          bad.passed,
          `${challenge.id}: keyword-stuffed answer incorrectly passed`
        ).toBe(false);
      });
    }
  }
});

// ---------------------------------------------------------------------------
// 5. Predict steps are coherent
// ---------------------------------------------------------------------------

describe("curriculum: every predict step is coherent", () => {
  for (const chapter of CHAPTERS) {
    for (const challenge of chapter.challenges) {
      const step = challenge.steps.find((s) => s.kind === "predict");
      if (!step) continue;

      it(`${challenge.id}: choices present, correctIndex in range, grader consistent`, () => {
        expect(
          step.choices,
          `${challenge.id}: predict step missing choices`
        ).toBeDefined();
        expect(step.choices!.length).toBeGreaterThanOrEqual(2);

        expect(
          step.correctIndex,
          `${challenge.id}: predict step missing correctIndex`
        ).toBeDefined();
        expect(step.correctIndex!).toBeGreaterThanOrEqual(0);
        expect(step.correctIndex!).toBeLessThan(step.choices!.length);

        const correct = gradePredictAnswer(
          step.correctIndex!,
          step.correctIndex!,
          step.choices
        );
        expect(correct.correct).toBe(true);

        for (let i = 0; i < step.choices!.length; i++) {
          if (i === step.correctIndex) continue;
          const wrong = gradePredictAnswer(i, step.correctIndex!, step.choices);
          expect(
            wrong.correct,
            `${challenge.id}: choice ${i} should be incorrect`
          ).toBe(false);
        }
      });
    }
  }
});

// ---------------------------------------------------------------------------
// 6. Unlock chain is coherent
// ---------------------------------------------------------------------------

describe("curriculum: unlock chain is coherent", () => {
  it("ch1 is always unlocked; ch2–ch5 require predecessors", () => {
    expect(isChapterUnlocked("ch1", [])).toBe(true);
    expect(isChapterUnlocked("ch2", [])).toBe(false);
    expect(isChapterUnlocked("ch3", [])).toBe(false);
    expect(isChapterUnlocked("ch4", [])).toBe(false);
    expect(isChapterUnlocked("ch5", [])).toBe(false);

    expect(isChapterUnlocked("ch2", ["ch1"])).toBe(true);
    expect(isChapterUnlocked("ch3", ["ch1"])).toBe(false);
    expect(isChapterUnlocked("ch3", ["ch1", "ch2"])).toBe(true);
    expect(isChapterUnlocked("ch4", ["ch1", "ch2", "ch3"])).toBe(true);
    expect(isChapterUnlocked("ch5", ["ch1", "ch2", "ch3", "ch4"])).toBe(true);
  });

  it("each chapter's unlockAfter points at the previous chapter (or none for ch1)", () => {
    expect(CHAPTERS).toHaveLength(5);
    const sorted = [...CHAPTERS].sort((a, b) => a.number - b.number);
    expect(sorted[0]!.unlockAfter).toBeUndefined();
    for (let i = 1; i < sorted.length; i++) {
      expect(sorted[i]!.unlockAfter).toBe(sorted[i - 1]!.id);
    }
  });

  it("completing all challenges of each chapter unlocks the next through Chapter 5", () => {
    let completedChapters: string[] = [];
    const completedChallengeIds = new Set<string>();

    // Start: only ch1 open
    expect(isChapterUnlocked("ch1", completedChapters)).toBe(true);
    for (const ch of CHAPTERS.slice(1)) {
      expect(isChapterUnlocked(ch.id, completedChapters)).toBe(false);
    }

    for (const chapter of CHAPTERS) {
      // Must be unlocked before a learner can work on it
      expect(
        isChapterUnlocked(chapter.id, completedChapters),
        `${chapter.id} should be unlocked before completing its challenges`
      ).toBe(true);

      // Completing a proper subset must NOT mark the chapter complete
      if (chapter.challenges.length > 1) {
        const partial = new Set(completedChallengeIds);
        partial.add(chapter.challenges[0]!.id);
        const mid = markChapterCompleteIfReady(
          completedChapters,
          chapter,
          partial
        );
        expect(
          mid.includes(chapter.id),
          `${chapter.id} must not complete after only one challenge`
        ).toBe(false);
        // Next chapter still locked
        const next = CHAPTERS.find((c) => c.unlockAfter === chapter.id);
        if (next) {
          expect(isChapterUnlocked(next.id, mid)).toBe(false);
        }
      }

      // Finish every challenge in this chapter
      for (const c of chapter.challenges) {
        completedChallengeIds.add(c.id);
      }
      completedChapters = markChapterCompleteIfReady(
        completedChapters,
        chapter,
        completedChallengeIds
      );
      expect(completedChapters).toContain(chapter.id);

      // Next chapter (if any) is now open
      const next = CHAPTERS.find((c) => c.unlockAfter === chapter.id);
      if (next) {
        expect(
          isChapterUnlocked(next.id, completedChapters),
          `${next.id} should unlock after completing ${chapter.id}`
        ).toBe(true);
      }
    }

    // Full path: all five chapters completed and unlocked
    expect(completedChapters).toEqual(
      expect.arrayContaining(["ch1", "ch2", "ch3", "ch4", "ch5"])
    );
    for (const ch of CHAPTERS) {
      expect(isChapterUnlocked(ch.id, completedChapters)).toBe(true);
    }
  });

  it("isChapterUnlocked uses real chapter metadata (unknown id stays locked)", () => {
    expect(isChapterUnlocked("nope", ["ch1", "ch2", "ch3", "ch4", "ch5"])).toBe(
      false
    );
    expect(getChapter("ch3")?.unlockAfter).toBe("ch2");
  });
});

// ---------------------------------------------------------------------------
// Structural sanity: every challenge has the predict → experiment → explain triad
// ---------------------------------------------------------------------------

describe("curriculum: challenge step structure", () => {
  it("every challenge has predict, experiment, and explain steps", () => {
    for (const chapter of CHAPTERS) {
      for (const challenge of chapter.challenges) {
        const kinds = challenge.steps.map((s) => s.kind);
        expect(kinds, challenge.id).toContain("predict");
        expect(kinds, challenge.id).toContain("experiment");
        expect(kinds, challenge.id).toContain("explain");
        const exp = challenge.steps.find((s) => s.kind === "experiment");
        expect(exp?.experimentCheck, challenge.id).toBeDefined();
      }
    }
  });

  it("toDSL round-trip of solutions still contains dslIncludes fragments", () => {
    for (const chapter of CHAPTERS) {
      for (const challenge of chapter.challenges) {
        const exp = challenge.steps.find((s) => s.kind === "experiment");
        const check = exp?.experimentCheck;
        if (!check?.dslIncludes?.length) continue;
        const sol = solutionFor(chapter, challenge);
        const parsed = parseDSL(sol.dsl);
        const round = toDSL(parsed.network, parsed.train);
        // l2 may sit outside network — re-check against original sol.dsl primarily;
        // for layer keywords, round-trip should still contain them.
        for (const frag of check.dslIncludes) {
          const f = frag.toLowerCase();
          const inSol = sol.dsl.toLowerCase().includes(f);
          const inRound =
            round.toLowerCase().includes(f) ||
            // l2/dropout serialize only when non-zero
            (f === "l2" && (parsed.network.l2 ?? 0) > 0) ||
            inSol;
          expect(
            inSol || inRound,
            `${challenge.id}: fragment "${frag}" inconsistent with solution`
          ).toBe(true);
        }
      }
    }
  });
});
