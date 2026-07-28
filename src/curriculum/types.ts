export type ChallengeStepKind = "predict" | "experiment" | "explain";

/** One distinct idea the learner must touch (synonyms/inflections all count as one hit). */
export interface ExplainConceptGroup {
  /** Stable id used in grader output (matchedConcepts / missingConcepts). */
  id: string;
  /** Short human label for Socratic feedback. */
  label: string;
  /** Phrases/words that count as evidence for this concept. */
  synonyms: string[];
}

export interface ChallengeStep {
  kind: ChallengeStepKind;
  prompt: string;
  /** Optional multiple-choice for predict steps. */
  choices?: string[];
  /** Index of correct choice when choices present. */
  correctIndex?: number;
  /**
   * Distinct concept groups for offline explanation grading.
   * Prefer this over raw keyword lists — the grader requires hits across groups.
   */
  explainConcepts?: ExplainConceptGroup[];
  /**
   * @deprecated Flat keyword list — kept optional so older UI store paths still typecheck.
   * New code should use explainConcepts + gradeExplanation from @/tutor/explain.
   */
  explainKeywords?: string[];
  /** Experiment: min accuracy to achieve, or required config substring. */
  experimentCheck?: {
    minAccuracy?: number;
    maxLoss?: number;
    dslIncludes?: string[];
    dataset?: string;
  };
}

export interface Challenge {
  id: string;
  title: string;
  description: string;
  steps: ChallengeStep[];
}

/**
 * A runnable example embedded in the lesson text.
 *
 * The point of teaching inside the app rather than in a PDF: an example the
 * reader can put on screen in one click, in the same lab they're about to
 * experiment in. Every one of these is covered by tests that actually run it,
 * so the prose can't drift from what the code does.
 */
export interface LessonExample {
  /** Button label — say what will happen, e.g. "Run AND on one neuron". */
  label: string;
  dsl: string;
  /** What the reader should look for once it runs. */
  expect: string;
}

export interface LessonSection {
  heading: string;
  /** Markdown: **bold**, *italic*, `code`, bullet lists, formula lines. */
  body: string;
  example?: LessonExample;
}

export interface Chapter {
  id: string;
  number: number;
  title: string;
  subtitle: string;
  goals: string[];
  /** One-paragraph summary for the sidebar. The lesson is the long form. */
  theory: string;
  /**
   * The chapter's actual teaching text, read in the lesson view. Optional so
   * chapters can be written one at a time; the sidebar falls back to `theory`.
   */
  lesson?: LessonSection[];
  starterDSL: string;
  challenges: Challenge[];
  /** Chapter unlock requires previous chapter challenges complete. */
  unlockAfter?: string;
}

export interface ChallengeProgress {
  challengeId: string;
  stepIndex: number;
  completed: boolean;
  predictAnswer?: number;
  explainText?: string;
  experimentPassed?: boolean;
}

export interface CurriculumProgress {
  currentChapterId: string;
  completedChapters: string[];
  challenges: Record<string, ChallengeProgress>;
}
