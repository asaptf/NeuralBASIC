/**
 * Offline explanation + predict graders for curriculum challenges.
 * Deterministic, no network. Socratic feedback only — never the full answer.
 */

import type { ExplainConceptGroup } from "@/curriculum/types";

export interface ExplainGradeResult {
  passed: boolean;
  score: number;
  matchedConcepts: string[];
  missingConcepts: string[];
  feedback: string;
}

export interface PredictGradeResult {
  correct: boolean;
  nudge: string;
}

const MIN_CHARS = 24;
const MIN_CONTENT_TOKENS = 5;

/** Light inflection folding so "separability" hits "separable", etc. */
function stemToken(raw: string): string {
  let w = raw.toLowerCase().replace(/[^a-z0-9]/g, "");
  if (w.length <= 3) return w;
  const suffixes = [
    "ational",
    "tional",
    "ization",
    "iveness",
    "fulness",
    "ousness",
    "ations",
    "ations",
    "ities",
    "ively",
    "ation",
    "ments",
    "ment",
    "ness",
    "able",
    "ible",
    "ence",
    "ance",
    "ions",
    "ing",
    "ied",
    "ies",
    "ily",
    "ity",
    "ely",
    "ly",
    "ed",
    "es",
    "s",
  ];
  for (const suf of suffixes) {
    if (w.length > suf.length + 2 && w.endsWith(suf)) {
      w = w.slice(0, -suf.length);
      break;
    }
  }
  return w;
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s'-]/g, " ")
    .split(/\s+/)
    .map((t) => t.replace(/^['-]+|['-]+$/g, ""))
    .filter((t) => t.length > 0);
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** True if a text token is a reasonable inflection/stem match for a synonym word. */
function tokenMatchesSynonym(token: string, synonymWord: string): boolean {
  const raw = synonymWord.toLowerCase();
  const t = token.toLowerCase();
  if (t === raw) return true;

  // Short synonyms (and, or, l2, …) require exact token equality only
  if (raw.length <= 3) return false;

  const stem = stemToken(raw);
  const st = stemToken(t);
  if (st === stem) return true;

  // Inflection: "linearly" vs "linear", "separability" vs "separable"
  // Require a meaningful shared prefix (at least 4 chars) to avoid "a"→"and".
  const minShared = Math.min(4, stem.length, st.length);
  if (stem.length >= 4 && st.length >= 4) {
    if (st.startsWith(stem) || stem.startsWith(st)) return true;
    if (t.startsWith(stem) || raw.startsWith(st)) return true;
  } else if (stem.length >= minShared && st.length >= minShared) {
    if (st === stem) return true;
  }
  return false;
}

/** True if `phrase` (possibly multi-word) appears in text, allowing light inflections. */
function phraseMatches(text: string, phrase: string): boolean {
  const p = phrase.toLowerCase().trim();
  if (!p) return false;

  const words = p.split(/\s+/).filter(Boolean);
  if (words.length === 1) {
    const w = words[0]!;
    // Whole-word / whole-phrase substring with boundaries for short words
    if (w.length <= 3) {
      return new RegExp(`\\b${escapeRegExp(w)}\\b`, "i").test(text);
    }
    if (text.includes(p)) return true;
    const tokens = tokenize(text);
    return tokens.some((t) => tokenMatchesSynonym(t, w));
  }

  // Multi-word exact phrase first
  if (text.includes(p)) return true;

  // Multi-word: each word must appear in order (soft stem match)
  const tokens = tokenize(text);
  const needed = words;
  let ti = 0;
  for (const n of needed) {
    let found = false;
    while (ti < tokens.length) {
      const t = tokens[ti++]!;
      if (tokenMatchesSynonym(t, n) || (n.length > 3 && t.includes(n))) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

function conceptMatches(text: string, group: ExplainConceptGroup): boolean {
  return group.synonyms.some((syn) => phraseMatches(text, syn));
}

/**
 * Detect keyword stuffing / empty padding:
 * - same token repeated to hit length floor
 * - very few unique content stems relative to total tokens
 * - answer is almost only stopwords
 */
function isLowQualityPadding(text: string, tokens: string[]): boolean {
  if (tokens.length === 0) return true;

  const stems = tokens.map(stemToken).filter((s) => s.length > 1);
  const unique = new Set(stems);
  const stop = new Set([
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "to",
    "of",
    "and",
    "or",
    "but",
    "in",
    "on",
    "for",
    "with",
    "as",
    "at",
    "by",
    "it",
    "its",
    "this",
    "that",
    "i",
    "we",
    "you",
    "they",
    "my",
    "our",
    "can",
    "could",
    "would",
    "should",
    "just",
    "very",
    "really",
    "so",
    "too",
    "also",
    "than",
    "then",
    "from",
    "into",
    "about",
    "because",
    "when",
    "what",
    "which",
    "who",
    "how",
    "not",
    "no",
    "yes",
    "do",
    "does",
    "did",
    "if",
    "else",
  ]);
  const content = stems.filter((s) => !stop.has(s) && s.length > 2);
  const uniqueContent = new Set(content);

  // Dominant single stem (e.g. "linear linear linear …")
  const counts = new Map<string, number>();
  for (const s of content) counts.set(s, (counts.get(s) ?? 0) + 1);
  let maxCount = 0;
  for (const c of counts.values()) if (c > maxCount) maxCount = c;
  if (content.length >= 4 && maxCount / content.length >= 0.6) return true;

  // Few unique ideas for a long-ish answer
  if (tokens.length >= 8 && uniqueContent.size <= 2) return true;
  if (tokens.length >= 12 && unique.size <= 3) return true;

  // Almost no content after stopword strip
  if (tokens.length >= MIN_CONTENT_TOKENS && uniqueContent.size === 0) return true;

  return false;
}

function socraticFeedback(
  matched: string[],
  missing: string[],
  reason: "short" | "padding" | "partial" | "pass" | "empty_concepts"
): string {
  if (reason === "pass") {
    return "Solid — you connected more than one idea. What did you *observe* that supports that?";
  }
  if (reason === "short") {
    return "Can you say a bit more — what did you expect, and what did the model actually do?";
  }
  if (reason === "padding") {
    return "Repeating one word won't unlock the next step. Which two different ideas from the challenge matter here?";
  }
  if (reason === "empty_concepts") {
    return "There's no concept checklist for this step — still: what experiment would test your claim?";
  }
  if (matched.length === 0) {
    return "What is the core idea you're trying to express? Name one mechanism and one consequence you saw.";
  }
  if (matched.length === 1 && missing.length > 0) {
    return `You touched "${matched[0]}". What *else* has to be true for that to explain the result?`;
  }
  const hint = missing[0];
  return hint
    ? `Good start on ${matched.join(" / ")}. What about the role of ${hint}?`
    : "You're close — can you link two distinct ideas from the lab into one sentence?";
}

/**
 * Grade a free-text explanation against distinct concept groups (with synonyms).
 * Rejects keyword stuffing, single-concept answers on multi-concept challenges,
 * and empty padding. Accepts natural phrasing and light inflections.
 */
export function gradeExplanation(
  text: string,
  concepts: ExplainConceptGroup[],
  options?: { minGroups?: number; minChars?: number }
): ExplainGradeResult {
  const trimmed = text.trim();
  const minChars = options?.minChars ?? MIN_CHARS;
  const groups = concepts ?? [];

  if (groups.length === 0) {
    const tokens = tokenize(trimmed);
    const passed =
      trimmed.length >= minChars &&
      tokens.length >= MIN_CONTENT_TOKENS &&
      !isLowQualityPadding(trimmed, tokens);
    return {
      passed,
      score: passed ? 1 : 0,
      matchedConcepts: [],
      missingConcepts: [],
      feedback: passed
        ? socraticFeedback([], [], "pass")
        : socraticFeedback([], [], trimmed.length < minChars ? "short" : "padding"),
    };
  }

  const lower = trimmed.toLowerCase();
  const tokens = tokenize(trimmed);
  const matched: string[] = [];
  const missing: string[] = [];

  for (const g of groups) {
    if (conceptMatches(lower, g)) matched.push(g.id);
    else missing.push(g.id);
  }

  const score = matched.length / groups.length;
  // Multi-concept: require at least 2 distinct groups (or all if only one exists)
  const defaultMin =
    groups.length <= 1 ? 1 : Math.min(groups.length, Math.max(2, Math.ceil(groups.length * 0.4)));
  const minGroups = options?.minGroups ?? defaultMin;

  if (trimmed.length < minChars) {
    return {
      passed: false,
      score,
      matchedConcepts: matched,
      missingConcepts: missing,
      feedback: socraticFeedback(matched, missing, "short"),
    };
  }

  if (isLowQualityPadding(trimmed, tokens)) {
    return {
      passed: false,
      score: Math.min(score, 1 / groups.length),
      matchedConcepts: matched.slice(0, 1),
      missingConcepts: groups.map((g) => g.id).filter((id) => !matched.slice(0, 1).includes(id)),
      feedback: socraticFeedback(matched.slice(0, 1), missing, "padding"),
    };
  }

  const passed = matched.length >= minGroups && score >= minGroups / groups.length;

  return {
    passed,
    score,
    matchedConcepts: matched,
    missingConcepts: missing,
    feedback: socraticFeedback(
      matched,
      missing,
      passed ? "pass" : "partial"
    ),
  };
}

/**
 * Grade a predict multiple-choice without leaking the correct index.
 * On wrong answers, return a Socratic nudge only.
 */
export function gradePredictAnswer(
  choiceIndex: number,
  correctIndex: number,
  _choices?: string[]
): PredictGradeResult {
  void _choices;
  if (choiceIndex === correctIndex) {
    return {
      correct: true,
      nudge: "Nice — lock that prediction in, then run the experiment and see if the metrics agree.",
    };
  }
  return {
    correct: false,
    nudge:
      "Not quite. Before guessing again: what would you *observe* if that choice were true? Then pick the option that matches that observation.",
  };
}
