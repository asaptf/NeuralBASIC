/**
 * @vitest-environment jsdom
 *
 * Lesson text must not silently drift from the engine. Every chapter that
 * carries a `lesson` is checked for structure, every embedded example is
 * actually trained, numeric claims in `expect` strings are verified against
 * measured multi-run distributions, and `runExample` wiring is exercised so a
 * click loads that example rather than stale state.
 *
 * If a claim fails, fix the prose (or the engine) — not these tolerances by
 * widening them until the lie passes.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { parseDSL } from "@/engine/dsl";
import { createAndTrain } from "@/engine/train";
import { getDataset } from "@/engine/datasets";
import type { TrainConfig } from "@/engine/types";
import { CHAPTERS } from "./chapters";
import type { LessonExample, LessonSection } from "./types";
import type { useAppStore as UseAppStoreType } from "@/store/useAppStore";

// ---------------------------------------------------------------------------
// Constants — keep the suite fast; a handful of repeats, not dozens.
// ---------------------------------------------------------------------------

/** Repeats per example for distribution checks. */
const REPEATS = 5;
/** Soft cap so a future mega-epoch example cannot dominate `npm test`. */
const EPOCHS_CAP = 400;

const PLACEHOLDER_RE = /\b(TODO|TBD|lorem)\b/i;

const WORD_NUMBERS: Record<string, number> = {
  two: 2,
  three: 3,
  four: 4,
  five: 5,
  six: 6,
  seven: 7,
  eight: 8,
  nine: 9,
  ten: 10,
  twice: 2,
};

// ---------------------------------------------------------------------------
// Stats helpers
// ---------------------------------------------------------------------------

interface RunOutcome {
  accuracy: number;
  loss: number;
  finite: boolean;
}

interface Dist {
  accuracies: number[];
  losses: number[];
  meanAcc: number;
  meanLoss: number;
  minAcc: number;
  maxAcc: number;
  minLoss: number;
  maxLoss: number;
  stdAcc: number;
  stdLoss: number;
  allFinite: boolean;
}

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
}

function std(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
}

function toDist(runs: RunOutcome[]): Dist {
  const accuracies = runs.map((r) => r.accuracy);
  const losses = runs.map((r) => r.loss);
  return {
    accuracies,
    losses,
    meanAcc: mean(accuracies),
    meanLoss: mean(losses),
    minAcc: Math.min(...accuracies),
    maxAcc: Math.max(...accuracies),
    minLoss: Math.min(...losses),
    maxLoss: Math.max(...losses),
    stdAcc: std(accuracies),
    stdLoss: std(losses),
    allFinite: runs.every((r) => r.finite),
  };
}

/**
 * Half-width around a measured mean that still counts as "near" the claim.
 * Derived from the observed spread so random init does not flake, with a
 * floor so a zero-variance lucky batch cannot force an absurdly tight band.
 */
function nearBand(values: number[], floor: number): number {
  const halfRange = (Math.max(...values) - Math.min(...values)) / 2;
  return Math.max(floor, 2 * std(values), halfRange + floor * 0.25);
}

function trainExample(dsl: string, overrides?: Partial<TrainConfig>): RunOutcome {
  const parsed = parseDSL(dsl);
  const epochs = Math.min(parsed.train.epochs, EPOCHS_CAP);
  const { history } = createAndTrain(
    parsed.network,
    { ...parsed.train, ...overrides, epochs },
    { includeDecisionBoundary: false }
  );
  const { accuracy, loss } = history.final;
  const finite =
    Number.isFinite(accuracy) &&
    Number.isFinite(loss) &&
    !Number.isNaN(accuracy) &&
    !Number.isNaN(loss);
  return { accuracy, loss, finite };
}

function sampleDistribution(dsl: string, n = REPEATS): Dist {
  const runs: RunOutcome[] = [];
  for (let i = 0; i < n; i++) runs.push(trainExample(dsl));
  return toDist(runs);
}

// ---------------------------------------------------------------------------
// Claim parsing — numbers (and a few English number-words) out of `expect`.
// ---------------------------------------------------------------------------

type Claim =
  | { kind: "acc_perfect" }
  | { kind: "acc_near"; value: number }
  | { kind: "acc_range"; lo: number; hi: number }
  | { kind: "loss_around"; value: number }
  | { kind: "baseline_loss_around"; value: number }
  | { kind: "loss_ratio"; factor: number }
  | { kind: "baseline_lr"; value: number }
  | { kind: "all_points_correct"; count: number }
  | { kind: "never_solves" };

function parseNumberToken(raw: string): number | null {
  const lower = raw.toLowerCase();
  if (lower in WORD_NUMBERS) return WORD_NUMBERS[lower]!;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

/**
 * Pull checkable numeric teaching claims out of an example's `expect` string.
 * Designed so editing "82%" to "95%" fails against measured moons accuracy.
 */
function parseExpectClaims(expectText: string): Claim[] {
  const claims: Claim[] = [];
  const text = expectText;

  // "25% to 75%" / "25%–75%" / "from 25% to 75%"
  const range = text.match(
    /(\d+(?:\.\d+)?)\s*%\s*(?:to|–|-|—)\s*(\d+(?:\.\d+)?)\s*%/i
  );
  if (range) {
    claims.push({
      kind: "acc_range",
      lo: Number(range[1]) / 100,
      hi: Number(range[2]) / 100,
    });
  }

  // "near 82%" / "around 82%" / "about 82%" / "lands near 82%"
  for (const m of text.matchAll(
    /(?:near|around|about|roughly|~)\s*(\d+(?:\.\d+)?)\s*%/gi
  )) {
    const v = Number(m[1]) / 100;
    // Skip if this percent is one end of an already-captured range.
    if (range && (m[1] === range[1] || m[1] === range[2])) continue;
    claims.push({ kind: "acc_near", value: v });
  }

  // Bare "reaches 100%" / "100% accuracy" when not part of a range claim
  if (!range && /\b100\s*%/.test(text)) {
    claims.push({ kind: "acc_perfect" });
  }

  // "loss sits around 2.2" / "loss ... near 0.3" — loss figure after "loss"
  for (const m of text.matchAll(
    /\bloss\b[^.]{0,40}?(?:around|near|about|sits around|sits near)\s*(\d+(?:\.\d+)?)/gi
  )) {
    claims.push({ kind: "loss_around", value: Number(m[1]) });
  }

  // "instead of 0.3" → baseline loss the reader is told to compare against
  const instead = text.match(/instead of\s+(\d+(?:\.\d+)?)/i);
  if (instead) {
    claims.push({ kind: "baseline_loss_around", value: Number(instead[1]) });
  }

  // "roughly seven times higher" / "3 times" / "twice"
  const times = text.match(
    /(?:roughly|about|around|nearly)?\s*([a-z]+|\d+(?:\.\d+)?)\s+times(?:\s+(?:higher|larger|as high))?/i
  );
  if (times) {
    const factor = parseNumberToken(times[1]!);
    if (factor != null && factor > 1) {
      claims.push({ kind: "loss_ratio", factor });
    }
  }

  // "Edit lr back to 0.8" / "lr=0.8" mentioned as the comparison setting
  const lrBack = text.match(
    /lr\s*(?:back\s*to|to|=)\s*(\d+(?:\.\d+)?)/i
  );
  if (lrBack) {
    claims.push({ kind: "baseline_lr", value: Number(lrBack[1]) });
  }

  // "all 4 points are on the correct side"
  const pts = text.match(
    /all\s+(\d+)\s+points?\s+(?:are\s+)?(?:on the correct side|correct)/i
  );
  if (pts) {
    claims.push({ kind: "all_points_correct", count: Number(pts[1]) });
  }

  // "never solves" / "keeps ringing" / "no matter how long you train"
  if (
    /never solves|keeps ringing|no matter how long you train|the wall doesn't/i.test(
      text
    )
  ) {
    claims.push({ kind: "never_solves" });
  }

  return claims;
}

function withLearningRate(dsl: string, lr: number): string {
  if (/\blr\s*=\s*[\d.]+/i.test(dsl)) {
    return dsl.replace(/\blr\s*=\s*[\d.]+/i, `lr=${lr}`);
  }
  return dsl.replace(
    /(train\b[^\n]*)/i,
    (_, trainLine: string) => `${trainLine} lr=${lr}`
  );
}

function assertClaims(
  label: string,
  example: LessonExample,
  dist: Dist,
  claims: Claim[]
): void {
  const baselineLr = claims.find((c) => c.kind === "baseline_lr") as
    | { kind: "baseline_lr"; value: number }
    | undefined;
  const needsBaseline = claims.some(
    (c) =>
      c.kind === "baseline_loss_around" ||
      c.kind === "loss_ratio" ||
      (c.kind === "acc_near" && baselineLr)
  );

  let baseline: Dist | null = null;
  if (needsBaseline && baselineLr) {
    baseline = sampleDistribution(
      withLearningRate(example.dsl, baselineLr.value),
      REPEATS
    );
    expect(
      baseline.allFinite,
      `${label}: baseline lr=${baselineLr.value} produced non-finite metrics`
    ).toBe(true);
  }

  for (const claim of claims) {
    switch (claim.kind) {
      case "acc_perfect": {
        // Every run must hit (near) perfect — teaching "reaches 100%".
        expect(
          dist.minAcc,
          `${label}: claimed 100% accuracy but min over ${REPEATS} runs was ${(dist.minAcc * 100).toFixed(1)}% (mean ${(dist.meanAcc * 100).toFixed(1)}%)`
        ).toBeGreaterThanOrEqual(0.99);
        break;
      }
      case "acc_near": {
        const band = nearBand(dist.accuracies, 0.08);
        expect(
          Math.abs(dist.meanAcc - claim.value),
          `${label}: claimed accuracy near ${(claim.value * 100).toFixed(0)}% but measured mean ${(dist.meanAcc * 100).toFixed(1)}% over ${REPEATS} runs [${(dist.minAcc * 100).toFixed(1)}–${(dist.maxAcc * 100).toFixed(1)}%] (band ±${(band * 100).toFixed(1)}pp)`
        ).toBeLessThanOrEqual(band);

        // Comparison phrasing "much like lr=X": accuracies stay close.
        if (baseline) {
          const accGap = Math.abs(dist.meanAcc - baseline.meanAcc);
          expect(
            accGap,
            `${label}: claimed accuracy stays close to lr=${baselineLr!.value}, but |Δmean acc|=${(accGap * 100).toFixed(1)}pp (high-lr ${(dist.meanAcc * 100).toFixed(1)}% vs baseline ${(baseline.meanAcc * 100).toFixed(1)}%)`
          ).toBeLessThanOrEqual(0.15);
        }
        break;
      }
      case "acc_range": {
        for (const a of dist.accuracies) {
          expect(
            a,
            `${label}: claimed accuracy stays in ${(claim.lo * 100).toFixed(0)}–${(claim.hi * 100).toFixed(0)}% but saw ${(a * 100).toFixed(1)}%`
          ).toBeGreaterThanOrEqual(claim.lo - 1e-9);
          expect(a).toBeLessThanOrEqual(claim.hi + 1e-9);
        }
        break;
      }
      case "loss_around": {
        const band = nearBand(dist.losses, Math.max(0.25, claim.value * 0.35));
        expect(
          Math.abs(dist.meanLoss - claim.value),
          `${label}: claimed loss around ${claim.value} but measured mean ${dist.meanLoss.toFixed(3)} over ${REPEATS} runs [${dist.minLoss.toFixed(3)}–${dist.maxLoss.toFixed(3)}] (band ±${band.toFixed(3)})`
        ).toBeLessThanOrEqual(band);
        break;
      }
      case "baseline_loss_around": {
        expect(baseline, `${label}: baseline_loss claim needs a baseline_lr in expect`).not.toBeNull();
        const band = nearBand(
          baseline!.losses,
          Math.max(0.1, claim.value * 0.4)
        );
        expect(
          Math.abs(baseline!.meanLoss - claim.value),
          `${label}: claimed baseline loss around ${claim.value} (lr=${baselineLr!.value}) but measured mean ${baseline!.meanLoss.toFixed(3)} [${baseline!.minLoss.toFixed(3)}–${baseline!.maxLoss.toFixed(3)}]`
        ).toBeLessThanOrEqual(band);
        break;
      }
      case "loss_ratio": {
        expect(baseline, `${label}: loss_ratio claim needs a baseline_lr in expect`).not.toBeNull();
        const ratio = dist.meanLoss / Math.max(baseline!.meanLoss, 1e-9);
        // Band from relative spreads of both distributions.
        const rel =
          nearBand(dist.losses, 0.01) / Math.max(dist.meanLoss, 1e-9) +
          nearBand(baseline!.losses, 0.01) /
            Math.max(baseline!.meanLoss, 1e-9);
        const band = Math.max(1.5, claim.factor * Math.max(0.35, rel));
        expect(
          Math.abs(ratio - claim.factor),
          `${label}: claimed loss ~${claim.factor}× baseline but measured ratio ${ratio.toFixed(2)} (high-lr mean loss ${dist.meanLoss.toFixed(3)} / baseline ${baseline!.meanLoss.toFixed(3)})`
        ).toBeLessThanOrEqual(band);
        // "Several times" must at least be a clear multiple, not a coin flip.
        expect(
          ratio,
          `${label}: claimed ${claim.factor}× higher loss but ratio was only ${ratio.toFixed(2)}`
        ).toBeGreaterThanOrEqual(Math.min(2.5, claim.factor * 0.4));
        break;
      }
      case "baseline_lr":
        // Handled when sampling baseline; nothing to assert alone.
        break;
      case "all_points_correct": {
        const parsed = parseDSL(example.dsl);
        const ds = getDataset(parsed.train.dataset);
        expect(
          ds.samples.length,
          `${label}: expect says all ${claim.count} points correct but dataset "${parsed.train.dataset}" has ${ds.samples.length} samples`
        ).toBe(claim.count);
        // Perfect accuracy ⇔ every point on the correct side of 0.5.
        expect(
          dist.minAcc,
          `${label}: claimed all ${claim.count} points correct but min accuracy ${(dist.minAcc * 100).toFixed(1)}%`
        ).toBeGreaterThanOrEqual(0.99);
        break;
      }
      case "never_solves": {
        expect(
          dist.maxAcc,
          `${label}: claimed the problem is never solved but a run reached ${(dist.maxAcc * 100).toFixed(1)}%`
        ).toBeLessThan(0.99);
        break;
      }
      default: {
        const _exhaustive: never = claim;
        throw new Error(`unhandled claim ${JSON.stringify(_exhaustive)}`);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Lesson inventory
// ---------------------------------------------------------------------------

function lessonChapters() {
  return CHAPTERS.filter((c) => c.lesson && c.lesson.length > 0);
}

function exampleSections(
  chapterId: string
): Array<{ chapterId: string; section: LessonSection; example: LessonExample }> {
  const out: Array<{
    chapterId: string;
    section: LessonSection;
    example: LessonExample;
  }> = [];
  for (const ch of lessonChapters()) {
    if (chapterId !== "*" && ch.id !== chapterId) continue;
    for (const section of ch.lesson!) {
      if (section.example) {
        out.push({ chapterId: ch.id, section, example: section.example });
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// 1. Structural sanity
// ---------------------------------------------------------------------------

describe("lesson structure", () => {
  it("at least one chapter currently ships a lesson", () => {
    expect(lessonChapters().length).toBeGreaterThan(0);
  });

  for (const chapter of lessonChapters()) {
    describe(`${chapter.id} (${chapter.title})`, () => {
      it("every section has non-empty heading and body", () => {
        for (const [i, section] of chapter.lesson!.entries()) {
          expect(
            section.heading?.trim(),
            `${chapter.id} section[${i}] heading`
          ).toBeTruthy();
          expect(
            section.body?.trim(),
            `${chapter.id} section[${i}] ("${section.heading}") body`
          ).toBeTruthy();
        }
      });

      it("headings are unique within the chapter", () => {
        const headings = chapter.lesson!.map((s) => s.heading.trim());
        expect(new Set(headings).size, headings.join(" | ")).toBe(
          headings.length
        );
      });

      it("every example has a non-empty expect string", () => {
        for (const section of chapter.lesson!) {
          if (!section.example) continue;
          expect(
            section.example.expect?.trim(),
            `${chapter.id} / "${section.heading}" example.expect`
          ).toBeTruthy();
          expect(
            section.example.label?.trim(),
            `${chapter.id} / "${section.heading}" example.label`
          ).toBeTruthy();
          expect(
            section.example.dsl?.trim(),
            `${chapter.id} / "${section.heading}" example.dsl`
          ).toBeTruthy();
        }
      });

      it("no section body still holds an obvious placeholder", () => {
        for (const section of chapter.lesson!) {
          expect(
            PLACEHOLDER_RE.test(section.body),
            `${chapter.id} / "${section.heading}" body still contains TODO/TBD/lorem`
          ).toBe(false);
          if (section.example) {
            expect(
              PLACEHOLDER_RE.test(section.example.expect),
              `${chapter.id} / "${section.heading}" expect still contains TODO/TBD/lorem`
            ).toBe(false);
          }
        }
      });
    });
  }
});

// ---------------------------------------------------------------------------
// 2–3. Every example runs, is finite, and is teachably reproducible
// ---------------------------------------------------------------------------

describe("lesson examples: runnable + finite + reproducible", () => {
  const examples = exampleSections("*");

  it("there is at least one runnable lesson example", () => {
    expect(examples.length).toBeGreaterThan(0);
  });

  for (const { chapterId, section, example } of examples) {
    const label = `${chapterId} / "${section.heading}" (${example.label})`;

    it(`${label}: parseDSL builds a network and trains with finite metrics`, () => {
      const parsed = parseDSL(example.dsl);
      expect(parsed.network.layers.length).toBeGreaterThan(0);
      expect(parsed.train.epochs).toBeGreaterThan(0);

      const once = trainExample(example.dsl);
      expect(once.finite, `${label}: non-finite loss/accuracy`).toBe(true);
      expect(once.accuracy).toBeGreaterThanOrEqual(0);
      expect(once.accuracy).toBeLessThanOrEqual(1);
      expect(once.loss).toBeGreaterThanOrEqual(0);
    }, 60_000);

    // Attention layers train by finite differences here, so a 5-run sample of an
    // attention example needs well past vitest's 5s default.
    it(`${label}: distribution over ${REPEATS} runs stays finite and teachable`, () => {
      const dist = sampleDistribution(example.dsl, REPEATS);
      expect(
        dist.allFinite,
        `${label}: at least one of ${REPEATS} runs produced NaN/non-finite metrics\nacc=${dist.accuracies}\nloss=${dist.losses}`
      ).toBe(true);

      // A single lucky/unlucky init must not be the whole story the prose sells.
      // If accuracy swings across nearly the full [0,1] band with no range claim,
      // the expect text is teaching a coin flip.
      const claims = parseExpectClaims(example.expect);
      const hasRange = claims.some((c) => c.kind === "acc_range");
      const hasPerfect = claims.some(
        (c) => c.kind === "acc_perfect" || c.kind === "all_points_correct"
      );
      const span = dist.maxAcc - dist.minAcc;
      if (hasPerfect) {
        expect(
          span,
          `${label}: claimed reliable success but accuracy span across runs is ${(span * 100).toFixed(1)}pp`
        ).toBeLessThanOrEqual(0.05);
      } else if (!hasRange) {
        expect(
          span,
          `${label}: accuracy span ${(span * 100).toFixed(1)}pp across ${REPEATS} runs is too wide to teach a point estimate from`
        ).toBeLessThanOrEqual(0.35);
      }
    }, 120_000);
  }
});

// ---------------------------------------------------------------------------
// 4. Numeric claims in `expect` vs measured reality
// ---------------------------------------------------------------------------

describe("lesson examples: expect-string numeric claims", () => {
  for (const { chapterId, section, example } of exampleSections("*")) {
    const label = `${chapterId} / "${section.heading}"`;
    const claims = parseExpectClaims(example.expect);

    it(`${label}: ${claims.length} claim(s) hold under random init`, () => {
      expect(
        claims.length,
        `${label}: expect string has no parseable numeric claims — add phrasing the suite understands, or the prose is untestable:\n${example.expect}`
      ).toBeGreaterThan(0);

      const dist = sampleDistribution(example.dsl, REPEATS);
      expect(dist.allFinite).toBe(true);
      assertClaims(label, example, dist, claims);
      // Attention examples train by finite differences and need more than the 5s default.
    }, 120_000);
  }
});

// ---------------------------------------------------------------------------
// 5. runExample wiring (store)
// ---------------------------------------------------------------------------

type Store = typeof UseAppStoreType;

function installEnv(opts: { reducedMotion?: boolean } = {}) {
  const { reducedMotion = false } = opts;
  Object.defineProperty(document, "hidden", {
    configurable: true,
    get: () => false,
  });
  window.matchMedia = vi.fn((query: string) => ({
    matches:
      reducedMotion && query.includes("prefers-reduced-motion: reduce"),
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(() => false),
  }));
}

async function loadStore(env: { reducedMotion?: boolean } = {}): Promise<Store> {
  localStorage.clear();
  vi.resetModules();
  installEnv(env);
  const mod = await import("@/store/useAppStore");
  return mod.useAppStore;
}

async function flushTraining(store: Store, maxSteps = 500) {
  for (let i = 0; i < maxSteps; i++) {
    const s = store.getState();
    if (!s.isTraining) return;
    if (s.isPaused) return;
    await vi.advanceTimersByTimeAsync(50);
  }
  throw new Error(
    `training still running after ${maxSteps} timer steps ` +
      `(epoch ${store.getState().epochsRun}/${store.getState().totalEpochs})`
  );
}

const STALE_XOR_DSL = `network StaleXor {
  dense 2 -> 1 activation=sigmoid
}
train dataset=xor lr=0.8 epochs=8
`;

const AND_EXAMPLE_DSL = `network Perceptron {
  dense 2 -> 1 activation=sigmoid
}
train dataset=and lr=0.8 epochs=200`;

describe("runExample wiring", () => {
  let store: Store;

  beforeEach(() => {
    vi.useFakeTimers({
      toFake: [
        "setTimeout",
        "clearTimeout",
        "setInterval",
        "clearInterval",
        "requestAnimationFrame",
        "cancelAnimationFrame",
        "performance",
        "Date",
      ],
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    localStorage.clear();
  });

  it("loads the DSL, tags lastTrigger, clears stale metrics, and trains the example", async () => {
    store = await loadStore({ reducedMotion: true });

    // Leave XOR results on screen first — the bug class is "previous state sticks".
    store.getState().setDsl(STALE_XOR_DSL);
    store.getState().trainNow("manual");
    const stale = store.getState();
    expect(stale.isTraining).toBe(false);
    expect(stale.history.losses.length).toBe(8);
    expect(stale.lastSnapshot).not.toBeNull();
    expect(stale.lastTrigger).toBe("manual");
    expect(stale.dsl).toContain("dataset=xor");
    const staleLosses = [...stale.history.losses];

    store.getState().runExample(AND_EXAMPLE_DSL);

    const s = store.getState();
    expect(s.dsl).toBe(AND_EXAMPLE_DSL);
    expect(s.lastTrigger).toBe("lesson-example");
    expect(s.parseError).toBeNull();
    expect(s.actionError).toBeNull();
    expect(s.isTraining).toBe(false);
    expect(s.isPaused).toBe(false);
    // Reduced-motion path finishes inside trainNow — results describe AND.
    expect(s.history.losses.length).toBeGreaterThan(0);
    expect(s.history.losses).not.toEqual(staleLosses);
    expect(s.epochsRun).toBe(s.totalEpochs);
    expect(s.totalEpochs).toBe(200);
    expect(s.trainConfig.dataset).toBe("and");
    expect(s.lastSnapshot).not.toBeNull();
    expect(Number.isFinite(s.lastSnapshot!.loss)).toBe(true);
    expect(s.lastSnapshot!.accuracy).toBeGreaterThanOrEqual(0.99);
    expect(s.weights.length).toBeGreaterThan(0);
  });

  it("mid-run handoff: abandons the previous session and trains the new DSL", async () => {
    store = await loadStore({ reducedMotion: false });

    store.getState().setDsl(
      `network LongXor {
  dense 2 -> 1 activation=sigmoid
}
train dataset=xor lr=0.5 epochs=40
`
    );
    store.getState().trainNow("manual");
    expect(store.getState().isTraining).toBe(true);
    await vi.advanceTimersByTimeAsync(16);
    const mid = store.getState();
    expect(mid.isTraining).toBe(true);
    expect(mid.epochsRun).toBeGreaterThan(0);
    expect(mid.epochsRun).toBeLessThan(40);
    expect(mid.dsl).toContain("dataset=xor");

    const shortAnd = `network QuickAnd {
  dense 2 -> 1 activation=sigmoid
}
train dataset=and lr=0.8 epochs=12
`;
    store.getState().runExample(shortAnd);

    // Immediately after the call the DSL is the example; training has restarted.
    expect(store.getState().dsl).toBe(shortAnd);
    expect(store.getState().lastTrigger).toBe("lesson-example");
    // Clear happened before trainNow re-armed the loop.
    expect(store.getState().trainConfig.dataset).toBe("and");

    await flushTraining(store);

    const done = store.getState();
    expect(done.isTraining).toBe(false);
    expect(done.dsl).toBe(shortAnd);
    expect(done.epochsRun).toBe(12);
    expect(done.history.losses).toHaveLength(12);
    expect(done.trainConfig.dataset).toBe("and");
    expect(done.lastSnapshot).not.toBeNull();
    expect(Number.isFinite(done.lastSnapshot!.accuracy)).toBe(true);
  });
});
