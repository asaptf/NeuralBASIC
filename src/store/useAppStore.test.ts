/**
 * @vitest-environment jsdom
 *
 * Store tests for the animated training loop and curriculum progression.
 * Isolation: each test re-imports the store module (vi.resetModules) so the
 * Zustand singleton and module-level loop state (session, forceTimer, rAF ids)
 * cannot leak between cases. localStorage is cleared every time.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { useAppStore as UseAppStoreType } from "./useAppStore";
import type { TrainStepResult } from "@/engine/types";

type Store = typeof UseAppStoreType;

const SMALL_AND_DSL = `network FastNet {
  dense 2 -> 1 activation=sigmoid
}
train dataset=and lr=0.8 epochs=6
`;

const SMALL_XOR_DSL = `network FastNet {
  dense 2 -> 1 activation=sigmoid
}
train dataset=xor lr=0.8 epochs=6
`;

const PLAUSIBLE_EXPLAIN: Record<string, string> = {
  "ch1-c1":
    "AND is linearly separable so a single straight boundary works, but XOR classes cannot be separated by one line.",
  "ch1-c2":
    "With a very large learning rate the updates jumped wildly, the loss oscillated, and training looked unstable.",
  "ch1-c3":
    "XOR is not linearly separable, so one neuron is stuck near chance and accuracy plateaus well below perfect.",
};

type EnvOpts = {
  reducedMotion?: boolean;
  hidden?: boolean;
  /** rAF never invokes its callback (forces watchdog / timer path). */
  rafNever?: boolean;
};

function installEnv(opts: EnvOpts = {}) {
  const { reducedMotion = false, hidden = false, rafNever = false } = opts;

  Object.defineProperty(document, "hidden", {
    configurable: true,
    get: () => hidden,
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

  if (rafNever) {
    vi.stubGlobal("requestAnimationFrame", () => 1);
    vi.stubGlobal("cancelAnimationFrame", () => {});
  }
}

async function loadStore(
  env: EnvOpts = {},
  opts: { clearStorage?: boolean } = {}
): Promise<Store> {
  const { clearStorage = true } = opts;
  if (clearStorage) localStorage.clear();
  vi.resetModules();
  installEnv(env);
  const mod = await import("./useAppStore");
  return mod.useAppStore;
}

/** Drive the animated loop until training ends (or max steps). */
async function flushTraining(store: Store, maxSteps = 400) {
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

function fakeSnapshot(
  partial: Partial<TrainStepResult> &
    Pick<TrainStepResult, "loss" | "accuracy">
): TrainStepResult {
  return {
    epoch: partial.epoch ?? 1,
    loss: partial.loss,
    accuracy: partial.accuracy,
    layerSnapshots: partial.layerSnapshots ?? [],
    decisionGrid: partial.decisionGrid,
  };
}

describe("useAppStore — training loop", () => {
  let store: Store;

  beforeEach(async () => {
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
    store = await loadStore();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    localStorage.clear();
  });

  it("1. full animated run reaches totalEpochs and populates results", async () => {
    store.getState().setDsl(SMALL_AND_DSL);
    store.getState().trainNow();

    expect(store.getState().isTraining).toBe(true);
    expect(store.getState().totalEpochs).toBe(6);

    await flushTraining(store);

    const s = store.getState();
    expect(s.isTraining).toBe(false);
    expect(s.isPaused).toBe(false);
    expect(s.epochsRun).toBe(s.totalEpochs);
    expect(s.totalEpochs).toBe(6);
    expect(s.history.losses).toHaveLength(6);
    expect(s.history.accuracies).toHaveLength(6);
    expect(s.weights.length).toBeGreaterThan(0);
    expect(s.lastSnapshot).not.toBeNull();
    expect(s.lastSnapshot!.loss).toBeGreaterThanOrEqual(0);
    expect(s.lastSnapshot!.accuracy).toBeGreaterThanOrEqual(0);
    expect(s.lastSnapshot!.accuracy).toBeLessThanOrEqual(1);
  });

  it("2. epochs arrive incrementally (not 0 then done in one step)", async () => {
    store.getState().setDsl(
      `network SlowVis {
  dense 2 -> 1 activation=sigmoid
}
train dataset=and lr=0.5 epochs=20
`
    );
    store.getState().trainNow();
    expect(store.getState().epochsRun).toBe(0);
    expect(store.getState().isTraining).toBe(true);

    // One frame / short advance — pacing should leave the run mid-way.
    await vi.advanceTimersByTimeAsync(16);

    const mid = store.getState();
    expect(mid.isTraining).toBe(true);
    expect(mid.epochsRun).toBeGreaterThan(0);
    expect(mid.epochsRun).toBeLessThan(mid.totalEpochs);
    expect(mid.history.losses.length).toBe(mid.epochsRun);
    expect(mid.history.losses.length).toBeLessThan(20);

    await flushTraining(store);
    expect(store.getState().isTraining).toBe(false);
    expect(store.getState().epochsRun).toBe(20);
  });

  it("3. hidden tab (document.hidden) still completes the run", async () => {
    store = await loadStore({ hidden: true });
    store.getState().setDsl(SMALL_AND_DSL);
    store.getState().trainNow();

    expect(store.getState().isTraining).toBe(true);
    await flushTraining(store);

    const s = store.getState();
    expect(s.isTraining).toBe(false);
    expect(s.epochsRun).toBe(6);
    expect(s.history.losses).toHaveLength(6);
    expect(s.lastSnapshot).not.toBeNull();
  });

  it("4. watchdog: dead rAF still completes via timer fallback after ~1s", async () => {
    store = await loadStore({ rafNever: true, hidden: false });
    store.getState().setDsl(SMALL_AND_DSL);
    store.getState().trainNow();

    expect(store.getState().isTraining).toBe(true);
    expect(store.getState().epochsRun).toBe(0);

    // Before the watchdog (~1s), a dead rAF must not have advanced epochs.
    await vi.advanceTimersByTimeAsync(500);
    expect(store.getState().epochsRun).toBe(0);
    expect(store.getState().isTraining).toBe(true);

    // Watchdog arms at 1000ms, then timer driver drains the run.
    await flushTraining(store);

    const s = store.getState();
    expect(s.isTraining).toBe(false);
    expect(s.epochsRun).toBe(6);
    expect(s.history.losses).toHaveLength(6);
  });

  it("5. prefers-reduced-motion completes synchronously inside trainNow()", async () => {
    store = await loadStore({ reducedMotion: true });
    store.getState().setDsl(SMALL_AND_DSL);
    store.getState().trainNow();

    // No timer flush — reduced-motion path finishes before trainNow returns.
    const s = store.getState();
    expect(s.isTraining).toBe(false);
    expect(s.epochsRun).toBe(6);
    expect(s.history.losses).toHaveLength(6);
    expect(s.weights.length).toBeGreaterThan(0);
    expect(s.lastSnapshot).not.toBeNull();
  });

  it("6. pause stops advancement; resume continues and finishes", async () => {
    store.getState().setDsl(
      `network PauseNet {
  dense 2 -> 1 activation=sigmoid
}
train dataset=and lr=0.5 epochs=30
`
    );
    store.getState().trainNow();
    await vi.advanceTimersByTimeAsync(16);

    const atPause = store.getState().epochsRun;
    expect(atPause).toBeGreaterThan(0);
    expect(atPause).toBeLessThan(30);

    store.getState().pauseTraining();
    expect(store.getState().isPaused).toBe(true);
    expect(store.getState().isTraining).toBe(true);

    const frozen = store.getState().epochsRun;
    // Plenty of fake time — a paused run must not silently keep training.
    await vi.advanceTimersByTimeAsync(5000);
    expect(store.getState().epochsRun).toBe(frozen);
    expect(store.getState().isPaused).toBe(true);
    expect(store.getState().isTraining).toBe(true);

    store.getState().resumeTraining();
    expect(store.getState().isPaused).toBe(false);
    expect(store.getState().isTraining).toBe(true);

    await flushTraining(store);
    const s = store.getState();
    expect(s.isTraining).toBe(false);
    expect(s.epochsRun).toBe(30);
    expect(s.history.losses).toHaveLength(30);
  });

  it("7. stepEpoch from idle advances exactly one epoch and stays paused", async () => {
    store = await loadStore({ reducedMotion: true });
    store.getState().setDsl(SMALL_AND_DSL);

    // From idle — must not train the whole model even under reduced-motion.
    store.getState().stepEpoch();
    let s = store.getState();
    expect(s.epochsRun).toBe(1);
    expect(s.history.losses).toHaveLength(1);
    expect(s.isTraining).toBe(true);
    expect(s.isPaused).toBe(true);
    expect(s.isTraining && s.isPaused).toBe(true);

    store.getState().stepEpoch();
    s = store.getState();
    expect(s.epochsRun).toBe(2);
    expect(s.history.losses).toHaveLength(2);
    expect(s.isPaused).toBe(true);
    expect(s.isTraining).toBe(true);

    // Further presses keep walking one at a time.
    store.getState().stepEpoch();
    expect(store.getState().epochsRun).toBe(3);
  });

  it("8. resetWeights mid-run stops the loop and clears results", async () => {
    store.getState().setDsl(
      `network ResetNet {
  dense 2 -> 1 activation=sigmoid
}
train dataset=and lr=0.5 epochs=40
`
    );
    store.getState().trainNow();
    await vi.advanceTimersByTimeAsync(16);
    expect(store.getState().isTraining).toBe(true);
    expect(store.getState().epochsRun).toBeGreaterThan(0);

    store.getState().resetWeights();

    const s = store.getState();
    expect(s.isTraining).toBe(false);
    expect(s.isPaused).toBe(false);
    expect(s.epochsRun).toBe(0);
    expect(s.totalEpochs).toBe(0);
    expect(s.weights).toEqual([]);
    expect(s.history).toEqual({ losses: [], accuracies: [] });
    expect(s.lastSnapshot).toBeNull();

    // Loop must stay dead after reset.
    await vi.advanceTimersByTimeAsync(3000);
    expect(store.getState().epochsRun).toBe(0);
    expect(store.getState().isTraining).toBe(false);
  });

  it("9. parse failure sets parseError and does not leave isTraining stuck true", async () => {
    // parseDSL now rejects invalid programs; use real garbage (no mock needed).
    store.getState().setDsl("this is not valid neuralbasic at all");
    store.getState().trainNow();

    const s = store.getState();
    expect(s.isTraining).toBe(false);
    expect(s.parseError).toBeTruthy();
    expect(s.parseError).toMatch(/line\s+\d+/i);
    expect(s.parseError).toMatch(/unknown top-level|this/i);
  });
});

describe("useAppStore — state coherence", () => {
  let store: Store;

  beforeEach(async () => {
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
    store = await loadStore({ reducedMotion: true });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    localStorage.clear();
  });

  it("10. setDataset to a different dataset clears run state; same dataset does not", () => {
    store.getState().setDsl(SMALL_AND_DSL);
    store.getState().trainNow();

    const finished = store.getState();
    expect(finished.lastSnapshot).not.toBeNull();
    expect(finished.history.losses.length).toBe(6);
    expect(finished.epochsRun).toBe(6);
    const snap = finished.lastSnapshot;
    const losses = [...finished.history.losses];

    // Same dataset — must not wipe a finished run.
    store.getState().setDataset("and");
    expect(store.getState().lastSnapshot).toBe(snap);
    expect(store.getState().history.losses).toEqual(losses);
    expect(store.getState().epochsRun).toBe(6);

    // Different dataset — previous numbers must not survive under the new name.
    store.getState().setDataset("xor");
    const cleared = store.getState();
    expect(cleared.trainConfig.dataset).toBe("xor");
    expect(cleared.lastSnapshot).toBeNull();
    expect(cleared.history).toEqual({ losses: [], accuracies: [] });
    expect(cleared.epochsRun).toBe(0);
    expect(cleared.totalEpochs).toBe(0);
    expect(cleared.weights).toEqual([]);
    expect(cleared.isTraining).toBe(false);
  });

  it("11. loadLocal after save restores experiment A and rebuilds its snapshot", () => {
    // Train A on AND.
    store.getState().setDsl(SMALL_AND_DSL);
    store.getState().trainNow();
    const a = store.getState();
    expect(a.trainConfig.dataset).toBe("and");
    expect(a.weights.length).toBeGreaterThan(0);
    const aWeightsJson = JSON.stringify(a.weights);
    const aLosses = [...a.history.losses];
    const aAcc = a.lastSnapshot!.accuracy;
    const aLoss = a.lastSnapshot!.loss;

    store.getState().saveLocal();

    // Train something different in between.
    store.getState().setDsl(SMALL_XOR_DSL);
    store.getState().trainNow();
    const mid = store.getState();
    expect(mid.trainConfig.dataset).toBe("xor");
    expect(JSON.stringify(mid.weights)).not.toBe(aWeightsJson);

    // Load A back — panels must describe the loaded model, not the XOR run.
    store.getState().loadLocal();
    const loaded = store.getState();
    expect(loaded.trainConfig.dataset).toBe("and");
    expect(loaded.dsl).toContain("dataset=and");
    expect(loaded.history.losses).toEqual(aLosses);
    expect(JSON.stringify(loaded.weights)).toBe(aWeightsJson);
    expect(loaded.lastSnapshot).not.toBeNull();
    // Snapshot is recomputed from weights; metrics should match the loaded model.
    expect(loaded.lastSnapshot!.accuracy).toBeCloseTo(aAcc, 5);
    expect(loaded.lastSnapshot!.loss).toBeCloseTo(aLoss, 5);
    expect(loaded.isTraining).toBe(false);
  });

  it("12. importExperimentFile with malformed JSON sets parseError and keeps state", () => {
    store.getState().setDsl(SMALL_AND_DSL);
    store.getState().trainNow();
    const before = store.getState();
    const snapshot = before.lastSnapshot;
    const losses = [...before.history.losses];
    const dsl = before.dsl;
    const weightsJson = JSON.stringify(before.weights);

    expect(() => {
      store.getState().importExperimentFile("{not valid json!!!");
    }).not.toThrow();

    const after = store.getState();
    expect(after.parseError).toMatch(/import failed/i);
    expect(after.dsl).toBe(dsl);
    expect(after.history.losses).toEqual(losses);
    expect(after.lastSnapshot).toBe(snapshot);
    expect(JSON.stringify(after.weights)).toBe(weightsJson);
    expect(after.isTraining).toBe(false);
  });
});

describe("useAppStore — curriculum progression", () => {
  let store: Store;

  beforeEach(async () => {
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
    store = await loadStore({ reducedMotion: true });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    localStorage.clear();
  });

  it("13. submitPredict records feedback and does not leak correctness into tutorMessages", () => {
    const beforeMsgs = store.getState().tutorMessages.map((m) => m.content);
    const beforeLen = beforeMsgs.length;

    const ok = store.getState().submitPredict("ch1-c1", 0);
    expect(ok).toBe(true);

    const fb = store.getState().challengeFeedback["ch1-c1"];
    expect(fb?.predictCorrect).toBe(true);
    expect(typeof fb?.predictNudge).toBe("string");

    const msgs = store.getState().tutorMessages;
    expect(msgs.length).toBe(beforeLen);
    // Transcript must not reveal correctness (no new tutor/user lines about right/wrong).
    for (const m of msgs) {
      if (beforeMsgs.includes(m.content)) continue;
      expect(m.content.toLowerCase()).not.toMatch(
        /correct|incorrect|right answer|wrong answer|option\s*1/
      );
    }

    const wrong = store.getState().submitPredict("ch1-c1", 2);
    expect(wrong).toBe(false);
    const fb2 = store.getState().challengeFeedback["ch1-c1"];
    expect(fb2?.predictCorrect).toBe(false);
    expect(fb2?.predictNudge).toBeTruthy();
    // Still no tutor transcript growth from predict grading.
    expect(store.getState().tutorMessages.length).toBe(beforeLen);
  });

  it("14. wrong prediction does not unlock experiment step; right one does", () => {
    // Wrong first.
    store.getState().submitPredict("ch1-c1", 3);
    let cp = store.getState().progress.challenges["ch1-c1"];
    expect(cp?.predictAnswer).toBe(3);
    expect(cp?.stepIndex ?? 0).toBe(0);
    expect(cp?.completed).toBeFalsy();

    // Right unlocks past predict (stepIndex >= 1).
    store.getState().submitPredict("ch1-c1", 0);
    cp = store.getState().progress.challenges["ch1-c1"];
    expect(cp?.predictAnswer).toBe(0);
    expect(cp!.stepIndex).toBeGreaterThanOrEqual(1);
  });

  it("15. checkExperimentGate only passes when experimentCheck is satisfied", () => {
    store.getState().setDsl(SMALL_AND_DSL);
    store.getState().parseAndApplyDsl();

    // No snapshot → fail.
    expect(store.getState().checkExperimentGate("ch1-c1")).toBe(false);

    // Wrong dataset / low accuracy → fail.
    store.setState({
      lastSnapshot: fakeSnapshot({ loss: 0.5, accuracy: 0.5 }),
      trainConfig: {
        ...store.getState().trainConfig,
        dataset: "xor",
      },
    });
    expect(store.getState().checkExperimentGate("ch1-c1")).toBe(false);
    expect(
      store.getState().progress.challenges["ch1-c1"]?.experimentPassed
    ).not.toBe(true);

    // Satisfying outcome for ch1-c1: dataset=and, dense, accuracy ≥ 0.99.
    store.setState({
      lastSnapshot: fakeSnapshot({ loss: 0.01, accuracy: 0.995 }),
      trainConfig: {
        ...store.getState().trainConfig,
        dataset: "and",
      },
      dsl: SMALL_AND_DSL,
    });
    expect(store.getState().checkExperimentGate("ch1-c1")).toBe(true);
    expect(
      store.getState().progress.challenges["ch1-c1"]?.experimentPassed
    ).toBe(true);
    expect(
      store.getState().progress.challenges["ch1-c1"]!.stepIndex
    ).toBeGreaterThanOrEqual(2);
  });

  it("16. submitExplain: keyword stuffing fails; genuine answer + all gates completes", () => {
    const challengeId = "ch1-c1";

    // Stuffing alone must fail and leave challenge incomplete.
    const stuffed = Array(12).fill("linear").join(" ");
    expect(store.getState().submitExplain(challengeId, stuffed)).toBe(false);
    expect(
      store.getState().progress.challenges[challengeId]?.completed
    ).toBeFalsy();

    // Predict + experiment required for full completion.
    store.getState().submitPredict(challengeId, 0);
    store.setState({
      lastSnapshot: fakeSnapshot({ loss: 0.01, accuracy: 1 }),
      trainConfig: {
        ...store.getState().trainConfig,
        dataset: "and",
      },
      dsl: SMALL_AND_DSL,
    });
    expect(store.getState().checkExperimentGate(challengeId)).toBe(true);

    const good = PLAUSIBLE_EXPLAIN[challengeId]!;
    expect(store.getState().submitExplain(challengeId, good)).toBe(true);
    const cp = store.getState().progress.challenges[challengeId]!;
    expect(cp.completed).toBe(true);
    expect(cp.stepIndex).toBeGreaterThanOrEqual(3);

    // Explain graded ok without prior predict/experiment must not complete.
    store.setState({
      progress: {
        ...store.getState().progress,
        challenges: {
          ...store.getState().progress.challenges,
          "ch1-c3": {
            challengeId: "ch1-c3",
            stepIndex: 0,
            completed: false,
          },
        },
      },
    });
    expect(
      store.getState().submitExplain("ch1-c3", PLAUSIBLE_EXPLAIN["ch1-c3"]!)
    ).toBe(true);
    expect(store.getState().progress.challenges["ch1-c3"]?.completed).toBe(
      false
    );
  });

  it("17. completing every chapter challenge unlocks next; loadChapter refuses locked", () => {
    const ch1Ids = ["ch1-c1", "ch1-c2", "ch1-c3"] as const;
    const datasets: Record<(typeof ch1Ids)[number], "and" | "or" | "xor"> = {
      "ch1-c1": "and",
      "ch1-c2": "or",
      "ch1-c3": "xor",
    };
    const correctPredict: Record<(typeof ch1Ids)[number], number> = {
      "ch1-c1": 0,
      "ch1-c2": 1,
      "ch1-c3": 1,
    };

    // ch2 is locked before ch1 is done.
    const beforeChapter = store.getState().progress.currentChapterId;
    store.getState().loadChapter("ch2");
    expect(store.getState().progress.currentChapterId).toBe(beforeChapter);
    expect(store.getState().progress.completedChapters).not.toContain("ch1");

    for (const id of ch1Ids) {
      expect(store.getState().submitPredict(id, correctPredict[id])).toBe(true);
      store.setState({
        lastSnapshot: fakeSnapshot({ loss: 0.01, accuracy: 1 }),
        trainConfig: {
          ...store.getState().trainConfig,
          dataset: datasets[id],
        },
        dsl: `network N {\n  dense 2 -> 1 activation=sigmoid\n}\ntrain dataset=${datasets[id]} lr=0.5 epochs=10\n`,
      });
      expect(store.getState().checkExperimentGate(id)).toBe(true);
      expect(
        store.getState().submitExplain(id, PLAUSIBLE_EXPLAIN[id]!)
      ).toBe(true);
      expect(store.getState().progress.challenges[id]?.completed).toBe(true);
    }

    expect(store.getState().progress.completedChapters).toContain("ch1");

    store.getState().loadChapter("ch2");
    expect(store.getState().progress.currentChapterId).toBe("ch2");
    expect(store.getState().dsl).toContain("network");

    // ch3 still locked (needs ch2 completed).
    store.getState().loadChapter("ch3");
    expect(store.getState().progress.currentChapterId).toBe("ch2");
  });

  it("18. progress survives reload via saveProgress/hydrate round-trip", async () => {
    store.getState().submitPredict("ch1-c1", 0);
    store.setState({
      lastSnapshot: fakeSnapshot({ loss: 0.01, accuracy: 1 }),
      trainConfig: {
        ...store.getState().trainConfig,
        dataset: "and",
      },
      dsl: SMALL_AND_DSL,
    });
    store.getState().checkExperimentGate("ch1-c1");
    store.getState().submitExplain("ch1-c1", PLAUSIBLE_EXPLAIN["ch1-c1"]!);

    const saved = structuredClone(store.getState().progress);
    expect(saved.challenges["ch1-c1"]?.completed).toBe(true);

    // Simulate a full page reload: fresh module + hydrate from localStorage.
    // Keep localStorage — only the in-memory Zustand singleton is replaced.
    const store2 = await loadStore(
      { reducedMotion: true },
      { clearStorage: false }
    );
    expect(store2.getState().progress.challenges["ch1-c1"]).toBeUndefined();

    store2.getState().hydrate();
    const hydrated = store2.getState().progress;
    expect(hydrated.challenges["ch1-c1"]?.completed).toBe(true);
    expect(hydrated.challenges["ch1-c1"]?.predictAnswer).toBe(
      saved.challenges["ch1-c1"]?.predictAnswer
    );
    expect(hydrated.challenges["ch1-c1"]?.experimentPassed).toBe(true);
    expect(hydrated.completedChapters).toEqual(saved.completedChapters);
  });
});
