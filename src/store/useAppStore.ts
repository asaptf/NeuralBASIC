"use client";

import { create } from "zustand";
import {
  CHAPTERS,
  getChapter,
  isChapterUnlocked,
} from "@/curriculum/chapters";
import { experimentCheckPasses } from "@/curriculum/gates";
import type {
  ChallengeProgress,
  CurriculumProgress,
} from "@/curriculum/types";
import {
  boundsFromSamples,
  buildDecisionGrid,
  cloneModelFromWeights,
  createTrainingSession,
  DSLInputShapeError,
  evaluateModel,
  forward,
  getDataset,
  parseDSL,
  prepareNetworkConfig,
  snapshotLayers,
} from "@/engine";
import type { TrainingSession } from "@/engine/train";
import type {
  LayerWeights,
  NetworkConfig,
  TrainConfig,
  TrainStepResult,
} from "@/engine/types";
import {
  buildModelExport,
  modelExportToJSON,
  toPyTorchSnippet,
} from "@/engine/export";
import {
  downloadText,
  exportExperimentJSON,
  importExperimentJSON,
  loadExperiment,
  loadProgress,
  loadTheme,
  saveExperiment,
  saveProgress,
  saveTheme,
  type ExperimentState,
} from "@/lib/persistence";
import {
  enforceSocratic,
  mockTutorReply,
  gradeExplanation,
  gradePredictAnswer,
  type ExplainGradeResult,
  type TutorContext,
  type TutorMessage,
} from "@/tutor";
import { defaultStarterDSL } from "@/engine/dsl";

export type ThemeId = "modern" | "retro";

/** Inline feedback shown next to a challenge step. */
export interface ChallengeFeedback {
  predictCorrect?: boolean;
  predictNudge?: string;
  explain?: ExplainGradeResult;
}

interface AppState {
  theme: ThemeId;
  dsl: string;
  network: NetworkConfig;
  trainConfig: TrainConfig;
  weights: LayerWeights[];
  history: {
    losses: number[];
    accuracies: number[];
    /** Held-out series. Empty when the dataset was too small to split. */
    valLosses: number[];
    valAccuracies: number[];
  };
  lastSnapshot: TrainStepResult | null;
  isTraining: boolean;
  isPaused: boolean;
  epochsRun: number;
  totalEpochs: number;
  /** DSL problems — surfaced under the editor, because that's where the code is. */
  parseError: string | null;
  /** Failures of a command the user just issued (Load, Import, a run that threw).
      Not syntax, so it must not appear under the editor. */
  actionError: string | null;
  progress: CurriculumProgress;
  tutorMessages: TutorMessage[];
  activeChallengeId: string | null;
  challengeFeedback: Record<string, ChallengeFeedback>;
  /** Immediate Mode: last control that triggered train */
  lastTrigger: string | null;

  setTheme: (t: ThemeId) => void;
  setDsl: (dsl: string) => void;
  setLearningRate: (lr: number) => void;
  setEpochs: (n: number) => void;
  setDataset: (d: TrainConfig["dataset"]) => void;
  parseAndApplyDsl: () => boolean;
  clearActionError: () => void;
  /** Load a lesson's runnable example into the editor and train it. */
  runExample: (dsl: string) => void;
  lessonOpen: boolean;
  setLessonOpen: (open: boolean) => void;
  trainNow: (reason?: string) => void;
  pauseTraining: () => void;
  resumeTraining: () => void;
  stepEpoch: () => void;
  resetWeights: () => void;
  loadChapter: (chapterId: string) => void;
  setActiveChallenge: (id: string | null) => void;
  submitPredict: (challengeId: string, choiceIndex: number) => boolean;
  submitExplain: (challengeId: string, text: string) => boolean;
  checkExperimentGate: (challengeId: string) => boolean;
  sendTutorMessage: (text: string) => void;
  saveLocal: () => void;
  loadLocal: () => void;
  exportExperimentFile: () => void;
  importExperimentFile: (json: string) => void;
  exportModel: () => { json: string; pytorch: string };
  hydrate: () => void;
}

function defaultProgress(): CurriculumProgress {
  return {
    currentChapterId: "ch1",
    completedChapters: [],
    challenges: {},
  };
}

function initialFromChapter(chapterId: string) {
  const ch = getChapter(chapterId);
  const dsl = ch?.starterDSL ?? defaultStarterDSL("ch1");
  const parsed = parseDSL(dsl);
  return { dsl, network: parsed.network, trainConfig: parsed.train };
}

const init = initialFromChapter("ch1");

function mid(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function tutorCtx(s: {
  progress: CurriculumProgress;
  dsl: string;
  lastSnapshot: TrainStepResult | null;
  activeChallengeId: string | null;
}): TutorContext {
  const ch = getChapter(s.progress.currentChapterId);
  return {
    chapterId: s.progress.currentChapterId,
    chapterTitle: ch?.title ?? "NeuralBASIC",
    theorySnippet: ch?.theory?.slice(0, 200),
    dsl: s.dsl,
    metrics: s.lastSnapshot
      ? {
          loss: s.lastSnapshot.loss,
          accuracy: s.lastSnapshot.accuracy,
          epoch: s.lastSnapshot.epoch,
        }
      : null,
    challengeId: s.activeChallengeId,
    challengeStep: s.activeChallengeId
      ? s.progress.challenges[s.activeChallengeId]
        ? `step ${s.progress.challenges[s.activeChallengeId]!.stepIndex}`
        : null
      : null,
  };
}

/**
 * Rebuild a displayable snapshot from restored weights.
 *
 * Loading an experiment restores weights but no rendered state, so without this
 * every panel keeps showing the *previous* run: the Data Lab draws the old
 * boundary under the loaded dataset's points and the metric cards report the old
 * loss. Recomputing here makes all panels agree with the model that's loaded.
 */
function snapshotFromWeights(
  network: NetworkConfig,
  trainConfig: TrainConfig,
  weights: LayerWeights[]
): TrainStepResult | null {
  if (!weights.length) return null;
  try {
    const prepared = prepareNetworkConfig(network, trainConfig.dataset);
    const model = cloneModelFromWeights(
      prepared.config,
      weights,
      prepared.inputDim
    );
    const ds = getDataset(trainConfig.dataset);
    // Populate activations so the network panel lights up.
    if (ds.samples[0]) forward(model, ds.samples[0].x);
    const { loss, accuracy } = evaluateModel(model, trainConfig.dataset);
    const is2d = ds.inputShape.length === 1 && ds.inputShape[0] === 2;
    const b = is2d ? boundsFromSamples(ds.samples) : null;
    return {
      epoch: 0,
      loss,
      accuracy,
      layerSnapshots: snapshotLayers(model),
      decisionGrid:
        is2d && b
          ? buildDecisionGrid(model, 24, b.xMin, b.xMax, b.yMin, b.yMax)
          : undefined,
    };
  } catch {
    // A restored file can disagree with the current engine; degrade to "untrained"
    // rather than taking the whole app down.
    return null;
  }
}

/* ── Animated training loop ──
   Training runs epoch-by-epoch off requestAnimationFrame so the learner
   actually watches the network learn. The session lives outside React state
   because it is mutable and re-created on every run. */

let session: TrainingSession | null = null;
let rafId: number | null = null;
let timerId: ReturnType<typeof setTimeout> | null = null;
/** Wall-clock anchor for pacing, so a throttled rAF doesn't stretch the run. */
let runStartedAt = 0;
let epochsAtStart = 0;

/** A full run animates over roughly this long, whatever the epoch count. */
const TARGET_RUN_MS = 2500;

function now(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function isHidden(): boolean {
  return typeof document !== "undefined" && document.hidden;
}

/**
 * Set once we've seen rAF fail to fire: some environments report the document
 * as visible while never painting (offscreen panes, embedded webviews), which
 * would wedge a run at epoch 0 with the controls disabled forever.
 */
let forceTimer = false;

/**
 * requestAnimationFrame never fires while the tab is hidden either, so fall
 * back to a timer whenever rAF can't be trusted. The run always finishes.
 */
function schedule(cb: () => void) {
  if (isHidden() || forceTimer) {
    timerId = setTimeout(cb, 32);
  } else {
    rafId = requestAnimationFrame(cb);
  }
}

let watchdogId: ReturnType<typeof setTimeout> | null = null;

function cancelLoop() {
  if (rafId != null && typeof cancelAnimationFrame !== "undefined") {
    cancelAnimationFrame(rafId);
  }
  rafId = null;
  if (timerId != null) {
    clearTimeout(timerId);
    timerId = null;
  }
  if (watchdogId != null) {
    clearTimeout(watchdogId);
    watchdogId = null;
  }
}

function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

export const useAppStore = create<AppState>((set, get) => {
  /**
   * Adopt a restored experiment. Recomputes the snapshot from its weights so the
   * Data Lab, network panel and metric cards all describe the loaded model
   * instead of whatever was on screen from the previous run.
   */
  const applyLoadedExperiment = (exp: ExperimentState) => {
    cancelLoop();
    session = null;
    // Files saved before held-out metrics existed carry no val series.
    const history = {
      losses: exp.history?.losses ?? [],
      accuracies: exp.history?.accuracies ?? [],
      valLosses: exp.history?.valLosses ?? [],
      valAccuracies: exp.history?.valAccuracies ?? [],
    };
    const weights = exp.weights ?? [];
    const snapshot = snapshotFromWeights(exp.network, exp.trainConfig, weights);
    set({
      dsl: exp.dsl,
      network: exp.network,
      trainConfig: exp.trainConfig,
      weights,
      history,
      lastSnapshot: snapshot,
      isTraining: false,
      isPaused: false,
      epochsRun: history.losses.length,
      totalEpochs: history.losses.length,
      parseError: null,
      actionError: null,
      lastTrigger: "loaded",
    });
  };

  /** Discard results that no longer describe the current dataset/model. */
  const clearRunState = () => {
    cancelLoop();
    session = null;
    set({
      weights: [],
      history: { losses: [], accuracies: [], valLosses: [], valAccuracies: [] },
      lastSnapshot: null,
      isTraining: false,
      isPaused: false,
      epochsRun: 0,
      totalEpochs: 0,
    });
  };

  /** Push the session's current state into React. */
  const syncFromSession = () => {
    if (!session) return;
    set({
      history: {
        losses: session.losses.slice(),
        accuracies: session.accuracies.slice(),
        valLosses: session.valLosses ? session.valLosses.slice() : [],
        valAccuracies: session.valAccuracies
          ? session.valAccuracies.slice()
          : [],
      },
      lastSnapshot: session.lastSnapshot,
      epochsRun: session.epochsRun,
    });
  };

  const finishRun = () => {
    cancelLoop();
    if (session) {
      set({ weights: session.exportWeights() });
      syncFromSession();
    }
    set({ isTraining: false, isPaused: false });
    const active = get().activeChallengeId;
    if (active) get().checkExperimentGate(active);
  };

  const frame = () => {
    rafId = null;
    timerId = null;
    if (!session || get().isPaused) return;

    const frameStart = now();

    if (isHidden()) {
      // Nobody is watching — drain as fast as the slice allows instead of
      // pacing an animation nobody can see.
      while (!session.isDone && now() - frameStart < 100) {
        session.runEpoch();
      }
    } else {
      // Pace against the wall clock rather than the frame count: if the browser
      // throttles rAF, we catch up instead of dragging the run out. The 12ms
      // budget keeps any single frame from blocking input on heavy models.
      const elapsed = now() - runStartedAt;
      const shouldHaveRun =
        epochsAtStart +
        Math.ceil((session.totalEpochs * elapsed) / TARGET_RUN_MS);

      while (
        !session.isDone &&
        session.epochsRun < shouldHaveRun &&
        now() - frameStart < 12
      ) {
        session.runEpoch();
      }
      // Always make progress, even if the budget was already blown.
      if (!session.isDone && session.epochsRun === epochsAtStart) {
        session.runEpoch();
      }
    }

    syncFromSession();

    if (session.isDone) {
      finishRun();
      return;
    }
    schedule(frame);
  };

  /** Re-anchor pacing when the tab comes back, so we don't cram the backlog. */
  const onVisibilityChange = () => {
    if (!session || !get().isTraining || get().isPaused) return;
    runStartedAt = now();
    epochsAtStart = session.epochsRun;
    // A hidden-tab timer and a visible-tab rAF are different schedulers.
    cancelLoop();
    schedule(frame);
  };

  if (typeof document !== "undefined") {
    document.addEventListener("visibilitychange", onVisibilityChange);
  }

  /**
   * If rAF hasn't advanced a single epoch after a second, it isn't going to.
   * Switch this session — and every later one — to the timer driver.
   */
  const armWatchdog = () => {
    if (watchdogId != null) clearTimeout(watchdogId);
    const startedAtEpoch = session?.epochsRun ?? 0;
    watchdogId = setTimeout(() => {
      watchdogId = null;
      const s = get();
      if (!session || !s.isTraining || s.isPaused) return;
      if (session.epochsRun > startedAtEpoch) return; // progressing fine
      forceTimer = true;
      cancelLoop();
      runStartedAt = now();
      epochsAtStart = session.epochsRun;
      schedule(frame);
    }, 1000);
  };

  const startLoop = () => {
    cancelLoop();
    runStartedAt = now();
    epochsAtStart = session?.epochsRun ?? 0;
    schedule(frame);
    if (!forceTimer && !isHidden()) armWatchdog();
  };

  return {
    theme: "modern",
    dsl: init.dsl,
    network: init.network,
    trainConfig: init.trainConfig,
    weights: [],
    history: { losses: [], accuracies: [], valLosses: [], valAccuracies: [] },
    lastSnapshot: null,
    isTraining: false,
    isPaused: false,
    epochsRun: 0,
    totalEpochs: 0,
    parseError: null,
    actionError: null,
    progress: defaultProgress(),
    tutorMessages: [
      {
        id: mid(),
        role: "tutor",
        content:
          "Welcome to NeuralBASIC. I'm your Socratic Tutor — I won't hand you full solutions. Predict, train, observe, explain. Ready for Chapter 1?",
        ts: Date.now(),
      },
    ],
    activeChallengeId: null,
    challengeFeedback: {},
    lessonOpen: false,
    lastTrigger: null,

    setTheme: (t) => {
      set({ theme: t });
      saveTheme(t);
      if (typeof document !== "undefined") {
        document.documentElement.dataset.theme = t;
      }
    },

    setDsl: (dsl) => set({ dsl }),

    setLearningRate: (lr) => {
      const trainConfig = { ...get().trainConfig, learningRate: lr };
      const dsl = syncTrainInDsl(get().dsl, trainConfig);
      set({ trainConfig, dsl });
    },

    setEpochs: (n) => {
      const trainConfig = { ...get().trainConfig, epochs: n };
      const dsl = syncTrainInDsl(get().dsl, trainConfig);
      set({ trainConfig, dsl });
    },

    setDataset: (d) => {
      if (d === get().trainConfig.dataset) return;
      const trainConfig = { ...get().trainConfig, dataset: d };
      const dsl = syncTrainInDsl(get().dsl, trainConfig);
      // The previous run's loss curve and accuracy describe a different dataset;
      // keeping them on screen under the new dataset's name is a lie.
      clearRunState();
      const shapeError = shapeErrorFor(dsl);
      set({
        trainConfig,
        dsl,
        parseError:
          shapeError === undefined ? get().parseError : shapeError,
      });
    },

    parseAndApplyDsl: () => {
      try {
        const parsed = parseDSL(get().dsl);
        set({
          network: parsed.network,
          trainConfig: parsed.train,
          parseError: null,
        });
        return true;
      } catch (e) {
        set({ parseError: e instanceof Error ? e.message : "Parse error" });
        return false;
      }
    },

    clearActionError: () => set({ actionError: null }),

    /**
     * The lesson's examples are the reason to teach inside the app: one click
     * puts the described network on screen in the same lab the reader is about
     * to experiment in.
     */
    runExample: (dsl) => {
      cancelLoop();
      session = null;
      set({
        dsl,
        weights: [],
        history: { losses: [], accuracies: [], valLosses: [], valAccuracies: [] },
        lastSnapshot: null,
        isTraining: false,
        isPaused: false,
        epochsRun: 0,
        totalEpochs: 0,
        parseError: null,
        actionError: null,
      });
      get().trainNow("lesson-example");
    },

    setLessonOpen: (open) => set({ lessonOpen: open }),

    trainNow: (reason = "manual") => {
      cancelLoop();
      set({ lastTrigger: reason, parseError: null, actionError: null });

      let parsed;
      try {
        parsed = parseDSL(get().dsl);
      } catch (e) {
        set({
          isTraining: false,
          parseError: e instanceof Error ? e.message : "Parse error",
        });
        return;
      }

      try {
        const epochs = Math.min(
          parsed.train.epochs,
          maxEpochsFor(parsed.network)
        );
        const runConfig = { ...parsed.train, epochs };

        session = createTrainingSession(parsed.network, runConfig, {
          includeDecisionBoundary: true,
          maxSamples: maxSamplesFor(parsed.network),
        });

        set({
          network: parsed.network,
          trainConfig: parsed.train,
          weights: [],
          history: { losses: [], accuracies: [], valLosses: [], valAccuracies: [] },
          lastSnapshot: null,
          isTraining: true,
          isPaused: false,
          epochsRun: 0,
          totalEpochs: epochs,
        });

        // Respect reduced-motion: run straight to the end, no animation.
        if (prefersReducedMotion()) {
          while (!session.isDone) session.runEpoch();
          finishRun();
          return;
        }

        startLoop();
      } catch (e) {
        session = null;
        set({
          isTraining: false,
          isPaused: false,
          actionError: e instanceof Error ? e.message : "Training failed",
        });
      }
    },

    pauseTraining: () => {
      if (!get().isTraining) return;
      cancelLoop();
      set({ isPaused: true });
    },

    resumeTraining: () => {
      if (!get().isTraining || !get().isPaused) return;
      set({ isPaused: false });
      startLoop();
    },

    /**
     * Advance exactly one epoch — the "productive struggle" microscope.
     * From idle this must still move a single epoch, so it opens a paused
     * session directly rather than going through trainNow() (which would start
     * the animation loop, or under reduced-motion run the whole thing).
     */
    stepEpoch: () => {
      cancelLoop();

      if (!session) {
        let parsed;
        try {
          parsed = parseDSL(get().dsl);
        } catch (e) {
          set({
            parseError: e instanceof Error ? e.message : "Parse error",
          });
          return;
        }
        const epochs = Math.min(
          parsed.train.epochs,
          maxEpochsFor(parsed.network)
        );
        try {
          session = createTrainingSession(
            parsed.network,
            { ...parsed.train, epochs },
            {
              includeDecisionBoundary: true,
              maxSamples: maxSamplesFor(parsed.network),
            }
          );
        } catch (e) {
          session = null;
          set({
            actionError: e instanceof Error ? e.message : "Training failed",
          });
          return;
        }
        set({
          network: parsed.network,
          trainConfig: parsed.train,
          weights: [],
          history: { losses: [], accuracies: [], valLosses: [], valAccuracies: [] },
          lastSnapshot: null,
          totalEpochs: epochs,
          epochsRun: 0,
          parseError: null,
          lastTrigger: "step",
        });
      }

      if (session.isDone) {
        set({ isTraining: false, isPaused: false });
        return;
      }

      session.runEpoch();
      syncFromSession();
      set({ lastTrigger: "step" });

      if (session.isDone) {
        finishRun();
      } else {
        // Stay paused so repeated presses walk forward one epoch at a time.
        set({ isTraining: true, isPaused: true });
      }
    },

    resetWeights: () => {
      cancelLoop();
      session = null;
      set({
        weights: [],
        history: { losses: [], accuracies: [], valLosses: [], valAccuracies: [] },
        lastSnapshot: null,
        isTraining: false,
        isPaused: false,
        epochsRun: 0,
        totalEpochs: 0,
      });
    },

    loadChapter: (chapterId) => {
      const ch = getChapter(chapterId);
      if (!ch) return;
      const { progress } = get();
      if (!isChapterUnlocked(chapterId, progress.completedChapters)) return;
      cancelLoop();
      session = null;
      const dsl = ch.starterDSL;
      const parsed = parseDSL(dsl);
      set({
        dsl,
        network: parsed.network,
        trainConfig: parsed.train,
        weights: [],
        history: { losses: [], accuracies: [], valLosses: [], valAccuracies: [] },
        lastSnapshot: null,
        isTraining: false,
        isPaused: false,
        epochsRun: 0,
        totalEpochs: 0,
        progress: { ...progress, currentChapterId: chapterId },
        activeChallengeId: ch.challenges[0]?.id ?? null,
        tutorMessages: [
          ...get().tutorMessages,
          {
            id: mid(),
            role: "tutor",
            content: `Chapter ${ch.number}: ${ch.title}. ${ch.subtitle}. Predict something before you train.`,
            ts: Date.now(),
          },
        ],
      });
      saveProgress({ ...get().progress, currentChapterId: chapterId });
    },

    setActiveChallenge: (id) => set({ activeChallengeId: id }),

    submitPredict: (challengeId, choiceIndex) => {
      const ch = CHAPTERS.flatMap((c) => c.challenges).find(
        (c) => c.id === challengeId
      );
      if (!ch) return false;
      const step = ch.steps.find((s) => s.kind === "predict");
      if (!step || step.correctIndex === undefined) return false;

      const graded = gradePredictAnswer(
        choiceIndex,
        step.correctIndex,
        step.choices
      );
      const prev = get().progress.challenges[challengeId] ?? {
        challengeId,
        stepIndex: 0,
        completed: false,
      };
      const next: ChallengeProgress = {
        ...prev,
        predictAnswer: choiceIndex,
        stepIndex: graded.correct ? Math.max(prev.stepIndex, 1) : prev.stepIndex,
      };
      const progress = {
        ...get().progress,
        challenges: { ...get().progress.challenges, [challengeId]: next },
      };
      set({
        progress,
        challengeFeedback: {
          ...get().challengeFeedback,
          [challengeId]: {
            ...get().challengeFeedback[challengeId],
            predictCorrect: graded.correct,
            predictNudge: graded.nudge,
          },
        },
      });
      saveProgress(progress);
      return graded.correct;
    },

    submitExplain: (challengeId, text) => {
      const ch = CHAPTERS.flatMap((c) => c.challenges).find(
        (c) => c.id === challengeId
      );
      if (!ch) return false;
      const step = ch.steps.find((s) => s.kind === "explain");

      const graded = gradeExplanation(text, step?.explainConcepts ?? []);
      const ok = graded.passed;

      const prev = get().progress.challenges[challengeId] ?? {
        challengeId,
        stepIndex: 0,
        completed: false,
      };
      const next: ChallengeProgress = {
        ...prev,
        explainText: text,
        stepIndex: ok ? Math.max(prev.stepIndex, 3) : prev.stepIndex,
        completed: prev.completed,
      };
      // Full completion still requires predict + experiment + explain.
      next.completed =
        ok &&
        prev.predictAnswer !== undefined &&
        (prev.experimentPassed === true ||
          !ch.steps.some((s) => s.kind === "experiment"));

      let progress: CurriculumProgress = {
        ...get().progress,
        challenges: { ...get().progress.challenges, [challengeId]: next },
      };

      const chapter = CHAPTERS.find((c) =>
        c.challenges.some((x) => x.id === challengeId)
      );
      if (chapter) {
        const allDone = chapter.challenges.every(
          (c) => progress.challenges[c.id]?.completed
        );
        if (allDone && !progress.completedChapters.includes(chapter.id)) {
          progress = {
            ...progress,
            completedChapters: [...progress.completedChapters, chapter.id],
          };
        }
      }

      set({
        progress,
        challengeFeedback: {
          ...get().challengeFeedback,
          [challengeId]: {
            ...get().challengeFeedback[challengeId],
            explain: graded,
          },
        },
      });
      saveProgress(progress);

      set({
        tutorMessages: [
          ...get().tutorMessages,
          { id: mid(), role: "user", content: text, ts: Date.now() },
          {
            id: mid(),
            role: "tutor",
            content: enforceSocratic(graded.feedback),
            ts: Date.now(),
          },
        ],
      });
      return ok;
    },

    checkExperimentGate: (challengeId) => {
      const ch = CHAPTERS.flatMap((c) => c.challenges).find(
        (c) => c.id === challengeId
      );
      if (!ch) return false;
      const step = ch.steps.find((s) => s.kind === "experiment");
      const check = step?.experimentCheck;
      if (!check) return true;

      const { lastSnapshot, trainConfig, dsl } = get();
      if (!lastSnapshot) return false;

      const passed = experimentCheckPasses(check, {
        accuracy: lastSnapshot.accuracy,
        loss: lastSnapshot.loss,
        dataset: trainConfig.dataset,
        dsl,
      });
      if (!passed) return false;

      const prev = get().progress.challenges[challengeId] ?? {
        challengeId,
        stepIndex: 0,
        completed: false,
      };
      const next: ChallengeProgress = {
        ...prev,
        experimentPassed: true,
        stepIndex: Math.max(prev.stepIndex, 2),
      };
      const progress = {
        ...get().progress,
        challenges: { ...get().progress.challenges, [challengeId]: next },
      };
      set({ progress });
      saveProgress(progress);
      return true;
    },

    sendTutorMessage: (text) => {
      const userMsg: TutorMessage = {
        id: mid(),
        role: "user",
        content: text,
        ts: Date.now(),
      };
      const reply = enforceSocratic(
        mockTutorReply(text, tutorCtx(get()), get().tutorMessages)
      );
      const tutorMsg: TutorMessage = {
        id: mid(),
        role: "tutor",
        content: reply,
        ts: Date.now(),
      };
      set({ tutorMessages: [...get().tutorMessages, userMsg, tutorMsg] });
    },

    saveLocal: () => {
      const s = get();
      const exp: ExperimentState = {
        dsl: s.dsl,
        network: s.network,
        trainConfig: s.trainConfig,
        weights: s.weights,
        history: s.history,
        name: s.network.name,
      };
      saveExperiment(exp);
      saveProgress(s.progress);
    },

    loadLocal: () => {
      const exp = loadExperiment();
      if (!exp) {
        set({ actionError: "Nothing saved yet — press Save first." });
        return;
      }
      applyLoadedExperiment(exp);
    },

    exportExperimentFile: () => {
      const s = get();
      const json = exportExperimentJSON({
        dsl: s.dsl,
        network: s.network,
        trainConfig: s.trainConfig,
        weights: s.weights,
        history: s.history,
        name: s.network.name,
      });
      downloadText("neuralbasic-experiment.json", json);
    },

    importExperimentFile: (json) => {
      try {
        applyLoadedExperiment(importExperimentJSON(json));
      } catch (e) {
        set({
          actionError:
            e instanceof Error
              ? `Import failed: ${e.message}`
              : "Import failed: not a NeuralBASIC experiment file",
        });
      }
    },

    exportModel: () => {
      const s = get();
      const exp = buildModelExport(
        s.network.name ?? "NeuralBASICNet",
        s.network,
        s.trainConfig,
        s.weights,
        s.lastSnapshot
          ? { loss: s.lastSnapshot.loss, accuracy: s.lastSnapshot.accuracy }
          : undefined
      );
      const json = modelExportToJSON(exp);
      const pytorch = toPyTorchSnippet(s.network, s.trainConfig);
      downloadText("neuralbasic-model.json", json);
      downloadText("neuralbasic-model.py", pytorch, "text/x-python");
      return { json, pytorch };
    },

    hydrate: () => {
      const theme = loadTheme();
      const progress = loadProgress();
      if (theme) {
        set({ theme });
        if (typeof document !== "undefined") {
          document.documentElement.dataset.theme = theme;
        }
      }
      if (progress) set({ progress });
    },
  };
});

/**
 * What the editor strip should say after the dataset changed:
 * - a string: this dataset cannot supply the declared input shape. That is a DSL
 *   problem, so it belongs under the editor — and the learner should see it on
 *   the dropdown change, not only after pressing Train (where the mismatch would
 *   otherwise sit as NaN loss or a crashed run). Same wording and line number
 *   Train gives it.
 * - null: the program is fine against the new dataset; clear the strip.
 * - undefined: some other parse failure. Leave the strip exactly as it is — a
 *   half-typed program must not start nagging from a dropdown change, and a
 *   syntax error Train already reported must not be erased by one.
 */
function shapeErrorFor(dsl: string): string | null | undefined {
  try {
    parseDSL(dsl);
    return null;
  } catch (e) {
    return e instanceof DSLInputShapeError ? e.message : undefined;
  }
}

function syncTrainInDsl(dsl: string, train: TrainConfig): string {
  const trainLine = `train dataset=${train.dataset} lr=${train.learningRate} epochs=${train.epochs}`;
  if (/^train\b/im.test(dsl)) {
    return dsl.replace(/^train\b.*$/im, trainLine);
  }
  return `${dsl.trimEnd()}\n${trainLine}\n`;
}

function maxEpochsFor(network: NetworkConfig): number {
  const hasAttn = network.layers.some(
    (l) => l.type === "attention" || l.type === "transformer_block"
  );
  // Attention uses finite-diff — more epochs help; UI still caps for snappiness
  if (hasAttn) return 80;
  return 400;
}

function maxSamplesFor(network: NetworkConfig): number {
  const hasAttn = network.layers.some(
    (l) => l.type === "attention" || l.type === "transformer_block"
  );
  // Conv uses analytical backprop — use full toy sets
  if (hasAttn) return 16;
  return 128;
}
