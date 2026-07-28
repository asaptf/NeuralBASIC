"use client";

import { useState } from "react";
import { Markdown } from "@/components/markdown/Markdown";
import { CHAPTERS, isChapterUnlocked } from "@/curriculum/chapters";
import type { ChallengeStep } from "@/curriculum/types";
import { useAppStore, type ChallengeFeedback } from "@/store/useAppStore";

export function ChapterNav() {
  const progress = useAppStore((s) => s.progress);
  const loadChapter = useAppStore((s) => s.loadChapter);
  const activeChallengeId = useAppStore((s) => s.activeChallengeId);
  const setActiveChallenge = useAppStore((s) => s.setActiveChallenge);
  const submitPredict = useAppStore((s) => s.submitPredict);
  const submitExplain = useAppStore((s) => s.submitExplain);
  const trainNow = useAppStore((s) => s.trainNow);
  const feedback = useAppStore((s) => s.challengeFeedback);

  const chapter = CHAPTERS.find((c) => c.id === progress.currentChapterId);

  return (
    <div className="panel flex h-full min-h-0 flex-col" data-testid="curriculum-panel">
      <div className="panel-header">
        <span>Curriculum</span>
        <span className="panel-header-note">
          {progress.completedChapters.length}/{CHAPTERS.length} done
        </span>
      </div>
      <div className="flex gap-1 overflow-x-auto border-b border-[var(--border)] p-2">
        {CHAPTERS.map((ch) => {
          const unlocked = isChapterUnlocked(ch.id, progress.completedChapters);
          const active = progress.currentChapterId === ch.id;
          const done = progress.completedChapters.includes(ch.id);
          return (
            <button
              key={ch.id}
              type="button"
              disabled={!unlocked}
              data-testid={`chapter-tab-${ch.id}`}
              className={`chapter-tab ${active ? "active" : ""} ${done ? "done" : ""}`}
              onClick={() => loadChapter(ch.id)}
              title={
                unlocked
                  ? `Ch ${ch.number}: ${ch.title}`
                  : "Complete the previous chapter's challenges first"
              }
              aria-label={`Chapter ${ch.number}: ${ch.title}${unlocked ? "" : " (locked)"}`}
            >
              {done ? "✓" : ch.number}
            </button>
          );
        })}
      </div>
      {chapter && (
        <div className="min-h-0 flex-1 overflow-auto p-3">
          <h2 className="text-base font-semibold" data-testid="chapter-title">
            Ch {chapter.number}: {chapter.title}
          </h2>
          <p className="mt-1 text-xs opacity-70">{chapter.subtitle}</p>

          <Markdown text={chapter.theory} className="mt-3 opacity-95" />

          <h3 className="mt-4 text-[10px] font-semibold uppercase tracking-[0.08em] opacity-60">
            Challenges
          </h3>
          <ul className="mt-2 space-y-2">
            {chapter.challenges.map((c) => {
              const cp = progress.challenges[c.id];
              const active = activeChallengeId === c.id;
              return (
                <li key={c.id}>
                  <button
                    type="button"
                    className={`challenge-item w-full text-left ${active ? "active" : ""} ${
                      cp?.completed ? "completed" : ""
                    }`}
                    data-testid={`challenge-${c.id}`}
                    aria-expanded={active}
                    onClick={() => setActiveChallenge(active ? null : c.id)}
                  >
                    <div className="text-sm font-medium">
                      {cp?.completed ? "✓ " : ""}
                      {c.title}
                    </div>
                    <div className="text-[11px] opacity-70">{c.description}</div>
                  </button>
                  {active && (
                    <ChallengeBody
                      challengeId={c.id}
                      steps={c.steps}
                      progress={cp}
                      feedback={feedback[c.id]}
                      onPredict={(idx) => submitPredict(c.id, idx)}
                      onExplain={(t) => submitExplain(c.id, t)}
                      onExperiment={() => trainNow("challenge-experiment")}
                    />
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}

function ChallengeBody({
  challengeId,
  steps,
  progress,
  feedback,
  onPredict,
  onExplain,
  onExperiment,
}: {
  challengeId: string;
  steps: ChallengeStep[];
  progress?: {
    stepIndex: number;
    predictAnswer?: number;
    experimentPassed?: boolean;
    completed?: boolean;
  };
  feedback?: ChallengeFeedback;
  onPredict: (i: number) => boolean;
  onExplain: (t: string) => boolean;
  onExperiment: () => void;
}) {
  const isTraining = useAppStore((s) => s.isTraining);
  const predictDone = feedback?.predictCorrect === true;
  const experimentPassed = !!progress?.experimentPassed;

  return (
    <div
      className="challenge-body mt-2 rounded border border-[var(--border)] p-2"
      data-testid={`challenge-body-${challengeId}`}
    >
      {steps.map((step, i) => {
        const locked =
          (step.kind === "experiment" && !predictDone) ||
          (step.kind === "explain" && !experimentPassed);
        const done =
          (step.kind === "predict" && predictDone) ||
          (step.kind === "experiment" && experimentPassed) ||
          (step.kind === "explain" && feedback?.explain?.passed === true);

        return (
          <div
            key={`${step.kind}-${i}`}
            className={`challenge-step ${locked ? "is-locked" : ""} ${done ? "is-done" : ""}`}
          >
            <div className="step-kind">
              <span>{step.kind}</span>
              {done && <span className="step-badge done">done</span>}
              {locked && <span className="step-badge">locked</span>}
            </div>
            <p className="mb-2 text-xs leading-relaxed opacity-90">
              {step.prompt}
            </p>

            {step.kind === "predict" && step.choices && (
              <>
                <div>
                  {step.choices.map((choice, ci) => {
                    const chosen = progress?.predictAnswer === ci;
                    const tone = !chosen
                      ? ""
                      : feedback?.predictCorrect
                        ? "chosen-right"
                        : "chosen-wrong";
                    return (
                      <button
                        key={ci}
                        type="button"
                        className={`btn choice-btn ${tone}`}
                        data-testid={`predict-${challengeId}-${ci}`}
                        aria-pressed={chosen}
                        onClick={() => onPredict(ci)}
                      >
                        {String.fromCharCode(65 + ci)}. {choice}
                      </button>
                    );
                  })}
                </div>
                {feedback?.predictNudge && (
                  <div
                    className={`step-feedback ${feedback.predictCorrect ? "ok" : "nudge"}`}
                    role="status"
                    data-testid={`predict-feedback-${challengeId}`}
                  >
                    <Markdown text={feedback.predictNudge} />
                  </div>
                )}
              </>
            )}

            {step.kind === "experiment" && (
              <>
                <button
                  type="button"
                  className="btn btn-primary text-xs"
                  disabled={locked || isTraining}
                  data-testid={`experiment-${challengeId}`}
                  onClick={onExperiment}
                >
                  {isTraining
                    ? "Training…"
                    : experimentPassed
                      ? "Passed ✓ — train again"
                      : "Train & check"}
                </button>
                {locked && (
                  <div className="locked-note mt-1">
                    Make your prediction first — that&apos;s the point.
                  </div>
                )}
                {!locked && !experimentPassed && progress?.stepIndex != null && (
                  <div className="locked-note mt-1">
                    Target not reached yet. Adjust the DSL or controls and train
                    again.
                  </div>
                )}
              </>
            )}

            {step.kind === "explain" && (
              <>
                <ExplainBox
                  challengeId={challengeId}
                  disabled={locked}
                  onSubmit={onExplain}
                />
                {locked && (
                  <div className="locked-note mt-1">
                    Run the experiment first — explain what you actually saw.
                  </div>
                )}
                {feedback?.explain && (
                  <div
                    className={`step-feedback ${feedback.explain.passed ? "ok" : "nudge"}`}
                    role="status"
                    data-testid={`explain-feedback-${challengeId}`}
                  >
                    <Markdown text={feedback.explain.feedback} />
                    {feedback.explain.matchedConcepts.length > 0 && (
                      <div className="mt-1 text-[11px] opacity-70">
                        Covered: {feedback.explain.matchedConcepts.join(", ")}
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        );
      })}
    </div>
  );
}

function ExplainBox({
  challengeId,
  disabled,
  onSubmit,
}: {
  challengeId: string;
  disabled: boolean;
  onSubmit: (t: string) => boolean;
}) {
  const [text, setText] = useState("");

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit(text);
      }}
    >
      <textarea
        name="explain"
        className="input min-h-[70px] w-full text-xs"
        disabled={disabled}
        value={text}
        onChange={(e) => setText(e.target.value)}
        data-testid={`explain-${challengeId}`}
        placeholder="Explain in your own words — name the mechanism and what you observed…"
      />
      <button
        type="submit"
        className="btn btn-primary mt-1 text-xs"
        disabled={disabled || text.trim().length === 0}
        data-testid={`explain-submit-${challengeId}`}
      >
        Submit explanation
      </button>
    </form>
  );
}
