"use client";

import { useEffect, useRef } from "react";
import { ChapterNav } from "@/components/curriculum/ChapterNav";
import { ImmediateControls } from "@/components/controls/ImmediateControls";
import { CodeEditor } from "@/components/editor/CodeEditor";
import { DataLab } from "@/components/lab/DataLab";
import { GitHubLink } from "@/components/ide/GitHubLink";
import { Logo } from "@/components/ide/Logo";
import { LessonView } from "@/components/curriculum/LessonView";
import { MetricsPanel } from "@/components/metrics/MetricsPanel";
import { TutorPanel } from "@/components/tutor/TutorPanel";
import { NetworkVisualizer } from "@/components/visualizer/NetworkVisualizer";
import { WelcomeGuide } from "@/components/onboarding/WelcomeGuide";
import { useAppStore } from "@/store/useAppStore";

export function IdeShell() {
  const theme = useAppStore((s) => s.theme);
  const setTheme = useAppStore((s) => s.setTheme);
  const hydrate = useAppStore((s) => s.hydrate);
  const saveLocal = useAppStore((s) => s.saveLocal);
  const loadLocal = useAppStore((s) => s.loadLocal);
  const exportExperimentFile = useAppStore((s) => s.exportExperimentFile);
  const exportModel = useAppStore((s) => s.exportModel);
  const importExperimentFile = useAppStore((s) => s.importExperimentFile);
  const trainNow = useAppStore((s) => s.trainNow);
  const setWelcomeOpen = useAppStore((s) => s.setWelcomeOpen);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    hydrate();
  }, [hydrate]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  // Keyboard: Ctrl/Cmd+Enter trains
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        trainNow("keyboard");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [trainNow]);

  return (
    <div className="ide-root" data-theme={theme} data-testid="ide-shell">
      <header className="ide-header">
        <div className="flex items-center gap-3">
          <div className="logo" data-testid="app-logo">
            <Logo />
            <span className="logo-text">NeuralBASIC</span>
          </div>
          <span className="hidden text-xs opacity-60 sm:inline">
            NeuronPad · AI Tablet · Immediate Mode
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div className="btn-cluster" role="group" aria-label="Experiment file actions">
            <button
              type="button"
              className="btn btn-ghost"
              data-testid="btn-save"
              onClick={() => saveLocal()}
            >
              Save
            </button>
            <button
              type="button"
              className="btn btn-ghost"
              data-testid="btn-load"
              onClick={() => loadLocal()}
            >
              Load
            </button>
            <button
              type="button"
              className="btn btn-ghost"
              data-testid="btn-export-exp"
              onClick={() => exportExperimentFile()}
            >
              Export JSON
            </button>
            <button
              type="button"
              className="btn btn-ghost"
              data-testid="btn-import"
              onClick={() => fileRef.current?.click()}
            >
              Import
            </button>
          </div>
          <input
            ref={fileRef}
            type="file"
            accept="application/json,.json"
            className="hidden"
            onChange={async (e) => {
              const f = e.target.files?.[0];
              if (!f) return;
              const text = await f.text();
              importExperimentFile(text);
              e.target.value = "";
            }}
          />
          <button
            type="button"
            className="btn btn-primary"
            data-testid="btn-export-model"
            onClick={() => exportModel()}
          >
            Export Model
          </button>
          <div
            className="theme-toggle"
            data-testid="theme-toggle"
            role="group"
            aria-label="Theme"
          >
            <button
              type="button"
              className={theme === "modern" ? "active" : ""}
              data-testid="theme-modern"
              data-theme-marker="modern-dark"
              onClick={() => setTheme("modern")}
            >
              Modern Dark
            </button>
            <button
              type="button"
              className={theme === "retro" ? "active" : ""}
              data-testid="theme-retro"
              data-theme-marker="retro-blue"
              onClick={() => setTheme("retro")}
            >
              Retro Blue
            </button>
          </div>
          {/* Icon-only for the same reason the source link is: the header is
              already at its limit of worded buttons, and a reader looking for
              the guide is looking for a question mark. */}
          <button
            type="button"
            className="btn btn-ghost btn-icon"
            data-testid="btn-welcome"
            aria-label="Open the welcome guide"
            title="Welcome guide"
            onClick={() => setWelcomeOpen(true)}
          >
            <svg
              viewBox="0 0 16 16"
              width="15"
              height="15"
              aria-hidden="true"
              focusable="false"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
            >
              <circle cx="8" cy="8" r="6.7" />
              <path
                d="M6.1 6.1a1.95 1.95 0 1 1 2.6 1.85c-.45.17-.7.55-.7 1.02v.28"
                strokeLinecap="round"
              />
              <circle cx="8" cy="11.7" r="0.85" fill="currentColor" stroke="none" />
            </svg>
          </button>
          <GitHubLink />
        </div>
      </header>

      <ImmediateControls />

      <main className="ide-grid">
        <section className="cell cell-curriculum">
          <ChapterNav />
        </section>
        <section className="cell cell-editor">
          <CodeEditor />
        </section>
        <section className="cell cell-lab">
          <DataLab />
        </section>
        <section className="cell cell-viz">
          <NetworkVisualizer />
        </section>
        <section className="cell cell-metrics">
          <MetricsPanel />
        </section>
        <section className="cell cell-tutor">
          <TutorPanel />
        </section>
      </main>

      <LessonView />
      <WelcomeGuide />

      <footer className="ide-footer">
        <span>Ctrl/⌘+Enter = Train</span>
        <span className="opacity-60">
          Socratic · Immediate Mode · No notebook cells
        </span>
      </footer>
    </div>
  );
}
