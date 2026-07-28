"use client";

import { useEffect, useRef } from "react";
import { Markdown } from "@/components/markdown/Markdown";
import { CHAPTERS } from "@/curriculum/chapters";
import type { LessonExample } from "@/curriculum/types";
import { useAppStore } from "@/store/useAppStore";

/**
 * The reading surface for a chapter's lesson.
 *
 * A textbook cannot live in a 270px sidebar — prose needs a measure of roughly
 * 60–75 characters to be readable at all. So the lesson opens as a sheet over
 * the workspace, and closes back to it. Its examples run in the lab underneath,
 * which is the whole argument for teaching inside the app instead of in a PDF.
 */
export function LessonView() {
  const open = useAppStore((s) => s.lessonOpen);
  const setOpen = useAppStore((s) => s.setLessonOpen);
  const chapterId = useAppStore((s) => s.progress.currentChapterId);
  const runExample = useAppStore((s) => s.runExample);
  const sheetRef = useRef<HTMLDivElement>(null);

  const chapter = CHAPTERS.find((c) => c.id === chapterId);

  // Escape closes, and focus moves into the sheet so the reader can scroll it
  // with the keyboard straight away.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    sheetRef.current?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [open, setOpen]);

  if (!open || !chapter) return null;

  const lesson = chapter.lesson;

  return (
    <div className="lesson-scrim" data-testid="lesson-view">
      <div
        className="lesson-sheet"
        role="dialog"
        aria-modal="true"
        aria-label={`Chapter ${chapter.number}: ${chapter.title}`}
        tabIndex={-1}
        ref={sheetRef}
      >
        <header className="lesson-head">
          <div>
            <p className="lesson-eyebrow">Chapter {chapter.number}</p>
            <h1 className="lesson-title">{chapter.title}</h1>
            <p className="lesson-subtitle">{chapter.subtitle}</p>
          </div>
          <button
            type="button"
            className="btn"
            onClick={() => setOpen(false)}
            data-testid="lesson-close"
          >
            Close ✕
          </button>
        </header>

        {chapter.goals.length > 0 && (
          <section className="lesson-goals" aria-label="What you'll be able to do">
            <h2 className="lesson-goals-title">By the end of this chapter</h2>
            <ul>
              {chapter.goals.map((g) => (
                <li key={g}>{g}</li>
              ))}
            </ul>
          </section>
        )}

        {lesson ? (
          <div className="lesson-body">
            {lesson.map((section, i) => (
              <section key={section.heading} className="lesson-section">
                <h2 className="lesson-heading">
                  <span className="lesson-heading-num">{i + 1}</span>
                  {section.heading}
                </h2>
                <Markdown text={section.body} className="lesson-prose" />
                {section.example && (
                  <ExampleBlock
                    example={section.example}
                    onRun={() => {
                      runExample(section.example!.dsl);
                      setOpen(false);
                    }}
                  />
                )}
              </section>
            ))}
          </div>
        ) : (
          <div className="lesson-body">
            <Markdown text={chapter.theory} className="lesson-prose" />
            <p className="lesson-pending">
              The long-form lesson for this chapter is still being written. The
              summary above and the challenges are complete and usable.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function ExampleBlock({
  example,
  onRun,
}: {
  example: LessonExample;
  onRun: () => void;
}) {
  return (
    <figure className="lesson-example">
      <pre className="lesson-example-code">
        <code>{example.dsl}</code>
      </pre>
      <figcaption className="lesson-example-note">
        <strong>Look for:</strong> {example.expect}
      </figcaption>
      <button
        type="button"
        className="btn btn-primary lesson-example-run"
        onClick={onRun}
      >
        {example.label} ▶
      </button>
    </figure>
  );
}
