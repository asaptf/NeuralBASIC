"use client";

import { useEffect, useRef, useState } from "react";
import { Markdown } from "@/components/markdown/Markdown";
import { useAppStore } from "@/store/useAppStore";

export function TutorPanel() {
  const messages = useAppStore((s) => s.tutorMessages);
  const send = useAppStore((s) => s.sendTutorMessage);
  const chapterId = useAppStore((s) => s.progress.currentChapterId);
  const [text, setText] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [messages.length]);

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;
    send(text.trim());
    setText("");
  };

  return (
    <div className="panel flex h-full min-h-0 flex-col" data-testid="tutor-panel">
      <div className="panel-header">
        <span>Socratic Tutor</span>
        <span className="panel-header-note">{chapterId}</span>
      </div>
      <div
        className="flex min-h-0 flex-1 flex-col gap-2 overflow-auto p-3"
        aria-live="polite"
      >
        {messages.map((m) => (
          <div
            key={m.id}
            className={
              m.role === "user"
                ? "chat-bubble chat-user"
                : "chat-bubble chat-tutor"
            }
            data-role={m.role}
          >
            <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider opacity-55">
              {m.role === "user" ? "You" : "Tutor"}
            </div>
            <Markdown text={m.content} />
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      <form onSubmit={onSubmit} className="flex gap-2 border-t border-[var(--border)] p-2">
        <input
          className="input min-w-0 flex-1"
          data-testid="tutor-input"
          aria-label="Ask the tutor a question"
          placeholder="Ask a question (I won't paste full solutions)…"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button type="submit" className="btn btn-primary" data-testid="tutor-send">
          Send
        </button>
      </form>
    </div>
  );
}
