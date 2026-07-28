"use client";

import dynamic from "next/dynamic";
import type { Monaco as MonacoApi } from "@monaco-editor/react";
import { useAppStore } from "@/store/useAppStore";

const Monaco = dynamic(() => import("@monaco-editor/react"), { ssr: false });

/**
 * NeuralBASIC DSL highlighting + themes that match the app shell. Without a
 * registered retro theme Monaco falls back to a white "vs" editor, which tears
 * a hole in the Retro Blue screen.
 */
function registerNeuralBasic(monaco: MonacoApi) {
  const existing = monaco.languages
    .getLanguages()
    .some((l: { id: string }) => l.id === "neuralbasic");

  if (!existing) {
    monaco.languages.register({ id: "neuralbasic" });
    monaco.languages.setMonarchTokensProvider("neuralbasic", {
      keywords: ["network", "train"],
      layers: [
        "dense",
        "conv2d",
        "flatten",
        "attention",
        "transformer",
        "transformer_block",
      ],
      tokenizer: {
        root: [
          [/#.*$/, "comment"],
          [/\b(network|train)\b/, "keyword"],
          [
            /\b(dense|conv2d|flatten|attention|transformer_block|transformer)\b/,
            "type",
          ],
          [
            /\b(activation|dataset|lr|epochs|l2|dropout|kernel|filters|heads|d_model|dff)\b(?==)/,
            "attribute.name",
          ],
          [/\b(sigmoid|relu|tanh|softmax|linear)\b/, "string"],
          [/->|=>/, "operator"],
          [/\d+(\.\d+)?/, "number"],
          [/[{}]/, "delimiter.bracket"],
        ],
      },
    });
    monaco.languages.setLanguageConfiguration("neuralbasic", {
      comments: { lineComment: "#" },
      brackets: [["{", "}"]],
      autoClosingPairs: [{ open: "{", close: "}" }],
    });
  }

  monaco.editor.defineTheme("neuralbasic-modern", {
    base: "vs-dark",
    inherit: true,
    rules: [
      { token: "keyword", foreground: "00ffc8", fontStyle: "bold" },
      { token: "type", foreground: "7c5cff", fontStyle: "bold" },
      { token: "attribute.name", foreground: "8aa89c" },
      { token: "string", foreground: "ffd166" },
      { token: "number", foreground: "ff6b9d" },
      { token: "operator", foreground: "00ffc8" },
      { token: "comment", foreground: "4a6d63", fontStyle: "italic" },
    ],
    colors: {
      "editor.background": "#0a0e17",
      "editor.foreground": "#e8f5f1",
      "editorLineNumber.foreground": "#37514a",
      "editorLineNumber.activeForeground": "#00ffc8",
      "editor.lineHighlightBackground": "#111c2a",
      "editorCursor.foreground": "#00ffc8",
      "editor.selectionBackground": "#00ffc833",
    },
  });

  monaco.editor.defineTheme("neuralbasic-retro", {
    base: "vs-dark",
    inherit: true,
    rules: [
      { token: "keyword", foreground: "ffff55", fontStyle: "bold" },
      { token: "type", foreground: "55ff55", fontStyle: "bold" },
      { token: "attribute.name", foreground: "55ffff" },
      { token: "string", foreground: "ffffff" },
      { token: "number", foreground: "ff5555" },
      { token: "operator", foreground: "ffff55" },
      { token: "comment", foreground: "8888ff", fontStyle: "italic" },
    ],
    colors: {
      "editor.background": "#0000aa",
      "editor.foreground": "#ffffff",
      "editorLineNumber.foreground": "#5555dd",
      "editorLineNumber.activeForeground": "#ffff55",
      "editor.lineHighlightBackground": "#0000cc",
      "editorCursor.foreground": "#ffff55",
      "editor.selectionBackground": "#5555ff88",
    },
  });
}

export function CodeEditor() {
  const dsl = useAppStore((s) => s.dsl);
  const setDsl = useAppStore((s) => s.setDsl);
  const theme = useAppStore((s) => s.theme);
  const parseError = useAppStore((s) => s.parseError);

  return (
    <div className="panel flex h-full min-h-0 flex-col" data-testid="editor-panel">
      <div className="panel-header">
        <span>DSL Editor</span>
        <span className="panel-header-note">NeuralBASIC</span>
      </div>
      <div className="min-h-0 flex-1">
        <Monaco
          height="100%"
          defaultLanguage="neuralbasic"
          language="neuralbasic"
          beforeMount={registerNeuralBasic}
          theme={theme === "retro" ? "neuralbasic-retro" : "neuralbasic-modern"}
          value={dsl}
          onChange={(v) => setDsl(v ?? "")}
          options={{
            fontSize: 14,
            fontFamily:
              theme === "retro"
                ? "'IBM Plex Mono', 'Courier New', monospace"
                : "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
            minimap: { enabled: false },
            wordWrap: "on",
            lineNumbers: "on",
            scrollBeyondLastLine: false,
            automaticLayout: true,
            tabSize: 2,
            padding: { top: 10 },
            overviewRulerLanes: 0,
            renderLineHighlight: "line",
            scrollbar: { verticalScrollbarSize: 8, horizontalScrollbarSize: 8 },
          }}
        />
      </div>
      {parseError && (
        <div className="parse-error" role="alert" data-testid="parse-error">
          {parseError}
        </div>
      )}
    </div>
  );
}
