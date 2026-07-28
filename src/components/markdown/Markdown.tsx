"use client";

import { Fragment, type ReactNode } from "react";

/**
 * Minimal markdown renderer for curriculum theory and tutor replies.
 *
 * Supports exactly what the content uses: **bold**, *italic*, `code`,
 * paragraph breaks and single line breaks. No HTML is injected — every node
 * is a real React element, so untrusted tutor text can never inject markup.
 */

const INLINE = /(\*\*[^*]+\*\*|\*[^*\n]+\*|`[^`]+`)/g;

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const parts = text.split(INLINE).filter((p) => p !== "");
  return parts.map((part, i) => {
    const key = `${keyPrefix}-${i}`;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return (
        <strong key={key} className="font-semibold text-[var(--accent)]">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return (
        <code key={key} className="md-code">
          {part.slice(1, -1)}
        </code>
      );
    }
    if (part.startsWith("*") && part.endsWith("*") && part.length > 2) {
      return (
        <em key={key} className="italic opacity-95">
          {part.slice(1, -1)}
        </em>
      );
    }
    return <Fragment key={key}>{part}</Fragment>;
  });
}

/** A line that is only a formula (no prose) gets centred monospace treatment. */
function isFormulaLine(line: string): boolean {
  const t = line.trim();
  if (t.length === 0 || t.length > 60) return false;
  return /^[a-zA-Zσσ₀-₉\s()=+\-*/^·×÷.,₁₂₃ᵢⱼwxyzbWXYZ]+$/.test(t) && /=/.test(t);
}

export function Markdown({
  text,
  className = "",
}: {
  text: string;
  className?: string;
}) {
  const blocks = text.trim().split(/\n{2,}/);

  return (
    <div className={`md ${className}`}>
      {blocks.map((block, bi) => {
        const lines = block.split("\n");

        // Bullet list block
        if (lines.every((l) => /^\s*[-*·]\s+/.test(l))) {
          return (
            <ul key={bi} className="md-list">
              {lines.map((l, li) => (
                <li key={li}>
                  {renderInline(l.replace(/^\s*[-*·]\s+/, ""), `${bi}-${li}`)}
                </li>
              ))}
            </ul>
          );
        }

        if (lines.length === 1 && isFormulaLine(lines[0]!)) {
          return (
            <div key={bi} className="md-formula">
              {lines[0]!.trim()}
            </div>
          );
        }

        return (
          <p key={bi} className="md-p">
            {lines.map((line, li) => (
              <Fragment key={li}>
                {li > 0 && <br />}
                {renderInline(line, `${bi}-${li}`)}
              </Fragment>
            ))}
          </p>
        );
      })}
    </div>
  );
}
