"use client";

import type { ProbeResult } from "@/engine/probe";

/**
 * What the network said about one input the learner handed it.
 *
 * Everywhere else in the app the model is only ever shown answering the
 * training set, so "is it right?" is a count of rings on a scatter plot. This
 * is the other question — *how* sure is it, and on which side of the line — and
 * a percentage alone does not answer it. The bar puts the answer in a place:
 * left of the threshold or right of it, near the edge or hard against the wall.
 */
export function PredictionReadout({
  result,
  trueClass,
  caption,
  testId,
}: {
  result: ProbeResult;
  /** The label the dataset would give this input, when that is known. */
  trueClass?: number | null;
  /** Where the input came from — a coordinate, "your drawing". */
  caption?: string;
  testId?: string;
}) {
  const { classIndex, confidence, p1 } = result;
  const pct = Math.round(confidence * 100);
  const correct = trueClass == null ? null : trueClass === classIndex;

  return (
    <div className="probe-readout" data-testid={testId}>
      <div className="probe-verdict">
        <span className={`probe-chip probe-chip-${classIndex}`}>
          class {classIndex}
        </span>
        <span className="probe-confidence" data-testid="probe-confidence">
          {pct}% sure
        </span>
        {correct != null && (
          <span className={`probe-truth ${correct ? "good" : "bad"}`}>
            {correct ? "matches the label" : `label says class ${trueClass}`}
          </span>
        )}
        {caption && <span className="probe-caption">{caption}</span>}
      </div>

      {/* The threshold is the whole story, so it is drawn, not described: the
          needle's distance from the centre line is the model's conviction. */}
      <div
        className="probe-bar"
        role="img"
        aria-label={`Predicts class ${classIndex} with ${pct}% confidence. Probability of class 1 is ${p1.toFixed(2)}, decision threshold 0.5.`}
      >
        <span className="probe-bar-threshold" />
        <span
          className={`probe-bar-needle probe-needle-${classIndex}`}
          style={{ left: `${Math.min(100, Math.max(0, p1 * 100))}%` }}
        />
      </div>
      <div className="probe-bar-scale" aria-hidden="true">
        <span>class 0</span>
        <span>0.5</span>
        <span>class 1</span>
      </div>
    </div>
  );
}
