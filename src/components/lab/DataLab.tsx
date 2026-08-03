"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getDataset } from "@/engine";
import { createModelProbe, type ProbeResult } from "@/engine/probe";
import type { Sample, TrainStepResult } from "@/engine/types";
import { ImageLab } from "@/components/lab/ImageLab";
import { PredictionReadout } from "@/components/lab/PredictionReadout";
import { PanelState } from "@/components/ui/PanelState";
import { useAppStore } from "@/store/useAppStore";

/**
 * The primary teaching surface: the dataset itself, with the model's decision
 * boundary underneath it. Seeing *which* points the network gets wrong is the
 * whole point — a boundary without the data on top teaches nothing.
 *
 * It is also where the learner tests the thing they trained. Everything else on
 * screen is the model's opinion of the training set; tapping the plot hands it a
 * point that was never in the data and reads the answer back.
 */

interface Palette {
  class0: string;
  class1: string;
  wrong: string;
  axis: string;
  text: string;
  bg: string;
}

/** Literals are the fallback only — the CSS variables below are the truth. */
const FALLBACK: Record<string, Palette> = {
  retro: {
    class0: "#ff5555",
    class1: "#55ff55",
    wrong: "#ffff55",
    axis: "#55ffff",
    text: "#ffffff",
    bg: "#000088",
  },
  modern: {
    class0: "#ff6b9d",
    class1: "#00ffc8",
    wrong: "#ffd166",
    axis: "rgba(255,255,255,0.22)",
    text: "#cfe8e0",
    bg: "#0a0e17",
  },
};

/**
 * Read the plot colours off the theme's CSS variables.
 *
 * The canvas and the prediction readout below it colour the same two classes,
 * and a canvas that keeps its own copy of the palette drifts from the stylesheet
 * the moment either is edited. Falls back to literals when there is no computed
 * style to read (server render, tests).
 */
function paletteFor(theme: string, el: Element | null): Palette {
  const fallback = FALLBACK[theme] ?? FALLBACK.modern!;
  if (!el || typeof getComputedStyle !== "function") return fallback;
  const cs = getComputedStyle(el);
  const read = (name: string, fb: string) =>
    cs.getPropertyValue(name).trim() || fb;
  return {
    class0: read("--class-0", fallback.class0),
    class1: read("--class-1", fallback.class1),
    wrong: read("--class-wrong", fallback.wrong),
    axis: read("--plot-axis", fallback.axis),
    text: read("--plot-text", fallback.text),
    bg: read("--plot-bg", fallback.bg),
  };
}

/** A point the learner handed the model, and what came back. */
interface ProbePoint {
  x: number;
  y: number;
  result: ProbeResult;
}

/** Mapping between the plot's pixels and feature space, as last drawn. */
interface PlotView {
  padL: number;
  padT: number;
  plotW: number;
  plotH: number;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

function is2d(samples: Sample[]): boolean {
  return samples.length > 0 && samples[0]!.x.length === 2;
}

/**
 * Which class the model assigns at a point, read off the boundary grid it
 * already computed. Shared by the canvas and the text summary so the two can't
 * disagree.
 */
function predictedAt(
  grid: NonNullable<TrainStepResult["decisionGrid"]>,
  x: number,
  y: number
): number {
  const rangeX = grid.xMax - grid.xMin || 1;
  const rangeY = grid.yMax - grid.yMin || 1;
  const gi = Math.min(
    grid.width - 1,
    Math.max(0, Math.round(((x - grid.xMin) / rangeX) * (grid.width - 1)))
  );
  const gj = Math.min(
    grid.height - 1,
    Math.max(0, Math.round(((grid.yMax - y) / rangeY) * (grid.height - 1)))
  );
  return (grid.values[gj * grid.width + gi] ?? 0.5) >= 0.5 ? 1 : 0;
}

function countMisclassified(
  samples: Sample[],
  grid: NonNullable<TrainStepResult["decisionGrid"]>
): number {
  let wrong = 0;
  for (const s of samples) {
    const label = (s.y[0] ?? 0) >= 0.5 ? 1 : 0;
    if (predictedAt(grid, s.x[0]!, s.x[1]!) !== label) wrong++;
  }
  return wrong;
}

export function DataLab() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const snapshot = useAppStore((s) => s.lastSnapshot);
  const dataset = useAppStore((s) => s.trainConfig.dataset);
  const theme = useAppStore((s) => s.theme);
  const network = useAppStore((s) => s.network);
  const trainConfig = useAppStore((s) => s.trainConfig);
  const weights = useAppStore((s) => s.weights);

  // A live copy of the trained network, for inputs that were never in the data.
  // Null until a run finishes — the store empties `weights` when one starts and
  // fills them at the end — and null again if those weights describe no model
  // this config can be rebuilt into.
  const probe = useMemo(
    () => createModelProbe(network, trainConfig, weights),
    [network, trainConfig, weights]
  );

  const [probes, setProbes] = useState<ProbePoint[]>([]);
  const [hovered, setHovered] = useState<ProbePoint | null>(null);
  const viewRef = useRef<PlotView | null>(null);

  // Answers from a model that no longer exists are worse than no answers: they
  // sit on the plot looking current. Drop them whenever the model changes.
  useEffect(() => {
    setProbes([]);
    setHovered(null);
  }, [probe]);

  /** Pixel position within the canvas → the feature-space point it stands for. */
  const featureAt = useCallback(
    (clientX: number, clientY: number): { x: number; y: number } | null => {
      const view = viewRef.current;
      const canvas = canvasRef.current;
      if (!view || !canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const px = clientX - rect.left;
      const py = clientY - rect.top;
      // Outside the plot frame there is no point being asked about.
      if (
        px < view.padL ||
        px > view.padL + view.plotW ||
        py < view.padT ||
        py > view.padT + view.plotH
      ) {
        return null;
      }
      return {
        x: view.xMin + ((px - view.padL) / view.plotW) * (view.xMax - view.xMin),
        y:
          view.yMin +
          ((view.padT + view.plotH - py) / view.plotH) * (view.yMax - view.yMin),
      };
    },
    []
  );

  const askAt = useCallback(
    (clientX: number, clientY: number): ProbePoint | null => {
      if (!probe) return null;
      const at = featureAt(clientX, clientY);
      if (!at) return null;
      return { x: at.x, y: at.y, result: probe.run([at.x, at.y]) };
    },
    [probe, featureAt]
  );

  /**
   * The canvas is invisible to screen readers, and "how many points is it still
   * getting wrong" is the single most important number on this panel — so it
   * also exists as text.
   */
  const summary = useMemo(() => {
    const ds = getDataset(dataset);
    if (!is2d(ds.samples)) {
      if (ds.inputShape.length === 3) {
        const [, h, w] = ds.inputShape;
        const kernels = snapshot?.layerSnapshots?.find(
          (l) => l.type === "conv2d"
        )?.weights?.length;
        return kernels
          ? `${dataset}: ${h}×${w} images, showing ${kernels} learned ${kernels === 1 ? "kernel" : "kernels"}.`
          : `${dataset}: ${h}×${w} images. Train a conv2d layer to see its kernels.`;
      }
      return `${dataset}: not a 2-D dataset — see the network panel and metrics.`;
    }
    const total = ds.samples.length;
    const grid = snapshot?.decisionGrid;
    if (!grid) {
      return `${dataset}: ${total} points plotted. Not trained yet — press Train to draw the decision boundary.`;
    }
    const wrong = countMisclassified(ds.samples, grid);
    return wrong === 0
      ? `${dataset}: all ${total} points on the correct side of the boundary.`
      : `${dataset}: ${wrong} of ${total} points on the wrong side of the boundary.`;
  }, [dataset, snapshot]);

  const plottable = is2d(getDataset(dataset).samples);
  // Image datasets get their own view rather than an apology for having no plane.
  const isImageDataset = getDataset(dataset).inputShape.length === 3;

  /** Latest draw, so the resize observer can call it without being rebuilt. */
  const drawRef = useRef<() => void>(() => {});

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const draw = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const rect = wrap.getBoundingClientRect();
      const w = Math.max(1, rect.width);
      const h = Math.max(1, rect.height);
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const p = paletteFor(theme, wrap);
      ctx.fillStyle = p.bg;
      ctx.fillRect(0, 0, w, h);

      const ds = getDataset(dataset);

      // Non-2D datasets (images / text) have no meaningful scatter plane.
      // The explanation lives in the PanelState overlay, not in the bitmap.
      if (!is2d(ds.samples)) {
        viewRef.current = null;
        return;
      }

      // Plot frame: prefer the grid's own bounds so boundary and points align.
      const grid = snapshot?.decisionGrid;
      let xMin: number;
      let xMax: number;
      let yMin: number;
      let yMax: number;
      if (grid) {
        ({ xMin, xMax, yMin, yMax } = grid);
      } else {
        const xs = ds.samples.map((s) => s.x[0]!);
        const ys = ds.samples.map((s) => s.x[1]!);
        const padX = (Math.max(...xs) - Math.min(...xs) || 1) * 0.2;
        const padY = (Math.max(...ys) - Math.min(...ys) || 1) * 0.2;
        xMin = Math.min(...xs) - padX;
        xMax = Math.max(...xs) + padX;
        yMin = Math.min(...ys) - padY;
        yMax = Math.max(...ys) + padY;
      }

      // The two axes are both feature space, so the plot must stay square —
      // otherwise `circles` reads as ellipses and the geometry lies.
      // Equalise the two ranges as well, so one unit of x is one unit of y.
      const rangeX = xMax - xMin || 1;
      const rangeY = yMax - yMin || 1;
      const range = Math.max(rangeX, rangeY);
      const midX = (xMin + xMax) / 2;
      const midY = (yMin + yMax) / 2;
      xMin = midX - range / 2;
      xMax = midX + range / 2;
      yMin = midY - range / 2;
      yMax = midY + range / 2;

      const marginL = 34;
      const marginR = 12;
      const marginT = 12;
      const marginB = 40;
      const availW = Math.max(1, w - marginL - marginR);
      const availH = Math.max(1, h - marginT - marginB);
      const side = Math.max(1, Math.min(availW, availH));
      const plotW = side;
      const plotH = side;
      const padL = marginL + (availW - side) / 2;
      const padT = marginT + (availH - side) / 2;

      const sx = (x: number) => padL + ((x - xMin) / (xMax - xMin || 1)) * plotW;
      const sy = (y: number) =>
        padT + plotH - ((y - yMin) / (yMax - yMin || 1)) * plotH;

      // Publish the frame so a tap can be turned back into a feature-space
      // point. Inverting the mapping here rather than recomputing it in the
      // handler is what keeps the marker under the finger that placed it.
      viewRef.current = { padL, padT, plotW, plotH, xMin, xMax, yMin, yMax };

      // ── decision boundary ──
      // Drawn through sx/sy in the grid's OWN bounds: the displayed frame was
      // widened to square up the axes, so assuming the grid fills the plot
      // would slide the boundary off the points.
      if (grid) {
        const gx0 = sx(grid.xMin);
        const gx1 = sx(grid.xMax);
        const gy0 = sy(grid.yMax);
        const gy1 = sy(grid.yMin);
        const cw = (gx1 - gx0) / grid.width;
        const ch = (gy1 - gy0) / grid.height;
        for (let j = 0; j < grid.height; j++) {
          for (let i = 0; i < grid.width; i++) {
            const v = grid.values[j * grid.width + i] ?? 0.5;
            const conf = Math.min(1, Math.abs(v - 0.5) * 2);
            const color = v >= 0.5 ? p.class1 : p.class0;
            ctx.globalAlpha = 0.12 + conf * 0.4;
            ctx.fillStyle = color;
            ctx.fillRect(gx0 + i * cw, gy0 + j * ch, cw + 0.6, ch + 0.6);
          }
        }
        ctx.globalAlpha = 1;
      } else {
        ctx.fillStyle = "rgba(255,255,255,0.03)";
        ctx.fillRect(padL, padT, plotW, plotH);
      }

      // ── axes ──
      ctx.strokeStyle = p.axis;
      ctx.lineWidth = 1;
      ctx.strokeRect(padL, padT, plotW, plotH);

      if (xMin < 0 && xMax > 0) {
        ctx.beginPath();
        ctx.moveTo(sx(0), padT);
        ctx.lineTo(sx(0), padT + plotH);
        ctx.stroke();
      }
      if (yMin < 0 && yMax > 0) {
        ctx.beginPath();
        ctx.moveTo(padL, sy(0));
        ctx.lineTo(padL + plotW, sy(0));
        ctx.stroke();
      }

      ctx.fillStyle = p.text;
      ctx.globalAlpha = 0.7;
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillText(xMin.toFixed(1), padL, padT + plotH + 14);
      ctx.textAlign = "right";
      ctx.fillText(xMax.toFixed(1), padL + plotW, padT + plotH + 14);
      ctx.fillText(yMax.toFixed(1), padL - 5, padT + 9);
      ctx.fillText(yMin.toFixed(1), padL - 5, padT + plotH);
      ctx.textAlign = "left";
      ctx.globalAlpha = 1;

      // ── data points, with misclassified ones ringed ──
      let wrong = 0;
      for (const s of ds.samples) {
        const px = sx(s.x[0]!);
        const py = sy(s.x[1]!);
        const label = (s.y[0] ?? 0) >= 0.5 ? 1 : 0;

        // Read the model's opinion at this point off the grid it already computed.
        const predicted = grid ? predictedAt(grid, s.x[0]!, s.x[1]!) : null;
        const isWrong = predicted != null && predicted !== label;
        if (isWrong) wrong++;

        if (isWrong) {
          ctx.beginPath();
          ctx.arc(px, py, 8.5, 0, Math.PI * 2);
          ctx.strokeStyle = p.wrong;
          ctx.lineWidth = 2;
          ctx.stroke();
        }

        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fillStyle = label === 1 ? p.class1 : p.class0;
        ctx.fill();
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = theme === "retro" ? "#ffffff" : "rgba(0,0,0,0.55)";
        ctx.stroke();
      }

      // ── the learner's own test points ──
      // Diamonds, deliberately not circles: these are questions the learner
      // asked, not data the network was trained on, and the plot must not blur
      // the two. Opacity tracks confidence, so a marker sitting on the boundary
      // looks as undecided as the model is.
      const marker = (pt: ProbePoint, pinned: boolean) => {
        const px = sx(pt.x);
        const py = sy(pt.y);
        const color = pt.result.classIndex === 1 ? p.class1 : p.class0;
        const r = 7.5;

        ctx.beginPath();
        ctx.moveTo(px, py - r);
        ctx.lineTo(px + r, py);
        ctx.lineTo(px, py + r);
        ctx.lineTo(px - r, py);
        ctx.closePath();

        ctx.globalAlpha = 0.25 + pt.result.confidence * 0.55;
        ctx.fillStyle = color;
        ctx.fill();
        ctx.globalAlpha = 1;

        ctx.lineWidth = pinned ? 2 : 1.5;
        ctx.strokeStyle = color;
        if (!pinned) ctx.setLineDash([3, 2]);
        ctx.stroke();
        ctx.setLineDash([]);

        if (pinned) {
          ctx.fillStyle = p.text;
          ctx.globalAlpha = 0.85;
          ctx.font = "10px ui-monospace, monospace";
          ctx.fillText(pt.result.p1.toFixed(2), px + r + 3, py + 3.5);
          ctx.globalAlpha = 1;
        }
      };

      for (const pt of probes) marker(pt, true);
      if (hovered) marker(hovered, false);

      // ── legend ──
      const ly = h - 14;
      const chip = (x: number, color: string, label: string, ring = false) => {
        ctx.beginPath();
        ctx.arc(x, ly - 3, 4.5, 0, Math.PI * 2);
        if (ring) {
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.stroke();
        } else {
          ctx.fillStyle = color;
          ctx.fill();
        }
        ctx.fillStyle = p.text;
        ctx.globalAlpha = 0.8;
        ctx.font = "10px ui-monospace, monospace";
        ctx.fillText(label, x + 9, ly);
        ctx.globalAlpha = 1;
        return x + 9 + ctx.measureText(label).width + 16;
      };
      let lx = padL;
      lx = chip(lx, p.class0, "class 0");
      lx = chip(lx, p.class1, "class 1");
      if (grid) {
        chip(lx, p.wrong, `misclassified: ${wrong}/${ds.samples.length}`, true);
      }
    };

    drawRef.current = draw;
    draw();
  }, [snapshot, dataset, theme, probes, hovered]);

  // Kept out of the draw effect on purpose: hovering the plot redraws on every
  // pointer move, and rebuilding the observer at that rate would be pure waste.
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const ro = new ResizeObserver(() => drawRef.current());
    ro.observe(wrap);
    return () => ro.disconnect();
  }, []);

  return (
    <div className="panel flex h-full min-h-0 flex-col" data-testid="data-lab-panel">
      <div className="panel-header">
        <span>Data &amp; Decision Boundary</span>
        <span className="panel-header-note">{dataset}</span>
      </div>
      <div ref={wrapRef} className="panel-body">
        <canvas
          ref={canvasRef}
          className={`block${probe && plottable ? " is-probeable" : ""}`}
          role="img"
          aria-label={summary}
          // Hover previews the answer, a tap pins it. Touch fires both, which
          // is the behaviour we want: the finger lands and the marker stays.
          onPointerMove={
            probe ? (e) => setHovered(askAt(e.clientX, e.clientY)) : undefined
          }
          onPointerLeave={probe ? () => setHovered(null) : undefined}
          onPointerDown={
            probe
              ? (e) => {
                  const pt = askAt(e.clientX, e.clientY);
                  if (pt) setProbes((prev) => [...prev, pt]);
                }
              : undefined
          }
        />
        {!plottable ? (
          isImageDataset ? (
            <ImageLab />
          ) : (
            <PanelState
              title={`“${dataset}” has no 2-D plane to plot`}
              hint="Its inputs aren't two features, so there's no boundary to draw. Watch the network panel and metrics instead."
              testId="lab-state"
            />
          )
        ) : null}
        {/* No overlay for "plotted but not trained": the summary line below the
            canvas already says exactly that, and it covers nothing. An overlay
            here can only sit on top of the data — centred it hid the lower
            crescent of `moons`, and anchored to the bottom it hid XOR's bottom
            row along with the axis labels and the legend. The panel isn't
            empty, so it doesn't need an empty state. */}
      </div>
      <p
        className="lab-summary"
        role="status"
        data-testid="lab-summary"
      >
        {summary}
      </p>
      {plottable && (
        <TestStrip
          probe={probe != null}
          trained={weights.length > 0}
          shown={hovered ?? probes[probes.length - 1] ?? null}
          pinnedCount={probes.length}
          onClear={() => setProbes([])}
        />
      )}
    </div>
  );
}

/**
 * The "test it" strip under the plot.
 *
 * A trained network that is only ever shown answering its own training set
 * teaches the wrong lesson — that accuracy is a property of the model rather
 * than of the model *and* the inputs you try. This is where a learner finds out
 * what happens between the clusters, or outside them.
 */
function TestStrip({
  probe,
  trained,
  shown,
  pinnedCount,
  onClear,
}: {
  probe: boolean;
  /** Whether a finished run left weights behind at all. */
  trained: boolean;
  shown: ProbePoint | null;
  pinnedCount: number;
  onClear: () => void;
}) {
  if (!probe) {
    return (
      <div className="lab-teststrip" data-testid="lab-teststrip">
        <span className="lab-teststrip-hint">
          {trained
            ? // Weights exist but no model can be rebuilt from them. Saying
              // "train it first" here would be a lie the learner can disprove
              // by looking at the epoch counter.
              "This run's weights don't fit the network in the editor, so there is nothing to test. Check the loss for NaN, then train again."
            : "Train the network, then tap the plot to ask it about a point that was never in the data."}
        </span>
      </div>
    );
  }

  return (
    <div className="lab-teststrip" data-testid="lab-teststrip">
      <div className="lab-teststrip-head">
        <span className="lab-teststrip-title">Test it</span>
        {pinnedCount > 0 && (
          <button
            type="button"
            className="btn btn-ghost btn-tiny"
            data-testid="btn-clear-probes"
            onClick={onClear}
          >
            Clear {pinnedCount}
          </button>
        )}
      </div>
      {shown ? (
        <PredictionReadout
          result={shown.result}
          caption={`at (${shown.x.toFixed(2)}, ${shown.y.toFixed(2)})`}
          testId="lab-probe-readout"
        />
      ) : (
        <span className="lab-teststrip-hint">
          Move over the plot for a live answer; tap to pin one. Walking a line
          across the boundary is the fastest way to see where the network stops
          being sure.
        </span>
      )}
    </div>
  );
}
