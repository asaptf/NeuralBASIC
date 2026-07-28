"use client";

import { useEffect, useMemo, useRef } from "react";
import { getDataset } from "@/engine";
import type { Sample, TrainStepResult } from "@/engine/types";
import { PanelState } from "@/components/ui/PanelState";
import { useAppStore } from "@/store/useAppStore";

/**
 * The primary teaching surface: the dataset itself, with the model's decision
 * boundary underneath it. Seeing *which* points the network gets wrong is the
 * whole point — a boundary without the data on top teaches nothing.
 */

interface Palette {
  class0: string;
  class1: string;
  wrong: string;
  axis: string;
  text: string;
  bg: string;
}

function paletteFor(theme: string): Palette {
  if (theme === "retro") {
    return {
      class0: "#ff5555",
      class1: "#55ff55",
      wrong: "#ffff55",
      axis: "#55ffff",
      text: "#ffffff",
      bg: "#000088",
    };
  }
  return {
    class0: "#ff6b9d",
    class1: "#00ffc8",
    wrong: "#ffd166",
    axis: "rgba(255,255,255,0.22)",
    text: "#cfe8e0",
    bg: "#0a0e17",
  };
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

  /**
   * The canvas is invisible to screen readers, and "how many points is it still
   * getting wrong" is the single most important number on this panel — so it
   * also exists as text.
   */
  const summary = useMemo(() => {
    const ds = getDataset(dataset);
    if (!is2d(ds.samples)) {
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

      const p = paletteFor(theme);
      ctx.fillStyle = p.bg;
      ctx.fillRect(0, 0, w, h);

      const ds = getDataset(dataset);

      // Non-2D datasets (images / text) have no meaningful scatter plane.
      // The explanation lives in the PanelState overlay, not in the bitmap.
      if (!is2d(ds.samples)) return;

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

    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [snapshot, dataset, theme]);

  return (
    <div className="panel flex h-full min-h-0 flex-col" data-testid="data-lab-panel">
      <div className="panel-header">
        <span>Data &amp; Decision Boundary</span>
        <span className="panel-header-note">{dataset}</span>
      </div>
      <div ref={wrapRef} className="panel-body">
        <canvas
          ref={canvasRef}
          className="block"
          role="img"
          aria-label={summary}
        />
        {!plottable ? (
          <PanelState
            title={`“${dataset}” has no 2-D plane to plot`}
            hint="Its inputs aren't two features, so there's no boundary to draw. Watch the network panel and metrics instead."
            testId="lab-state"
          />
        ) : (
          !snapshot?.decisionGrid && (
            <PanelState
              title="Not trained yet"
              hint="The data is plotted. Press Train to draw the decision boundary over it."
              testId="lab-state"
            />
          )
        )}
      </div>
      <p
        className="lab-summary"
        role="status"
        data-testid="lab-summary"
      >
        {summary}
      </p>
    </div>
  );
}
