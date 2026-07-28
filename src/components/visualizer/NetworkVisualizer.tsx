"use client";

import { useEffect, useRef } from "react";
import type { LayerConfig } from "@/engine/types";
import { PanelState } from "@/components/ui/PanelState";
import { useAppStore } from "@/store/useAppStore";

/**
 * The network itself: neurons glowing with their activation, edges coloured and
 * weighted by their sign and magnitude. The decision boundary lives in the Data
 * Lab panel at full size — duplicating it here as a thumbnail helped nobody.
 */

function layerLabel(l: LayerConfig): string {
  switch (l.type) {
    case "dense":
      return `dense ${l.units}${l.activation ? ` ${l.activation}` : ""}`;
    case "conv2d":
      return `conv ${l.filters}×${l.kernelSize}`;
    case "flatten":
      return "flatten";
    case "attention":
      return `attn d${l.dModel}`;
    case "transformer_block":
      return `transformer d${l.dModel}`;
  }
}

export function NetworkVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const snapshot = useAppStore((s) => s.lastSnapshot);
  const network = useAppStore((s) => s.network);
  const theme = useAppStore((s) => s.theme);
  const isTraining = useAppStore((s) => s.isTraining);
  const hasSnapshot = !!snapshot?.layerSnapshots?.length;

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

      const isRetro = theme === "retro";
      const posColor = isRetro ? "85,255,85" : "0,255,200";
      const negColor = isRetro ? "255,85,85" : "255,107,157";
      const textColor = isRetro ? "#ffffff" : "#cfe8e0";

      ctx.fillStyle = isRetro ? "#0000aa" : "#0a0e17";
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = isRetro
        ? "rgba(85,255,255,0.10)"
        : "rgba(0,255,200,0.05)";
      ctx.lineWidth = 1;
      for (let x = 0; x < w; x += 24) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let y = 0; y < h; y += 24) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      // Structure comes from the DSL and is real even before training; weights
      // and activations do not exist yet. Rendering placeholder 0.2 weights and
      // 0.3 activations made an untrained net look like a weakly trained one,
      // so an untrained net is drawn deliberately inert instead.
      const trained = !!snapshot?.layerSnapshots?.length;

      const layers = network.layers;

      // Column sizes: input column, then one per layer (capped for legibility).
      const layerSizes: number[] = [];
      const labels: string[] = [];

      const first = layers[0];
      if (first?.type === "dense" && first.inputDim) {
        layerSizes.push(Math.min(first.inputDim, 12));
        labels.push(`input ${first.inputDim}`);
      } else if (first?.type === "conv2d") {
        layerSizes.push(4);
        labels.push("input 4×4");
      } else if (
        first?.type === "attention" ||
        first?.type === "transformer_block"
      ) {
        layerSizes.push(Math.min(first.dModel, 8));
        labels.push(`input d${first.dModel}`);
      } else {
        layerSizes.push(2);
        labels.push("input");
      }

      for (const l of layers) {
        if (l.type === "dense") layerSizes.push(Math.min(l.units, 12));
        else if (l.type === "conv2d") layerSizes.push(Math.min(l.filters, 8));
        else if (l.type === "flatten")
          layerSizes.push(layerSizes[layerSizes.length - 1] ?? 4);
        else layerSizes.push(Math.min(l.dModel, 8));
        labels.push(layerLabel(l));
      }

      const nLayers = layerSizes.length;
      const marginX = 34;
      const marginTop = 30;
      const marginBottom = 26;
      const usableW = w - marginX * 2;
      const usableH = Math.max(20, h - marginTop - marginBottom);

      const positions: { x: number; y: number; act: number }[][] = [];
      for (let li = 0; li < nLayers; li++) {
        const count = layerSizes[li]!;
        const x =
          marginX +
          (nLayers === 1 ? usableW / 2 : (usableW * li) / (nLayers - 1));
        const col: { x: number; y: number; act: number }[] = [];
        // Keep dense columns from overflowing tall panels.
        const spacing = Math.min(usableH / Math.max(count, 1), 46);
        const colH = spacing * Math.max(count - 1, 0);
        const top = marginTop + (usableH - colH) / 2;
        for (let n = 0; n < count; n++) {
          const y = count === 1 ? marginTop + usableH / 2 : top + spacing * n;
          let act = 0;
          const snapIdx = li - 1;
          const acts = snapshot?.layerSnapshots?.[snapIdx]?.activations;
          if (acts && acts.length) act = acts[n] ?? acts[n % acts.length]!;
          col.push({ x, y, act });
        }
        positions.push(col);
      }

      // edges
      for (let li = 0; li < positions.length - 1; li++) {
        const a = positions[li]!;
        const b = positions[li + 1]!;
        const snap = snapshot?.layerSnapshots?.[li];
        for (let i = 0; i < a.length; i++) {
          for (let j = 0; j < b.length; j++) {
            const weight = snap?.weights?.[j]?.[i];
            ctx.beginPath();
            ctx.moveTo(a[i]!.x, a[i]!.y);
            ctx.lineTo(b[j]!.x, b[j]!.y);
            if (!trained || weight == null) {
              // No sign and no magnitude to show — just that a connection exists.
              ctx.strokeStyle = isRetro
                ? "rgba(85,255,255,0.22)"
                : "rgba(255,255,255,0.13)";
              ctx.lineWidth = 1;
            } else {
              const mag = Math.min(1, Math.abs(weight) * 2);
              ctx.strokeStyle = `rgba(${weight >= 0 ? posColor : negColor},${
                0.08 + mag * 0.6
              })`;
              ctx.lineWidth = 0.5 + mag * 2.5;
            }
            ctx.stroke();
          }
        }
      }

      // neurons
      const radius = Math.max(5, Math.min(9, usableH / 24));
      for (const col of positions) {
        for (const node of col) {
          const glow = trained
            ? Math.max(0, Math.min(1, Math.abs(node.act)))
            : 0;
          const r = radius + glow * 4;

          if (!isRetro && trained) {
            const g = ctx.createRadialGradient(
              node.x,
              node.y,
              0,
              node.x,
              node.y,
              r * 3
            );
            g.addColorStop(0, `rgba(0,255,200,${0.3 + glow * 0.45})`);
            g.addColorStop(1, "rgba(0,255,200,0)");
            ctx.fillStyle = g;
            ctx.beginPath();
            ctx.arc(node.x, node.y, r * 3, 0, Math.PI * 2);
            ctx.fill();
          }

          ctx.beginPath();
          ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
          if (!trained) {
            // Outline only: the unit exists, its activation doesn't yet.
            ctx.fillStyle = isRetro ? "#000088" : "#0d1522";
            ctx.strokeStyle = isRetro
              ? "rgba(85,255,255,0.5)"
              : "rgba(255,255,255,0.28)";
          } else if (isRetro) {
            ctx.fillStyle = glow > 0.5 ? "#ffff55" : "#55ffff";
            ctx.strokeStyle = "#ffffff";
          } else {
            ctx.fillStyle = `rgba(${30 + glow * 40}, ${80 + glow * 175}, ${
              120 + glow * 100
            }, 0.95)`;
            ctx.strokeStyle = "rgba(255,255,255,0.35)";
          }
          ctx.fill();
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }

      // layer labels, clamped so the outermost columns don't get clipped
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillStyle = textColor;
      ctx.globalAlpha = 0.75;
      ctx.textAlign = "center";
      for (let li = 0; li < labels.length; li++) {
        const label = labels[li]!;
        const half = ctx.measureText(label).width / 2 + 3;
        const x = positions[li]?.[0]?.x ?? marginX;
        ctx.fillText(label, Math.min(Math.max(x, half), w - half), 14);
      }
      ctx.textAlign = "left";
      ctx.globalAlpha = 1;

      // Weight-sign legend, abbreviated when the panel is too narrow for the
      // full wording rather than letting it run off the edge.
      ctx.font = "10px ui-monospace, monospace";
      ctx.globalAlpha = trained ? 0.8 : 0;
      const longLegend = w >= 270;
      const posLabel = longLegend ? "— positive weight" : "— positive";
      const negLabel = longLegend ? "— negative weight" : "— negative";
      ctx.fillStyle = `rgb(${posColor})`;
      ctx.fillText(posLabel, 10, h - 10);
      ctx.fillStyle = `rgb(${negColor})`;
      ctx.fillText(negLabel, 10 + ctx.measureText(posLabel).width + 12, h - 10);
      ctx.globalAlpha = 1;

      // attention heatmap (belongs to the network, not the data plane)
      const attn = snapshot?.attentionMaps?.[0];
      if (attn && attn.length) {
        const n = attn.length;
        const aw = Math.min(96, h - 60);
        if (aw > 24) {
          const cell = aw / n;
          const ax = w - aw - 12;
          const ay = h - aw - 22;
          ctx.fillStyle = isRetro ? "#000055" : "rgba(0,0,0,0.6)";
          ctx.fillRect(ax - 4, ay - 15, aw + 8, aw + 22);
          ctx.fillStyle = isRetro ? "#ffff55" : "#f0c87a";
          ctx.fillText("attention", ax, ay - 4);
          for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
              const v = attn[i]![j] ?? 0;
              const g = Math.round(v * 255);
              ctx.fillStyle = isRetro
                ? `rgb(${g},${g},0)`
                : `rgba(255, ${100 + g / 2}, 50, ${0.25 + v * 0.75})`;
              ctx.fillRect(ax + j * cell, ay + i * cell, cell, cell);
            }
          }
          ctx.strokeStyle = isRetro ? "#ffffff" : "rgba(255,255,255,0.3)";
          ctx.strokeRect(ax, ay, aw, aw);
        }
      }

      // No "TRAINING" stamp here — it collided with the layer labels, and the
      // transport bar above already reports epoch progress.
    };

    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [snapshot, network, theme, isTraining]);

  return (
    <div className="panel flex h-full min-h-0 flex-col" data-testid="visualizer-panel">
      <div className="panel-header">
        <span>Network</span>
        <span className="panel-header-note">
          {isTraining ? "updating…" : "activations · weights"}
        </span>
      </div>
      <div ref={wrapRef} className="panel-body">
        <canvas ref={canvasRef} className="block" />
        {!hasSnapshot && (
          <PanelState
            title="Not trained yet"
            hint="The architecture comes from your DSL. Weights and activations appear once you train."
            testId="network-state"
          />
        )}
      </div>
    </div>
  );
}
