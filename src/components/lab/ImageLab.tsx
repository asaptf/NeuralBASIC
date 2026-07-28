"use client";

import { useMemo } from "react";
import { getDataset } from "@/engine";
import type { Dataset } from "@/engine/types";
import { useAppStore } from "@/store/useAppStore";

/**
 * The Data Lab's view for image datasets.
 *
 * On an image dataset the scatter plot has nothing to draw, so the largest panel
 * on screen used to go blank with a "not a 2-D dataset" notice — exactly during
 * the chapter about convolution. This shows what actually matters there instead:
 * the inputs themselves, and the kernels the network learned to slide over them.
 *
 * Kernels are the subject of Chapter 4 and were previously invisible, even
 * though the engine has been reporting them in `layerSnapshots` all along.
 */

function imageShape(ds: Dataset): { c: number; h: number; w: number } | null {
  if (ds.inputShape.length !== 3) return null;
  const [c, h, w] = ds.inputShape;
  if (!c || !h || !w) return null;
  return { c, h, w };
}

function Swatch({
  values,
  w,
  h,
  cell = 14,
  signed = false,
}: {
  values: number[];
  w: number;
  h: number;
  cell?: number;
  signed?: boolean;
}) {
  // Signed grids (kernels) need a symmetric scale so 0 reads as neutral;
  // unsigned grids (pixels) just span their own range.
  const max = Math.max(1e-6, ...values.map((v) => Math.abs(v)));
  return (
    <svg
      width={w * cell}
      height={h * cell}
      className="swatch"
      role="img"
      aria-hidden="true"
    >
      {values.slice(0, w * h).map((v, i) => {
        const x = (i % w) * cell;
        const y = Math.floor(i / w) * cell;
        let fill: string;
        if (signed) {
          const t = Math.min(1, Math.abs(v) / max);
          fill =
            v >= 0
              ? `color-mix(in srgb, var(--good) ${Math.round(t * 100)}%, transparent)`
              : `color-mix(in srgb, var(--bad) ${Math.round(t * 100)}%, transparent)`;
        } else {
          const t = Math.min(1, Math.max(0, v));
          fill = `color-mix(in srgb, var(--text) ${Math.round(t * 92)}%, transparent)`;
        }
        return <rect key={i} x={x} y={y} width={cell} height={cell} fill={fill} />;
      })}
    </svg>
  );
}

export function ImageLab() {
  const dataset = useAppStore((s) => s.trainConfig.dataset);
  const snapshot = useAppStore((s) => s.lastSnapshot);

  const ds = useMemo(() => getDataset(dataset), [dataset]);
  const shape = imageShape(ds);

  const kernels = useMemo(() => {
    const conv = snapshot?.layerSnapshots?.find((l) => l.type === "conv2d");
    if (!conv?.weights || !conv.shape) return null;
    const [filters, channels, k] = conv.shape;
    if (!filters || !channels || !k) return null;
    // weights is [filter][channels * k * k]; show the first channel of each.
    return {
      k,
      list: conv.weights.map((flat) => flat.slice(0, k * k)),
      filters,
    };
  }, [snapshot]);

  if (!shape) return null;

  // One clean example per class, so the two categories sit side by side.
  const perClass = new Map<number, number[]>();
  for (const s of ds.samples) {
    const label = s.y.length > 1 ? s.y.indexOf(Math.max(...s.y)) : s.y[0]! >= 0.5 ? 1 : 0;
    if (!perClass.has(label)) perClass.set(label, s.x);
  }

  return (
    <div className="image-lab" data-testid="image-lab">
      <section>
        <h3 className="image-lab-heading">Inputs · {shape.h}×{shape.w}</h3>
        <div className="swatch-row">
          {[...perClass.entries()].map(([label, x]) => (
            <figure key={label} className="swatch-fig">
              <Swatch values={x} w={shape.w} h={shape.h} cell={16} />
              <figcaption>class {label}</figcaption>
            </figure>
          ))}
        </div>
      </section>

      <section>
        <h3 className="image-lab-heading">
          Learned kernels
          {kernels ? ` · ${kernels.filters} × ${kernels.k}×${kernels.k}` : ""}
        </h3>
        {kernels ? (
          <>
            <div className="swatch-row">
              {kernels.list.map((k, i) => (
                <figure key={i} className="swatch-fig">
                  <Swatch values={k} w={kernels.k} h={kernels.k} cell={18} signed />
                  <figcaption>filter {i}</figcaption>
                </figure>
              ))}
            </div>
            <p className="image-lab-note">
              Each kernel slides over every position in the image, using the same
              weights everywhere. Green is positive, red negative.
            </p>
          </>
        ) : (
          <p className="image-lab-note">
            No convolution layer in this network yet — add a{" "}
            <code className="md-code">conv2d</code> layer and train to see its
            kernels.
          </p>
        )}
      </section>
    </div>
  );
}
