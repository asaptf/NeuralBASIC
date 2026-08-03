"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getDataset } from "@/engine";
import { createModelProbe } from "@/engine/probe";
import type { Dataset } from "@/engine/types";
import { PredictionReadout } from "@/components/lab/PredictionReadout";
import { useAppStore } from "@/store/useAppStore";

/**
 * The Data Lab's view for image datasets.
 *
 * On an image dataset the scatter plot has nothing to draw, so the largest panel
 * on screen used to go blank with a "not a 2-D dataset" notice — exactly during
 * the chapter about convolution. This shows what actually matters there instead:
 * the inputs themselves, the kernels the network learned to slide over them, and
 * a grid the learner can draw on to test the thing they trained.
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

/** Label a sample carries, whichever output shape the dataset uses. */
function labelOf(s: { y: number[] }): number {
  return s.y.length > 1
    ? s.y.indexOf(Math.max(...s.y))
    : s.y[0]! >= 0.5
      ? 1
      : 0;
}

/**
 * A canvas the learner paints an input on, one pixel at a time.
 *
 * This is where Chapter 4 stops being an assertion. Draw a bar in the middle and
 * the network is right; drag the same bar into a corner and a dense readout
 * falls apart while a convolutional one does not. Nothing in the app could show
 * that before, because every input on screen came out of the training set.
 */
function DrawableInput({
  h,
  w,
  ds,
}: {
  h: number;
  w: number;
  ds: Dataset;
}) {
  const network = useAppStore((s) => s.network);
  const trainConfig = useAppStore((s) => s.trainConfig);
  const weights = useAppStore((s) => s.weights);

  const probe = useMemo(
    () => createModelProbe(network, trainConfig, weights),
    [network, trainConfig, weights]
  );

  const [pixels, setPixels] = useState<number[]>(() =>
    Array.from({ length: h * w }, () => 0)
  );
  const [exampleIndex, setExampleIndex] = useState(0);
  const gridRef = useRef<HTMLDivElement>(null);
  /** Value the current drag is painting — null when no drag is in progress. */
  const painting = useRef<number | null>(null);

  // A drawing sized for the previous dataset is not an input for this one.
  useEffect(() => {
    setPixels(Array.from({ length: h * w }, () => 0));
  }, [h, w, ds.name]);

  /** Every labelled input, keyed by its pixels — built once per dataset. */
  const labelled = useMemo(() => {
    const byPixels = new Map<string, number>();
    for (const s of ds.samples) byPixels.set(s.x.join(","), labelOf(s));
    return byPixels;
  }, [ds]);

  /**
   * The dataset's own answer, when the drawing happens to *be* one of its
   * samples. Drawing something new leaves this null, which is the honest
   * report: nobody labelled it, so there is only the model's opinion.
   */
  const truth = useMemo(
    () => labelled.get(pixels.join(",")) ?? null,
    [pixels, labelled]
  );

  const result = useMemo(
    () => (probe ? probe.run(pixels) : null),
    [probe, pixels]
  );

  const setPixel = useCallback((index: number, value: number) => {
    setPixels((prev) => {
      if (prev[index] === value) return prev;
      const next = prev.slice();
      next[index] = value;
      return next;
    });
  }, []);

  /** Which cell a pointer is over, from the grid's geometry. */
  const cellAt = useCallback(
    (clientX: number, clientY: number): number | null => {
      const el = gridRef.current;
      if (!el) return null;
      const rect = el.getBoundingClientRect();
      const col = Math.floor(((clientX - rect.left) / rect.width) * w);
      const row = Math.floor(((clientY - rect.top) / rect.height) * h);
      if (col < 0 || col >= w || row < 0 || row >= h) return null;
      return row * w + col;
    },
    [h, w]
  );

  const isOn = (v: number | undefined) => (v ?? 0) >= 0.5;

  return (
    <div className="draw-lab">
      <div className="draw-lab-row">
        {/* Painting is handled on the container, not the cells: capturing the
            pointer here is what lets a single drag sweep across many pixels,
            on touch as well as with a mouse. The cells stay buttons so the grid
            is still reachable from the keyboard. */}
        <div
          ref={gridRef}
          className="draw-grid"
          style={{ gridTemplateColumns: `repeat(${w}, 1fr)` }}
          data-testid="draw-grid"
          onPointerDown={(e) => {
            const i = cellAt(e.clientX, e.clientY);
            if (i == null) return;
            const value = isOn(pixels[i]) ? 0 : 1;
            painting.current = value;
            setPixel(i, value);
            // Capture is what lets one drag sweep the whole grid, but it is an
            // improvement on the interaction, not a precondition for it: a
            // pointer the browser no longer considers active throws here, and
            // that must not cost the learner the pixel they just painted.
            try {
              e.currentTarget.setPointerCapture(e.pointerId);
            } catch {
              /* no capture — per-cell painting still works */
            }
          }}
          onPointerMove={(e) => {
            if (painting.current == null) return;
            const i = cellAt(e.clientX, e.clientY);
            if (i != null) setPixel(i, painting.current);
          }}
          onPointerUp={() => {
            painting.current = null;
          }}
          onPointerCancel={() => {
            painting.current = null;
          }}
        >
          {pixels.map((v, i) => (
            <button
              key={i}
              type="button"
              className={`draw-cell${isOn(v) ? " is-on" : ""}`}
              aria-pressed={isOn(v)}
              aria-label={`pixel row ${Math.floor(i / w) + 1}, column ${(i % w) + 1}`}
              // detail === 0 means the click came from the keyboard, so the
              // pointer path above has not already handled this cell.
              onClick={(e) => {
                if (e.detail === 0) setPixel(i, isOn(v) ? 0 : 1);
              }}
            />
          ))}
        </div>

        <div className="draw-lab-side">
          <div className="draw-lab-buttons">
            <button
              type="button"
              className="btn btn-ghost btn-tiny"
              data-testid="btn-draw-clear"
              onClick={() => setPixels(Array.from({ length: h * w }, () => 0))}
            >
              Clear
            </button>
            <button
              type="button"
              className="btn btn-ghost btn-tiny"
              data-testid="btn-draw-example"
              onClick={() => {
                const sample = ds.samples[exampleIndex % ds.samples.length];
                if (sample) setPixels(sample.x.slice(0, h * w));
                setExampleIndex((n) => n + 1);
              }}
            >
              Load an example
            </button>
          </div>
          {result ? (
            <PredictionReadout
              result={result}
              trueClass={truth}
              caption={truth == null ? "your drawing" : "from the training set"}
              testId="draw-probe-readout"
            />
          ) : weights.length ? (
            <p className="image-lab-note">
              This run&rsquo;s weights don&rsquo;t fit the network in the editor,
              so there is nothing to test. Check the loss for NaN, then train
              again.
            </p>
          ) : (
            <p className="image-lab-note">
              Train the network first — then whatever you draw here gets an
              answer.
            </p>
          )}
        </div>
      </div>

      <p className="image-lab-note">
        Draw a shape, then rub it out and draw the same shape somewhere else. A
        readout wired to fixed positions scores <em>where</em> the pixels are;
        one built on convolution scores <em>what</em> they make.
      </p>
    </div>
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
    const label = labelOf(s);
    if (!perClass.has(label)) perClass.set(label, s.x);
  }

  return (
    <div className="image-lab" data-testid="image-lab">
      <section>
        <h3 className="image-lab-heading">Test it · draw an input</h3>
        <DrawableInput h={shape.h} w={shape.w} ds={ds} />
      </section>

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
