/**
 * Inference probe — a small, stable surface for "test what you trained".
 *
 * The store already rebuilds a live model from saved weights for display
 * (`prepareNetworkConfig` → `cloneModelFromWeights`). This module walks the
 * same path so a probe and a snapshot can never disagree about which model
 * is loaded. Failures degrade to `null` rather than taking the app down.
 */

import { createModel, predict } from "./model";
import type { Model } from "./model";
import { cloneModelFromWeights, prepareNetworkConfig } from "./train";
import type {
  ActivationName,
  LayerWeights,
  NetworkConfig,
  TrainConfig,
} from "./types";

export interface ProbeResult {
  /** Raw model output vector, as the network produced it. */
  output: number[];
  /** Index of the class the model picks. */
  classIndex: number;
  /** Probability it assigns to that class, 0–1. */
  confidence: number;
  /** Probability of class 1 — the quantity the 0.5 decision threshold applies to. */
  p1: number;
}

export interface ModelProbe {
  /** Number of values an input vector must carry. */
  readonly inputSize: number;
  /** Number of classes the readout distinguishes. */
  readonly outputSize: number;
  run(x: number[]): ProbeResult;
}

/** Clamp a probability-like value into [0, 1]. */
function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0;
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}

/**
 * Coerce a caller vector to exactly `inputSize`: pad short, truncate long,
 * replace non-finite entries with 0. A UI mid-edit must never crash the panel.
 */
function coerceInput(x: number[], inputSize: number): number[] {
  const out = new Array<number>(inputSize);
  for (let i = 0; i < inputSize; i++) {
    const v = i < x.length ? x[i]! : 0;
    out[i] = Number.isFinite(v) ? v : 0;
  }
  return out;
}

/** True when `m` is a rows×cols matrix (every row the same length). */
function matrixShapeMatches(
  m: number[][] | undefined,
  rows: number,
  cols: number
): boolean {
  if (!m || m.length !== rows) return false;
  for (const row of m) {
    if (row.length !== cols) return false;
  }
  return true;
}

/**
 * Whether exported weights can actually be loaded into this model shape.
 * `loadWeights` is intentionally permissive (partial restore for viz); a probe
 * needs a stricter check so a mismatched save file yields `null` instead of
 * silent garbage predictions.
 */
function weightsMatchModel(model: Model, weights: LayerWeights[]): boolean {
  if (weights.length !== model.layers.length) return false;

  for (let i = 0; i < model.layers.length; i++) {
    const layer = model.layers[i]!;
    const w = weights[i]!;
    if (w.type !== layer.type) return false;

    if (layer.type === "dense") {
      if (!w.weights || !w.biases) return false;
      if (w.weights.length !== layer.weights.length) return false;
      if (w.biases.length !== layer.biases.length) return false;
      const inDim = layer.weights[0]?.length ?? 0;
      for (const row of w.weights) {
        if (row.length !== inDim) return false;
      }
    } else if (layer.type === "conv2d") {
      if (!w.weights || !w.biases) return false;
      if (w.weights.length !== layer.filters) return false;
      if (w.biases.length !== layer.filters) return false;
      const flatLen =
        layer.inChannels * layer.kernelSize * layer.kernelSize;
      for (const row of w.weights) {
        if (row.length !== flatLen) return false;
      }
    } else if (layer.type === "attention") {
      // Same strictness as dense/conv: names alone are not enough — an
      // imported experiment from a different d_model would otherwise load
      // and produce NaNs that look like 100% confidence on class 0.
      const d = layer.dModel;
      if (
        !matrixShapeMatches(w.params?.Wq, d, d) ||
        !matrixShapeMatches(w.params?.Wk, d, d) ||
        !matrixShapeMatches(w.params?.Wv, d, d) ||
        !matrixShapeMatches(w.params?.Wo, d, d)
      )
        return false;
    } else if (layer.type === "transformer_block") {
      const d = layer.attention.dModel;
      const dff = layer.b1.length;
      if (
        !matrixShapeMatches(w.params?.Wq, d, d) ||
        !matrixShapeMatches(w.params?.Wk, d, d) ||
        !matrixShapeMatches(w.params?.Wv, d, d) ||
        !matrixShapeMatches(w.params?.Wo, d, d) ||
        !matrixShapeMatches(w.params?.W1, dff, d) ||
        !matrixShapeMatches(w.params?.W2, d, dff)
      )
        return false;
      // exportWeights concatenates b1|b2; require the full vector so a
      // restore cannot silently zero the feed-forward biases.
      if (!w.biases || w.biases.length !== dff + d) return false;
    }
    // flatten / pool carry no parameters — type match is enough
  }
  return true;
}

function outputSizeOf(model: Model): number {
  for (let i = model.layers.length - 1; i >= 0; i--) {
    const layer = model.layers[i]!;
    if (layer.type === "dense") return layer.weights.length;
  }
  return 1;
}

/** Activation of the final dense readout, if the model has one. */
function lastDenseActivation(model: Model): ActivationName | undefined {
  for (let i = model.layers.length - 1; i >= 0; i--) {
    const layer = model.layers[i]!;
    if (layer.type === "dense") return layer.activation;
  }
  return undefined;
}

/**
 * Normalise a raw readout into class index / confidence / p1.
 *
 * Softmax heads produce a distribution, so confidence is a share of that
 * mass. Independent heads (sigmoid, etc.) do not — each unit is its own
 * score, and dividing by the sum fabricates certainty from mutual "no"
 * answers (e.g. [0.2, 0.1] is not 67% confidence).
 *
 * For independent multi-unit heads, `p1` is the unit-1 score itself (not a
 * normalised share): the UI draws it against a 0.5 threshold the same way
 * it does for a single sigmoid, so it must stay a number that threshold
 * still means something for.
 */
function interpretOutput(
  raw: number[],
  activation: ActivationName | undefined
): ProbeResult {
  const output = raw.slice();

  if (output.length <= 1) {
    const p1 = clamp01(output[0] ?? 0);
    const classIndex = p1 >= 0.5 ? 1 : 0;
    const confidence = classIndex === 1 ? p1 : 1 - p1;
    return { output, classIndex, confidence, p1 };
  }

  const n = output.length;

  // Softmax (or anything already a distribution): share-of-mass reading.
  if (activation === "softmax") {
    let sum = 0;
    for (const v of output) sum += Number.isFinite(v) ? Math.max(0, v) : 0;

    let probs: number[];
    if (sum < 1e-12) {
      // Dead readout — fall back to uniform rather than divide by zero.
      probs = Array.from({ length: n }, () => 1 / n);
    } else {
      probs = output.map((v) => {
        const x = Number.isFinite(v) ? Math.max(0, v) : 0;
        return x / sum;
      });
    }

    let classIndex = 0;
    for (let i = 1; i < n; i++) {
      if (probs[i]! > probs[classIndex]!) classIndex = i;
    }

    return {
      output,
      classIndex,
      confidence: probs[classIndex]!,
      p1: probs[1] ?? 0,
    };
  }

  // Independent multi-unit head: pick argmax, confidence = that unit's own
  // score, p1 = unit 1's score (thresholdable at 0.5; see comment above).
  let classIndex = 0;
  for (let i = 1; i < n; i++) {
    const a = Number.isFinite(output[i]!) ? output[i]! : -Infinity;
    const b = Number.isFinite(output[classIndex]!)
      ? output[classIndex]!
      : -Infinity;
    if (a > b) classIndex = i;
  }

  return {
    output,
    classIndex,
    confidence: clamp01(output[classIndex] ?? 0),
    p1: clamp01(output[1] ?? 0),
  };
}

/**
 * Build a live inference probe from stored weights.
 * Returns `null` when the weights are empty or incompatible with the config —
 * the same degradation rationale as the store's `snapshotFromWeights`.
 */
export function createModelProbe(
  network: NetworkConfig,
  trainConfig: TrainConfig,
  weights: LayerWeights[]
): ModelProbe | null {
  if (!weights.length) return null;

  try {
    const prepared = prepareNetworkConfig(network, trainConfig.dataset);
    const shapeModel = createModel(prepared.config, prepared.inputDim);
    if (!weightsMatchModel(shapeModel, weights)) return null;

    const model = cloneModelFromWeights(
      prepared.config,
      weights,
      prepared.inputDim
    );
    const inputSize = prepared.inputDim;
    const outputSize = outputSizeOf(model);
    const headActivation = lastDenseActivation(model);

    return {
      inputSize,
      outputSize,
      run(x: number[]): ProbeResult {
        const coerced = coerceInput(x, inputSize);
        const raw = predict(model, coerced);
        return interpretOutput(raw, headActivation);
      },
    };
  } catch {
    // A restored file can disagree with the current engine; degrade to
    // "you can't probe this" rather than taking the whole app down.
    return null;
  }
}
