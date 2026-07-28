import { getDataset } from "./datasets";
import { binaryCrossEntropy, clip, mean, mse } from "./math";
import {
  createModel,
  exportWeights,
  forward,
  loadWeights,
  predict,
  snapshotLayers,
  type Conv2dLayerState,
  type DenseLayerState,
  type Model,
  type PoolLayerState,
} from "./model";
import {
  resolveValRatio,
  splitTrainVal,
  type DataSplit,
} from "./split";
import type {
  LayerWeights,
  NetworkConfig,
  Sample,
  TrainConfig,
  TrainHistory,
  TrainStepResult,
} from "./types";

function l2Penalty(model: Model, coeff: number): number {
  if (!coeff) return 0;
  let s = 0;
  for (const layer of model.layers) {
    if (layer.type === "dense") {
      for (const row of layer.weights) for (const w of row) s += w * w;
    } else if (layer.type === "conv2d") {
      for (const f of layer.kernels)
        for (const c of f) for (const row of c) for (const w of row) s += w * w;
    }
  }
  return coeff * s;
}

function flattenCHW(t: number[][][]): number[] {
  const out: number[] = [];
  for (const plane of t)
    for (const row of plane) for (const v of row) out.push(v);
  return out;
}

function reshapeCHW(
  flat: number[],
  c: number,
  h: number,
  w: number
): number[][][] {
  const out: number[][][] = [];
  let idx = 0;
  for (let ch = 0; ch < c; ch++) {
    const plane: number[][] = [];
    for (let y = 0; y < h; y++) {
      const row: number[] = [];
      for (let x = 0; x < w; x++) row.push(flat[idx++] ?? 0);
      plane.push(row);
    }
    out.push(plane);
  }
  return out;
}

/** Analytical dense layer backward + SGD update. Returns dL/dInput. */
function backwardDense(
  layer: DenseLayerState,
  deltaOut: number[],
  lr: number,
  l2: number,
  isOutputSigmoid: boolean
): number[] {
  if (!layer.lastInput || !layer.lastPre || !layer.lastOut) {
    return new Array(layer.weights[0]?.length ?? 0).fill(0);
  }

  const act = layer.activation;
  let gradPre: number[];
  if (isOutputSigmoid && act === "sigmoid") {
    // BCE + sigmoid: dL/dz = pred - target (already in deltaOut)
    gradPre = deltaOut;
  } else if (act === "sigmoid") {
    gradPre = deltaOut.map(
      (d, i) => d * layer.lastOut![i]! * (1 - layer.lastOut![i]!)
    );
  } else if (act === "relu") {
    gradPre = deltaOut.map((d, i) => (layer.lastPre![i]! > 0 ? d : 0));
  } else if (act === "tanh") {
    gradPre = deltaOut.map(
      (d, i) => d * (1 - layer.lastOut![i]! * layer.lastOut![i]!)
    );
  } else {
    gradPre = deltaOut.slice();
  }

  const input = layer.lastInput;
  const nextDelta = new Array(input.length).fill(0) as number[];

  for (let o = 0; o < layer.weights.length; o++) {
    const g = gradPre[o]!;
    layer.biases[o]! -= lr * g;
    for (let i = 0; i < input.length; i++) {
      const w = layer.weights[o]![i]!;
      const gw = g * input[i]! + 2 * l2 * w;
      layer.weights[o]![i]! = w - lr * gw;
      nextDelta[i]! += g * w;
    }
  }
  return nextDelta;
}

/** Analytical conv2d backward + SGD. deltaOut is dL/dActivatedOutput [F][H'][W']. */
function backwardConv(
  layer: Conv2dLayerState,
  deltaOut: number[][][],
  lr: number,
  l2: number
): number[][][] | null {
  if (!layer.lastInput || !layer.lastPre || !layer.lastOut) return null;

  const input = layer.lastInput;
  const inC = input.length;
  const h = input[0]!.length;
  const w = input[0]![0]!.length;
  const k = layer.kernelSize;
  const outH = layer.lastOut[0]!.length;
  const outW = layer.lastOut[0]![0]!.length;

  // dL/dPre
  const dPre: number[][][] = [];
  for (let f = 0; f < layer.filters; f++) {
    const plane: number[][] = [];
    for (let y = 0; y < outH; y++) {
      const row: number[] = [];
      for (let x = 0; x < outW; x++) {
        const dout = deltaOut[f]?.[y]?.[x] ?? 0;
        const pre = layer.lastPre[f]![y]![x]!;
        let factor = 1;
        if (layer.activation === "relu") factor = pre > 0 ? 1 : 0;
        else if (layer.activation === "sigmoid") {
          const a = layer.lastOut[f]![y]![x]!;
          factor = a * (1 - a);
        }
        row.push(dout * factor);
      }
      plane.push(row);
    }
    dPre.push(plane);
  }

  // dInput
  const dInput: number[][][] = [];
  for (let c = 0; c < inC; c++) {
    const plane = Array.from({ length: h }, () =>
      Array.from({ length: w }, () => 0)
    );
    dInput.push(plane);
  }

  // accumulate kernel grads then apply (use pre-update kernels for dInput)
  const dKernels: number[][][][] = layer.kernels.map((f) =>
    f.map((c) => c.map((row) => row.map(() => 0)))
  );
  const dBias = new Array(layer.filters).fill(0) as number[];

  for (let f = 0; f < layer.filters; f++) {
    for (let y = 0; y < outH; y++) {
      for (let x = 0; x < outW; x++) {
        const g = dPre[f]![y]![x]!;
        dBias[f]! += g;
        for (let c = 0; c < inC; c++) {
          for (let ky = 0; ky < k; ky++) {
            for (let kx = 0; kx < k; kx++) {
              const iv = input[c]![y + ky]![x + kx] ?? 0;
              const kw = layer.kernels[f]![c]![ky]![kx]!;
              dKernels[f]![c]![ky]![kx]! += g * iv;
              dInput[c]![y + ky]![x + kx]! += g * kw;
            }
          }
        }
      }
    }
  }

  for (let f = 0; f < layer.filters; f++) {
    layer.biases[f]! -= lr * dBias[f]!;
    for (let c = 0; c < inC; c++) {
      for (let ky = 0; ky < k; ky++) {
        for (let kx = 0; kx < k; kx++) {
          const kw = layer.kernels[f]![c]![ky]![kx]!;
          const gk = dKernels[f]![c]![ky]![kx]! + 2 * l2 * kw;
          layer.kernels[f]![c]![ky]![kx]! = kw - lr * gk;
        }
      }
    }
  }

  return dInput;
}

/**
 * Analytical pool backward. Max routes gradient to the argmax cell;
 * avg spreads it evenly over the window (clamped to input bounds).
 * No learnable parameters.
 */
function backwardPool(
  layer: PoolLayerState,
  deltaOut: number[][][]
): number[][][] | null {
  if (!layer.lastInput || !layer.lastOut) return null;

  const input = layer.lastInput;
  const c = input.length;
  const h = input[0]!.length;
  const w = input[0]![0]!.length;
  const outH = layer.lastOut[0]!.length;
  const outW = layer.lastOut[0]![0]!.length;
  const size = layer.global ? Math.max(h, w, 1) : layer.size;
  const stride = layer.global ? Math.max(h, w, 1) : layer.stride;

  const dInput: number[][][] = [];
  for (let ch = 0; ch < c; ch++) {
    dInput.push(
      Array.from({ length: h }, () => Array.from({ length: w }, () => 0))
    );
  }

  if (layer.mode === "max") {
    const maxY = layer.lastMaxY;
    const maxX = layer.lastMaxX;
    if (!maxY || !maxX) return dInput;
    for (let ch = 0; ch < c; ch++) {
      for (let oy = 0; oy < outH; oy++) {
        for (let ox = 0; ox < outW; ox++) {
          const g = deltaOut[ch]?.[oy]?.[ox] ?? 0;
          const iy = maxY[ch]![oy]![ox]!;
          const ix = maxX[ch]![oy]![ox]!;
          if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
            dInput[ch]![iy]![ix]! += g;
          }
        }
      }
    }
  } else {
    for (let ch = 0; ch < c; ch++) {
      for (let oy = 0; oy < outH; oy++) {
        for (let ox = 0; ox < outW; ox++) {
          const g = deltaOut[ch]?.[oy]?.[ox] ?? 0;
          const y0 = oy * stride;
          const x0 = ox * stride;
          const y1 = Math.min(h, y0 + size);
          const x1 = Math.min(w, x0 + size);
          const n = Math.max(1, (y1 - y0) * (x1 - x0));
          const share = g / n;
          for (let y = y0; y < y1; y++) {
            for (let x = x0; x < x1; x++) {
              dInput[ch]![y]![x]! += share;
            }
          }
        }
      }
    }
  }

  return dInput;
}

/**
 * Unified analytical step for dense and conv architectures.
 * Skips attention/transformer (returns false so caller can fall back).
 */
function trainStepAnalytical(
  model: Model,
  x: number[],
  y: number[],
  lr: number,
  l2: number
): number | null {
  const hasAttn = model.layers.some(
    (l) => l.type === "attention" || l.type === "transformer_block"
  );
  if (hasAttn) return null;

  const { output } = forward(model, x);
  const loss = binaryCrossEntropy(output, y) + l2Penalty(model, l2);

  // dL/doutput for BCE+sigmoid units
  let deltaFlat: number[] | null = output.map((p, i) => p - y[i]!);
  let deltaSpatial: number[][][] | null = null;

  for (let li = model.layers.length - 1; li >= 0; li--) {
    const layer = model.layers[li]!;

    if (layer.type === "dense") {
      if (!deltaFlat) {
        if (deltaSpatial) deltaFlat = flattenCHW(deltaSpatial);
        else break;
      }
      const isOut = li === model.layers.length - 1;
      deltaFlat = backwardDense(
        layer,
        deltaFlat,
        lr,
        l2,
        isOut && layer.activation === "sigmoid"
      );
      deltaSpatial = null;
    } else if (layer.type === "flatten") {
      if (deltaFlat && layer.lastShape) {
        const [c, h, w] = layer.lastShape;
        deltaSpatial = reshapeCHW(deltaFlat, c!, h!, w!);
        deltaFlat = null;
      }
    } else if (layer.type === "pool") {
      if (!deltaSpatial) {
        if (deltaFlat && layer.lastOut) {
          const c = layer.lastOut.length;
          const h = layer.lastOut[0]!.length;
          const w = layer.lastOut[0]![0]!.length;
          deltaSpatial = reshapeCHW(deltaFlat, c, h, w);
          deltaFlat = null;
        } else break;
      }
      const dIn = backwardPool(layer, deltaSpatial);
      deltaSpatial = dIn;
      deltaFlat = null;
    } else if (layer.type === "conv2d") {
      if (!deltaSpatial) {
        if (deltaFlat && layer.lastOut) {
          const f = layer.filters;
          const h = layer.lastOut[0]!.length;
          const w = layer.lastOut[0]![0]!.length;
          deltaSpatial = reshapeCHW(deltaFlat, f, h, w);
          deltaFlat = null;
        } else break;
      }
      const dIn = backwardConv(layer, deltaSpatial, lr, l2);
      deltaSpatial = dIn;
      deltaFlat = null;
    }
  }

  return loss;
}

type ParamRef = {
  get: () => number;
  set: (v: number) => void;
};

function collectParams(model: Model): ParamRef[] {
  const params: ParamRef[] = [];
  const addMat = (m: number[][]) => {
    for (let i = 0; i < m.length; i++) {
      for (let j = 0; j < m[i]!.length; j++) {
        const ii = i;
        const jj = j;
        params.push({
          get: () => m[ii]![jj]!,
          set: (v) => {
            m[ii]![jj] = v;
          },
        });
      }
    }
  };
  const addVec = (v: number[]) => {
    for (let i = 0; i < v.length; i++) {
      const ii = i;
      params.push({
        get: () => v[ii]!,
        set: (val) => {
          v[ii] = val;
        },
      });
    }
  };

  for (const layer of model.layers) {
    if (layer.type === "dense") {
      addMat(layer.weights);
      addVec(layer.biases);
    } else if (layer.type === "conv2d") {
      for (let f = 0; f < layer.filters; f++) {
        for (let c = 0; c < layer.inChannels; c++) {
          for (let ky = 0; ky < layer.kernelSize; ky++) {
            for (let kx = 0; kx < layer.kernelSize; kx++) {
              const ff = f,
                cc = c,
                yy = ky,
                xx = kx;
              params.push({
                get: () => layer.kernels[ff]![cc]![yy]![xx]!,
                set: (v) => {
                  layer.kernels[ff]![cc]![yy]![xx] = v;
                },
              });
            }
          }
        }
      }
      addVec(layer.biases);
    } else if (layer.type === "attention") {
      addMat(layer.Wq);
      addMat(layer.Wk);
      addMat(layer.Wv);
      addMat(layer.Wo);
    } else if (layer.type === "transformer_block") {
      addMat(layer.attention.Wq);
      addMat(layer.attention.Wk);
      addMat(layer.attention.Wv);
      addMat(layer.attention.Wo);
      addMat(layer.W1);
      addMat(layer.W2);
      addVec(layer.b1);
      addVec(layer.b2);
    }
  }
  return params;
}

/**
 * Correct finite-difference SGD:
 * - central differences
 * - ALL gradients computed before ANY weight update (critical)
 * - gradient clipping + lr cap for stability
 */
function trainStepFiniteDiff(
  model: Model,
  x: number[],
  y: number[],
  lr: number,
  l2: number
): number {
  const lossFn = () => {
    const out = predict(model, x);
    return binaryCrossEntropy(out, y) + l2Penalty(model, l2);
  };

  const params = collectParams(model);
  const eps = 1e-3;
  const grads = new Array(params.length).fill(0) as number[];

  for (let i = 0; i < params.length; i++) {
    const p = params[i]!;
    const orig = p.get();
    p.set(orig + eps);
    const up = lossFn();
    p.set(orig - eps);
    const down = lossFn();
    p.set(orig);
    grads[i] = (up - down) / (2 * eps);
  }

  // Cap step size: finite-diff noise + many params need modest updates
  const stepLr = Math.min(Math.max(lr, 1e-4), 0.15);
  for (let i = 0; i < params.length; i++) {
    const g = clip(grads[i]!, -2, 2);
    const p = params[i]!;
    p.set(p.get() - stepLr * g);
  }

  return lossFn();
}

function hasAttention(model: Model): boolean {
  return model.layers.some(
    (l) => l.type === "attention" || l.type === "transformer_block"
  );
}

/**
 * Hybrid step for attention/transformer + dense tail:
 * 1) analytical SGD on dense layers
 * 2) deferred central-diff on attention/transformer params only
 * Dense head learns quickly; attention still adapts without full-model FD cost.
 */
function trainStepHybridAttention(
  model: Model,
  x: number[],
  y: number[],
  lr: number,
  l2: number
): number {
  const { output } = forward(model, x);
  let delta = output.map((p, i) => p - y[i]!);

  // Dense layers reverse (analytical)
  for (let li = model.layers.length - 1; li >= 0; li--) {
    const layer = model.layers[li]!;
    if (layer.type !== "dense") continue;
    const isOut = li === model.layers.length - 1;
    delta = backwardDense(
      layer,
      delta,
      lr,
      l2,
      isOut && layer.activation === "sigmoid"
    );
  }

  // Finite-diff only non-dense params
  const lossFn = () => {
    const out = predict(model, x);
    return binaryCrossEntropy(out, y) + l2Penalty(model, l2);
  };

  const params: ParamRef[] = [];
  for (const layer of model.layers) {
    if (layer.type === "attention") {
      for (const mat of [layer.Wq, layer.Wk, layer.Wv, layer.Wo]) {
        for (let i = 0; i < mat.length; i++)
          for (let j = 0; j < mat[i]!.length; j++) {
            const ii = i,
              jj = j,
              m = mat;
            params.push({
              get: () => m[ii]![jj]!,
              set: (v) => {
                m[ii]![jj] = v;
              },
            });
          }
      }
    } else if (layer.type === "transformer_block") {
      for (const mat of [
        layer.attention.Wq,
        layer.attention.Wk,
        layer.attention.Wv,
        layer.attention.Wo,
        layer.W1,
        layer.W2,
      ]) {
        for (let i = 0; i < mat.length; i++)
          for (let j = 0; j < mat[i]!.length; j++) {
            const ii = i,
              jj = j,
              m = mat;
            params.push({
              get: () => m[ii]![jj]!,
              set: (v) => {
                m[ii]![jj] = v;
              },
            });
          }
      }
      for (const vec of [layer.b1, layer.b2]) {
        for (let i = 0; i < vec.length; i++) {
          const ii = i,
            v = vec;
          params.push({
            get: () => v[ii]!,
            set: (val) => {
              v[ii] = val;
            },
          });
        }
      }
    }
  }

  // Subsample params each step for speed (still unbiased enough for toys)
  const eps = 1e-3;
  const stepLr = Math.min(Math.max(lr, 1e-4), 0.08);
  const maxParams = 64;
  const stride = Math.max(1, Math.floor(params.length / maxParams));
  for (let i = 0; i < params.length; i += stride) {
    const p = params[i]!;
    const orig = p.get();
    p.set(orig + eps);
    const up = lossFn();
    p.set(orig - eps);
    const down = lossFn();
    p.set(orig);
    const g = clip((up - down) / (2 * eps), -2, 2);
    p.set(orig - stepLr * g);
  }

  return lossFn();
}

/** One training step: analytical when possible, hybrid/finite-diff otherwise. */
export function trainStep(
  model: Model,
  x: number[],
  y: number[],
  lr: number,
  l2: number,
  forceFiniteDiff = false
): number {
  if (forceFiniteDiff) {
    return trainStepFiniteDiff(model, x, y, lr, l2);
  }
  if (hasAttention(model)) {
    return trainStepHybridAttention(model, x, y, lr, l2);
  }
  const loss = trainStepAnalytical(model, x, y, lr, l2);
  if (loss != null) return loss;
  return trainStepFiniteDiff(model, x, y, lr, l2);
}

function accuracyOn(
  model: Model,
  samples: { x: number[]; y: number[] }[]
): number {
  if (!samples.length) return 0;
  let correct = 0;
  for (const s of samples) {
    const p = predict(model, s.x);
    if (p.length === 1) {
      const pred = p[0]! >= 0.5 ? 1 : 0;
      const tgt = s.y[0]! >= 0.5 ? 1 : 0;
      if (pred === tgt) correct++;
    } else {
      let pi = 0;
      let ti = 0;
      for (let i = 1; i < p.length; i++) if (p[i]! > p[pi]!) pi = i;
      for (let i = 1; i < s.y.length; i++) if (s.y[i]! > s.y[ti]!) ti = i;
      if (pi === ti) correct++;
    }
  }
  return correct / samples.length;
}

function epochLoss(
  model: Model,
  samples: { x: number[]; y: number[] }[],
  l2: number
): number {
  const losses = samples.map((s) => {
    const p = predict(model, s.x);
    return binaryCrossEntropy(p, s.y);
  });
  return mean(losses) + l2Penalty(model, l2);
}

export interface DecisionBounds {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

/**
 * 2-D axis-aligned bounds from sample features (first two dims) with margin.
 * Never returns a zero-width range — collapsed axes expand to at least `minSpan`.
 * Margin is ~15–20% of each axis range (default 18%).
 */
export function boundsFromSamples(
  samples: { x: number[] }[],
  marginRatio = 0.18,
  minSpan = 1
): DecisionBounds {
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;

  for (const s of samples) {
    if (s.x.length < 2) continue;
    const x = s.x[0]!;
    const y = s.x[1]!;
    if (x < xMin) xMin = x;
    if (x > xMax) xMax = x;
    if (y < yMin) yMin = y;
    if (y > yMax) yMax = y;
  }

  if (!Number.isFinite(xMin) || !Number.isFinite(yMin)) {
    return { xMin: -1.5, xMax: 1.5, yMin: -1.5, yMax: 1.5 };
  }

  let xSpan = xMax - xMin;
  let ySpan = yMax - yMin;

  if (xSpan < minSpan) {
    const mid = (xMin + xMax) / 2;
    xMin = mid - minSpan / 2;
    xMax = mid + minSpan / 2;
    xSpan = minSpan;
  }
  if (ySpan < minSpan) {
    const mid = (yMin + yMax) / 2;
    yMin = mid - minSpan / 2;
    yMax = mid + minSpan / 2;
    ySpan = minSpan;
  }

  const xPad = xSpan * marginRatio;
  const yPad = ySpan * marginRatio;
  return {
    xMin: xMin - xPad,
    xMax: xMax + xPad,
    yMin: yMin - yPad,
    yMax: yMax + yPad,
  };
}

/**
 * Evaluate the model on a regular 2-D grid.
 * Explicit xMin/xMax/yMin/yMax always win; callers that omit them get the
 * legacy default box [-1.5, 1.5]². Prefer passing boundsFromSamples(...) when
 * plotting against a dataset so scatter points share the frame.
 */
export function buildDecisionGrid(
  model: Model,
  resolution = 24,
  xMin = -1.5,
  xMax = 1.5,
  yMin = -1.5,
  yMax = 1.5
): TrainStepResult["decisionGrid"] {
  const values: number[] = [];
  for (let j = 0; j < resolution; j++) {
    const y = yMax - ((yMax - yMin) * j) / (resolution - 1 || 1);
    for (let i = 0; i < resolution; i++) {
      const x = xMin + ((xMax - xMin) * i) / (resolution - 1 || 1);
      const p = predict(model, [x, y]);
      values.push(p.length === 1 ? p[0]! : p[1] ?? p[0]!);
    }
  }
  return {
    width: resolution,
    height: resolution,
    values,
    xMin,
    xMax,
    yMin,
    yMax,
  };
}

function collectAttention(model: Model): number[][][] | undefined {
  for (const layer of model.layers) {
    if (layer.type === "attention" && layer.lastAttn) return layer.lastAttn;
    if (layer.type === "transformer_block" && layer.attention.lastAttn)
      return layer.attention.lastAttn;
  }
  return undefined;
}

export interface TrainOptions {
  onEpoch?: (result: TrainStepResult) => void;
  /** If true, use finite-diff even for dense/conv. */
  forceFiniteDiff?: boolean;
  includeDecisionBoundary?: boolean;
  /** Max samples per epoch (for heavy models). */
  maxSamples?: number;
  /** Optional initial weights (e.g. resume / deterministic tests). */
  initialWeights?: LayerWeights[];
}

export interface TrainingSession {
  /** Advance exactly one epoch; past the end is a safe no-op returning last snapshot. */
  runEpoch(): TrainStepResult;
  readonly epochsRun: number;
  readonly totalEpochs: number;
  readonly isDone: boolean;
  readonly losses: number[];
  readonly accuracies: number[];
  /** Per-epoch held-out loss (empty when no validation split). */
  readonly valLosses: number[];
  /** Per-epoch held-out accuracy (empty when no validation split). */
  readonly valAccuracies: number[];
  /** True when this session computed a held-out split. */
  readonly hasValidation: boolean;
  readonly lastSnapshot: TrainStepResult | null;
  exportWeights(): LayerWeights[];
  /** Fresh random init, clears history, epochsRun = 0. */
  reset(): void;
}

/** Shared input-dim / conv-shape / post-transformer-dense inference. */
export function prepareNetworkConfig(
  config: NetworkConfig,
  datasetName: TrainConfig["dataset"]
): { config: NetworkConfig; inputDim: number } {
  const ds = getDataset(datasetName);
  let inputDim = 2;
  if (ds.inputShape.length === 1) inputDim = ds.inputShape[0]!;
  else inputDim = ds.inputShape.reduce((a, b) => a * b, 1);

  const layers = config.layers.map((l, i) => {
    if (i === 0 && l.type === "dense" && !l.inputDim) {
      return { ...l, inputDim };
    }
    if (i === 0 && l.type === "conv2d") {
      return {
        ...l,
        inputChannels: l.inputChannels ?? ds.inputShape[0] ?? 1,
        inputHeight: l.inputHeight ?? ds.inputShape[1] ?? 4,
        inputWidth: l.inputWidth ?? ds.inputShape[2] ?? 4,
      };
    }
    return l;
  });

  // For transformer on bag-of-features: ensure first dense after transformer
  // has correct inputDim = dModel
  for (let i = 0; i < layers.length; i++) {
    const l = layers[i]!;
    if (
      l.type === "dense" &&
      !l.inputDim &&
      i > 0 &&
      (layers[i - 1]!.type === "transformer_block" ||
        layers[i - 1]!.type === "attention")
    ) {
      const prev = layers[i - 1]!;
      const dm =
        prev.type === "transformer_block"
          ? prev.dModel
          : prev.type === "attention"
            ? prev.dModel
            : inputDim;
      layers[i] = { ...l, inputDim: dm };
    }
  }

  return { config: { ...config, layers }, inputDim };
}

function shuffleInPlace<T>(arr: T[]): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j]!, arr[i]!];
  }
}

/**
 * Build the train/val split once per run from the cached dataset.
 * Training code only ever receives `split.train` as the optimizable sample list.
 */
export function prepareDataSplit(trainConfig: TrainConfig): DataSplit {
  const ds = getDataset(trainConfig.dataset);
  return splitTrainVal(ds.samples, resolveValRatio(trainConfig.valRatio));
}

/** Run a single epoch body (identical path for train() and TrainingSession). */
function runOneEpoch(
  model: Model,
  trainConfig: TrainConfig,
  options: TrainOptions,
  epoch: number,
  samplesWork: Sample[],
  valSamples: Sample[] | null
): TrainStepResult {
  const ds = getDataset(trainConfig.dataset);
  const l2 = model.config.l2 ?? 0;
  const lr = trainConfig.learningRate;
  const heavy = hasAttention(model);
  const maxSamples =
    options.maxSamples ??
    (heavy ? Math.min(samplesWork.length, 14) : samplesWork.length);

  // Shuffle only the training copy — val is never mixed into the optimizer path.
  if (trainConfig.shuffle !== false) {
    shuffleInPlace(samplesWork);
  }

  const batch = samplesWork.slice(0, maxSamples);
  for (const s of batch) {
    trainStep(model, s.x, s.y, lr, l2, options.forceFiniteDiff === true);
  }

  if (samplesWork[0]) forward(model, samplesWork[0].x);

  const loss = epochLoss(model, samplesWork, l2);
  const accuracy = accuracyOn(model, samplesWork);

  let valLoss: number | null = null;
  let valAccuracy: number | null = null;
  if (valSamples && valSamples.length > 0) {
    valLoss = epochLoss(model, valSamples, l2);
    valAccuracy = accuracyOn(model, valSamples);
  }

  const is2d =
    ds.inputShape.length === 1 && ds.inputShape[0] === 2;
  let decisionGrid: TrainStepResult["decisionGrid"];
  if (options.includeDecisionBoundary !== false && is2d) {
    const b = boundsFromSamples(ds.samples);
    decisionGrid = buildDecisionGrid(
      model,
      24,
      b.xMin,
      b.xMax,
      b.yMin,
      b.yMax
    );
  }

  const result: TrainStepResult = {
    epoch,
    loss,
    accuracy,
    valLoss,
    valAccuracy,
    layerSnapshots: snapshotLayers(model),
    predictions: samplesWork.slice(0, 8).map((s) => predict(model, s.x)),
    decisionGrid,
    attentionMaps: collectAttention(model),
  };
  options.onEpoch?.(result);
  return result;
}

/**
 * Train model for N epochs. Immediate Mode: call with new configs freely.
 * Metrics `loss`/`accuracy` are train-set only; `valLoss`/`valAccuracy` are held-out.
 */
export function train(
  model: Model,
  trainConfig: TrainConfig,
  options: TrainOptions = {}
): TrainHistory {
  const split = prepareDataSplit(trainConfig);
  // Working train copy only — structurally cannot train on val.
  const samples = split.train.slice();
  const valSamples = split.val;
  const losses: number[] = [];
  const accuracies: number[] = [];
  const valLosses: number[] = [];
  const valAccuracies: number[] = [];
  let last: TrainStepResult = {
    epoch: 0,
    loss: 0,
    accuracy: 0,
    valLoss: null,
    valAccuracy: null,
    layerSnapshots: snapshotLayers(model),
  };

  for (let epoch = 1; epoch <= trainConfig.epochs; epoch++) {
    last = runOneEpoch(
      model,
      trainConfig,
      options,
      epoch,
      samples,
      valSamples
    );
    losses.push(last.loss);
    accuracies.push(last.accuracy);
    if (last.valLoss != null) valLosses.push(last.valLoss);
    if (last.valAccuracy != null) valAccuracies.push(last.valAccuracy);
  }

  return { losses, accuracies, valLosses, valAccuracies, final: last };
}

/**
 * Incremental training session — one epoch per runEpoch() call.
 * Keeps model state alive between epochs so a rAF loop can animate learning.
 */
export function createTrainingSession(
  config: NetworkConfig,
  trainConfig: TrainConfig,
  options: TrainOptions = {}
): TrainingSession {
  const prepared = prepareNetworkConfig(config, trainConfig.dataset);
  const split = prepareDataSplit(trainConfig);
  // Val list is frozen for the session lifetime; only train is reshuffled.
  const valSamples = split.val;
  let model = createModel(prepared.config, prepared.inputDim);
  if (options.initialWeights) {
    loadWeights(model, options.initialWeights);
  }
  let epochsRun = 0;
  const losses: number[] = [];
  const accuracies: number[] = [];
  const valLosses: number[] = [];
  const valAccuracies: number[] = [];
  let lastSnapshot: TrainStepResult | null = null;
  // Working train copy only — never includes held-out samples.
  let samplesWork = split.train.slice();

  const reinitModel = () => {
    model = createModel(prepared.config, prepared.inputDim);
    if (options.initialWeights) {
      loadWeights(model, options.initialWeights);
    }
  };

  const session: TrainingSession = {
    runEpoch(): TrainStepResult {
      if (epochsRun >= trainConfig.epochs) {
        if (lastSnapshot) return lastSnapshot;
        // Never trained — return a zero snapshot without advancing
        const empty: TrainStepResult = {
          epoch: 0,
          loss: 0,
          accuracy: 0,
          valLoss: null,
          valAccuracy: null,
          layerSnapshots: snapshotLayers(model),
        };
        lastSnapshot = empty;
        return empty;
      }
      const epoch = epochsRun + 1;
      const result = runOneEpoch(
        model,
        trainConfig,
        options,
        epoch,
        samplesWork,
        valSamples
      );
      epochsRun = epoch;
      losses.push(result.loss);
      accuracies.push(result.accuracy);
      if (result.valLoss != null) valLosses.push(result.valLoss);
      if (result.valAccuracy != null) valAccuracies.push(result.valAccuracy);
      lastSnapshot = result;
      return result;
    },
    get epochsRun() {
      return epochsRun;
    },
    get totalEpochs() {
      return trainConfig.epochs;
    },
    get isDone() {
      return epochsRun >= trainConfig.epochs;
    },
    get losses() {
      return losses;
    },
    get accuracies() {
      return accuracies;
    },
    get valLosses() {
      return valLosses;
    },
    get valAccuracies() {
      return valAccuracies;
    },
    get hasValidation() {
      return split.splitApplied;
    },
    get lastSnapshot() {
      return lastSnapshot;
    },
    exportWeights(): LayerWeights[] {
      return exportWeights(model);
    },
    reset(): void {
      reinitModel();
      epochsRun = 0;
      losses.length = 0;
      accuracies.length = 0;
      valLosses.length = 0;
      valAccuracies.length = 0;
      lastSnapshot = null;
      samplesWork = split.train.slice();
    },
  };

  return session;
}

export function createAndTrain(
  config: NetworkConfig,
  trainConfig: TrainConfig,
  options?: TrainOptions
): { model: Model; history: TrainHistory; weights: LayerWeights[] } {
  const { config: prepared, inputDim } = prepareNetworkConfig(
    config,
    trainConfig.dataset
  );
  const model = createModel(prepared, inputDim);
  const history = train(model, trainConfig, options);
  return { model, history, weights: exportWeights(model) };
}

export function evaluateModel(
  model: Model,
  datasetName: TrainConfig["dataset"]
): { loss: number; accuracy: number } {
  const ds = getDataset(datasetName);
  const l2 = model.config.l2 ?? 0;
  return {
    loss: epochLoss(model, ds.samples, l2),
    accuracy: accuracyOn(model, ds.samples),
  };
}

export function cloneModelFromWeights(
  config: NetworkConfig,
  weights: LayerWeights[],
  inputDim = 2
): Model {
  const model = createModel(config, inputDim);
  loadWeights(model, weights);
  return model;
}

export { clip, mse };
