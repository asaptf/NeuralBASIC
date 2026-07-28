import {
  activationPrime,
  addBias,
  applyActivation,
  he,
  matVec,
  sigmoid,
  softmax,
  xavier,
  zeros,
  zeros2d,
} from "./math";
import type {
  ActivationName,
  LayerSnapshot,
  LayerWeights,
  NetworkConfig,
} from "./types";

export interface DenseLayerState {
  type: "dense";
  weights: number[][]; // [out][in]
  biases: number[];
  activation: ActivationName;
  lastInput?: number[];
  lastPre?: number[];
  lastOut?: number[];
}

export interface Conv2dLayerState {
  type: "conv2d";
  filters: number; // out channels
  kernelSize: number;
  inChannels: number;
  // kernels[out][in][kh][kw]
  kernels: number[][][][];
  biases: number[];
  activation: ActivationName;
  lastInput?: number[][][]; // C,H,W
  lastPre?: number[][][];
  lastOut?: number[][][];
}

export interface FlattenLayerState {
  type: "flatten";
  lastShape?: number[];
  lastOut?: number[];
}

export interface AttentionLayerState {
  type: "attention";
  dModel: number;
  nHeads: number;
  Wq: number[][];
  Wk: number[][];
  Wv: number[][];
  Wo: number[][];
  lastInput?: number[][]; // [seq][d]
  lastAttn?: number[][][]; // [heads][seq][seq]
  lastOut?: number[][];
}

export interface TransformerBlockState {
  type: "transformer_block";
  attention: AttentionLayerState;
  W1: number[][];
  b1: number[];
  W2: number[][];
  b2: number[];
  lastOut?: number[][];
}

export type LayerState =
  | DenseLayerState
  | Conv2dLayerState
  | FlattenLayerState
  | AttentionLayerState
  | TransformerBlockState;

export interface Model {
  config: NetworkConfig;
  layers: LayerState[];
}

function initDense(
  inputDim: number,
  units: number,
  activation: ActivationName = "sigmoid"
): DenseLayerState {
  const weights = zeros2d(units, inputDim);
  for (let o = 0; o < units; o++) {
    for (let i = 0; i < inputDim; i++) {
      weights[o]![i] =
        activation === "relu" ? he(inputDim) : xavier(inputDim, units);
    }
  }
  return {
    type: "dense",
    weights,
    biases: zeros(units),
    activation,
  };
}

function initConv(
  inChannels: number,
  filters: number,
  kernelSize: number,
  activation: ActivationName = "relu"
): Conv2dLayerState {
  const kernels: number[][][][] = [];
  const fanIn = inChannels * kernelSize * kernelSize;
  for (let f = 0; f < filters; f++) {
    const perIn: number[][][] = [];
    for (let c = 0; c < inChannels; c++) {
      const k: number[][] = [];
      for (let y = 0; y < kernelSize; y++) {
        const row: number[] = [];
        for (let x = 0; x < kernelSize; x++) {
          row.push(he(fanIn) * 0.5);
        }
        k.push(row);
      }
      perIn.push(k);
    }
    kernels.push(perIn);
  }
  return {
    type: "conv2d",
    filters,
    kernelSize,
    inChannels,
    kernels,
    biases: zeros(filters),
    activation,
  };
}

function initAttention(dModel: number, nHeads = 2): AttentionLayerState {
  const initMat = (r: number, c: number) => {
    const m = zeros2d(r, c);
    for (let i = 0; i < r; i++)
      for (let j = 0; j < c; j++) m[i]![j] = xavier(c, r) * 0.5;
    return m;
  };
  return {
    type: "attention",
    dModel,
    nHeads,
    Wq: initMat(dModel, dModel),
    Wk: initMat(dModel, dModel),
    Wv: initMat(dModel, dModel),
    Wo: initMat(dModel, dModel),
  };
}

function initTransformer(
  dModel: number,
  nHeads = 2,
  dff = 8
): TransformerBlockState {
  const initMat = (r: number, c: number) => {
    const m = zeros2d(r, c);
    for (let i = 0; i < r; i++)
      for (let j = 0; j < c; j++) m[i]![j] = xavier(c, r) * 0.5;
    return m;
  };
  return {
    type: "transformer_block",
    attention: initAttention(dModel, nHeads),
    W1: initMat(dff, dModel),
    b1: zeros(dff),
    W2: initMat(dModel, dff),
    b2: zeros(dModel),
  };
}

/** Infer sequential layer input dims and construct model. */
export function createModel(
  config: NetworkConfig,
  defaultInputDim = 2
): Model {
  const layers: LayerState[] = [];
  let flatDim = defaultInputDim;
  let channels = 1;
  let height = 0;
  let width = 0;
  let isSpatial = false;

  for (const lc of config.layers) {
    if (lc.type === "dense") {
      const inDim = lc.inputDim ?? flatDim;
      const layer = initDense(inDim, lc.units, lc.activation ?? "sigmoid");
      layers.push(layer);
      flatDim = lc.units;
      isSpatial = false;
    } else if (lc.type === "conv2d") {
      const inCh = lc.inputChannels ?? (isSpatial ? channels : 1);
      if (!isSpatial && lc.inputHeight && lc.inputWidth) {
        height = lc.inputHeight;
        width = lc.inputWidth;
        channels = inCh;
        isSpatial = true;
      } else if (!isSpatial) {
        // assume square from flatDim
        const side = Math.round(Math.sqrt(flatDim / inCh));
        height = side;
        width = side;
        channels = inCh;
        isSpatial = true;
      }
      const layer = initConv(
        inCh,
        lc.filters,
        lc.kernelSize,
        lc.activation ?? "relu"
      );
      layers.push(layer);
      channels = lc.filters;
      // valid padding: H' = H - k + 1
      height = Math.max(1, height - lc.kernelSize + 1);
      width = Math.max(1, width - lc.kernelSize + 1);
      flatDim = channels * height * width;
    } else if (lc.type === "flatten") {
      layers.push({ type: "flatten" });
      isSpatial = false;
    } else if (lc.type === "attention") {
      const dm = lc.dModel;
      layers.push(initAttention(dm, lc.nHeads ?? 2));
      flatDim = dm;
    } else if (lc.type === "transformer_block") {
      const dm = lc.dModel;
      layers.push(initTransformer(dm, lc.nHeads ?? 2, lc.dff ?? dm * 2));
      flatDim = dm;
    }
  }

  return { config, layers };
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
      for (let x = 0; x < w; x++) {
        row.push(flat[idx++] ?? 0);
      }
      plane.push(row);
    }
    out.push(plane);
  }
  return out;
}

function flattenCHW(t: number[][][]): number[] {
  const out: number[] = [];
  for (const plane of t)
    for (const row of plane) for (const v of row) out.push(v);
  return out;
}

function forwardDense(layer: DenseLayerState, input: number[]): number[] {
  layer.lastInput = input;
  const pre = addBias(matVec(layer.weights, input), layer.biases);
  layer.lastPre = pre;
  const out = applyActivation(pre, layer.activation);
  layer.lastOut = out;
  return out;
}

function forwardConv(layer: Conv2dLayerState, input: number[][][]): number[][][] {
  layer.lastInput = input;
  const inC = input.length;
  const h = input[0]!.length;
  const w = input[0]![0]!.length;
  const k = layer.kernelSize;
  const outH = Math.max(1, h - k + 1);
  const outW = Math.max(1, w - k + 1);
  const pre: number[][][] = [];
  const out: number[][][] = [];

  for (let f = 0; f < layer.filters; f++) {
    const plane: number[][] = [];
    const prePlane: number[][] = [];
    for (let y = 0; y < outH; y++) {
      const row: number[] = [];
      const preRow: number[] = [];
      for (let x = 0; x < outW; x++) {
        let sum = layer.biases[f] ?? 0;
        for (let c = 0; c < inC; c++) {
          for (let ky = 0; ky < k; ky++) {
            for (let kx = 0; kx < k; kx++) {
              sum +=
                (input[c]![y + ky]![x + kx] ?? 0) *
                (layer.kernels[f]![c]![ky]![kx] ?? 0);
            }
          }
        }
        preRow.push(sum);
        row.push(
          layer.activation === "relu"
            ? Math.max(0, sum)
            : layer.activation === "sigmoid"
              ? sigmoid(sum)
              : sum
        );
      }
      plane.push(row);
      prePlane.push(preRow);
    }
    out.push(plane);
    pre.push(prePlane);
  }
  layer.lastPre = pre;
  layer.lastOut = out;
  return out;
}

function matMulVecRows(m: number[][], v: number[]): number[] {
  // m is [out][in], v is [in]
  return matVec(m, v);
}

function forwardAttention(
  layer: AttentionLayerState,
  tokens: number[][]
): number[][] {
  layer.lastInput = tokens;
  const seq = tokens.length;
  const d = layer.dModel;
  const nHeads = layer.nHeads;
  const dHead = Math.max(1, Math.floor(d / nHeads));

  const project = (W: number[][], x: number[]) => matMulVecRows(W, x);

  const Q = tokens.map((t) => project(layer.Wq, t));
  const K = tokens.map((t) => project(layer.Wk, t));
  const V = tokens.map((t) => project(layer.Wv, t));

  const allAttn: number[][][] = [];
  const headOuts: number[][][] = []; // [heads][seq][dHead]

  for (let h = 0; h < nHeads; h++) {
    const qs = Q.map((q) => q.slice(h * dHead, h * dHead + dHead));
    const ks = K.map((k) => k.slice(h * dHead, h * dHead + dHead));
    const vs = V.map((v) => v.slice(h * dHead, h * dHead + dHead));
    const scale = 1 / Math.sqrt(dHead);
    const attn: number[][] = [];
    const hout: number[][] = [];
    for (let i = 0; i < seq; i++) {
      const scores: number[] = [];
      for (let j = 0; j < seq; j++) {
        let s = 0;
        for (let t = 0; t < dHead; t++) s += (qs[i]![t] ?? 0) * (ks[j]![t] ?? 0);
        scores.push(s * scale);
      }
      const weights = softmax(scores);
      attn.push(weights);
      const o = zeros(dHead);
      for (let j = 0; j < seq; j++) {
        for (let t = 0; t < dHead; t++) {
          o[t]! += weights[j]! * (vs[j]![t] ?? 0);
        }
      }
      hout.push(o);
    }
    allAttn.push(attn);
    headOuts.push(hout);
  }

  layer.lastAttn = allAttn;

  // concat heads
  const concat: number[][] = [];
  for (let i = 0; i < seq; i++) {
    const row: number[] = [];
    for (let h = 0; h < nHeads; h++) {
      const piece = headOuts[h]![i]!;
      for (const v of piece) row.push(v);
    }
    // pad/truncate to dModel
    while (row.length < d) row.push(0);
    concat.push(row.slice(0, d));
  }

  const out = concat.map((t) => project(layer.Wo, t));
  layer.lastOut = out;
  return out;
}

function forwardTransformer(
  layer: TransformerBlockState,
  tokens: number[][]
): number[][] {
  const attnOut = forwardAttention(layer.attention, tokens);
  // residual
  const res1 = attnOut.map((row, i) =>
    row.map((v, j) => v + (tokens[i]![j] ?? 0))
  );
  // FFN
  const out = res1.map((tok) => {
    const h = addBias(matVec(layer.W1, tok), layer.b1).map((x) =>
      Math.max(0, x)
    );
    const y = addBias(matVec(layer.W2, h), layer.b2);
    return y.map((v, j) => v + (tok[j] ?? 0));
  });
  layer.lastOut = out;
  return out;
}

export type ForwardCache = {
  output: number[];
  spatial?: number[][][];
  tokens?: number[][];
};

/**
 * Forward pass. Input is flat vector. For conv, uses first conv's
 * declared spatial dims or sqrt inference.
 */
export function forward(model: Model, input: number[]): ForwardCache {
  let flat = input.slice();
  let spatial: number[][][] | undefined;
  let tokens: number[][] | undefined;

  // If first layer is conv, reshape
  const first = model.layers[0];
  if (first?.type === "conv2d") {
    const c = first.inChannels;
    const side = Math.round(Math.sqrt(flat.length / c));
    spatial = reshapeCHW(flat, c, side, side);
  }

  // If attention/transformer and input is flat sequence of features,
  // treat as single token or reshape by dModel
  if (
    first?.type === "attention" ||
    first?.type === "transformer_block"
  ) {
    const d =
      first.type === "attention" ? first.dModel : first.attention.dModel;
    if (flat.length === d) {
      tokens = [flat];
    } else {
      const seq = Math.max(1, Math.floor(flat.length / d));
      tokens = [];
      for (let i = 0; i < seq; i++) {
        tokens.push(flat.slice(i * d, i * d + d));
      }
      // pad last
      const last = tokens[tokens.length - 1]!;
      while (last.length < d) last.push(0);
    }
  }

  for (const layer of model.layers) {
    if (layer.type === "dense") {
      if (tokens) {
        // pool tokens then dense, or apply to mean
        const meanTok = zeros(tokens[0]!.length);
        for (const t of tokens) {
          for (let i = 0; i < meanTok.length; i++) meanTok[i]! += t[i]!;
        }
        for (let i = 0; i < meanTok.length; i++)
          meanTok[i]! /= tokens.length;
        flat = forwardDense(layer, meanTok);
        tokens = undefined;
        spatial = undefined;
      } else if (spatial) {
        flat = forwardDense(layer, flattenCHW(spatial));
        spatial = undefined;
      } else {
        flat = forwardDense(layer, flat);
      }
    } else if (layer.type === "conv2d") {
      if (!spatial) {
        const c = layer.inChannels;
        const side = Math.round(Math.sqrt(flat.length / c));
        spatial = reshapeCHW(flat, c, side, side);
      }
      spatial = forwardConv(layer, spatial);
      flat = flattenCHW(spatial);
    } else if (layer.type === "flatten") {
      if (spatial) {
        layer.lastShape = [
          spatial.length,
          spatial[0]!.length,
          spatial[0]![0]!.length,
        ];
        flat = flattenCHW(spatial);
        layer.lastOut = flat;
        spatial = undefined;
      } else {
        layer.lastOut = flat;
      }
    } else if (layer.type === "attention") {
      if (!tokens) {
        tokens = [flat.slice(0, layer.dModel)];
        while (tokens[0]!.length < layer.dModel) tokens[0]!.push(0);
      }
      tokens = forwardAttention(layer, tokens);
      flat = tokens[0]!.slice(); // first token / CLS-like
      // mean pool for multi-token
      if (tokens.length > 1) {
        const m = zeros(layer.dModel);
        for (const t of tokens)
          for (let i = 0; i < m.length; i++) m[i]! += t[i]!;
        for (let i = 0; i < m.length; i++) m[i]! /= tokens.length;
        flat = m;
      }
    } else if (layer.type === "transformer_block") {
      if (!tokens) {
        const d = layer.attention.dModel;
        tokens = [flat.slice(0, d)];
        while (tokens[0]!.length < d) tokens[0]!.push(0);
      }
      tokens = forwardTransformer(layer, tokens);
      const d = layer.attention.dModel;
      const m = zeros(d);
      for (const t of tokens) for (let i = 0; i < d; i++) m[i]! += t[i]!;
      for (let i = 0; i < d; i++) m[i]! /= tokens.length;
      flat = m;
    }
  }

  return { output: flat, spatial, tokens };
}

export function predict(model: Model, input: number[]): number[] {
  return forward(model, input).output;
}

/** Collect visualization snapshots after a forward pass. */
export function snapshotLayers(model: Model): LayerSnapshot[] {
  return model.layers.map((layer) => {
    if (layer.type === "dense") {
      return {
        type: "dense",
        weights: layer.weights.map((r) => r.slice()),
        biases: layer.biases.slice(),
        activations: layer.lastOut?.slice(),
        shape: [layer.weights[0]?.length ?? 0, layer.weights.length],
      };
    }
    if (layer.type === "conv2d") {
      const flatW = layer.kernels.map((f) =>
        f.flatMap((c) => c.flatMap((r) => r))
      );
      return {
        type: "conv2d",
        weights: flatW,
        biases: layer.biases.slice(),
        activations: layer.lastOut ? flattenCHW(layer.lastOut) : undefined,
        shape: [layer.filters, layer.inChannels, layer.kernelSize],
      };
    }
    if (layer.type === "flatten") {
      return {
        type: "flatten",
        activations: layer.lastOut?.slice(),
        shape: layer.lastShape,
      };
    }
    if (layer.type === "attention") {
      return {
        type: "attention",
        activations: layer.lastOut?.[0]?.slice(),
        attention: layer.lastAttn?.map((h) => h.map((r) => r.slice())),
        shape: [layer.dModel, layer.nHeads],
      };
    }
    return {
      type: "transformer_block",
      activations: layer.lastOut?.[0]?.slice(),
      attention: layer.attention.lastAttn?.map((h) =>
        h.map((r) => r.slice())
      ),
      shape: [layer.attention.dModel, layer.attention.nHeads],
    };
  });
}

export function exportWeights(model: Model): LayerWeights[] {
  const out: LayerWeights[] = [];
  for (const layer of model.layers) {
    if (layer.type === "dense") {
      out.push({
        type: "dense",
        weights: layer.weights.map((r) => r.slice()),
        biases: layer.biases.slice(),
      });
    } else if (layer.type === "conv2d") {
      out.push({
        type: "conv2d",
        weights: layer.kernels.map((f) =>
          f.flatMap((c) => c.flatMap((r) => r))
        ),
        biases: layer.biases.slice(),
      });
    } else if (layer.type === "flatten") {
      out.push({ type: "flatten" });
    } else if (layer.type === "attention") {
      const params: Record<string, number[][]> = {
        Wq: layer.Wq.map((r) => r.slice()),
        Wk: layer.Wk.map((r) => r.slice()),
        Wv: layer.Wv.map((r) => r.slice()),
        Wo: layer.Wo.map((r) => r.slice()),
      };
      out.push({ type: "attention", params });
    } else {
      const params: Record<string, number[][]> = {
        Wq: layer.attention.Wq.map((r) => r.slice()),
        Wk: layer.attention.Wk.map((r) => r.slice()),
        Wv: layer.attention.Wv.map((r) => r.slice()),
        Wo: layer.attention.Wo.map((r) => r.slice()),
        W1: layer.W1.map((r) => r.slice()),
        W2: layer.W2.map((r) => r.slice()),
      };
      out.push({
        type: "transformer_block",
        params,
        biases: [...layer.b1, ...layer.b2],
      });
    }
  }
  return out;
}

export function loadWeights(model: Model, weights: LayerWeights[]): void {
  for (let i = 0; i < model.layers.length && i < weights.length; i++) {
    const layer = model.layers[i]!;
    const w = weights[i]!;
    if (layer.type === "dense" && w.weights && w.biases) {
      layer.weights = w.weights.map((r) => r.slice());
      layer.biases = w.biases.slice();
    } else if (layer.type === "conv2d" && w.weights && w.biases) {
      // rebuild kernels from flat
      const k = layer.kernelSize;
      const inC = layer.inChannels;
      for (let f = 0; f < layer.filters; f++) {
        const flat = w.weights[f] ?? [];
        let idx = 0;
        for (let c = 0; c < inC; c++) {
          for (let y = 0; y < k; y++) {
            for (let x = 0; x < k; x++) {
              layer.kernels[f]![c]![y]![x] = flat[idx++] ?? 0;
            }
          }
        }
      }
      layer.biases = w.biases.slice();
    } else if (layer.type === "attention" && w.params) {
      if (w.params.Wq) layer.Wq = w.params.Wq.map((r) => r.slice());
      if (w.params.Wk) layer.Wk = w.params.Wk.map((r) => r.slice());
      if (w.params.Wv) layer.Wv = w.params.Wv.map((r) => r.slice());
      if (w.params.Wo) layer.Wo = w.params.Wo.map((r) => r.slice());
    } else if (layer.type === "transformer_block" && w.params) {
      if (w.params.Wq)
        layer.attention.Wq = w.params.Wq.map((r) => r.slice());
      if (w.params.Wk)
        layer.attention.Wk = w.params.Wk.map((r) => r.slice());
      if (w.params.Wv)
        layer.attention.Wv = w.params.Wv.map((r) => r.slice());
      if (w.params.Wo)
        layer.attention.Wo = w.params.Wo.map((r) => r.slice());
      if (w.params.W1) layer.W1 = w.params.W1.map((r) => r.slice());
      if (w.params.W2) layer.W2 = w.params.W2.map((r) => r.slice());
    }
  }
}

// re-export activation prime for train
export { activationPrime };
