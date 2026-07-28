/** Small numeric helpers for the educational neural runtime. */

export function zeros(n: number): number[] {
  return Array.from({ length: n }, () => 0);
}

export function zeros2d(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () => zeros(cols));
}

export function randn(mean = 0, std = 1): number {
  // Box-Muller
  const u = 1 - Math.random();
  const v = Math.random();
  const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  return mean + z * std;
}

export function xavier(fanIn: number, fanOut: number): number {
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  return (Math.random() * 2 - 1) * limit;
}

export function he(fanIn: number): number {
  return randn(0, Math.sqrt(2 / fanIn));
}

export function clip(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

export function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

export function sigmoidPrime(y: number): number {
  // y is already sigmoid(x)
  return y * (1 - y);
}

export function relu(x: number): number {
  return x > 0 ? x : 0;
}

export function reluPrime(x: number): number {
  return x > 0 ? 1 : 0;
}

export function tanh(x: number): number {
  return Math.tanh(x);
}

export function tanhPrime(y: number): number {
  return 1 - y * y;
}

export function softmax(xs: number[]): number[] {
  const max = Math.max(...xs);
  const exps = xs.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / sum);
}

export function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

export function matVec(m: number[][], v: number[]): number[] {
  return m.map((row) => dot(row, v));
}

export function addBias(v: number[], b: number[]): number[] {
  return v.map((x, i) => x + (b[i] ?? 0));
}

export function applyActivation(
  v: number[],
  name: string,
): number[] {
  switch (name) {
    case "sigmoid":
      return v.map(sigmoid);
    case "relu":
      return v.map(relu);
    case "tanh":
      return v.map(tanh);
    case "softmax":
      return softmax(v);
    case "linear":
    default:
      return v.slice();
  }
}

export function activationPrime(
  name: string,
  activated: number[],
  pre: number[]
): number[] {
  switch (name) {
    case "sigmoid":
      return activated.map(sigmoidPrime);
    case "relu":
      return pre.map(reluPrime);
    case "tanh":
      return activated.map(tanhPrime);
    case "softmax":
      // Used with cross-entropy: combined gradient handled outside.
      return activated.map(() => 1);
    case "linear":
    default:
      return activated.map(() => 1);
  }
}

export function mse(pred: number[], target: number[]): number {
  let s = 0;
  for (let i = 0; i < pred.length; i++) {
    const d = pred[i]! - target[i]!;
    s += d * d;
  }
  return s / (pred.length || 1);
}

export function binaryCrossEntropy(pred: number[], target: number[]): number {
  const eps = 1e-7;
  let s = 0;
  for (let i = 0; i < pred.length; i++) {
    const p = clip(pred[i]!, eps, 1 - eps);
    const t = target[i]!;
    s += -(t * Math.log(p) + (1 - t) * Math.log(1 - p));
  }
  return s / (pred.length || 1);
}

export function argmax(v: number[]): number {
  let best = 0;
  for (let i = 1; i < v.length; i++) {
    if (v[i]! > v[best]!) best = i;
  }
  return best;
}

export function mean(xs: number[]): number {
  if (!xs.length) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}
