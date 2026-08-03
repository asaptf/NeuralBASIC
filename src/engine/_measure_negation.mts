/**
 * Measure full-train accuracy distribution for negation attention + baselines.
 *
 * Matches the training recipe in negation.test.ts `trainOnSamples` exactly
 * (createModel + Fisher–Yates shuffle + trainStep per sample — NOT createAndTrain).
 *
 * Run:
 *   npx --yes tsx src/engine/_measure_negation.mts
 *   npx --yes tsx src/engine/_measure_negation.mts /tmp/negation-accs.json
 * Fallback if extensionless imports fail:
 *   npx --yes vite-node src/engine/_measure_negation.mts
 */
import * as fs from "node:fs";
import { getDataset, NEGATION_D_MODEL, NEGATION_SEQ_LEN } from "./datasets";
import { createModel, predict, type Model } from "./model";
import { trainStep } from "./train";
import type { NetworkConfig, Sample } from "./types";

const V = NEGATION_D_MODEL;
const SEQ = NEGATION_SEQ_LEN;
const FLAT = SEQ * V;
const EPOCHS = 150;
const BOOTSTRAP_TRIALS = 20_000;
const K_VALUES = [8, 12, 16, 20, 24, 32] as const;
const BUCKET = 0.05;

// ── stats ──────────────────────────────────────────────────────────────────

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function std(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  const v = xs.reduce((s, x) => s + (x - m) * (x - m), 0) / (xs.length - 1);
  return Math.sqrt(v);
}

function median(xs: number[]): number {
  const s = xs.slice().sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[mid - 1]! + s[mid]!) / 2 : s[mid]!;
}

function fmt(x: number, d = 4): string {
  return x.toFixed(d);
}

function fmtProb(p: number): string {
  if (p < 1e-4) return "<1e-4";
  return p.toFixed(4);
}

// ── training recipe (mirrors negation.test.ts trainOnSamples) ──────────────

function accuracyOn(model: Model, samples: Sample[]): number {
  if (!samples.length) return 0;
  let correct = 0;
  for (const s of samples) {
    const p = predict(model, s.x);
    const pred = p[0]! >= 0.5 ? 1 : 0;
    const tgt = s.y[0]! >= 0.5 ? 1 : 0;
    if (pred === tgt) correct++;
  }
  return correct / samples.length;
}

/** Sum of per-position one-hots → vocab count vector (bag-of-words). */
function bagOfTokens(x: number[]): number[] {
  const b = Array.from({ length: V }, () => 0);
  for (let i = 0; i < SEQ; i++) {
    for (let j = 0; j < V; j++) b[j]! += x[i * V + j]!;
  }
  return b;
}

function trainOnSamples(
  config: NetworkConfig,
  samples: Sample[],
  lr: number,
  epochs: number,
  inputDim: number
): Model {
  const model = createModel(config, inputDim);
  for (let e = 0; e < epochs; e++) {
    const order = samples.slice();
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [order[i], order[j]] = [order[j]!, order[i]!];
    }
    for (const s of order) trainStep(model, s.x, s.y, lr, 0);
  }
  return model;
}

// ── configurations ─────────────────────────────────────────────────────────

type Spec = {
  name: string;
  config: NetworkConfig;
  lr: number;
  n: number;
  inputDim: number;
  /** Optional sample transform (e.g. bag-of-tokens). */
  transform?: (s: Sample) => Sample;
};

const attentionConfig: NetworkConfig = {
  layers: [
    { type: "attention", dModel: 4, nHeads: 2 },
    { type: "dense", units: 1, activation: "sigmoid", inputDim: 4 },
  ],
};

const linearFlatConfig: NetworkConfig = {
  layers: [{ type: "dense", units: 1, activation: "sigmoid", inputDim: 16 }],
};

const linearBagConfig: NetworkConfig = {
  layers: [{ type: "dense", units: 1, activation: "sigmoid", inputDim: 4 }],
};

const denseFlatConfig: NetworkConfig = {
  layers: [
    { type: "dense", units: 16, activation: "relu", inputDim: 16 },
    { type: "dense", units: 8, activation: "relu" },
    { type: "dense", units: 1, activation: "sigmoid" },
  ],
};

const SPECS: Spec[] = [
  {
    name: "attention",
    config: attentionConfig,
    lr: 0.1,
    n: 200,
    inputDim: FLAT,
  },
  {
    name: "linearFlat",
    config: linearFlatConfig,
    lr: 0.1,
    n: 60,
    inputDim: FLAT,
  },
  {
    name: "linearBag_lr0.1",
    config: linearBagConfig,
    lr: 0.1,
    n: 60,
    inputDim: V,
    transform: (s) => ({ x: bagOfTokens(s.x), y: s.y }),
  },
  {
    name: "linearBag_lr0.5",
    config: linearBagConfig,
    lr: 0.5,
    n: 60,
    inputDim: V,
    transform: (s) => ({ x: bagOfTokens(s.x), y: s.y }),
  },
  {
    name: "denseFlat",
    config: denseFlatConfig,
    lr: 0.12,
    n: 60,
    inputDim: FLAT,
  },
];

// ── histogram + summary ────────────────────────────────────────────────────

function histogram(accs: number[], bucket = BUCKET): string[] {
  const lines: string[] = [];
  // Bucket keys: floor(acc / bucket) * bucket; cover [0, 1]
  const nBuckets = Math.round(1 / bucket);
  const counts = new Array(nBuckets + 1).fill(0) as number[];
  for (const a of accs) {
    const idx = Math.min(nBuckets, Math.floor(a / bucket + 1e-12));
    counts[idx]!++;
  }
  for (let i = 0; i <= nBuckets; i++) {
    const lo = i * bucket;
    const hi = Math.min(1, (i + 1) * bucket);
    const c = counts[i]!;
    if (c === 0 && (lo > 1 || hi < 0)) continue;
    // Always show buckets that have counts; also show full range summary style
    if (c === 0) continue;
    const bar = "#".repeat(Math.min(60, c));
    lines.push(
      `  [${fmt(lo, 2)}, ${fmt(hi, 2)}${i === nBuckets ? "]" : ")"}  n=${String(c).padStart(3)}  ${bar}`
    );
  }
  return lines;
}

function fraction(accs: number[], pred: (a: number) => boolean): number {
  return accs.filter(pred).length / accs.length;
}

function printSummary(name: string, accs: number[]): void {
  const sorted = accs.slice().sort((a, b) => a - b);
  console.log("");
  console.log("=".repeat(72));
  console.log(`CONFIG: ${name}`);
  console.log("=".repeat(72));
  console.log(
    `n=${accs.length}  mean=${fmt(mean(accs))}  std=${fmt(std(accs))}  min=${fmt(Math.min(...accs))}  max=${fmt(Math.max(...accs))}`
  );
  console.log("histogram (bucket=0.05):");
  for (const line of histogram(accs)) console.log(line);
  console.log(
    `frac >=0.9: ${fmt(fraction(accs, (a) => a >= 0.9))}  ` +
      `>=0.85: ${fmt(fraction(accs, (a) => a >= 0.85))}  ` +
      `>=0.75: ${fmt(fraction(accs, (a) => a >= 0.75))}  ` +
      `<=0.6: ${fmt(fraction(accs, (a) => a <= 0.6))}`
  );
  if (name === "attention") {
    const n90 = accs.filter((a) => a >= 0.9).length;
    const n75 = accs.filter((a) => a >= 0.75).length;
    console.log(
      `empirical P(acc>=0.9) = ${n90}/${accs.length} = ${fmt(n90 / accs.length)}`
    );
    console.log(
      `empirical P(acc>=0.75) = ${n75}/${accs.length} = ${fmt(n75 / accs.length)}`
    );
  }
  console.log("sorted per-run accuracies:");
  console.log(JSON.stringify(sorted.map((x) => +x.toFixed(6))));
}

// ── bootstrap: P(assertion fails) ──────────────────────────────────────────

function bootstrapFailRates(accs: number[]): void {
  const n = accs.length;
  const rules: { key: string; fail: (sample: number[]) => boolean }[] = [
    { key: "mean<0.75", fail: (s) => mean(s) < 0.75 },
    { key: "med<0.75", fail: (s) => median(s) < 0.75 },
    { key: "max<0.9", fail: (s) => Math.max(...s) < 0.9 },
    ...([1, 2, 3, 4] as const).map((c) => ({
      key: `n>=0.9 <${c}`,
      fail: (s: number[]) => s.filter((a) => a >= 0.9).length < c,
    })),
    ...([1, 2, 3, 4] as const).map((c) => ({
      key: `n>=0.75 <${c}`,
      fail: (s: number[]) => s.filter((a) => a >= 0.75).length < c,
    })),
  ];

  const colW = 12;
  const header =
    "k".padStart(4) +
    rules.map((r) => r.key.padStart(colW)).join("");
  console.log("bootstrap P(assertion FAILS) — resample with replacement, " +
    `${BOOTSTRAP_TRIALS} trials`);
  console.log(header);
  console.log("-".repeat(header.length));

  for (const k of K_VALUES) {
    const fails = rules.map(() => 0);
    for (let t = 0; t < BOOTSTRAP_TRIALS; t++) {
      const sample: number[] = [];
      for (let i = 0; i < k; i++) {
        sample.push(accs[Math.floor(Math.random() * n)]!);
      }
      for (let r = 0; r < rules.length; r++) {
        if (rules[r]!.fail(sample)) fails[r]!++;
      }
    }
    const row =
      String(k).padStart(4) +
      fails
        .map((f) => fmtProb(f / BOOTSTRAP_TRIALS).padStart(colW))
        .join("");
    console.log(row);
  }
}

// ── main ───────────────────────────────────────────────────────────────────

const rawSamples = getDataset("negation").samples;
const outPath = process.argv[2];

console.log("negation attention accuracy measurement");
console.log("UNSEEDED (Math.random drives weight init + shuffle; no seed hook)");
console.log(`dataset samples: ${rawSamples.length}  epochs: ${EPOCHS}`);
console.log(
  `configs: ${SPECS.map((s) => `${s.name}(n=${s.n},lr=${s.lr})`).join(", ")}`
);
console.log(`bootstrap trials: ${BOOTSTRAP_TRIALS}  k in [${K_VALUES.join(", ")}]`);
console.log("");

const allResults: Record<string, number[]> = {};

for (const spec of SPECS) {
  const samples = spec.transform
    ? rawSamples.map(spec.transform)
    : rawSamples;
  const accs: number[] = [];
  const t0 = Date.now();
  process.stderr.write(
    `[${spec.name}] starting ${spec.n} runs (lr=${spec.lr}, epochs=${EPOCHS})...\n`
  );

  for (let r = 0; r < spec.n; r++) {
    const model = trainOnSamples(
      spec.config,
      samples,
      spec.lr,
      EPOCHS,
      spec.inputDim
    );
    accs.push(accuracyOn(model, samples));
    if ((r + 1) % 10 === 0 || r + 1 === spec.n) {
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      process.stderr.write(
        `[${spec.name}] ${r + 1}/${spec.n}  meansofar=${fmt(mean(accs))}  ${elapsed}s\n`
      );
    }
  }

  allResults[spec.name] = accs;
  printSummary(spec.name, accs);
  console.log("");
  bootstrapFailRates(accs);
  console.log("");
}

if (outPath) {
  fs.writeFileSync(outPath, JSON.stringify(allResults, null, 2) + "\n");
  process.stderr.write(`wrote raw arrays to ${outPath}\n`);
}

console.log("done.");
