import type { Dataset, DatasetName, Sample } from "./types";

function makeXor(): Dataset {
  const samples: Sample[] = [
    { x: [0, 0], y: [0] },
    { x: [0, 1], y: [1] },
    { x: [1, 0], y: [1] },
    { x: [1, 1], y: [0] },
  ];
  return {
    name: "xor",
    samples,
    inputShape: [2],
    outputDim: 1,
    kind: "classification",
    featureNames: ["x1", "x2"],
  };
}

function makeAnd(): Dataset {
  return {
    name: "and",
    samples: [
      { x: [0, 0], y: [0] },
      { x: [0, 1], y: [0] },
      { x: [1, 0], y: [0] },
      { x: [1, 1], y: [1] },
    ],
    inputShape: [2],
    outputDim: 1,
    kind: "classification",
    featureNames: ["x1", "x2"],
  };
}

function makeOr(): Dataset {
  return {
    name: "or",
    samples: [
      { x: [0, 0], y: [0] },
      { x: [0, 1], y: [1] },
      { x: [1, 0], y: [1] },
      { x: [1, 1], y: [1] },
    ],
    inputShape: [2],
    outputDim: 1,
    kind: "classification",
    featureNames: ["x1", "x2"],
  };
}

/**
 * Deterministic unit noise in [-1, 1] from sample index + salt.
 * Used so toy / noisy_* / image datasets never call Math.random() and stay
 * cache-stable across processes.
 */
function detUnit(i: number, salt: number): number {
  let h = Math.imul(i + 1, 2654435761) ^ Math.imul(salt, 1597334677);
  h = Math.imul(h ^ (h >>> 16), 2246822519);
  h = Math.imul(h ^ (h >>> 13), 3266489917);
  return ((h >>> 0) / 4294967296) * 2 - 1;
}

/**
 * Half-unit noise in [-0.5, 0.5] — matches the support of the former
 * `(Math.random() - 0.5)` jitter so geometry/spread stay the same after
 * the deterministic conversion.
 */
function detHalf(i: number, salt: number): number {
  return detUnit(i, salt) * 0.5;
}

/**
 * Linearly separable 2-D cloud. Fully deterministic (no Math.random).
 * Points are uniform in the square via detUnit; label is sign of x+y.
 */
function makeLinear(): Dataset {
  const samples: Sample[] = [];
  for (let i = 0; i < 40; i++) {
    // salts 5/9 yield a balanced 20/20 label split on n=40
    const x = detUnit(i, 5);
    const y = detUnit(i, 9);
    samples.push({ x: [x, y], y: [x + y > 0 ? 1 : 0] });
  }
  return {
    name: "linear",
    samples,
    inputShape: [2],
    outputDim: 1,
    kind: "classification",
    featureNames: ["x", "y"],
  };
}

/**
 * Two interleaving moons (classic sklearn-style toy set).
 * Positional jitter uses detHalf so the spread matches the former
 * `(Math.random()-0.5)*noise` draws. Fully deterministic.
 */
function makeMoons(n = 80, noise = 0.08): Dataset {
  const samples: Sample[] = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    const t = (Math.PI * i) / (half - 1 || 1);
    samples.push({
      x: [
        Math.cos(t) + detHalf(i, 11) * noise,
        Math.sin(t) + detHalf(i, 22) * noise,
      ],
      y: [0],
    });
  }
  for (let i = 0; i < n - half; i++) {
    const t = (Math.PI * i) / (n - half - 1 || 1);
    samples.push({
      x: [
        1 - Math.cos(t) + detHalf(i + half, 11) * noise,
        0.5 - Math.sin(t) + detHalf(i + half, 22) * noise,
      ],
      y: [1],
    });
  }
  return {
    name: "moons",
    samples,
    inputShape: [2],
    outputDim: 1,
    kind: "classification",
    featureNames: ["x", "y"],
  };
}

/**
 * Concentric circles. Jitter via detHalf (same magnitude as the former
 * Math.random draws). Fully deterministic.
 */
function makeCircles(n = 80, noise = 0.05, factor = 0.5): Dataset {
  const samples: Sample[] = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    const t = (2 * Math.PI * i) / half;
    samples.push({
      x: [
        Math.cos(t) + detHalf(i, 13) * noise,
        Math.sin(t) + detHalf(i, 17) * noise,
      ],
      y: [0],
    });
  }
  for (let i = 0; i < n - half; i++) {
    const t = (2 * Math.PI * i) / (n - half);
    samples.push({
      x: [
        factor * Math.cos(t) + detHalf(i + half, 13) * noise,
        factor * Math.sin(t) + detHalf(i + half, 17) * noise,
      ],
      y: [1],
    });
  }
  return {
    name: "circles",
    samples,
    inputShape: [2],
    outputDim: 1,
    kind: "classification",
    featureNames: ["x", "y"],
  };
}

/**
 * Two interleaved spirals. Radial noise on x only (same as before), via
 * detHalf so magnitude matches `(Math.random()-0.5)*0.1`. Fully deterministic.
 */
function makeSpiral(n = 60): Dataset {
  const samples: Sample[] = [];
  const spiralNoise = 0.1;
  for (let i = 0; i < n; i++) {
    const r = i / n;
    const t = (4 * Math.PI * i) / n;
    samples.push({
      x: [r * Math.cos(t) + detHalf(i, 19) * spiralNoise, r * Math.sin(t)],
      y: [0],
    });
    samples.push({
      x: [
        r * Math.cos(t + Math.PI) + detHalf(i + n, 19) * spiralNoise,
        r * Math.sin(t + Math.PI),
      ],
      y: [1],
    });
  }
  return {
    name: "spiral",
    samples,
    inputShape: [2],
    outputDim: 1,
    kind: "classification",
    featureNames: ["x", "y"],
  };
}

/** Map detUnit → [0, 1) for Bernoulli-style flips. */
function detUnit01(i: number, salt: number): number {
  return (detUnit(i, salt) + 1) / 2;
}

/**
 * 4×4 binary patterns — vertical vs horizontal bar (tiny image subset).
 * Clean bars are fixed; noisy variants use detUnit (no Math.random).
 */
function makeTinyImages(): Dataset {
  const samples: Sample[] = [];
  let sampleIdx = 0;
  // Vertical bars class 0
  for (let col = 0; col < 4; col++) {
    const img = Array.from({ length: 16 }, () => 0);
    for (let row = 0; row < 4; row++) img[row * 4 + col] = 1;
    // slight noise variants
    samples.push({ x: img.slice(), y: [1, 0] });
    sampleIdx++;
    const noisy = img.map((v, pi) =>
      detUnit01(sampleIdx * 16 + pi, 31) < 0.1 ? 1 - v : v
    );
    samples.push({ x: noisy, y: [1, 0] });
    sampleIdx++;
  }
  // Horizontal bars class 1
  for (let row = 0; row < 4; row++) {
    const img = Array.from({ length: 16 }, () => 0);
    for (let col = 0; col < 4; col++) img[row * 4 + col] = 1;
    samples.push({ x: img.slice(), y: [0, 1] });
    sampleIdx++;
    const noisy = img.map((v, pi) =>
      detUnit01(sampleIdx * 16 + pi, 31) < 0.1 ? 1 - v : v
    );
    samples.push({ x: noisy, y: [0, 1] });
    sampleIdx++;
  }
  return {
    name: "tiny_images",
    samples,
    inputShape: [1, 4, 4], // C,H,W
    outputDim: 2,
    kind: "classification",
  };
}

/**
 * 8×8 images: a short vertical bar vs a short horizontal bar at many
 * translations. Deterministic (no Math.random).
 *
 * Motivation: `tiny_images` exhausts every bar position in 16 samples, so a
 * dense net memorizes locations and weight sharing buys nothing. Here there
 * are many more placements (and mild noise copies), which is the regime
 * where sharing *should* help. With global pooling after conv2d the dense head
 * is translation-invariant and a small CNN beats a larger dense baseline on
 * held-out accuracy (see shifted_bars tests). Without pooling, flatten leaves
 * the head position-specific and dense can match by memorizing locations.
 *
 * Layout (BAR_LEN=3 on 8×8):
 *   vertical placements:   (8-3+1)×8 = 48
 *   horizontal placements: 8×(8-3+1) = 48
 *   ×2 (clean + noisy) → 192 samples, balanced 96/96.
 */
function makeShiftedBars(size = 8, barLen = 3, noiseRate = 0.04): Dataset {
  const samples: Sample[] = [];
  let sampleIdx = 0;

  const blank = (): number[] => Array.from({ length: size * size }, () => 0);

  const addNoisy = (img: number[], y: number[]): void => {
    samples.push({ x: img.slice(), y: y.slice() });
    sampleIdx++;
    const noisy = img.map((v, pi) =>
      detUnit01(sampleIdx * size * size + pi, 47) < noiseRate ? 1 - v : v
    );
    samples.push({ x: noisy, y: y.slice() });
    sampleIdx++;
  };

  // Class 0: vertical bar of length barLen
  for (let r = 0; r <= size - barLen; r++) {
    for (let c = 0; c < size; c++) {
      const img = blank();
      for (let k = 0; k < barLen; k++) img[(r + k) * size + c] = 1;
      addNoisy(img, [1, 0]);
    }
  }
  // Class 1: horizontal bar of length barLen
  for (let r = 0; r < size; r++) {
    for (let c = 0; c <= size - barLen; c++) {
      const img = blank();
      for (let k = 0; k < barLen; k++) img[r * size + (c + k)] = 1;
      addNoisy(img, [0, 1]);
    }
  }

  return {
    name: "shifted_bars",
    samples,
    inputShape: [1, size, size], // C,H,W
    outputDim: 2,
    kind: "classification",
  };
}

/**
 * Flip a deterministic subset of labels (label noise). Order is content-hash
 * based so the same geometric points always receive the same flipped labels.
 */
function flipLabelsDeterministic(samples: Sample[], flipRate: number): void {
  if (!(flipRate > 0) || samples.length === 0) return;
  const order = samples
    .map((s, i) => {
      let h = (i + 1) * 2654435761;
      for (const v of s.x) {
        h = Math.imul(h ^ (Math.floor(v * 1e6) | 0), 1597334677);
      }
      return { i, k: h >>> 0 };
    })
    .sort((a, b) => a.k - b.k || a.i - b.i);
  const nFlip = Math.min(
    samples.length,
    Math.max(0, Math.round(samples.length * flipRate))
  );
  for (let k = 0; k < nFlip; k++) {
    const idx = order[k]!.i;
    const y = samples[idx]!.y[0]!;
    samples[idx]!.y = [y >= 0.5 ? 0 : 1];
  }
}

/**
 * Two interleaving moons with mild positional jitter **and** label noise.
 *
 * Unlike plain `moons` (positional noise only, which a wide net still
 * generalises), a fixed fraction of labels are flipped. Memorising those
 * flips drives train accuracy up while held-out accuracy stalls — the
 * signature Chapter 3 needs. Fully deterministic (no Math.random).
 *
 * Measured defaults (n=44, noise=0.12, flip=0.23), 40 runs, val=0.3:
 *   64×64 l2=0  e500 lr=0.08 → train ~96%  val ~64%  gap ~32pp (min ~21pp)
 *   64×64 l2=0.005 e300      → train ~82%  val ~75%  (held-out +12pp vs above)
 *   dense 6 units            → train ~80%  val ~69%  (held-out +5pp vs above)
 */
function makeNoisyMoons(
  n = 44,
  noise = 0.12,
  flipRate = 0.23
): Dataset {
  const samples: Sample[] = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    const t = (Math.PI * i) / (half - 1 || 1);
    samples.push({
      x: [
        Math.cos(t) + detUnit(i, 11) * noise,
        Math.sin(t) + detUnit(i, 22) * noise,
      ],
      y: [0],
    });
  }
  for (let i = 0; i < n - half; i++) {
    const t = (Math.PI * i) / (n - half - 1 || 1);
    samples.push({
      x: [
        1 - Math.cos(t) + detUnit(i + half, 11) * noise,
        0.5 - Math.sin(t) + detUnit(i + half, 22) * noise,
      ],
      y: [1],
    });
  }
  flipLabelsDeterministic(samples, flipRate);
  return {
    name: "noisy_moons",
    samples,
    inputShape: [2],
    outputDim: 1,
    kind: "classification",
    featureNames: ["x", "y"],
  };
}

/**
 * Tiny text: bag-of-words style binary sentiment on fixed 8-dim one-hot features.
 * Features: [good, bad, love, hate, yes, no, great, awful]
 */
function makeTinyText(): Dataset {
  // Positions: 0 good, 1 bad, 2 love, 3 hate, 4 yes, 5 no, 6 great, 7 awful
  const samples: Sample[] = [
    { x: [1, 0, 0, 0, 0, 0, 0, 0], y: [1] }, // good → pos
    { x: [0, 1, 0, 0, 0, 0, 0, 0], y: [0] }, // bad → neg
    { x: [0, 0, 1, 0, 0, 0, 0, 0], y: [1] },
    { x: [0, 0, 0, 1, 0, 0, 0, 0], y: [0] },
    { x: [0, 0, 0, 0, 1, 0, 0, 0], y: [1] },
    { x: [0, 0, 0, 0, 0, 1, 0, 0], y: [0] },
    { x: [0, 0, 0, 0, 0, 0, 1, 0], y: [1] },
    { x: [0, 0, 0, 0, 0, 0, 0, 1], y: [0] },
    { x: [1, 0, 1, 0, 0, 0, 0, 0], y: [1] }, // good love
    { x: [0, 1, 0, 1, 0, 0, 0, 0], y: [0] }, // bad hate
    { x: [1, 0, 0, 0, 1, 0, 0, 0], y: [1] },
    { x: [0, 1, 0, 0, 0, 1, 0, 0], y: [0] },
    { x: [0, 0, 0, 0, 0, 0, 1, 1], y: [0.5] }, // mixed — treat as soft
    { x: [1, 1, 0, 0, 0, 0, 0, 0], y: [0.5] },
  ];
  // Harden soft labels for classification accuracy
  const hardened = samples.map((s) => ({
    x: s.x,
    y: [s.y[0]! >= 0.5 ? 1 : 0],
  }));
  return {
    name: "tiny_text",
    samples: hardened,
    inputShape: [8],
    outputDim: 1,
    kind: "classification",
  };
}

/**
 * Multi-token negation sequences for Chapter 5.
 *
 * Vocabulary (one-hot, d_model = 4): PAD=0, NOT=1, GOOD=2, BAD=3.
 * Sequence length = 4. Flat input length = 16; with `attention`/`transformer`
 * `d_model=4` the engine yields 4 tokens.
 *
 * Label rule (multiset XOR / compositional negation):
 *   - Exactly one sentiment token (GOOD or BAD) per sample.
 *   - Base polarity: GOOD → 1, BAD → 0.
 *   - If NOT appears anywhere in the sequence, flip polarity.
 *   - So: good → +, not good → −, bad → −, not bad → +.
 *
 * Why bag-of-words (linear) fails:
 *   Count vectors form an XOR over {NOT present} × {sentiment}. A linear
 *   classifier on the bag can get at most 3 of the 4 bag-types right; with
 *   sample weights (4 bare GOOD, 4 bare BAD, 12 NOT+GOOD, 12 NOT+BAD) the
 *   hard ceiling is 28/32 = 0.875 (drop a size-4 class). Empirically linear
 *   plateaus near ~0.6–0.7.
 *
 * Why position-tied dense fails held-out positions:
 *   Flattening 4×4 one-hots gives position-specific weights. Holding out all
 *   sequences that use slot 3, a large MLP memorizes train slots (~100% train)
 *   and collapses on test (~30%). Content-based attention transfers.
 *
 * Sample count: 4 bare GOOD + 4 bare BAD + 12 NOT+GOOD + 12 NOT+BAD = 32,
 * balanced 16/16. Fully deterministic (no Math.random).
 */
export const NEGATION_VOCAB = ["PAD", "NOT", "GOOD", "BAD"] as const;
export const NEGATION_SEQ_LEN = 4;
export const NEGATION_D_MODEL = 4;
const NEG_PAD = 0;
const NEG_NOT = 1;
const NEG_GOOD = 2;
const NEG_BAD = 3;

function negationOneHot(id: number): number[] {
  const v = Array.from({ length: NEGATION_D_MODEL }, () => 0);
  v[id] = 1;
  return v;
}

function negationEncode(ids: number[]): number[] {
  return ids.flatMap(negationOneHot);
}

function makeNegation(): Dataset {
  const samples: Sample[] = [];

  // Bare sentiment at each position
  for (let p = 0; p < NEGATION_SEQ_LEN; p++) {
    const good = Array.from({ length: NEGATION_SEQ_LEN }, () => NEG_PAD);
    good[p] = NEG_GOOD;
    samples.push({ x: negationEncode(good), y: [1] });
    const bad = Array.from({ length: NEGATION_SEQ_LEN }, () => NEG_PAD);
    bad[p] = NEG_BAD;
    samples.push({ x: negationEncode(bad), y: [0] });
  }

  // NOT + sentiment at all ordered pairs of distinct positions
  for (let i = 0; i < NEGATION_SEQ_LEN; i++) {
    for (let j = 0; j < NEGATION_SEQ_LEN; j++) {
      if (i === j) continue;
      const ng = Array.from({ length: NEGATION_SEQ_LEN }, () => NEG_PAD);
      ng[i] = NEG_NOT;
      ng[j] = NEG_GOOD;
      samples.push({ x: negationEncode(ng), y: [0] });
      const nb = Array.from({ length: NEGATION_SEQ_LEN }, () => NEG_PAD);
      nb[i] = NEG_NOT;
      nb[j] = NEG_BAD;
      samples.push({ x: negationEncode(nb), y: [1] });
    }
  }

  return {
    name: "negation",
    samples,
    inputShape: [NEGATION_SEQ_LEN * NEGATION_D_MODEL],
    outputDim: 1,
    kind: "classification",
  };
}

const CACHE: Partial<Record<DatasetName, Dataset>> = {};

export function getDataset(name: DatasetName, seedRefresh = false): Dataset {
  if (!seedRefresh && CACHE[name]) return CACHE[name]!;
  let ds: Dataset;
  switch (name) {
    case "xor":
      ds = makeXor();
      break;
    case "and":
      ds = makeAnd();
      break;
    case "or":
      ds = makeOr();
      break;
    case "linear":
      ds = makeLinear();
      break;
    case "moons":
      ds = makeMoons();
      break;
    case "circles":
      ds = makeCircles();
      break;
    case "spiral":
      ds = makeSpiral();
      break;
    case "tiny_images":
      ds = makeTinyImages();
      break;
    case "tiny_text":
      ds = makeTinyText();
      break;
    case "noisy_moons":
      ds = makeNoisyMoons();
      break;
    case "shifted_bars":
      ds = makeShiftedBars();
      break;
    case "negation":
      ds = makeNegation();
      break;
    default:
      ds = makeXor();
  }
  CACHE[name] = ds;
  return ds;
}

export const DATASET_NAMES: DatasetName[] = [
  "xor",
  "and",
  "or",
  "linear",
  "moons",
  "circles",
  "spiral",
  "tiny_images",
  "tiny_text",
  "noisy_moons",
  "shifted_bars",
  "negation",
];
