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

function makeLinear(): Dataset {
  const samples: Sample[] = [];
  for (let i = 0; i < 40; i++) {
    const x = Math.random() * 2 - 1;
    const y = Math.random() * 2 - 1;
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

/** Two interleaving moons (classic sklearn-style toy set). */
function makeMoons(n = 80, noise = 0.08): Dataset {
  const samples: Sample[] = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    const t = (Math.PI * i) / (half - 1 || 1);
    const x = Math.cos(t) + (Math.random() - 0.5) * noise;
    const y = Math.sin(t) + (Math.random() - 0.5) * noise;
    samples.push({ x: [x, y], y: [0] });
  }
  for (let i = 0; i < n - half; i++) {
    const t = (Math.PI * i) / (n - half - 1 || 1);
    const x = 1 - Math.cos(t) + (Math.random() - 0.5) * noise;
    const y = 0.5 - Math.sin(t) + (Math.random() - 0.5) * noise;
    samples.push({ x: [x, y], y: [1] });
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

function makeCircles(n = 80, noise = 0.05, factor = 0.5): Dataset {
  const samples: Sample[] = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    const t = (2 * Math.PI * i) / half;
    samples.push({
      x: [
        Math.cos(t) + (Math.random() - 0.5) * noise,
        Math.sin(t) + (Math.random() - 0.5) * noise,
      ],
      y: [0],
    });
  }
  for (let i = 0; i < n - half; i++) {
    const t = (2 * Math.PI * i) / (n - half);
    samples.push({
      x: [
        factor * Math.cos(t) + (Math.random() - 0.5) * noise,
        factor * Math.sin(t) + (Math.random() - 0.5) * noise,
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

function makeSpiral(n = 60): Dataset {
  const samples: Sample[] = [];
  for (let i = 0; i < n; i++) {
    const r = i / n;
    const t = (4 * Math.PI * i) / n;
    samples.push({
      x: [r * Math.cos(t) + (Math.random() - 0.5) * 0.1, r * Math.sin(t)],
      y: [0],
    });
    samples.push({
      x: [
        r * Math.cos(t + Math.PI) + (Math.random() - 0.5) * 0.1,
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

/** 4×4 binary patterns — vertical vs horizontal bar (tiny image subset). */
function makeTinyImages(): Dataset {
  const samples: Sample[] = [];
  // Vertical bars class 0
  for (let col = 0; col < 4; col++) {
    const img = Array.from({ length: 16 }, () => 0);
    for (let row = 0; row < 4; row++) img[row * 4 + col] = 1;
    // slight noise variants
    samples.push({ x: img.slice(), y: [1, 0] });
    const noisy = img.map((v) => (Math.random() < 0.1 ? 1 - v : v));
    samples.push({ x: noisy, y: [1, 0] });
  }
  // Horizontal bars class 1
  for (let row = 0; row < 4; row++) {
    const img = Array.from({ length: 16 }, () => 0);
    for (let col = 0; col < 4; col++) img[row * 4 + col] = 1;
    samples.push({ x: img.slice(), y: [0, 1] });
    const noisy = img.map((v) => (Math.random() < 0.1 ? 1 - v : v));
    samples.push({ x: noisy, y: [0, 1] });
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
 * Deterministic unit noise in [-1, 1] from sample index + salt.
 * Used so noisy_* datasets never call Math.random() and stay cache-stable.
 */
function detUnit(i: number, salt: number): number {
  let h = Math.imul(i + 1, 2654435761) ^ Math.imul(salt, 1597334677);
  h = Math.imul(h ^ (h >>> 16), 2246822519);
  h = Math.imul(h ^ (h >>> 13), 3266489917);
  return ((h >>> 0) / 4294967296) * 2 - 1;
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
];
