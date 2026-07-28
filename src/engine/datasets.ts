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
];
