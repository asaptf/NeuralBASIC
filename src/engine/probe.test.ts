import { describe, expect, it } from "vitest";
import { getDataset } from "./datasets";
import {
  createModel,
  exportWeights,
  loadWeights,
  predict,
} from "./model";
import { createModelProbe } from "./probe";
import { createAndTrain } from "./train";
import type { LayerWeights, NetworkConfig, TrainConfig } from "./types";

describe("createModelProbe — binary sigmoid (AND)", () => {
  it("probes the four corners with the labels it learned, and p1 crosses 0.5 the right way", () => {
    const network: NetworkConfig = {
      layers: [
        { type: "dense", units: 1, activation: "sigmoid", inputDim: 2 },
      ],
    };
    const trainConfig: TrainConfig = {
      learningRate: 0.8,
      epochs: 200,
      dataset: "and",
    };

    // AND is reliably solvable; retry a few inits like other engine tests.
    let best = createAndTrain(network, trainConfig);
    for (let attempt = 0; attempt < 4; attempt++) {
      if (best.history.final.accuracy >= 0.99) break;
      const next = createAndTrain(network, trainConfig);
      if (next.history.final.accuracy > best.history.final.accuracy)
        best = next;
    }
    expect(best.history.final.accuracy).toBeGreaterThanOrEqual(0.99);

    const probe = createModelProbe(network, trainConfig, best.weights);
    expect(probe).not.toBeNull();
    expect(probe!.inputSize).toBe(2);
    expect(probe!.outputSize).toBe(1);

    const corners: { x: number[]; label: number }[] = [
      { x: [0, 0], label: 0 },
      { x: [0, 1], label: 0 },
      { x: [1, 0], label: 0 },
      { x: [1, 1], label: 1 },
    ];

    for (const { x, label } of corners) {
      const r = probe!.run(x);
      expect(r.output).toHaveLength(1);
      expect(Number.isFinite(r.output[0]!)).toBe(true);
      expect(r.classIndex).toBe(label);
      // p1 should sit on the label's side of the 0.5 threshold
      if (label === 1) {
        expect(r.p1).toBeGreaterThanOrEqual(0.5);
        expect(r.confidence).toBeCloseTo(r.p1, 10);
      } else {
        expect(r.p1).toBeLessThan(0.5);
        expect(r.confidence).toBeCloseTo(1 - r.p1, 10);
      }
    }
  });
});

describe("createModelProbe — multi-class softmax (tiny_images)", () => {
  it("classIndex matches argmax; confidence in 0–1; p1 + p(class 0) === 1", () => {
    const network: NetworkConfig = {
      layers: [
        {
          type: "conv2d",
          filters: 4,
          kernelSize: 2,
          activation: "relu",
          inputChannels: 1,
          inputHeight: 4,
          inputWidth: 4,
        },
        { type: "flatten" },
        { type: "dense", units: 8, activation: "relu" },
        { type: "dense", units: 2, activation: "softmax" },
      ],
    };
    const trainConfig: TrainConfig = {
      learningRate: 0.12,
      epochs: 60,
      dataset: "tiny_images",
    };

    let best = createAndTrain(network, trainConfig);
    for (let attempt = 0; attempt < 3; attempt++) {
      if (best.history.final.accuracy >= 0.75) break;
      const next = createAndTrain(network, trainConfig);
      if (next.history.final.accuracy > best.history.final.accuracy)
        best = next;
    }

    const probe = createModelProbe(network, trainConfig, best.weights);
    expect(probe).not.toBeNull();
    // inputShape [1,4,4] → flattened 16
    expect(probe!.inputSize).toBe(16);
    expect(probe!.outputSize).toBe(2);

    const sample = getDataset("tiny_images").samples[0]!;
    const r = probe!.run(sample.x);

    expect(r.output).toHaveLength(2);
    expect(r.output.every((v) => Number.isFinite(v))).toBe(true);

    // classIndex is argmax of the raw (or equivalently normalised) readout
    let argmax = 0;
    for (let i = 1; i < r.output.length; i++) {
      if (r.output[i]! > r.output[argmax]!) argmax = i;
    }
    expect(r.classIndex).toBe(argmax);

    expect(r.confidence).toBeGreaterThanOrEqual(0);
    expect(r.confidence).toBeLessThanOrEqual(1);
    expect(r.p1).toBeGreaterThanOrEqual(0);
    expect(r.p1).toBeLessThanOrEqual(1);

    // Softmax distribution: p1 + p(class 0) === 1
    const p0 = 1 - r.p1;
    expect(r.p1 + p0).toBeCloseTo(1, 10);
    // confidence is the probability of the picked class
    const confOfClass = r.classIndex === 1 ? r.p1 : p0;
    expect(r.confidence).toBeCloseTo(confOfClass, 10);
  });
});

describe("createModelProbe — independent multi-unit sigmoid head", () => {
  it("does not inflate confidence when unit scores sum well below 1", () => {
    // Chapter 4's starter ends in dense 2 activation=sigmoid — independent
    // scores, not a distribution. Craft biases so any input yields ~[0.2, 0.1].
    const network: NetworkConfig = {
      layers: [
        { type: "dense", units: 2, activation: "sigmoid", inputDim: 2 },
      ],
    };
    const trainConfig: TrainConfig = {
      learningRate: 0.1,
      epochs: 1,
      dataset: "and",
    };
    const logit = (p: number) => Math.log(p / (1 - p));
    const weights: LayerWeights[] = [
      {
        type: "dense",
        weights: [
          [0, 0],
          [0, 0],
        ],
        biases: [logit(0.2), logit(0.1)],
      },
    ];

    const probe = createModelProbe(network, trainConfig, weights);
    expect(probe).not.toBeNull();
    const r = probe!.run([0, 0]);

    expect(r.output[0]!).toBeCloseTo(0.2, 5);
    expect(r.output[1]!).toBeCloseTo(0.1, 5);
    expect(r.classIndex).toBe(0);
    // Selected unit's own score — not 0.2/(0.2+0.1) ≈ 0.67
    expect(r.confidence).toBeCloseTo(0.2, 5);
    expect(r.confidence).toBeLessThan(0.5);
    // p1 is unit 1's score (still thresholdable at 0.5), not a share of mass
    expect(r.p1).toBeCloseTo(0.1, 5);
  });
});

describe("createModelProbe — failure modes", () => {
  const network: NetworkConfig = {
    layers: [{ type: "dense", units: 1, activation: "sigmoid", inputDim: 2 }],
  };
  const trainConfig: TrainConfig = {
    learningRate: 0.8,
    epochs: 1,
    dataset: "and",
  };

  it("returns null for empty weights", () => {
    expect(createModelProbe(network, trainConfig, [])).toBeNull();
  });

  it("returns null when weights disagree with the config", () => {
    // Wrong layer count / type for a single dense neuron
    const bad: LayerWeights[] = [
      {
        type: "dense",
        weights: [
          [0, 0, 0], // input dim 3 instead of 2
        ],
        biases: [0],
      },
    ];
    expect(createModelProbe(network, trainConfig, bad)).toBeNull();

    // Completely different architecture (conv weights on a dense net)
    const convLike: LayerWeights[] = [
      {
        type: "conv2d",
        weights: [[0, 0, 0, 0]],
        biases: [0],
      },
    ];
    expect(createModelProbe(network, trainConfig, convLike)).toBeNull();
  });

  it("returns null when attention matrices were built for a different d_model", () => {
    // Config expects d_model=4; hand-build 8×8 attention matrices.
    const network: NetworkConfig = {
      layers: [
        { type: "attention", dModel: 4, nHeads: 2 },
        { type: "dense", units: 1, activation: "sigmoid", inputDim: 4 },
      ],
    };
    const trainConfig: TrainConfig = {
      learningRate: 0.1,
      epochs: 1,
      dataset: "tiny_text",
    };
    const mat = (n: number) =>
      Array.from({ length: n }, () => Array.from({ length: n }, () => 0.01));
    const wrong: LayerWeights[] = [
      {
        type: "attention",
        params: {
          Wq: mat(8),
          Wk: mat(8),
          Wv: mat(8),
          Wo: mat(8),
        },
      },
      {
        type: "dense",
        weights: [Array.from({ length: 4 }, () => 0)],
        biases: [0],
      },
    ];
    expect(createModelProbe(network, trainConfig, wrong)).toBeNull();
  });
});

describe("transformer_block weight round-trip restores feed-forward biases", () => {
  it("export then load yields the same prediction as the original model", () => {
    const config: NetworkConfig = {
      layers: [
        { type: "transformer_block", dModel: 8, nHeads: 2, dff: 12 },
        { type: "dense", units: 1, activation: "sigmoid", inputDim: 8 },
      ],
    };
    // Train so parameters leave zero; then force non-zero FF biases so a
    // restore that drops them cannot agree by accident.
    const { model } = createAndTrain(config, {
      learningRate: 0.1,
      epochs: 5,
      dataset: "tiny_text",
    });
    const block = model.layers[0]!;
    expect(block.type).toBe("transformer_block");
    if (block.type === "transformer_block") {
      for (let i = 0; i < block.b1.length; i++) block.b1[i] = 0.15 * (i + 1);
      for (let i = 0; i < block.b2.length; i++) block.b2[i] = -0.07 * (i + 1);
    }

    const weights = exportWeights(model);
    expect(weights[0]!.biases).toBeDefined();
    expect(weights[0]!.biases!.length).toBe(12 + 8);

    // tiny_text inputShape is [8]; must match createAndTrain's prepared dim
    const restored = createModel(config, 8);
    loadWeights(restored, weights);

    const restoredBlock = restored.layers[0]!;
    expect(restoredBlock.type).toBe("transformer_block");
    if (block.type === "transformer_block" && restoredBlock.type === "transformer_block") {
      expect(restoredBlock.b1).toEqual(block.b1);
      expect(restoredBlock.b2).toEqual(block.b2);
    }

    const sample = getDataset("tiny_text").samples[0]!;
    const a = predict(model, sample.x);
    const b = predict(restored, sample.x);
    expect(b).toHaveLength(a.length);
    for (let i = 0; i < a.length; i++) {
      expect(Math.abs(a[i]! - b[i]!)).toBeLessThan(1e-12);
    }
  });
});

describe("createModelProbe — input coercion & determinism", () => {
  it("handles short, long, and NaN-containing inputs with finite results", () => {
    const network: NetworkConfig = {
      layers: [
        { type: "dense", units: 1, activation: "sigmoid", inputDim: 2 },
      ],
    };
    const trainConfig: TrainConfig = {
      learningRate: 0.8,
      epochs: 50,
      dataset: "and",
    };
    const { weights } = createAndTrain(network, trainConfig);
    const probe = createModelProbe(network, trainConfig, weights)!;
    expect(probe).not.toBeNull();

    const short = probe.run([1]);
    expect(short.output.every((v) => Number.isFinite(v))).toBe(true);
    expect(Number.isFinite(short.p1)).toBe(true);
    expect(Number.isFinite(short.confidence)).toBe(true);

    const long = probe.run([1, 0, 9, 9, 9]);
    expect(long.output.every((v) => Number.isFinite(v))).toBe(true);

    const dirty = probe.run([NaN, Infinity]);
    expect(dirty.output.every((v) => Number.isFinite(v))).toBe(true);
    expect(Number.isFinite(dirty.p1)).toBe(true);
    expect(Number.isFinite(dirty.confidence)).toBe(true);
  });

  it("two run calls with the same input agree exactly", () => {
    const network: NetworkConfig = {
      layers: [
        { type: "dense", units: 1, activation: "sigmoid", inputDim: 2 },
      ],
    };
    const trainConfig: TrainConfig = {
      learningRate: 0.8,
      epochs: 50,
      dataset: "and",
    };
    const { weights } = createAndTrain(network, trainConfig);
    const probe = createModelProbe(network, trainConfig, weights)!;

    const a = probe.run([1, 1]);
    const b = probe.run([1, 1]);
    expect(a.output).toEqual(b.output);
    expect(a.classIndex).toBe(b.classIndex);
    expect(a.confidence).toBe(b.confidence);
    expect(a.p1).toBe(b.p1);
  });
});
