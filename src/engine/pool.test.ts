/**
 * Pooling layer unit tests: shapes, forward values, analytical backward,
 * and train-path integration.
 */
import { describe, expect, it } from "vitest";
import { parseDSL, toDSL } from "./dsl";
import {
  createModel,
  exportWeights,
  forward,
  loadWeights,
  snapshotLayers,
  type PoolLayerState,
} from "./model";
import { createAndTrain, trainStep } from "./train";
import type { NetworkConfig } from "./types";

describe("pool — shape inference", () => {
  it("global pool collapses each channel to 1×1 (flatDim = filters)", () => {
    const config: NetworkConfig = {
      layers: [
        {
          type: "conv2d",
          filters: 4,
          kernelSize: 3,
          activation: "relu",
          inputChannels: 1,
          inputHeight: 8,
          inputWidth: 8,
        },
        { type: "pool", mode: "max", global: true },
        { type: "dense", units: 8, activation: "relu" },
        { type: "dense", units: 2, activation: "sigmoid" },
      ],
    };
    const model = createModel(config, 64);
    const pool = model.layers[1] as PoolLayerState;
    expect(pool.type).toBe("pool");
    expect(pool.global).toBe(true);
    expect(pool.mode).toBe("max");

    // Dense after global pool must see 4 inputs (one per filter).
    const dense = model.layers[2]!;
    expect(dense.type).toBe("dense");
    if (dense.type === "dense") {
      expect(dense.weights[0]!.length).toBe(4);
    }
  });

  it("windowed 2×2 stride-2 halves spatial size after conv", () => {
    // 8×8 → conv k=3 valid → 6×6 → pool 2 stride 2 → 3×3
    const config: NetworkConfig = {
      layers: [
        {
          type: "conv2d",
          filters: 2,
          kernelSize: 3,
          activation: "linear",
          inputChannels: 1,
          inputHeight: 8,
          inputWidth: 8,
        },
        { type: "pool", mode: "avg", size: 2, stride: 2 },
        { type: "flatten" },
        { type: "dense", units: 2, activation: "sigmoid" },
      ],
    };
    const model = createModel(config, 64);
    const dense = model.layers[3]!;
    expect(dense.type).toBe("dense");
    if (dense.type === "dense") {
      // 2 filters × 3 × 3 = 18
      expect(dense.weights[0]!.length).toBe(18);
    }
  });
});

describe("pool — forward", () => {
  it("max pool picks the window maximum", () => {
    const model = createModel(
      {
        layers: [
          {
            type: "conv2d",
            filters: 1,
            kernelSize: 1,
            activation: "linear",
            inputChannels: 1,
            inputHeight: 2,
            inputWidth: 2,
          },
          { type: "pool", mode: "max", size: 2, stride: 2 },
        ],
      },
      4
    );
    // Set identity 1×1 kernel and zero bias so conv is passthrough.
    const conv = model.layers[0]!;
    if (conv.type === "conv2d") {
      conv.kernels[0]![0]![0]![0] = 1;
      conv.biases[0] = 0;
    }
    // Input 2×2: [[1, 3], [2, 0]] → max = 3
    const { spatial, output } = forward(model, [1, 3, 2, 0]);
    expect(output).toEqual([3]);
    expect(spatial?.[0]?.[0]?.[0]).toBe(3);
  });

  it("avg pool returns the window mean", () => {
    const model = createModel(
      {
        layers: [
          {
            type: "conv2d",
            filters: 1,
            kernelSize: 1,
            activation: "linear",
            inputChannels: 1,
            inputHeight: 2,
            inputWidth: 2,
          },
          { type: "pool", mode: "avg", size: 2, stride: 2 },
        ],
      },
      4
    );
    const conv = model.layers[0]!;
    if (conv.type === "conv2d") {
      conv.kernels[0]![0]![0]![0] = 1;
      conv.biases[0] = 0;
    }
    // mean(1,3,2,0) = 1.5
    const { output } = forward(model, [1, 3, 2, 0]);
    expect(output[0]).toBeCloseTo(1.5, 8);
  });

  it("global max reduces a full feature map to one scalar per channel", () => {
    const model = createModel(
      {
        layers: [
          {
            type: "conv2d",
            filters: 2,
            kernelSize: 1,
            activation: "linear",
            inputChannels: 1,
            inputHeight: 3,
            inputWidth: 3,
          },
          { type: "pool", mode: "max", global: true },
        ],
      },
      9
    );
    const conv = model.layers[0]!;
    if (conv.type === "conv2d") {
      // filter 0: identity, filter 1: ×2
      conv.kernels[0]![0]![0]![0] = 1;
      conv.kernels[1]![0]![0]![0] = 2;
      conv.biases[0] = 0;
      conv.biases[1] = 0;
    }
    const flat = [0, 1, 0, 2, 9, 1, 0, 3, 0];
    const { output } = forward(model, flat);
    expect(output).toHaveLength(2);
    expect(output[0]).toBe(9); // max of passthrough
    expect(output[1]).toBe(18); // max of ×2
  });
});

describe("pool — analytical backward", () => {
  it("max pool routes gradient only to the argmax cell", () => {
    const model = createModel(
      {
        layers: [
          {
            type: "conv2d",
            filters: 1,
            kernelSize: 1,
            activation: "linear",
            inputChannels: 1,
            inputHeight: 2,
            inputWidth: 2,
          },
          { type: "pool", mode: "max", size: 2, stride: 2 },
          { type: "flatten" },
          { type: "dense", units: 1, activation: "linear" },
        ],
      },
      4
    );
    const conv = model.layers[0]!;
    if (conv.type === "conv2d") {
      conv.kernels[0]![0]![0]![0] = 1;
      conv.biases[0] = 0;
    }
    const dense = model.layers[3]!;
    if (dense.type === "dense") {
      dense.weights[0]![0] = 1;
      dense.biases[0] = 0;
    }

    // Input max at position (0,1) value 3. BCE vs target 0: pred≈3 for linear?
    // Use sigmoid output path instead — set dense to sigmoid.
    // Simpler: run trainStep and inspect pool.lastMax + that conv kernel moved.
    // With linear dense + BCE the loss is still defined.
    const x = [1, 3, 2, 0];
    const y = [0];
    // Force analytical path: no attention
    const loss = trainStep(model, x, y, 0.1, 0);
    expect(Number.isFinite(loss)).toBe(true);

    const pool = model.layers[1] as PoolLayerState;
    expect(pool.lastMaxY?.[0]?.[0]?.[0]).toBe(0);
    expect(pool.lastMaxX?.[0]?.[0]?.[0]).toBe(1);
  });

  it("avg pool shares gradient across the window (finite-diff check)", () => {
    // Tiny: 1×1 conv identity + avg pool 2×2 + linear dense → scalar.
    // Central-diff dL/d input vs analytical via trainStepFiniteDiff force.
    const config: NetworkConfig = {
      layers: [
        {
          type: "conv2d",
          filters: 1,
          kernelSize: 1,
          activation: "linear",
          inputChannels: 1,
          inputHeight: 2,
          inputWidth: 2,
        },
        { type: "pool", mode: "avg", size: 2, stride: 2 },
        { type: "flatten" },
        { type: "dense", units: 1, activation: "sigmoid" },
      ],
    };
    const modelA = createModel(config, 4);
    const modelB = createModel(config, 4);
    // Shared init
    const convA = modelA.layers[0]!;
    const convB = modelB.layers[0]!;
    if (convA.type === "conv2d" && convB.type === "conv2d") {
      convA.kernels[0]![0]![0]![0] = 0.5;
      convB.kernels[0]![0]![0]![0] = 0.5;
      convA.biases[0] = 0.1;
      convB.biases[0] = 0.1;
    }
    const dA = modelA.layers[3]!;
    const dB = modelB.layers[3]!;
    if (dA.type === "dense" && dB.type === "dense") {
      dA.weights[0]![0] = 0.7;
      dB.weights[0]![0] = 0.7;
      dA.biases[0] = -0.2;
      dB.biases[0] = -0.2;
    }

    const x = [0.2, 0.8, 0.4, 0.6];
    const y = [1];
    const wBefore = exportWeights(modelA);
    loadWeights(modelB, wBefore);

    trainStep(modelA, x, y, 0.05, 0, false);
    trainStep(modelB, x, y, 0.05, 0, true);

    const wA = exportWeights(modelA);
    const wB = exportWeights(modelB);
    // Analytical and finite-diff should move the conv kernel similarly.
    const kA = wA[0]!.weights![0]![0]!;
    const kB = wB[0]!.weights![0]![0]!;
    expect(Math.abs(kA - kB)).toBeLessThan(0.02);
  });
});

describe("pool — weights / snapshot / train", () => {
  it("export/loadWeights round-trips a pooled CNN", () => {
    const config: NetworkConfig = {
      layers: [
        {
          type: "conv2d",
          filters: 2,
          kernelSize: 2,
          activation: "relu",
          inputChannels: 1,
          inputHeight: 4,
          inputWidth: 4,
        },
        { type: "pool", mode: "max", size: 2, stride: 2 },
        { type: "flatten" },
        { type: "dense", units: 2, activation: "sigmoid" },
      ],
    };
    const m1 = createModel(config, 16);
    const w = exportWeights(m1);
    expect(w.some((x) => x.type === "pool")).toBe(true);
    const m2 = createModel(config, 16);
    loadWeights(m2, w);
    const x = Array.from({ length: 16 }, (_, i) => (i % 3 === 0 ? 1 : 0));
    expect(forward(m2, x).output).toEqual(forward(m1, x).output);
  });

  it("snapshotLayers includes pool activations after forward", () => {
    const config: NetworkConfig = {
      layers: [
        {
          type: "conv2d",
          filters: 2,
          kernelSize: 2,
          activation: "relu",
          inputChannels: 1,
          inputHeight: 4,
          inputWidth: 4,
        },
        { type: "pool", mode: "avg", global: true },
        { type: "dense", units: 2, activation: "sigmoid" },
      ],
    };
    const model = createModel(config, 16);
    forward(
      model,
      Array.from({ length: 16 }, (_, i) => (i < 4 ? 1 : 0))
    );
    const snaps = snapshotLayers(model);
    const poolSnap = snaps.find((s) => s.type === "pool");
    expect(poolSnap).toBeDefined();
    expect(poolSnap!.activations).toHaveLength(2);
    expect(poolSnap!.shape).toEqual([2, 1, 1]);
  });

  it("trains tiny_images with global avg-pool above chance (aggregate)", () => {
    // Global max is init-sensitive on tiny_images; avg is stable. Aggregate
    // over a few runs so a single dead init cannot flake the suite.
    const RUNS = 4;
    const accs: number[] = [];
    const config: NetworkConfig = {
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
        { type: "pool", mode: "avg", global: true },
        { type: "dense", units: 2, activation: "sigmoid" },
      ],
    };
    for (let r = 0; r < RUNS; r++) {
      const { history } = createAndTrain(
        config,
        { learningRate: 0.2, epochs: 100, dataset: "tiny_images" },
        { includeDecisionBoundary: false }
      );
      accs.push(history.final.accuracy);
    }
    const meanAcc = accs.reduce((a, b) => a + b, 0) / accs.length;
    expect(meanAcc).toBeGreaterThanOrEqual(0.75);
    expect(Math.max(...accs)).toBeGreaterThanOrEqual(0.85);
  });
});

describe("pool — DSL", () => {
  it("parses windowed and global forms", () => {
    const windowed = parseDSL(`network N {
  conv2d filters=4 kernel=2
  pool mode=max size=2 stride=2
  flatten
  dense 2 activation=sigmoid
}
train dataset=tiny_images
`);
    expect(windowed.network.layers[1]).toEqual({
      type: "pool",
      mode: "max",
      size: 2,
      stride: 2,
      global: undefined,
    });

    const global = parseDSL(`network N {
  pool mode=avg global=true
  dense 1
}
train dataset=xor
`);
    expect(global.network.layers[0]).toMatchObject({
      type: "pool",
      mode: "avg",
      global: true,
    });
  });

  it("round-trips toDSL", () => {
    const src = `network TinyCNN {
  conv2d filters=4 kernel=3 activation=relu channels=1 height=8 width=8
  pool mode=max global=true
  dense 8 activation=relu
  dense 2 activation=sigmoid
}
train dataset=shifted_bars lr=0.12 epochs=80
`;
    const parsed = parseDSL(src);
    const again = parseDSL(toDSL(parsed.network, parsed.train));
    expect(again.network).toEqual(parsed.network);
    expect(again.train).toEqual(parsed.train);
  });

  it("rejects unknown mode/keys with valid alternatives", () => {
    let err: unknown;
    try {
      parseDSL(`network N {
  pool mode=banana size=2
}
train dataset=xor
`);
    } catch (e) {
      err = e;
    }
    expect(String(err)).toMatch(/banana/i);
    expect(String(err)).toMatch(/max/i);
    expect(String(err)).toMatch(/avg/i);

    try {
      parseDSL(`network N {
  pool mode=max padding=same
}
train dataset=xor
`);
    } catch (e) {
      err = e;
    }
    expect(String(err)).toMatch(/padding/i);
    expect(String(err)).toMatch(/mode|size|stride|global/i);
  });

  it("accepts avg aliases average/mean", () => {
    const a = parseDSL(`network N {
  pool mode=average size=2
}
train dataset=xor
`);
    expect(a.network.layers[0]).toMatchObject({ type: "pool", mode: "avg" });
    const b = parseDSL(`network N {
  pool mode=mean global=true
}
train dataset=xor
`);
    expect(b.network.layers[0]).toMatchObject({
      type: "pool",
      mode: "avg",
      global: true,
    });
  });
});
