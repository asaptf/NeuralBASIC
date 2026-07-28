/**
 * `shifted_bars` + conv-vs-dense with pooling.
 *
 * Aggregates only — per-run assertions on stochastic training have flaked
 * repeatedly in this repo.
 *
 * ## Why this dataset exists
 * `tiny_images` has only four bar positions and lists every one in the 16
 * training samples. A dense net memorizes them; weight sharing buys nothing.
 * `shifted_bars` places length-3 bars on an 8×8 grid at every valid
 * translation (96 clean + 96 noisy = 192).
 *
 * ## Why pooling is required
 * Valid-pad stride-1 conv2d alone leaves a position-specific dense head after
 * flatten. Global average pooling collapses each feature map to one number
 * per filter, so the readout is translation-invariant and weight sharing wins.
 *
 * ## Why held-out *positions* (not random val)
 * Random val still overlaps the same lattice of placements (clean + noise
 * copies). Dense can memorise train positions and score well on nearby/noise
 * variants. A checkerboard position hold-out is the honest generalisation test.
 *
 * ## Measured (position hold-out, 12 runs, 150 epochs)
 * | architecture                                      | params | mean test | notes        |
 * |---------------------------------------------------|-------:|----------:|--------------|
 * | conv 4×k=3 → global avg → dense 2                 |     50 |   ~0.98   | stable       |
 * | dense 16-8-2                                      |   1194 |   ~0.77   | memorizes    |
 * | dense 32-16-2                                     |   2642 |   ~0.78   | memorizes    |
 */
import { describe, expect, it } from "vitest";
import { getDataset, DATASET_NAMES } from "./datasets";
import { parseDSL, toDSL } from "./dsl";
import { createModel, predict, type Model } from "./model";
import { trainStep, prepareNetworkConfig } from "./train";
import type { NetworkConfig, Sample, TrainConfig } from "./types";

function countParams(
  config: NetworkConfig,
  dataset: TrainConfig["dataset"] = "shifted_bars"
): number {
  const { config: cfg, inputDim } = prepareNetworkConfig(config, dataset);
  const model = createModel(cfg, inputDim);
  let n = 0;
  for (const layer of model.layers) {
    if (layer.type === "dense") {
      for (const row of layer.weights) n += row.length;
      n += layer.biases.length;
    } else if (layer.type === "conv2d") {
      for (const f of layer.kernels)
        for (const c of f) for (const row of c) n += row.length;
      n += layer.biases.length;
    }
  }
  return n;
}

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function std(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  const v = xs.reduce((s, x) => s + (x - m) * (x - m), 0) / (xs.length - 1);
  return Math.sqrt(v);
}

function accuracyOn(model: Model, samples: Sample[]): number {
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

/**
 * Checkerboard position hold-out for `shifted_bars`.
 * Generation order (see datasets.ts): vertical placements r∈[0,5], c∈[0,7]
 * then horizontal r∈[0,7], c∈[0,5]; each placement yields clean+noisy.
 * Hold out positions where (r+c) is odd.
 */
function positionHoldout(): { train: Sample[]; test: Sample[] } {
  const ds = getDataset("shifted_bars");
  const train: Sample[] = [];
  const test: Sample[] = [];

  for (let i = 0; i < 48; i++) {
    const r = Math.floor(i / 8);
    const c = i % 8;
    const hold = (r + c) % 2 === 1;
    const pair = [ds.samples[i * 2]!, ds.samples[i * 2 + 1]!];
    (hold ? test : train).push(...pair);
  }
  for (let i = 0; i < 48; i++) {
    const r = Math.floor(i / 6);
    const c = i % 6;
    const hold = (r + c) % 2 === 1;
    const base = 96;
    const pair = [ds.samples[base + i * 2]!, ds.samples[base + i * 2 + 1]!];
    (hold ? test : train).push(...pair);
  }
  return { train, test };
}

function trainOnSamples(
  config: NetworkConfig,
  samples: Sample[],
  lr: number,
  epochs: number
): Model {
  const model = createModel(config, 64);
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

describe("tiny_images determinism", () => {
  it("is stable across cache hits and seedRefresh (no Math.random)", () => {
    const a = getDataset("tiny_images");
    const b = getDataset("tiny_images");
    expect(a.samples.length).toBe(16);
    expect(a.samples.map((s) => [...s.x, ...s.y])).toEqual(
      b.samples.map((s) => [...s.x, ...s.y])
    );
    const c = getDataset("tiny_images", true);
    expect(c.samples.map((s) => [...s.x, ...s.y])).toEqual(
      a.samples.map((s) => [...s.x, ...s.y])
    );
    const class0 = a.samples.filter((s) => s.y[0]! > s.y[1]!).length;
    expect(class0).toBe(8);
    expect(a.samples.length - class0).toBe(8);
  });
});

describe("shifted_bars dataset", () => {
  it("is registered and listed", () => {
    expect(DATASET_NAMES).toContain("shifted_bars");
    const ds = getDataset("shifted_bars");
    expect(ds.name).toBe("shifted_bars");
    expect(ds.inputShape).toEqual([1, 8, 8]);
    expect(ds.outputDim).toBe(2);
    expect(ds.samples.length).toBe(192);
    const c0 = ds.samples.filter((s) => s.y[0]! > s.y[1]!).length;
    expect(c0).toBe(96);
  });

  it("is deterministic (same pixels + labels every call)", () => {
    const a = getDataset("shifted_bars");
    const b = getDataset("shifted_bars");
    expect(a.samples.map((s) => [...s.x, ...s.y])).toEqual(
      b.samples.map((s) => [...s.x, ...s.y])
    );
    const c = getDataset("shifted_bars", true);
    expect(c.samples.map((s) => [...s.x, ...s.y])).toEqual(
      a.samples.map((s) => [...s.x, ...s.y])
    );
  });

  it("parses pooled CNN in DSL and round-trips via toDSL", () => {
    const src = `network ShiftedCNN {
  conv2d filters=4 kernel=3 activation=relu channels=1 height=8 width=8
  pool mode=avg global=true
  dense 2 activation=sigmoid
}
train dataset=shifted_bars lr=0.2 epochs=150
`;
    const parsed = parseDSL(src);
    expect(parsed.train.dataset).toBe("shifted_bars");
    expect(parsed.network.layers[0]).toMatchObject({
      type: "conv2d",
      filters: 4,
      kernelSize: 3,
      inputHeight: 8,
      inputWidth: 8,
    });
    expect(parsed.network.layers[1]).toMatchObject({
      type: "pool",
      mode: "avg",
      global: true,
    });
    const again = parseDSL(toDSL(parsed.network, parsed.train));
    expect(again.train.dataset).toBe("shifted_bars");
    expect(again.network).toEqual(parsed.network);
  });

  it(
    "is learnable by a small global-avg CNN (aggregate train accuracy)",
    () => {
      const RUNS = 6;
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
          { type: "pool", mode: "avg", global: true },
          { type: "dense", units: 2, activation: "sigmoid" },
        ],
      };
      const { train } = positionHoldout();
      const accs: number[] = [];
      for (let r = 0; r < RUNS; r++) {
        const model = trainOnSamples(config, train, 0.2, 100);
        accs.push(accuracyOn(model, train));
      }
      expect(mean(accs)).toBeGreaterThanOrEqual(0.9);
      expect(Math.max(...accs)).toBeGreaterThanOrEqual(0.95);
    }
  );
});

/**
 * Prove-it suite: CNN + global avg-pool must beat denser dense baselines on
 * position-held-out accuracy with fewer parameters.
 */
describe("conv with pool vs dense on shifted_bars", () => {
  it(
    "global-avg CNN beats ≥-param dense baselines on held-out positions",
    () => {
      const RUNS = 12;
      const EPOCHS = 150;

      // 50 params: conv 4×k=3 (40) + dense 4→2 (10)
      const cnn: NetworkConfig = {
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
          { type: "pool", mode: "avg", global: true },
          { type: "dense", units: 2, activation: "sigmoid" },
        ],
      };
      // 1194 params
      const denseMatched: NetworkConfig = {
        layers: [
          { type: "dense", units: 16, activation: "relu", inputDim: 64 },
          { type: "dense", units: 8, activation: "relu" },
          { type: "dense", units: 2, activation: "sigmoid" },
        ],
      };
      // 2642 params
      const denseLarger: NetworkConfig = {
        layers: [
          { type: "dense", units: 32, activation: "relu", inputDim: 64 },
          { type: "dense", units: 16, activation: "relu" },
          { type: "dense", units: 2, activation: "sigmoid" },
        ],
      };

      const cnnParams = countParams(cnn);
      const denseMatchedParams = countParams(denseMatched);
      const denseLargerParams = countParams(denseLarger);

      expect(denseMatchedParams).toBeGreaterThanOrEqual(cnnParams);
      expect(denseLargerParams).toBeGreaterThan(cnnParams);
      expect(cnnParams).toBeLessThanOrEqual(denseMatchedParams);
      expect(cnnParams).toBeLessThan(100);

      const { train, test } = positionHoldout();
      expect(train.length).toBe(96);
      expect(test.length).toBe(96);

      const cnnTest: number[] = [];
      const d1Test: number[] = [];
      const d2Test: number[] = [];
      const cnnTrain: number[] = [];
      const d1Train: number[] = [];
      const d2Train: number[] = [];

      for (let r = 0; r < RUNS; r++) {
        const cnnModel = trainOnSamples(cnn, train, 0.2, EPOCHS);
        cnnTest.push(accuracyOn(cnnModel, test));
        cnnTrain.push(accuracyOn(cnnModel, train));

        const d1 = trainOnSamples(denseMatched, train, 0.15, EPOCHS);
        d1Test.push(accuracyOn(d1, test));
        d1Train.push(accuracyOn(d1, train));

        const d2 = trainOnSamples(denseLarger, train, 0.12, EPOCHS);
        d2Test.push(accuracyOn(d2, test));
        d2Train.push(accuracyOn(d2, train));
      }

      const cnnMean = mean(cnnTest);
      const d1Mean = mean(d1Test);
      const d2Mean = mean(d2Test);

      const report = {
        RUNS,
        EPOCHS,
        params: {
          cnn: cnnParams,
          dense16: denseMatchedParams,
          dense32: denseLargerParams,
        },
        testMean: { cnn: cnnMean, dense16: d1Mean, dense32: d2Mean },
        testStd: {
          cnn: std(cnnTest),
          dense16: std(d1Test),
          dense32: std(d2Test),
        },
        trainMean: {
          cnn: mean(cnnTrain),
          dense16: mean(d1Train),
          dense32: mean(d2Train),
        },
        testMin: {
          cnn: Math.min(...cnnTest),
          dense16: Math.min(...d1Test),
          dense32: Math.min(...d2Test),
        },
      };
      console.log(
        "[shifted_bars position hold-out]",
        JSON.stringify(report, null, 2)
      );

      // Dense baselines fit the training lattice (not broken / handicapped).
      expect(mean(d1Train)).toBeGreaterThanOrEqual(0.95);
      expect(mean(d2Train)).toBeGreaterThanOrEqual(0.95);

      // CNN also fits train (global avg is expressive enough for bars).
      expect(mean(cnnTrain)).toBeGreaterThanOrEqual(0.9);

      // The claim: fewer-param CNN wins position-held-out accuracy vs both
      // denser nets. Margin leaves room for init noise while remaining clear.
      expect(cnnMean).toBeGreaterThan(d1Mean + 0.08);
      expect(cnnMean).toBeGreaterThan(d2Mean + 0.08);
      expect(cnnMean).toBeGreaterThanOrEqual(0.9);
    }
  );
});
