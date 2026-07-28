import { describe, expect, it } from "vitest";
import { getDataset } from "./datasets";
import { parseDSL, toDSL } from "./dsl";
import {
  classKey,
  DEFAULT_VAL_RATIO,
  MIN_SAMPLES_FOR_VAL_SPLIT,
  resolveValRatio,
  splitTrainVal,
} from "./split";
import {
  createAndTrain,
  createTrainingSession,
  prepareDataSplit,
  prepareNetworkConfig,
  train,
} from "./train";
import { createModel, exportWeights, loadWeights } from "./model";
import type { NetworkConfig, Sample, TrainConfig } from "./types";

function countByClass(samples: Sample[]): Map<string, number> {
  const m = new Map<string, number>();
  for (const s of samples) {
    const k = classKey(s.y);
    m.set(k, (m.get(k) ?? 0) + 1);
  }
  return m;
}

describe("splitTrainVal — deterministic + stratified", () => {
  it("is deterministic for the same cached dataset", () => {
    const ds = getDataset("moons");
    const a = splitTrainVal(ds.samples, 0.25);
    const b = splitTrainVal(ds.samples, 0.25);
    expect(a.splitApplied).toBe(true);
    expect(a.train.length + a.val!.length).toBe(ds.samples.length);
    expect(a.train.map((s) => s.x.join(","))).toEqual(
      b.train.map((s) => s.x.join(","))
    );
    expect(a.val!.map((s) => s.x.join(","))).toEqual(
      b.val!.map((s) => s.x.join(","))
    );
  });

  it("leaves the hold-out off unless asked", () => {
    // Opt-in by design: defaulting it on withheld a quarter of the data from
    // every chapter that predates it, and moved figures already written into
    // the lesson prose. DEFAULT_VAL_RATIO remains the suggested value for
    // callers that want one, not an implicit default.
    expect(resolveValRatio(undefined)).toBe(0);
    expect(resolveValRatio(0)).toBe(0);
    expect(resolveValRatio(0.3)).toBe(0.3);
    expect(resolveValRatio(1)).toBe(0);
    expect(DEFAULT_VAL_RATIO).toBeGreaterThan(0);
  });

  it("keeps class balance within one sample of the target ratio on moons", () => {
    const ds = getDataset("moons");
    const split = splitTrainVal(ds.samples, 0.25);
    expect(split.splitApplied).toBe(true);

    const full = countByClass(ds.samples);
    const trainC = countByClass(split.train);
    const valC = countByClass(split.val!);

    for (const [cls, n] of full) {
      const nVal = valC.get(cls) ?? 0;
      const nTrain = trainC.get(cls) ?? 0;
      expect(nTrain + nVal).toBe(n);
      // ~25% hold-out, at least 1 per class, not all
      expect(nVal).toBeGreaterThanOrEqual(1);
      expect(nTrain).toBeGreaterThanOrEqual(1);
      const expected = Math.floor(n * 0.25);
      // floor or floor+1 after the "at least 1" rule
      expect(Math.abs(nVal - Math.max(1, expected))).toBeLessThanOrEqual(1);
    }
  });

  it("never puts the same sample in both train and val", () => {
    const ds = getDataset("circles");
    const split = splitTrainVal(ds.samples, 0.3);
    const trainKeys = new Set(split.train.map((s) => s.x.join("|")));
    for (const s of split.val!) {
      expect(trainKeys.has(s.x.join("|"))).toBe(false);
    }
  });

  it("skips tiny logic datasets (xor / and / or)", () => {
    for (const name of ["xor", "and", "or"] as const) {
      const ds = getDataset(name);
      expect(ds.samples.length).toBeLessThan(MIN_SAMPLES_FOR_VAL_SPLIT);
      const split = splitTrainVal(ds.samples, 0.25);
      expect(split.splitApplied).toBe(false);
      expect(split.val).toBeNull();
      expect(split.train).toHaveLength(ds.samples.length);
    }
  });

  it("disables when valRatio is 0 even on large datasets", () => {
    const ds = getDataset("moons");
    const split = splitTrainVal(ds.samples, 0);
    expect(split.splitApplied).toBe(false);
    expect(split.val).toBeNull();
  });

  it("applies a split to moons, circles, spiral, linear, tiny_*, noisy_moons, shifted_bars, negation", () => {
    for (const name of [
      "moons",
      "circles",
      "spiral",
      "linear",
      "tiny_images",
      "tiny_text",
      "noisy_moons",
      "shifted_bars",
      "negation",
    ] as const) {
      const ds = getDataset(name);
      if (ds.samples.length < MIN_SAMPLES_FOR_VAL_SPLIT) continue;
      const split = splitTrainVal(ds.samples, DEFAULT_VAL_RATIO);
      expect(split.splitApplied, name).toBe(true);
      expect(split.val!.length, name).toBeGreaterThan(0);
      expect(split.train.length, name).toBeGreaterThan(0);
    }
  });
});

describe("train path — structural hold-out", () => {
  it("prepareDataSplit respects valRatio=0 and default", () => {
    const off = prepareDataSplit({
      learningRate: 0.2,
      epochs: 1,
      dataset: "moons",
      valRatio: 0,
    });
    expect(off.splitApplied).toBe(false);

    const omitted = prepareDataSplit({
      learningRate: 0.2,
      epochs: 1,
      dataset: "moons",
    });
    expect(omitted.splitApplied).toBe(false);

    const on = prepareDataSplit({
      learningRate: 0.2,
      epochs: 1,
      dataset: "moons",
      valRatio: 0.3,
    });
    expect(on.splitApplied).toBe(true);
  });

  it("train metrics are train-set; val metrics present on moons", () => {
    const { history } = createAndTrain(
      {
        layers: [
          { type: "dense", units: 8, activation: "relu", inputDim: 2 },
          { type: "dense", units: 1, activation: "sigmoid" },
        ],
      },
      {
        learningRate: 0.25,
        epochs: 15,
        dataset: "moons",
        shuffle: false,
        valRatio: 0.25,
      }
    );

    expect(history.valLosses).toBeDefined();
    expect(history.valAccuracies).toBeDefined();
    expect(history.valLosses!).toHaveLength(history.losses.length);
    expect(history.valAccuracies!).toHaveLength(history.accuracies.length);
    expect(history.final.valLoss).not.toBeNull();
    expect(history.final.valAccuracy).not.toBeNull();
    expect(Number.isFinite(history.final.valLoss!)).toBe(true);
    expect(history.final.valAccuracy!).toBeGreaterThanOrEqual(0);
    expect(history.final.valAccuracy!).toBeLessThanOrEqual(1);
    // Existing field names still work
    expect(history.final.loss).toBeDefined();
    expect(history.final.accuracy).toBeDefined();
  });

  it("xor has no val metrics (tiny-dataset rule)", () => {
    const { history } = createAndTrain(
      {
        layers: [
          { type: "dense", units: 4, activation: "relu", inputDim: 2 },
          { type: "dense", units: 1, activation: "sigmoid" },
        ],
      },
      { learningRate: 0.35, epochs: 20, dataset: "xor", shuffle: false }
    );
    expect(history.final.valLoss).toBeNull();
    expect(history.final.valAccuracy).toBeNull();
    expect(history.valLosses ?? []).toHaveLength(0);
    expect(history.valAccuracies ?? []).toHaveLength(0);
  });

  it("session and train() match including val series (shuffle off)", () => {
    const config: NetworkConfig = {
      layers: [
        { type: "dense", units: 6, activation: "relu", inputDim: 2 },
        { type: "dense", units: 1, activation: "sigmoid" },
      ],
    };
    const trainConfig: TrainConfig = {
      learningRate: 0.3,
      epochs: 12,
      dataset: "moons",
      shuffle: false,
      valRatio: 0.25,
    };
    const prepared = prepareNetworkConfig(config, "moons");
    const seed = createModel(prepared.config, prepared.inputDim);
    const seedWeights = exportWeights(seed);

    const modelA = createModel(prepared.config, prepared.inputDim);
    loadWeights(modelA, seedWeights);
    const history = train(modelA, trainConfig);

    const sess = createTrainingSession(config, trainConfig, {
      initialWeights: seedWeights,
    });
    expect(sess.hasValidation).toBe(true);
    for (let i = 0; i < 12; i++) sess.runEpoch();

    expect(sess.valLosses).toHaveLength(12);
    expect(history.valLosses).toHaveLength(12);
    for (let i = 0; i < 12; i++) {
      expect(sess.losses[i]).toBeCloseTo(history.losses[i]!, 10);
      expect(sess.accuracies[i]).toBeCloseTo(history.accuracies[i]!, 10);
      expect(sess.valLosses[i]).toBeCloseTo(history.valLosses![i]!, 10);
      expect(sess.valAccuracies[i]).toBeCloseTo(history.valAccuracies![i]!, 10);
    }
    expect(sess.lastSnapshot!.valAccuracy).toBeCloseTo(
      history.final.valAccuracy!,
      10
    );
  });
});

describe("DSL val= knob", () => {
  it("parses val= and round-trips via toDSL", () => {
    const source = `network Net {
  dense 2 -> 4 activation=relu
  dense 4 -> 1 activation=sigmoid
}
train dataset=moons lr=0.2 epochs=50 val=0.3
`;
    const parsed = parseDSL(source);
    expect(parsed.train.valRatio).toBe(0.3);
    const again = parseDSL(toDSL(parsed.network, parsed.train));
    expect(again.train.valRatio).toBe(0.3);
  });

  it("accepts val=0 to disable", () => {
    const parsed = parseDSL(`network N {
  dense 2 -> 1
}
train dataset=moons lr=0.2 epochs=10 val=0
`);
    expect(parsed.train.valRatio).toBe(0);
  });

  it("rejects malformed val", () => {
    expect(() =>
      parseDSL(`network N {
  dense 2 -> 1
}
train dataset=moons val=1
`)
    ).toThrow(/val/);
    expect(() =>
      parseDSL(`network N {
  dense 2 -> 1
}
train dataset=moons val=-0.1
`)
    ).toThrow(/val/);
    expect(() =>
      parseDSL(`network N {
  dense 2 -> 1
}
train dataset=moons val=banana
`)
    ).toThrow(/val/);
  });
});

/**
 * Chapter 3 needs a dataset where overfitting is reliably observable and
 * fixable. Clean sets (moons/circles) generalise at any capacity; spiral
 * underfits both sides. `noisy_moons` flips a fixed fraction of labels so
 * memorising individual points actively hurts held-out accuracy.
 *
 * Aggregates only — per-run assertions on stochastic training have flaked
 * twice in this repo already.
 *
 * Measured over 40 runs (val=0.3), defaults of `noisy_moons`:
 *   64×64 l2=0  e500 lr=0.08 → train ~96%  val ~64%  gap ~32pp (min ~21pp)
 *   64×64 l2=0.005 e300      → train ~82%  val ~75%  (held-out +12pp)
 *   dense 6 units            → train ~80%  val ~69%  (held-out +5pp)
 */
describe("overfitting signature (noisy_moons)", () => {
  it("is deterministic (same points + labels every call)", () => {
    const a = getDataset("noisy_moons");
    const b = getDataset("noisy_moons");
    expect(a.samples.length).toBe(b.samples.length);
    expect(a.samples.length).toBeGreaterThanOrEqual(MIN_SAMPLES_FOR_VAL_SPLIT);
    for (let i = 0; i < a.samples.length; i++) {
      expect(a.samples[i]!.x).toEqual(b.samples[i]!.x);
      expect(a.samples[i]!.y).toEqual(b.samples[i]!.y);
    }
    // Refresh regenerates identical geometry + flips (no Math.random).
    const c = getDataset("noisy_moons", true);
    expect(c.samples.map((s) => [...s.x, ...s.y])).toEqual(
      a.samples.map((s) => [...s.x, ...s.y])
    );
  });

  it("high capacity overfits; L2 / small capacity improve held-out", () => {
    const RUNS = 20;

    const overfitNet: NetworkConfig = {
      name: "OverfitDemo",
      layers: [
        { type: "dense", units: 64, activation: "relu", inputDim: 2 },
        { type: "dense", units: 64, activation: "relu" },
        { type: "dense", units: 1, activation: "sigmoid" },
      ],
      l2: 0,
    };
    const overfitTrain: TrainConfig = {
      learningRate: 0.08,
      epochs: 500,
      dataset: "noisy_moons",
      shuffle: true,
      valRatio: 0.3,
    };

    const l2Net: NetworkConfig = {
      name: "L2Fix",
      layers: [
        { type: "dense", units: 64, activation: "relu", inputDim: 2 },
        { type: "dense", units: 64, activation: "relu" },
        { type: "dense", units: 1, activation: "sigmoid" },
      ],
      l2: 0.005,
    };
    const l2Train: TrainConfig = {
      learningRate: 0.12,
      epochs: 300,
      dataset: "noisy_moons",
      shuffle: true,
      valRatio: 0.3,
    };

    const smallNet: NetworkConfig = {
      name: "SmallFix",
      layers: [
        { type: "dense", units: 6, activation: "relu", inputDim: 2 },
        { type: "dense", units: 1, activation: "sigmoid" },
      ],
      l2: 0,
    };
    const smallTrain: TrainConfig = {
      learningRate: 0.3,
      epochs: 250,
      dataset: "noisy_moons",
      shuffle: true,
      valRatio: 0.3,
    };

    const overfitTrainAccs: number[] = [];
    const overfitValAccs: number[] = [];
    const overfitGaps: number[] = [];
    const l2ValAccs: number[] = [];
    const smallValAccs: number[] = [];

    for (let r = 0; r < RUNS; r++) {
      const over = createAndTrain(overfitNet, overfitTrain, {
        includeDecisionBoundary: false,
      });
      const t = over.history.final.accuracy;
      const v = over.history.final.valAccuracy;
      expect(v, "overfit run must produce held-out metrics").not.toBeNull();
      expect(Number.isFinite(v!)).toBe(true);
      overfitTrainAccs.push(t);
      overfitValAccs.push(v!);
      overfitGaps.push(t - v!);

      const reg = createAndTrain(l2Net, l2Train, {
        includeDecisionBoundary: false,
      });
      expect(reg.history.final.valAccuracy).not.toBeNull();
      l2ValAccs.push(reg.history.final.valAccuracy!);

      const sm = createAndTrain(smallNet, smallTrain, {
        includeDecisionBoundary: false,
      });
      expect(sm.history.final.valAccuracy).not.toBeNull();
      smallValAccs.push(sm.history.final.valAccuracy!);
    }

    const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
    const meanTrain = mean(overfitTrainAccs);
    const meanVal = mean(overfitValAccs);
    const meanGap = mean(overfitGaps);
    const meanL2Val = mean(l2ValAccs);
    const meanSmallVal = mean(smallValAccs);

    const summary =
      `overfit train=${(meanTrain * 100).toFixed(1)}% val=${(meanVal * 100).toFixed(1)}% ` +
      `gap=${(meanGap * 100).toFixed(1)}pp (n=${RUNS}); ` +
      `L2 val=${(meanL2Val * 100).toFixed(1)}% small val=${(meanSmallVal * 100).toFixed(1)}%`;

    // 1. High capacity: near-perfect train, clearly worse held-out.
    expect(
      meanTrain,
      `high-capacity mean train should be near-perfect.\n${summary}`
    ).toBeGreaterThan(0.9);
    expect(
      meanGap,
      `mean train−val gap must be unmistakable (≫ sampling noise).\n${summary}`
    ).toBeGreaterThan(0.15);
    expect(
      meanVal,
      `held-out must lag train — not a joint underfit.\n${summary}`
    ).toBeLessThan(meanTrain - 0.12);

    // 2. Fix direction: L2 and/or reduced capacity improve held-out accuracy.
    // Either fix is enough for the chapter; both should help on this set.
    const bestFixVal = Math.max(meanL2Val, meanSmallVal);
    expect(
      bestFixVal,
      `L2 or small net must improve held-out vs unregularised high-cap.\n${summary}`
    ).toBeGreaterThan(meanVal + 0.03);
  }, 180_000);
});
