import { describe, expect, it } from "vitest";
import { getDataset } from "./datasets";
import {
  createModel,
  exportWeights,
  loadWeights,
  predict,
} from "./model";
import {
  boundsFromSamples,
  buildDecisionGrid,
  createAndTrain,
  createTrainingSession,
  prepareNetworkConfig,
  train,
} from "./train";
import type { NetworkConfig, TrainConfig } from "./types";

const andConfig: NetworkConfig = {
  layers: [{ type: "dense", units: 1, activation: "sigmoid", inputDim: 2 }],
};

describe("TrainingSession (incremental epochs)", () => {
  it("N runEpoch() calls match train() on a fixed setup (shuffle off)", () => {
    const trainConfig: TrainConfig = {
      learningRate: 0.5,
      epochs: 25,
      dataset: "and",
      shuffle: false,
    };
    const prepared = prepareNetworkConfig(andConfig, "and");
    const seedModel = createModel(prepared.config, prepared.inputDim);
    const seedWeights = exportWeights(seedModel);

    const modelA = createModel(prepared.config, prepared.inputDim);
    loadWeights(modelA, seedWeights);
    const history = train(modelA, trainConfig);

    const sess = createTrainingSession(andConfig, trainConfig, {
      initialWeights: seedWeights,
    });
    for (let i = 0; i < 25; i++) sess.runEpoch();

    expect(sess.epochsRun).toBe(25);
    expect(sess.isDone).toBe(true);
    expect(sess.losses).toHaveLength(25);
    expect(history.losses).toHaveLength(25);

    for (let i = 0; i < 25; i++) {
      expect(sess.losses[i]).toBeCloseTo(history.losses[i]!, 10);
      expect(sess.accuracies[i]).toBeCloseTo(history.accuracies[i]!, 10);
    }

    expect(sess.accuracies[sess.accuracies.length - 1]!).toBeGreaterThanOrEqual(
      0.75
    );
    expect(sess.lastSnapshot!.epoch).toBe(25);
    expect(sess.lastSnapshot!.layerSnapshots.length).toBeGreaterThan(0);

    // Final weights should match the batch train path
    expect(JSON.stringify(sess.exportWeights())).toBe(
      JSON.stringify(exportWeights(modelA))
    );
  });

  it("runEpoch past totalEpochs is a safe no-op", () => {
    const session = createTrainingSession(andConfig, {
      learningRate: 0.4,
      epochs: 3,
      dataset: "and",
      shuffle: false,
    });
    for (let i = 0; i < 3; i++) session.runEpoch();
    const snap = session.lastSnapshot!;
    const w = session.exportWeights();
    const extra = session.runEpoch();
    const extra2 = session.runEpoch();

    expect(session.epochsRun).toBe(3);
    expect(session.isDone).toBe(true);
    expect(session.losses).toHaveLength(3);
    expect(extra.epoch).toBe(snap.epoch);
    expect(extra.loss).toBe(snap.loss);
    expect(extra2.loss).toBe(snap.loss);

    // Weights unchanged after no-op epochs
    const w2 = session.exportWeights();
    expect(JSON.stringify(w2)).toBe(JSON.stringify(w));
  });

  it("reset() restores a trainable fresh state", () => {
    const session = createTrainingSession(andConfig, {
      learningRate: 0.6,
      epochs: 10,
      dataset: "and",
      shuffle: false,
    });
    for (let i = 0; i < 5; i++) session.runEpoch();
    expect(session.epochsRun).toBe(5);
    expect(session.losses.length).toBe(5);
    expect(session.lastSnapshot).not.toBeNull();

    const midWeights = session.exportWeights();
    session.reset();

    expect(session.epochsRun).toBe(0);
    expect(session.isDone).toBe(false);
    expect(session.losses).toHaveLength(0);
    expect(session.accuracies).toHaveLength(0);
    expect(session.lastSnapshot).toBeNull();

    // After reset, mid-training weights are gone (new random init)
    const freshWeights = session.exportWeights();
    expect(JSON.stringify(freshWeights)).not.toBe(JSON.stringify(midWeights));

    const r1 = session.runEpoch();
    expect(session.epochsRun).toBe(1);
    expect(r1.epoch).toBe(1);
    expect(Number.isFinite(r1.loss)).toBe(true);

    while (!session.isDone) session.runEpoch();
    expect(session.epochsRun).toBe(10);
    expect(session.accuracies[session.accuracies.length - 1]!).toBeGreaterThanOrEqual(
      0.75
    );
  });

  it("createAndTrain still works and session exportWeights is usable", () => {
    const { history, weights } = createAndTrain(andConfig, {
      learningRate: 0.5,
      epochs: 15,
      dataset: "and",
      shuffle: false,
    });
    expect(history.final.accuracy).toBeGreaterThanOrEqual(0.99);
    expect(weights[0]?.weights).toBeDefined();

    const session = createTrainingSession(andConfig, {
      learningRate: 0.5,
      epochs: 5,
      dataset: "or",
    });
    session.runEpoch();
    const w = session.exportWeights();
    const prepared = prepareNetworkConfig(andConfig, "or");
    const m = createModel(prepared.config, prepared.inputDim);
    loadWeights(m, w);
    const p = predict(m, [1, 1]);
    expect(Number.isFinite(p[0]!)).toBe(true);
  });
});

describe("decision grid bounds from data", () => {
  it("bounds contain every 2-D sample of xor, moons, circles, spiral", () => {
    for (const name of ["xor", "moons", "circles", "spiral"] as const) {
      const ds = getDataset(name);
      const b = boundsFromSamples(ds.samples);
      expect(b.xMax).toBeGreaterThan(b.xMin);
      expect(b.yMax).toBeGreaterThan(b.yMin);
      for (const s of ds.samples) {
        expect(s.x[0]!).toBeGreaterThanOrEqual(b.xMin);
        expect(s.x[0]!).toBeLessThanOrEqual(b.xMax);
        expect(s.x[1]!).toBeGreaterThanOrEqual(b.yMin);
        expect(s.x[1]!).toBeLessThanOrEqual(b.yMax);
      }
    }
  });

  it("train snapshots populate decisionGrid with data-aware bounds for 2-D sets", () => {
    const { history } = createAndTrain(
      {
        layers: [
          { type: "dense", units: 4, activation: "relu", inputDim: 2 },
          { type: "dense", units: 1, activation: "sigmoid" },
        ],
      },
      { learningRate: 0.3, epochs: 5, dataset: "xor", shuffle: false }
    );
    const grid = history.final.decisionGrid;
    expect(grid).toBeDefined();
    const ds = getDataset("xor");
    for (const s of ds.samples) {
      expect(s.x[0]!).toBeGreaterThanOrEqual(grid!.xMin);
      expect(s.x[0]!).toBeLessThanOrEqual(grid!.xMax);
      expect(s.x[1]!).toBeGreaterThanOrEqual(grid!.yMin);
      expect(s.x[1]!).toBeLessThanOrEqual(grid!.yMax);
    }
  });

  it("explicit bounds passed to buildDecisionGrid win over data defaults", () => {
    const model = createModel(andConfig, 2);
    const grid = buildDecisionGrid(model, 4, -3, 3, -2, 2);
    expect(grid!.xMin).toBe(-3);
    expect(grid!.xMax).toBe(3);
    expect(grid!.yMin).toBe(-2);
    expect(grid!.yMax).toBe(2);
    expect(grid!.values).toHaveLength(16);
  });

  it("zero-width axis still yields a sane padded box", () => {
    const b = boundsFromSamples([{ x: [0.5, 0.5] }]);
    expect(b.xMax - b.xMin).toBeGreaterThan(0.5);
    expect(b.yMax - b.yMin).toBeGreaterThan(0.5);
    expect(0.5).toBeGreaterThanOrEqual(b.xMin);
    expect(0.5).toBeLessThanOrEqual(b.xMax);
  });
});
