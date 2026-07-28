import { describe, expect, it } from "vitest";
import {
  exportExperimentJSON,
  importExperimentJSON,
} from "./persistence";
import {
  buildModelExport,
  modelExportToJSON,
  parseModelExport,
  toPyTorchSnippet,
} from "@/engine/export";
import { createAndTrain } from "@/engine/train";

describe("experiment save/load round-trip", () => {
  it("JSON export/import preserves dsl, network, trainConfig", () => {
    const state = {
      dsl: "network A {\n  dense 2 -> 1 activation=sigmoid\n}\ntrain dataset=and lr=0.5 epochs=50\n",
      network: {
        name: "A",
        layers: [
          {
            type: "dense" as const,
            units: 1,
            activation: "sigmoid" as const,
            inputDim: 2,
          },
        ],
      },
      trainConfig: {
        learningRate: 0.5,
        epochs: 50,
        dataset: "and" as const,
      },
      weights: [] as [],
      history: { losses: [1, 0.5], accuracies: [0.5, 1] },
      name: "A",
    };
    const json = exportExperimentJSON(state);
    const loaded = importExperimentJSON(json);
    expect(loaded.dsl).toBe(state.dsl);
    expect(loaded.network.name).toBe("A");
    expect(loaded.trainConfig.dataset).toBe("and");
    expect(loaded.history?.losses).toEqual([1, 0.5]);
  });
});

describe("model export artifacts", () => {
  it("produces model JSON + non-empty pytorch snippet from trained model", () => {
    const config = {
      name: "PersistNet",
      layers: [
        { type: "dense" as const, units: 1, activation: "sigmoid" as const, inputDim: 2 },
      ],
    };
    const trainConfig = {
      learningRate: 0.6,
      epochs: 30,
      dataset: "or" as const,
    };
    const { weights, history } = createAndTrain(config, trainConfig);
    const exp = buildModelExport("PersistNet", config, trainConfig, weights, {
      loss: history.final.loss,
      accuracy: history.final.accuracy,
    });
    const json = modelExportToJSON(exp);
    const round = parseModelExport(json);
    expect(round.weights[0]?.weights).toBeDefined();
    const pt = toPyTorchSnippet(config, trainConfig);
    expect(pt.trim().length).toBeGreaterThan(0);
    expect(pt).toMatch(/import torch/);
  });
});
