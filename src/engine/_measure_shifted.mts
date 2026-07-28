import { createAndTrain, prepareNetworkConfig } from "./train";
import { createModel } from "./model";
import { getDataset } from "./datasets";
import type { NetworkConfig, TrainConfig } from "./types";

function countParams(config: NetworkConfig, dataset: TrainConfig["dataset"]): number {
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

function mean(xs: number[]) { return xs.reduce((a,b)=>a+b,0)/xs.length; }
function min(xs: number[]) { return Math.min(...xs); }
function max(xs: number[]) { return Math.max(...xs); }

const ds = getDataset("shifted_bars");
console.log("shifted_bars samples:", ds.samples.length, "shape", ds.inputShape);
const c0 = ds.samples.filter(s => s.y[0]! > s.y[1]!).length;
console.log("class balance:", c0, ds.samples.length - c0);

const a = getDataset("tiny_images");
const b = getDataset("tiny_images", true);
console.log("tiny_images det:", JSON.stringify(a.samples) === JSON.stringify(b.samples), "n=", a.samples.length);

type Spec = { name: string; config: NetworkConfig; lr: number; epochs: number; val?: number };

const specs: Spec[] = [
  {
    name: "cnn2f-k3-d4",
    config: {
      layers: [
        { type: "conv2d", filters: 2, kernelSize: 3, activation: "relu", inputChannels: 1, inputHeight: 8, inputWidth: 8 },
        { type: "flatten" },
        { type: "dense", units: 4, activation: "relu" },
        { type: "dense", units: 2, activation: "sigmoid" },
      ],
    },
    lr: 0.15, epochs: 200, val: 0.3,
  },
  {
    name: "cnn4f-k3-d8",
    config: {
      layers: [
        { type: "conv2d", filters: 4, kernelSize: 3, activation: "relu", inputChannels: 1, inputHeight: 8, inputWidth: 8 },
        { type: "flatten" },
        { type: "dense", units: 8, activation: "relu" },
        { type: "dense", units: 2, activation: "sigmoid" },
      ],
    },
    lr: 0.12, epochs: 200, val: 0.3,
  },
  {
    name: "cnn2f-k2-d8",
    config: {
      layers: [
        { type: "conv2d", filters: 2, kernelSize: 2, activation: "relu", inputChannels: 1, inputHeight: 8, inputWidth: 8 },
        { type: "flatten" },
        { type: "dense", units: 8, activation: "relu" },
        { type: "dense", units: 2, activation: "sigmoid" },
      ],
    },
    lr: 0.12, epochs: 200, val: 0.3,
  },
  {
    name: "dense64-16-2",
    config: {
      layers: [
        { type: "dense", units: 64, activation: "relu", inputDim: 64 },
        { type: "dense", units: 16, activation: "relu" },
        { type: "dense", units: 2, activation: "sigmoid" },
      ],
    },
    lr: 0.12, epochs: 200, val: 0.3,
  },
  {
    name: "dense32-16-2",
    config: {
      layers: [
        { type: "dense", units: 32, activation: "relu", inputDim: 64 },
        { type: "dense", units: 16, activation: "relu" },
        { type: "dense", units: 2, activation: "sigmoid" },
      ],
    },
    lr: 0.12, epochs: 200, val: 0.3,
  },
  {
    name: "dense16-8-2",
    config: {
      layers: [
        { type: "dense", units: 16, activation: "relu", inputDim: 64 },
        { type: "dense", units: 8, activation: "relu" },
        { type: "dense", units: 2, activation: "sigmoid" },
      ],
    },
    lr: 0.15, epochs: 200, val: 0.3,
  },
  {
    name: "dense8-2",
    config: {
      layers: [
        { type: "dense", units: 8, activation: "relu", inputDim: 64 },
        { type: "dense", units: 2, activation: "sigmoid" },
      ],
    },
    lr: 0.2, epochs: 200, val: 0.3,
  },
  {
    name: "dense128-32-2",
    config: {
      layers: [
        { type: "dense", units: 128, activation: "relu", inputDim: 64 },
        { type: "dense", units: 32, activation: "relu" },
        { type: "dense", units: 2, activation: "sigmoid" },
      ],
    },
    lr: 0.1, epochs: 200, val: 0.3,
  },
];

const RUNS = 6;
for (const s of specs) {
  const params = countParams(s.config, "shifted_bars");
  const trains: number[] = [];
  const vals: number[] = [];
  for (let r = 0; r < RUNS; r++) {
    const { history } = createAndTrain(s.config, {
      learningRate: s.lr,
      epochs: s.epochs,
      dataset: "shifted_bars",
      shuffle: true,
      valRatio: s.val,
    }, { includeDecisionBoundary: false });
    trains.push(history.final.accuracy);
    vals.push(history.final.valAccuracy ?? NaN);
  }
  console.log(
    `${s.name.padEnd(16)} params=${String(params).padStart(5)}  train ${ (mean(trains)*100).toFixed(1)}% [${(min(trains)*100).toFixed(0)}-${(max(trains)*100).toFixed(0)}]  val ${(mean(vals)*100).toFixed(1)}% [${(min(vals)*100).toFixed(0)}-${(max(vals)*100).toFixed(0)}]`
  );
}
