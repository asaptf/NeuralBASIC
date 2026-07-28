/** Core neural engine types — pure TypeScript educational runtime. */

export type ActivationName =
  | "linear"
  | "sigmoid"
  | "relu"
  | "tanh"
  | "softmax";

export type LayerConfig =
  | {
      type: "dense";
      units: number;
      activation?: ActivationName;
      inputDim?: number;
    }
  | {
      type: "conv2d";
      filters: number;
      kernelSize: number;
      activation?: ActivationName;
      inputChannels?: number;
      inputHeight?: number;
      inputWidth?: number;
    }
  | {
      type: "flatten";
    }
  | {
      type: "attention";
      dModel: number;
      nHeads?: number;
    }
  | {
      type: "transformer_block";
      dModel: number;
      nHeads?: number;
      dff?: number;
    };

export interface NetworkConfig {
  name?: string;
  layers: LayerConfig[];
  /** L2 weight decay coefficient (0 = off). */
  l2?: number;
  /** Dropout rate 0–1 applied during training only (dense). */
  dropout?: number;
}

export interface TrainConfig {
  learningRate: number;
  epochs: number;
  batchSize?: number;
  dataset: DatasetName;
  shuffle?: boolean;
}

export type DatasetName =
  | "xor"
  | "moons"
  | "circles"
  | "and"
  | "or"
  | "linear"
  | "spiral"
  | "tiny_images"
  | "tiny_text";

export interface Sample {
  x: number[];
  y: number[];
}

export interface Dataset {
  name: DatasetName;
  samples: Sample[];
  inputShape: number[];
  outputDim: number;
  kind: "classification" | "regression";
  /** Optional 2D feature labels for decision boundary (first two dims). */
  featureNames?: [string, string];
}

export interface LayerSnapshot {
  type: string;
  weights?: number[][];
  biases?: number[];
  activations?: number[];
  /** Attention weights [heads][seq][seq] if present. */
  attention?: number[][][];
  shape?: number[];
}

export interface TrainStepResult {
  epoch: number;
  loss: number;
  accuracy: number;
  layerSnapshots: LayerSnapshot[];
  predictions?: number[][];
  decisionGrid?: {
    width: number;
    height: number;
    values: number[];
    xMin: number;
    xMax: number;
    yMin: number;
    yMax: number;
  };
  attentionMaps?: number[][][];
}

export interface TrainHistory {
  losses: number[];
  accuracies: number[];
  final: TrainStepResult;
}

export interface ModelExport {
  format: "neuralbasic-model-v1";
  name: string;
  config: NetworkConfig;
  trainConfig: TrainConfig;
  weights: LayerWeights[];
  metrics?: { loss: number; accuracy: number };
  exportedAt: string;
}

export interface LayerWeights {
  type: string;
  weights?: number[][];
  biases?: number[];
  /** Attention / transformer params as flat named matrices. */
  params?: Record<string, number[][]>;
}

export interface EngineState {
  config: NetworkConfig;
  trainConfig: TrainConfig;
  weights: LayerWeights[];
  history: { losses: number[]; accuracies: number[] };
  lastSnapshot: TrainStepResult | null;
  isTraining: boolean;
}
