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
  /**
   * Fraction of each class held out for validation metrics (0–1 exclusive).
   * Omitted → engine default (see DEFAULT_VAL_RATIO). Explicit 0 disables the split
   * (useful for tiny logic sets where a hold-out is meaningless).
   */
  valRatio?: number;
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
  | "tiny_text"
  /** Two moons + deterministic label noise — Chapter 3 overfitting demo. */
  | "noisy_moons"
  /**
   * Short vertical vs horizontal bars at many 8×8 translations — Chapter 4
   * demo where weight sharing beats a parameter-matched dense net.
   */
  | "shifted_bars";

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
  /** Mean train-set loss (samples the optimizer saw this run). */
  loss: number;
  /** Train-set accuracy. */
  accuracy: number;
  /**
   * Held-out loss for this epoch, or `null` when no validation split was applied
   * (tiny datasets, valRatio=0). Always present so UI can branch on null vs number.
   */
  valLoss?: number | null;
  /**
   * Held-out accuracy for this epoch, or `null` when no validation split.
   */
  valAccuracy?: number | null;
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
  /** Per-epoch train loss (same length as `accuracies`). */
  losses: number[];
  /** Per-epoch train accuracy. */
  accuracies: number[];
  /**
   * Per-epoch held-out loss. Empty when no validation split was applied;
   * otherwise same length as `losses`.
   */
  valLosses?: number[];
  /**
   * Per-epoch held-out accuracy. Empty when no validation split;
   * otherwise same length as `accuracies`.
   */
  valAccuracies?: number[];
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
