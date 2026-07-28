import type {
  LayerWeights,
  ModelExport,
  NetworkConfig,
  TrainConfig,
} from "./types";

export function buildModelExport(
  name: string,
  config: NetworkConfig,
  trainConfig: TrainConfig,
  weights: LayerWeights[],
  metrics?: { loss: number; accuracy: number }
): ModelExport {
  return {
    format: "neuralbasic-model-v1",
    name,
    config,
    trainConfig,
    weights,
    metrics,
    exportedAt: new Date().toISOString(),
  };
}

export function modelExportToJSON(exp: ModelExport): string {
  return JSON.stringify(exp, null, 2);
}

export function parseModelExport(json: string): ModelExport {
  const data = JSON.parse(json) as ModelExport;
  if (data.format !== "neuralbasic-model-v1") {
    throw new Error("Unsupported model export format");
  }
  return data;
}

/** Generate a simple PyTorch-equivalent snippet for educational export. */
export function toPyTorchSnippet(
  config: NetworkConfig,
  trainConfig: TrainConfig
): string {
  const lines: string[] = [];
  lines.push("import torch");
  lines.push("import torch.nn as nn");
  lines.push("import torch.optim as optim");
  lines.push("");
  lines.push(`# Auto-generated from NeuralBASIC — ${config.name ?? "Net"}`);
  lines.push("# Educational equivalent; not a bit-exact port.");
  lines.push("");
  lines.push("class Net(nn.Module):");
  lines.push("    def __init__(self):");
  lines.push("        super().__init__()");
  lines.push("        layers = []");

  let prev: number | string = "in_features";
  let idx = 0;
  for (const l of config.layers) {
    if (l.type === "dense") {
      const inF = l.inputDim ?? prev;
      const act =
        l.activation === "relu"
          ? "nn.ReLU()"
          : l.activation === "tanh"
            ? "nn.Tanh()"
            : l.activation === "sigmoid"
              ? "nn.Sigmoid()"
              : null;
      lines.push(
        `        layers.append(nn.Linear(${inF}, ${l.units}))`
      );
      if (act) lines.push(`        layers.append(${act})`);
      prev = l.units;
      idx++;
    } else if (l.type === "conv2d") {
      lines.push(
        `        layers.append(nn.Conv2d(${l.inputChannels ?? 1}, ${l.filters}, kernel_size=${l.kernelSize}))`
      );
      if ((l.activation ?? "relu") === "relu")
        lines.push("        layers.append(nn.ReLU())");
      idx++;
    } else if (l.type === "flatten") {
      lines.push("        layers.append(nn.Flatten())");
    } else if (l.type === "pool") {
      if (l.global) {
        lines.push(
          l.mode === "max"
            ? "        layers.append(nn.AdaptiveMaxPool2d(1))"
            : "        layers.append(nn.AdaptiveAvgPool2d(1))"
        );
      } else {
        const size = l.size ?? 2;
        const stride = l.stride ?? size;
        lines.push(
          l.mode === "max"
            ? `        layers.append(nn.MaxPool2d(kernel_size=${size}, stride=${stride}))`
            : `        layers.append(nn.AvgPool2d(kernel_size=${size}, stride=${stride}))`
        );
      }
    } else if (l.type === "attention" || l.type === "transformer_block") {
      const d = l.dModel;
      const h = l.nHeads ?? 2;
      lines.push(
        `        # Multi-head attention / transformer block (d_model=${d}, heads=${h})`
      );
      lines.push(
        `        self.attn_${idx} = nn.MultiheadAttention(${d}, ${h}, batch_first=True)`
      );
      prev = d;
      idx++;
    }
  }
  lines.push("        self.net = nn.Sequential(*layers)");
  lines.push("");
  lines.push("    def forward(self, x):");
  lines.push("        return self.net(x)");
  lines.push("");
  lines.push("model = Net()");
  lines.push(
    `opt = optim.SGD(model.parameters(), lr=${trainConfig.learningRate}${
      config.l2 ? `, weight_decay=${config.l2}` : ""
    })`
  );
  lines.push("loss_fn = nn.BCELoss()  # or CrossEntropyLoss for multi-class");
  lines.push("");
  lines.push(`# Train loop sketch (${trainConfig.epochs} epochs, dataset=${trainConfig.dataset})`);
  lines.push(`for epoch in range(${trainConfig.epochs}):`);
  lines.push("    opt.zero_grad()");
  lines.push("    # pred = model(batch_x)");
  lines.push("    # loss = loss_fn(pred, batch_y)");
  lines.push("    # loss.backward(); opt.step()");
  lines.push("    pass");
  lines.push("");
  return lines.join("\n");
}
