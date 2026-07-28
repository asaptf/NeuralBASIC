import type {
  ActivationName,
  DatasetName,
  LayerConfig,
  NetworkConfig,
  TrainConfig,
} from "./types";

export interface ParsedProgram {
  network: NetworkConfig;
  train: TrainConfig;
  raw: string;
}

/** Parse failure with a 1-based line number. `message` is self-sufficient for UI. */
export class DSLParseError extends Error {
  readonly line: number;

  constructor(message: string, line: number) {
    const full =
      line > 0 && !/^line\s+\d+/i.test(message)
        ? `Line ${line}: ${message}`
        : message;
    super(full);
    this.name = "DSLParseError";
    this.line = line;
  }
}

const ACTIVATION_LIST = [
  "linear",
  "sigmoid",
  "relu",
  "tanh",
  "softmax",
] as const;

const ACTIVATIONS = new Set<string>(ACTIVATION_LIST);

const DATASET_LIST = [
  "xor",
  "moons",
  "circles",
  "and",
  "or",
  "linear",
  "spiral",
  "tiny_images",
  "tiny_text",
  "noisy_moons",
  "shifted_bars",
  "negation",
] as const;

const DATASETS = new Set<string>(DATASET_LIST);

const LAYER_TYPES = new Set([
  "dense",
  "conv2d",
  "flatten",
  "pool",
  "attention",
  "transformer",
]);

const TRAIN_KEYS = new Set([
  "lr",
  "epochs",
  "dataset",
  "batch",
  "batchsize",
  "shuffle",
  "val",
  "valratio",
]);

const DENSE_KEYS = new Set(["activation"]);

const CONV_KEYS = new Set([
  "filters",
  "kernel",
  "kernelsize",
  "activation",
  "channels",
  "height",
  "width",
]);

const POOL_KEYS = new Set(["mode", "size", "stride", "global"]);

const POOL_MODES = new Set(["max", "avg", "average", "mean"]);

const ATTENTION_KEYS = new Set(["d_model", "dmodel", "heads"]);

const TRANSFORMER_KEYS = new Set(["d_model", "dmodel", "heads", "dff"]);

interface SourceLine {
  /** 1-based original line number in the source. */
  line: number;
  /** Comment-stripped, trimmed text (non-empty). */
  text: string;
}

/**
 * Minimal NeuralBASIC DSL
 *
 * network MyNet {
 *   dense 2 -> 4 activation=relu
 *   dense 4 -> 1 activation=sigmoid
 *   # or: conv2d filters=4 kernel=2 activation=relu
 *   # or: pool mode=max size=2 stride=2
 *   # or: pool mode=avg global=true
 *   # or: attention d_model=8 heads=2
 *   # or: transformer d_model=8 heads=2
 * }
 * train dataset=xor lr=0.5 epochs=200
 * # optional: l2=0.01 dropout=0.2
 */
export function parseDSL(source: string): ParsedProgram {
  const lines = preprocess(source);

  if (lines.length === 0) {
    throw new DSLParseError(
      "Program is empty. Expected a `network` block (e.g. `network MyNet { ... }`).",
      1
    );
  }

  let name = "Net";
  const layers: LayerConfig[] = [];
  let l2 = 0;
  let dropout = 0;
  let train: TrainConfig = {
    learningRate: 0.5,
    epochs: 100,
    dataset: "xor",
  };

  let inNetwork = false;
  let sawNetwork = false;
  let networkOpenLine = 0;

  for (const { line, text } of lines) {
    if (text === "}") {
      if (!inNetwork) {
        throw new DSLParseError("Unexpected `}`. No open `network` block.", line);
      }
      inNetwork = false;
      continue;
    }

    const netOpen = text.match(/^network\s+(\w+)\s*\{?$/i);
    if (netOpen) {
      if (inNetwork) {
        throw new DSLParseError(
          "Unexpected `network` while a previous `network` block is still open. Close it with `}`.",
          line
        );
      }
      name = netOpen[1]!;
      inNetwork = true;
      sawNetwork = true;
      networkOpenLine = line;
      continue;
    }

    // `network` with missing/invalid name
    if (/^network\b/i.test(text)) {
      throw new DSLParseError(
        "Malformed `network` header. Expected `network Name {` (name must be an identifier).",
        line
      );
    }

    if (/^train\b/i.test(text)) {
      if (inNetwork) {
        throw new DSLParseError(
          "`train` is not allowed inside a `network` block. Close the network with `}` first.",
          line
        );
      }
      train = { ...train, ...parseTrainLine(text, line) };
      continue;
    }

    const l2m = text.match(/^l2\s*=\s*(.+)$/i);
    if (l2m) {
      if (inNetwork) {
        throw new DSLParseError(
          "`l2` is a top-level setting; place it outside the `network` block.",
          line
        );
      }
      l2 = parseNonNegNumber(l2m[1]!.trim(), "l2", line);
      continue;
    }

    const drm = text.match(/^dropout\s*=\s*(.+)$/i);
    if (drm) {
      if (inNetwork) {
        throw new DSLParseError(
          "`dropout` is a top-level setting; place it outside the `network` block.",
          line
        );
      }
      dropout = parseUnitInterval(drm[1]!.trim(), "dropout", line);
      continue;
    }

    if (inNetwork) {
      layers.push(parseLayerLine(text, line));
      continue;
    }

    // Outside network: unknown top-level keyword / stray prose
    const first = text.split(/\s+/)[0] ?? text;
    if (LAYER_TYPES.has(first.toLowerCase())) {
      throw new DSLParseError(
        `Layer \`${first}\` must appear inside a \`network\` block.`,
        line
      );
    }
    throw new DSLParseError(
      `Unknown top-level keyword \`${first}\`. Expected \`network\`, \`train\`, \`l2=...\`, or \`dropout=...\`.`,
      line
    );
  }

  if (inNetwork) {
    throw new DSLParseError(
      "Unclosed `network` block. Expected a closing `}`.",
      networkOpenLine || lines[lines.length - 1]!.line
    );
  }

  if (!sawNetwork) {
    throw new DSLParseError(
      "Program has no `network` block. Expected `network Name { ... }`.",
      lines[0]!.line
    );
  }

  if (!layers.length) {
    throw new DSLParseError(
      "Network has no layers. Add at least one layer (e.g. `dense 2 -> 1 activation=sigmoid`).",
      networkOpenLine || lines[0]!.line
    );
  }

  return {
    network: { name, layers, l2, dropout },
    train,
    raw: source,
  };
}

function preprocess(source: string): SourceLine[] {
  const rawLines = source.split(/\r?\n/);
  const out: SourceLine[] = [];
  for (let i = 0; i < rawLines.length; i++) {
    const stripped = rawLines[i]!.replace(/#.*$/, "").trim();
    if (stripped) out.push(...expandBraces(i + 1, stripped));
  }
  return out;
}

/**
 * Split a physical line so `{` / `}` become their own logical lines (same 1-based
 * line number). Lets compact one-liners like `network P { dense 2 -> 1 }` parse
 * like the multi-line form without inventing new line numbers.
 */
function expandBraces(line: number, text: string): SourceLine[] {
  // Fast path: no braces, or a lone `}` (already a full logical line).
  if (!/[{}]/.test(text)) return [{ line, text }];
  if (text === "}") return [{ line, text }];

  // Tokenize on braces, keeping them as separate tokens.
  const tokens: string[] = [];
  let buf = "";
  for (const ch of text) {
    if (ch === "{" || ch === "}") {
      const piece = buf.trim();
      if (piece) tokens.push(piece);
      tokens.push(ch);
      buf = "";
    } else {
      buf += ch;
    }
  }
  const tail = buf.trim();
  if (tail) tokens.push(tail);

  // Re-attach `{` to a preceding `network Name` header so existing
  // `/^network\s+(\w+)\s*\{?$/i` matching still works.
  const result: SourceLine[] = [];
  for (const t of tokens) {
    if (t === "{") {
      const prev = result[result.length - 1];
      if (prev && /^network\s+\w+$/i.test(prev.text)) {
        result[result.length - 1] = { line, text: `${prev.text} {` };
      } else {
        // Lone / unexpected `{` — surface as its own token for clear errors.
        result.push({ line, text: "{" });
      }
    } else {
      result.push({ line, text: t });
    }
  }
  return result.length ? result : [{ line, text }];
}

function parseTrainLine(line: string, lineNo: number): Partial<TrainConfig> {
  // Strip leading `train`
  const rest = line.replace(/^train\b/i, "").trim();
  const kvs = parseKeyValues(rest, lineNo, "train");

  for (const key of kvs.keys()) {
    if (!TRAIN_KEYS.has(key.toLowerCase())) {
      throw new DSLParseError(
        `Unknown train parameter \`${key}\`. Valid keys: lr, epochs, dataset, batch, shuffle, val.`,
        lineNo
      );
    }
  }

  const out: Partial<TrainConfig> = {};

  if (kvs.has("lr")) {
    const v = kvs.get("lr")!;
    const n = parseFiniteNumber(v, "lr", lineNo);
    if (!(n > 0)) {
      throw new DSLParseError(
        `\`lr\` must be a positive number, got \`${v}\`.`,
        lineNo
      );
    }
    out.learningRate = n;
  }

  if (kvs.has("epochs")) {
    const v = kvs.get("epochs")!;
    const n = parsePositiveInt(v, "epochs", lineNo);
    out.epochs = n;
  }

  if (kvs.has("dataset")) {
    const v = kvs.get("dataset")!;
    if (!DATASETS.has(v)) {
      throw new DSLParseError(
        `Unknown dataset \`${v}\`. Valid datasets: ${DATASET_LIST.join(", ")}.`,
        lineNo
      );
    }
    out.dataset = v as DatasetName;
  }

  const batchRaw = kvs.get("batch") ?? kvs.get("batchsize");
  if (batchRaw !== undefined) {
    out.batchSize = parsePositiveInt(batchRaw, "batch", lineNo);
  }

  if (kvs.has("shuffle")) {
    out.shuffle = parseBoolean(kvs.get("shuffle")!, "shuffle", lineNo);
  }

  const valRaw = kvs.get("val") ?? kvs.get("valratio");
  if (valRaw !== undefined) {
    // Allow 0 (disable split) through values in (0, 1). Reject 1 and above —
    // a 100% hold-out would leave nothing to train on.
    const n = parseFiniteNumber(valRaw, "val", lineNo);
    if (n < 0 || n >= 1) {
      throw new DSLParseError(
        `\`val\` must be in [0, 1), got \`${valRaw}\`. Use 0 to disable the hold-out, e.g. \`val=0.25\`.`,
        lineNo
      );
    }
    out.valRatio = n;
  }

  // Reject leftover non-kv tokens after `train`
  if (rest && kvs.size === 0 && !/^\w+\s*=/.test(rest)) {
    // e.g. `train xor` without keys
    throw new DSLParseError(
      "Malformed `train` line. Expected key=value pairs (e.g. `train dataset=xor lr=0.5 epochs=100`).",
      lineNo
    );
  }

  return out;
}

function parseLayerLine(line: string, lineNo: number): LayerConfig {
  const first = (line.split(/\s+/)[0] ?? "").toLowerCase();

  if (first === "dense") return parseDense(line, lineNo);
  if (first === "conv2d") return parseConv2d(line, lineNo);
  if (first === "flatten") return parseFlatten(line, lineNo);
  if (first === "pool") return parsePool(line, lineNo);
  if (first === "attention") return parseAttention(line, lineNo);
  if (first === "transformer") return parseTransformer(line, lineNo);

  throw new DSLParseError(
    `Unknown layer type \`${first}\`. Valid types: dense, conv2d, flatten, pool, attention, transformer.`,
    lineNo
  );
}

function parseDense(line: string, lineNo: number): LayerConfig {
  // dense [N ->] M [activation=name]
  const m = line.match(
    /^dense(?:\s+(\S+)\s*->\s*(\S+)|\s+(\S+))?(?:\s+(.*))?$/i
  );
  if (!m || (m[1] == null && m[3] == null)) {
    throw new DSLParseError(
      "Malformed `dense` layer. Expected `dense units` or `dense input -> units` (e.g. `dense 2 -> 4 activation=relu`).",
      lineNo
    );
  }

  let inputDim: number | undefined;
  let units: number;
  let kvRest: string;

  if (m[1] != null) {
    inputDim = parsePositiveInt(m[1], "input width", lineNo);
    units = parsePositiveInt(m[2]!, "units", lineNo);
    kvRest = (m[4] ?? "").trim();
  } else {
    units = parsePositiveInt(m[3]!, "units", lineNo);
    kvRest = (m[4] ?? "").trim();
  }

  const kvs = parseKeyValues(kvRest, lineNo, "dense");
  for (const key of kvs.keys()) {
    if (!DENSE_KEYS.has(key.toLowerCase())) {
      throw new DSLParseError(
        `Unknown dense parameter \`${key}\`. Valid keys: activation.`,
        lineNo
      );
    }
  }

  let activation: ActivationName = "sigmoid";
  if (kvs.has("activation")) {
    activation = parseActivation(kvs.get("activation")!, lineNo);
  }

  return {
    type: "dense",
    units,
    inputDim,
    activation,
  };
}

function parseConv2d(line: string, lineNo: number): LayerConfig {
  const rest = line.replace(/^conv2d\b/i, "").trim();
  const kvs = parseKeyValues(rest, lineNo, "conv2d");

  for (const key of kvs.keys()) {
    if (!CONV_KEYS.has(key.toLowerCase())) {
      throw new DSLParseError(
        `Unknown conv2d parameter \`${key}\`. Valid keys: filters, kernel, activation, channels, height, width.`,
        lineNo
      );
    }
  }

  const filters = kvs.has("filters")
    ? parsePositiveInt(kvs.get("filters")!, "filters", lineNo)
    : 4;
  const kernelRaw = kvs.get("kernel") ?? kvs.get("kernelsize");
  const kernelSize = kernelRaw
    ? parsePositiveInt(kernelRaw, "kernel", lineNo)
    : 2;
  const actRaw = kvs.get("activation");
  const activation: ActivationName = actRaw
    ? parseActivation(actRaw, lineNo)
    : "relu";

  const inputChannels = kvs.has("channels")
    ? parsePositiveInt(kvs.get("channels")!, "channels", lineNo)
    : undefined;
  const inputHeight = kvs.has("height")
    ? parsePositiveInt(kvs.get("height")!, "height", lineNo)
    : undefined;
  const inputWidth = kvs.has("width")
    ? parsePositiveInt(kvs.get("width")!, "width", lineNo)
    : undefined;

  return {
    type: "conv2d",
    filters,
    kernelSize,
    activation,
    inputChannels,
    inputHeight,
    inputWidth,
  };
}

function parseFlatten(line: string, lineNo: number): LayerConfig {
  const rest = line.replace(/^flatten\b/i, "").trim();
  if (rest) {
    throw new DSLParseError(
      `\`flatten\` takes no parameters; unexpected \`${rest.split(/\s+/)[0]}\`.`,
      lineNo
    );
  }
  return { type: "flatten" };
}

function parsePool(line: string, lineNo: number): LayerConfig {
  const rest = line.replace(/^pool\b/i, "").trim();
  const kvs = parseKeyValues(rest, lineNo, "pool");

  for (const key of kvs.keys()) {
    if (!POOL_KEYS.has(key.toLowerCase())) {
      throw new DSLParseError(
        `Unknown pool parameter \`${key}\`. Valid keys: mode, size, stride, global.`,
        lineNo
      );
    }
  }

  const modeRaw = (kvs.get("mode") ?? "max").toLowerCase();
  if (!POOL_MODES.has(modeRaw)) {
    throw new DSLParseError(
      `Unknown pool mode \`${kvs.get("mode") ?? modeRaw}\`. Valid modes: max, avg.`,
      lineNo
    );
  }
  const mode: "max" | "avg" =
    modeRaw === "max" ? "max" : "avg";

  let global = false;
  if (kvs.has("global")) {
    global = parseBoolean(kvs.get("global")!, "global", lineNo);
  }

  let size: number | undefined;
  let stride: number | undefined;

  if (global) {
    // size/stride are ignored for global pool; reject only if contradictory nonsense.
    if (kvs.has("size")) {
      size = parsePositiveInt(kvs.get("size")!, "size", lineNo);
    }
    if (kvs.has("stride")) {
      stride = parsePositiveInt(kvs.get("stride")!, "stride", lineNo);
    }
  } else {
    size = kvs.has("size")
      ? parsePositiveInt(kvs.get("size")!, "size", lineNo)
      : 2;
    stride = kvs.has("stride")
      ? parsePositiveInt(kvs.get("stride")!, "stride", lineNo)
      : size;
  }

  return {
    type: "pool",
    mode,
    size,
    stride,
    global: global || undefined,
  };
}

function parseAttention(line: string, lineNo: number): LayerConfig {
  const rest = line.replace(/^attention\b/i, "").trim();
  const kvs = parseKeyValues(rest, lineNo, "attention");

  for (const key of kvs.keys()) {
    if (!ATTENTION_KEYS.has(key.toLowerCase())) {
      throw new DSLParseError(
        `Unknown attention parameter \`${key}\`. Valid keys: d_model, heads.`,
        lineNo
      );
    }
  }

  const dModelRaw = kvs.get("d_model") ?? kvs.get("dmodel");
  const dModel = dModelRaw
    ? parsePositiveInt(dModelRaw, "d_model", lineNo)
    : 8;
  const nHeads = kvs.has("heads")
    ? parsePositiveInt(kvs.get("heads")!, "heads", lineNo)
    : 2;

  return { type: "attention", dModel, nHeads };
}

function parseTransformer(line: string, lineNo: number): LayerConfig {
  const rest = line.replace(/^transformer\b/i, "").trim();
  const kvs = parseKeyValues(rest, lineNo, "transformer");

  for (const key of kvs.keys()) {
    if (!TRANSFORMER_KEYS.has(key.toLowerCase())) {
      throw new DSLParseError(
        `Unknown transformer parameter \`${key}\`. Valid keys: d_model, heads, dff.`,
        lineNo
      );
    }
  }

  const dModelRaw = kvs.get("d_model") ?? kvs.get("dmodel");
  const dModel = dModelRaw
    ? parsePositiveInt(dModelRaw, "d_model", lineNo)
    : 8;
  const nHeads = kvs.has("heads")
    ? parsePositiveInt(kvs.get("heads")!, "heads", lineNo)
    : 2;
  const dff = kvs.has("dff")
    ? parsePositiveInt(kvs.get("dff")!, "dff", lineNo)
    : dModel * 2;

  return { type: "transformer_block", dModel, nHeads, dff };
}

/** Parse `key=value` pairs; keys stored lowercased for case-insensitive lookup. */
function parseKeyValues(
  rest: string,
  lineNo: number,
  context: string
): Map<string, string> {
  const map = new Map<string, string>();
  if (!rest) return map;

  let remaining = rest;
  const pairRe = /^(\w+)\s*=\s*(\S+)(?:\s+|$)/;

  while (remaining.length > 0) {
    const m = remaining.match(pairRe);
    if (!m) {
      const bad = remaining.split(/\s+/)[0] ?? remaining;
      throw new DSLParseError(
        `Malformed ${context} parameter near \`${bad}\`. Expected key=value pairs.`,
        lineNo
      );
    }
    const key = m[1]!;
    const value = m[2]!;
    const lower = key.toLowerCase();
    if (map.has(lower)) {
      throw new DSLParseError(
        `Duplicate parameter \`${key}\` on ${context}.`,
        lineNo
      );
    }
    map.set(lower, value);
    remaining = remaining.slice(m[0].length).trimStart();
  }

  return map;
}

function parseActivation(raw: string, lineNo: number): ActivationName {
  const act = raw.toLowerCase();
  if (!ACTIVATIONS.has(act)) {
    throw new DSLParseError(
      `Unknown activation \`${raw}\`. Valid activations: ${ACTIVATION_LIST.join(", ")}.`,
      lineNo
    );
  }
  return act as ActivationName;
}

function parseFiniteNumber(raw: string, label: string, lineNo: number): number {
  if (!/^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$/.test(raw)) {
    throw new DSLParseError(
      `\`${label}\` must be a number, got \`${raw}\`.`,
      lineNo
    );
  }
  const n = Number(raw);
  if (!Number.isFinite(n)) {
    throw new DSLParseError(
      `\`${label}\` must be a finite number, got \`${raw}\`.`,
      lineNo
    );
  }
  return n;
}

function parseNonNegNumber(raw: string, label: string, lineNo: number): number {
  const n = parseFiniteNumber(raw, label, lineNo);
  if (n < 0) {
    throw new DSLParseError(
      `\`${label}\` must be ≥ 0, got \`${raw}\`.`,
      lineNo
    );
  }
  return n;
}

function parseUnitInterval(raw: string, label: string, lineNo: number): number {
  const n = parseFiniteNumber(raw, label, lineNo);
  if (n < 0 || n > 1) {
    throw new DSLParseError(
      `\`${label}\` must be between 0 and 1, got \`${raw}\`.`,
      lineNo
    );
  }
  return n;
}

function parsePositiveInt(raw: string, label: string, lineNo: number): number {
  if (!/^\d+$/.test(raw)) {
    throw new DSLParseError(
      `\`${label}\` must be a positive integer, got \`${raw}\`.`,
      lineNo
    );
  }
  const n = Number(raw);
  if (!Number.isInteger(n) || n <= 0) {
    throw new DSLParseError(
      `\`${label}\` must be a positive integer, got \`${raw}\`.`,
      lineNo
    );
  }
  return n;
}

function parseBoolean(raw: string, label: string, lineNo: number): boolean {
  const v = raw.toLowerCase();
  if (v === "true" || v === "1" || v === "yes") return true;
  if (v === "false" || v === "0" || v === "no") return false;
  throw new DSLParseError(
    `\`${label}\` must be true or false, got \`${raw}\`.`,
    lineNo
  );
}

/** Serialize config back to DSL (pretty). */
export function toDSL(network: NetworkConfig, train: TrainConfig): string {
  const lines: string[] = [];
  lines.push(`network ${network.name ?? "Net"} {`);
  for (const l of network.layers) {
    if (l.type === "dense") {
      const inPart = l.inputDim != null ? `${l.inputDim} -> ` : "";
      lines.push(
        `  dense ${inPart}${l.units} activation=${l.activation ?? "sigmoid"}`
      );
    } else if (l.type === "conv2d") {
      const parts = [
        `conv2d filters=${l.filters} kernel=${l.kernelSize} activation=${l.activation ?? "relu"}`,
      ];
      // Emit spatial input shape when present so parseDSL(toDSL(x)) keeps it.
      if (l.inputChannels != null) parts.push(`channels=${l.inputChannels}`);
      if (l.inputHeight != null) parts.push(`height=${l.inputHeight}`);
      if (l.inputWidth != null) parts.push(`width=${l.inputWidth}`);
      lines.push(`  ${parts.join(" ")}`);
    } else if (l.type === "flatten") {
      lines.push(`  flatten`);
    } else if (l.type === "pool") {
      const parts = [`pool mode=${l.mode}`];
      if (l.global) {
        parts.push("global=true");
      } else {
        parts.push(`size=${l.size ?? 2}`);
        const stride = l.stride ?? l.size ?? 2;
        parts.push(`stride=${stride}`);
      }
      lines.push(`  ${parts.join(" ")}`);
    } else if (l.type === "attention") {
      lines.push(
        `  attention d_model=${l.dModel} heads=${l.nHeads ?? 2}`
      );
    } else if (l.type === "transformer_block") {
      const parts = [
        `transformer d_model=${l.dModel} heads=${l.nHeads ?? 2}`,
      ];
      if (l.dff != null) parts.push(`dff=${l.dff}`);
      lines.push(`  ${parts.join(" ")}`);
    }
  }
  lines.push(`}`);
  if (network.l2) lines.push(`l2=${network.l2}`);
  if (network.dropout) lines.push(`dropout=${network.dropout}`);
  const trainParts = [
    `train dataset=${train.dataset}`,
    `lr=${train.learningRate}`,
    `epochs=${train.epochs}`,
  ];
  if (train.batchSize != null) trainParts.push(`batch=${train.batchSize}`);
  if (train.shuffle != null) trainParts.push(`shuffle=${train.shuffle}`);
  if (train.valRatio != null) trainParts.push(`val=${train.valRatio}`);
  lines.push(trainParts.join(" "));
  return lines.join("\n");
}

export function defaultStarterDSL(chapterId = "ch1"): string {
  switch (chapterId) {
    case "ch1":
      return `network Perceptron {
  dense 2 -> 1 activation=sigmoid
}
train dataset=xor lr=0.8 epochs=200
`;
    case "ch2":
      return `network MLP {
  dense 2 -> 8 activation=relu
  dense 8 -> 4 activation=relu
  dense 4 -> 1 activation=sigmoid
}
train dataset=xor lr=0.3 epochs=300
`;
    case "ch3":
      // Measured on noisy_moons (40 runs): ~96% train / ~64% val with this setup.
      // Fix direction: l2=0.005 (same net) or dense 2->6->1 raises held-out accuracy.
      return `network OverfitDemo {
  dense 2 -> 64 activation=relu
  dense 64 -> 64 activation=relu
  dense 64 -> 1 activation=sigmoid
}
l2=0.0
train dataset=noisy_moons lr=0.08 epochs=400 val=0.3
`;
    case "ch4":
      // The chapter's headline result: shared kernels plus a position-discarding
      // readout. 50 parameters reach ~97% held-out where a 2,642-parameter dense
      // net reaches ~86%. Swapping `pool` for `flatten` collapses it to ~62%.
      return `network ShiftedCNN {
  conv2d filters=4 kernel=3 activation=relu channels=1 height=8 width=8
  pool mode=avg global=true
  dense 2 activation=sigmoid
}
train dataset=shifted_bars lr=0.2 epochs=150 val=0.3
`;
    case "ch5":
      // d_model MUST divide the 16-feature input on token boundaries: negation is
      // 4 tokens of 4, so d_model=4. With d_model=8 you get two half-token
      // fragments and a 1×1 attention map, which teaches nothing.
      return `network TinyAttn {
  attention d_model=4 heads=2
  dense 4 -> 1 activation=sigmoid
}
train dataset=negation lr=0.1 epochs=80 val=0.3
`;
    default:
      return defaultStarterDSL("ch1");
  }
}
