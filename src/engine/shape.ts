import { DATASET_NAMES, getDataset } from "./datasets";
import type { DatasetName, LayerConfig, NetworkConfig } from "./types";

/**
 * Description of a declared-input / dataset mismatch.
 * Pure data — callers (parseDSL, the store) turn it into a DSLParseError or UI strip.
 */
export interface InputShapeMismatch {
  /** 0-based index of the first layer that declares a conflicting input shape. */
  layerIndex: number;
  /** Flat feature count the layer declares (product for spatial shapes). */
  declaredSize: number;
  /** Flat feature count the dataset supplies. */
  datasetSize: number;
  /** Ready-made human-readable sentence for the editor / DSLParseError. */
  message: string;
}

/** Effective first-layer input shape used by the size/geometry checks. */
interface EffectiveInput {
  size: number;
  shapeLabel: string;
  c?: number;
  h?: number;
  w?: number;
  /**
   * True when the learner wrote no channels/height/width on a conv2d — the
   * C×H×W above is purely runtime defaults (prepareNetworkConfig at index 0,
   * createModel elsewhere). Error text must not parrot those as declarations.
   */
  inferredOnly?: boolean;
}

function flatSize(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/** Format a dataset (or layer) shape for error text: `2` or `1×8×8 = 64`. */
function formatShape(shape: number[]): string {
  const flat = flatSize(shape);
  if (shape.length === 3) {
    return `${shape[0]}×${shape[1]}×${shape[2]} = ${flat}`;
  }
  if (shape.length === 1) return String(flat);
  return `${shape.join("×")} = ${flat}`;
}

function datasetFlatSize(name: DatasetName): number {
  return flatSize(getDataset(name).inputShape);
}

function datasetsWithSize(size: number): DatasetName[] {
  return DATASET_NAMES.filter((n) => datasetFlatSize(n) === size);
}

/**
 * Datasets whose flat size matches a declared conv shape *and* whose samples
 * the engine would reshape to the same C×H×W (square read from channels).
 * Listing a size-only match that the geometry rule then rejects is worse
 * than no advice.
 */
function datasetsCompatibleWithConv(declared: {
  size: number;
  c: number;
  h: number;
  w: number;
}): DatasetName[] {
  return DATASET_NAMES.filter((n) => {
    const size = datasetFlatSize(n);
    if (size !== declared.size) return false;
    const engine = engineConvReadShape(size, declared.c);
    return declared.h === engine.h && declared.w === engine.w;
  });
}

function formatDatasetList(names: DatasetName[]): string {
  if (names.length === 0) return "(none available)";
  if (names.length === 1) return names[0]!;
  if (names.length === 2) return `${names[0]} or ${names[1]}`;
  return `${names.slice(0, -1).join(", ")}, or ${names[names.length - 1]}`;
}

/** Built-in datasets with a true C×H×W layout (images a bare conv2d can read). */
function spatialDatasetNames(): DatasetName[] {
  return DATASET_NAMES.filter((n) => getDataset(n).inputShape.length === 3);
}

/**
 * How `forward` in model.ts reshapes a flat sample into CHW for the first
 * conv layer: always a square, `side = round(sqrt(flat / channels))`.
 * createModel sizes the dense tail from the *declared* H×W, so a product
 * match with different geometry corrupts weight-row widths at train time.
 */
function engineConvReadShape(
  datasetSize: number,
  channels: number
): { c: number; h: number; w: number; label: string } {
  const side = Math.round(Math.sqrt(datasetSize / channels));
  return {
    c: channels,
    h: side,
    w: side,
    label: `${channels}×${side}×${side}`,
  };
}

/**
 * Effective dataset-facing input shape, or null when there is nothing to check
 * (unset dense inputDim; non-input layers).
 *
 * For conv2d, unwritten dimensions follow the runtime path that actually runs:
 * - layerIndex === 0: prepareNetworkConfig fills channels/height/width from the
 *   dataset (train.ts, `i === 0` only).
 * - layerIndex > 0: prepareNetworkConfig leaves the layer alone; createModel
 *   defaults channels to 1 (when not already spatial) and, if height/width are
 *   not both declared, assumes a square side from the flat sample size.
 *
 * Using prepareNetworkConfig fallbacks for a conv that only becomes
 * dataset-facing after a skipped leading flatten would invent a shape the
 * engine never builds (e.g. channels from a 1-D inputShape on xor).
 */
function firstLayerEffectiveShape(
  layer: LayerConfig,
  dataset: DatasetName,
  layerIndex: number
): EffectiveInput | null {
  if (layer.type === "dense") {
    // Only validate when the learner wrote `dense N -> M` (inputDim set).
    if (layer.inputDim == null) return null;
    return { size: layer.inputDim, shapeLabel: String(layer.inputDim) };
  }

  if (layer.type === "conv2d") {
    const wroteAny =
      layer.inputChannels != null ||
      layer.inputHeight != null ||
      layer.inputWidth != null;

    let c: number;
    let h: number;
    let w: number;

    if (layerIndex === 0) {
      // Same defaults as prepareNetworkConfig (train.ts) for i === 0.
      const ds = getDataset(dataset);
      c = layer.inputChannels ?? ds.inputShape[0] ?? 1;
      h = layer.inputHeight ?? ds.inputShape[1] ?? 4;
      w = layer.inputWidth ?? ds.inputShape[2] ?? 4;
    } else {
      // createModel (model.ts): non-spatial, prepareNetworkConfig did not fill.
      c = layer.inputChannels ?? 1;
      if (layer.inputHeight != null && layer.inputWidth != null) {
        h = layer.inputHeight;
        w = layer.inputWidth;
      } else {
        const side = Math.round(Math.sqrt(datasetFlatSize(dataset) / c));
        h = side;
        w = side;
      }
    }

    return {
      size: c * h * w,
      shapeLabel: `${c}×${h}×${w} = ${c * h * w}`,
      c,
      h,
      w,
      inferredOnly: !wroteAny,
    };
  }

  // attention / transformer / flatten / pool: out of scope for this check.
  return null;
}

function denseFixSnippet(
  layer: Extract<LayerConfig, { type: "dense" }>,
  datasetSize: number
): string {
  const act = layer.activation ?? "sigmoid";
  return `dense ${datasetSize} -> ${layer.units} activation=${act}`;
}

function convFixSnippet(dataset: DatasetName): string | null {
  const shape = getDataset(dataset).inputShape;
  if (shape.length === 3) {
    return `channels=${shape[0]} height=${shape[1]} width=${shape[2]}`;
  }
  return null;
}

function convFixFromRead(c: number, h: number, w: number): string {
  return `channels=${c} height=${h} width=${w}`;
}

/**
 * Bare `conv2d` on a non-image (1-D) dataset: the engine would invent
 * channels=flat height=4 width=4 and then crash. Nothing was declared, so
 * do not echo that nonsense — say the samples are not an image and point
 * at spatial datasets or a dense first layer.
 */
function buildBareConvMessage(
  dataset: DatasetName,
  datasetSize: number
): string {
  const spatial = spatialDatasetNames();
  const spatialList = formatDatasetList(spatial);
  return (
    `Input shape mismatch: dataset \`${dataset}\` samples are ${datasetSize} values, ` +
    `not an image a conv2d can read. ` +
    `Use a spatial dataset (${spatialList}), or start with a dense layer instead.`
  );
}

function buildSizeMessage(
  layer: LayerConfig,
  declared: EffectiveInput,
  dataset: DatasetName,
  datasetSize: number,
  datasetLabel: string
): string {
  let compatible: DatasetName[];
  if (
    layer.type === "conv2d" &&
    declared.c != null &&
    declared.h != null &&
    declared.w != null
  ) {
    // Size-only matches that fail the geometry rule are not usable advice.
    compatible = datasetsCompatibleWithConv({
      size: declared.size,
      c: declared.c,
      h: declared.h,
      w: declared.w,
    });
  } else {
    compatible = datasetsWithSize(declared.size);
  }

  const datasetAlt =
    compatible.length > 0
      ? `or a dataset with ${declared.size} features (${formatDatasetList(compatible)})`
      : layer.type === "conv2d"
        ? `or change the declared shape (no built-in dataset matches this geometry with ${declared.size} features)`
        : `or change the declared size (no built-in dataset supplies ${declared.size} features)`;

  if (layer.type === "dense") {
    const fix = denseFixSnippet(layer, datasetSize);
    return (
      `Input size mismatch: this layer declares ${declared.shapeLabel}, ` +
      `but dataset \`${dataset}\` supplies ${datasetLabel}. ` +
      `Use \`${fix}\`, ${datasetAlt}.`
    );
  }

  // conv2d — product / size mismatch
  const convFix = convFixSnippet(dataset);
  const layerFix = convFix
    ? `Use \`${convFix}\` on this conv2d, ${datasetAlt}`
    : `Match the layer to the dataset (flat size ${datasetSize}), ${datasetAlt}`;
  return (
    `Input shape mismatch: this layer declares ${declared.shapeLabel}, ` +
    `but dataset \`${dataset}\` supplies ${datasetLabel}. ` +
    `${layerFix}.`
  );
}

/**
 * Product matches but H×W (and/or the square the engine will reshape to)
 * does not — a different failure mode from the size case.
 */
function buildGeometryMessage(
  declared: { shapeLabel: string },
  dataset: DatasetName,
  datasetSize: number,
  engine: { c: number; h: number; w: number; label: string }
): string {
  // The engine's square only spans the whole sample when `channels` divides it
  // into a square. With an unworkable channel count (2 channels over 64 values)
  // that square is a rounding artefact, so suggest the dataset's own shape
  // instead of a "fix" that would not read the sample either.
  const squareFits = engine.c * engine.h * engine.w === datasetSize;
  const fix = squareFits
    ? convFixFromRead(engine.c, engine.h, engine.w)
    : convFixSnippet(dataset);
  const advice = fix
    ? `Use \`${fix}\` on this conv2d.`
    : `No square reading of a ${datasetSize}-value sample exists for ${engine.c} channels — declare a channel count that squares.`;
  const reads = squareFits
    ? `the engine reads each \`${dataset}\` sample as ${engine.label}`
    : `\`${dataset}\` samples are ${datasetSize} values, which ${engine.c} channels cannot square`;
  return (
    `Input geometry mismatch: this layer declares ${declared.shapeLabel}, ` +
    `but ${reads}. ${advice}`
  );
}

/**
 * Pure, non-throwing check: does the network's dataset-facing input declaration
 * match what the dataset can supply?
 *
 * Dense: only when the learner wrote `dense N -> M` (inputDim set). Unset
 * dense inputDim stays free so prepareNetworkConfig can keep inferring.
 *
 * Conv2d: always checks the *effective* shape — declared dims plus the same
 * runtime defaults the engine will apply at that layer index (dataset fill
 * only for layers[0] via prepareNetworkConfig; createModel defaults after).
 * A bare `conv2d` on a 1-D dataset is rejected (the first-layer fallback path
 * invents channels=flat and then crashes in forwardConv); a bare `conv2d` on
 * a true image dataset still passes.
 *
 * Leading layers that cannot change the flat input size are skipped before
 * picking the layer to validate. Today that is only `flatten` (a no-op on
 * already-flat samples). `pool` resizes, and `conv2d` consumes the input —
 * neither is transparent. `layerIndex` always points at the declaring layer
 * so parseDSL can map it to the source line.
 *
 * For conv2d, matching the flat product is not enough: `forward` always
 * reshapes samples as a square (`side = round(sqrt(flat / channels))`), while
 * `createModel` sizes the dense tail from the declared H×W. A declaration
 * whose product matches but whose geometry does not must be rejected.
 *
 * @returns null when compatible, otherwise a mismatch description.
 */
export function checkInputShape(
  network: NetworkConfig,
  dataset: DatasetName
): InputShapeMismatch | null {
  const layers = network.layers;
  if (!layers.length) return null;

  // Walk past leading shape-preserving no-ops so a `flatten` in front of a
  // dense/conv declaration cannot hide the dataset-facing check.
  let layerIndex = 0;
  while (layerIndex < layers.length && layers[layerIndex]!.type === "flatten") {
    layerIndex++;
  }
  if (layerIndex >= layers.length) return null;

  const layer = layers[layerIndex]!;
  const declared = firstLayerEffectiveShape(layer, dataset, layerIndex);
  if (!declared) return null;

  const ds = getDataset(dataset);
  const datasetSize = flatSize(ds.inputShape);
  const datasetLabel = formatShape(ds.inputShape);

  const sizeMismatch = declared.size !== datasetSize;
  let geometryMismatch = false;
  let engine: ReturnType<typeof engineConvReadShape> | null = null;
  if (
    layer.type === "conv2d" &&
    declared.c != null &&
    declared.h != null &&
    declared.w != null
  ) {
    engine = engineConvReadShape(datasetSize, declared.c);
    if (declared.h !== engine.h || declared.w !== engine.w) {
      geometryMismatch = true;
    }
  }

  if (!sizeMismatch && !geometryMismatch) {
    return null;
  }

  // Bare conv: nothing was written — do not parrot prepareNetworkConfig's
  // invented C×H×W. Report that the samples are not an image.
  if (declared.inferredOnly) {
    return {
      layerIndex,
      declaredSize: declared.size,
      datasetSize,
      message: buildBareConvMessage(dataset, datasetSize),
    };
  }

  if (sizeMismatch) {
    return {
      layerIndex,
      declaredSize: declared.size,
      datasetSize,
      message: buildSizeMessage(
        layer,
        declared,
        dataset,
        datasetSize,
        datasetLabel
      ),
    };
  }

  // Same product — geometry does not match what forward reads.
  return {
    layerIndex,
    declaredSize: declared.size,
    datasetSize,
    message: buildGeometryMessage(declared, dataset, datasetSize, engine!),
  };
}
