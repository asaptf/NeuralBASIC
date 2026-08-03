import { describe, expect, it } from "vitest";
import {
  DSLInputShapeError,
  DSLParseError,
  defaultStarterDSL,
  parseDSL,
  toDSL,
} from "./dsl";
import { exportWeights } from "./model";
import { checkInputShape } from "./shape";
import { createAndTrain, prepareNetworkConfig } from "./train";
import type { NetworkConfig } from "./types";

const DENSE_MISMATCH = `network Perceptron {
  dense 2 -> 1 activation=sigmoid
}
train dataset=shifted_bars lr=0.8 epochs=50
`;

const DENSE_FIXED = `network Perceptron {
  dense 64 -> 1 activation=sigmoid
}
train dataset=shifted_bars lr=0.8 epochs=30
`;

const CONV_MISMATCH = `network ShiftedCNN {
  conv2d filters=4 kernel=3 activation=relu channels=1 height=8 width=8
  pool mode=avg global=true
  dense 2 activation=sigmoid
}
train dataset=circles lr=0.2 epochs=50
`;

const CONV_FIXED = `network ShiftedCNN {
  conv2d filters=4 kernel=3 activation=relu channels=1 height=8 width=8
  pool mode=avg global=true
  dense 2 activation=sigmoid
}
train dataset=shifted_bars lr=0.2 epochs=40
`;

/** Product matches tiny_images (16) but geometry is 2×8, not the engine's 4×4. */
const CONV_GEOMETRY_MISMATCH = `network TinyCNN {
  conv2d filters=2 kernel=2 activation=relu channels=1 height=2 width=8
  pool mode=avg global=true
  dense 2 activation=sigmoid
}
train dataset=tiny_images lr=0.2 epochs=20
`;

describe("checkInputShape", () => {
  it("returns null for compatible chapter starters", () => {
    for (const id of ["ch1", "ch2", "ch3", "ch4", "ch5"] as const) {
      const parsed = parseDSL(defaultStarterDSL(id));
      expect(checkInputShape(parsed.network, parsed.train.dataset)).toBeNull();
    }
  });

  it("returns null when dense inputDim is unset (inference path)", () => {
    const network: NetworkConfig = {
      layers: [{ type: "dense", units: 4, activation: "relu" }],
    };
    expect(checkInputShape(network, "shifted_bars")).toBeNull();
  });

  it("returns null when bare conv2d infers from a spatial dataset (shifted_bars)", () => {
    const network: NetworkConfig = {
      layers: [
        { type: "conv2d", filters: 4, kernelSize: 3, activation: "relu" },
      ],
    };
    expect(checkInputShape(network, "shifted_bars")).toBeNull();
  });

  it("returns null when bare conv2d infers from tiny_images (1×4×4)", () => {
    const network: NetworkConfig = {
      layers: [
        { type: "conv2d", filters: 2, kernelSize: 2, activation: "relu" },
      ],
    };
    expect(checkInputShape(network, "tiny_images")).toBeNull();
  });

  it("rejects bare conv2d on a 1-D dataset (xor) without parroting inferred dims", () => {
    const network: NetworkConfig = {
      layers: [
        { type: "conv2d", filters: 4, kernelSize: 3, activation: "relu" },
      ],
    };
    const m = checkInputShape(network, "xor");
    expect(m).not.toBeNull();
    // Honest report: samples are not an image — not "declares 2×4×4 = 32".
    expect(m!.message).toBe(
      "Input shape mismatch: dataset `xor` samples are 2 values, " +
        "not an image a conv2d can read. " +
        "Use a spatial dataset (tiny_images or shifted_bars), " +
        "or start with a dense layer instead."
    );
    expect(m!.message).not.toMatch(/2×4×4|declares/);
  });

  it("rejects bare conv2d on negation (flat=16) the same way", () => {
    const network: NetworkConfig = {
      layers: [
        { type: "conv2d", filters: 2, kernelSize: 2, activation: "relu" },
      ],
    };
    const m = checkInputShape(network, "negation");
    expect(m).not.toBeNull();
    expect(m!.message).toBe(
      "Input shape mismatch: dataset `negation` samples are 16 values, " +
        "not an image a conv2d can read. " +
        "Use a spatial dataset (tiny_images or shifted_bars), " +
        "or start with a dense layer instead."
    );
    // Must not advertise the nonsense channel count prepareNetworkConfig would invent.
    expect(m!.message).not.toMatch(/16×4×4|256/);
  });

  it("returns null for 1×8×8 on shifted_bars", () => {
    const network: NetworkConfig = {
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
      ],
    };
    expect(checkInputShape(network, "shifted_bars")).toBeNull();
  });

  it("returns null for 1×4×4 on tiny_images", () => {
    const network: NetworkConfig = {
      layers: [
        {
          type: "conv2d",
          filters: 2,
          kernelSize: 2,
          activation: "relu",
          inputChannels: 1,
          inputHeight: 4,
          inputWidth: 4,
        },
      ],
    };
    expect(checkInputShape(network, "tiny_images")).toBeNull();
  });

  it("describes dense vs shifted_bars mismatch with sizes and alternatives", () => {
    const network: NetworkConfig = {
      layers: [
        {
          type: "dense",
          units: 1,
          inputDim: 2,
          activation: "sigmoid",
        },
      ],
    };
    const m = checkInputShape(network, "shifted_bars");
    expect(m).not.toBeNull();
    expect(m!.layerIndex).toBe(0);
    expect(m!.declaredSize).toBe(2);
    expect(m!.datasetSize).toBe(64);
    expect(m!.message).toMatch(/2/);
    expect(m!.message).toMatch(/shifted_bars/);
    expect(m!.message).toMatch(/64/);
    expect(m!.message).toMatch(/1×8×8/);
    expect(m!.message).toMatch(/xor|and|or|moons|circles/i);
  });

  it("describes conv2d vs circles mismatch with C×H×W form", () => {
    const network: NetworkConfig = {
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
      ],
    };
    const m = checkInputShape(network, "circles");
    expect(m).not.toBeNull();
    expect(m!.layerIndex).toBe(0);
    expect(m!.declaredSize).toBe(64);
    expect(m!.datasetSize).toBe(2);
    expect(m!.message).toMatch(/1×8×8/);
    expect(m!.message).toMatch(/circles/);
    expect(m!.message).toMatch(/shifted_bars/);
    // Size case — not the geometry wording.
    expect(m!.message).toMatch(/Input shape mismatch/);
    expect(m!.message).not.toMatch(/Input geometry mismatch/);
  });

  it("rejects conv geometry when product matches but H×W does not (2×8 on tiny_images)", () => {
    const network: NetworkConfig = {
      layers: [
        {
          type: "conv2d",
          filters: 2,
          kernelSize: 2,
          activation: "relu",
          inputChannels: 1,
          inputHeight: 2,
          inputWidth: 8,
        },
      ],
    };
    const m = checkInputShape(network, "tiny_images");
    expect(m).not.toBeNull();
    // Same flat size — this is the geometry hole, not the product hole.
    expect(m!.declaredSize).toBe(16);
    expect(m!.datasetSize).toBe(16);
    expect(m!.message).toBe(
      "Input geometry mismatch: this layer declares 1×2×8 = 16, " +
        "but the engine reads each `tiny_images` sample as 1×4×4. " +
        "Use `channels=1 height=4 width=4` on this conv2d."
    );
  });

  it("does not advise a square that would not read the sample either", () => {
    // 2 channels over 64 values has no square reading (sqrt(32) is not an
    // integer), so the engine's rounded 2×6×6 is an artefact — pointing the
    // learner at it would be advice that fails the same way.
    const network: NetworkConfig = {
      layers: [
        {
          type: "conv2d",
          filters: 2,
          kernelSize: 2,
          activation: "relu",
          inputChannels: 2,
          inputHeight: 4,
          inputWidth: 8,
        },
      ],
    };
    const m = checkInputShape(network, "shifted_bars");
    expect(m).not.toBeNull();
    expect(m!.declaredSize).toBe(64);
    expect(m!.datasetSize).toBe(64);
    expect(m!.message).toMatch(/Input geometry mismatch/);
    expect(m!.message).not.toMatch(/2×6×6/);
    // Advises the dataset's own shape instead.
    expect(m!.message).toContain("channels=1 height=8 width=8");
  });

  it("size-mismatch alternatives for conv only name geometry-compatible datasets", () => {
    // 1×2×8 = 16 on circles: size-only matches are tiny_images and negation,
    // but both are read as 1×4×4 — switching to either fails geometry.
    const network: NetworkConfig = {
      layers: [
        {
          type: "conv2d",
          filters: 2,
          kernelSize: 2,
          activation: "relu",
          inputChannels: 1,
          inputHeight: 2,
          inputWidth: 8,
        },
      ],
    };
    const m = checkInputShape(network, "circles");
    expect(m).not.toBeNull();
    expect(m!.declaredSize).toBe(16);
    expect(m!.datasetSize).toBe(2);
    expect(m!.message).toMatch(/Input shape mismatch/);
    expect(m!.message).toContain("1×2×8 = 16");
    expect(m!.message).toContain("circles");
    // Must not advertise datasets the geometry rule would then reject.
    expect(m!.message).not.toMatch(/tiny_images|negation/);
    expect(m!.message).toContain(
      "no built-in dataset matches this geometry with 16 features"
    );
  });

  it("still names geometry-compatible size alternatives for a square declaration", () => {
    // 1×4×4 = 16 on circles: tiny_images (and negation, same flat+square) work.
    const network: NetworkConfig = {
      layers: [
        {
          type: "conv2d",
          filters: 2,
          kernelSize: 2,
          activation: "relu",
          inputChannels: 1,
          inputHeight: 4,
          inputWidth: 4,
        },
      ],
    };
    const m = checkInputShape(network, "circles");
    expect(m).not.toBeNull();
    expect(m!.message).toMatch(/Input shape mismatch/);
    expect(m!.message).toMatch(/tiny_images/);
    // Dense-style size list still allowed when geometry agrees.
    expect(m!.message).not.toMatch(/no built-in dataset matches this geometry/);
  });

  it("walks past leading flatten so a mismatched dense is still rejected", () => {
    // flatten on flat input is a no-op — dense still faces the dataset.
    const network: NetworkConfig = {
      layers: [
        { type: "flatten" },
        {
          type: "dense",
          units: 1,
          inputDim: 64,
          activation: "sigmoid",
        },
      ],
    };
    const m = checkInputShape(network, "xor");
    expect(m).not.toBeNull();
    expect(m!.layerIndex).toBe(1);
    expect(m!.declaredSize).toBe(64);
    expect(m!.datasetSize).toBe(2);
    expect(m!.message).toBe(
      "Input size mismatch: this layer declares 64, " +
        "but dataset `xor` supplies 2. " +
        "Use `dense 2 -> 1 activation=sigmoid`, " +
        "or a dataset with 64 features (shifted_bars)."
    );
  });

  it("accepts leading flatten in front of a matching dense declaration", () => {
    const network: NetworkConfig = {
      layers: [
        { type: "flatten" },
        {
          type: "dense",
          units: 1,
          inputDim: 2,
          activation: "sigmoid",
        },
      ],
    };
    expect(checkInputShape(network, "xor")).toBeNull();
  });

  it("rejects flatten then conv with height/width only on xor (createModel defaults)", () => {
    // prepareNetworkConfig only fills channels from inputShape at i===0.
    // After a leading flatten the conv is at index 1, so createModel uses
    // inputChannels ?? 1 — not xor's inputShape[0]=2. Declaring height=1
    // width=1 with no channels must be seen as 1×1×1 (size 1), not 2×1×1.
    const network: NetworkConfig = {
      layers: [
        { type: "flatten" },
        {
          type: "conv2d",
          filters: 1,
          kernelSize: 1,
          activation: "relu",
          inputHeight: 1,
          inputWidth: 1,
        },
      ],
    };
    const m = checkInputShape(network, "xor");
    expect(m).not.toBeNull();
    expect(m!.layerIndex).toBe(1);
    expect(m!.declaredSize).toBe(1);
    expect(m!.datasetSize).toBe(2);
    expect(m!.message).toBe(
      "Input shape mismatch: this layer declares 1×1×1 = 1, " +
        "but dataset `xor` supplies 2. " +
        "Match the layer to the dataset (flat size 2), " +
        "or change the declared shape (no built-in dataset matches this geometry with 1 features)."
    );
    // Must not pretend channels came from the 1-D inputShape.
    expect(m!.message).not.toMatch(/2×1×1/);
  });

  it("still accepts bare conv2d after leading flatten on spatial datasets", () => {
    // createModel squares flatDim/channels after flatten — same as a first
    // layer bare conv on a true image dataset once channels default to 1.
    for (const dataset of ["tiny_images", "shifted_bars"] as const) {
      const network: NetworkConfig = {
        layers: [
          { type: "flatten" },
          {
            type: "conv2d",
            filters: 2,
            kernelSize: 2,
            activation: "relu",
          },
        ],
      };
      expect(checkInputShape(network, dataset)).toBeNull();
    }
  });

  it("does not treat leading pool as transparent", () => {
    // pool changes size — it is not a no-op, so we must not walk past it
    // and validate a later dense against the raw dataset size.
    const network: NetworkConfig = {
      layers: [
        { type: "pool", mode: "avg", global: true },
        {
          type: "dense",
          units: 1,
          inputDim: 64,
          activation: "sigmoid",
        },
      ],
    };
    expect(checkInputShape(network, "xor")).toBeNull();
  });
});

describe("input shape mismatch never reaches training", () => {
  it("rejects dense/shifted_bars repro at parse time", () => {
    expect(() => parseDSL(DENSE_MISMATCH)).toThrow(DSLParseError);
    expect(() => parseDSL(DENSE_MISMATCH)).toThrow(DSLInputShapeError);
    try {
      parseDSL(DENSE_MISMATCH);
    } catch (e) {
      const pe = e as DSLParseError;
      expect(pe).toBeInstanceOf(DSLInputShapeError);
      expect(pe.line).toBe(2);
      expect(pe.message).toMatch(/2/);
      expect(pe.message).toMatch(/shifted_bars/);
      expect(pe.message).toMatch(/64|1×8×8/);
      expect(pe.message).toMatch(/^Line 2:/);
    }
  });

  it("rejects conv/circles repro at parse time", () => {
    expect(() => parseDSL(CONV_MISMATCH)).toThrow(DSLParseError);
    expect(() => parseDSL(CONV_MISMATCH)).toThrow(DSLInputShapeError);
    try {
      parseDSL(CONV_MISMATCH);
    } catch (e) {
      const pe = e as DSLParseError;
      expect(pe).toBeInstanceOf(DSLInputShapeError);
      expect(pe.line).toBe(2);
      expect(pe.message).toMatch(/64|1×8×8/);
      expect(pe.message).toMatch(/circles/);
      expect(pe.message).toMatch(/shifted_bars/);
    }
  });

  it("rejects conv geometry mismatch at parse time", () => {
    expect(() => parseDSL(CONV_GEOMETRY_MISMATCH)).toThrow(DSLParseError);
    expect(() => parseDSL(CONV_GEOMETRY_MISMATCH)).toThrow(DSLInputShapeError);
    try {
      parseDSL(CONV_GEOMETRY_MISMATCH);
    } catch (e) {
      const pe = e as DSLInputShapeError;
      expect(pe).toBeInstanceOf(DSLInputShapeError);
      expect(pe).toBeInstanceOf(DSLParseError);
      expect(pe.line).toBe(2);
      expect(pe.message).toMatch(/^Line 2:/);
      expect(pe.message).toContain(
        "Input geometry mismatch: this layer declares 1×2×8 = 16, " +
          "but the engine reads each `tiny_images` sample as 1×4×4. " +
          "Use `channels=1 height=4 width=4` on this conv2d."
      );
    }
  });

  it("rejects bare conv2d + 1-D dataset at parse time", () => {
    const src = `network N {
  conv2d filters=4 kernel=3 activation=relu
  flatten
  dense 2 activation=sigmoid
}
train dataset=xor lr=0.2 epochs=10
`;
    expect(() => parseDSL(src)).toThrow(DSLInputShapeError);
    try {
      parseDSL(src);
    } catch (e) {
      const pe = e as DSLInputShapeError;
      expect(pe.line).toBe(2);
      expect(pe.message).toContain(
        "dataset `xor` samples are 2 values, not an image a conv2d can read"
      );
      expect(pe.message).toContain("tiny_images or shifted_bars");
      expect(pe.message).toContain("dense layer");
    }
  });

  it("rejects leading flatten + mismatched dense at the dense line (not flatten)", () => {
    // Repro: flatten hid layers[0] so dense 64 on xor trained with NaN.
    const src = `network N {
  flatten
  dense 64 -> 1 activation=sigmoid
}
train dataset=xor lr=0.5 epochs=3
`;
    expect(() => parseDSL(src)).toThrow(DSLInputShapeError);
    try {
      parseDSL(src);
    } catch (e) {
      const pe = e as DSLInputShapeError;
      expect(pe).toBeInstanceOf(DSLInputShapeError);
      // layerIndex=1 → source line of `dense`, not the leading `flatten`.
      expect(pe.line).toBe(3);
      expect(pe.message).toMatch(/^Line 3:/);
      expect(pe.message).toContain(
        "Input size mismatch: this layer declares 64, " +
          "but dataset `xor` supplies 2. " +
          "Use `dense 2 -> 1 activation=sigmoid`, " +
          "or a dataset with 64 features (shifted_bars)."
      );
    }
  });

  it("accepts leading flatten + matching dense declaration", () => {
    expect(() =>
      parseDSL(`network N {
  flatten
  dense 2 -> 1 activation=sigmoid
}
train dataset=xor lr=0.5 epochs=3
`)
    ).not.toThrow();
  });

  it("rejects flatten + conv height/width-only on xor at parse time", () => {
    // Same position-dependent fallback hole: channels must not be taken from
    // xor's inputShape when the conv is not layers[0].
    const src = `network N {
  flatten
  conv2d filters=1 kernel=1 activation=relu height=1 width=1
  dense 2 activation=sigmoid
}
train dataset=xor lr=0.2 epochs=3
`;
    expect(() => parseDSL(src)).toThrow(DSLInputShapeError);
    try {
      parseDSL(src);
    } catch (e) {
      const pe = e as DSLInputShapeError;
      expect(pe.line).toBe(3);
      expect(pe.message).toMatch(/^Line 3:/);
      expect(pe.message).toContain("1×1×1 = 1");
      expect(pe.message).toContain("dataset `xor` supplies 2");
      expect(pe.message).not.toMatch(/2×1×1/);
    }
  });

  it("accepts bare conv2d + tiny_images at parse time", () => {
    expect(() =>
      parseDSL(`network N {
  conv2d filters=2 kernel=2 activation=relu
  pool mode=avg global=true
  dense 2 activation=sigmoid
}
train dataset=tiny_images lr=0.2 epochs=10
`)
    ).not.toThrow();
  });

  it("corrected dense program trains with finite loss and weights (no NaN corruption)", () => {
    const parsed = parseDSL(DENSE_FIXED);
    const { history, weights } = createAndTrain(parsed.network, {
      ...parsed.train,
      epochs: 20,
    });
    const last = history.losses[history.losses.length - 1]!;
    expect(Number.isFinite(last)).toBe(true);
    expect(Number.isNaN(last)).toBe(false);
    for (const layer of weights) {
      if (layer.weights) {
        for (const row of layer.weights) {
          for (const w of row) {
            expect(Number.isFinite(w)).toBe(true);
            expect(Number.isNaN(w)).toBe(false);
          }
        }
      }
      if (layer.biases) {
        for (const b of layer.biases) {
          expect(Number.isFinite(b)).toBe(true);
        }
      }
    }
    // Dense first layer must stay width-64, not silently grow.
    expect(weights[0]?.weights?.[0]?.length).toBe(64);
  });

  it("corrected conv program trains with finite loss and weights", () => {
    const parsed = parseDSL(CONV_FIXED);
    const { history, weights } = createAndTrain(parsed.network, {
      ...parsed.train,
      epochs: 15,
    });
    const last = history.losses[history.losses.length - 1]!;
    expect(Number.isFinite(last)).toBe(true);
    for (const layer of weights) {
      if (layer.weights) {
        for (const row of layer.weights) {
          for (const w of row) {
            expect(Number.isFinite(w)).toBe(true);
          }
        }
      }
    }
    // Sanity: exportWeights still returns something usable.
    expect(weights.length).toBeGreaterThan(0);
  });
});

describe("DSLInputShapeError is distinguishable from syntax DSLParseError", () => {
  it("shape mismatch is DSLInputShapeError (and still a DSLParseError)", () => {
    let err: unknown;
    try {
      parseDSL(DENSE_MISMATCH);
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(DSLParseError);
    expect(err).toBeInstanceOf(DSLInputShapeError);
    expect((err as Error).name).toBe("DSLInputShapeError");
  });

  it("bad activation is DSLParseError but not DSLInputShapeError", () => {
    let err: unknown;
    try {
      parseDSL(`network N {
  dense 2 -> 1 activation=nope
}
train dataset=xor lr=0.5 epochs=10
`);
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(DSLParseError);
    expect(err).not.toBeInstanceOf(DSLInputShapeError);
    expect((err as Error).name).toBe("DSLParseError");
  });

  it("unclosed network block is DSLParseError but not DSLInputShapeError", () => {
    let err: unknown;
    try {
      parseDSL(`network N {
  dense 2 -> 1 activation=sigmoid
train dataset=xor lr=0.5 epochs=10
`);
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(DSLParseError);
    expect(err).not.toBeInstanceOf(DSLInputShapeError);
    expect((err as Error).name).toBe("DSLParseError");
  });
});

describe("unset input shape still infers from dataset", () => {
  it("infers dense inputDim from shifted_bars (64)", () => {
    const parsed = parseDSL(`network N {
  dense 4 activation=relu
  dense 1 activation=sigmoid
}
train dataset=shifted_bars lr=0.1 epochs=5
`);
    const { config, inputDim } = prepareNetworkConfig(
      parsed.network,
      parsed.train.dataset
    );
    expect(inputDim).toBe(64);
    expect(config.layers[0]).toMatchObject({
      type: "dense",
      units: 4,
      inputDim: 64,
    });
  });

  it("infers conv2d spatial dims from shifted_bars (1×8×8)", () => {
    const parsed = parseDSL(`network N {
  conv2d filters=4 kernel=3 activation=relu
  flatten
  dense 2 activation=sigmoid
}
train dataset=shifted_bars lr=0.1 epochs=5
`);
    const { config } = prepareNetworkConfig(
      parsed.network,
      parsed.train.dataset
    );
    expect(config.layers[0]).toMatchObject({
      type: "conv2d",
      inputChannels: 1,
      inputHeight: 8,
      inputWidth: 8,
    });
  });
});

describe("compatible programs stay green", () => {
  it("every chapter starter still parses", () => {
    for (const id of ["ch1", "ch2", "ch3", "ch4", "ch5"] as const) {
      expect(() => parseDSL(defaultStarterDSL(id))).not.toThrow();
    }
  });

  it("parseDSL(toDSL(...)) round-trips for starters", () => {
    for (const id of ["ch1", "ch2", "ch3", "ch4", "ch5"] as const) {
      const parsed = parseDSL(defaultStarterDSL(id));
      const again = parseDSL(toDSL(parsed.network, parsed.train));
      expect(again.network).toEqual(parsed.network);
      expect(again.train).toEqual(parsed.train);
    }
  });

  it("exportWeights after a short compatible train is finite", () => {
    const parsed = parseDSL(defaultStarterDSL("ch1"));
    const { model, history } = createAndTrain(parsed.network, {
      ...parsed.train,
      epochs: 10,
    });
    expect(Number.isFinite(history.final.loss)).toBe(true);
    const w = exportWeights(model);
    expect(w[0]?.weights?.[0]?.length).toBe(2);
  });
});
