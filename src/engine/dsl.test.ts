import { describe, expect, it } from "vitest";
import { CHAPTERS } from "@/curriculum/chapters";
import {
  DSLParseError,
  defaultStarterDSL,
  parseDSL,
  toDSL,
} from "./dsl";

/** Intended learner solutions from curriculum.completeness.test.ts (must keep parsing). */
const INTENDED_SOLUTION_DSLS: { id: string; dsl: string }[] = [
  {
    id: "ch1-c1",
    dsl: `network Perceptron {
  dense 2 -> 1 activation=sigmoid
}
train dataset=and lr=0.8 epochs=250
`,
  },
  {
    id: "ch1-c2",
    dsl: `network Perceptron {
  dense 2 -> 1 activation=sigmoid
}
train dataset=or lr=0.8 epochs=250
`,
  },
  {
    id: "ch1-c3",
    dsl: defaultStarterDSL("ch1").replace(/dataset=\w+/, "dataset=xor"),
  },
  {
    id: "ch2-c1",
    dsl: `network MLP {
  dense 2 -> 12 activation=relu
  dense 12 -> 8 activation=relu
  dense 8 -> 1 activation=sigmoid
}
train dataset=xor lr=0.35 epochs=500
`,
  },
  {
    id: "ch2-c2",
    dsl: `network MLP {
  dense 2 -> 16 activation=relu
  dense 16 -> 8 activation=relu
  dense 8 -> 1 activation=sigmoid
}
train dataset=moons lr=0.25 epochs=200
`,
  },
  {
    id: "ch2-c3",
    dsl: `network MLP {
  dense 2 -> 16 activation=relu
  dense 16 -> 8 activation=relu
  dense 8 -> 1 activation=sigmoid
}
train dataset=circles lr=0.25 epochs=250
`,
  },
  {
    id: "ch3-c1",
    dsl: defaultStarterDSL("ch3").replace(/dataset=\w+/, "dataset=moons"),
  },
  {
    id: "ch3-c2",
    dsl: `network OverfitDemo {
  dense 2 -> 32 activation=relu
  dense 32 -> 32 activation=relu
  dense 32 -> 1 activation=sigmoid
}
l2=0.01
train dataset=moons lr=0.2 epochs=80
`,
  },
  {
    id: "ch3-c3",
    dsl: `network OverfitDemo {
  dense 2 -> 32 activation=relu
  dense 32 -> 32 activation=relu
  dense 32 -> 1 activation=sigmoid
}
train dataset=moons lr=0.2 epochs=300
`,
  },
  {
    id: "ch4-c1",
    dsl: `network TinyCNN {
  conv2d filters=4 kernel=2 activation=relu channels=1 height=4 width=4
  flatten
  dense 8 activation=relu
  dense 2 activation=sigmoid
}
train dataset=tiny_images lr=0.12 epochs=100
`,
  },
  {
    id: "ch4-c3",
    dsl: defaultStarterDSL("ch4").replace(
      /dataset=\w+/,
      "dataset=tiny_images"
    ),
  },
  {
    id: "ch5-c1",
    dsl: `network TinyTransformer {
  transformer d_model=8 heads=2
  dense 8 -> 1 activation=sigmoid
}
train dataset=tiny_text lr=0.1 epochs=80
`,
  },
  {
    id: "ch5-c2",
    dsl: `network TinyTransformer {
  transformer d_model=8 heads=2
  dense 8 -> 1 activation=sigmoid
}
train dataset=tiny_text lr=0.1 epochs=40
`,
  },
  {
    id: "ch5-c3",
    dsl: `network TextDense {
  dense 8 -> 16 activation=relu
  dense 16 -> 1 activation=sigmoid
}
train dataset=tiny_text lr=0.2 epochs=120
`,
  },
];

function expectParseError(
  source: string,
  opts: { line?: number; message?: RegExp | string }
): DSLParseError {
  let err: unknown;
  try {
    parseDSL(source);
  } catch (e) {
    err = e;
  }
  expect(err).toBeInstanceOf(DSLParseError);
  const pe = err as DSLParseError;
  if (opts.line != null) expect(pe.line).toBe(opts.line);
  if (opts.message != null) {
    if (typeof opts.message === "string") {
      expect(pe.message).toContain(opts.message);
    } else {
      expect(pe.message).toMatch(opts.message);
    }
  }
  // Message must be self-sufficient for the editor strip (include line + detail)
  expect(pe.message).toMatch(/line\s+\d+/i);
  return pe;
}

describe("parseDSL — valid programs", () => {
  it("parses a minimal dense network + train", () => {
    const parsed = parseDSL(`network Perceptron {
  dense 2 -> 1 activation=sigmoid
}
train dataset=xor lr=0.5 epochs=100
`);
    expect(parsed.network.name).toBe("Perceptron");
    expect(parsed.network.layers).toEqual([
      { type: "dense", units: 1, inputDim: 2, activation: "sigmoid" },
    ]);
    expect(parsed.train).toMatchObject({
      dataset: "xor",
      learningRate: 0.5,
      epochs: 100,
    });
  });

  it("keeps optional params optional (no activation, no l2/dropout/batch)", () => {
    const parsed = parseDSL(`network N {
  dense 2 -> 4
  dense 1
  conv2d
  pool
  flatten
  attention
  transformer
}
train dataset=and
`);
    expect(parsed.network.layers).toHaveLength(7);
    expect(parsed.network.layers[0]).toMatchObject({
      type: "dense",
      units: 4,
      activation: "sigmoid",
    });
    expect(parsed.network.layers[2]).toMatchObject({
      type: "conv2d",
      filters: 4,
      kernelSize: 2,
      activation: "relu",
    });
    expect(parsed.network.layers[3]).toMatchObject({
      type: "pool",
      mode: "max",
      size: 2,
      stride: 2,
    });
    expect(parsed.train.dataset).toBe("and");
    expect(parsed.train.learningRate).toBe(0.5);
    expect(parsed.train.epochs).toBe(100);
  });

  it("accepts comments, blank lines, l2, dropout, batch, shuffle", () => {
    const parsed = parseDSL(`
# header
network Net {
  # hidden
  dense 2 -> 3 activation=relu
}

l2=0.01
dropout=0.2
train dataset=moons lr=0.1 epochs=50 batch=4 shuffle=false
`);
    expect(parsed.network.l2).toBe(0.01);
    expect(parsed.network.dropout).toBe(0.2);
    expect(parsed.train.batchSize).toBe(4);
    expect(parsed.train.shuffle).toBe(false);
  });

  it("round-trips toDSL(parseDSL(x)) with deep-equal network config", () => {
    for (const id of ["ch1", "ch2", "ch3", "ch4", "ch5"] as const) {
      const source = defaultStarterDSL(id);
      const parsed = parseDSL(source);
      const again = parseDSL(toDSL(parsed.network, parsed.train));
      // Full network config must survive serialization (incl. conv2d spatial shape).
      expect(again.network).toEqual(parsed.network);
      expect(again.train).toEqual(parsed.train);
    }
  });

  it("accepts a single-line network block (body on the header line)", () => {
    const parsed = parseDSL(
      `network P { dense 2 -> 1 activation=sigmoid }
train dataset=xor lr=0.8 epochs=200
`
    );
    expect(parsed.network.name).toBe("P");
    expect(parsed.network.layers).toEqual([
      { type: "dense", units: 1, inputDim: 2, activation: "sigmoid" },
    ]);
    expect(parsed.train).toMatchObject({
      dataset: "xor",
      learningRate: 0.8,
      epochs: 200,
    });
  });

  it("accepts a closing brace on the last layer line", () => {
    const parsed = parseDSL(`network P {
  dense 2 -> 1 }
train dataset=xor
`);
    expect(parsed.network.name).toBe("P");
    expect(parsed.network.layers).toEqual([
      { type: "dense", units: 1, inputDim: 2, activation: "sigmoid" },
    ]);
  });
});

describe("parseDSL — curriculum regression (must not throw)", () => {
  for (const chapter of CHAPTERS) {
    it(`${chapter.id} starterDSL parses`, () => {
      expect(() => parseDSL(chapter.starterDSL)).not.toThrow();
      const p = parseDSL(chapter.starterDSL);
      expect(p.network.layers.length).toBeGreaterThan(0);
    });
  }

  for (const id of ["ch1", "ch2", "ch3", "ch4", "ch5"] as const) {
    it(`defaultStarterDSL(${id}) parses`, () => {
      expect(() => parseDSL(defaultStarterDSL(id))).not.toThrow();
    });
  }

  for (const sol of INTENDED_SOLUTION_DSLS) {
    it(`intended solution ${sol.id} parses`, () => {
      expect(() => parseDSL(sol.dsl)).not.toThrow();
      const p = parseDSL(sol.dsl);
      expect(p.network.layers.length).toBeGreaterThan(0);
    });
  }
});

describe("parseDSL — rejections", () => {
  it("rejects empty / whitespace-only program", () => {
    expectParseError("", { line: 1, message: /empty/i });
    expectParseError("   \n  \n", { line: 1, message: /empty/i });
    expectParseError("# only a comment\n\n", { line: 1, message: /empty/i });
  });

  it("rejects unknown top-level keyword / stray prose", () => {
    expectParseError("this is not valid neuralbasic at all", {
      line: 1,
      message: /unknown top-level|this/i,
    });
    expectParseError("netwrk Perceptron { dense 2 -> 1 }", {
      line: 1,
      message: /netwrk/i,
    });
  });

  it("rejects a program with no network block", () => {
    expectParseError("train dataset=xor lr=0.5 epochs=10", {
      message: /no `network` block/i,
    });
  });

  it("rejects unknown layer type inside a network", () => {
    expectParseError(
      `network N {
  banana 2 -> 1
}
train dataset=xor
`,
      { line: 2, message: /banana|unknown layer/i }
    );
  });

  it("rejects unknown activation and names valid ones", () => {
    const pe = expectParseError(
      `network P {
  dense 2 -> 1 activation=banana
}
train dataset=xor
`,
      { line: 2, message: /banana/i }
    );
    expect(pe.message).toMatch(/sigmoid/i);
    expect(pe.message).toMatch(/relu/i);
  });

  it("rejects unknown dataset and names valid ones", () => {
    const pe = expectParseError(
      `network N {
  dense 2 -> 1
}
train dataset=not_a_dataset lr=0.5 epochs=10
`,
      { line: 4, message: /not_a_dataset/i }
    );
    expect(pe.message).toMatch(/xor/i);
    expect(pe.message).toMatch(/moons/i);
  });

  it("rejects unknown parameter keys on layer and train", () => {
    expectParseError(
      `network N {
  dense 2 -> 1 activation=sigmoid foo=bar
}
train dataset=xor
`,
      { line: 2, message: /foo/i }
    );
    expectParseError(
      `network N {
  dense 2 -> 1
}
train dataset=xor lr=0.5 epochs=10 momentum=0.9
`,
      { line: 4, message: /momentum/i }
    );
    expectParseError(
      `network N {
  conv2d filters=4 stride=2
}
train dataset=xor
`,
      { line: 2, message: /stride/i }
    );
  });

  it("rejects non-numeric dense widths and non-positive sizes", () => {
    expectParseError(
      `network P {
  dense two -> one activation=sigmoid
}
train dataset=xor
`,
      { line: 2, message: /two|positive integer|number/i }
    );
    expectParseError(
      `network P {
  dense 0 -> 1 activation=sigmoid
}
train dataset=xor
`,
      { line: 2, message: /positive/i }
    );
    expectParseError(
      `network P {
  dense 2 -> 0 activation=sigmoid
}
train dataset=xor
`,
      { line: 2, message: /positive/i }
    );
  });

  it("rejects malformed / unclosed network block", () => {
    expectParseError(
      `network P {
  dense 2 -> 1 activation=sigmoid
train dataset=xor
`,
      { message: /unclosed|train.*inside|network/i }
    );
    // train inside open network is reported at the train line
    expectParseError(
      `network P {
  dense 2 -> 1
train dataset=xor lr=0.5 epochs=10
`,
      { line: 3, message: /train|network/i }
    );
    expectParseError(
      `network P {
  dense 2 -> 1 activation=sigmoid
`,
      { message: /unclosed/i }
    );
    expectParseError(
      `}
network P {
  dense 2 -> 1
}
train dataset=xor
`,
      { line: 1, message: /unexpected/i }
    );
  });

  it("rejects empty network body", () => {
    expectParseError(
      `network Empty {
}
train dataset=xor
`,
      { message: /no layers/i }
    );
  });

  it("rejects layer outside network", () => {
    expectParseError(
      `dense 2 -> 1 activation=sigmoid
train dataset=xor
`,
      { line: 1, message: /inside a `network`/i }
    );
  });

  it("rejects bad train numbers", () => {
    expectParseError(
      `network N {
  dense 2 -> 1
}
train dataset=xor lr=fast epochs=10
`,
      { line: 4, message: /lr|number|fast/i }
    );
    expectParseError(
      `network N {
  dense 2 -> 1
}
train dataset=xor lr=0 epochs=10
`,
      { line: 4, message: /lr|positive/i }
    );
  });
});
