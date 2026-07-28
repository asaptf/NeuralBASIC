/**
 * `negation` + attention-vs-dense (position hold-out).
 *
 * Aggregates only — per-run assertions on stochastic training have flaked
 * repeatedly in this repo.
 *
 * ## Why this dataset exists
 * `tiny_text` is an 8-dim bag of word-presence flags. With d_model=8 the
 * engine builds a single token and attention is a 1×1 map of [[1]]. Nothing
 * requires looking at one position in light of another; a linear neuron
 * solves it.
 *
 * `negation` is a 4-token sequence (d_model=4, flat length 16) where NOT
 * flips the polarity of GOOD/BAD. The same multiset interaction is an XOR
 * that a linear bag-of-words classifier cannot express.
 *
 * ## Why position hold-out
 * Flattened dense nets can memorize every train placement of the one-hots.
 * Holding out all sequences that use slot 3 is the honest generalisation
 * test: content-based attention transfers; position-tied dense does not.
 *
 * ## Measured (position hold-out on slot 3, 10 runs, 200 epochs)
 * | architecture                         | params | mean train | mean test |
 * |--------------------------------------|-------:|-----------:|----------:|
 * | attention d=4 heads=2 → dense 1      |     69 |     ~0.88  |    ~0.92  |
 * | dense 16-8-1 (flat 16)               |    417 |     ~0.98  |    ~0.33  |
 * | dense 32-16-1                        |   1089 |     ~1.00  |    ~0.30  |
 * | linear bag (sum one-hots → 1)        |      5 |     ~0.60  |    ~0.69  |
 * | mlp bag 16-1                         |     97 |     ~1.00  |    ~1.00* |
 *
 * *MLP-on-bag succeeds because bags of train and test coincide; linear bag
 * still plateaus (XOR). The chapter contrast is attention vs position-tied
 * dense of ≥ params, plus linear bag ceiling.
 *
 * Linear bag hard ceiling: ≤ 28/32 = 0.875 (best halfspace drops one size-4
 * bag-type of the four XOR classes).
 */
import { describe, expect, it } from "vitest";
import {
  getDataset,
  DATASET_NAMES,
  NEGATION_D_MODEL,
  NEGATION_SEQ_LEN,
  NEGATION_VOCAB,
} from "./datasets";
import { parseDSL, toDSL } from "./dsl";
import { createModel, predict, forward, type Model } from "./model";
import { trainStep, prepareNetworkConfig } from "./train";
import type { NetworkConfig, Sample, TrainConfig } from "./types";

const V = NEGATION_D_MODEL;
const SEQ = NEGATION_SEQ_LEN;
const FLAT = SEQ * V;

function countParams(
  config: NetworkConfig,
  dataset: TrainConfig["dataset"] = "negation"
): number {
  const { config: cfg, inputDim } = prepareNetworkConfig(config, dataset);
  const model = createModel(cfg, inputDim);
  let n = 0;
  for (const layer of model.layers) {
    if (layer.type === "dense") {
      for (const row of layer.weights) n += row.length;
      n += layer.biases.length;
    } else if (layer.type === "attention") {
      for (const m of [layer.Wq, layer.Wk, layer.Wv, layer.Wo])
        for (const r of m) n += r.length;
    } else if (layer.type === "transformer_block") {
      for (const m of [
        layer.attention.Wq,
        layer.attention.Wk,
        layer.attention.Wv,
        layer.attention.Wo,
        layer.W1,
        layer.W2,
      ])
        for (const r of m) n += r.length;
      n += layer.b1.length + layer.b2.length;
    }
  }
  return n;
}

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function std(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  const v = xs.reduce((s, x) => s + (x - m) * (x - m), 0) / (xs.length - 1);
  return Math.sqrt(v);
}

function accuracyOn(model: Model, samples: Sample[]): number {
  if (!samples.length) return 0;
  let correct = 0;
  for (const s of samples) {
    const p = predict(model, s.x);
    const pred = p[0]! >= 0.5 ? 1 : 0;
    const tgt = s.y[0]! >= 0.5 ? 1 : 0;
    if (pred === tgt) correct++;
  }
  return correct / samples.length;
}

/** Sum of per-position one-hots → vocab count vector (true bag-of-words). */
function bagOfTokens(x: number[]): number[] {
  const b = Array.from({ length: V }, () => 0);
  for (let i = 0; i < SEQ; i++) {
    for (let j = 0; j < V; j++) b[j]! += x[i * V + j]!;
  }
  return b;
}

/**
 * Hold out every sequence that places a non-PAD token in slot `holdSlot`
 * (default 3). Train only sees content in the earlier slots.
 */
function positionHoldout(holdSlot = 3): { train: Sample[]; test: Sample[] } {
  const ds = getDataset("negation");
  const train: Sample[] = [];
  const test: Sample[] = [];
  for (const s of ds.samples) {
    // Token at holdSlot is non-PAD iff its one-hot is not e_PAD
    const base = holdSlot * V;
    const isPad =
      s.x[base]! === 1 &&
      s.x[base + 1]! === 0 &&
      s.x[base + 2]! === 0 &&
      s.x[base + 3]! === 0;
    (isPad ? train : test).push(s);
  }
  return { train, test };
}

function trainOnSamples(
  config: NetworkConfig,
  samples: Sample[],
  lr: number,
  epochs: number,
  inputDim: number
): Model {
  const model = createModel(config, inputDim);
  for (let e = 0; e < epochs; e++) {
    const order = samples.slice();
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [order[i], order[j]] = [order[j]!, order[i]!];
    }
    for (const s of order) trainStep(model, s.x, s.y, lr, 0);
  }
  return model;
}

/** Decode flat one-hot sequence back to token ids (for debugging / maps). */
function decodeIds(x: number[]): number[] {
  const ids: number[] = [];
  for (let i = 0; i < SEQ; i++) {
    let best = 0;
    for (let j = 1; j < V; j++) {
      if (x[i * V + j]! > x[i * V + best]!) best = j;
    }
    ids.push(best);
  }
  return ids;
}

describe("negation dataset", () => {
  it("is registered and listed", () => {
    expect(DATASET_NAMES).toContain("negation");
    const ds = getDataset("negation");
    expect(ds.name).toBe("negation");
    expect(ds.inputShape).toEqual([FLAT]);
    expect(ds.outputDim).toBe(1);
    expect(ds.samples.length).toBe(32);
    const pos = ds.samples.filter((s) => s.y[0]! >= 0.5).length;
    expect(pos).toBe(16);
    expect(NEGATION_VOCAB).toEqual(["PAD", "NOT", "GOOD", "BAD"]);
  });

  it("is deterministic (same tokens + labels every call)", () => {
    const a = getDataset("negation");
    const b = getDataset("negation");
    expect(a.samples.map((s) => [...s.x, ...s.y])).toEqual(
      b.samples.map((s) => [...s.x, ...s.y])
    );
    const c = getDataset("negation", true);
    expect(c.samples.map((s) => [...s.x, ...s.y])).toEqual(
      a.samples.map((s) => [...s.x, ...s.y])
    );
  });

  it("is multi-token under d_model=4 (seq×d_model reshape)", () => {
    const ds = getDataset("negation");
    const sample = ds.samples[0]!;
    expect(sample.x.length).toBe(FLAT);

    const config: NetworkConfig = {
      layers: [
        { type: "attention", dModel: NEGATION_D_MODEL, nHeads: 2 },
        {
          type: "dense",
          units: 1,
          activation: "sigmoid",
          inputDim: NEGATION_D_MODEL,
        },
      ],
    };
    const model = createModel(config, FLAT);
    forward(model, sample.x);
    const attn = model.layers[0]!;
    expect(attn.type).toBe("attention");
    if (attn.type === "attention") {
      expect(attn.lastAttn).toBeDefined();
      expect(attn.lastAttn!.length).toBe(2); // heads
      expect(attn.lastAttn![0]!.length).toBe(SEQ);
      expect(attn.lastAttn![0]![0]!.length).toBe(SEQ);
      // Not a 1×1 decoration
      expect(SEQ).toBeGreaterThan(1);
    }
  });

  it("labels follow compositional negation (NOT flips GOOD/BAD)", () => {
    const ds = getDataset("negation");
    for (const s of ds.samples) {
      const ids = decodeIds(s.x);
      const hasNot = ids.includes(1);
      const hasGood = ids.includes(2);
      const hasBad = ids.includes(3);
      expect(Number(hasGood) + Number(hasBad)).toBe(1);
      const base = hasGood ? 1 : 0;
      const expected = hasNot ? 1 - base : base;
      expect(s.y[0]).toBe(expected);
    }
  });

  it("parses attention DSL and round-trips via toDSL", () => {
    const src = `network NegationAttn {
  attention d_model=4 heads=2
  dense 4 -> 1 activation=sigmoid
}
train dataset=negation lr=0.1 epochs=150
`;
    const parsed = parseDSL(src);
    expect(parsed.train.dataset).toBe("negation");
    expect(parsed.network.layers[0]).toMatchObject({
      type: "attention",
      dModel: 4,
      nHeads: 2,
    });
    const again = parseDSL(toDSL(parsed.network, parsed.train));
    expect(again.train.dataset).toBe("negation");
    expect(again.network).toEqual(parsed.network);
  });

  it(
    "is learnable by attention (aggregate full-train accuracy)",
    () => {
      const RUNS = 8;
      const config: NetworkConfig = {
        layers: [
          { type: "attention", dModel: 4, nHeads: 2 },
          { type: "dense", units: 1, activation: "sigmoid", inputDim: 4 },
        ],
      };
      const samples = getDataset("negation").samples;
      const accs: number[] = [];
      for (let r = 0; r < RUNS; r++) {
        const model = trainOnSamples(config, samples, 0.1, 150, FLAT);
        accs.push(accuracyOn(model, samples));
      }
      // FD training is noisy; mean should clearly beat chance
      expect(mean(accs)).toBeGreaterThanOrEqual(0.75);
      expect(Math.max(...accs)).toBeGreaterThanOrEqual(0.9);
    }
  );
});

/**
 * Prove-it suite: attention must beat ≥-param position-tied dense baselines
 * on held-out slots, and linear bag-of-words must plateau at the XOR ceiling.
 */
describe("attention vs dense on negation", () => {
  it(
    "attention beats ≥-param flat dense on held-out positions; linear bag plateaus",
    () => {
      const RUNS = 10;
      const EPOCHS = 200;

      const attn: NetworkConfig = {
        layers: [
          { type: "attention", dModel: 4, nHeads: 2 },
          { type: "dense", units: 1, activation: "sigmoid", inputDim: 4 },
        ],
      };
      // 417 params — well above attention's 69
      const denseMatched: NetworkConfig = {
        layers: [
          { type: "dense", units: 16, activation: "relu", inputDim: FLAT },
          { type: "dense", units: 8, activation: "relu" },
          { type: "dense", units: 1, activation: "sigmoid" },
        ],
      };
      // 1089 params
      const denseLarger: NetworkConfig = {
        layers: [
          { type: "dense", units: 32, activation: "relu", inputDim: FLAT },
          { type: "dense", units: 16, activation: "relu" },
          { type: "dense", units: 1, activation: "sigmoid" },
        ],
      };
      // Linear bag-of-words on count vector (true BoW baseline)
      const linearBag: NetworkConfig = {
        layers: [
          { type: "dense", units: 1, activation: "sigmoid", inputDim: V },
        ],
      };

      const attnParams = countParams(attn);
      const d1Params = countParams(denseMatched);
      const d2Params = countParams(denseLarger);
      const bagParams = countParams(linearBag);

      expect(d1Params).toBeGreaterThanOrEqual(attnParams);
      expect(d2Params).toBeGreaterThan(attnParams);
      expect(attnParams).toBeLessThan(100);
      expect(bagParams).toBeLessThan(attnParams);

      const { train, test } = positionHoldout(3);
      expect(train.length).toBe(18);
      expect(test.length).toBe(14);

      const attnTest: number[] = [];
      const attnTrain: number[] = [];
      const d1Test: number[] = [];
      const d1Train: number[] = [];
      const d2Test: number[] = [];
      const d2Train: number[] = [];
      const bagAcc: number[] = [];

      const bagAll = getDataset("negation").samples.map((s) => ({
        x: bagOfTokens(s.x),
        y: s.y,
      }));

      for (let r = 0; r < RUNS; r++) {
        const am = trainOnSamples(attn, train, 0.1, EPOCHS, FLAT);
        attnTest.push(accuracyOn(am, test));
        attnTrain.push(accuracyOn(am, train));

        const d1 = trainOnSamples(denseMatched, train, 0.12, EPOCHS, FLAT);
        d1Test.push(accuracyOn(d1, test));
        d1Train.push(accuracyOn(d1, train));

        const d2 = trainOnSamples(denseLarger, train, 0.1, EPOCHS, FLAT);
        d2Test.push(accuracyOn(d2, test));
        d2Train.push(accuracyOn(d2, train));

        // Linear bag on full data (ceiling is structural, not a hold-out story)
        const bm = trainOnSamples(linearBag, bagAll, 0.5, EPOCHS, V);
        bagAcc.push(accuracyOn(bm, bagAll));
      }

      const attnMean = mean(attnTest);
      const d1Mean = mean(d1Test);
      const d2Mean = mean(d2Test);
      const bagMean = mean(bagAcc);

      // Hard linear-bag ceiling: ≤ 28/32 = 0.875
      const LINEAR_BAG_CEILING = 28 / 32;

      const report = {
        RUNS,
        EPOCHS,
        params: {
          attn: attnParams,
          dense16: d1Params,
          dense32: d2Params,
          linearBag: bagParams,
        },
        holdoutTestMean: {
          attn: attnMean,
          dense16: d1Mean,
          dense32: d2Mean,
        },
        holdoutTestStd: {
          attn: std(attnTest),
          dense16: std(d1Test),
          dense32: std(d2Test),
        },
        holdoutTrainMean: {
          attn: mean(attnTrain),
          dense16: mean(d1Train),
          dense32: mean(d2Train),
        },
        linearBagFullMean: bagMean,
        linearBagCeiling: LINEAR_BAG_CEILING,
      };
      console.log(
        "[negation position hold-out]",
        JSON.stringify(report, null, 2)
      );

      // Dense fits the train lattice (not broken / handicapped).
      expect(mean(d1Train)).toBeGreaterThanOrEqual(0.9);
      expect(mean(d2Train)).toBeGreaterThanOrEqual(0.95);

      // Attention clearly beats both denser flat baselines on held-out slots.
      expect(attnMean).toBeGreaterThanOrEqual(0.75);
      expect(attnMean).toBeGreaterThan(d1Mean + 0.2);
      expect(attnMean).toBeGreaterThan(d2Mean + 0.2);

      // Dense collapses toward chance on held-out positions.
      expect(d1Mean).toBeLessThan(0.55);
      expect(d2Mean).toBeLessThan(0.55);

      // Linear bag plateaus at/below the XOR ceiling (never solves the task).
      expect(bagMean).toBeLessThanOrEqual(LINEAR_BAG_CEILING + 1e-9);
      expect(bagMean).toBeLessThan(0.9);
      expect(Math.max(...bagAcc)).toBeLessThanOrEqual(LINEAR_BAG_CEILING + 1e-9);
    }
  );

  it(
    "trained attention map is seq×seq and measurably non-uniform",
    () => {
      // Aggregate across independent trainings — do NOT simplify back to a
      // single-run threshold on globalMax. That peak is a stochastic quantity
      // over a soft attention matrix: a CI flake saw 0.295 against
      // uniform+0.05=0.30. A 30-run single-train probe of models with
      // full-train acc ≥ 0.85 gave globalMax mean≈0.81, min≈0.26, max≈0.996
      // (26/27 cleared 0.30; one solved the task with a nearly flat head-0
      // map). Assert the mean of that distribution, not one draw
      // (CONTRIBUTING.md: aggregates across runs).
      const ATTEMPTS = 12;
      const MIN_GOOD = 6;
      const EPOCHS = 200;
      const samples = getDataset("negation").samples;
      const config: NetworkConfig = {
        layers: [
          { type: "attention", dModel: 4, nHeads: 2 },
          { type: "dense", units: 1, activation: "sigmoid", inputDim: 4 },
        ],
      };

      // Probe sequences that readers will see in the heatmap
      const probes = samples.filter((s) => {
        const ids = decodeIds(s.x);
        // Prefer content in early slots for a clean map dump
        return ids[0] !== 0 || ids[1] !== 0;
      });
      expect(probes.length).toBeGreaterThan(0);
      const probeSlice = probes.slice(0, 8);

      const uniform = 1 / SEQ;
      const runMaxes: number[] = [];
      const runRanges: number[] = [];
      const runL1s: number[] = [];
      const allAccs: number[] = [];
      let best: Model | null = null;
      let bestAcc = 0;
      // One dump of maps from the best run (for humans reading the suite log)
      let dumpMaps: { ids: string[]; pred: number; head0: number[][] }[] = [];

      for (let r = 0; r < ATTEMPTS; r++) {
        const model = trainOnSamples(config, samples, 0.1, EPOCHS, FLAT);
        const acc = accuracyOn(model, samples);
        allAccs.push(acc);

        let globalMax = 0;
        let globalMin = 1;
        let totalUniformL1 = 0;
        let nMaps = 0;
        const mapsThisRun: typeof dumpMaps = [];

        for (const s of probeSlice) {
          forward(model, s.x);
          const layer = model.layers[0]!;
          expect(layer.type).toBe("attention");
          if (layer.type !== "attention" || !layer.lastAttn) continue;
          const head0 = layer.lastAttn[0]!;
          // Genuine seq×seq (not 1×1 decoration) — deterministic shape check
          expect(head0.length).toBe(SEQ);
          expect(head0[0]!.length).toBe(SEQ);

          // Rows are distributions
          for (const row of head0) {
            const sum = row.reduce((a, b) => a + b, 0);
            expect(sum).toBeCloseTo(1, 5);
          }

          let mapL1 = 0;
          for (const row of head0) {
            for (const p of row) {
              globalMax = Math.max(globalMax, p);
              globalMin = Math.min(globalMin, p);
              mapL1 += Math.abs(p - uniform);
            }
          }
          totalUniformL1 += mapL1;
          nMaps++;

          mapsThisRun.push({
            ids: decodeIds(s.x).map((i) => NEGATION_VOCAB[i]!),
            pred: predict(model, s.x)[0]!,
            head0: head0.map((row) => row.map((v) => +v.toFixed(3))),
          });
        }

        // Only well-trained runs enter the non-uniformity aggregate — a failed
        // init is not a counterexample to "trained maps are readable".
        if (acc >= 0.85) {
          runMaxes.push(globalMax);
          runRanges.push(globalMax - globalMin);
          runL1s.push(totalUniformL1 / Math.max(1, nMaps));
        }

        if (acc > bestAcc) {
          bestAcc = acc;
          best = model;
          dumpMaps = mapsThisRun;
        }
      }

      expect(best).not.toBeNull();
      expect(bestAcc).toBeGreaterThanOrEqual(0.85);
      // Enough successful trainings to estimate the map distribution
      expect(runMaxes.length).toBeGreaterThanOrEqual(MIN_GOOD);

      const meanMax = mean(runMaxes);
      const meanRange = mean(runRanges);
      const meanL1 = mean(runL1s);
      const clearMargin = runMaxes.filter((m) => m > uniform + 0.05).length;
      console.log(
        "[negation trained attention maps]",
        JSON.stringify(
          {
            ATTEMPTS,
            goodRuns: runMaxes.length,
            fullTrainAccMean: +mean(allAccs).toFixed(3),
            bestAcc: +bestAcc.toFixed(3),
            globalMax: {
              mean: +meanMax.toFixed(3),
              min: +Math.min(...runMaxes).toFixed(3),
              max: +Math.max(...runMaxes).toFixed(3),
              clearUniformPlus05: clearMargin,
              perRun: runMaxes.map((x) => +x.toFixed(3)),
            },
            rangeMean: +meanRange.toFixed(3),
            meanL1FromUniform: +meanL1.toFixed(3),
            maps: dumpMaps,
          },
          null,
          2
        )
      );

      // Non-uniformity of the *distribution* of trained maps.
      // Uniform baseline: max = min = 1/seq = 0.25, L1 = 0, range = 0.
      // Thresholds sit well below measured means (max≈0.81, range≈0.7+, L1≈2+)
      // but far above "decoration" — near-flat maps every run still fail.
      // Do not replace these with a single-run `expect(globalMax) > …`.
      expect(meanMax).toBeGreaterThan(uniform + 0.15); // mean max ≳ 0.40
      expect(meanRange).toBeGreaterThan(0.15);
      expect(meanL1).toBeGreaterThan(0.5);
      // Majority of successful trainings clear the old single-run margin, so a
      // rare flat head-0 solution cannot carry the mean alone.
      expect(clearMargin).toBeGreaterThanOrEqual(
        Math.ceil(runMaxes.length / 2)
      );

      // Behavioral check on the best-trained model: GOOD alone vs NOT+GOOD
      const goodAlone = samples.find((s) => {
        const ids = decodeIds(s.x);
        return (
          ids.filter((t) => t === 2).length === 1 &&
          !ids.includes(1) &&
          !ids.includes(3)
        );
      })!;
      const notGood = samples.find((s) => {
        const ids = decodeIds(s.x);
        return ids.includes(1) && ids.includes(2) && !ids.includes(3);
      })!;
      expect(goodAlone).toBeDefined();
      expect(notGood).toBeDefined();
      expect(predict(best!, goodAlone.x)[0]!).toBeGreaterThan(0.5);
      expect(predict(best!, notGood.x)[0]!).toBeLessThan(0.5);

      // Maps for those two sequences must differ (attending is input-dependent)
      forward(best!, goodAlone.x);
      const mapGood =
        best!.layers[0]!.type === "attention"
          ? (best!.layers[0] as { lastAttn?: number[][][] }).lastAttn![0]!
          : null;
      forward(best!, notGood.x);
      const mapNotGood =
        best!.layers[0]!.type === "attention"
          ? (best!.layers[0] as { lastAttn?: number[][][] }).lastAttn![0]!
          : null;
      expect(mapGood).not.toBeNull();
      expect(mapNotGood).not.toBeNull();
      let diff = 0;
      for (let i = 0; i < SEQ; i++)
        for (let j = 0; j < SEQ; j++)
          diff += Math.abs(mapGood![i]![j]! - mapNotGood![i]![j]!);
      expect(diff).toBeGreaterThan(0.15);
    }
  );
});
