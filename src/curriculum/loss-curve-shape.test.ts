/**
 * Locks Chapter 1's load-bearing claims about *loss-curve shape*.
 *
 * Lesson prose (not `expect` strings) asserts:
 *   1. On `or` the curve "slides down smoothly".
 *   2. On `moons` or `xor` it is "visibly jagged".
 *   3. The cause is disagreement between samples, not difficulty / lr.
 *   4. Raising lr on `moons` does not make the curve noticeably more jagged.
 *
 * If a claim fails against measured numbers, fix the prose (or engine) —
 * do not widen thresholds until the lie passes.
 *
 * Metric: direction-change ratio (DCR) of the per-epoch loss series.
 *   For interior points i, count the fraction where sign(L[i]-L[i-1]) flips
 *   relative to sign(L[i+1]-L[i]). Flat segments (both diffs ~0) are skipped.
 *
 *   Smooth monotonic descent → DCR ≈ 0.
 *   A curve that tugs up and down each epoch → DCR well above 0 (random walk
 *   of signed diffs tends toward ~0.5; real jagged SGD curves land ~0.6–0.7).
 *
 * Thresholds (explicit, not eyeballed after a single run):
 *   SMOOTH_MAX  = 0.15  — mean DCR must sit at or below this to count as smooth
 *   JAGGED_MIN  = 0.40  — mean DCR must sit at or above this to count as jagged
 *   LR_JAG_SLACK = 0.12 — high-lr mean DCR may not exceed low-lr mean by more
 *
 * These bands leave ~0.25 of headroom from the observed means (or ≈ 0.00,
 * moons/xor ≈ 0.65–0.70) so random init does not flake, while a real
 * prose/engine drift that collapses the smooth/jagged gap will fail.
 */
import { describe, expect, it } from "vitest";
import { createAndTrain } from "@/engine/train";
import { createModel, predict, exportWeights, loadWeights } from "@/engine/model";
import { getDataset } from "@/engine/datasets";
import type { Model } from "@/engine/model";
import type { DatasetName, Sample } from "@/engine/types";

// ---------------------------------------------------------------------------
// Harness knobs — keep the suite fast (modest epochs, handful of repeats).
// ---------------------------------------------------------------------------

const REPEATS = 6;
const EPOCHS = 50;
/** Learning rate used in the "smooth / jagged by dataset" claims. */
const BASE_LR = 0.8;
/** Large learning rate used in claim 4 (matches the lesson's lr=20 example). */
const HIGH_LR = 20;

/** Mean DCR ≤ this → "slides down smoothly". */
const SMOOTH_MAX = 0.15;
/** Mean DCR ≥ this → "visibly jagged". */
const JAGGED_MIN = 0.40;
/**
 * Claim 4: high-lr mean DCR must not exceed low-lr mean by more than this.
 * Hand / pilot measurements put moons@0.8 ≈ 0.69 and moons@20 ≈ 0.64–0.68,
 * so "not noticeably more jagged" is operationalised as "not higher by >0.12".
 */
const LR_JAG_SLACK = 0.12;

const PERCEPTRON = {
  layers: [{ type: "dense" as const, units: 1, activation: "sigmoid" as const }],
};

// ---------------------------------------------------------------------------
// Shape metric
// ---------------------------------------------------------------------------

/**
 * Direction-change ratio of a loss series.
 * Returns the fraction of interior steps where the first difference flips sign.
 * Range is [0, 1]; empty / too-short series yield 0.
 */
function directionChangeRatio(losses: number[]): number {
  if (losses.length < 3) return 0;
  let flips = 0;
  let total = 0;
  for (let i = 1; i < losses.length - 1; i++) {
    const d1 = losses[i]! - losses[i - 1]!;
    const d2 = losses[i + 1]! - losses[i]!;
    // Skip perfectly flat corners so a plateau does not inflate the denominator.
    if (Math.abs(d1) < 1e-15 && Math.abs(d2) < 1e-15) continue;
    total++;
    if (d1 * d2 < 0) flips++;
  }
  return total === 0 ? 0 : flips / total;
}

interface ShapeDist {
  ratios: number[];
  mean: number;
  min: number;
  max: number;
  std: number;
  /** Final accuracies of the same runs — used for the difficulty check. */
  accuracies: number[];
  meanAcc: number;
}

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
}

function std(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
}

function sampleShape(
  dataset: DatasetName,
  lr: number,
  epochs = EPOCHS,
  n = REPEATS
): ShapeDist {
  const ratios: number[] = [];
  const accuracies: number[] = [];
  for (let i = 0; i < n; i++) {
    const { history } = createAndTrain(
      PERCEPTRON,
      {
        dataset,
        learningRate: lr,
        epochs,
        shuffle: true,
      },
      { includeDecisionBoundary: false }
    );
    ratios.push(directionChangeRatio(history.losses));
    accuracies.push(history.final.accuracy);
  }
  return {
    ratios,
    mean: mean(ratios),
    min: Math.min(...ratios),
    max: Math.max(...ratios),
    std: std(ratios),
    accuracies,
    meanAcc: mean(accuracies),
  };
}

function fmtDist(d: ShapeDist): string {
  return (
    `mean=${d.mean.toFixed(3)} std=${d.std.toFixed(3)} ` +
    `range=[${d.min.toFixed(3)}, ${d.max.toFixed(3)}] ` +
    `n=${d.ratios.length} meanAcc=${d.meanAcc.toFixed(3)}`
  );
}

// ---------------------------------------------------------------------------
// Claim 3 helpers — sample disagreement, independent of the DCR metric.
// ---------------------------------------------------------------------------

/** Flatten dense weights+biases of a single-output perceptron. */
function getParams(model: Model): number[] {
  const w = exportWeights(model);
  const out: number[] = [];
  for (const layer of w) {
    if (layer.weights) for (const row of layer.weights) for (const v of row) out.push(v);
    if (layer.biases) for (const v of layer.biases) out.push(v);
  }
  return out;
}

function setParams(model: Model, flat: number[]): void {
  const w = exportWeights(model);
  let idx = 0;
  for (const layer of w) {
    if (layer.weights) {
      for (let i = 0; i < layer.weights.length; i++) {
        for (let j = 0; j < layer.weights[i]!.length; j++) {
          layer.weights[i]![j] = flat[idx++]!;
        }
      }
    }
    if (layer.biases) {
      for (let i = 0; i < layer.biases.length; i++) layer.biases[i] = flat[idx++]!;
    }
  }
  loadWeights(model, w);
}

function epochLoss(model: Model, samples: Sample[]): number {
  let loss = 0;
  for (const s of samples) {
    const p = Math.max(1e-9, Math.min(1 - 1e-9, predict(model, s.x)[0]!));
    const y = s.y[0]!;
    loss += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
  }
  return loss / samples.length;
}

/**
 * One analytical SGD step for a single dense sigmoid + BCE unit.
 * Mirrors the engine's dense backward for this architecture so claim-3
 * diagnostics stay faithful without depending on private train helpers.
 */
function sgdStep(model: Model, x: number[], y: number[], lr: number): void {
  const p = predict(model, x)[0]!;
  const err = p - y[0]!;
  const params = getParams(model);
  const input = [...x, 1];
  for (let i = 0; i < params.length; i++) params[i]! -= lr * err * input[i]!;
  setParams(model, params);
}

/**
 * Path-length / net-displacement of the weight trajectory over training.
 * When samples agree, steps compound and the ratio stays modest (~3–5).
 * When samples cancel each other, the path is long relative to net progress
 * (moons/circles typically >10). Averaged over REPEATS random inits.
 */
function meanPathDisplacementRatio(
  dataset: DatasetName,
  lr = BASE_LR,
  epochs = EPOCHS,
  n = REPEATS
): number {
  const samples = getDataset(dataset).samples;
  const ratios: number[] = [];
  for (let run = 0; run < n; run++) {
    const model = createModel(
      {
        layers: [
          {
            type: "dense",
            units: 1,
            activation: "sigmoid",
            inputDim: 2,
          },
        ],
      },
      2
    );
    let pathSum = 0;
    let dispSum = 0;
    for (let ep = 0; ep < epochs; ep++) {
      const before = getParams(model);
      const order = samples.slice();
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [order[i], order[j]] = [order[j]!, order[i]!];
      }
      let prev = before.slice();
      for (const s of order) {
        sgdStep(model, s.x, s.y, lr);
        const after = getParams(model);
        let step = 0;
        for (let k = 0; k < after.length; k++) {
          const d = after[k]! - prev[k]!;
          step += d * d;
        }
        pathSum += Math.sqrt(step);
        prev = after;
      }
      const after = getParams(model);
      let disp = 0;
      for (let k = 0; k < after.length; k++) {
        const d = after[k]! - before[k]!;
        disp += d * d;
      }
      dispSum += Math.sqrt(disp);
    }
    ratios.push(dispSum < 1e-12 ? 1e6 : pathSum / dispSum);
  }
  return mean(ratios);
}

/**
 * Train with either per-sample SGD or full-batch GD and return mean DCR.
 * Full-batch applies the *average* gradient once per epoch — samples cannot
 * tug the weights in opposing directions within an epoch. If claim 3 is right,
 * batch GD should erase jaggedness even on moons/xor.
 */
function meanDcrUnderUpdateRule(
  dataset: DatasetName,
  mode: "sgd" | "batch",
  lr = BASE_LR,
  epochs = EPOCHS,
  n = REPEATS
): number {
  const samples = getDataset(dataset).samples;
  const ratios: number[] = [];
  for (let run = 0; run < n; run++) {
    const model = createModel(
      {
        layers: [
          {
            type: "dense",
            units: 1,
            activation: "sigmoid",
            inputDim: 2,
          },
        ],
      },
      2
    );
    const losses: number[] = [];
    for (let ep = 0; ep < epochs; ep++) {
      if (mode === "sgd") {
        const order = samples.slice();
        for (let i = order.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [order[i], order[j]] = [order[j]!, order[i]!];
        }
        for (const s of order) sgdStep(model, s.x, s.y, lr);
      } else {
        const dim = getParams(model).length;
        const g = new Array(dim).fill(0) as number[];
        for (const s of samples) {
          const p = predict(model, s.x)[0]!;
          const err = p - s.y[0]!;
          const input = [...s.x, 1];
          for (let i = 0; i < dim; i++) g[i]! += err * input[i]!;
        }
        for (let i = 0; i < dim; i++) g[i]! /= samples.length;
        const params = getParams(model);
        for (let i = 0; i < params.length; i++) params[i]! -= lr * g[i]!;
        setParams(model, params);
      }
      losses.push(epochLoss(model, samples));
    }
    ratios.push(directionChangeRatio(losses));
  }
  return mean(ratios);
}

// ---------------------------------------------------------------------------
// Unit: metric itself
// ---------------------------------------------------------------------------

describe("directionChangeRatio metric", () => {
  it("is ~0 on a smooth monotonic descent", () => {
    const series = Array.from({ length: 40 }, (_, i) => 1 / (i + 1));
    expect(directionChangeRatio(series)).toBe(0);
  });

  it("is high on a pure up-down zigzag", () => {
    const series = Array.from({ length: 40 }, (_, i) => (i % 2 === 0 ? 1 : 0));
    // Every interior point flips: ratio = 1.
    expect(directionChangeRatio(series)).toBe(1);
  });

  it("ignores fully flat plateaus", () => {
    expect(directionChangeRatio([1, 1, 1, 1, 1])).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Claims 1 & 2 — smooth on or, jagged on moons/xor
// ---------------------------------------------------------------------------

describe("Ch1 loss-curve shape: smooth vs jagged by dataset", () => {
  it(`claim 1: or at lr=${BASE_LR} slides down smoothly (mean DCR ≤ ${SMOOTH_MAX})`, () => {
    const d = sampleShape("or", BASE_LR);
    expect(
      d.mean,
      `or claimed smooth but measured ${fmtDist(d)} (threshold ≤ ${SMOOTH_MAX})`
    ).toBeLessThanOrEqual(SMOOTH_MAX);
  });

  it(`claim 2a: moons at lr=${BASE_LR} is visibly jagged (mean DCR ≥ ${JAGGED_MIN})`, () => {
    const d = sampleShape("moons", BASE_LR);
    expect(
      d.mean,
      `moons claimed jagged but measured ${fmtDist(d)} (threshold ≥ ${JAGGED_MIN})`
    ).toBeGreaterThanOrEqual(JAGGED_MIN);
  });

  it(`claim 2b: xor at lr=${BASE_LR} is visibly jagged (mean DCR ≥ ${JAGGED_MIN})`, () => {
    const d = sampleShape("xor", BASE_LR);
    expect(
      d.mean,
      `xor claimed jagged but measured ${fmtDist(d)} (threshold ≥ ${JAGGED_MIN})`
    ).toBeGreaterThanOrEqual(JAGGED_MIN);
  });

  it("smooth and jagged regimes are cleanly separated (or ≪ moons, xor)", () => {
    const orD = sampleShape("or", BASE_LR);
    const moonsD = sampleShape("moons", BASE_LR);
    const xorD = sampleShape("xor", BASE_LR);
    // Gap must survive the worst-case ends of the observed ranges.
    expect(
      moonsD.mean - orD.mean,
      `moons−or gap too small: or ${fmtDist(orD)} vs moons ${fmtDist(moonsD)}`
    ).toBeGreaterThan(0.25);
    expect(
      xorD.mean - orD.mean,
      `xor−or gap too small: or ${fmtDist(orD)} vs xor ${fmtDist(xorD)}`
    ).toBeGreaterThan(0.25);
  });
});

// ---------------------------------------------------------------------------
// Claim 4 — high lr does not add jaggedness on moons
// ---------------------------------------------------------------------------

describe("Ch1 loss-curve shape: learning rate vs jaggedness", () => {
  it(`claim 4: moons lr=${HIGH_LR} is not noticeably more jagged than lr=${BASE_LR}`, () => {
    const low = sampleShape("moons", BASE_LR);
    const high = sampleShape("moons", HIGH_LR);
    expect(
      high.mean,
      `high-lr moons became more jagged than allowed: ` +
        `lr=${BASE_LR} ${fmtDist(low)} vs lr=${HIGH_LR} ${fmtDist(high)} ` +
        `(slack ${LR_JAG_SLACK})`
    ).toBeLessThanOrEqual(low.mean + LR_JAG_SLACK);

    // Both should still be in the jagged regime — lr must not *smooth* the
    // curve either in a way that would invert the teaching point.
    expect(
      high.mean,
      `high-lr moons lost jaggedness entirely: ${fmtDist(high)}`
    ).toBeGreaterThanOrEqual(JAGGED_MIN);
  });
});

// ---------------------------------------------------------------------------
// Claim 3 — jaggedness tracks sample disagreement, not difficulty
// ---------------------------------------------------------------------------

describe("Ch1 loss-curve shape: disagreement, not difficulty (claim 3)", () => {
  /**
   * Difficulty counterexample: moons reaches teachably-high accuracy under
   * the same single-neuron setup, yet its loss curve is as jagged as xor's.
   * If jaggedness were "the problem is hard", moons should be smooth once
   * accuracy is high. It is not.
   */
  it("difficulty is not the driver: moons is accurate yet jagged; or is both accurate and smooth", () => {
    const orD = sampleShape("or", BASE_LR);
    const moonsD = sampleShape("moons", BASE_LR);

    expect(
      orD.meanAcc,
      `or should be easy (acc≈1); got ${fmtDist(orD)}`
    ).toBeGreaterThanOrEqual(0.99);
    expect(
      orD.mean,
      `or should be smooth; got ${fmtDist(orD)}`
    ).toBeLessThanOrEqual(SMOOTH_MAX);

    // Single neuron on moons typically lands ~0.80–0.90 under these knobs.
    // A floor of 0.70 keeps this from flaking while still counting as
    // "substantially learned", not a coin flip.
    expect(
      moonsD.meanAcc,
      `moons should reach teachable accuracy (not a failed run); got ${fmtDist(moonsD)}`
    ).toBeGreaterThanOrEqual(0.7);
    expect(
      moonsD.mean,
      `moons should stay jagged despite decent accuracy; got ${fmtDist(moonsD)}`
    ).toBeGreaterThanOrEqual(JAGGED_MIN);
  });

  /**
   * Weight-trajectory cancellation: path length / net displacement per epoch
   * is a direct geometric measure of samples tugging in different directions.
   * Agreeing datasets (or, and) stay low; moons / xor / circles run high.
   * This is independent of the DCR shape metric asserted above.
   */
  it("sample-disagreement metric (path/displacement) is low on or and high on moons/xor", () => {
    const orPD = meanPathDisplacementRatio("or");
    const andPD = meanPathDisplacementRatio("and");
    const moonsPD = meanPathDisplacementRatio("moons");
    const xorPD = meanPathDisplacementRatio("xor");

    // Pilot ranges: or≈3, and≈4, xor≈7, moons≈12. Thresholds leave headroom.
    expect(
      orPD,
      `or path/disp should be low (samples agree); measured ${orPD.toFixed(2)}`
    ).toBeLessThan(6);
    expect(
      andPD,
      `and path/disp should be low; measured ${andPD.toFixed(2)}`
    ).toBeLessThan(7);
    expect(
      moonsPD,
      `moons path/disp should be high (samples cancel); measured ${moonsPD.toFixed(2)}`
    ).toBeGreaterThan(7);
    expect(
      xorPD,
      `xor path/disp should be high; measured ${xorPD.toFixed(2)}`
    ).toBeGreaterThan(5);

    expect(
      moonsPD,
      `moons should cancel more than or: moons=${moonsPD.toFixed(2)} or=${orPD.toFixed(2)}`
    ).toBeGreaterThan(orPD * 1.5);
  });

  /**
   * Causal check: if jaggedness is produced by within-epoch sample tug-of-war,
   * replacing per-sample SGD with full-batch GD (one averaged step per epoch)
   * must erase the jaggedness even on moons and xor. Difficulty of the
   * objective is unchanged; only the disagreement channel is removed.
   *
   * Note: this diagnostic uses a minimal analytical trainer equivalent to the
   * engine's dense+sigmoid+BCE path, because the production engine only
   * exposes per-sample SGD. The claim under test is about the *mechanism the
   * lesson describes*, not about a user-facing batch mode.
   */
  it("removing within-epoch disagreement (batch GD) erases jaggedness on moons and xor", () => {
    const moonsSgd = meanDcrUnderUpdateRule("moons", "sgd");
    const moonsBatch = meanDcrUnderUpdateRule("moons", "batch");
    const xorSgd = meanDcrUnderUpdateRule("xor", "sgd");
    const xorBatch = meanDcrUnderUpdateRule("xor", "batch");
    const orBatch = meanDcrUnderUpdateRule("or", "batch");

    expect(
      moonsSgd,
      `diagnostic SGD on moons should still be jagged; got ${moonsSgd.toFixed(3)}`
    ).toBeGreaterThanOrEqual(JAGGED_MIN);
    expect(
      xorSgd,
      `diagnostic SGD on xor should still be jagged; got ${xorSgd.toFixed(3)}`
    ).toBeGreaterThanOrEqual(JAGGED_MIN);

    expect(
      moonsBatch,
      `batch GD on moons should be smooth if jaggedness is sample disagreement; got ${moonsBatch.toFixed(3)}`
    ).toBeLessThanOrEqual(SMOOTH_MAX);
    expect(
      xorBatch,
      `batch GD on xor should be smooth if jaggedness is sample disagreement; got ${xorBatch.toFixed(3)}`
    ).toBeLessThanOrEqual(SMOOTH_MAX);
    expect(
      orBatch,
      `batch GD on or stays smooth; got ${orBatch.toFixed(3)}`
    ).toBeLessThanOrEqual(SMOOTH_MAX);
  });
});
