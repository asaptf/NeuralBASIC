import type { Sample } from "./types";

/**
 * Default held-out fraction when `TrainConfig.valRatio` is omitted.
 * Classic 3:1 train/val; large enough for stable metrics on moons (80 pts)
 * without starving training.
 */
export const DEFAULT_VAL_RATIO = 0.25;

/**
 * Datasets smaller than this never receive a validation split.
 *
 * Rationale: xor / and / or have only 4 points. Any non-empty hold-out leaves
 * a 1-sample "validation set" whose accuracy is pure noise (0 or 1) and would
 * mislead the overfitting chapter. Require enough rows that stratified hold-out
 * can keep ≥1 train and ≥1 val sample for every class that has ≥2 members.
 */
export const MIN_SAMPLES_FOR_VAL_SPLIT = 10;

export interface DataSplit {
  /** Samples the optimizer is allowed to see. */
  train: Sample[];
  /**
   * Held-out samples for metrics only. `null` when no split was applied
   * (tiny dataset, valRatio ≤ 0, or stratification could not reserve both sides).
   */
  val: Sample[] | null;
  /** Effective ratio actually used (0 when no split). */
  valRatio: number;
  /** True iff `val` is non-null and non-empty. */
  splitApplied: boolean;
}

/** Stable class key for stratification (binary threshold or multi-class argmax). */
export function classKey(y: number[]): string {
  if (y.length <= 1) {
    return String(y[0]! >= 0.5 ? 1 : 0);
  }
  let best = 0;
  for (let i = 1; i < y.length; i++) {
    if (y[i]! > y[best]!) best = i;
  }
  return String(best);
}

/**
 * Deterministic content hash — used only to order samples inside a class so the
 * hold-out is not "first N in generation order". No Math.random().
 */
function stableSampleOrderKey(s: Sample, originalIndex: number): number {
  let h = (originalIndex + 1) * 2654435761;
  for (const v of s.x) {
    h = Math.imul(h ^ (Math.floor(v * 1e6) | 0), 1597334677);
  }
  for (const v of s.y) {
    h = Math.imul(h ^ (Math.floor(v * 1e6) | 0), 2246822519);
  }
  return h >>> 0;
}

/**
 * Stratified train/validation split.
 *
 * - Deterministic for a fixed sample array (dataset cache → same split every run).
 * - Class-balanced: each class contributes ≈ `valRatio` of its members to val.
 * - Never uses Math.random().
 * - Returns `splitApplied: false` (all samples in train, val=null) when a split
 *   would be meaningless or impossible.
 */
export function splitTrainVal(
  samples: Sample[],
  valRatio: number = DEFAULT_VAL_RATIO
): DataSplit {
  const n = samples.length;
  if (
    !(valRatio > 0) ||
    valRatio >= 1 ||
    n < MIN_SAMPLES_FOR_VAL_SPLIT
  ) {
    return {
      train: samples.slice(),
      val: null,
      valRatio: 0,
      splitApplied: false,
    };
  }

  // Group original indices by class.
  const byClass = new Map<string, number[]>();
  for (let i = 0; i < n; i++) {
    const key = classKey(samples[i]!.y);
    let list = byClass.get(key);
    if (!list) {
      list = [];
      byClass.set(key, list);
    }
    list.push(i);
  }

  const valIndexSet = new Set<number>();

  for (const indices of byClass.values()) {
    const c = indices.length;
    // Need ≥2 members to put at least one on each side.
    if (c < 2) continue;

    let nVal = Math.floor(c * valRatio);
    if (nVal < 1) nVal = 1;
    if (nVal >= c) nVal = c - 1;

    // Deterministic order inside the class (not generation order).
    const ordered = indices
      .slice()
      .sort(
        (a, b) =>
          stableSampleOrderKey(samples[a]!, a) -
            stableSampleOrderKey(samples[b]!, b) || a - b
      );

    for (let k = 0; k < nVal; k++) {
      valIndexSet.add(ordered[k]!);
    }
  }

  if (valIndexSet.size === 0 || valIndexSet.size >= n) {
    return {
      train: samples.slice(),
      val: null,
      valRatio: 0,
      splitApplied: false,
    };
  }

  const train: Sample[] = [];
  const val: Sample[] = [];
  for (let i = 0; i < n; i++) {
    if (valIndexSet.has(i)) val.push(samples[i]!);
    else train.push(samples[i]!);
  }

  return {
    train,
    val,
    valRatio,
    splitApplied: true,
  };
}

/**
 * Resolve the effective val ratio from optional TrainConfig field.
 * `undefined` → default; explicit `0` → no split.
 */
export function resolveValRatio(valRatio: number | undefined): number {
  if (valRatio === undefined) return DEFAULT_VAL_RATIO;
  if (!Number.isFinite(valRatio) || valRatio <= 0) return 0;
  if (valRatio >= 1) return 0;
  return valRatio;
}
