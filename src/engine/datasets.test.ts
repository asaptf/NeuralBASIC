import { describe, it, expect } from "vitest";
import { getDataset, DATASET_NAMES } from "./datasets";
import type { Dataset, DatasetName } from "./types";

/** Integer checksum of all coords + labels (floor to 1e-6). */
function checksum(ds: Dataset): number {
  let h = 0;
  for (const s of ds.samples) {
    for (const v of s.x) h = Math.imul(h ^ (Math.floor(v * 1e6) | 0), 2654435761);
    for (const v of s.y) h = Math.imul(h ^ (Math.floor(v * 1e6) | 0), 1597334677);
  }
  return h >>> 0;
}

function classCounts(ds: Dataset): { c0: number; c1: number } {
  const c0 = ds.samples.filter((s) => (s.y[0] ?? 0) < 0.5).length;
  return { c0, c1: ds.samples.length - c0 };
}

/** First three samples flattened — pins geometry without float-literal noise. */
function headFingerprint(ds: Dataset, n = 3): number[] {
  return ds.samples.slice(0, n).flatMap((s) => [...s.x, ...s.y]);
}

/**
 * Frozen fingerprints for the four generators that used to call Math.random().
 * A silent geometry change flips checksum and/or head values.
 */
const FINGERPRINTS: Record<
  "linear" | "moons" | "circles" | "spiral",
  { n: number; c0: number; c1: number; checksum: number; head: number[] }
> = {
  linear: {
    n: 40,
    c0: 20,
    c1: 20,
    checksum: 1402053166,
    head: [
      0.21129974816, 0.463505432475, 1, 0.332980767824, -0.309290983714, 1,
      -0.494467716198, -0.283938608598, 0,
    ],
  },
  moons: {
    n: 80,
    c0: 40,
    c1: 40,
    checksum: 1987007559,
    head: [
      1.038260778636, -0.003562228996, 0, 1.022323464473, 0.092997499504, 0,
      0.950416822158, 0.12778253231, 0,
    ],
  },
  circles: {
    n: 80,
    c0: 40,
    c1: 40,
    checksum: 1883091326,
    head: [
      0.999421134824, 0.000487298437, 0, 1.004595334408, 0.145069466465, 0,
      0.948692967763, 0.306163672253, 0,
    ],
  },
  spiral: {
    n: 120,
    c0: 60,
    c1: 60,
    checksum: 2975151956,
    head: [
      -0.037451800751, 0, 0, -0.017650940898, 0, 1, 0.000822487077,
      0.003465194847, 0,
    ],
  },
};

describe.each(Object.keys(FINGERPRINTS) as (keyof typeof FINGERPRINTS)[])(
  "%s dataset determinism",
  (name) => {
    const expected = FINGERPRINTS[name];

    it("is registered", () => {
      expect(DATASET_NAMES).toContain(name);
    });

    it("is stable across cache hits and seedRefresh (no Math.random)", () => {
      const a = getDataset(name);
      const b = getDataset(name);
      expect(a.samples.map((s) => [...s.x, ...s.y])).toEqual(
        b.samples.map((s) => [...s.x, ...s.y])
      );
      const c = getDataset(name, true);
      expect(c.samples.map((s) => [...s.x, ...s.y])).toEqual(
        a.samples.map((s) => [...s.x, ...s.y])
      );
    });

    it("pins sample count, class balance, checksum, and head geometry", () => {
      const ds = getDataset(name, true);
      expect(ds.samples.length).toBe(expected.n);
      const { c0, c1 } = classCounts(ds);
      expect(c0).toBe(expected.c0);
      expect(c1).toBe(expected.c1);
      expect(checksum(ds)).toBe(expected.checksum);

      const head = headFingerprint(ds);
      expect(head).toHaveLength(expected.head.length);
      for (let i = 0; i < head.length; i++) {
        expect(head[i]).toBeCloseTo(expected.head[i]!, 9);
      }
    });
  }
);

describe("toy datasets no longer depend on Math.random at generation", () => {
  it("linear/moons/circles/spiral match after forced refresh in a fresh cache path", () => {
    // seedRefresh=true bypasses cache; equality implies pure detUnit generation.
    const names: DatasetName[] = ["linear", "moons", "circles", "spiral"];
    for (const name of names) {
      const a = getDataset(name, true);
      const b = getDataset(name, true);
      expect(checksum(a)).toBe(checksum(b));
      expect(a.samples.length).toBe(b.samples.length);
    }
  });
});
