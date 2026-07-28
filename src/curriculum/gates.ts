import type { ChallengeStep } from "./types";

/** Everything an experiment gate is allowed to look at. */
export interface ExperimentOutcome {
  accuracy: number;
  loss: number;
  dataset: string;
  dsl: string;
}

/**
 * Does this training outcome satisfy a challenge's experiment gate?
 *
 * Single source of truth on purpose: the store decides real progression with
 * this, and the curriculum tests prove every gate is winnable with it. A second
 * copy would let the two drift while both looked green.
 */
export function experimentCheckPasses(
  check: NonNullable<ChallengeStep["experimentCheck"]>,
  outcome: ExperimentOutcome
): boolean {
  if (check.dataset && outcome.dataset !== check.dataset) return false;
  if (check.minAccuracy != null && outcome.accuracy < check.minAccuracy) {
    return false;
  }
  if (check.maxLoss != null && outcome.loss > check.maxLoss) return false;
  if (check.dslIncludes) {
    const lower = outcome.dsl.toLowerCase();
    for (const frag of check.dslIncludes) {
      if (!lower.includes(frag.toLowerCase())) return false;
    }
  }
  return true;
}
