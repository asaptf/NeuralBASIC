import type { CurriculumProgress } from "@/curriculum/types";
import type { LayerWeights, NetworkConfig, TrainConfig } from "@/engine/types";

const EXPERIMENT_KEY = "neuralbasic:experiment:v1";
const PROGRESS_KEY = "neuralbasic:progress:v1";
const THEME_KEY = "neuralbasic:theme:v1";

export interface ExperimentState {
  dsl: string;
  network: NetworkConfig;
  trainConfig: TrainConfig;
  weights?: LayerWeights[];
  history?: { losses: number[]; accuracies: number[] };
  name?: string;
  savedAt?: string;
}

export function saveExperiment(state: ExperimentState): void {
  if (typeof window === "undefined") return;
  const payload: ExperimentState = {
    ...state,
    savedAt: new Date().toISOString(),
  };
  localStorage.setItem(EXPERIMENT_KEY, JSON.stringify(payload));
}

export function loadExperiment(): ExperimentState | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(EXPERIMENT_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as ExperimentState;
  } catch {
    return null;
  }
}

export function exportExperimentJSON(state: ExperimentState): string {
  return JSON.stringify(
    { ...state, savedAt: new Date().toISOString(), format: "neuralbasic-experiment-v1" },
    null,
    2
  );
}

export function importExperimentJSON(json: string): ExperimentState {
  const data = JSON.parse(json) as ExperimentState & { format?: string };
  if (!data.dsl || !data.network || !data.trainConfig) {
    throw new Error("Invalid experiment JSON");
  }
  return data;
}

export function saveProgress(progress: CurriculumProgress): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(PROGRESS_KEY, JSON.stringify(progress));
}

export function loadProgress(): CurriculumProgress | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(PROGRESS_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as CurriculumProgress;
  } catch {
    return null;
  }
}

export function saveTheme(theme: "modern" | "retro"): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(THEME_KEY, theme);
}

export function loadTheme(): "modern" | "retro" | null {
  if (typeof window === "undefined") return null;
  const t = localStorage.getItem(THEME_KEY);
  if (t === "modern" || t === "retro") return t;
  return null;
}

export function downloadText(filename: string, text: string, mime = "application/json") {
  if (typeof window === "undefined") return;
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
