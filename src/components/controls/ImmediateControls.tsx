"use client";

import { DATASET_NAMES } from "@/engine/datasets";
import type { DatasetName } from "@/engine/types";
import { useAppStore } from "@/store/useAppStore";

export function ImmediateControls() {
  const trainConfig = useAppStore((s) => s.trainConfig);
  const setLearningRate = useAppStore((s) => s.setLearningRate);
  const setEpochs = useAppStore((s) => s.setEpochs);
  const setDataset = useAppStore((s) => s.setDataset);
  const trainNow = useAppStore((s) => s.trainNow);
  const isTraining = useAppStore((s) => s.isTraining);
  const isPaused = useAppStore((s) => s.isPaused);
  const epochsRun = useAppStore((s) => s.epochsRun);
  const totalEpochs = useAppStore((s) => s.totalEpochs);
  const pauseTraining = useAppStore((s) => s.pauseTraining);
  const resumeTraining = useAppStore((s) => s.resumeTraining);
  const stepEpoch = useAppStore((s) => s.stepEpoch);
  const resetWeights = useAppStore((s) => s.resetWeights);
  const parseAndApplyDsl = useAppStore((s) => s.parseAndApplyDsl);

  const actionError = useAppStore((s) => s.actionError);
  const clearActionError = useAppStore((s) => s.clearActionError);

  const pct = totalEpochs > 0 ? Math.min(100, (epochsRun / totalEpochs) * 100) : 0;

  return (
    <div className="immediate-bar" data-testid="immediate-controls">
      <label className="control">
        <span>LR</span>
        <input
          type="number"
          inputMode="decimal"
          step="0.05"
          min="0.001"
          max="50"
          value={trainConfig.learningRate}
          data-testid="control-lr"
          disabled={isTraining && !isPaused}
          onChange={(e) => setLearningRate(Number(e.target.value))}
        />
      </label>
      <label className="control">
        <span>Epochs</span>
        <input
          type="number"
          inputMode="numeric"
          step="10"
          min="1"
          max="2000"
          value={trainConfig.epochs}
          data-testid="control-epochs"
          disabled={isTraining && !isPaused}
          onChange={(e) => setEpochs(Number(e.target.value))}
        />
      </label>
      <label className="control">
        <span>Dataset</span>
        <select
          value={trainConfig.dataset}
          data-testid="control-dataset"
          disabled={isTraining && !isPaused}
          onChange={(e) => setDataset(e.target.value as DatasetName)}
        >
          {DATASET_NAMES.map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
      </label>

      <button
        type="button"
        className="btn btn-primary"
        data-testid="btn-train"
        disabled={isTraining && !isPaused}
        onClick={() => {
          parseAndApplyDsl();
          trainNow("immediate-train");
        }}
      >
        {isTraining && !isPaused ? "Training…" : "Train ▶"}
      </button>

      {isTraining &&
        (isPaused ? (
          <button
            type="button"
            className="btn"
            data-testid="btn-resume"
            onClick={() => resumeTraining()}
          >
            Resume ▶
          </button>
        ) : (
          <button
            type="button"
            className="btn"
            data-testid="btn-pause"
            onClick={() => pauseTraining()}
          >
            Pause ⏸
          </button>
        ))}

      <button
        type="button"
        className="btn"
        data-testid="btn-step"
        title="Advance exactly one epoch"
        disabled={isTraining && !isPaused}
        onClick={() => stepEpoch()}
      >
        Step +1
      </button>

      <button
        type="button"
        className="btn"
        data-testid="btn-reset"
        onClick={() => resetWeights()}
      >
        Reset
      </button>

      <div className="train-progress" aria-hidden="true">
        <div className="train-progress-fill" style={{ width: `${pct}%` }} />
      </div>
      <span
        className="train-epoch-readout"
        data-testid="epoch-readout"
        role="status"
        aria-label={`epoch ${epochsRun} of ${totalEpochs}`}
      >
        {totalEpochs > 0 ? `${epochsRun}/${totalEpochs} ep` : "idle"}
      </span>

      {actionError && (
        <div className="action-error" role="alert" data-testid="action-error">
          <span>{actionError}</span>
          <button
            type="button"
            className="action-error-dismiss"
            aria-label="Dismiss"
            onClick={() => clearActionError()}
          >
            ×
          </button>
        </div>
      )}
    </div>
  );
}
