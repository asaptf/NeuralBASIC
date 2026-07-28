"use client";

import { useAppStore } from "@/store/useAppStore";

const W = 300;
const H = 82;
const PAD_L = 34;
const PAD_R = 6;
const PAD_T = 8;
const PAD_B = 16;

function polyline(
  values: number[],
  yMin: number,
  yMax: number
): string {
  if (values.length < 2) return "";
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const span = yMax - yMin || 1;
  return values
    .map((v, i) => {
      const x = PAD_L + (i / (values.length - 1)) * plotW;
      const y = PAD_T + plotH - ((v - yMin) / span) * plotH;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function Chart({
  label,
  values,
  yMin,
  yMax,
  color,
  format,
  baseline,
  baselineLabel,
  testId,
}: {
  label: string;
  values: number[];
  yMin: number;
  yMax: number;
  color: string;
  format: (n: number) => string;
  baseline?: number;
  baselineLabel?: string;
  testId?: string;
}) {
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const span = yMax - yMin || 1;
  const baselineY =
    baseline != null
      ? PAD_T + plotH - ((baseline - yMin) / span) * plotH
      : null;

  return (
    <div>
      <div className="chart-label">{label}</div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="chart-svg"
        data-testid={testId}
        role="img"
        aria-label={`${label} over ${values.length} epochs`}
      >
        {/* plot frame */}
        <rect
          x={PAD_L}
          y={PAD_T}
          width={plotW}
          height={plotH}
          className="chart-frame"
        />

        {/* y axis labels */}
        <text x={PAD_L - 4} y={PAD_T + 7} className="chart-tick" textAnchor="end">
          {format(yMax)}
        </text>
        <text
          x={PAD_L - 4}
          y={PAD_T + plotH}
          className="chart-tick"
          textAnchor="end"
        >
          {format(yMin)}
        </text>

        {baselineY != null && (
          <>
            <line
              x1={PAD_L}
              x2={PAD_L + plotW}
              y1={baselineY}
              y2={baselineY}
              className="chart-baseline"
            />
            {baselineLabel && (
              <text
                x={PAD_L + plotW - 2}
                y={baselineY - 3}
                className="chart-tick"
                textAnchor="end"
              >
                {baselineLabel}
              </text>
            )}
          </>
        )}

        {values.length > 1 ? (
          <polyline
            fill="none"
            stroke={color}
            strokeWidth="1.5"
            vectorEffect="non-scaling-stroke"
            points={polyline(values, yMin, yMax)}
          />
        ) : (
          <text x={PAD_L + 8} y={H / 2} className="chart-empty">
            train to see the curve
          </text>
        )}

        {/* x axis */}
        <text x={PAD_L} y={H - 3} className="chart-tick">
          0
        </text>
        <text
          x={PAD_L + plotW}
          y={H - 3}
          className="chart-tick"
          textAnchor="end"
        >
          {values.length ? `${values.length} ep` : "epoch"}
        </text>
      </svg>
    </div>
  );
}

export function MetricsPanel() {
  const history = useAppStore((s) => s.history);
  const snapshot = useAppStore((s) => s.lastSnapshot);
  const trainConfig = useAppStore((s) => s.trainConfig);
  const lastTrigger = useAppStore((s) => s.lastTrigger);

  const losses = history.losses;
  const accuracies = history.accuracies;

  // Autoscale loss to the range actually observed, so small-but-real changes
  // are visible instead of being flattened against a hardcoded ceiling.
  const lossMax = losses.length ? Math.max(...losses) : 1;
  const lossMin = losses.length ? Math.min(...losses) : 0;
  const lossPad = (lossMax - lossMin) * 0.12 || Math.max(lossMax * 0.1, 0.01);
  const lossTop = lossMax + lossPad;
  const lossBottom = Math.max(0, lossMin - lossPad);

  const first = losses[0];
  const last = losses[losses.length - 1];
  const delta = first != null && last != null ? last - first : null;

  return (
    <div className="panel flex h-full min-h-0 flex-col" data-testid="metrics-panel">
      <div className="panel-header">
        <span>Metrics</span>
        <span className="panel-header-note">loss · accuracy</span>
      </div>
      <div className="flex flex-1 flex-col gap-2 overflow-auto p-2.5">
        <div className="grid grid-cols-2 gap-2">
          <Metric
            label="Loss"
            value={snapshot ? snapshot.loss.toFixed(4) : "—"}
            testId="metric-loss"
            hint={
              delta != null
                ? `${delta <= 0 ? "▼" : "▲"} ${Math.abs(delta).toFixed(4)}`
                : undefined
            }
            hintTone={delta == null ? undefined : delta <= 0 ? "good" : "bad"}
          />
          <Metric
            label="Accuracy"
            value={snapshot ? `${(snapshot.accuracy * 100).toFixed(1)}%` : "—"}
            testId="metric-accuracy"
          />
          <Metric
            label="Epoch"
            value={snapshot ? String(snapshot.epoch) : "—"}
            testId="metric-epoch"
          />
          <Metric
            label="LR"
            value={String(trainConfig.learningRate)}
            testId="metric-lr"
          />
        </div>

        <Chart
          label="Loss"
          values={losses}
          yMin={lossBottom}
          yMax={lossTop}
          color="var(--accent)"
          format={(n) => n.toFixed(3)}
          testId="loss-chart"
        />

        <Chart
          label="Accuracy"
          values={accuracies}
          yMin={0}
          yMax={1}
          color="var(--accent-2)"
          format={(n) => `${Math.round(n * 100)}%`}
          baseline={0.5}
          baselineLabel="chance"
          testId="accuracy-chart"
        />

        <div className="text-xs opacity-70">
          Dataset: <strong>{trainConfig.dataset}</strong>
          {lastTrigger && (
            <>
              {" "}
              · last train:{" "}
              <span data-testid="last-trigger">{lastTrigger}</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function Metric({
  label,
  value,
  testId,
  hint,
  hintTone,
}: {
  label: string;
  value: string;
  testId?: string;
  hint?: string;
  hintTone?: "good" | "bad";
}) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value" data-testid={testId}>
        {value}
      </div>
      {hint && <div className={`metric-hint ${hintTone ?? ""}`}>{hint}</div>}
    </div>
  );
}
