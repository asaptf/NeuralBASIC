"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { useAppStore } from "@/store/useAppStore";

/**
 * The first-run guide.
 *
 * Written against a constraint the rest of the app takes seriously: this is not
 * a passive course, and a tour is the easiest way to accidentally build one. So
 * every step ends in something to do rather than something more to watch, the
 * whole thing is four screens long, and it explains only what a reader cannot
 * discover by pressing Train — where the panels are, that progress is gated on
 * explaining, and the two gestures that exist nowhere else in the interface.
 *
 * The figures animate because motion is the honest medium for what they show:
 * a network learning over epochs, and a loop you go round. Each one holds a
 * readable final state when animation is off, so `prefers-reduced-motion`
 * removes the movement without removing the diagram.
 */

interface Step {
  id: string;
  eyebrow: string;
  title: string;
  body: ReactNode;
  figure: ReactNode;
  figureLabel: string;
  /** Something to do now, instead of reading the next screen. */
  action?: { label: string; kind: "train" | "lesson" };
}

const STEPS: Step[] = [
  {
    id: "map",
    eyebrow: "The workspace",
    title: "Six panels, one experiment",
    figureLabel:
      "Layout of the workspace: Curriculum on the left, Editor above Network, the Data Lab in the middle, Metrics above the Tutor on the right.",
    figure: <WorkspaceMap />,
    body: (
      <>
        Every panel describes the same run. The <strong>Data Lab</strong> is the
        biggest one because seeing <em>which</em> points a network gets wrong
        teaches more than any number does — the rest are instruments around it.
      </>
    ),
  },
  {
    id: "immediate",
    eyebrow: "Immediate Mode",
    title: "You watch it learn",
    figureLabel:
      "A decision boundary swinging into place over scattered points while a loss curve draws itself.",
    figure: <LiveTraining />,
    body: (
      <>
        Training advances epoch by epoch, so you see the boundary move and the
        loss fall rather than jumping from before to after. Change the learning
        rate, the dataset or the program and it retrains at once. There are no
        cells to re-run.
      </>
    ),
    action: { label: "Train something now", kind: "train" },
  },
  {
    id: "loop",
    eyebrow: "How you make progress",
    title: "Predict first, then find out",
    figureLabel:
      "A four-step cycle: predict, experiment, observe, explain, returning to predict.",
    figure: <LearningLoop />,
    body: (
      <>
        Chapters unlock when you <strong>explain what happened</strong>, not when
        you finish reading. Guessing wrong is part of it — a wrong prediction
        gets a nudge rather than the answer, and the tutor will refuse to hand
        you a working solution however you ask.
      </>
    ),
    action: { label: "Open Chapter 1's lesson", kind: "lesson" },
  },
  {
    id: "gestures",
    eyebrow: "Worth knowing",
    title: "Two gestures you would not guess",
    figureLabel:
      "Left: a pointer crossing a scatter plot leaves marked test points and moves a confidence needle. Right: pixels lighting up one by one on a small grid.",
    figure: <TwoGestures />,
    body: (
      <>
        Once a network is trained, <strong>tap the plot</strong> to ask it about
        a point that was never in the data. On image datasets that plot becomes
        a grid you <strong>paint an input on</strong> — draw a shape, then draw
        the same shape somewhere else and watch what survives the move.
      </>
    ),
  },
];

export function WelcomeGuide() {
  const open = useAppStore((s) => s.welcomeOpen);
  const setOpen = useAppStore((s) => s.setWelcomeOpen);
  const trainNow = useAppStore((s) => s.trainNow);
  const setLessonOpen = useAppStore((s) => s.setLessonOpen);
  const [index, setIndex] = useState(0);
  const sheetRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    sheetRef.current?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [open, setOpen]);

  // Reopening should start at the beginning; someone who came back for the map
  // is not looking for wherever they happened to stop last time.
  useEffect(() => {
    if (open) setIndex(0);
  }, [open]);

  if (!open) return null;

  const step = STEPS[index]!;
  const last = index === STEPS.length - 1;

  const runAction = () => {
    if (step.action?.kind === "train") trainNow("welcome-guide");
    if (step.action?.kind === "lesson") setLessonOpen(true);
    setOpen(false);
  };

  return (
    <div className="wg-scrim" data-testid="welcome-guide">
      <div
        className="wg-sheet"
        role="dialog"
        aria-modal="true"
        aria-label="Welcome to NeuralBASIC"
        tabIndex={-1}
        ref={sheetRef}
      >
        <header className="wg-head">
          <p className="wg-brand">
            NeuralBASIC <span>· a lab, not a lecture</span>
          </p>
          <button
            type="button"
            className="btn btn-ghost"
            data-testid="welcome-skip"
            onClick={() => setOpen(false)}
          >
            Skip ✕
          </button>
        </header>

        {/* Keyed so each step's figure remounts and its animation starts fresh
            as the reader arrives, rather than mid-cycle. */}
        <figure className="wg-figure" key={step.id}>
          <div className="wg-figure-frame" role="img" aria-label={step.figureLabel}>
            {step.figure}
          </div>
        </figure>

        <div className="wg-copy">
          <p className="wg-eyebrow">{step.eyebrow}</p>
          <h2 className="wg-title">{step.title}</h2>
          <p className="wg-body">{step.body}</p>
        </div>

        <footer className="wg-foot">
          <ol className="wg-dots" aria-label={`Step ${index + 1} of ${STEPS.length}`}>
            {STEPS.map((s, i) => (
              <li key={s.id}>
                <button
                  type="button"
                  className={`wg-dot${i === index ? " is-current" : ""}`}
                  aria-label={`Step ${i + 1}: ${s.title}`}
                  aria-current={i === index ? "step" : undefined}
                  onClick={() => setIndex(i)}
                />
              </li>
            ))}
          </ol>

          <div className="wg-actions">
            {step.action && (
              <button
                type="button"
                className="btn"
                data-testid="welcome-action"
                onClick={runAction}
              >
                {step.action.label}
              </button>
            )}
            {index > 0 && (
              <button
                type="button"
                className="btn btn-ghost"
                onClick={() => setIndex((i) => i - 1)}
              >
                Back
              </button>
            )}
            <button
              type="button"
              className="btn btn-primary"
              data-testid="welcome-next"
              onClick={() => (last ? setOpen(false) : setIndex((i) => i + 1))}
            >
              {/* Not "Start Chapter 1": the guide can be reopened from any
                  chapter, and closing it would not take the reader there. */}
              {last ? "Get started" : "Next →"}
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}

/* ── Figures ──
   Plain SVG on the theme's own tokens, so Retro Blue gets its own palette for
   free. Animation lives in CSS (see globals.css) and is defined so that the
   un-animated state is the finished one. */

interface MapPanel {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  primary?: boolean;
}

/** The desktop grid: curriculum | editor/network | lab | metrics/tutor. */
const PANELS_WIDE: MapPanel[] = [
  { x: 6, y: 6, w: 104, h: 158, label: "Curriculum" },
  { x: 118, y: 6, w: 128, h: 74, label: "Editor" },
  { x: 118, y: 90, w: 128, h: 74, label: "Network" },
  { x: 254, y: 6, w: 170, h: 158, label: "Data Lab", primary: true },
  { x: 432, y: 6, w: 122, h: 74, label: "Metrics" },
  { x: 432, y: 90, w: 122, h: 74, label: "Tutor" },
];

/**
 * The phone layout, in the order the panels actually stack there — the lab
 * first, then the rest. A map is a claim about the reader's own screen, so
 * showing them four columns they do not have would be a small lie on the very
 * first thing they see. Kept in step with the `max-width: 767px` grid in
 * globals.css.
 */
const PANELS_NARROW: MapPanel[] = [
  { x: 6, y: 6, w: 208, h: 60, label: "Data Lab", primary: true },
  { x: 6, y: 72, w: 208, h: 38, label: "Curriculum" },
  { x: 6, y: 116, w: 208, h: 38, label: "Editor" },
  { x: 6, y: 160, w: 208, h: 38, label: "Network" },
  { x: 6, y: 204, w: 208, h: 38, label: "Metrics" },
  { x: 6, y: 248, w: 208, h: 38, label: "Tutor" },
];

function MapPanels({ panels }: { panels: MapPanel[] }) {
  return (
    <>
      {panels.map((p, i) => (
        <g
          key={p.label}
          className={`wg-panel${p.primary ? " is-primary" : ""}`}
          style={{ ["--i" as string]: i }}
        >
          <rect x={p.x} y={p.y} width={p.w} height={p.h} rx="7" />
          <text x={p.x + p.w / 2} y={p.y + p.h / 2 + 4}>
            {p.label}
          </text>
        </g>
      ))}
    </>
  );
}

function WorkspaceMap() {
  // Both are rendered and one is hidden by media query, so the diagram matches
  // the layout without JS deciding at runtime and disagreeing with the server.
  return (
    <>
      <svg viewBox="0 0 560 170" className="wg-svg wg-map-wide" aria-hidden="true">
        <MapPanels panels={PANELS_WIDE} />
      </svg>
      <svg viewBox="0 0 220 292" className="wg-svg wg-map-narrow" aria-hidden="true">
        <MapPanels panels={PANELS_NARROW} />
      </svg>
    </>
  );
}

function LiveTraining() {
  // The line below runs from (20,40) to (156,140); every class-0 dot sits on its
  // lower-left side and every class-1 dot on its upper-right. Worth checking if
  // these ever move: a boundary that does not actually separate the classes is
  // exactly the wrong first picture for an app about decision boundaries.
  const dots = [
    [34, 82, 0], [52, 108, 0], [30, 122, 0], [68, 136, 0], [48, 146, 0],
    [100, 40, 1], [126, 58, 1], [146, 34, 1], [108, 74, 1], [134, 92, 1],
  ] as const;

  return (
    <svg viewBox="0 0 560 170" className="wg-svg" aria-hidden="true">
      <rect x="6" y="6" width="164" height="158" rx="7" className="wg-plot" />
      <g className="wg-boundary-spin">
        <line x1="20" y1="40" x2="156" y2="140" className="wg-boundary" />
      </g>
      {dots.map(([cx, cy, c], i) => (
        <circle
          key={i}
          cx={cx}
          cy={cy}
          r="5"
          className={c === 1 ? "wg-dot-1" : "wg-dot-0"}
        />
      ))}

      <rect x="196" y="6" width="358" height="158" rx="7" className="wg-plot" />
      <text x="208" y="26" className="wg-axis-label">
        loss
      </text>
      {/* A curve that falls fast and then flattens — the shape of a run that
          worked, which is what a first-time reader should expect to look for. */}
      <path
        d="M 208 40 C 250 132, 268 138, 300 142 C 340 147, 380 149, 420 150 C 470 151, 510 151, 544 152"
        className="wg-loss-curve"
      />
      <text x="208" y="158" className="wg-axis-label">
        0
      </text>
      <text x="544" y="158" className="wg-axis-label" textAnchor="end">
        epochs
      </text>
    </svg>
  );
}

const LOOP_NODES = [
  { x: 280, y: 22, label: "predict", tx: 280, ty: 12, anchor: "middle" },
  { x: 346, y: 88, label: "experiment", tx: 358, ty: 92, anchor: "start" },
  { x: 280, y: 154, label: "observe", tx: 280, ty: 176, anchor: "middle" },
  { x: 214, y: 88, label: "explain", tx: 202, ty: 92, anchor: "end" },
] as const;

/**
 * Chevrons at the midpoint of each quarter-arc. A ring of four labels reads
 * equally well backwards, and "explain, then predict" is not the loop — so the
 * direction has to be drawn, not left to the reader's assumption.
 */
const LOOP_ARROWS = [45, 135, 225, 315].map((deg) => {
  const rad = (deg * Math.PI) / 180;
  return {
    deg,
    x: 280 + 66 * Math.sin(rad),
    y: 88 - 66 * Math.cos(rad),
  };
});

function LearningLoop() {
  return (
    <svg viewBox="0 0 560 190" className="wg-svg" aria-hidden="true">
      <circle cx="280" cy="88" r="66" className="wg-ring" />
      {LOOP_ARROWS.map((a) => (
        <path
          key={a.deg}
          d="M -4 -4 L 4 0 L -4 4 z"
          className="wg-ring-arrow"
          transform={`translate(${a.x.toFixed(1)} ${a.y.toFixed(1)}) rotate(${a.deg})`}
        />
      ))}
      {/* cx/cy stay at the origin: offset-path translates an element from its
          own coordinates, so a centred circle would orbit at double the offset. */}
      <circle cx="0" cy="0" r="5" className="wg-ring-runner" />
      {LOOP_NODES.map((n) => (
        <g key={n.label} className="wg-node">
          <circle cx={n.x} cy={n.y} r="7" />
          <text x={n.tx} y={n.ty} textAnchor={n.anchor}>
            {n.label}
          </text>
        </g>
      ))}
      <text x="280" y="93" className="wg-ring-centre">
        repeat
      </text>
    </svg>
  );
}

function TwoGestures() {
  const dots = [
    [46, 116, 0], [74, 130, 0], [38, 92, 0], [96, 104, 0],
    [150, 52, 1], [178, 68, 1], [136, 80, 1], [190, 40, 1],
  ] as const;
  // A three-cell vertical bar on a 6x6 grid — the motif Chapter 4 is about.
  const lit = [8, 14, 20];

  return (
    <svg viewBox="0 0 560 180" className="wg-svg" aria-hidden="true">
      <rect x="6" y="6" width="228" height="122" rx="7" className="wg-plot" />
      {dots.map(([cx, cy, c], i) => (
        <circle key={i} cx={cx} cy={cy} r="5" className={c === 1 ? "wg-dot-1" : "wg-dot-0"} />
      ))}
      {/* Answers already pinned. Each sits in the region of the class it is
          coloured for, because in the real panel the marker's colour *is* the
          model's verdict — a mismatch here would teach the wrong thing. */}
      <path d="M 128 54 l 8 8 l -8 8 l -8 -8 z" className="wg-pin wg-pin-1" />
      <path d="M 62 106 l 8 8 l -8 8 l -8 -8 z" className="wg-pin wg-pin-0" />
      <g className="wg-cursor-track">
        <line x1="0" y1="14" x2="0" y2="120" className="wg-cursor-line" />
        <path d="M 0 60 l 8 8 l -8 8 l -8 -8 z" className="wg-cursor-pin" />
      </g>

      <rect x="6" y="144" width="228" height="10" rx="5" className="wg-needle-track" />
      <rect x="118" y="140" width="2" height="18" className="wg-needle-mid" />
      <rect y="141" width="4" height="16" rx="2" className="wg-needle" />

      <line x1="270" y1="14" x2="270" y2="166" className="wg-divider" />

      <g transform="translate(306 18)">
        {Array.from({ length: 36 }, (_, i) => {
          const on = lit.indexOf(i);
          return (
            <rect
              key={i}
              x={(i % 6) * 24}
              y={Math.floor(i / 6) * 24}
              width="21"
              height="21"
              rx="3"
              className={on >= 0 ? "wg-pixel is-on" : "wg-pixel"}
              style={on >= 0 ? { ["--i" as string]: on } : undefined}
            />
          );
        })}
      </g>
      <text x="306" y="176" className="wg-axis-label">
        paint an input
      </text>
    </svg>
  );
}
