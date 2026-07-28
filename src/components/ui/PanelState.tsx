/**
 * One treatment for "this panel has nothing to show yet" and "this panel
 * failed".
 *
 * These states used to be expressed four different ways — text baked into a
 * canvas bitmap, text inside an SVG, em-dashes in the metric cards, and in the
 * network panel's case nothing at all. Canvas- and SVG-drawn copy also can't
 * be selected, can't be read by a screen reader, and doesn't inherit the type
 * scale, so it drifted from the rest of the UI.
 *
 * Rendered as a DOM overlay so the panel underneath can keep drawing whatever
 * real context it has (axes, layer structure) behind the message.
 */
export function PanelState({
  title,
  hint,
  tone = "idle",
  anchor = "center",
  testId,
}: {
  title: string;
  hint?: string;
  tone?: "idle" | "error";
  /**
   * Where the plate sits. `bottom` when the panel is already drawing something
   * the reader should see — a scatter plot, a layer diagram. Centring the plate
   * there covers the very thing the copy points at: on `moons` the message
   * "the data is plotted" sat on top of the lower crescent and hid it.
   */
  anchor?: "center" | "bottom";
  testId?: string;
}) {
  return (
    <div
      className={`panel-state panel-state-${tone} panel-state-at-${anchor}`}
      role={tone === "error" ? "alert" : "status"}
      data-testid={testId}
    >
      <div className="panel-state-plate">
        <p className="panel-state-title">{title}</p>
        {hint && <p className="panel-state-hint">{hint}</p>}
      </div>
    </div>
  );
}
