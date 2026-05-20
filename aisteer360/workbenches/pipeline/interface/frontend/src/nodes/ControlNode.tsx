import { memo, type MouseEvent } from "react";
import { Handle, Position, type NodeProps } from "reactflow";
import type { ControlNodeData, ControlNodeParam } from "../types";
import { usePipelineStore } from "../store/pipelineStore";

const CATEGORY_FOOTER_LABEL: Record<string, string> = {
  input_control: "input control",
  structural_control: "structural control",
  state_control: "state control",
  output_control: "output control",
};

const GRIP_DOT_COUNT = 12;

function GripDots() {
  return (
    <div className="ctrl-bar-grip" aria-hidden>
      {Array.from({ length: GRIP_DOT_COUNT }).map((_, i) => (
        <span key={i} />
      ))}
    </div>
  );
}

function CloseIcon() {
  return (
    <svg
      width="9"
      height="9"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

function ControlNodeImpl({ id, data, selected }: NodeProps<ControlNodeData>) {
  const requestDeleteNode = usePipelineStore((s) => s.requestDeleteNode);
  const grouped = usePipelineStore((s) => {
    const g = s.lockedGroups.find((grp) => grp.members.includes(id));
    return Boolean(g && g.members.length >= 2);
  });

  const title = data.label || data.method || "unset";

  const displayParams: ControlNodeParam[] = (
    data.params && data.params.length > 0
      ? data.params
      : Object.entries(data.args)
          .filter(([, v]) => v != null)
          .map(([key, val]) => ({ label: key, value: String(val) }))
  ).slice(0, 3);

  const onClose = (event: MouseEvent) => {
    event.stopPropagation();
    if (data.onClose) data.onClose();
    else requestDeleteNode(id);
  };

  const footerLabel =
    data.status ?? (data.category ? CATEGORY_FOOTER_LABEL[data.category] : "control");

  return (
    <div
      className={`ctrl-node${selected ? " selected" : ""}${grouped ? " grouped" : ""}`}
      data-category={data.category ?? "neutral"}
    >
      {/* when grouped, members accept inputs only — outputs leave from the
          synthetic group node behind them. flipping handle type to target
          tells React Flow the handle can't source connections. */}
      <Handle type={grouped ? "target" : "source"} position={Position.Left} id="left" />
      <Handle type={grouped ? "target" : "source"} position={Position.Right} id="right" />
      <Handle type={grouped ? "target" : "source"} position={Position.Top} id="top" />
      <Handle type={grouped ? "target" : "source"} position={Position.Bottom} id="bottom" />

      <div className="ctrl-bar">
        <span className="ctrl-bar-title" title={title}>
          {title}
        </span>
        <div className="ctrl-bar-spacer" />
        <GripDots />
        <button
          type="button"
          className="ctrl-bar-btn"
          onClick={onClose}
          aria-label={`Remove ${data.method}`}
          title="Close"
        >
          <CloseIcon />
        </button>
      </div>

      <div className="ctrl-body">
        {displayParams.length === 0 ? (
          <div className="ctrl-empty">no parameters set</div>
        ) : (
          displayParams.map((p, idx) => (
            <div key={`${p.label}-${idx}`} className="ctrl-card">
              <div className="ctrl-card-label">
                {p.icon ? <span className="ctrl-card-icon">{p.icon}</span> : null}
                {p.label}
              </div>
              <div className="ctrl-card-value" title={p.value}>
                {p.value}
              </div>
            </div>
          ))
        )}
      </div>

      {footerLabel ? <div className="ctrl-footer">{footerLabel}</div> : null}
    </div>
  );
}

export const ControlNode = memo(ControlNodeImpl);
