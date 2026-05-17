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

const GRIP_DOT_COUNT = 24;

function GripDots() {
  return (
    <div className="ctrl-bar-grip" aria-hidden>
      {Array.from({ length: GRIP_DOT_COUNT }).map((_, i) => (
        <span key={i} />
      ))}
    </div>
  );
}

function SettingsIcon() {
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
      <path d="M12.22 2h-.44a2 2 0 00-2 2v.18a2 2 0 01-1 1.73l-.43.25a2 2 0 01-2 0l-.15-.08a2 2 0 00-2.73.73l-.22.38a2 2 0 00.73 2.73l.15.1a2 2 0 011 1.72v.51a2 2 0 01-1 1.74l-.15.09a2 2 0 00-.73 2.73l.22.38a2 2 0 002.73.73l.15-.08a2 2 0 012 0l.43.25a2 2 0 011 1.73V20a2 2 0 002 2h.44a2 2 0 002-2v-.18a2 2 0 011-1.73l.43-.25a2 2 0 012 0l.15.08a2 2 0 002.73-.73l.22-.39a2 2 0 00-.73-2.73l-.15-.08a2 2 0 01-1-1.74v-.5a2 2 0 011-1.74l.15-.09a2 2 0 00.73-2.73l-.22-.38a2 2 0 00-2.73-.73l-.15.08a2 2 0 01-2 0l-.43-.25a2 2 0 01-1-1.73V4a2 2 0 00-2-2z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
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
  const removeNode = usePipelineStore((s) => s.removeNode);
  const setSelectedNodeId = usePipelineStore((s) => s.setSelectedNodeId);

  const title = data.label ?? data.method;

  const displayParams: ControlNodeParam[] =
    data.params && data.params.length > 0
      ? data.params
      : Object.entries(data.args)
          .filter(([, v]) => v != null)
          .slice(0, 3)
          .map(([key, val]) => ({ label: key, value: String(val) }));

  const onSettings = (event: MouseEvent) => {
    event.stopPropagation();
    if (data.onSettings) data.onSettings();
    else setSelectedNodeId(id);
  };

  const onClose = (event: MouseEvent) => {
    event.stopPropagation();
    if (data.onClose) data.onClose();
    else removeNode(id);
  };

  const footerLabel = data.status ?? CATEGORY_FOOTER_LABEL[data.category] ?? data.category;

  return (
    <div
      className={`ctrl-node${selected ? " selected" : ""}`}
      data-category={data.category}
    >
      <Handle type="target" position={Position.Left} id="in" />
      <Handle type="source" position={Position.Right} id="out" />

      <div className="ctrl-bar">
        <span className="ctrl-bar-title" title={title}>
          {title}
        </span>
        <div className="ctrl-bar-spacer" />
        <GripDots />
        <button
          type="button"
          className="ctrl-bar-btn"
          onClick={onSettings}
          aria-label="Settings"
          title="Settings"
        >
          <SettingsIcon />
        </button>
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
