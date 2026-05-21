import { memo, type MouseEvent } from "react";
import { Handle, Position, type NodeProps } from "reactflow";
import type { ControlNodeData, ControlNodeParam } from "../types";
import { usePipelineStore } from "../store/pipelineStore";

/* ── category metadata ──────────────────────────────────────────── */

const CATEGORY_SIDEBAR_LABEL: Record<string, string> = {
  input_control: "input",
  structural_control: "structural",
  state_control: "state",
  output_control: "output",
};

/* 12×12 inline SVGs — one per category */

function InputIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M15 3h4a2 2 0 012 2v14a2 2 0 01-2 2h-4" />
      <polyline points="10 17 15 12 10 7" />
      <line x1="15" y1="12" x2="3" y2="12" />
    </svg>
  );
}

function StructuralIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="12" cy="12" r="3" />
      <circle cx="4" cy="6" r="2" />
      <circle cx="20" cy="6" r="2" />
      <circle cx="4" cy="18" r="2" />
      <circle cx="20" cy="18" r="2" />
      <line x1="9.5" y1="10" x2="5.5" y2="7.5" />
      <line x1="14.5" y1="10" x2="18.5" y2="7.5" />
      <line x1="9.5" y1="14" x2="5.5" y2="16.5" />
      <line x1="14.5" y1="14" x2="18.5" y2="16.5" />
    </svg>
  );
}

function StateIcon() {
  /* stack of transformer layers with a probe arrow tapping the middle one —
     evokes activation steering / layer-level state manipulation. */
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <line x1="6" y1="6" x2="20" y2="6" />
      <line x1="6" y1="12" x2="20" y2="12" />
      <line x1="6" y1="18" x2="20" y2="18" />
      <line x1="2" y1="12" x2="6" y2="12" />
      <polyline points="4 10 6 12 4 14" />
    </svg>
  );
}

function OutputIcon() {
  /* mirror of InputIcon: tray on the left, arrow leaving toward the right. */
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M9 3H5a2 2 0 00-2 2v14a2 2 0 002 2h4" />
      <polyline points="15 7 20 12 15 17" />
      <line x1="9" y1="12" x2="21" y2="12" />
    </svg>
  );
}

const CATEGORY_ICON: Record<string, () => JSX.Element> = {
  input_control: InputIcon,
  structural_control: StructuralIcon,
  state_control: StateIcon,
  output_control: OutputIcon,
};

function CloseIcon() {
  return (
    <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

/* ── component ──────────────────────────────────────────────────── */

function ControlNodeImpl({ id, data, selected }: NodeProps<ControlNodeData>) {
  const requestDeleteNode = usePipelineStore((s) => s.requestDeleteNode);
  const grouped = usePipelineStore((s) => {
    const g = s.lockedGroups.find((grp) => grp.members.includes(id));
    return Boolean(g && g.members.length >= 2);
  });

  const onClose = (event: MouseEvent) => {
    event.stopPropagation();
    if (data.onClose) data.onClose();
    else requestDeleteNode(id);
  };

  const methodName = data.label || data.method || "unset";

  const displayParams: ControlNodeParam[] = (
    data.params && data.params.length > 0
      ? data.params
      : Object.entries(data.args)
          .filter(([, v]) => v != null)
          .map(([key, val]) => ({ label: key, value: String(val) }))
  ).slice(0, 3);

  const sidebarLabel =
    data.category ? CATEGORY_SIDEBAR_LABEL[data.category] ?? "" : "";

  const IconComponent = data.category ? CATEGORY_ICON[data.category] : null;

  return (
    <div
      className={`ctrl-node${selected ? " selected" : ""}${grouped ? " grouped" : ""}`}
      data-category={data.category ?? "neutral"}
    >
      <Handle type={grouped ? "target" : "source"} position={Position.Left} id="left" />
      <Handle type={grouped ? "target" : "source"} position={Position.Right} id="right" />
      <Handle type={grouped ? "target" : "source"} position={Position.Top} id="top" />
      <Handle type={grouped ? "target" : "source"} position={Position.Bottom} id="bottom" />

      <button
        type="button"
        className="ctrl-close"
        onClick={onClose}
        aria-label={`Remove ${methodName}`}
        title="close"
      >
        <CloseIcon />
      </button>

      <div className="ctrl-face">
        {/* coloured sidebar */}
        <div className="ctrl-sidebar">
          {IconComponent ? (
            <span className="ctrl-sidebar-icon">
              <IconComponent />
            </span>
          ) : null}
          <div className="ctrl-sidebar-spacer" />
          {sidebarLabel ? (
            <span className="ctrl-sidebar-label">{sidebarLabel}</span>
          ) : null}
        </div>

        {/* content area */}
        <div className="ctrl-content">
          <div className="ctrl-method-name" title={methodName}>
            {methodName}
          </div>

          <div className="ctrl-rows">
            {displayParams.length === 0 ? (
              <div className="ctrl-empty">no parameters set</div>
            ) : (
              displayParams.map((p, idx) => (
                <div key={`${p.label}-${idx}`} className="ctrl-row">
                  <span className="ctrl-row-key">{p.label}</span>
                  <span className="ctrl-row-value" title={p.value}>
                    {p.value}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export const ControlNode = memo(ControlNodeImpl);
