import { memo, type MouseEvent } from "react";
import { Handle, Position, type NodeProps } from "reactflow";
import type { ModelNodeData } from "../types";
import { usePipelineStore } from "../store/pipelineStore";

/* ── icons ───────────────────────────────────────────────────────── */

function BoxIcon() {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M21 8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z" />
      <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
      <line x1="12" y1="22.08" x2="12" y2="12" />
    </svg>
  );
}

/* gold cube with three faceted faces (top / front-left / front-right) and a
   thin outline; rendered when the model node is the steering target. */
function TargetBoxIcon() {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 24 24"
      aria-hidden
    >
      <polygon points="12,2 21,7 12,12 3,7" fill="#f6d76b" />
      <polygon points="3,7 12,12 12,22 3,17" fill="#b8902f" />
      <polygon points="21,7 12,12 12,22 21,17" fill="#d9ad3f" />
      <path
        d="M21 8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z"
        fill="none"
        stroke="#5a4612"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <polyline
        points="3.27 6.96 12 12.01 20.73 6.96"
        fill="none"
        stroke="#5a4612"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <line
        x1="12"
        y1="22.08"
        x2="12"
        y2="12"
        stroke="#5a4612"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
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

/* ── helpers ──────────────────────────────────────────────────────── */

/** Strip the org prefix: "ibm-granite/granite-4.0-h-micro" → "granite-4.0-h-micro" */
function shortModelId(id: string): string {
  const idx = id.lastIndexOf("/");
  return idx >= 0 ? id.slice(idx + 1) : id;
}

/* ── component ───────────────────────────────────────────────────── */

function ModelNodeImpl({ id, data, selected }: NodeProps<ModelNodeData>) {
  const requestDeleteNode = usePipelineStore((s) => s.requestDeleteNode);
  const isTarget = usePipelineStore((s) => s.targetModelNodeId === id);

  const display = data.modelId ? shortModelId(data.modelId) : "unset";
  const loaded = data.params.length > 0;

  const onClose = (event: MouseEvent) => {
    event.stopPropagation();
    requestDeleteNode(id);
  };

  return (
    <div className={`model-wrap${loaded ? " loaded" : ""}${selected ? " selected" : ""}`}>
      <Handle type="source" position={Position.Left} id="left" />
      <Handle type="source" position={Position.Right} id="right" />
      <Handle type="source" position={Position.Top} id="top" />
      <Handle type="source" position={Position.Bottom} id="bottom" />

      <button
        type="button"
        className="model-close"
        onClick={onClose}
        aria-label="Remove model"
        title="close"
      >
        <CloseIcon />
      </button>

      {/* offset shadow */}
      <div className="model-shadow" />

      {/* face */}
      <div className="model-face">
        {/* ── sidebar ── */}
        <div className="model-sidebar">
          <span className={`model-sidebar-icon${isTarget ? " target" : ""}`}>
            {isTarget ? <TargetBoxIcon /> : <BoxIcon />}
          </span>
          <div className="model-sidebar-spacer" />
          <span className="model-sidebar-label">model</span>
        </div>

        {/* ── content ── */}
        <div className="model-content">
          <div className="model-head">
            <span className="model-id" title={data.modelId || "no model selected"}>
              {display}
            </span>
          </div>

          {data.params.length > 0 ? (
            <div className="model-rows">
              {data.params.map((p, idx) => (
                <div key={`${p.label}-${idx}`} className="model-row">
                  <span className="model-row-key">{p.label}</span>
                  <span className="model-row-value" title={p.value}>
                    {p.value}
                  </span>
                </div>
              ))}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

export const ModelNode = memo(ModelNodeImpl);