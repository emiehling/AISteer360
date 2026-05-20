import { memo, type MouseEvent } from "react";
import { Handle, Position, type NodeProps } from "reactflow";
import { usePipelineStore } from "../store/pipelineStore";

interface ModelNodeParam {
  label: string;
  value: string;
}

interface ModelNodeData {
  modelId: string;
  loaded: boolean;
  params: ModelNodeParam[];
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

function StarIcon({ filled }: { filled: boolean }) {
  return (
    <svg
      width="11"
      height="11"
      viewBox="0 0 24 24"
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <polygon points="12 2 15 9 22 9.5 17 14.5 18.5 22 12 18 5.5 22 7 14.5 2 9.5 9 9" />
    </svg>
  );
}

function ModelNodeImpl({ id, data, selected }: NodeProps<ModelNodeData>) {
  const requestDeleteNode = usePipelineStore((s) => s.requestDeleteNode);
  const targetModelNodeId = usePipelineStore((s) => s.targetModelNodeId);
  const setTargetModelNodeId = usePipelineStore((s) => s.setTargetModelNodeId);

  const display = data.modelId || "‹ select model ›";
  const isTarget = targetModelNodeId === id;

  const onClose = (event: MouseEvent) => {
    event.stopPropagation();
    requestDeleteNode(id);
  };

  const onToggleTarget = (event: MouseEvent) => {
    event.stopPropagation();
    setTargetModelNodeId(isTarget ? null : id);
  };

  return (
    <div className={`model-wrap${selected ? " selected" : ""}`}>
      <Handle type="source" position={Position.Left} id="left" />
      <Handle type="source" position={Position.Right} id="right" />
      <Handle type="source" position={Position.Top} id="top" />
      <Handle type="source" position={Position.Bottom} id="bottom" />

      <div className="model-face">
        <div className="model-bar">
          <span className="model-bar-title">{isTarget ? "target model" : "model"}</span>
          <span
            className={`model-bar-dot ${data.loaded ? "loaded" : "unloaded"}`}
            aria-label={data.loaded ? "model loaded" : "model not loaded"}
            title={data.loaded ? "model loaded" : "model not loaded"}
          />
          <button
            type="button"
            className={`model-bar-btn model-bar-star${isTarget ? " active" : ""}`}
            onClick={onToggleTarget}
            aria-label={isTarget ? "Unset as target model" : "Set as target model"}
            aria-pressed={isTarget}
            title={isTarget ? "target model" : "set as target"}
          >
            <StarIcon filled={isTarget} />
          </button>
          <button
            type="button"
            className="model-bar-btn model-bar-close"
            onClick={onClose}
            aria-label="Remove model"
            title="close"
          >
            <CloseIcon />
          </button>
        </div>

        <div className="model-id-row" title={data.modelId || "no model selected"}>
          {display}
        </div>

        {data.params.length > 0 ? (
          <div className="model-params">
            {data.params.map((p, idx) => (
              <div key={`${p.label}-${idx}`} className="model-param-row">
                <span className="model-param-label">{p.label}</span>
                <span className="model-param-value" title={p.value}>
                  {p.value}
                </span>
              </div>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

export const ModelNode = memo(ModelNodeImpl);
