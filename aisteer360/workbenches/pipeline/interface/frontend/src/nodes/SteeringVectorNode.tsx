import { memo, type MouseEvent } from "react";
import { Handle, Position, type NodeProps } from "reactflow";
import { usePipelineStore } from "../store/pipelineStore";

interface SteeringVectorNodeData {
  name: string;
  path?: string | null;
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

function VectorArrowIcon() {
  return (
    <svg
      width="36"
      height="42"
      viewBox="0 0 36 42"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinejoin="round"
      aria-hidden
    >
      <path
        d="M9 4 H21 L29 12 V36 a1 1 0 0 1 -1 1 H9 a1 1 0 0 1 -1 -1 V5 a1 1 0 0 1 1 -1 Z"
        fill="var(--node-face-bg)"
      />
      <path d="M21 4 V12 H29" />
      <line
        x1="13"
        y1="28"
        x2="24"
        y2="17"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
      <polyline
        points="18 17 24 17 24 23"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}

function SteeringVectorNodeImpl({ id, data, selected }: NodeProps<SteeringVectorNodeData>) {
  const requestDeleteNode = usePipelineStore((s) => s.requestDeleteNode);

  const onClose = (event: MouseEvent) => {
    event.stopPropagation();
    requestDeleteNode(id);
  };

  return (
    <div className={`steering-vector-node${selected ? " selected" : ""}`}>
      <Handle type="source" position={Position.Left} id="left" />
      <Handle type="source" position={Position.Right} id="right" />
      <Handle type="source" position={Position.Top} id="top" />
      <Handle type="source" position={Position.Bottom} id="bottom" />

      <button
        type="button"
        className="steering-vector-close"
        onClick={onClose}
        aria-label={`Remove ${data.name}`}
        title="close"
      >
        <CloseIcon />
      </button>

      <div className="steering-vector-icon-wrap">
        <VectorArrowIcon />
      </div>
      <div className="steering-vector-name" title={data.name}>
        {data.name || "vector"}
      </div>
    </div>
  );
}

export const SteeringVectorNode = memo(SteeringVectorNodeImpl);
