import { memo, type MouseEvent } from "react";
import { Handle, Position, type NodeProps } from "reactflow";
import type { DatasetNodeData } from "../types";
import { usePipelineStore } from "../store/pipelineStore";

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

function StackedDocIcon() {
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
        transform="translate(4 0)"
      />
      <path
        d="M9 4 H21 L29 12 V36 a1 1 0 0 1 -1 1 H9 a1 1 0 0 1 -1 -1 V5 a1 1 0 0 1 1 -1 Z"
        fill="var(--node-face-bg)"
        transform="translate(2 2)"
      />
      <path
        d="M9 4 H21 L29 12 V36 a1 1 0 0 1 -1 1 H9 a1 1 0 0 1 -1 -1 V5 a1 1 0 0 1 1 -1 Z"
        fill="var(--node-face-bg)"
      />
      <path d="M21 4 V12 H29" />
    </svg>
  );
}

function DatasetNodeImpl({ id, data, selected }: NodeProps<DatasetNodeData>) {
  const requestDeleteNode = usePipelineStore((s) => s.requestDeleteNode);

  const onClose = (event: MouseEvent) => {
    event.stopPropagation();
    if (data.onClose) data.onClose();
    else requestDeleteNode(id);
  };

  return (
    <div className={`dataset-node${selected ? " selected" : ""}`}>
      <Handle type="source" position={Position.Left} id="left" />
      <Handle type="source" position={Position.Right} id="right" />
      <Handle type="source" position={Position.Top} id="top" />
      <Handle type="source" position={Position.Bottom} id="bottom" />

      <button
        type="button"
        className="dataset-close"
        onClick={onClose}
        aria-label={`Remove ${data.name}`}
        title="close"
      >
        <CloseIcon />
      </button>

      <div className="dataset-icon-wrap">
        <StackedDocIcon />
      </div>
      <div className="dataset-name" title={data.name}>
        {data.name || "dataset"}
      </div>
    </div>
  );
}

export const DatasetNode = memo(DatasetNodeImpl);
