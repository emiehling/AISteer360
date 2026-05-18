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

function DatasetIcon() {
  return (
    <svg
      width="48"
      height="56"
      viewBox="0 0 48 56"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      aria-hidden
    >
      <ellipse cx="24" cy="10" rx="18" ry="6" fill="var(--dataset-disk-fill)" />
      <path d="M6 10 V22" />
      <path d="M42 10 V22" />
      <ellipse cx="24" cy="22" rx="18" ry="6" fill="var(--dataset-disk-fill)" />
      <path d="M6 22 V34" />
      <path d="M42 22 V34" />
      <ellipse cx="24" cy="34" rx="18" ry="6" fill="var(--dataset-disk-fill)" />
      <path d="M6 34 V46" />
      <path d="M42 34 V46" />
      <ellipse cx="24" cy="46" rx="18" ry="6" fill="var(--dataset-disk-fill)" />
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

      <div className="dataset-bar">
        <span className="dataset-bar-title" title={data.name}>
          {data.name || "dataset"}
        </span>
        <div className="dataset-bar-spacer" />
        <button
          type="button"
          className="dataset-bar-btn"
          onClick={onClose}
          aria-label={`Remove ${data.name}`}
          title="Close"
        >
          <CloseIcon />
        </button>
      </div>

      <div className="dataset-body">
        <DatasetIcon />
        <div className="dataset-meta">
          <div className="dataset-name" title={data.name}>
            {data.name || "dataset"}
          </div>
          {data.rowCount != null ? (
            <div className="dataset-rowcount">{data.rowCount} rows</div>
          ) : (
            <div className="dataset-rowcount muted">unbound</div>
          )}
        </div>
      </div>

      <div className="dataset-footer">dataset</div>
    </div>
  );
}

export const DatasetNode = memo(DatasetNodeImpl);
