import { memo, type MouseEvent } from "react";
import { Handle, Position, type NodeProps } from "reactflow";
import type { ControlNodeData } from "../types";
import { usePipelineStore } from "../store/pipelineStore";

const CATEGORY_LABEL: Record<string, string> = {
  input_control: "INPUT",
  structural_control: "STRUCTURAL",
  state_control: "STATE",
  output_control: "OUTPUT",
};

interface PortLayout {
  topSource?: boolean;
  bottomSource?: boolean;
  leftTarget?: boolean;
  rightSource?: boolean;
}

function portsFor(category: string): PortLayout {
  switch (category) {
    case "input_control":
      return { leftTarget: true, rightSource: true };
    case "structural_control":
      return { bottomSource: true };
    case "state_control":
      return { topSource: true };
    case "output_control":
      return { leftTarget: true, rightSource: true };
    default:
      return {};
  }
}

function ControlNodeImpl({ id, data, selected }: NodeProps<ControlNodeData>) {
  const removeNode = usePipelineStore((s) => s.removeNode);
  const ports = portsFor(data.category);
  const onDelete = (e: MouseEvent) => {
    e.stopPropagation();
    removeNode(id);
  };
  return (
    <div
      className={`control-node${selected ? " selected" : ""}`}
      data-category={data.category}
    >
      <div className="control-header">{CATEGORY_LABEL[data.category] ?? data.category}</div>
      <div className="control-body" title={data.method}>
        {data.method}
      </div>
      <button
        type="button"
        className="control-delete"
        onClick={onDelete}
        aria-label={`Remove ${data.method}`}
        title="Remove"
      >
        ×
      </button>
      {ports.leftTarget && (
        <Handle type="target" position={Position.Left} id="in" className="port port-target" />
      )}
      {ports.rightSource && (
        <Handle type="source" position={Position.Right} id="out" className="port port-source" />
      )}
      {ports.topSource && (
        <Handle type="source" position={Position.Top} id="out" className="port port-source" />
      )}
      {ports.bottomSource && (
        <Handle
          type="source"
          position={Position.Bottom}
          id="out"
          className="port port-source"
        />
      )}
    </div>
  );
}

export const ControlNode = memo(ControlNodeImpl);
