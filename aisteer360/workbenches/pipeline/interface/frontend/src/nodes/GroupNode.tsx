import { memo } from "react";
import { Handle, Position, type NodeProps } from "reactflow";

export interface GroupNodeData {
  category: string | null;
  width: number;
  height: number;
}

function GroupNodeImpl({ data }: NodeProps<GroupNodeData>) {
  return (
    <div
      className="group-node"
      data-category={data.category ?? "neutral"}
      style={{ width: data.width, height: data.height }}
    >
      <Handle type="source" position={Position.Left} id="left" />
      <Handle type="source" position={Position.Right} id="right" />
      <Handle type="source" position={Position.Top} id="top" />
      <Handle type="source" position={Position.Bottom} id="bottom" />
    </div>
  );
}

export const GroupNode = memo(GroupNodeImpl);
