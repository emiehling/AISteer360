import { memo } from "react";
import { Handle, Position, type NodeProps } from "reactflow";

interface ModelNodeData {
  modelNameOrPath: string;
}

function ModelNodeImpl({ data }: NodeProps<ModelNodeData>) {
  const display = data.modelNameOrPath || "‹ select model ›";
  return (
    <div className="model-node">
      <div className="model-header">TARGET MODEL</div>
      <div className="model-body" title={data.modelNameOrPath || "no model selected"}>
        {display}
      </div>
      <Handle
        type="target"
        position={Position.Left}
        id="input"
        className="port port-target"
      />
      <Handle
        type="target"
        position={Position.Top}
        id="structural"
        className="port port-target"
      />
      <Handle
        type="target"
        position={Position.Bottom}
        id="state"
        className="port port-target"
      />
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        className="port port-source"
      />
    </div>
  );
}

export const ModelNode = memo(ModelNodeImpl);
