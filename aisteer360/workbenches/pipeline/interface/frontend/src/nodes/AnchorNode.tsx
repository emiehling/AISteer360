import { memo } from "react";
import { Handle, Position, type NodeProps } from "reactflow";

export type AnchorVariant = "prompt" | "response";

interface AnchorNodeData {
  variant: AnchorVariant;
}

function AnchorNodeImpl({ data }: NodeProps<AnchorNodeData>) {
  const isPrompt = data.variant === "prompt";
  return (
    <div className="anchor-node" data-variant={data.variant}>
      <div className="anchor-label">{isPrompt ? "prompt" : "response"}</div>
      {isPrompt ? (
        <Handle
          type="source"
          position={Position.Right}
          id="out"
          className="port port-source"
        />
      ) : (
        <Handle
          type="target"
          position={Position.Left}
          id="in"
          className="port port-target"
        />
      )}
    </div>
  );
}

export const AnchorNode = memo(AnchorNodeImpl);
