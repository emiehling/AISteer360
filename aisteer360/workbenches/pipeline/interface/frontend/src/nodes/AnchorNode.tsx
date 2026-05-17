import { memo } from "react";
import { Handle, Position, type NodeProps } from "reactflow";

export type AnchorVariant = "prompt" | "response";

interface AnchorNodeData {
  variant: AnchorVariant;
}

const handleStyle = {
  position: "relative" as const,
  transform: "none",
  top: "auto",
  right: "auto",
  left: "auto",
  bottom: "auto",
  width: 8,
  height: 8,
  border: "1.5px solid var(--text-dim)",
  background: "transparent",
  flexShrink: 0,
};

function AnchorNodeImpl({ data }: NodeProps<AnchorNodeData>) {
  const isPrompt = data.variant === "prompt";

  const body = (
    <div className="anchor-body">
      <span className="anchor-label">{isPrompt ? "prompt" : "response"}</span>
    </div>
  );

  if (isPrompt) {
    return (
      <div className="anchor-wrap">
        {body}
        <div className="anchor-stem" aria-hidden />
        <Handle
          type="source"
          position={Position.Right}
          id="out"
          style={{ ...handleStyle, borderRadius: "50%" }}
        />
      </div>
    );
  }

  return (
    <div className="anchor-wrap">
      <Handle
        type="target"
        position={Position.Left}
        id="in"
        style={{ ...handleStyle, borderRadius: 1 }}
      />
      <div className="anchor-stem" aria-hidden />
      {body}
    </div>
  );
}

export const AnchorNode = memo(AnchorNodeImpl);
