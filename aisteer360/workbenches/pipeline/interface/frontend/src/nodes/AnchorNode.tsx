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
  width: 10,
  height: 10,
  border: "1.5px solid var(--text-dim)",
  background: "transparent",
  flexShrink: 0,
  opacity: 1,
};

function AnchorNodeImpl({ data }: NodeProps<AnchorNodeData>) {
  const isPrompt = data.variant === "prompt";

  if (isPrompt) {
    return (
      <div className="anchor-wrap">
        <div className="anchor-body">
          <span className="anchor-label">prompt</span>
        </div>
        <div className="anchor-stem" aria-hidden />
        <Handle
          type="source"
          position={Position.Right}
          id="out"
          className="anchor-handle anchor-handle-prompt"
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
        className="anchor-handle anchor-handle-response"
        style={{ ...handleStyle, borderRadius: 1 }}
      />
      <div className="anchor-stem" aria-hidden />
      <div className="anchor-body">
        <span className="anchor-label">response</span>
      </div>
    </div>
  );
}

export const AnchorNode = memo(AnchorNodeImpl);
