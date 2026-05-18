import { memo } from "react";
import { getSmoothStepPath, type EdgeProps } from "reactflow";

function PipelineEdgeImpl(props: EdgeProps) {
  const {
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    selected,
    style,
    markerEnd,
  } = props;

  const [path] = getSmoothStepPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    borderRadius: 5,
  });

  return (
    <path
      id={id}
      d={path}
      style={style}
      fill="none"
      markerEnd={markerEnd}
      className={`pipeline-edge${selected ? " selected" : ""}`}
    />
  );
}

export const PipelineEdge = memo(PipelineEdgeImpl);
