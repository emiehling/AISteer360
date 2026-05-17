import { memo } from "react";
import { EdgeLabelRenderer, getSmoothStepPath, type EdgeProps } from "reactflow";

const CHEVRON_HALF = 5;

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
  } = props;

  const [path, labelX, labelY] = getSmoothStepPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    borderRadius: 8,
  });

  const dx = targetX - sourceX;
  const dy = targetY - sourceY;
  const angle = (Math.atan2(dy, dx) * 180) / Math.PI;

  return (
    <>
      <path
        id={id}
        d={path}
        style={style}
        fill="none"
        className={`pipeline-edge${selected ? " selected" : ""}`}
      />
      <EdgeLabelRenderer>
        <div
          className="pipeline-edge-chevron"
          style={{
            transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px) rotate(${angle}deg)`,
          }}
        >
          <svg width={CHEVRON_HALF * 2} height={CHEVRON_HALF * 2} viewBox={`0 0 ${CHEVRON_HALF * 2} ${CHEVRON_HALF * 2}`}>
            <path
              d={`M 0 0 L ${CHEVRON_HALF * 2} ${CHEVRON_HALF} L 0 ${CHEVRON_HALF * 2} Z`}
              fill="currentColor"
            />
          </svg>
        </div>
      </EdgeLabelRenderer>
    </>
  );
}

export const PipelineEdge = memo(PipelineEdgeImpl);
