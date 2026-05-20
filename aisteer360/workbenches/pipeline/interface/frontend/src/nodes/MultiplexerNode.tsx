import { memo, useMemo, type MouseEvent } from "react";
import { Handle, Position, type NodeProps } from "reactflow";
import { usePipelineStore } from "../store/pipelineStore";

export type MultiplexerOrientation = "vertical" | "horizontal";

export interface MultiplexerNodeData {
  name: string;
  orientation: MultiplexerOrientation;
}

// minimum spacing between adjacent input handles, in px. used to size the bar
// so handles are easy to grab.
const MIN_PORT_SPACING = 20;
// minimum bar long-axis size when there are 0 connected inputs (1 empty port).
const MIN_LONG_AXIS = 80;
// short-axis size of the bar (the thin dimension).
const SHORT_AXIS = 20;

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

function MultiplexerNodeImpl({ id, data, selected }: NodeProps<MultiplexerNodeData>) {
  const requestDeleteNode = usePipelineStore((s) => s.requestDeleteNode);
  const edges = usePipelineStore((s) => s.edges);

  const orientation = data.orientation ?? "vertical";

  // count distinct input handle ids in use on this mux. always render one extra
  // (empty) port so the user has a target for the next connection.
  const usedInputHandles = useMemo(() => {
    const seen = new Set<string>();
    for (const e of edges) {
      if (e.target !== id) continue;
      const h = e.targetHandle ?? "";
      if (h.startsWith("in-")) seen.add(h);
    }
    return seen;
  }, [edges, id]);

  const inputCount = usedInputHandles.size + 1;
  const portIds = useMemo(() => {
    const ids: string[] = [];
    let i = 0;
    while (ids.length < inputCount) {
      const candidate = `in-${i}`;
      if (usedInputHandles.has(candidate) || ids.length === inputCount - 1) {
        ids.push(candidate);
      }
      i += 1;
      if (i > 1000) break;  // safety
    }
    return ids;
  }, [inputCount, usedInputHandles]);

  const onClose = (event: MouseEvent) => {
    event.stopPropagation();
    requestDeleteNode(id);
  };

  const isVertical = orientation === "vertical";
  const longAxis = Math.max(MIN_LONG_AXIS, inputCount * MIN_PORT_SPACING);
  const wrapStyle = isVertical
    ? { width: SHORT_AXIS, height: longAxis }
    : { width: longAxis, height: SHORT_AXIS };

  const inputPosition = isVertical ? Position.Left : Position.Top;
  const outputPosition = isVertical ? Position.Right : Position.Bottom;

  return (
    <div
      className={`mux-node${selected ? " selected" : ""}`}
      data-orientation={orientation}
      style={wrapStyle}
    >
      {portIds.map((portId, idx) => {
        const fraction = (idx + 1) / (inputCount + 1);
        const positionStyle = isVertical
          ? { top: `${fraction * 100}%` }
          : { left: `${fraction * 100}%` };
        const isEmptySlot = !usedInputHandles.has(portId);
        const cls = `mux-input-handle${isEmptySlot ? " mux-input-handle-empty" : ""}`;
        return (
          <Handle
            key={portId}
            id={portId}
            type="target"
            position={inputPosition}
            style={positionStyle}
            className={cls}
          />
        );
      })}

      <Handle
        id="out"
        type="source"
        position={outputPosition}
        className="mux-output-handle"
      />

      <button
        type="button"
        className="mux-close"
        onClick={onClose}
        aria-label={`Remove ${data.name || "multiplexer"}`}
        title="close"
      >
        <CloseIcon />
      </button>
    </div>
  );
}

export const MultiplexerNode = memo(MultiplexerNodeImpl);
