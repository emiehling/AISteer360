import {
  useEffect,
  useRef,
  type CSSProperties,
  type DragEvent as ReactDragEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { usePipelineStore } from "../store/pipelineStore";
import { DRAG_MIME_BLANK } from "../panels/LibraryPanel";

type ToolbarKind = "control" | "dataset" | "model" | "steering_vector" | "multiplexer";

function makeBlankDragStart(kind: ToolbarKind) {
  return (event: ReactDragEvent<HTMLButtonElement>) => {
    event.dataTransfer.setData(DRAG_MIME_BLANK, kind);
    event.dataTransfer.effectAllowed = "copy";
    // cancel any active click-to-place mode so the user only acts on the drag.
    usePipelineStore.getState().cancelPlacement();
  };
}

const DEFAULT_OFFSET_RIGHT = 12;
const DEFAULT_OFFSET_TOP = 12;
const EDGE_PADDING = 6;

function GripIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <circle cx="9" cy="6" r="1.4" />
      <circle cx="15" cy="6" r="1.4" />
      <circle cx="9" cy="12" r="1.4" />
      <circle cx="15" cy="12" r="1.4" />
      <circle cx="9" cy="18" r="1.4" />
      <circle cx="15" cy="18" r="1.4" />
    </svg>
  );
}

function LockIcon({ locked }: { locked: boolean }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="5" y="11" width="14" height="9" rx="1.5" />
      {locked ? (
        <path d="M8 11 V8 a4 4 0 0 1 8 0 V11" />
      ) : (
        <path d="M8 11 V8 a4 4 0 0 1 7.5 -1.8" />
      )}
    </svg>
  );
}

function ChevronIcon({ up }: { up: boolean }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points={up ? "6 15 12 9 18 15" : "6 9 12 15 18 9"} />
    </svg>
  );
}

function CursorIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M5 3 L19 12 L12 13 L9 20 Z" />
    </svg>
  );
}

function ControlIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="4" y="6" width="16" height="12" rx="1.5" />
      <path d="M4 10 H20" />
      <path d="M9 14 H15" />
    </svg>
  );
}

function DocumentIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M7 3 H14 L19 8 V20 a1 1 0 0 1 -1 1 H7 a1 1 0 0 1 -1 -1 V4 a1 1 0 0 1 1 -1 Z" />
      <path d="M14 3 V8 H19" />
      <path d="M9 13 H16" />
      <path d="M9 17 H14" />
    </svg>
  );
}

function ChipIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="6" y="6" width="12" height="12" rx="1.5" />
      <rect x="9" y="9" width="6" height="6" />
      <path d="M3 9 H6" />
      <path d="M3 15 H6" />
      <path d="M18 9 H21" />
      <path d="M18 15 H21" />
      <path d="M9 3 V6" />
      <path d="M15 3 V6" />
      <path d="M9 18 V21" />
      <path d="M15 18 V21" />
    </svg>
  );
}

function VectorIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <line x1="5" y1="19" x2="19" y2="5" />
      <polyline points="12 5 19 5 19 12" />
      <circle cx="5" cy="19" r="1.4" fill="currentColor" stroke="none" />
    </svg>
  );
}

function MultiplexerIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="10" y="4" width="4" height="16" rx="0.5" />
      <line x1="3" y1="7" x2="10" y2="7" />
      <line x1="3" y1="12" x2="10" y2="12" />
      <line x1="3" y1="17" x2="10" y2="17" />
      <line x1="14" y1="12" x2="21" y2="12" />
    </svg>
  );
}

export function CanvasToolbar() {
  const placement = usePipelineStore((s) => s.placement);
  const startPlacement = usePipelineStore((s) => s.startPlacement);
  const cancelPlacement = usePipelineStore((s) => s.cancelPlacement);
  const toolbarPosition = usePipelineStore((s) => s.toolbarPosition);
  const toolbarLocked = usePipelineStore((s) => s.toolbarLocked);
  const toolbarMinimized = usePipelineStore((s) => s.toolbarMinimized);
  const setToolbarPosition = usePipelineStore((s) => s.setToolbarPosition);
  const toggleToolbarLocked = usePipelineStore((s) => s.toggleToolbarLocked);
  const toggleToolbarMinimized = usePipelineStore((s) => s.toggleToolbarMinimized);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const dragStateRef = useRef<{
    pointerId: number;
    offsetX: number;
    offsetY: number;
    parentRect: DOMRect;
    selfWidth: number;
    selfHeight: number;
  } | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLElement) {
        const tag = e.target.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        if (e.target.isContentEditable) return;
      }
      if (e.key === "Escape" && placement) {
        cancelPlacement();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [placement, cancelPlacement]);

  const onGripPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (toolbarLocked) return;
    const container = containerRef.current;
    const parent = container?.parentElement;
    if (!container || !parent) return;
    event.preventDefault();
    event.stopPropagation();
    const rect = container.getBoundingClientRect();
    const parentRect = parent.getBoundingClientRect();
    dragStateRef.current = {
      pointerId: event.pointerId,
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top,
      parentRect,
      selfWidth: rect.width,
      selfHeight: rect.height,
    };
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
  };

  const onGripPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragStateRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const { parentRect, selfWidth, selfHeight, offsetX, offsetY } = drag;
    const rawX = event.clientX - parentRect.left - offsetX;
    const rawY = event.clientY - parentRect.top - offsetY;
    const maxX = Math.max(EDGE_PADDING, parentRect.width - selfWidth - EDGE_PADDING);
    const maxY = Math.max(EDGE_PADDING, parentRect.height - selfHeight - EDGE_PADDING);
    const x = Math.min(maxX, Math.max(EDGE_PADDING, rawX));
    const y = Math.min(maxY, Math.max(EDGE_PADDING, rawY));
    setToolbarPosition({ x, y });
  };

  const onGripPointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragStateRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    dragStateRef.current = null;
    try {
      (event.currentTarget as HTMLElement).releasePointerCapture(event.pointerId);
    } catch {
      // ignore — capture may already be released
    }
  };

  const positioned: CSSProperties = toolbarPosition
    ? { left: toolbarPosition.x, top: toolbarPosition.y, right: "auto" }
    : { right: DEFAULT_OFFSET_RIGHT, top: DEFAULT_OFFSET_TOP };

  const placementKind = placement?.kind ?? null;

  const onSelectClick = () => {
    if (placement) cancelPlacement();
  };

  const onNewControlClick = () => {
    if (placementKind === "control") {
      cancelPlacement();
    } else {
      startPlacement({ kind: "control" });
    }
  };

  const onDatasetClick = () => {
    if (placementKind === "dataset") {
      cancelPlacement();
    } else {
      startPlacement({ kind: "dataset" });
    }
  };

  const onModelClick = () => {
    if (placementKind === "model") {
      cancelPlacement();
    } else {
      startPlacement({ kind: "model" });
    }
  };

  const onSteeringVectorClick = () => {
    if (placementKind === "steering_vector") {
      cancelPlacement();
    } else {
      startPlacement({ kind: "steering_vector" });
    }
  };

  const onMultiplexerClick = () => {
    if (placementKind === "multiplexer") {
      cancelPlacement();
    } else {
      startPlacement({ kind: "multiplexer" });
    }
  };

  return (
    <div
      ref={containerRef}
      className={`canvas-toolbar${toolbarMinimized ? " minimized" : ""}${toolbarLocked ? " locked" : ""}`}
      role="toolbar"
      aria-label="Canvas tools"
      style={positioned}
    >
      <div className="toolbar-header">
        <div
          className={`toolbar-grip${toolbarLocked ? " locked" : ""}`}
          onPointerDown={onGripPointerDown}
          onPointerMove={onGripPointerMove}
          onPointerUp={onGripPointerUp}
          onPointerCancel={onGripPointerUp}
          title={toolbarLocked ? "locked — unlock to drag" : "drag to move"}
          aria-hidden
        >
          <GripIcon />
        </div>
        <button
          type="button"
          className={`toolbar-header-btn${toolbarLocked ? " active" : ""}`}
          onClick={toggleToolbarLocked}
          title={toolbarLocked ? "Unlock toolbar" : "Lock toolbar"}
          aria-label={toolbarLocked ? "Unlock toolbar" : "Lock toolbar"}
          aria-pressed={toolbarLocked}
        >
          <LockIcon locked={toolbarLocked} />
        </button>
        <button
          type="button"
          className="toolbar-header-btn"
          onClick={toggleToolbarMinimized}
          title={toolbarMinimized ? "Expand toolbar" : "Minimize toolbar"}
          aria-label={toolbarMinimized ? "Expand toolbar" : "Minimize toolbar"}
          aria-pressed={toolbarMinimized}
        >
          <ChevronIcon up={toolbarMinimized} />
        </button>
      </div>

      {!toolbarMinimized && (
        <div className="toolbar-body">
          <div className="toolbar-row">
            <button
              type="button"
              className={`toolbar-btn${placement === null ? " active" : ""}`}
              title="Select (Esc cancels placement)"
              aria-label="Select"
              aria-pressed={placement === null}
              onClick={onSelectClick}
            >
              <CursorIcon />
            </button>
            <button
              type="button"
              className={`toolbar-btn${placementKind === "control" ? " active" : ""}`}
              title="New control (click then click canvas, or drag onto canvas)"
              aria-label="New control"
              aria-pressed={placementKind === "control"}
              onClick={onNewControlClick}
              draggable
              onDragStart={makeBlankDragStart("control")}
            >
              <ControlIcon />
            </button>
          </div>
          <div className="toolbar-row">
            <button
              type="button"
              className={`toolbar-btn${placementKind === "dataset" ? " active" : ""}`}
              title="New dataset (click then click canvas, or drag onto canvas)"
              aria-label="New dataset"
              aria-pressed={placementKind === "dataset"}
              onClick={onDatasetClick}
              draggable
              onDragStart={makeBlankDragStart("dataset")}
            >
              <DocumentIcon />
            </button>
            <button
              type="button"
              className={`toolbar-btn${placementKind === "model" ? " active" : ""}`}
              title="New model (click then click canvas, or drag onto canvas)"
              aria-label="New model"
              aria-pressed={placementKind === "model"}
              onClick={onModelClick}
              draggable
              onDragStart={makeBlankDragStart("model")}
            >
              <ChipIcon />
            </button>
          </div>
          <div className="toolbar-row">
            <button
              type="button"
              className={`toolbar-btn${placementKind === "steering_vector" ? " active" : ""}`}
              title="New steering vector (click then click canvas, or drag onto canvas)"
              aria-label="New steering vector"
              aria-pressed={placementKind === "steering_vector"}
              onClick={onSteeringVectorClick}
              draggable
              onDragStart={makeBlankDragStart("steering_vector")}
            >
              <VectorIcon />
            </button>
            <button
              type="button"
              className={`toolbar-btn${placementKind === "multiplexer" ? " active" : ""}`}
              title="New multiplexer (click then click canvas, or drag onto canvas)"
              aria-label="New multiplexer"
              aria-pressed={placementKind === "multiplexer"}
              onClick={onMultiplexerClick}
              draggable
              onDragStart={makeBlankDragStart("multiplexer")}
            >
              <MultiplexerIcon />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
