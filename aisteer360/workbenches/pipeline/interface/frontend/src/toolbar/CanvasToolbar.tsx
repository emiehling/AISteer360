import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { usePipelineStore } from "../store/pipelineStore";
import type { ControlCategory } from "../types";

const DEFAULT_OFFSET_RIGHT = 12;
const DEFAULT_OFFSET_TOP = 12;
const EDGE_PADDING = 6;

const CONTROL_CATEGORIES: { value: ControlCategory; label: string }[] = [
  { value: "input_control", label: "Input" },
  { value: "structural_control", label: "Structural" },
  { value: "state_control", label: "State" },
  { value: "output_control", label: "Output" },
];

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

function ChevronRightIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points="9 6 15 12 9 18" />
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

  const [submenuOpen, setSubmenuOpen] = useState(false);
  const submenuRef = useRef<HTMLDivElement | null>(null);
  const submenuButtonRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!submenuOpen) return;
    const onDown = (event: globalThis.MouseEvent) => {
      const target = event.target as Node | null;
      if (submenuRef.current?.contains(target as Node)) return;
      if (submenuButtonRef.current?.contains(target as Node)) return;
      setSubmenuOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [submenuOpen]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLElement) {
        const tag = e.target.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        if (e.target.isContentEditable) return;
      }
      if (e.key === "Escape") {
        if (placement) cancelPlacement();
        if (submenuOpen) setSubmenuOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [placement, cancelPlacement, submenuOpen]);

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
  const placementCategory = placement?.kind === "control" ? placement.category : null;

  const onSelectClick = () => {
    if (placement) cancelPlacement();
    setSubmenuOpen(false);
  };

  const onNewControlClick = () => {
    setSubmenuOpen((v) => !v);
  };

  const onPickCategory = (category: ControlCategory) => {
    setSubmenuOpen(false);
    startPlacement({ kind: "control", category });
  };

  const onDatasetClick = () => {
    setSubmenuOpen(false);
    if (placementKind === "dataset") {
      cancelPlacement();
    } else {
      startPlacement({ kind: "dataset" });
    }
  };

  const onModelClick = () => {
    setSubmenuOpen(false);
    if (placementKind === "model") {
      cancelPlacement();
    } else {
      startPlacement({ kind: "model" });
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
            ref={submenuButtonRef}
            type="button"
            className={`toolbar-btn${placementKind === "control" ? " active" : ""}${submenuOpen ? " expanded" : ""}`}
            title="New control"
            aria-label="New control"
            aria-haspopup="menu"
            aria-expanded={submenuOpen}
            onClick={onNewControlClick}
          >
            <ControlIcon />
          </button>
          <button
            type="button"
            className={`toolbar-btn${placementKind === "dataset" ? " active" : ""}`}
            title="New dataset"
            aria-label="New dataset"
            aria-pressed={placementKind === "dataset"}
            onClick={onDatasetClick}
          >
            <DocumentIcon />
          </button>
          <button
            type="button"
            className={`toolbar-btn${placementKind === "model" ? " active" : ""}`}
            title="New model"
            aria-label="New model"
            aria-pressed={placementKind === "model"}
            onClick={onModelClick}
          >
            <ChipIcon />
          </button>

          {submenuOpen && (
            <div
              ref={submenuRef}
              className="toolbar-submenu"
              role="menu"
              aria-label="Pick control category"
            >
              {CONTROL_CATEGORIES.map((c) => (
                <button
                  key={c.value}
                  type="button"
                  role="menuitem"
                  className={`toolbar-submenu-item${placementCategory === c.value ? " active" : ""}`}
                  data-category={c.value}
                  onClick={() => onPickCategory(c.value)}
                >
                  <span className="toolbar-submenu-dot" aria-hidden />
                  <span className="toolbar-submenu-label">{c.label}</span>
                  <ChevronRightIcon />
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
