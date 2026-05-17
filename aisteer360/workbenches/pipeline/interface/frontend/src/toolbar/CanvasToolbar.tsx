import { useEffect } from "react";
import { usePipelineStore } from "../store/pipelineStore";
import type { ToolMode } from "../types";

const TOOLS: { mode: ToolMode; label: string; key: string; icon: JSX.Element }[] = [
  {
    mode: "select",
    label: "Select",
    key: "V",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M5 3 L19 12 L12 13 L9 20 Z" />
      </svg>
    ),
  },
  {
    mode: "connect",
    label: "Connect",
    key: "C",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="6" cy="12" r="2.5" />
        <circle cx="18" cy="12" r="2.5" />
        <path d="M8.5 12 H15.5" />
      </svg>
    ),
  },
  {
    mode: "erase",
    label: "Erase",
    key: "X",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M16 3 L21 8 L9 20 L4 20 L4 15 Z" />
        <path d="M11 6 L18 13" />
      </svg>
    ),
  },
];

export function CanvasToolbar() {
  const activeTool = usePipelineStore((s) => s.activeTool);
  const setActiveTool = usePipelineStore((s) => s.setActiveTool);
  const resetCanvas = usePipelineStore((s) => s.resetCanvas);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLElement) {
        const tag = e.target.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        if (e.target.isContentEditable) return;
      }
      const k = e.key.toLowerCase();
      if (k === "v") setActiveTool("select");
      else if (k === "c") setActiveTool("connect");
      else if (k === "x") setActiveTool("erase");
      else if (e.key === "Escape") setActiveTool("select");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setActiveTool]);

  return (
    <div className="canvas-toolbar" role="toolbar" aria-label="Canvas tools">
      {TOOLS.map((t) => (
        <button
          key={t.mode}
          type="button"
          className={`toolbar-btn${activeTool === t.mode ? " active" : ""}`}
          title={`${t.label} (${t.key})`}
          aria-label={t.label}
          aria-pressed={activeTool === t.mode}
          onClick={() => setActiveTool(t.mode)}
        >
          {t.icon}
        </button>
      ))}
      <div className="toolbar-divider" />
      <button
        type="button"
        className="toolbar-btn"
        title="Reset canvas"
        aria-label="Reset canvas"
        onClick={() => {
          if (confirm("Remove all controls and connections from the canvas?")) {
            resetCanvas();
          }
        }}
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M3 12 a9 9 0 1 0 3-6.7" />
          <path d="M3 4 V10 H9" />
        </svg>
      </button>
    </div>
  );
}
