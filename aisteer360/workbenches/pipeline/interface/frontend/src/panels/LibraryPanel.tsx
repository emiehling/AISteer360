import type { DragEvent } from "react";
import type { ControlCategory } from "../types";
import { usePipelineStore } from "../store/pipelineStore";

const CATEGORY_ORDER: { category: ControlCategory; label: string }[] = [
  { category: "input_control", label: "Input" },
  { category: "structural_control", label: "Structural" },
  { category: "state_control", label: "State" },
  { category: "output_control", label: "Output" },
];

export const DRAG_MIME = "application/x-aisteer-control";

export function LibraryPanel() {
  const methods = usePipelineStore((s) => s.methods);

  const onDragStart = (
    event: DragEvent<HTMLDivElement>,
    category: ControlCategory,
    method: string,
  ) => {
    event.dataTransfer.setData(DRAG_MIME, JSON.stringify({ category, method }));
    event.dataTransfer.effectAllowed = "copy";
  };

  return (
    <div className="library-row" role="region" aria-label="Steering controls library">
      {CATEGORY_ORDER.map((col) => {
        const items = methods.filter((m) => m.category === col.category);
        return (
          <div className="library-column" data-category={col.category} key={col.category}>
            <div className="library-header">
              <span className="library-dot" />
              <span className="library-label">{col.label}</span>
            </div>
            <div className="library-body">
              {items.length === 0 ? (
                <div className="library-empty">no methods registered</div>
              ) : (
                items.map((m) => (
                  <div
                    className="library-pill"
                    key={m.method}
                    title={m.method}
                    draggable
                    onDragStart={(e) => onDragStart(e, m.category, m.method)}
                  >
                    {m.method}
                  </div>
                ))
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
