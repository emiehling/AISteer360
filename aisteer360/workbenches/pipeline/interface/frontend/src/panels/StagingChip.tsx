import type { DragEvent } from "react";
import type { ControlCategory, ControlNodeParam } from "../types";

const CATEGORY_FOOTER_LABEL: Record<string, string> = {
  input_control: "input control",
  structural_control: "structural control",
  state_control: "state control",
  output_control: "output control",
  neutral: "control",
};

interface StagingChipProps {
  category: ControlCategory | "neutral";
  title: string;
  params: ControlNodeParam[];
  draggable: boolean;
  onDragStart?: (event: DragEvent<HTMLDivElement>) => void;
  onDragEnd?: (event: DragEvent<HTMLDivElement>) => void;
}

export function StagingChip({
  category,
  title,
  params,
  draggable,
  onDragStart,
  onDragEnd,
}: StagingChipProps) {
  return (
    <div
      className={`ctrl-node staging-chip${draggable ? " draggable" : ""}`}
      data-category={category}
      draggable={draggable}
      onDragStart={onDragStart}
      onDragEnd={onDragEnd}
      title={draggable ? "drag onto canvas" : undefined}
    >
      <div className="ctrl-bar">
        <span className="ctrl-bar-title" title={title}>
          {title || " "}
        </span>
        <div className="ctrl-bar-spacer" />
      </div>
      <div className="ctrl-body">
        {params.length === 0 ? (
          <div className="ctrl-empty">no parameters set</div>
        ) : (
          params.map((p, idx) => (
            <div key={`${p.label}-${idx}`} className="ctrl-card">
              <div className="ctrl-card-label">{p.label}</div>
              <div className="ctrl-card-value" title={p.value}>
                {p.value}
              </div>
            </div>
          ))
        )}
      </div>
      <div className="ctrl-footer">{CATEGORY_FOOTER_LABEL[category]}</div>
    </div>
  );
}
