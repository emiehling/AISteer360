import { useMemo, useState, type DragEvent } from "react";
import type { ControlCategory } from "../types";
import { usePipelineStore } from "../store/pipelineStore";

const CATEGORIES: { value: ControlCategory; label: string }[] = [
  { value: "input_control", label: "Input" },
  { value: "structural_control", label: "Structural" },
  { value: "state_control", label: "State" },
  { value: "output_control", label: "Output" },
];

export const DRAG_MIME = "application/x-aisteer-control";

interface DraggableChipProps {
  category: ControlCategory;
  method: string;
  label: string;
  onAfterDrop: () => void;
}

function DraggableChip({ category, method, label, onAfterDrop }: DraggableChipProps) {
  const onDragStart = (event: DragEvent<HTMLDivElement>) => {
    event.dataTransfer.setData(DRAG_MIME, JSON.stringify({ category, method }));
    event.dataTransfer.effectAllowed = "copy";
  };
  const onDragEnd = (event: DragEvent<HTMLDivElement>) => {
    if (event.dataTransfer.dropEffect !== "none") {
      onAfterDrop();
    }
  };
  return (
    <div
      className="palette-chip"
      data-category={category}
      draggable
      onDragStart={onDragStart}
      onDragEnd={onDragEnd}
      title="drag onto canvas"
    >
      <span className="palette-chip-dot" />
      <span className="palette-chip-cat">
        {CATEGORIES.find((c) => c.value === category)?.label}
      </span>
      <span className="palette-chip-sep">·</span>
      <span className="palette-chip-method">{label}</span>
    </div>
  );
}

function AddNewControlSection() {
  const methods = usePipelineStore((s) => s.methods);
  const [category, setCategory] = useState<ControlCategory | "">("");
  const [method, setMethod] = useState<string>("");

  const methodOptions = useMemo(
    () => (category ? methods.filter((m) => m.category === category) : []),
    [category, methods],
  );

  const reset = () => {
    setCategory("");
    setMethod("");
  };

  const ready = Boolean(category && method);

  return (
    <div className="palette-section">
      <div className="palette-section-head">Add registered control</div>
      <div className="palette-section-body">
        <label className="palette-field">
          <span className="palette-field-label">category</span>
          <select
            className="palette-input"
            value={category}
            onChange={(e) => {
              const next = e.target.value as ControlCategory | "";
              setCategory(next);
              setMethod("");
            }}
          >
            <option value="">— select —</option>
            {CATEGORIES.map((c) => (
              <option key={c.value} value={c.value}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
        <label className="palette-field">
          <span className="palette-field-label">method</span>
          <select
            className="palette-input"
            value={method}
            onChange={(e) => setMethod(e.target.value)}
            disabled={!category}
          >
            <option value="">{category ? "— select —" : "select category first"}</option>
            {methodOptions.map((m) => (
              <option key={m.method} value={m.method}>
                {m.method}
              </option>
            ))}
          </select>
        </label>
        <div className="palette-chip-slot">
          {ready ? (
            <DraggableChip
              category={category as ControlCategory}
              method={method}
              label={method}
              onAfterDrop={reset}
            />
          ) : (
            <div className="palette-chip-empty">configure both fields, then drag</div>
          )}
        </div>
      </div>
    </div>
  );
}

function CustomControlSection() {
  const [category, setCategory] = useState<ControlCategory | "">("");
  const [name, setName] = useState<string>("");

  const reset = () => {
    setCategory("");
    setName("");
  };

  const trimmed = name.trim();
  const ready = Boolean(category && trimmed);

  return (
    <div className="palette-section">
      <div className="palette-section-head">Add custom control</div>
      <div className="palette-section-body">
        <label className="palette-field">
          <span className="palette-field-label">category</span>
          <select
            className="palette-input"
            value={category}
            onChange={(e) => setCategory(e.target.value as ControlCategory | "")}
          >
            <option value="">— select —</option>
            {CATEGORIES.map((c) => (
              <option key={c.value} value={c.value}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
        <label className="palette-field">
          <span className="palette-field-label">name</span>
          <input
            className="palette-input"
            type="text"
            value={name}
            placeholder="my_control"
            onChange={(e) => setName(e.target.value)}
          />
        </label>
        <div className="palette-chip-slot">
          {ready ? (
            <DraggableChip
              category={category as ControlCategory}
              method={trimmed}
              label={trimmed}
              onAfterDrop={reset}
            />
          ) : (
            <div className="palette-chip-empty">configure both fields, then drag</div>
          )}
        </div>
      </div>
    </div>
  );
}

export function LibraryPanel() {
  return (
    <div className="library-row" role="region" aria-label="Steering controls library">
      <AddNewControlSection />
      <CustomControlSection />
    </div>
  );
}
