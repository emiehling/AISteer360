import { useMemo, type DragEvent } from "react";
import type { ControlCategory, ControlNodeParam } from "../types";
import { usePipelineStore } from "../store/pipelineStore";
import { StagingChip } from "./StagingChip";

export const DRAG_MIME = "application/x-aisteer-control";

const CATEGORY_LABEL: Record<ControlCategory, string> = {
  input_control: "Input",
  structural_control: "Structural",
  state_control: "State",
  output_control: "Output",
};

function paramsFromArgs(args: Record<string, unknown>): ControlNodeParam[] {
  return Object.entries(args)
    .filter(([, v]) => v != null && v !== "")
    .slice(0, 3)
    .map(([k, v]) => ({ label: k, value: String(v) }));
}

function ModeToggle() {
  const stagingMode = usePipelineStore((s) => s.stagingMode);
  const setStagingMode = usePipelineStore((s) => s.setStagingMode);
  return (
    <div className="palette-mode-toggle" role="tablist">
      <button
        type="button"
        role="tab"
        aria-selected={stagingMode === "new"}
        className={`palette-mode-btn${stagingMode === "new" ? " active" : ""}`}
        onClick={() => setStagingMode("new")}
      >
        new
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={stagingMode === "load"}
        className={`palette-mode-btn${stagingMode === "load" ? " active" : ""}`}
        onClick={() => setStagingMode("load")}
      >
        load
      </button>
    </div>
  );
}

function LoadModeDropdown() {
  const methods = usePipelineStore((s) => s.methods);
  const stagingMethod = usePipelineStore((s) => s.stagingMethod);
  const setStagingMethod = usePipelineStore((s) => s.setStagingMethod);

  const sorted = useMemo(
    () =>
      [...methods].sort((a, b) =>
        a.category === b.category
          ? a.method.localeCompare(b.method)
          : a.category.localeCompare(b.category),
      ),
    [methods],
  );

  return (
    <label className="palette-field">
      <span className="palette-field-label">control</span>
      <select
        className="palette-input"
        value={stagingMethod ?? ""}
        onChange={(e) => setStagingMethod(e.target.value || null)}
      >
        <option value="">— select —</option>
        {sorted.map((m) => (
          <option key={`${m.category}:${m.method}`} value={m.method}>
            {`◆ ${CATEGORY_LABEL[m.category]} · ${m.method}`}
          </option>
        ))}
      </select>
    </label>
  );
}

export function LibraryPanel() {
  const stagingMode = usePipelineStore((s) => s.stagingMode);
  const stagingMethod = usePipelineStore((s) => s.stagingMethod);
  const stagingCategory = usePipelineStore((s) => s.stagingCategory);
  const stagingName = usePipelineStore((s) => s.stagingName);
  const stagingArgs = usePipelineStore((s) => s.stagingArgs);
  const resetStaging = usePipelineStore((s) => s.resetStaging);

  const ready =
    stagingMode === "load"
      ? Boolean(stagingMethod && stagingCategory)
      : Boolean(stagingCategory && stagingName.trim());

  const onDragStart = (event: DragEvent<HTMLDivElement>) => {
    if (!ready || !stagingCategory) return;
    const method =
      stagingMode === "load" ? stagingMethod ?? stagingName.trim() : stagingName.trim();
    const payload = {
      category: stagingCategory,
      method,
      args: stagingArgs,
    };
    event.dataTransfer.setData(DRAG_MIME, JSON.stringify(payload));
    event.dataTransfer.effectAllowed = "copy";
  };

  const onDragEnd = (event: DragEvent<HTMLDivElement>) => {
    if (event.dataTransfer.dropEffect !== "none") {
      resetStaging();
    }
  };

  const chipCategory: ControlCategory | "neutral" =
    stagingCategory ?? (stagingMode === "new" ? "neutral" : "neutral");
  const chipTitle = stagingMode === "new" ? stagingName.trim() : stagingMethod ?? "";
  const chipParams = paramsFromArgs(stagingArgs);

  return (
    <div className="palette-section add-control-section" role="region" aria-label="Add control">
      <div className="palette-section-head">Add control</div>
      <div className="palette-section-body">
        <ModeToggle />
        {stagingMode === "load" && <LoadModeDropdown />}
        <div className="palette-stage" aria-label="Control staging area">
          <StagingChip
            category={chipCategory}
            title={chipTitle}
            params={chipParams}
            draggable={ready}
            onDragStart={onDragStart}
            onDragEnd={onDragEnd}
          />
        </div>
      </div>
    </div>
  );
}
