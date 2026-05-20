import type { DragEvent } from "react";
import type { ControlCategory, ControlNodeParam } from "../types";
import type { StagingKind } from "../store/pipelineStore";
import { usePipelineStore } from "../store/pipelineStore";
import { StagingChip } from "./StagingChip";

export const DRAG_MIME = "application/x-aisteer-control";
export const DRAG_MIME_DATASET = "application/x-aisteer-dataset";
export const DRAG_MIME_MODEL = "application/x-aisteer-model";

// blank-from-toolbar sentinel: payload is just a kind, node is created
// empty (matching click-to-place from the toolbar).
export const DRAG_MIME_BLANK = "application/x-aisteer-blank";

const KIND_BUTTONS: { value: StagingKind; label: string }[] = [
  { value: "control", label: "control" },
  { value: "dataset", label: "dataset" },
  { value: "model", label: "model" },
];

function paramsFromArgs(args: Record<string, unknown>): ControlNodeParam[] {
  return Object.entries(args)
    .filter(([, v]) => v != null && v !== "")
    .slice(0, 3)
    .map(([k, v]) => ({ label: k, value: String(v) }));
}

function KindToggle() {
  const stagingKind = usePipelineStore((s) => s.stagingKind);
  const setStagingKind = usePipelineStore((s) => s.setStagingKind);
  return (
    <div className="palette-mode-toggle" role="tablist">
      {KIND_BUTTONS.map((b) => (
        <button
          key={b.value}
          type="button"
          role="tab"
          aria-selected={stagingKind === b.value}
          className={`palette-mode-btn${stagingKind === b.value ? " active" : ""}`}
          onClick={() => setStagingKind(stagingKind === b.value ? null : b.value)}
        >
          {b.label}
        </button>
      ))}
    </div>
  );
}

function ControlChip() {
  const stagingMethod = usePipelineStore((s) => s.stagingMethod);
  const stagingCategory = usePipelineStore((s) => s.stagingCategory);
  const stagingName = usePipelineStore((s) => s.stagingName);
  const stagingArgs = usePipelineStore((s) => s.stagingArgs);
  const resetStaging = usePipelineStore((s) => s.resetStaging);

  const ready = Boolean(stagingCategory && (stagingMethod || stagingName.trim()));

  const onDragStart = (event: DragEvent<HTMLDivElement>) => {
    if (!ready || !stagingCategory) return;
    const method = stagingMethod ?? stagingName.trim();
    const label = stagingName.trim() || method;
    const payload = {
      category: stagingCategory,
      method,
      args: stagingArgs,
      label,
    };
    event.dataTransfer.setData(DRAG_MIME, JSON.stringify(payload));
    event.dataTransfer.effectAllowed = "copy";
  };

  const onDragEnd = (event: DragEvent<HTMLDivElement>) => {
    if (event.dataTransfer.dropEffect !== "none") {
      resetStaging();
    }
  };

  const chipCategory: ControlCategory | "neutral" = stagingCategory ?? "neutral";
  const chipTitle = stagingName.trim() || stagingMethod || "";

  return (
    <StagingChip
      category={chipCategory}
      title={chipTitle}
      params={paramsFromArgs(stagingArgs)}
      draggable={ready}
      onDragStart={onDragStart}
      onDragEnd={onDragEnd}
    />
  );
}

function DatasetChip() {
  const stagingDatasetPath = usePipelineStore((s) => s.stagingDatasetPath);
  const stagingDatasetName = usePipelineStore((s) => s.stagingDatasetName);
  const resetStaging = usePipelineStore((s) => s.resetStaging);

  const ready = Boolean(stagingDatasetPath);
  const title = stagingDatasetName || "select file…";

  const onDragStart = (event: DragEvent<HTMLDivElement>) => {
    if (!ready) return;
    const payload = {
      name: stagingDatasetName || "dataset",
      path: stagingDatasetPath,
    };
    event.dataTransfer.setData(DRAG_MIME_DATASET, JSON.stringify(payload));
    event.dataTransfer.effectAllowed = "copy";
  };

  const onDragEnd = (event: DragEvent<HTMLDivElement>) => {
    if (event.dataTransfer.dropEffect !== "none") {
      resetStaging();
    }
  };

  return (
    <div
      className={`dataset-chip${ready ? " draggable" : ""}`}
      draggable={ready}
      onDragStart={onDragStart}
      onDragEnd={onDragEnd}
      title={ready ? "drag onto canvas" : "load a file in Settings"}
    >
      <div className="dataset-chip-title" title={title}>
        {title}
      </div>
      <div className="dataset-chip-footer">dataset</div>
    </div>
  );
}

function ModelChip() {
  const stagingModelId = usePipelineStore((s) => s.stagingModelId);
  const resetStaging = usePipelineStore((s) => s.resetStaging);

  const ready = Boolean(stagingModelId.trim());
  const title = stagingModelId.trim() || "enter HF id…";

  const onDragStart = (event: DragEvent<HTMLDivElement>) => {
    if (!ready) return;
    const payload = { modelId: stagingModelId.trim() };
    event.dataTransfer.setData(DRAG_MIME_MODEL, JSON.stringify(payload));
    event.dataTransfer.effectAllowed = "copy";
  };

  const onDragEnd = (event: DragEvent<HTMLDivElement>) => {
    if (event.dataTransfer.dropEffect !== "none") {
      resetStaging();
    }
  };

  return (
    <div
      className={`model-chip${ready ? " draggable" : ""}`}
      draggable={ready}
      onDragStart={onDragStart}
      onDragEnd={onDragEnd}
      title={ready ? "drag onto canvas" : "enter an HF id in Settings"}
    >
      <div className="model-chip-title" title={title}>
        {title}
      </div>
      <div className="model-chip-footer">model</div>
    </div>
  );
}

function StageContent() {
  const stagingKind = usePipelineStore((s) => s.stagingKind);
  if (stagingKind === "control") return <ControlChip />;
  if (stagingKind === "dataset") return <DatasetChip />;
  if (stagingKind === "model") return <ModelChip />;
  return <div className="palette-stage-empty">pick an element type above</div>;
}

export function LibraryPanel() {
  return (
    <div className="palette-section add-control-section" role="region" aria-label="Add element">
      <div className="palette-section-head">Add element</div>
      <div className="palette-section-body">
        <KindToggle />
        <div className="palette-stage" aria-label="Element staging area">
          <StageContent />
        </div>
      </div>
    </div>
  );
}
