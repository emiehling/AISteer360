import { memo, useEffect, useRef, useState, type MouseEvent } from "react";
import { Handle, Position, type NodeProps } from "reactflow";
import type { CatalogTargetEntry } from "../types";

interface ModelNodeParam {
  label: string;
  value: string;
}

interface ModelNodeData {
  modelId: string;
  loaded: boolean;
  params: ModelNodeParam[];
  entries?: CatalogTargetEntry[];
  onChangeModel?: (modelId: string) => void;
}

function SwapIcon() {
  return (
    <svg
      width="11"
      height="11"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M12 3a9 9 0 1 0 9 9" />
      <path d="M21 3v5h-5" />
      <path d="M21 12A9 9 0 0 0 12 3" />
    </svg>
  );
}

function ModelNodeImpl({ data }: NodeProps<ModelNodeData>) {
  const display = data.modelId || "‹ select model ›";
  const [open, setOpen] = useState(false);
  const popoverRef = useRef<HTMLDivElement | null>(null);
  const buttonRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (event: globalThis.MouseEvent) => {
      const target = event.target as Node | null;
      if (popoverRef.current?.contains(target as Node)) return;
      if (buttonRef.current?.contains(target as Node)) return;
      setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  const onSwap = (event: MouseEvent) => {
    event.stopPropagation();
    setOpen((prev) => !prev);
  };

  const onPick = (event: MouseEvent, modelId: string) => {
    event.stopPropagation();
    data.onChangeModel?.(modelId);
    setOpen(false);
  };

  const entries = data.entries ?? [];

  return (
    <div className="model-wrap">
      <Handle type="target" position={Position.Left} id="input" />
      <Handle type="source" position={Position.Right} id="output" />

      <div className="model-face">
        <div className="model-bar">
          <span className="model-bar-title">target model</span>
          <button
            ref={buttonRef}
            type="button"
            className="model-bar-btn"
            onClick={onSwap}
            aria-label="Select model from catalog"
            aria-haspopup="listbox"
            aria-expanded={open}
            title="change model"
          >
            <SwapIcon />
          </button>
          <span
            className={`model-bar-dot ${data.loaded ? "loaded" : "unloaded"}`}
            aria-label={data.loaded ? "model loaded" : "model not loaded"}
            title={data.loaded ? "model loaded" : "model not loaded"}
          />
        </div>

        <div className="model-id-row" title={data.modelId || "no model selected"}>
          {display}
        </div>

        {data.params.length > 0 ? (
          <div className="model-params">
            {data.params.map((p, idx) => (
              <div key={`${p.label}-${idx}`} className="model-param-row">
                <span className="model-param-label">{p.label}</span>
                <span className="model-param-value" title={p.value}>
                  {p.value}
                </span>
              </div>
            ))}
          </div>
        ) : null}
      </div>

      {open ? (
        <div
          ref={popoverRef}
          className="model-picker-popover"
          role="listbox"
          onClick={(e) => e.stopPropagation()}
        >
          {entries.length === 0 ? (
            <div className="model-picker-empty">no target-eligible models in catalog</div>
          ) : (
            entries.map((entry) => {
              const selected = entry.model_id === data.modelId;
              return (
                <button
                  key={entry.model_id}
                  type="button"
                  className={`model-picker-row${selected ? " selected" : ""}`}
                  role="option"
                  aria-selected={selected}
                  onClick={(e) => onPick(e, entry.model_id)}
                >
                  <span className="model-picker-label">{entry.label}</span>
                  <span className="model-picker-id" title={entry.model_id}>
                    {entry.model_id}
                  </span>
                </button>
              );
            })
          )}
        </div>
      ) : null}
    </div>
  );
}

export const ModelNode = memo(ModelNodeImpl);
