import { memo, type MouseEvent } from "react";
import { Handle, Position, type NodeProps } from "reactflow";

interface ModelNodeParam {
  label: string;
  value: string;
}

interface ModelNodeData {
  modelId: string;
  loaded: boolean;
  params: ModelNodeParam[];
  onChangeModel?: () => void;
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
  const onSwap = (event: MouseEvent) => {
    event.stopPropagation();
    data.onChangeModel?.();
  };

  return (
    <div className="model-wrap">
      <Handle type="target" position={Position.Left} id="input" />
      <Handle type="source" position={Position.Right} id="output" />

      <div className="model-shadow" aria-hidden />
      <div className="model-face">
        <div className="model-bar">
          <span className="model-bar-title">target model</span>
          <button
            type="button"
            className="model-bar-btn"
            onClick={onSwap}
            aria-label="Select model from Hugging Face"
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
    </div>
  );
}

export const ModelNode = memo(ModelNodeImpl);
