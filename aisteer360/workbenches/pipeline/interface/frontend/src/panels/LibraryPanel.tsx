import { useEffect, useState, type ChangeEvent } from "react";
import { probeModel, searchModels, type ModelSearchHit } from "../api/model";
import type {
  ControlCategory,
  ControlNodeData,
  DatasetNodeData,
  DatasetSource,
  ModelNodeData,
  ModelProbe,
} from "../types";
import { usePipelineStore } from "../store/pipelineStore";

const CATEGORIES: { value: ControlCategory; label: string }[] = [
  { value: "input_control", label: "Input" },
  { value: "structural_control", label: "Structural" },
  { value: "state_control", label: "State" },
  { value: "output_control", label: "Output" },
];

function fmtInt(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  return n.toLocaleString();
}

/** Format a parameter count with B/M/K suffix (e.g. 1_234_000_000 → "1.23B"). */
function fmtParams(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

function ControlConfigure({ node }: { node: { id: string; data: ControlNodeData } }) {
  const methods = usePipelineStore((s) => s.methods);
  const setNodeMethod = usePipelineStore((s) => s.setNodeMethod);
  const setNodeCategory = usePipelineStore((s) => s.setNodeCategory);

  const data = node.data;
  const categoryIsSet = Boolean(data.category);
  const categoryMethods = data.category
    ? methods.filter((m) => m.category === data.category)
    : [];

  return (
    <>
      <header className="panel-head control-settings-head">
        <div className="control-settings-field">
          <span className="control-settings-label">category</span>
          <div className="palette-mode-toggle palette-mode-toggle-fill" role="tablist">
            {CATEGORIES.map((c) => (
              <button
                key={c.value}
                type="button"
                role="tab"
                aria-selected={data.category === c.value}
                className={`palette-mode-btn${data.category === c.value ? " active" : ""}`}
                onClick={() =>
                  setNodeCategory(node.id, data.category === c.value ? null : c.value)
                }
              >
                {c.label}
              </button>
            ))}
          </div>
        </div>
      </header>
      <div className="panel-scroll">
        {categoryIsSet ? (
          <div className="panel-section open">
            <div className="panel-section-head static">
              <span className="panel-section-arrow">▾</span>
              <span className="panel-section-title">Method</span>
            </div>
            <div className="panel-section-body">
              <select
                className="palette-input palette-select"
                value={data.method ?? ""}
                onChange={(e) => {
                  const next = e.target.value;
                  if (next) setNodeMethod(node.id, next);
                }}
              >
                <option value="">— select method —</option>
                {categoryMethods.map((m) => (
                  <option key={m.method} value={m.method}>
                    {m.method}
                  </option>
                ))}
              </select>
              {categoryMethods.length === 0 && (
                <div className="panel-empty">no methods registered for this category</div>
              )}
            </div>
          </div>
        ) : (
          <div className="panel-section open">
            <div className="panel-section-head static">
              <span className="panel-section-arrow">▾</span>
              <span className="panel-section-title">Category</span>
            </div>
            <div className="panel-section-body">
              <div className="panel-empty">pick a category above to choose a method</div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}

function ModelConfigure({ node }: { node: { id: string; data: ModelNodeData } }) {
  const setModelNodeId = usePipelineStore((s) => s.setModelNodeId);
  const setModelNodeParams = usePipelineStore((s) => s.setModelNodeParams);
  const targetModelNodeId = usePipelineStore((s) => s.targetModelNodeId);
  const setTargetModelNodeId = usePipelineStore((s) => s.setTargetModelNodeId);

  const currentId = node.data.modelId ?? "";
  const [draftId, setDraftId] = useState<string>(currentId);
  const [editing, setEditing] = useState<boolean>(!currentId);
  const isTarget = targetModelNodeId === node.id;

  const [suggestions, setSuggestions] = useState<ModelSearchHit[]>([]);
  const [openSuggestions, setOpenSuggestions] = useState(false);
  const [highlightIdx, setHighlightIdx] = useState<number>(-1);

  const [probe, setProbe] = useState<ModelProbe | null>(null);
  const [probeError, setProbeError] = useState<string | null>(null);
  const [probing, setProbing] = useState(false);

  useEffect(() => {
    if (!editing) return;
    const q = draftId.trim();
    if (q.length < 2) {
      setSuggestions([]);
      setOpenSuggestions(false);
      return;
    }
    let cancelled = false;
    const timer = setTimeout(() => {
      searchModels(q, 10)
        .then((res) => {
          if (cancelled) return;
          setSuggestions(res.results);
          setOpenSuggestions(res.results.length > 0);
          setHighlightIdx(-1);
        })
        .catch(() => {
          if (cancelled) return;
          setSuggestions([]);
          setOpenSuggestions(false);
        });
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [draftId, editing]);

  useEffect(() => {
    const id = currentId.trim();
    if (!id) {
      setProbe(null);
      setProbeError(null);
      setProbing(false);
      return;
    }
    let cancelled = false;
    setProbing(true);
    setProbeError(null);
    probeModel(id)
      .then((res) => {
        if (cancelled) return;
        setProbe(res);
        setProbing(false);
      })
      .catch((err: Error) => {
        if (cancelled) return;
        setProbe(null);
        setProbeError(err?.message ?? "probe failed");
        setProbing(false);
      });
    return () => {
      cancelled = true;
    };
  }, [currentId]);

  // mirror selected probe fields onto the canvas node so they show up under
  // the model id. only the most useful ones — params, layers, hidden_dim, type.
  useEffect(() => {
    if (!probe) {
      setModelNodeParams(node.id, []);
      return;
    }
    const rows: { label: string; value: string }[] = [];
    if (probe.total_params != null) rows.push({ label: "parameters", value: fmtParams(probe.total_params) });
    if (probe.num_hidden_layers != null) rows.push({ label: "layers", value: fmtInt(probe.num_hidden_layers) });
    if (probe.hidden_size != null) rows.push({ label: "hidden_dim", value: fmtInt(probe.hidden_size) });
    if (probe.model_type) rows.push({ label: "type", value: probe.model_type });
    setModelNodeParams(node.id, rows);
  }, [probe, node.id, setModelNodeParams]);

  const commit = (id: string) => {
    setModelNodeId(node.id, id.trim());
    setEditing(false);
    setOpenSuggestions(false);
  };

  const onConfirm = () => commit(draftId);

  const onPickSuggestion = (hit: ModelSearchHit) => {
    setDraftId(hit.model_id);
    commit(hit.model_id);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (openSuggestions && suggestions.length > 0) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setHighlightIdx((idx) => Math.min(suggestions.length - 1, idx + 1));
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setHighlightIdx((idx) => Math.max(-1, idx - 1));
        return;
      }
      if (e.key === "Tab" && highlightIdx >= 0) {
        e.preventDefault();
        setDraftId(suggestions[highlightIdx].model_id);
        return;
      }
      if (e.key === "Escape") {
        setOpenSuggestions(false);
        return;
      }
    }
    if (e.key === "Enter") {
      if (highlightIdx >= 0 && suggestions[highlightIdx]) {
        onPickSuggestion(suggestions[highlightIdx]);
      } else {
        onConfirm();
      }
    }
  };

  return (
    <>
      <div className="settings-toolbar">
        <button
          type="button"
          className="settings-load-btn"
          onClick={() => setEditing(true)}
          title="enter HF model id"
        >
          load
        </button>
        <button
          type="button"
          className={`settings-load-btn${isTarget ? " active" : ""}`}
          onClick={() => setTargetModelNodeId(isTarget ? null : node.id)}
          title={isTarget ? "this is the target model" : "set as target model"}
        >
          {isTarget ? "★ target" : "set target"}
        </button>
      </div>
      <header className="panel-head control-settings-head model-settings-head">
        <div className="control-settings-field model-id-field">
          <span className="control-settings-label">huggingface id</span>
          {editing ? (
            <div className="model-id-entry">
              <div className="model-id-input-wrap">
                <input
                  className="palette-input model-id-input"
                  type="text"
                  value={draftId}
                  placeholder="search or paste org/model-name"
                  onChange={(e) => {
                    setDraftId(e.target.value);
                    setOpenSuggestions(true);
                  }}
                  onFocus={() => {
                    if (suggestions.length > 0) setOpenSuggestions(true);
                  }}
                  onBlur={() => {
                    setTimeout(() => setOpenSuggestions(false), 120);
                  }}
                  onKeyDown={onKeyDown}
                  autoFocus
                />
                {openSuggestions && suggestions.length > 0 ? (
                  <ul className="model-id-suggestions" role="listbox">
                    {suggestions.map((hit, idx) => (
                      <li
                        key={hit.model_id}
                        role="option"
                        aria-selected={idx === highlightIdx}
                        className={`model-id-suggestion${idx === highlightIdx ? " highlighted" : ""}`}
                        onMouseDown={(e) => {
                          e.preventDefault();
                          onPickSuggestion(hit);
                        }}
                        onMouseEnter={() => setHighlightIdx(idx)}
                      >
                        <span className="model-id-suggestion-id">{hit.model_id}</span>
                        {hit.downloads != null ? (
                          <span className="model-id-suggestion-meta">
                            ↓ {fmtInt(hit.downloads)}
                          </span>
                        ) : null}
                      </li>
                    ))}
                  </ul>
                ) : null}
              </div>
              <button
                type="button"
                className="settings-load-btn"
                onClick={onConfirm}
                disabled={!draftId.trim()}
              >
                confirm
              </button>
            </div>
          ) : (
            <div className="model-id-display" title={currentId}>
              {currentId || "—"}
            </div>
          )}
        </div>
      </header>
      {currentId ? (
        <div className="panel-section model-probe-section">
          <div className="panel-section-head static">
            <span className="panel-section-arrow">▾</span>
            <span className="panel-section-title">architecture</span>
            {probing ? <span className="model-probe-status">probing…</span> : null}
          </div>
          <div className="panel-section-body">
            {probeError ? (
              <div className="model-probe-error">{probeError}</div>
            ) : probe ? (
              <div className="panel-fields-grid model-probe-grid">
                <div className="panel-field-row model-probe-row">
                  <span className="panel-field-name">parameters</span>
                  <span className="model-probe-value">{fmtParams(probe.total_params)}</span>
                </div>
                <div className="panel-field-row model-probe-row">
                  <span className="panel-field-name">type</span>
                  <span className="model-probe-value">{probe.model_type ?? "—"}</span>
                </div>
                <div className="panel-field-row model-probe-row">
                  <span className="panel-field-name">layers</span>
                  <span className="model-probe-value">{fmtInt(probe.num_hidden_layers)}</span>
                </div>
                <div className="panel-field-row model-probe-row">
                  <span className="panel-field-name">hidden_dim</span>
                  <span className="model-probe-value">{fmtInt(probe.hidden_size)}</span>
                </div>
                <div className="panel-field-row model-probe-row">
                  <span className="panel-field-name">attention heads</span>
                  <span className="model-probe-value">{fmtInt(probe.num_attention_heads)}</span>
                </div>
                <div className="panel-field-row model-probe-row">
                  <span className="panel-field-name">kv heads</span>
                  <span className="model-probe-value">{fmtInt(probe.num_key_value_heads)}</span>
                </div>
                <div className="panel-field-row model-probe-row">
                  <span className="panel-field-name">intermediate_dim</span>
                  <span className="model-probe-value">{fmtInt(probe.intermediate_size)}</span>
                </div>
                <div className="panel-field-row model-probe-row">
                  <span className="panel-field-name">vocab</span>
                  <span className="model-probe-value">{fmtInt(probe.vocab_size)}</span>
                </div>
                <div className="panel-field-row model-probe-row">
                  <span className="panel-field-name">max positions</span>
                  <span className="model-probe-value">{fmtInt(probe.max_position_embeddings)}</span>
                </div>
              </div>
            ) : (
              !probing && <div className="panel-empty">no probe data</div>
            )}
          </div>
        </div>
      ) : null}
    </>
  );
}

function DatasetConfigure({ node }: { node: { id: string; data: DatasetNodeData } }) {
  const updateDatasetNodeData = usePipelineStore((s) => s.updateDatasetNodeData);
  const data = node.data;
  const source: DatasetSource = data.source ?? "local";

  const onLoadFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    updateDatasetNodeData(node.id, {
      path: file.name,
      name: file.name,
    });
  };

  return (
    <>
      <header className="panel-head control-settings-head">
        <div className="control-settings-field">
          <span className="control-settings-label">name</span>
          <input
            className="palette-input"
            type="text"
            value={data.name}
            placeholder="dataset"
            onChange={(e) => updateDatasetNodeData(node.id, { name: e.target.value })}
          />
        </div>
        <div className="control-settings-field">
          <span className="control-settings-label">source</span>
          <div className="palette-mode-toggle" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={source === "local"}
              className={`palette-mode-btn${source === "local" ? " active" : ""}`}
              onClick={() => updateDatasetNodeData(node.id, { source: "local" })}
            >
              local
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={source === "huggingface"}
              className={`palette-mode-btn${source === "huggingface" ? " active" : ""}`}
              onClick={() => updateDatasetNodeData(node.id, { source: "huggingface" })}
            >
              huggingface
            </button>
          </div>
        </div>
      </header>
      <div className="panel-scroll">
        <div className="panel-section open">
          <div className="panel-section-head static">
            <span className="panel-section-arrow">▾</span>
            <span className="panel-section-title">
              {source === "local" ? "Local file" : "HuggingFace dataset"}
            </span>
          </div>
          <div className="panel-section-body">
            {source === "local" ? (
              <>
                <label className="settings-load-btn" style={{ display: "inline-block" }}>
                  load
                  <input
                    type="file"
                    accept=".csv,.json,.jsonl,.parquet"
                    onChange={onLoadFile}
                    style={{ display: "none" }}
                  />
                </label>
                {data.path ? (
                  <div className="dataset-source-row" title={data.path}>
                    {data.path}
                  </div>
                ) : (
                  <div className="panel-empty">no file loaded — click load above</div>
                )}
              </>
            ) : (
              <input
                className="palette-input"
                type="text"
                value={data.hfId ?? ""}
                placeholder="org/dataset-name"
                onChange={(e) => updateDatasetNodeData(node.id, { hfId: e.target.value })}
              />
            )}
          </div>
        </div>
      </div>
    </>
  );
}

const HEAD_LABEL_BY_TYPE: Record<string, string> = {
  control: "Control",
  model: "Model",
  dataset: "Dataset",
};

export function LibraryPanel() {
  const selectedNodeId = usePipelineStore((s) => s.selectedNodeId);
  const nodes = usePipelineStore((s) => s.nodes);
  const selected = selectedNodeId ? nodes.find((n) => n.id === selectedNodeId) : null;
  const selectedType = selected?.type ?? "";
  const headLabel = HEAD_LABEL_BY_TYPE[selectedType] ?? "Configure";

  return (
    <div className="palette-section add-control-section" role="region" aria-label={headLabel}>
      <div className="palette-section-head">{headLabel}</div>
      <div className="palette-section-body">
        {selectedType === "control" && selected ? (
          <ControlConfigure
            node={{ id: selected.id, data: selected.data as ControlNodeData }}
          />
        ) : selectedType === "model" && selected ? (
          <ModelConfigure
            node={{ id: selected.id, data: selected.data as ModelNodeData }}
          />
        ) : selectedType === "dataset" && selected ? (
          <DatasetConfigure
            node={{ id: selected.id, data: selected.data as DatasetNodeData }}
          />
        ) : (
          <div className="panel-placeholder">select a node to configure</div>
        )}
      </div>
    </div>
  );
}
