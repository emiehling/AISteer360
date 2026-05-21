import { useState } from "react";
import type {
  ControlNodeData,
  DatasetColumnSpec,
  DatasetNodeData,
  MethodFieldSpec,
  MethodSpec,
  ModelNodeData,
} from "../types";
import { usePipelineStore } from "../store/pipelineStore";
import { FieldWidget } from "./widgets";

const GEN_KWARG_FIELDS: MethodFieldSpec[] = [
  { name: "max_new_tokens", type: "int", default: 256, required: false, help: null },
  { name: "temperature", type: "float", default: 1.0, required: false, help: null },
  { name: "top_p", type: "float", default: 1.0, required: false, help: null },
  { name: "top_k", type: "int", default: 50, required: false, help: null },
  { name: "repetition_penalty", type: "float", default: 1.0, required: false, help: null },
  { name: "do_sample", type: "bool", default: true, required: false, help: null },
  { name: "seed", type: "int | None", default: null, required: false, help: null },
];

function sortedFields(fields: MethodFieldSpec[]): MethodFieldSpec[] {
  return [...fields].sort((a, b) => {
    if (a.required !== b.required) return a.required ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
}

interface SectionProps {
  title: string;
  fields: MethodFieldSpec[];
  values: Record<string, unknown>;
  onChange: (name: string, next: unknown) => void;
  emptyHint: string;
  preserveOrder?: boolean;
  hideHeader?: boolean;
}

function PanelSection({
  title,
  fields,
  values,
  onChange,
  emptyHint,
  preserveOrder = false,
  hideHeader = false,
}: SectionProps) {
  const [open, setOpen] = useState(true);
  const ordered = preserveOrder ? fields : sortedFields(fields);
  return (
    <div className={`panel-section${open ? " open" : " collapsed"}`}>
      {!hideHeader && (
        <button
          type="button"
          className="panel-section-head"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
        >
          <span className="panel-section-arrow">{open ? "▾" : "▸"}</span>
          <span className="panel-section-title">{title}</span>
        </button>
      )}
      {open && (
        <div className="panel-section-body">
          {ordered.length === 0 ? (
            <div className="panel-empty">{emptyHint}</div>
          ) : (
            <div className="panel-fields-grid">
              {ordered.map((f) => {
                const value = values[f.name] === undefined ? f.default : values[f.name];
                return (
                  <div className="panel-field-row" key={f.name} title={f.help ?? undefined}>
                    <label className="panel-field-label-inline">
                      <span className="panel-field-name">
                        {f.name}
                        {f.required ? <span className="panel-field-required">*</span> : null}
                      </span>
                      <span className="panel-field-type">({f.type})</span>
                    </label>
                    <FieldWidget
                      spec={f}
                      value={value}
                      placeholder={f.help ?? undefined}
                      onChange={(next) => onChange(f.name, next)}
                    />
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface FreeFormEntry {
  key: string;
  value: string;
}

function entriesFromArgs(args: Record<string, unknown>): FreeFormEntry[] {
  return Object.entries(args).map(([key, value]) => ({
    key,
    value: value === null || value === undefined ? "" : String(value),
  }));
}

function argsFromEntries(entries: FreeFormEntry[]): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const { key, value } of entries) {
    const k = key.trim();
    if (!k) continue;
    out[k] = value;
  }
  return out;
}

interface FreeFormEditorProps {
  args: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
}

function FreeFormEditor({ args, onChange }: FreeFormEditorProps) {
  const [entries, setEntries] = useState<FreeFormEntry[]>(() => entriesFromArgs(args));

  const commit = (next: FreeFormEntry[]) => {
    setEntries(next);
    onChange(argsFromEntries(next));
  };

  const update = (idx: number, patch: Partial<FreeFormEntry>) => {
    commit(entries.map((e, i) => (i === idx ? { ...e, ...patch } : e)));
  };

  const remove = (idx: number) => commit(entries.filter((_, i) => i !== idx));
  const add = () => commit([...entries, { key: "", value: "" }]);

  return (
    <div className="free-form-editor">
      {entries.length === 0 && (
        <div className="panel-empty">no parameters — click + to add</div>
      )}
      {entries.map((entry, idx) => (
        <div className="free-form-row" key={idx}>
          <input
            className="widget-input"
            type="text"
            placeholder="key"
            value={entry.key}
            onChange={(e) => update(idx, { key: e.target.value })}
          />
          <input
            className="widget-input"
            type="text"
            placeholder="value"
            value={entry.value}
            onChange={(e) => update(idx, { value: e.target.value })}
          />
          <button
            type="button"
            className="free-form-btn"
            onClick={() => remove(idx)}
            aria-label="remove parameter"
            title="remove"
          >
            ×
          </button>
        </div>
      ))}
      <button type="button" className="free-form-add" onClick={add}>
        + add parameter
      </button>
    </div>
  );
}

function ControlParameters({ node }: { node: { id: string; data: ControlNodeData } }) {
  const methods = usePipelineStore((s) => s.methods);
  const updateNodeArgs = usePipelineStore((s) => s.updateNodeArgs);
  const updateNodeRuntimeKwargs = usePipelineStore((s) => s.updateNodeRuntimeKwargs);
  const data = node.data;
  const methodIsSet = Boolean(data.method);
  const spec: MethodSpec | undefined = methodIsSet
    ? methods.find((m) => m.category === data.category && m.method === data.method)
    : undefined;

  if (!methodIsSet) {
    return <div className="panel-placeholder">select a method in Configure to edit parameters</div>;
  }

  if (!spec) {
    return (
      <div className="panel-scroll">
        <div className="panel-section open">
          <div className="panel-section-head static">
            <span className="panel-section-arrow">▾</span>
            <span className="panel-section-title">Parameters</span>
          </div>
          <div className="panel-section-body">
            <FreeFormEditor
              args={data.args}
              onChange={(next) => {
                const current = data.args;
                const merged: Record<string, unknown> = { ...next };
                for (const k of Object.keys(current)) {
                  if (!(k in merged)) merged[k] = undefined;
                }
                updateNodeArgs(node.id, merged);
              }}
            />
          </div>
        </div>
      </div>
    );
  }

  const hasAnyFields = spec.args.length + spec.runtime_kwargs.length > 0;

  return (
    <div className="panel-scroll model-params-split">
      <div className="model-params-left">
        {hasAnyFields ? (
          <>
            {spec.args.length > 0 ? (
              <PanelSection
                title="Fixed Parameters"
                fields={spec.args}
                values={data.args}
                onChange={(name, next) => updateNodeArgs(node.id, { [name]: next })}
                emptyHint="no fixed parameters"
                hideHeader
              />
            ) : null}
            {spec.runtime_kwargs.length > 0 ? (
              <PanelSection
                title="Runtime Kwargs"
                fields={spec.runtime_kwargs}
                values={data.runtimeKwargs}
                onChange={(name, next) => updateNodeRuntimeKwargs(node.id, { [name]: next })}
                emptyHint="no runtime kwargs for this control"
                hideHeader
              />
            ) : null}
          </>
        ) : (
          <div className="panel-empty">no parameters for this method</div>
        )}
      </div>
      <div className="model-params-divider" aria-hidden />
      <div className="model-params-right" />
    </div>
  );
}

function ModelParameters({ node }: { node: { id: string; data: ModelNodeData } }) {
  const updateModelNodeGenKwargs = usePipelineStore((s) => s.updateModelNodeGenKwargs);
  const genKwargs = node.data.genKwargs ?? {};
  return (
    <div className="panel-scroll model-params-split">
      <div className="model-params-left">
        <PanelSection
          title="Generation"
          fields={GEN_KWARG_FIELDS}
          values={genKwargs}
          onChange={(name, next) => updateModelNodeGenKwargs(node.id, { [name]: next })}
          emptyHint="no generation kwargs"
          preserveOrder
          hideHeader
        />
      </div>
      <div className="model-params-divider" aria-hidden />
      <div className="model-params-right" />
    </div>
  );
}

function DatasetParameters({ node }: { node: { id: string; data: DatasetNodeData } }) {
  const updateDatasetNodeData = usePipelineStore((s) => s.updateDatasetNodeData);
  const columns: DatasetColumnSpec[] = node.data.columns ?? [];

  const updateColumn = (idx: number, patch: Partial<DatasetColumnSpec>) => {
    const next = columns.map((c, i) => (i === idx ? { ...c, ...patch } : c));
    updateDatasetNodeData(node.id, { columns: next });
  };
  const removeColumn = (idx: number) => {
    updateDatasetNodeData(node.id, { columns: columns.filter((_, i) => i !== idx) });
  };
  const addColumn = () => {
    updateDatasetNodeData(node.id, {
      columns: [...columns, { name: "", active: true, renameTo: "" }],
    });
  };

  return (
    <div className="panel-scroll">
      <div className="panel-section open">
        <div className="panel-section-head static">
          <span className="panel-section-arrow">▾</span>
          <span className="panel-section-title">Columns</span>
        </div>
        <div className="panel-section-body">
          {columns.length === 0 && (
            <div className="panel-empty">no columns defined — click + to add</div>
          )}
          {columns.map((col, idx) => (
            <div className="free-form-row" key={idx}>
              <label className="widget-checkbox" title="active">
                <input
                  type="checkbox"
                  checked={col.active}
                  onChange={(e) => updateColumn(idx, { active: e.target.checked })}
                />
              </label>
              <input
                className="widget-input"
                type="text"
                placeholder="column"
                value={col.name}
                onChange={(e) => updateColumn(idx, { name: e.target.value })}
              />
              <input
                className="widget-input"
                type="text"
                placeholder="rename to (optional)"
                value={col.renameTo}
                onChange={(e) => updateColumn(idx, { renameTo: e.target.value })}
              />
              <button
                type="button"
                className="free-form-btn"
                onClick={() => removeColumn(idx)}
                aria-label="remove column"
                title="remove"
              >
                ×
              </button>
            </div>
          ))}
          <button type="button" className="free-form-add" onClick={addColumn}>
            + add column
          </button>
        </div>
      </div>
    </div>
  );
}

function MultiplexerParameters({
  node,
}: {
  node: { id: string; data: { name?: string; orientation?: "vertical" | "horizontal" } };
}) {
  const setMultiplexerOrientation = usePipelineStore((s) => s.setMultiplexerOrientation);
  const edges = usePipelineStore((s) => s.edges);

  const orientation = node.data.orientation ?? "vertical";
  const connectedInputs = edges.filter(
    (e) => e.target === node.id && (e.targetHandle ?? "").startsWith("in-"),
  ).length;

  return (
    <header className="panel-head control-settings-head">
      <div className="control-settings-field">
        <span className="control-settings-label">orientation</span>
        <div className="palette-mode-toggle" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={orientation === "vertical"}
            className={`palette-mode-btn${orientation === "vertical" ? " active" : ""}`}
            onClick={() => setMultiplexerOrientation(node.id, "vertical")}
          >
            vertical
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={orientation === "horizontal"}
            className={`palette-mode-btn${orientation === "horizontal" ? " active" : ""}`}
            onClick={() => setMultiplexerOrientation(node.id, "horizontal")}
          >
            horizontal
          </button>
        </div>
      </div>
      <div className="control-settings-field">
        <span className="control-settings-label">connected inputs</span>
        <div className="model-id-display">{connectedInputs}</div>
      </div>
    </header>
  );
}

export function ParameterPanel() {
  const selectedNodeId = usePipelineStore((s) => s.selectedNodeId);
  const nodes = usePipelineStore((s) => s.nodes);

  const selected = selectedNodeId ? nodes.find((n) => n.id === selectedNodeId) : null;
  const selectedType = selected?.type;

  return (
    <aside className="parameter-panel" aria-label="Parameters">
      <div className="palette-section-head control-settings-bar">Parameters</div>
      {selectedType === "control" && selected ? (
        <ControlParameters
          node={{ id: selected.id, data: selected.data as ControlNodeData }}
        />
      ) : selectedType === "model" && selected ? (
        <ModelParameters
          node={{ id: selected.id, data: selected.data as ModelNodeData }}
        />
      ) : selectedType === "dataset" && selected ? (
        <DatasetParameters
          node={{ id: selected.id, data: selected.data as DatasetNodeData }}
        />
      ) : selectedType === "multiplexer" && selected ? (
        <MultiplexerParameters
          node={{
            id: selected.id,
            data: selected.data as {
              name?: string;
              orientation?: "vertical" | "horizontal";
            },
          }}
        />
      ) : (
        <div className="panel-placeholder">select a node to edit parameters</div>
      )}
    </aside>
  );
}
