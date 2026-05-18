import { useState } from "react";
import type {
  ControlCategory,
  ControlNodeData,
  MethodFieldSpec,
  MethodSpec,
} from "../types";
import { usePipelineStore } from "../store/pipelineStore";
import { FieldWidget } from "./widgets";

const CATEGORIES: { value: ControlCategory; label: string }[] = [
  { value: "input_control", label: "Input" },
  { value: "structural_control", label: "Structural" },
  { value: "state_control", label: "State" },
  { value: "output_control", label: "Output" },
];

interface SectionProps {
  title: string;
  fields: MethodFieldSpec[];
  values: Record<string, unknown>;
  onChange: (name: string, next: unknown) => void;
  emptyHint: string;
}

function PanelSection({ title, fields, values, onChange, emptyHint }: SectionProps) {
  const [open, setOpen] = useState(true);
  return (
    <div className={`panel-section${open ? " open" : " collapsed"}`}>
      <button
        type="button"
        className="panel-section-head"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="panel-section-arrow">{open ? "▾" : "▸"}</span>
        <span className="panel-section-title">{title}</span>
      </button>
      {open && (
        <div className="panel-section-body">
          {fields.length === 0 ? (
            <div className="panel-empty">{emptyHint}</div>
          ) : (
            fields.map((f) => {
              const value = values[f.name] === undefined ? f.default : values[f.name];
              return (
                <div className="panel-field" key={f.name}>
                  <label className="panel-field-label" title={f.help ?? undefined}>
                    <span className="panel-field-name">{f.name}</span>
                    <span className="panel-field-type">{f.type}</span>
                  </label>
                  {f.help && <div className="panel-field-help">{f.help}</div>}
                  <FieldWidget
                    spec={f}
                    value={value}
                    onChange={(next) => onChange(f.name, next)}
                  />
                </div>
              );
            })
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

interface HeaderProps {
  name: string;
  category: ControlCategory | null;
  nameDisabled: boolean;
  categoryDisabled: boolean;
  onNameChange?: (next: string) => void;
  onCategoryChange?: (next: ControlCategory | null) => void;
}

function ControlHeader({
  name,
  category,
  nameDisabled,
  categoryDisabled,
  onNameChange,
  onCategoryChange,
}: HeaderProps) {
  return (
    <header className="panel-head control-settings-head">
      <div className="control-settings-field">
        <span className="control-settings-label">name</span>
        <input
          className="palette-input"
          type="text"
          value={name}
          placeholder="control_name"
          disabled={nameDisabled}
          onChange={(e) => onNameChange?.(e.target.value)}
        />
      </div>
      <div className="control-settings-field">
        <span className="control-settings-label">category</span>
        <select
          className="palette-input"
          value={category ?? ""}
          disabled={categoryDisabled}
          onChange={(e) => onCategoryChange?.((e.target.value || null) as ControlCategory | null)}
        >
          <option value="">— select —</option>
          {CATEGORIES.map((c) => (
            <option key={c.value} value={c.value}>
              {c.label}
            </option>
          ))}
        </select>
      </div>
    </header>
  );
}

function SelectedNodePanel({ node }: { node: { id: string; data: ControlNodeData } }) {
  const methods = usePipelineStore((s) => s.methods);
  const updateNodeArgs = usePipelineStore((s) => s.updateNodeArgs);
  const updateNodeRuntimeKwargs = usePipelineStore((s) => s.updateNodeRuntimeKwargs);

  const data = node.data;
  const spec: MethodSpec | undefined = methods.find(
    (m) => m.category === data.category && m.method === data.method,
  );

  return (
    <>
      <ControlHeader
        name={data.method}
        category={data.category}
        nameDisabled
        categoryDisabled
      />
      <div className="panel-scroll">
        {spec ? (
          <>
            <PanelSection
              title="Fixed parameters"
              fields={spec.args}
              values={data.args}
              onChange={(name, next) => updateNodeArgs(node.id, { [name]: next })}
              emptyHint="no fixed parameters"
            />
            <PanelSection
              title="Runtime kwargs"
              fields={spec.runtime_kwargs}
              values={data.runtimeKwargs}
              onChange={(name, next) => updateNodeRuntimeKwargs(node.id, { [name]: next })}
              emptyHint="no runtime kwargs for this control"
            />
          </>
        ) : (
          <div className="panel-section">
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
        )}
      </div>
    </>
  );
}

function StagingPanel() {
  const stagingMode = usePipelineStore((s) => s.stagingMode);
  const stagingMethod = usePipelineStore((s) => s.stagingMethod);
  const stagingCategory = usePipelineStore((s) => s.stagingCategory);
  const stagingName = usePipelineStore((s) => s.stagingName);
  const stagingArgs = usePipelineStore((s) => s.stagingArgs);
  const setStagingCategory = usePipelineStore((s) => s.setStagingCategory);
  const setStagingName = usePipelineStore((s) => s.setStagingName);
  const setStagingArgs = usePipelineStore((s) => s.setStagingArgs);
  const methods = usePipelineStore((s) => s.methods);

  const isLoad = stagingMode === "load";
  const spec: MethodSpec | undefined = isLoad
    ? methods.find((m) => m.method === stagingMethod)
    : undefined;

  if (isLoad && !stagingMethod) {
    return <div className="panel-placeholder">select a control to edit parameters</div>;
  }

  return (
    <>
      <ControlHeader
        name={stagingName}
        category={stagingCategory}
        nameDisabled={isLoad}
        categoryDisabled={isLoad}
        onNameChange={setStagingName}
        onCategoryChange={setStagingCategory}
      />
      <div className="panel-scroll">
        {spec ? (
          <>
            <PanelSection
              title="Fixed parameters"
              fields={spec.args}
              values={stagingArgs}
              onChange={(name, next) =>
                setStagingArgs({ ...stagingArgs, [name]: next })
              }
              emptyHint="no fixed parameters"
            />
            <PanelSection
              title="Runtime kwargs"
              fields={spec.runtime_kwargs}
              values={{}}
              onChange={() => {}}
              emptyHint="set on the canvas after dropping"
            />
          </>
        ) : (
          <div className="panel-section">
            <div className="panel-section-head static">
              <span className="panel-section-arrow">▾</span>
              <span className="panel-section-title">Parameters</span>
            </div>
            <div className="panel-section-body">
              <FreeFormEditor args={stagingArgs} onChange={setStagingArgs} />
            </div>
          </div>
        )}
      </div>
    </>
  );
}

export function ParameterPanel() {
  const selectedNodeId = usePipelineStore((s) => s.selectedNodeId);
  const nodes = usePipelineStore((s) => s.nodes);

  const selected = selectedNodeId ? nodes.find((n) => n.id === selectedNodeId) : null;
  const isControl = selected?.type === "control";

  return (
    <aside className="parameter-panel" aria-label="Control settings">
      <div className="palette-section-head control-settings-bar">Control settings</div>
      {isControl && selected ? (
        <SelectedNodePanel
          node={{ id: selected.id, data: selected.data as ControlNodeData }}
        />
      ) : (
        <StagingPanel />
      )}
    </aside>
  );
}
