import { useRef, useState, type ChangeEvent } from "react";
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

const CONTROL_FILE_ACCEPT = ".control,application/json";
const DATASET_FILE_ACCEPT = ".csv,.json,.jsonl,.parquet";

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
}

function PanelSection({ title, fields, values, onChange, emptyHint }: SectionProps) {
  const [open, setOpen] = useState(true);
  const ordered = sortedFields(fields);
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
                      <span className="panel-field-type">{f.type}</span>
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

interface HeaderProps {
  name: string;
  category: ControlCategory | null;
  nameDisabled: boolean;
  categoryDisabled: boolean;
  methodSubtitle?: string | null;
  onNameChange?: (next: string) => void;
  onCategoryChange?: (next: ControlCategory | null) => void;
}

function ControlHeader({
  name,
  category,
  nameDisabled,
  categoryDisabled,
  methodSubtitle,
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
        {methodSubtitle ? (
          <span className="control-settings-method-id" title={`method: ${methodSubtitle}`}>
            method: {methodSubtitle}
          </span>
        ) : null}
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
  const updateNodeLabel = usePipelineStore((s) => s.updateNodeLabel);
  const setNodeMethod = usePipelineStore((s) => s.setNodeMethod);

  const data = node.data;
  const methodIsSet = Boolean(data.method);
  const spec: MethodSpec | undefined = methodIsSet
    ? methods.find((m) => m.category === data.category && m.method === data.method)
    : undefined;
  const displayName = data.label || data.method || "";
  const categoryMethods = methods.filter((m) => m.category === data.category);

  return (
    <>
      <ControlHeader
        name={displayName}
        category={data.category}
        nameDisabled={false}
        categoryDisabled
        methodSubtitle={methodIsSet ? data.method : null}
        onNameChange={(next) => updateNodeLabel(node.id, next)}
      />
      <div className="panel-scroll">
        {!methodIsSet && (
          <div className="panel-section">
            <div className="panel-section-head static">
              <span className="panel-section-arrow">▾</span>
              <span className="panel-section-title">Method</span>
            </div>
            <div className="panel-section-body">
              <select
                className="palette-input"
                value=""
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
        )}
        {methodIsSet && spec ? (
          <>
            <PanelSection
              title="Fixed Parameters"
              fields={spec.args}
              values={data.args}
              onChange={(name, next) => updateNodeArgs(node.id, { [name]: next })}
              emptyHint="no fixed parameters"
            />
            <PanelSection
              title="Runtime Kwargs"
              fields={spec.runtime_kwargs}
              values={data.runtimeKwargs}
              onChange={(name, next) => updateNodeRuntimeKwargs(node.id, { [name]: next })}
              emptyHint="no runtime kwargs for this control"
            />
          </>
        ) : methodIsSet ? (
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
        ) : null}
      </div>
    </>
  );
}

function ControlStagingPanel() {
  const stagingMethod = usePipelineStore((s) => s.stagingMethod);
  const stagingCategory = usePipelineStore((s) => s.stagingCategory);
  const stagingName = usePipelineStore((s) => s.stagingName);
  const stagingArgs = usePipelineStore((s) => s.stagingArgs);
  const setStagingMethod = usePipelineStore((s) => s.setStagingMethod);
  const setStagingCategory = usePipelineStore((s) => s.setStagingCategory);
  const setStagingName = usePipelineStore((s) => s.setStagingName);
  const setStagingArgs = usePipelineStore((s) => s.setStagingArgs);
  const methods = usePipelineStore((s) => s.methods);

  const spec: MethodSpec | undefined = stagingMethod
    ? methods.find((m) => m.method === stagingMethod)
    : undefined;

  const onLoad = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const text = String(reader.result ?? "");
        const parsed = JSON.parse(text) as {
          category?: ControlCategory;
          method?: string;
          name?: string;
          args?: Record<string, unknown>;
        };
        if (parsed.method) setStagingMethod(parsed.method);
        if (parsed.category) setStagingCategory(parsed.category);
        if (parsed.name) setStagingName(parsed.name);
        if (parsed.args) setStagingArgs(parsed.args);
      } catch (err) {
        console.error("failed to parse .control file:", err);
      }
    };
    reader.readAsText(file);
  };

  return (
    <>
      <SettingsToolbar accept={CONTROL_FILE_ACCEPT} onLoadFile={onLoad} />
      <ControlHeader
        name={stagingName}
        category={stagingCategory}
        nameDisabled={false}
        categoryDisabled={false}
        methodSubtitle={stagingMethod}
        onNameChange={setStagingName}
        onCategoryChange={setStagingCategory}
      />
      <div className="panel-scroll">
        {spec ? (
          <>
            <PanelSection
              title="Fixed Parameters"
              fields={spec.args}
              values={stagingArgs}
              onChange={(name, next) =>
                setStagingArgs({ ...stagingArgs, [name]: next })
              }
              emptyHint="no fixed parameters"
            />
            <PanelSection
              title="Runtime Kwargs"
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

function DatasetStagingPanel() {
  const stagingDatasetPath = usePipelineStore((s) => s.stagingDatasetPath);
  const stagingDatasetName = usePipelineStore((s) => s.stagingDatasetName);
  const setStagingDatasetPath = usePipelineStore((s) => s.setStagingDatasetPath);
  const setStagingDatasetName = usePipelineStore((s) => s.setStagingDatasetName);

  const onLoad = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setStagingDatasetPath(file.name);
    setStagingDatasetName(file.name);
  };

  return (
    <>
      <SettingsToolbar accept={DATASET_FILE_ACCEPT} onLoadFile={onLoad} />
      <header className="panel-head control-settings-head">
        <div className="control-settings-field">
          <span className="control-settings-label">name</span>
          <input
            className="palette-input"
            type="text"
            value={stagingDatasetName}
            placeholder="dataset"
            onChange={(e) => setStagingDatasetName(e.target.value)}
          />
        </div>
      </header>
      <div className="panel-scroll">
        <div className="panel-section">
          <div className="panel-section-head static">
            <span className="panel-section-arrow">▾</span>
            <span className="panel-section-title">Source</span>
          </div>
          <div className="panel-section-body">
            {stagingDatasetPath ? (
              <div className="dataset-source-row" title={stagingDatasetPath}>
                {stagingDatasetPath}
              </div>
            ) : (
              <div className="panel-empty">no file loaded — click load above</div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

function ModelStagingPanel() {
  const stagingModelId = usePipelineStore((s) => s.stagingModelId);
  const setStagingModelId = usePipelineStore((s) => s.setStagingModelId);
  const [draftId, setDraftId] = useState<string>(stagingModelId);
  const [editing, setEditing] = useState<boolean>(!stagingModelId);

  const onConfirm = () => {
    setStagingModelId(draftId.trim());
    setEditing(false);
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
      </div>
      <header className="panel-head control-settings-head">
        <div className="control-settings-field">
          <span className="control-settings-label">huggingface id</span>
          {editing ? (
            <div className="model-id-entry">
              <input
                className="palette-input"
                type="text"
                value={draftId}
                placeholder="org/model-name"
                onChange={(e) => setDraftId(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") onConfirm();
                }}
                autoFocus
              />
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
            <div className="model-id-display" title={stagingModelId}>
              {stagingModelId || "—"}
            </div>
          )}
        </div>
      </header>
    </>
  );
}

interface SettingsToolbarProps {
  accept: string;
  onLoadFile: (event: ChangeEvent<HTMLInputElement>) => void;
}

function SettingsToolbar({ accept, onLoadFile }: SettingsToolbarProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  return (
    <div className="settings-toolbar">
      <button
        type="button"
        className="settings-load-btn"
        onClick={() => inputRef.current?.click()}
        title="open file"
      >
        load
      </button>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={onLoadFile}
        style={{ display: "none" }}
      />
    </div>
  );
}

function StagingPanel() {
  const stagingKind = usePipelineStore((s) => s.stagingKind);
  if (stagingKind === "control") return <ControlStagingPanel />;
  if (stagingKind === "dataset") return <DatasetStagingPanel />;
  if (stagingKind === "model") return <ModelStagingPanel />;
  return <div className="panel-placeholder">pick an element type to configure</div>;
}

export function ParameterPanel() {
  const selectedNodeId = usePipelineStore((s) => s.selectedNodeId);
  const nodes = usePipelineStore((s) => s.nodes);

  const selected = selectedNodeId ? nodes.find((n) => n.id === selectedNodeId) : null;
  const isControl = selected?.type === "control";

  return (
    <aside className="parameter-panel" aria-label="Settings">
      <div className="palette-section-head control-settings-bar">Settings</div>
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
