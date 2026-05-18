import { useState } from "react";
import type { ControlNodeData, MethodFieldSpec, MethodSpec } from "../types";
import { usePipelineStore } from "../store/pipelineStore";
import { FieldWidget } from "./widgets";

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

export function ParameterPanel() {
  const selectedNodeId = usePipelineStore((s) => s.selectedNodeId);
  const nodes = usePipelineStore((s) => s.nodes);
  const methods = usePipelineStore((s) => s.methods);
  const updateNodeArgs = usePipelineStore((s) => s.updateNodeArgs);
  const updateNodeRuntimeKwargs = usePipelineStore((s) => s.updateNodeRuntimeKwargs);

  const node = selectedNodeId ? nodes.find((n) => n.id === selectedNodeId) : null;
  const isControl = node?.type === "control";
  const data = isControl ? (node!.data as ControlNodeData) : null;
  const spec: MethodSpec | undefined = data
    ? methods.find((m) => m.category === data.category && m.method === data.method)
    : undefined;

  return (
    <aside className="parameter-panel" aria-label="Parameter panel">
      {!data ? (
        <div className="panel-placeholder">select a control to edit parameters</div>
      ) : (
        <>
          <header className="panel-head">
            <span className="panel-head-method">{data.method}</span>
            <span className="panel-head-category">{data.category}</span>
          </header>
          <div className="panel-scroll">
            <PanelSection
              title="Fixed parameters"
              fields={spec?.args ?? []}
              values={data.args}
              onChange={(name, next) => updateNodeArgs(node!.id, { [name]: next })}
              emptyHint="no fixed parameters"
            />
            <PanelSection
              title="Runtime kwargs"
              fields={spec?.runtime_kwargs ?? []}
              values={data.runtimeKwargs}
              onChange={(name, next) => updateNodeRuntimeKwargs(node!.id, { [name]: next })}
              emptyHint="no runtime kwargs for this control"
            />
          </div>
        </>
      )}
    </aside>
  );
}
