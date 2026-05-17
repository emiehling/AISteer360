import { useEffect, useState, type ChangeEvent } from "react";
import type { MethodFieldSpec } from "../types";

interface FieldWidgetProps {
  spec: MethodFieldSpec;
  value: unknown;
  onChange: (next: unknown) => void;
}

function isLiteral(type: string): string[] | null {
  const m = type.match(/^Literal\[(.+)\]$/);
  if (!m) return null;
  return m[1]
    .split(",")
    .map((s) => s.trim().replace(/^['"](.*)['"]$/, "$1"));
}

function widgetKindFor(type: string): "text" | "number" | "checkbox" | "select" | "json" {
  const stripped = type.replace(/\s*\|\s*None\s*$/, "").trim();
  if (isLiteral(stripped)) return "select";
  if (stripped === "bool") return "checkbox";
  if (stripped === "int" || stripped === "float") return "number";
  if (stripped === "str") return "text";
  return "json";
}

function TextWidget({ value, onChange }: FieldWidgetProps) {
  const v = value === null || value === undefined ? "" : String(value);
  return (
    <input
      className="widget-input"
      type="text"
      value={v}
      onChange={(e: ChangeEvent<HTMLInputElement>) =>
        onChange(e.target.value === "" ? null : e.target.value)
      }
    />
  );
}

function NumberWidget({ spec, value, onChange }: FieldWidgetProps) {
  const stripped = spec.type.replace(/\s*\|\s*None\s*$/, "").trim();
  const isInt = stripped === "int";
  const v = value === null || value === undefined ? "" : String(value);
  return (
    <input
      className="widget-input"
      type="number"
      step={isInt ? 1 : "any"}
      value={v}
      onChange={(e: ChangeEvent<HTMLInputElement>) => {
        const raw = e.target.value;
        if (raw === "") {
          onChange(null);
          return;
        }
        const num = isInt ? parseInt(raw, 10) : parseFloat(raw);
        if (!Number.isNaN(num)) onChange(num);
      }}
    />
  );
}

function CheckboxWidget({ value, onChange }: FieldWidgetProps) {
  return (
    <label className="widget-checkbox">
      <input
        type="checkbox"
        checked={Boolean(value)}
        onChange={(e: ChangeEvent<HTMLInputElement>) => onChange(e.target.checked)}
      />
      <span>{value ? "true" : "false"}</span>
    </label>
  );
}

function SelectWidget({ spec, value, onChange }: FieldWidgetProps) {
  const stripped = spec.type.replace(/\s*\|\s*None\s*$/, "").trim();
  const options = isLiteral(stripped) ?? [];
  const v = value === null || value === undefined ? "" : String(value);
  return (
    <select
      className="widget-input"
      value={v}
      onChange={(e: ChangeEvent<HTMLSelectElement>) => onChange(e.target.value)}
    >
      {options.map((opt) => (
        <option key={opt} value={opt}>
          {opt}
        </option>
      ))}
    </select>
  );
}

function JsonWidget({ value, onChange }: FieldWidgetProps) {
  const initial = value === undefined ? "" : JSON.stringify(value, null, 2);
  const [draft, setDraft] = useState(initial);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDraft(value === undefined ? "" : JSON.stringify(value, null, 2));
    setError(null);
  }, [value]);

  const commit = () => {
    if (draft.trim() === "") {
      setError(null);
      onChange(null);
      return;
    }
    try {
      const parsed = JSON.parse(draft);
      setError(null);
      onChange(parsed);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="widget-json">
      <textarea
        className="widget-textarea"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        rows={Math.min(8, Math.max(2, draft.split("\n").length))}
        spellCheck={false}
      />
      {error && <div className="widget-json-error">{error}</div>}
    </div>
  );
}

export function FieldWidget(props: FieldWidgetProps) {
  const kind = widgetKindFor(props.spec.type);
  switch (kind) {
    case "text":
      return <TextWidget {...props} />;
    case "number":
      return <NumberWidget {...props} />;
    case "checkbox":
      return <CheckboxWidget {...props} />;
    case "select":
      return <SelectWidget {...props} />;
    default:
      return <JsonWidget {...props} />;
  }
}
