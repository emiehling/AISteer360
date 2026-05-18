import type { ReactNode } from "react";

export type ControlCategory =
  | "input_control"
  | "structural_control"
  | "state_control"
  | "output_control";

export interface ControlNode {
  id: string;
  category: ControlCategory;
  method: string;
  args: Record<string, unknown>;
  position: [number, number];
}

export interface PipelineDefinition {
  model_name_or_path: string;
  nodes: ControlNode[];
}

export interface MethodFieldSpec {
  name: string;
  type: string;
  default: unknown;
  required: boolean;
  help: string | null;
}

export interface MethodSpec {
  category: ControlCategory;
  method: string;
  args: MethodFieldSpec[];
  runtime_kwargs: MethodFieldSpec[];
}

export interface MethodsResponse {
  methods: MethodSpec[];
}

export interface CatalogEntry {
  label: string;
  model_id: string;
  provider: string;
  endpoint: string | null;
  roles: string[];
}

export interface CatalogResponse {
  entries: CatalogEntry[];
  providers: string[];
  roles: string[];
}

export interface CatalogTargetEntry {
  label: string;
  model_id: string;
}

export interface ModelProbe {
  model_id: string;
  num_hidden_layers: number | null;
  hidden_size: number | null;
  num_attention_heads: number | null;
  num_key_value_heads: number | null;
  intermediate_size: number | null;
  vocab_size: number | null;
  max_position_embeddings: number | null;
  model_type: string | null;
  source: string;
}

export type ToolMode = "select" | "connect" | "erase";

export interface ControlNodeParam {
  icon?: ReactNode;
  label: string;
  value: string;
}

export interface ControlNodeData {
  category: ControlCategory;
  method: string;
  args: Record<string, unknown>;
  runtimeKwargs: Record<string, unknown>;
  label?: string;
  status?: string;
  params?: ControlNodeParam[];
  onClose?: () => void;
}

export interface DatasetNodeData {
  name: string;
  rowCount?: number | null;
  onClose?: () => void;
}
