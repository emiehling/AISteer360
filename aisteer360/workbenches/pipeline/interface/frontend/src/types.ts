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

export type ToolMode = "select" | "connect" | "erase";

export interface ControlNodeData {
  category: ControlCategory;
  method: string;
  args: Record<string, unknown>;
  runtimeKwargs: Record<string, unknown>;
}
