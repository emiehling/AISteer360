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
