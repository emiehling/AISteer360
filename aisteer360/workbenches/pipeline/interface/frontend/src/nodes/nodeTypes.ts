import type { NodeTypes } from "reactflow";
import { AnchorNode } from "./AnchorNode";
import { ControlNode } from "./ControlNode";
import { ModelNode } from "./ModelNode";

export const nodeTypes: NodeTypes = {
  prompt_anchor: AnchorNode,
  response_anchor: AnchorNode,
  target_model: ModelNode,
  control: ControlNode,
};
