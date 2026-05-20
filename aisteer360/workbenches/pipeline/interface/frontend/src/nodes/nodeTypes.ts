import type { NodeTypes } from "reactflow";
import { AnchorNode } from "./AnchorNode";
import { ControlNode } from "./ControlNode";
import { DatasetNode } from "./DatasetNode";
import { ModelNode } from "./ModelNode";
import { MultiplexerNode } from "./MultiplexerNode";
import { SteeringVectorNode } from "./SteeringVectorNode";

export const nodeTypes: NodeTypes = {
  prompt_anchor: AnchorNode,
  response_anchor: AnchorNode,
  model: ModelNode,
  control: ControlNode,
  dataset: DatasetNode,
  steering_vector: SteeringVectorNode,
  multiplexer: MultiplexerNode,
};
