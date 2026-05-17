import { useEffect, useMemo } from "react";
import ReactFlow, {
  Background,
  BackgroundVariant,
  Controls,
  type Node,
  ReactFlowProvider,
  useReactFlow,
} from "reactflow";
import "reactflow/dist/style.css";

import { nodeTypes } from "./nodes/nodeTypes";
import { usePipelineStore } from "./store/pipelineStore";

const ANCHOR_LEFT_X = 80;
const ANCHOR_RIGHT_X = 820;
const MODEL_X = 410;
const ROW_Y = 220;

function buildInitialNodes(modelNameOrPath: string): Node[] {
  return [
    {
      id: "anchor-prompt",
      type: "prompt_anchor",
      position: { x: ANCHOR_LEFT_X, y: ROW_Y + 6 },
      data: { variant: "prompt" },
      draggable: false,
      selectable: false,
      deletable: false,
    },
    {
      id: "model",
      type: "target_model",
      position: { x: MODEL_X, y: ROW_Y - 18 },
      data: { modelNameOrPath },
      deletable: false,
    },
    {
      id: "anchor-response",
      type: "response_anchor",
      position: { x: ANCHOR_RIGHT_X, y: ROW_Y + 6 },
      data: { variant: "response" },
      draggable: false,
      selectable: false,
      deletable: false,
    },
  ];
}

function CanvasInner() {
  const nodes = usePipelineStore((s) => s.nodes);
  const edges = usePipelineStore((s) => s.edges);
  const onNodesChange = usePipelineStore((s) => s.onNodesChange);
  const onEdgesChange = usePipelineStore((s) => s.onEdgesChange);
  const setNodes = usePipelineStore((s) => s.setNodes);
  const modelNameOrPath = usePipelineStore((s) => s.modelNameOrPath);
  const { fitView } = useReactFlow();

  const initialNodes = useMemo(() => buildInitialNodes(modelNameOrPath), [modelNameOrPath]);

  useEffect(() => {
    if (nodes.length === 0) {
      setNodes(initialNodes);
      requestAnimationFrame(() => fitView({ padding: 0.18, duration: 0 }));
    }
  }, [nodes.length, initialNodes, setNodes, fitView]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      nodeTypes={nodeTypes}
      nodesConnectable={false}
      connectOnClick={false}
      proOptions={{ hideAttribution: true }}
      minZoom={0.5}
      maxZoom={2}
      fitView
      fitViewOptions={{ padding: 0.18 }}
    >
      <Background variant={BackgroundVariant.Dots} gap={18} size={1} />
      <Controls position="bottom-left" showInteractive={false} />
    </ReactFlow>
  );
}

export function PipelineCanvas() {
  return (
    <div className="canvas-area">
      <ReactFlowProvider>
        <CanvasInner />
      </ReactFlowProvider>
    </div>
  );
}
