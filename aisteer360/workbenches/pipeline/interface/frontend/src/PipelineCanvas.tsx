import { useCallback, useEffect, useMemo, useRef, type DragEvent } from "react";
import ReactFlow, {
  type Edge,
  type Node,
  type OnSelectionChangeParams,
  ReactFlowProvider,
  useReactFlow,
} from "reactflow";
import "reactflow/dist/style.css";

import { edgeTypes } from "./edges/edgeTypes";
import { nodeTypes } from "./nodes/nodeTypes";
import { usePipelineStore } from "./store/pipelineStore";
import { makeIsValidConnection } from "./canvas/validation";
import { DRAG_MIME } from "./panels/LibraryPanel";
import type { ControlCategory } from "./types";

const ANCHOR_LEFT_X = 40;
const ANCHOR_RIGHT_X = 1100;
const MODEL_X = 500;
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
      data: {
        modelId: modelNameOrPath,
        loaded: false,
        params: [],
        onChangeModel: undefined,
      },
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
  const onConnect = usePipelineStore((s) => s.onConnect);
  const setNodes = usePipelineStore((s) => s.setNodes);
  const addControlNode = usePipelineStore((s) => s.addControlNode);
  const removeEdge = usePipelineStore((s) => s.removeEdge);
  const setSelectedNodeId = usePipelineStore((s) => s.setSelectedNodeId);
  const activeTool = usePipelineStore((s) => s.activeTool);
  const modelNameOrPath = usePipelineStore((s) => s.modelNameOrPath);

  const { fitView, screenToFlowPosition } = useReactFlow();
  const wrapperRef = useRef<HTMLDivElement | null>(null);

  const initialNodes = useMemo(() => buildInitialNodes(modelNameOrPath), [modelNameOrPath]);

  useEffect(() => {
    if (nodes.length === 0) {
      setNodes(initialNodes);
      requestAnimationFrame(() => fitView({ padding: 0.18, duration: 0 }));
    }
  }, [nodes.length, initialNodes, setNodes, fitView]);

  const isValidConnection = useMemo(
    () => makeIsValidConnection(() => usePipelineStore.getState().edges),
    [],
  );

  const onSelectionChange = useCallback(
    (params: OnSelectionChangeParams) => {
      const node = params.nodes.find((n) => n.type === "control") ?? params.nodes[0];
      setSelectedNodeId(node?.id ?? null);
    },
    [setSelectedNodeId],
  );

  const onEdgeClick = useCallback(
    (_event: React.MouseEvent, edge: Edge) => {
      if (activeTool === "erase") {
        removeEdge(edge.id);
      }
    },
    [activeTool, removeEdge],
  );

  const onDragOver = useCallback((event: DragEvent<HTMLDivElement>) => {
    if (Array.from(event.dataTransfer.types).includes(DRAG_MIME)) {
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
    }
  }, []);

  const onDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      const raw = event.dataTransfer.getData(DRAG_MIME);
      if (!raw) return;
      event.preventDefault();
      let parsed: { category: ControlCategory; method: string };
      try {
        parsed = JSON.parse(raw);
      } catch {
        return;
      }
      const position = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      addControlNode(parsed.category, parsed.method, position);
    },
    [addControlNode, screenToFlowPosition],
  );

  return (
    <div
      className="canvas-area"
      data-active-tool={activeTool}
      ref={wrapperRef}
      onDragOver={onDragOver}
      onDrop={onDrop}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onSelectionChange={onSelectionChange}
        onEdgeClick={onEdgeClick}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        nodesConnectable={activeTool === "connect"}
        connectOnClick={false}
        isValidConnection={isValidConnection}
        defaultEdgeOptions={{ type: "pipeline" }}
        proOptions={{ hideAttribution: true }}
        zoomOnScroll={false}
        zoomOnPinch={false}
        zoomOnDoubleClick={false}
        panOnDrag={false}
        panOnScroll={false}
        minZoom={1}
        maxZoom={1}
        fitView
        fitViewOptions={{ padding: 0.18 }}
      />
    </div>
  );
}

export function PipelineCanvas() {
  return (
    <ReactFlowProvider>
      <CanvasInner />
    </ReactFlowProvider>
  );
}
