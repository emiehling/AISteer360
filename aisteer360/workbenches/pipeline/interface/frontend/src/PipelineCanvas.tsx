import { useCallback, useEffect, useMemo, useRef, type DragEvent } from "react";
import ReactFlow, {
  type Edge,
  MarkerType,
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
import { probeModel } from "./api/model";
import type { ControlCategory, ModelProbe } from "./types";

const EDGE_BUFFER = 56;
const ANCHOR_TOTAL_WIDTH = 124;  // 100px body + 24px stem
const ANCHOR_BODY_HEIGHT = 34;
const MODEL_WIDTH = 310;
const MODEL_HEIGHT = 160;  // approximate; used only for vertical centering

function computeFixtureLayout(canvasWidth: number, canvasHeight: number) {
  const promptX = EDGE_BUFFER;
  const responseX = Math.max(EDGE_BUFFER, canvasWidth - EDGE_BUFFER - ANCHOR_TOTAL_WIDTH);
  // 4/5 of the way from prompt anchor's right edge to response anchor's left edge
  const promptRight = promptX + ANCHOR_TOTAL_WIDTH;
  const targetCenter = promptRight + (responseX - promptRight) * 0.8;
  const modelX = Math.max(promptRight, Math.min(responseX - MODEL_WIDTH, targetCenter - MODEL_WIDTH / 2));
  const midY = Math.max(0, canvasHeight / 2);
  const anchorY = midY - ANCHOR_BODY_HEIGHT / 2;
  const modelY = midY - MODEL_HEIGHT / 2;
  return { promptX, modelX, responseX, anchorY, modelY };
}

function modelNodeOnChange(modelId: string) {
  usePipelineStore.getState().setModelNameOrPath(modelId);
}

function probeToParams(probe: ModelProbe | null) {
  if (!probe) return [];
  const rows: { label: string; value: string }[] = [];
  if (probe.model_type) rows.push({ label: "type", value: probe.model_type });
  if (probe.num_hidden_layers != null) rows.push({ label: "layers", value: String(probe.num_hidden_layers) });
  if (probe.hidden_size != null) rows.push({ label: "hidden dim", value: String(probe.hidden_size) });
  return rows;
}

const EDGE_ARROW_MARKER = { type: MarkerType.Arrow, width: 16, height: 16 };

function buildInitialEdges(): Edge[] {
  return [
    {
      id: "default-prompt-model",
      source: "anchor-prompt",
      sourceHandle: "out",
      target: "model",
      targetHandle: "input",
      type: "pipeline",
      markerEnd: EDGE_ARROW_MARKER,
    },
    {
      id: "default-model-response",
      source: "model",
      sourceHandle: "output",
      target: "anchor-response",
      targetHandle: "in",
      type: "pipeline",
      markerEnd: EDGE_ARROW_MARKER,
    },
  ];
}

function buildInitialNodes(
  modelNameOrPath: string,
  canvasWidth: number,
  canvasHeight: number,
  entries: { label: string; model_id: string }[],
): Node[] {
  const { promptX, modelX, responseX, anchorY, modelY } = computeFixtureLayout(
    canvasWidth,
    canvasHeight,
  );
  return [
    {
      id: "anchor-prompt",
      type: "prompt_anchor",
      position: { x: promptX, y: anchorY },
      data: { variant: "prompt" },
      draggable: false,
      selectable: false,
      deletable: false,
    },
    {
      id: "model",
      type: "target_model",
      position: { x: modelX, y: modelY },
      data: {
        modelId: modelNameOrPath,
        loaded: false,
        params: [],
        entries,
        onChangeModel: modelNodeOnChange,
      },
      deletable: false,
    },
    {
      id: "anchor-response",
      type: "response_anchor",
      position: { x: responseX, y: anchorY },
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
  const setEdges = usePipelineStore((s) => s.setEdges);
  const addControlNode = usePipelineStore((s) => s.addControlNode);
  const removeEdge = usePipelineStore((s) => s.removeEdge);
  const setSelectedNodeId = usePipelineStore((s) => s.setSelectedNodeId);
  const activeTool = usePipelineStore((s) => s.activeTool);
  const modelNameOrPath = usePipelineStore((s) => s.modelNameOrPath);
  const catalogTargetEntries = usePipelineStore((s) => s.catalogTargetEntries);
  const modelProbe = usePipelineStore((s) => s.modelProbe);

  const { screenToFlowPosition } = useReactFlow();
  const wrapperRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;

    if (usePipelineStore.getState().nodes.length === 0) {
      const width = wrapper.clientWidth || 1100;
      const height = wrapper.clientHeight || 600;
      setNodes(buildInitialNodes(modelNameOrPath, width, height, catalogTargetEntries));
      if (usePipelineStore.getState().edges.length === 0) {
        setEdges(buildInitialEdges());
      }
    }

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      const { promptX, modelX, responseX, anchorY, modelY } = computeFixtureLayout(width, height);
      const current = usePipelineStore.getState().nodes;
      const next = current.map((n) => {
        if (n.id === "anchor-prompt") return { ...n, position: { x: promptX, y: anchorY } };
        if (n.id === "model") return { ...n, position: { x: modelX, y: modelY } };
        if (n.id === "anchor-response") return { ...n, position: { x: responseX, y: anchorY } };
        return n;
      });
      setNodes(next);
    });
    observer.observe(wrapper);
    return () => observer.disconnect();
    // intentionally only re-run when the model id changes to avoid re-creating the observer.
    // catalog/entries reflection happens in the separate effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelNameOrPath, setNodes]);

  useEffect(() => {
    const current = usePipelineStore.getState().nodes;
    if (current.length === 0) return;
    const params = probeToParams(modelProbe?.model_id === modelNameOrPath ? modelProbe : null);
    const next = current.map((n) =>
      n.id === "model"
        ? {
            ...n,
            data: {
              ...n.data,
              modelId: modelNameOrPath,
              entries: catalogTargetEntries,
              params,
              onChangeModel: modelNodeOnChange,
            },
          }
        : n,
    );
    setNodes(next);
  }, [modelNameOrPath, catalogTargetEntries, modelProbe, setNodes]);

  useEffect(() => {
    const id = modelNameOrPath.trim();
    if (!id) {
      usePipelineStore.getState().setModelProbe(null);
      return;
    }
    const current = usePipelineStore.getState().modelProbe;
    if (current && current.model_id === id) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      probeModel(id)
        .then((probe) => {
          if (cancelled) return;
          if (usePipelineStore.getState().modelNameOrPath.trim() !== id) return;
          usePipelineStore.getState().setModelProbe(probe);
        })
        .catch((err) => {
          if (cancelled) return;
          console.warn("probeModel failed:", err);
          usePipelineStore.getState().setModelProbe(null);
        });
    }, 350);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [modelNameOrPath]);

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
      let parsed: { category: ControlCategory; method: string; args?: Record<string, unknown> };
      try {
        parsed = JSON.parse(raw);
      } catch {
        return;
      }
      const position = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      const id = addControlNode(parsed.category, parsed.method, position);
      if (parsed.args && Object.keys(parsed.args).length > 0) {
        usePipelineStore.getState().updateNodeArgs(id, parsed.args);
      }
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
        defaultEdgeOptions={{ type: "pipeline", markerEnd: EDGE_ARROW_MARKER }}
        proOptions={{ hideAttribution: true }}
        zoomOnScroll={false}
        zoomOnPinch={false}
        zoomOnDoubleClick={false}
        panOnDrag={false}
        panOnScroll={false}
        preventScrolling={false}
        autoPanOnNodeDrag={false}
        autoPanOnConnect={false}
        minZoom={1}
        maxZoom={1}
        defaultViewport={{ x: 0, y: 0, zoom: 1 }}
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
