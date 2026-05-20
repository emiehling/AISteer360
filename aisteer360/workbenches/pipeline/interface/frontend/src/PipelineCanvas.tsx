import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent } from "react";
import ReactFlow, {
  ConnectionMode,
  type Edge,
  MarkerType,
  type Node,
  type NodeChange,
  type OnSelectionChangeParams,
  ReactFlowProvider,
  useReactFlow,
  useViewport,
} from "reactflow";
import "reactflow/dist/style.css";

import { edgeTypes } from "./edges/edgeTypes";
import { nodeTypes } from "./nodes/nodeTypes";
import { usePipelineStore } from "./store/pipelineStore";
import { makeIsValidConnection } from "./canvas/validation";
import { ConfirmDialog } from "./canvas/ConfirmDialog";
import { DRAG_MIME, DRAG_MIME_BLANK, DRAG_MIME_DATASET, DRAG_MIME_MODEL } from "./panels/LibraryPanel";
import { probeModel } from "./api/model";
import type { ControlCategory } from "./types";
import type { PlacementRequest } from "./store/pipelineStore";

const EDGE_BUFFER = 32;
// outline-only anchors: [label 50] [gap 4] [stem 20] [gap 4] [handle 10] = 88 total.
// the prompt anchor renders this layout left-to-right (handle on the right);
// the response anchor mirrors it (handle on the left). centers below give the
// X coord of the handle's center, which is where edges visually attach.
const ANCHOR_LABEL_WIDTH = 50;
const ANCHOR_GAP = 4;
const ANCHOR_STEM_WIDTH = 20;
const ANCHOR_HANDLE_WIDTH = 10;
const ANCHOR_TOTAL_WIDTH =
  ANCHOR_LABEL_WIDTH + ANCHOR_GAP + ANCHOR_STEM_WIDTH + ANCHOR_GAP + ANCHOR_HANDLE_WIDTH;  // 88
const ANCHOR_PROMPT_HANDLE_OFFSET = ANCHOR_TOTAL_WIDTH - ANCHOR_HANDLE_WIDTH / 2;  // 83
const ANCHOR_RESPONSE_HANDLE_OFFSET = ANCHOR_HANDLE_WIDTH / 2;  // 5
const ANCHOR_BODY_HEIGHT = 40;
const MODEL_WIDTH = 340;
const MODEL_HEIGHT_FALLBACK = 240;

const SNAP_GRID: [number, number] = [20, 20];
const GRID_SIZE = SNAP_GRID[0];
const RAIL_BUFFER = 3 * GRID_SIZE;
const MARRY_THRESHOLD = 2 * GRID_SIZE;

// node sizes are integer (and even) multiples of GRID_SIZE so that left/right
// handles (placed at node center = position.y + height/2) sit on the snap grid.
// keeping height/2 a multiple of GRID_SIZE keeps controls/datasets aligned with
// anchors and the model after snap-to-grid drag.
const NODE_BOUNDS_FALLBACK = { width: 160, height: 120 };
const DATASET_FALLBACK = { width: 80, height: 80 };
const STEERING_VECTOR_FALLBACK = { width: 80, height: 80 };
// fallback for newly placed multiplexers (vertical default, 1 empty input port).
// after mount React Flow measures the actual size and node.width/height take over.
const MULTIPLEXER_FALLBACK = { width: 20, height: 80 };

const PALETTE_MINIMIZED_HEIGHT = 24;

interface CanvasGeometry {
  promptX: number;
  responseX: number;
  modelX: number;
  centerY: number;
  anchorY: number;
  modelY: number;
  promptStemMidX: number;
  responseStemMidX: number;
  width: number;
  height: number;
  // height the canvas would have if the palette were minimized; used for
  // bounds (drag arena + visual box) so the draggable region is the same
  // regardless of palette state.
  boundsHeight: number;
}

function snap(value: number, step: number): number {
  return Math.round(value / step) * step;
}

function computeGeometry(
  canvasWidth: number,
  canvasHeight: number,
  modelHeight: number,
  boundsHeight: number,
): CanvasGeometry {
  const promptX = EDGE_BUFFER;
  const responseX = Math.max(EDGE_BUFFER, canvasWidth - EDGE_BUFFER - ANCHOR_TOTAL_WIDTH);
  const promptRight = promptX + ANCHOR_TOTAL_WIDTH;
  // 4/5 of the way from prompt anchor's right edge to response anchor's left edge
  const targetCenter = promptRight + (responseX - promptRight) * 0.8;
  const modelX = Math.max(promptRight, Math.min(responseX - MODEL_WIDTH, targetCenter - MODEL_WIDTH / 2));
  // Snap centerY so dropped/snapped controls can land on it.
  const centerY = snap(Math.max(0, canvasHeight / 2), GRID_SIZE);
  const anchorY = centerY - ANCHOR_BODY_HEIGHT / 2;
  const modelY = centerY - modelHeight / 2;
  const promptStemMidX = promptX + ANCHOR_PROMPT_HANDLE_OFFSET;  // promptX + 83
  const responseStemMidX = responseX + ANCHOR_RESPONSE_HANDLE_OFFSET;  // responseX + 5
  return {
    promptX,
    responseX,
    modelX,
    centerY,
    anchorY,
    modelY,
    promptStemMidX,
    responseStemMidX,
    width: canvasWidth,
    height: canvasHeight,
    boundsHeight,
  };
}

const EDGE_ARROW_MARKER = { type: MarkerType.Arrow, width: 16, height: 16 };

function buildInitialNodes(geometry: CanvasGeometry): Node[] {
  const { promptX, responseX, anchorY } = geometry;
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

interface SeamIndicatorsProps {
  nodes: Node[];
  lockedGroups: string[][];
  sizeForNode: (node: Node) => { width: number; height: number };
  getBoundsFor: (width: number, height: number) => {
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
    railY: number;
  };
  viewportOffset: { x: number; y: number };
}

function SeamIndicators({ nodes, lockedGroups, sizeForNode, getBoundsFor, viewportOffset }: SeamIndicatorsProps) {
  const items: {
    key: string;
    left: number;
    top: number;
    leftIds: string[];
    rightIds: string[];
  }[] = [];
  const byId = new Map(nodes.map((n) => [n.id, n]));
  for (const group of lockedGroups) {
    const members = group
      .map((id) => byId.get(id))
      .filter((n): n is Node => Boolean(n) && n.type === "control");
    if (members.length < 2) continue;
    const sorted = [...members].sort((a, b) => a.position.x - b.position.x);
    for (let i = 0; i < sorted.length - 1; i++) {
      const a = sorted[i];
      const b = sorted[i + 1];
      const aSize = sizeForNode(a);
      const bSize = sizeForNode(b);
      const aRight = a.position.x + aSize.width;
      const bLeft = b.position.x;
      if (Math.abs(bLeft - aRight) > 4) continue;
      const seamX = (aRight + bLeft) / 2;
      const seamY = (a.position.y + aSize.height / 2 + b.position.y + bSize.height / 2) / 2;
      const leftIds = sorted.slice(0, i + 1).map((n) => n.id);
      const rightIds = sorted.slice(i + 1).map((n) => n.id);
      items.push({ key: `${a.id}-${b.id}`, left: seamX, top: seamY, leftIds, rightIds });
    }
  }
  if (items.length === 0) return null;
  const onSeamClick = (event: React.MouseEvent, leftIds: string[], rightIds: string[]) => {
    event.stopPropagation();
    usePipelineStore.getState().splitLockGroup(leftIds, rightIds);

    // visibly separate: try to nudge the right group rightward by a gap; if that
    // would clamp against the right rail, nudge the left group leftward instead.
    const GAP = 4 * GRID_SIZE;
    const allNodes = usePipelineStore.getState().nodes;
    const byIdMap = new Map(allNodes.map((n) => [n.id, n]));
    const rightMembers = rightIds.map((id) => byIdMap.get(id)).filter((n): n is Node => Boolean(n));
    const leftMembers = leftIds.map((id) => byIdMap.get(id)).filter((n): n is Node => Boolean(n));
    if (rightMembers.length === 0 && leftMembers.length === 0) return;

    const tryShift = (members: Node[], dx: number) => {
      const moves = new Map<string, { x: number; y: number }>();
      for (const m of members) {
        const { width, height } = sizeForNode(m);
        const { minX, maxX } = getBoundsFor(width, height);
        const x = m.position.x + dx;
        if (x < minX - 0.5 || x > maxX + 0.5) return null;
        moves.set(m.id, { x, y: m.position.y });
      }
      return moves;
    };

    let moves = tryShift(rightMembers, GAP);
    if (!moves) moves = tryShift(leftMembers, -GAP);
    if (!moves) return;

    const next = allNodes.map((n) => {
      const upd = moves!.get(n.id);
      return upd ? { ...n, position: upd } : n;
    });
    usePipelineStore.getState().setNodes(next);
  };
  return (
    <div
      className="lock-seam-layer"
      style={{ transform: `translate(${viewportOffset.x}px, ${viewportOffset.y}px)` }}
      aria-hidden={false}
    >
      {items.map((it) => (
        <button
          key={it.key}
          type="button"
          className="lock-seam"
          style={{ left: it.left, top: it.top }}
          onClick={(e) => onSeamClick(e, it.leftIds, it.rightIds)}
          aria-label="Break locked pair"
          title="click to break (or alt+drag)"
        >
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M10 14a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
            <path d="M14 10a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
          </svg>
        </button>
      ))}
    </div>
  );
}

interface GridBoundsProps {
  geometry: CanvasGeometry | null;
  viewportOffset: { x: number; y: number };
}

function GridBounds({ geometry, viewportOffset }: GridBoundsProps) {
  if (!geometry) return null;
  const left = geometry.promptStemMidX;
  const width = Math.max(0, geometry.responseStemMidX - geometry.promptStemMidX);
  const halfH = geometry.boundsHeight / 2;
  const top = geometry.centerY - halfH;
  const height = geometry.boundsHeight;
  // Align the dot pattern so dots fall at flow-coord multiples of GRID_SIZE
  // (where nodes' top-left corners snap to). The element's own (left, top) is
  // its origin; we shift by -GRID_SIZE/2 (radial-gradient draws at cell center)
  // and lock phase to flow coords so the dots stay aligned across viewport
  // translates / canvas resizes.
  const phaseX = ((left % GRID_SIZE) + GRID_SIZE) % GRID_SIZE;
  const phaseY = ((top % GRID_SIZE) + GRID_SIZE) % GRID_SIZE;
  const bgPosX = -GRID_SIZE / 2 - phaseX;
  const bgPosY = -GRID_SIZE / 2 - phaseY;
  return (
    <div
      className="canvas-grid-bounds"
      style={{
        left,
        top,
        width,
        height,
        backgroundPosition: `${bgPosX}px ${bgPosY}px`,
        transform: `translate(${viewportOffset.x}px, ${viewportOffset.y}px)`,
      }}
      aria-hidden
    />
  );
}

function CanvasInner() {
  const nodes = usePipelineStore((s) => s.nodes);
  const edges = usePipelineStore((s) => s.edges);
  const onEdgesChange = usePipelineStore((s) => s.onEdgesChange);
  const onConnect = usePipelineStore((s) => s.onConnect);
  const setNodes = usePipelineStore((s) => s.setNodes);
  const onNodesChangeStore = usePipelineStore((s) => s.onNodesChange);
  const addControlNode = usePipelineStore((s) => s.addControlNode);
  const addDatasetNode = usePipelineStore((s) => s.addDatasetNode);
  const addModelNode = usePipelineStore((s) => s.addModelNode);
  const removeEdge = usePipelineStore((s) => s.removeEdge);
  const setSelectedNodeId = usePipelineStore((s) => s.setSelectedNodeId);
  const activeTool = usePipelineStore((s) => s.activeTool);
  const modelNameOrPath = usePipelineStore((s) => s.modelNameOrPath);
  const lockedGroups = usePipelineStore((s) => s.lockedGroups);
  const selectedNodeId = usePipelineStore((s) => s.selectedNodeId);
  const paletteMinimized = usePipelineStore((s) => s.paletteMinimized);
  const placement = usePipelineStore((s) => s.placement);
  const cancelPlacement = usePipelineStore((s) => s.cancelPlacement);

  const { screenToFlowPosition, setViewport } = useReactFlow();
  const viewport = useViewport();
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const [geometry, setGeometry] = useState<CanvasGeometry | null>(null);
  // vertical layout (centerY/anchorY/modelY) is captured once at initial mount
  // and never recomputed from canvas height afterwards — palette resize/minimize
  // changes the canvas height but must not pull anchors or the model vertically.
  const lockedVerticalRef = useRef<{ centerY: number; anchorY: number; modelY: number } | null>(null);
  // group-drag bookkeeping: when a locked-group member is dragged, we keep
  // the per-member offset relative to the dragged node so groupmates follow.
  const dragGroupRef = useRef<{
    leaderId: string;
    offsets: Map<string, { dx: number; dy: number }>;
    leaderStart: { x: number; y: number };
    suppressLock: boolean;  // true when alt-drag breaks lock for this drag
  } | null>(null);

  const computeGeometryNow = useCallback((): CanvasGeometry => {
    const wrapper = wrapperRef.current;
    const w = wrapper?.clientWidth ?? 1100;
    const h = wrapper?.clientHeight ?? 600;
    const state = usePipelineStore.getState();
    const extra = state.paletteMinimized ? 0 : Math.max(0, state.paletteHeight - PALETTE_MINIMIZED_HEIGHT);
    const boundsHeight = h + extra;
    const geom = computeGeometry(w, h, MODEL_HEIGHT_FALLBACK, boundsHeight);
    const locked = lockedVerticalRef.current;
    if (!locked) return geom;
    return { ...geom, centerY: locked.centerY, anchorY: locked.anchorY, modelY: locked.modelY };
  }, []);

  const sizeForNode = useCallback((node: Node): { width: number; height: number } => {
    if (node.width != null && node.height != null) {
      return { width: node.width, height: node.height };
    }
    if (node.type === "dataset") return DATASET_FALLBACK;
    if (node.type === "steering_vector") return STEERING_VECTOR_FALLBACK;
    if (node.type === "multiplexer") return MULTIPLEXER_FALLBACK;
    if (node.type === "model") return { width: MODEL_WIDTH, height: MODEL_HEIGHT_FALLBACK };
    return NODE_BOUNDS_FALLBACK;
  }, []);

  const getBoundsFor = useCallback(
    (nodeWidth: number, nodeHeight: number) => {
      const geom = geometry ?? computeGeometryNow();
      const minX = geom.promptStemMidX + RAIL_BUFFER;
      const maxX = Math.max(minX, geom.responseStemMidX - RAIL_BUFFER - nodeWidth);
      const railY = geom.centerY - nodeHeight / 2;
      // Y bounds are an arena centered on the locked rail, sized to the
      // palette-collapsed (max) canvas height so the draggable region is
      // independent of palette state.
      const halfH = geom.boundsHeight / 2;
      const minY = geom.centerY - halfH + GRID_SIZE;
      const maxY = Math.max(minY, geom.centerY + halfH - nodeHeight - GRID_SIZE);
      return { minX, maxX, minY, maxY, railY };
    },
    [geometry, computeGeometryNow],
  );

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      const current = usePipelineStore.getState().nodes;
      const byId = new Map(current.map((n) => [n.id, n]));

      // control nodes are free on the grid once placed, but if a control was
      // just dropped or snapped to the rail with the fallback height, we
      // re-snap its Y once the real measured height arrives so its left/right
      // handles sit on centerY.
      const controlReSnaps = new Map<string, { x: number; y: number }>();
      const geomNow = geometry ?? computeGeometryNow();
      for (const change of changes) {
        if (change.type !== "dimensions" || !change.dimensions) continue;
        const node = byId.get(change.id);
        if (!node || node.type !== "control") continue;
        const newHeight = change.dimensions.height;
        const newWidth = change.dimensions.width;
        if (!newHeight || !newWidth) continue;
        const prevHeight = node.height ?? NODE_BOUNDS_FALLBACK.height;
        const wasOnRail = Math.abs(node.position.y - (geomNow.centerY - prevHeight / 2)) < 1.5;
        if (!wasOnRail) continue;
        const targetY = geomNow.centerY - newHeight / 2;
        if (Math.abs(targetY - node.position.y) < 0.5) continue;
        controlReSnaps.set(node.id, { x: node.position.x, y: targetY });
      }

      // clamp position changes for movable nodes. Control/dataset/model are
      // 2D-free within bounds.
      const clamped: NodeChange[] = [];
      const dragGroup = dragGroupRef.current;
      for (const change of changes) {
        if (change.type !== "position" || !change.position) {
          clamped.push(change);
          continue;
        }
        const node = byId.get(change.id);
        if (!node) {
          clamped.push(change);
          continue;
        }
        let clampedPos = change.position;
        if (
          node.type === "control" ||
          node.type === "dataset" ||
          node.type === "model" ||
          node.type === "steering_vector" ||
          node.type === "multiplexer"
        ) {
          const { width, height } = sizeForNode(node);
          const { minX, maxX, minY, maxY } = getBoundsFor(width, height);
          clampedPos = {
            x: Math.min(maxX, Math.max(minX, change.position.x)),
            y: Math.min(maxY, Math.max(minY, change.position.y)),
          };
        }
        clamped.push({ ...change, position: clampedPos });

        // synthesize matching position changes for locked groupmates
        if (
          dragGroup &&
          !dragGroup.suppressLock &&
          dragGroup.leaderId === change.id &&
          dragGroup.offsets.size > 0
        ) {
          for (const [memberId, offset] of dragGroup.offsets) {
            const member = byId.get(memberId);
            if (!member || member.type !== "control") continue;
            const { width, height } = sizeForNode(member);
            const { minX, maxX, minY, maxY } = getBoundsFor(width, height);
            const x = Math.min(maxX, Math.max(minX, clampedPos.x + offset.dx));
            const y = Math.min(maxY, Math.max(minY, clampedPos.y + offset.dy));
            clamped.push({
              id: memberId,
              type: "position",
              position: { x, y },
              dragging: change.dragging,
            });
          }
        }
      }
      onNodesChangeStore(clamped);

      if (controlReSnaps.size > 0) {
        const next = usePipelineStore.getState().nodes.map((n) => {
          const upd = controlReSnaps.get(n.id);
          return upd ? { ...n, position: upd } : n;
        });
        setNodes(next);
      }
    },
    [getBoundsFor, sizeForNode, onNodesChangeStore, computeGeometryNow, setNodes, geometry],
  );

  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;

    const initial = computeGeometryNow();
    if (!lockedVerticalRef.current) {
      lockedVerticalRef.current = {
        centerY: initial.centerY,
        anchorY: initial.anchorY,
        modelY: initial.modelY,
      };
    }
    setGeometry(initial);

    if (usePipelineStore.getState().nodes.length === 0) {
      setNodes(buildInitialNodes(initial));
    }

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      const state = usePipelineStore.getState();
      const extra = state.paletteMinimized ? 0 : Math.max(0, state.paletteHeight - PALETTE_MINIMIZED_HEIGHT);
      const boundsHeight = height + extra;
      const raw = computeGeometry(width, height, MODEL_HEIGHT_FALLBACK, boundsHeight);
      const locked = lockedVerticalRef.current;
      const geom: CanvasGeometry = locked
        ? { ...raw, centerY: locked.centerY, anchorY: locked.anchorY, modelY: locked.modelY }
        : raw;
      setGeometry(geom);
      // only reposition the prompt/response anchor fixtures along X so they
      // stay pinned to the canvas edges. user-placed nodes (control, dataset,
      // model, steering_vector, multiplexer) keep their absolute positions —
      // re-clamping on resize would silently rewrite the layout. nodes that
      // end up outside the new bounds remain there until the user drags them.
      const current = usePipelineStore.getState().nodes;
      let dirty = false;
      const next = current.map((n) => {
        if (n.id === "anchor-prompt" && n.position.x !== geom.promptX) {
          dirty = true;
          return { ...n, position: { x: geom.promptX, y: n.position.y } };
        }
        if (n.id === "anchor-response" && n.position.x !== geom.responseX) {
          dirty = true;
          return { ...n, position: { x: geom.responseX, y: n.position.y } };
        }
        return n;
      });
      if (dirty) setNodes(next);
    });
    observer.observe(wrapper);
    return () => observer.disconnect();
    // intentionally only re-run when the model id changes to avoid re-creating the observer.
    // catalog/entries reflection happens in the separate effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelNameOrPath, setNodes]);

  useEffect(() => {
    if (!geometry) return;
    const locked = lockedVerticalRef.current;
    if (!locked) return;
    const canvasHeight = geometry.height;

    let targetFlowCenterY: number | null = null;
    if (paletteMinimized) {
      targetFlowCenterY = locked.anchorY + ANCHOR_BODY_HEIGHT / 2;
    } else if (selectedNodeId) {
      const currentNodes = usePipelineStore.getState().nodes;
      const node = currentNodes.find((n) => n.id === selectedNodeId);
      if (
        node &&
        (node.type === "control" ||
          node.type === "dataset" ||
          node.type === "steering_vector" ||
          node.type === "multiplexer")
      ) {
        const { height } = sizeForNode(node);
        targetFlowCenterY = node.position.y + height / 2;
      }
    }

    const offsetY = targetFlowCenterY == null ? 0 : canvasHeight / 2 - targetFlowCenterY;
    setViewport({ x: 0, y: offsetY, zoom: 1 }, { duration: 200 });
  }, [paletteMinimized, selectedNodeId, geometry, sizeForNode, setViewport]);

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
    () =>
      makeIsValidConnection(
        () => usePipelineStore.getState().edges,
        () => usePipelineStore.getState().nodes,
      ),
    [],
  );

  const selectedEdgeIdRef = useRef<string | null>(null);

  const onSelectionChange = useCallback(
    (params: OnSelectionChangeParams) => {
      const node =
        params.nodes.find(
          (n) =>
            n.type === "control" ||
            n.type === "model" ||
            n.type === "dataset" ||
            n.type === "steering_vector" ||
            n.type === "multiplexer",
        ) ?? params.nodes[0];
      setSelectedNodeId(node?.id ?? null);
      selectedEdgeIdRef.current = params.edges[0]?.id ?? null;
    },
    [setSelectedNodeId],
  );

  const onNodeClick = useCallback((_event: React.MouseEvent, node: Node) => {
    if (
      node.type === "control" ||
      node.type === "dataset" ||
      node.type === "model" ||
      node.type === "steering_vector" ||
      node.type === "multiplexer"
    ) {
      usePipelineStore.getState().setPaletteMinimized(false);
    }
  }, []);

  const placeAtScreen = useCallback(
    (req: PlacementRequest, screenX: number, screenY: number) => {
      const dropped = screenToFlowPosition({ x: screenX, y: screenY });
      if (req.kind === "control") {
        const { width, height } = NODE_BOUNDS_FALLBACK;
        const { minX, maxX, minY, maxY } = getBoundsFor(width, height);
        const position = {
          x: Math.min(maxX, Math.max(minX, dropped.x - width / 2)),
          y: Math.min(maxY, Math.max(minY, dropped.y - height / 2)),
        };
        usePipelineStore.getState().addControlNode(null, "", position, "");
        return;
      }
      if (req.kind === "dataset") {
        const { width, height } = DATASET_FALLBACK;
        const { minX, maxX, minY, maxY } = getBoundsFor(width, height);
        const position = {
          x: Math.min(maxX, Math.max(minX, dropped.x - width / 2)),
          y: Math.min(maxY, Math.max(minY, dropped.y - height / 2)),
        };
        usePipelineStore.getState().addDatasetNode("dataset", position);
        return;
      }
      if (req.kind === "model") {
        const width = MODEL_WIDTH;
        const height = MODEL_HEIGHT_FALLBACK;
        const { minX, maxX, minY, maxY } = getBoundsFor(width, height);
        const position = {
          x: Math.min(maxX, Math.max(minX, dropped.x - width / 2)),
          y: Math.min(maxY, Math.max(minY, dropped.y - height / 2)),
        };
        usePipelineStore.getState().addModelNode("", position);
        return;
      }
      if (req.kind === "steering_vector") {
        const { width, height } = STEERING_VECTOR_FALLBACK;
        const { minX, maxX, minY, maxY } = getBoundsFor(width, height);
        const position = {
          x: Math.min(maxX, Math.max(minX, dropped.x - width / 2)),
          y: Math.min(maxY, Math.max(minY, dropped.y - height / 2)),
        };
        usePipelineStore.getState().addSteeringVectorNode("vector", position);
        return;
      }
      if (req.kind === "multiplexer") {
        const { width, height } = MULTIPLEXER_FALLBACK;
        const { minX, maxX, minY, maxY } = getBoundsFor(width, height);
        const position = {
          x: Math.min(maxX, Math.max(minX, dropped.x - width / 2)),
          y: Math.min(maxY, Math.max(minY, dropped.y - height / 2)),
        };
        usePipelineStore.getState().addMultiplexerNode(position);
      }
    },
    [getBoundsFor, screenToFlowPosition],
  );

  const onPaneClick = useCallback(
    (event: React.MouseEvent) => {
      const current = usePipelineStore.getState().placement;
      if (current) {
        placeAtScreen(current, event.clientX, event.clientY);
        cancelPlacement();
        return;
      }
      usePipelineStore.getState().setPaletteMinimized(true);
    },
    [cancelPlacement, placeAtScreen],
  );

  const onEdgeClick = useCallback(
    (_event: React.MouseEvent, edge: Edge) => {
      selectedEdgeIdRef.current = edge.id;
    },
    [],
  );

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLElement) {
        const tag = event.target.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        if (event.target.isContentEditable) return;
      }
      if (event.key !== "Delete" && event.key !== "Backspace") return;
      const edgeId = selectedEdgeIdRef.current;
      if (!edgeId) return;
      event.preventDefault();
      removeEdge(edgeId);
      selectedEdgeIdRef.current = null;
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [removeEdge]);

  const onDragOver = useCallback((event: DragEvent<HTMLDivElement>) => {
    const types = Array.from(event.dataTransfer.types);
    if (
      types.includes(DRAG_MIME) ||
      types.includes(DRAG_MIME_DATASET) ||
      types.includes(DRAG_MIME_MODEL) ||
      types.includes(DRAG_MIME_BLANK)
    ) {
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
    }
  }, []);

  const onNodeDragStart = useCallback(
    (event: React.MouseEvent, node: Node) => {
      if (node.type !== "control") {
        dragGroupRef.current = null;
        return;
      }
      const altPressed = event.altKey;
      if (altPressed) {
        // alt-drag: break this node out of any group, drag it solo
        usePipelineStore.getState().removeFromLockGroup(node.id);
        dragGroupRef.current = {
          leaderId: node.id,
          offsets: new Map(),
          leaderStart: { x: node.position.x, y: node.position.y },
          suppressLock: true,
        };
        return;
      }
      const group = usePipelineStore.getState().getGroupOf(node.id);
      if (!group || group.length < 2) {
        dragGroupRef.current = {
          leaderId: node.id,
          offsets: new Map(),
          leaderStart: { x: node.position.x, y: node.position.y },
          suppressLock: false,
        };
        return;
      }
      const offsets = new Map<string, { dx: number; dy: number }>();
      const allNodes = usePipelineStore.getState().nodes;
      for (const memberId of group) {
        if (memberId === node.id) continue;
        const member = allNodes.find((n) => n.id === memberId);
        if (!member) continue;
        offsets.set(memberId, {
          dx: member.position.x - node.position.x,
          dy: member.position.y - node.position.y,
        });
      }
      dragGroupRef.current = {
        leaderId: node.id,
        offsets,
        leaderStart: { x: node.position.x, y: node.position.y },
        suppressLock: false,
      };
    },
    [],
  );

  const onNodeDragStop = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      const dragGroup = dragGroupRef.current;
      dragGroupRef.current = null;
      if (node.type !== "control") return;
      if (dragGroup?.suppressLock) return;

      const allNodes = usePipelineStore.getState().nodes;
      const dragged = allNodes.find((n) => n.id === node.id);
      if (!dragged) return;
      const draggedCategory = (dragged.data as { category?: string } | undefined)?.category;
      if (!draggedCategory) return;

      const movedGroupIds = dragGroup?.offsets
        ? new Set<string>([dragged.id, ...dragGroup.offsets.keys()])
        : new Set<string>([dragged.id]);

      // group bounding box uses leftmost / rightmost outer edges
      const movedNodes = allNodes.filter((n) => movedGroupIds.has(n.id));
      let groupLeft = Infinity;
      let groupRight = -Infinity;
      for (const m of movedNodes) {
        const { width } = sizeForNode(m);
        if (m.position.x < groupLeft) groupLeft = m.position.x;
        if (m.position.x + width > groupRight) groupRight = m.position.x + width;
      }
      const draggedSize = sizeForNode(dragged);
      const draggedY = dragged.position.y;
      const draggedH = draggedSize.height;

      // find best snap candidate: same-category control NOT in the moved group,
      // whose right or left outer edge is within MARRY_THRESHOLD of the group's
      // outer edge, AND vertical overlap is significant (top/bottom within MARRY_THRESHOLD).
      type Candidate = {
        otherId: string;
        deltaX: number;  // dx to apply to entire moved group
        deltaY: number;
      };
      let best: Candidate | null = null;
      let bestScore = Infinity;
      for (const other of allNodes) {
        if (movedGroupIds.has(other.id)) continue;
        if (other.type !== "control") continue;
        const otherCat = (other.data as { category?: string } | undefined)?.category;
        if (otherCat !== draggedCategory) continue;
        const otherSize = sizeForNode(other);
        const otherLeft = other.position.x;
        const otherRight = other.position.x + otherSize.width;
        const otherTop = other.position.y;
        // vertical overlap test: dragged node's top within threshold of other's top.
        const dy = otherTop - draggedY;
        if (Math.abs(dy) > MARRY_THRESHOLD) continue;
        // horizontal vertical overlap (must overlap or be near in Y axis):
        const verticalOverlap =
          Math.min(draggedY + draggedH, otherTop + otherSize.height) -
          Math.max(draggedY, otherTop);
        if (verticalOverlap < draggedH * 0.5) continue;

        // case 1: snap group's right edge to other's left edge
        const dxLeftOfOther = otherLeft - groupRight;
        if (Math.abs(dxLeftOfOther) <= MARRY_THRESHOLD) {
          const score = Math.abs(dxLeftOfOther) + Math.abs(dy);
          if (score < bestScore) {
            bestScore = score;
            best = { otherId: other.id, deltaX: dxLeftOfOther, deltaY: dy };
          }
        }
        // case 2: snap group's left edge to other's right edge
        const dxRightOfOther = otherRight - groupLeft;
        if (Math.abs(dxRightOfOther) <= MARRY_THRESHOLD) {
          const score = Math.abs(dxRightOfOther) + Math.abs(dy);
          if (score < bestScore) {
            bestScore = score;
            best = { otherId: other.id, deltaX: dxRightOfOther, deltaY: dy };
          }
        }
      }

      if (!best) return;

      // verify the snap keeps the moved group within bounds
      let snapOk = true;
      const snapped: { id: string; x: number; y: number }[] = [];
      for (const m of movedNodes) {
        const { width, height } = sizeForNode(m);
        const { minX, maxX, minY, maxY } = getBoundsFor(width, height);
        const x = m.position.x + best.deltaX;
        const y = m.position.y + best.deltaY;
        if (x < minX - 0.5 || x > maxX + 0.5 || y < minY - 0.5 || y > maxY + 0.5) {
          snapOk = false;
          break;
        }
        snapped.push({ id: m.id, x, y });
      }
      if (!snapOk) return;

      const moveMap = new Map(snapped.map((s) => [s.id, { x: s.x, y: s.y }]));
      const next = allNodes.map((n) => {
        const upd = moveMap.get(n.id);
        return upd ? { ...n, position: { x: upd.x, y: upd.y } } : n;
      });
      setNodes(next);
      usePipelineStore.getState().mergeLockGroups(dragged.id, best.otherId);
    },
    [getBoundsFor, setNodes, sizeForNode],
  );

  const onDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      const blankRaw = event.dataTransfer.getData(DRAG_MIME_BLANK);
      const controlRaw = event.dataTransfer.getData(DRAG_MIME);
      const datasetRaw = event.dataTransfer.getData(DRAG_MIME_DATASET);
      const modelRaw = event.dataTransfer.getData(DRAG_MIME_MODEL);
      if (!blankRaw && !controlRaw && !datasetRaw && !modelRaw) return;
      event.preventDefault();

      if (
        blankRaw === "control" ||
        blankRaw === "dataset" ||
        blankRaw === "model" ||
        blankRaw === "steering_vector" ||
        blankRaw === "multiplexer"
      ) {
        placeAtScreen({ kind: blankRaw }, event.clientX, event.clientY);
        return;
      }

      const dropped = screenToFlowPosition({ x: event.clientX, y: event.clientY });

      if (controlRaw) {
        let parsed: {
          category: ControlCategory;
          method: string;
          args?: Record<string, unknown>;
          label?: string;
        };
        try {
          parsed = JSON.parse(controlRaw);
        } catch {
          return;
        }
        const { width, height } = NODE_BOUNDS_FALLBACK;
        const { minX, maxX, minY, maxY } = getBoundsFor(width, height);
        const position = {
          x: Math.min(maxX, Math.max(minX, dropped.x - width / 2)),
          y: Math.min(maxY, Math.max(minY, dropped.y - height / 2)),
        };
        const id = addControlNode(parsed.category, parsed.method, position, parsed.label);
        if (parsed.args && Object.keys(parsed.args).length > 0) {
          usePipelineStore.getState().updateNodeArgs(id, parsed.args);
        }
        return;
      }

      if (datasetRaw) {
        let parsed: { name?: string; path?: string };
        try {
          parsed = JSON.parse(datasetRaw);
        } catch {
          return;
        }
        const { width, height } = DATASET_FALLBACK;
        const { minX, maxX, minY, maxY } = getBoundsFor(width, height);
        const position = {
          x: Math.min(maxX, Math.max(minX, dropped.x)),
          y: Math.min(maxY, Math.max(minY, dropped.y)),
        };
        addDatasetNode(parsed.name || "dataset", position);
        return;
      }

      if (modelRaw) {
        let parsed: { modelId?: string };
        try {
          parsed = JSON.parse(modelRaw);
        } catch {
          return;
        }
        const width = MODEL_WIDTH;
        const height = MODEL_HEIGHT_FALLBACK;
        const { minX, maxX, minY, maxY } = getBoundsFor(width, height);
        const position = {
          x: Math.min(maxX, Math.max(minX, dropped.x)),
          y: Math.min(maxY, Math.max(minY, dropped.y)),
        };
        addModelNode(parsed.modelId || "", position);
        return;
      }
    },
    [addControlNode, addDatasetNode, addModelNode, screenToFlowPosition, getBoundsFor, placeAtScreen],
  );

  return (
    <div
      className="canvas-area"
      data-active-tool={activeTool}
      data-placing={placement ? placement.kind : undefined}
      ref={wrapperRef}
      onDragOver={onDragOver}
      onDrop={onDrop}
    >
      <GridBounds geometry={geometry} viewportOffset={{ x: viewport.x, y: viewport.y }} />
      <SeamIndicators
        nodes={nodes}
        lockedGroups={lockedGroups}
        sizeForNode={sizeForNode}
        getBoundsFor={getBoundsFor}
        viewportOffset={{ x: viewport.x, y: viewport.y }}
      />
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeDragStart={onNodeDragStart}
        onNodeDragStop={onNodeDragStop}
        onSelectionChange={onSelectionChange}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        onEdgeClick={onEdgeClick}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        nodesConnectable={true}
        connectionMode={ConnectionMode.Loose}
        connectionRadius={48}
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
        snapToGrid
        snapGrid={SNAP_GRID}
      />
      <ConfirmDialog />
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
