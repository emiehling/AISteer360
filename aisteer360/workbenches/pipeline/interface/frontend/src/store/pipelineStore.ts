import { create } from "zustand";
import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type Edge,
  type EdgeChange,
  MarkerType,
  type Node,
  type NodeChange,
  type XYPosition,
} from "reactflow";
import type {
  CatalogTargetEntry,
  ControlCategory,
  ControlNode,
  ControlNodeData,
  DatasetNodeData,
  MethodSpec,
  ModelNodeData,
  ModelProbe,
  PipelineDefinition,
  ToolMode,
} from "../types";

const FIXTURE_NODE_TYPES = new Set(["prompt_anchor", "response_anchor"]);

// synthetic group nodes have ids starting with this prefix. they're created
// by the canvas when a locked control group has 2+ members and removed when
// the group dissolves.
export const GROUP_NODE_ID_PREFIX = "group:";

const PALETTE_HEIGHT_DEFAULT = 440;
const PALETTE_HEIGHT_MIN = 440;
const PALETTE_HEIGHT_MAX = 720;
const PALETTE_HEIGHT_STEP = 20;

function isControlNode(n: Node): boolean {
  return n.type === "control";
}

export interface LockedGroup {
  id: string;
  members: string[];
}

function pruneLockedGroups(groups: LockedGroup[], removedId: string): LockedGroup[] {
  const next: LockedGroup[] = [];
  for (const group of groups) {
    const filtered = group.members.filter((id) => id !== removedId);
    if (filtered.length >= 2) next.push({ id: group.id, members: filtered });
  }
  return next;
}

/** Returns the set of group ids that existed in `before` but no longer survive
 *  in `after` — those groups have dissolved and any synthetic group nodes /
 *  edges referencing them should be dropped. */
function dissolvedGroupIds(before: LockedGroup[], after: LockedGroup[]): Set<string> {
  const surviving = new Set(after.map((g) => g.id));
  const dissolved = new Set<string>();
  for (const g of before) {
    if (!surviving.has(g.id)) dissolved.add(g.id);
  }
  return dissolved;
}

function uuid(): string {
  return crypto.randomUUID();
}

export type PlacementRequest =
  | { kind: "control" }
  | { kind: "dataset" }
  | { kind: "model" }
  | { kind: "steering_vector" }
  | { kind: "multiplexer" };

interface PipelineStoreState {
  nodes: Node[];
  edges: Edge[];
  modelNameOrPath: string;
  methods: MethodSpec[];
  catalogTargetEntries: CatalogTargetEntry[];
  modelProbe: ModelProbe | null;
  selectedNodeId: string | null;
  activeTool: ToolMode;
  sessionId: string | null;
  paletteHeight: number;
  paletteMinimized: boolean;
  toolbarPosition: { x: number; y: number } | null;
  toolbarLocked: boolean;
  toolbarMinimized: boolean;
  placement: PlacementRequest | null;
  targetModelNodeId: string | null;

  // raw canvas rails (without per-node-size offset). pushed into the store by
  // the canvas whenever geometry changes; consumed by collectNodes when called
  // from the toolbar (which doesn't know the canvas geometry directly).
  canvasBounds: { left: number; right: number; top: number; bottom: number } | null;
  // ID prefix for synthetic group nodes so callers can recognize them and
  // exclude them from pipeline definitions etc.
  // see also: GROUP_NODE_ID_PREFIX exported from the store module.
  pendingDeleteNodeId: string | null;
  pendingConfirm: {
    title: string;
    message: string;
    confirmLabel?: string;
    cancelLabel?: string;
    onConfirm: () => void;
  } | null;
  lockedGroups: LockedGroup[];

  setNodes: (nodes: Node[]) => void;
  setEdges: (edges: Edge[]) => void;
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;
  setModelNameOrPath: (value: string) => void;
  setMethods: (methods: MethodSpec[]) => void;
  setCatalogTargetEntries: (entries: CatalogTargetEntry[]) => void;
  setModelProbe: (probe: ModelProbe | null) => void;
  setSelectedNodeId: (id: string | null) => void;
  setActiveTool: (mode: ToolMode) => void;
  setSessionId: (id: string | null) => void;
  setPaletteHeight: (px: number) => void;
  setPaletteMinimized: (v: boolean) => void;
  togglePaletteMinimized: () => void;
  setToolbarPosition: (pos: { x: number; y: number } | null) => void;
  setToolbarLocked: (v: boolean) => void;
  toggleToolbarLocked: () => void;
  setToolbarMinimized: (v: boolean) => void;
  toggleToolbarMinimized: () => void;
  startPlacement: (req: PlacementRequest) => void;
  cancelPlacement: () => void;
  setTargetModelNodeId: (id: string | null) => void;
  setModelNodeId: (id: string, modelId: string) => void;
  updateModelNodeGenKwargs: (id: string, kwargs: Record<string, unknown>) => void;
  setModelNodeParams: (
    id: string,
    params: { label: string; value: string }[],
  ) => void;
  updateDatasetNodeData: (id: string, patch: Partial<DatasetNodeData>) => void;

  addControlNode: (
    category: ControlCategory | null,
    method: string,
    position: XYPosition,
    label?: string,
  ) => string;
  addDatasetNode: (name: string, position: XYPosition) => string;
  addModelNode: (modelId: string, position: XYPosition) => string;
  addSteeringVectorNode: (name: string, position: XYPosition) => string;
  addMultiplexerNode: (position: XYPosition) => string;
  setMultiplexerOrientation: (id: string, orientation: "vertical" | "horizontal") => void;
  removeNode: (id: string) => void;
  removeEdge: (id: string) => void;
  removeManyNodesAndEdges: (nodeIds: string[], edgeIds: string[]) => void;
  updateNodeArgs: (id: string, args: Record<string, unknown>) => void;
  updateNodeRuntimeKwargs: (id: string, kwargs: Record<string, unknown>) => void;
  updateNodeLabel: (id: string, label: string) => void;
  setNodeMethod: (id: string, method: string) => void;
  setNodeCategory: (id: string, category: ControlCategory | null) => void;
  requestDeleteNode: (id: string) => void;
  confirmDeleteNode: () => void;
  cancelDeleteNode: () => void;
  requestConfirm: (config: {
    title: string;
    message: string;
    confirmLabel?: string;
    cancelLabel?: string;
    onConfirm: () => void;
  }) => void;
  resolveConfirm: () => void;
  cancelConfirm: () => void;
  resetCanvas: () => void;
  setCanvasBounds: (bounds: { left: number; right: number; top: number; bottom: number } | null) => void;
  collectNodes: () => void;

  toPipelineDefinition: () => PipelineDefinition;
  getRuntimeKwargs: () => Record<string, unknown>;

  getGroupOf: (id: string) => LockedGroup | null;
  mergeLockGroups: (idA: string, idB: string) => void;
  removeFromLockGroup: (id: string) => void;
  splitLockGroup: (leftIds: string[], rightIds: string[]) => void;
}

export const usePipelineStore = create<PipelineStoreState>((set, get) => ({
  nodes: [],
  edges: [],
  modelNameOrPath: "",
  methods: [],
  catalogTargetEntries: [],
  modelProbe: null,
  selectedNodeId: null,
  activeTool: "select",
  sessionId: null,
  paletteHeight: PALETTE_HEIGHT_DEFAULT,
  paletteMinimized: true,
  toolbarPosition: null,
  toolbarLocked: false,
  toolbarMinimized: false,
  placement: null,
  targetModelNodeId: null,

  canvasBounds: null,
  pendingDeleteNodeId: null,
  pendingConfirm: null,
  lockedGroups: [],

  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),
  onNodesChange: (changes) => {
    set({ nodes: applyNodeChanges(changes, get().nodes) });
  },
  onEdgesChange: (changes) => {
    set({ edges: applyEdgeChanges(changes, get().edges) });
  },
  onConnect: (connection) => {
    set({
      edges: addEdge(
        {
          ...connection,
          type: "pipeline",
          markerEnd: { type: MarkerType.Arrow, width: 16, height: 16 },
        },
        get().edges,
      ),
    });
  },
  setModelNameOrPath: (value) => set({ modelNameOrPath: value }),
  setMethods: (methods) => set({ methods }),
  setCatalogTargetEntries: (entries) => set({ catalogTargetEntries: entries }),
  setModelProbe: (probe) => set({ modelProbe: probe }),
  setSelectedNodeId: (id) => set({ selectedNodeId: id }),
  setActiveTool: (mode) => set({ activeTool: mode }),
  setSessionId: (id) => set({ sessionId: id }),
  setPaletteHeight: (px) => {
    const snapped = Math.round(px / PALETTE_HEIGHT_STEP) * PALETTE_HEIGHT_STEP;
    const clamped = Math.max(PALETTE_HEIGHT_MIN, Math.min(PALETTE_HEIGHT_MAX, snapped));
    set({ paletteHeight: clamped });
  },
  setPaletteMinimized: (v) => set({ paletteMinimized: v }),
  togglePaletteMinimized: () => set({ paletteMinimized: !get().paletteMinimized }),
  setToolbarPosition: (pos) => set({ toolbarPosition: pos }),
  setToolbarLocked: (v) => set({ toolbarLocked: v }),
  toggleToolbarLocked: () => set({ toolbarLocked: !get().toolbarLocked }),
  setToolbarMinimized: (v) => set({ toolbarMinimized: v }),
  toggleToolbarMinimized: () => set({ toolbarMinimized: !get().toolbarMinimized }),
  startPlacement: (req) => set({ placement: req }),
  cancelPlacement: () => set({ placement: null }),
  setTargetModelNodeId: (id) => {
    set({ targetModelNodeId: id });
    if (id) {
      const node = get().nodes.find((n) => n.id === id);
      const modelId = (node?.data as { modelId?: string } | undefined)?.modelId;
      if (modelId !== undefined) set({ modelNameOrPath: modelId });
    } else {
      set({ modelNameOrPath: "" });
    }
  },
  setModelNodeId: (id, modelId) => {
    const trimmed = modelId.trim();
    set({
      nodes: get().nodes.map((n) =>
        n.id === id && n.type === "model"
          ? { ...n, data: { ...(n.data as Record<string, unknown>), modelId: trimmed } }
          : n,
      ),
    });
    if (get().targetModelNodeId === id) {
      set({ modelNameOrPath: trimmed });
    }
  },

  updateModelNodeGenKwargs: (id, kwargs) => {
    set({
      nodes: get().nodes.map((n) => {
        if (n.id !== id || n.type !== "model") return n;
        const data = n.data as ModelNodeData;
        return {
          ...n,
          data: { ...data, genKwargs: { ...(data.genKwargs ?? {}), ...kwargs } },
        };
      }),
    });
  },

  setModelNodeParams: (id, params) => {
    set({
      nodes: get().nodes.map((n) => {
        if (n.id !== id || n.type !== "model") return n;
        const data = n.data as ModelNodeData;
        const wasLoaded = data.params.length > 0;
        const willBeLoaded = params.length > 0;
        // model node grows from 80px (empty) to 120px (probe rows present).
        // shift Y by half the height delta so the rail-center stays fixed.
        let position = n.position;
        if (!wasLoaded && willBeLoaded) {
          position = { ...position, y: position.y - 20 };
        } else if (wasLoaded && !willBeLoaded) {
          position = { ...position, y: position.y + 20 };
        }
        return { ...n, position, data: { ...data, params } };
      }),
    });
  },

  updateDatasetNodeData: (id, patch) => {
    set({
      nodes: get().nodes.map((n) => {
        if (n.id !== id || n.type !== "dataset") return n;
        const data = n.data as DatasetNodeData;
        return { ...n, data: { ...data, ...patch } };
      }),
    });
  },

  addControlNode: (category, method, position, label) => {
    const id = uuid();
    const trimmedLabel = label?.trim();
    const data: ControlNodeData = {
      category,
      method,
      args: {},
      runtimeKwargs: {},
      label: trimmedLabel ? trimmedLabel : method,
      status: "",
      params: [],
    };
    const node: Node<ControlNodeData> = {
      id,
      type: "control",
      position,
      data,
    };
    set({ nodes: [...get().nodes, node] });
    return id;
  },

  addDatasetNode: (name, position) => {
    const id = uuid();
    const data: DatasetNodeData = {
      name: name && name.trim() ? name.trim() : "dataset",
      rowCount: null,
      source: "local",
      path: null,
      hfId: "",
      columns: [],
    };
    const node: Node = {
      id,
      type: "dataset",
      position,
      data,
    };
    set({ nodes: [...get().nodes, node] });
    return id;
  },

  addModelNode: (modelId, position) => {
    const id = uuid();
    const trimmed = modelId.trim();
    const data: ModelNodeData = {
      modelId: trimmed,
      loaded: false,
      params: [],
      genKwargs: {},
    };
    const node: Node = {
      id,
      type: "model",
      position,
      data,
    };
    const isFirstTarget = get().targetModelNodeId === null;
    set({ nodes: [...get().nodes, node] });
    if (isFirstTarget) {
      set({ targetModelNodeId: id, modelNameOrPath: trimmed });
    }
    return id;
  },

  addSteeringVectorNode: (name, position) => {
    const id = uuid();
    const node: Node = {
      id,
      type: "steering_vector",
      position,
      data: { name: name && name.trim() ? name.trim() : "vector", path: null },
    };
    set({ nodes: [...get().nodes, node] });
    return id;
  },

  addMultiplexerNode: (position) => {
    const id = uuid();
    const node: Node = {
      id,
      type: "multiplexer",
      position,
      data: { name: "mux", orientation: "vertical" },
    };
    set({ nodes: [...get().nodes, node] });
    return id;
  },

  setMultiplexerOrientation: (id, orientation) => {
    set({
      nodes: get().nodes.map((n) =>
        n.id === id && n.type === "multiplexer"
          ? { ...n, data: { ...(n.data as Record<string, unknown>), orientation } }
          : n,
      ),
    });
  },

  removeNode: (id) => {
    const node = get().nodes.find((n) => n.id === id);
    if (node && FIXTURE_NODE_TYPES.has(node.type ?? "")) return;
    // synthetic group nodes are managed via lockedGroups — no direct removal.
    if (node && node.type === "group") return;
    const wasTarget = get().targetModelNodeId === id;
    const beforeGroups = get().lockedGroups;
    const afterGroups = pruneLockedGroups(beforeGroups, id);
    const dissolved = dissolvedGroupIds(beforeGroups, afterGroups);
    set({
      nodes: get().nodes.filter((n) => n.id !== id && !dissolved.has(n.id)),
      edges: get().edges.filter(
        (e) =>
          e.source !== id &&
          e.target !== id &&
          !dissolved.has(e.source) &&
          !dissolved.has(e.target),
      ),
      selectedNodeId: get().selectedNodeId === id ? null : get().selectedNodeId,
      lockedGroups: afterGroups,
      ...(wasTarget ? { targetModelNodeId: null, modelNameOrPath: "" } : {}),
    });
  },

  removeManyNodesAndEdges: (nodeIds, edgeIds) => {
    const removableNodeIds = new Set<string>();
    const allNodes = get().nodes;
    for (const id of nodeIds) {
      const node = allNodes.find((n) => n.id === id);
      if (!node) continue;
      if (FIXTURE_NODE_TYPES.has(node.type ?? "")) continue;
      if (node.type === "group") continue;  // managed via lockedGroups
      removableNodeIds.add(id);
    }

    // Cascade: any edge feeding INTO a multiplexer input is selected for
    // deletion; its source node should also be removed (matches the existing
    // single-edge cascade behavior in removeEdge).
    const allEdges = get().edges;
    const removableEdgeIds = new Set(edgeIds);
    for (const eid of edgeIds) {
      const edge = allEdges.find((e) => e.id === eid);
      if (!edge) continue;
      const targetNode = allNodes.find((n) => n.id === edge.target);
      const isMuxInput =
        targetNode?.type === "multiplexer" &&
        typeof edge.targetHandle === "string" &&
        edge.targetHandle.startsWith("in-");
      if (!isMuxInput) continue;
      const sourceNode = allNodes.find((n) => n.id === edge.source);
      if (!sourceNode) continue;
      if (FIXTURE_NODE_TYPES.has(sourceNode.type ?? "")) continue;
      removableNodeIds.add(sourceNode.id);
    }

    if (removableNodeIds.size === 0 && removableEdgeIds.size === 0) return;

    const wasTargetRemoved =
      get().targetModelNodeId !== null && removableNodeIds.has(get().targetModelNodeId!);

    const beforeGroups = get().lockedGroups;
    let lockedGroups = beforeGroups;
    for (const id of removableNodeIds) {
      lockedGroups = pruneLockedGroups(lockedGroups, id);
    }
    const dissolved = dissolvedGroupIds(beforeGroups, lockedGroups);
    for (const gid of dissolved) removableNodeIds.add(gid);

    const nextNodes = allNodes.filter((n) => !removableNodeIds.has(n.id));
    const nextEdges = allEdges.filter(
      (e) =>
        !removableEdgeIds.has(e.id) &&
        !removableNodeIds.has(e.source) &&
        !removableNodeIds.has(e.target),
    );

    set({
      nodes: nextNodes,
      edges: nextEdges,
      lockedGroups,
      selectedNodeId:
        get().selectedNodeId && removableNodeIds.has(get().selectedNodeId!)
          ? null
          : get().selectedNodeId,
      ...(wasTargetRemoved ? { targetModelNodeId: null, modelNameOrPath: "" } : {}),
    });
  },

  removeEdge: (id) => {
    const edges = get().edges;
    const edge = edges.find((e) => e.id === id);
    if (!edge) return;
    // cascade: if the edge is feeding into a multiplexer's input, delete the
    // upstream source node too so the mux port count tracks the input set.
    const targetNode = get().nodes.find((n) => n.id === edge.target);
    const isMuxInput =
      targetNode?.type === "multiplexer" &&
      typeof edge.targetHandle === "string" &&
      edge.targetHandle.startsWith("in-");
    if (isMuxInput && edge.source) {
      const sourceNode = get().nodes.find((n) => n.id === edge.source);
      const sourceIsFixture = sourceNode && FIXTURE_NODE_TYPES.has(sourceNode.type ?? "");
      if (sourceNode && !sourceIsFixture) {
        const sourceId = sourceNode.id;
        const wasTarget = get().targetModelNodeId === sourceId;
        const beforeGroups = get().lockedGroups;
        const afterGroups = pruneLockedGroups(beforeGroups, sourceId);
        const dissolved = dissolvedGroupIds(beforeGroups, afterGroups);
        set({
          nodes: get().nodes.filter((n) => n.id !== sourceId && !dissolved.has(n.id)),
          edges: edges.filter(
            (e) =>
              e.id !== id &&
              e.source !== sourceId &&
              e.target !== sourceId &&
              !dissolved.has(e.source) &&
              !dissolved.has(e.target),
          ),
          selectedNodeId: get().selectedNodeId === sourceId ? null : get().selectedNodeId,
          lockedGroups: afterGroups,
          ...(wasTarget ? { targetModelNodeId: null, modelNameOrPath: "" } : {}),
        });
        return;
      }
    }
    set({ edges: edges.filter((e) => e.id !== id) });
  },

  updateNodeArgs: (id, args) => {
    set({
      nodes: get().nodes.map((n) => {
        if (n.id !== id || !isControlNode(n)) return n;
        const data = n.data as ControlNodeData;
        const merged = { ...data.args, ...args };
        const params = Object.entries(merged)
          .filter(([, v]) => v != null)
          .slice(0, 3)
          .map(([k, v]) => ({ label: k, value: String(v) }));
        return { ...n, data: { ...data, args: merged, params } };
      }),
    });
  },

  updateNodeLabel: (id, label) => {
    set({
      nodes: get().nodes.map((n) =>
        n.id === id && isControlNode(n)
          ? { ...n, data: { ...(n.data as ControlNodeData), label } }
          : n,
      ),
    });
  },

  setNodeMethod: (id, method) => {
    const spec = get().methods.find((m) => m.method === method);
    const defaults: Record<string, unknown> = {};
    if (spec) {
      for (const f of spec.args) {
        if (f.default !== undefined) defaults[f.name] = f.default;
      }
    }
    set({
      nodes: get().nodes.map((n) => {
        if (n.id !== id || !isControlNode(n)) return n;
        const data = n.data as ControlNodeData;
        const nextLabel = !data.label || data.label === data.method ? method : data.label;
        return {
          ...n,
          data: {
            ...data,
            method,
            args: defaults,
            params: [],
            label: nextLabel,
          },
        };
      }),
    });
  },

  setNodeCategory: (id, category) => {
    set({
      nodes: get().nodes.map((n) => {
        if (n.id !== id || !isControlNode(n)) return n;
        const data = n.data as ControlNodeData;
        if (data.category === category) return n;
        return {
          ...n,
          data: {
            ...data,
            category,
            method: "",
            args: {},
            params: [],
          },
        };
      }),
    });
  },

  requestDeleteNode: (id) => set({ pendingDeleteNodeId: id }),
  confirmDeleteNode: () => {
    const id = get().pendingDeleteNodeId;
    if (!id) return;
    const node = get().nodes.find((n) => n.id === id);
    if (node && FIXTURE_NODE_TYPES.has(node.type ?? "")) {
      set({ pendingDeleteNodeId: null });
      return;
    }
    const wasTarget = get().targetModelNodeId === id;
    const beforeGroups = get().lockedGroups;
    const afterGroups = pruneLockedGroups(beforeGroups, id);
    const dissolved = dissolvedGroupIds(beforeGroups, afterGroups);
    set({
      nodes: get().nodes.filter((n) => n.id !== id && !dissolved.has(n.id)),
      edges: get().edges.filter(
        (e) =>
          e.source !== id &&
          e.target !== id &&
          !dissolved.has(e.source) &&
          !dissolved.has(e.target),
      ),
      selectedNodeId: get().selectedNodeId === id ? null : get().selectedNodeId,
      pendingDeleteNodeId: null,
      lockedGroups: afterGroups,
      ...(wasTarget ? { targetModelNodeId: null, modelNameOrPath: "" } : {}),
    });
  },
  cancelDeleteNode: () => set({ pendingDeleteNodeId: null }),

  requestConfirm: (config) => set({ pendingConfirm: config }),
  resolveConfirm: () => {
    const cur = get().pendingConfirm;
    set({ pendingConfirm: null });
    cur?.onConfirm();
  },
  cancelConfirm: () => set({ pendingConfirm: null }),

  setCanvasBounds: (bounds) => set({ canvasBounds: bounds }),

  collectNodes: () => {
    const bounds = get().canvasBounds;
    if (!bounds) return;
    const SNAP = 20;
    const snap = (v: number) => Math.round(v / SNAP) * SNAP;
    // size fallbacks mirror the canvas's sizeForNode. measured node.width/height
    // (set by React Flow after mount) takes precedence when available.
    const fallbackSizeFor = (n: Node): { width: number; height: number } => {
      if (n.width != null && n.height != null) return { width: n.width, height: n.height };
      if (n.type === "dataset") return { width: 80, height: 80 };
      if (n.type === "steering_vector") return { width: 80, height: 80 };
      if (n.type === "multiplexer") return { width: 20, height: 80 };
      if (n.type === "model") return { width: 240, height: 80 };
      return { width: 160, height: 120 };
    };
    let dirty = false;
    const nextNodes = get().nodes.map((n) => {
      if (FIXTURE_NODE_TYPES.has(n.type ?? "")) return n;
      const { width, height } = fallbackSizeFor(n);
      const minX = bounds.left;
      const minY = bounds.top + SNAP;
      const maxX = Math.max(minX, bounds.right - width);
      const maxY = Math.max(minY, bounds.bottom - height - SNAP);
      const inX = n.position.x >= minX && n.position.x <= maxX;
      const inY = n.position.y >= minY && n.position.y <= maxY;
      if (inX && inY) return n;
      const clampedX = Math.min(maxX, Math.max(minX, n.position.x));
      const clampedY = Math.min(maxY, Math.max(minY, n.position.y));
      const x = Math.min(maxX, Math.max(minX, snap(clampedX)));
      const y = Math.min(maxY, Math.max(minY, snap(clampedY)));
      if (x === n.position.x && y === n.position.y) return n;
      dirty = true;
      return { ...n, position: { x, y } };
    });
    if (dirty) set({ nodes: nextNodes });
  },

  updateNodeRuntimeKwargs: (id, kwargs) => {
    set({
      nodes: get().nodes.map((n) =>
        n.id === id && isControlNode(n)
          ? {
              ...n,
              data: {
                ...(n.data as ControlNodeData),
                runtimeKwargs: { ...(n.data as ControlNodeData).runtimeKwargs, ...kwargs },
              },
            }
          : n,
      ),
    });
  },

  resetCanvas: () => {
    set({
      nodes: get().nodes.filter((n) => FIXTURE_NODE_TYPES.has(n.type ?? "")),
      edges: [],
      selectedNodeId: null,
      lockedGroups: [],
      targetModelNodeId: null,
      modelNameOrPath: "",
    });
  },

  getGroupOf: (id) => {
    const group = get().lockedGroups.find((g) => g.members.includes(id));
    return group ?? null;
  },
  mergeLockGroups: (idA, idB) => {
    const groups = get().lockedGroups;
    const groupA = groups.find((g) => g.members.includes(idA));
    const groupB = groups.find((g) => g.members.includes(idB));
    if (groupA && groupA === groupB) return;
    const members = new Set<string>();
    if (groupA) groupA.members.forEach((m) => members.add(m));
    else members.add(idA);
    if (groupB) groupB.members.forEach((m) => members.add(m));
    else members.add(idB);
    // pick a stable id: prefer existing groupA's id, else groupB's, else fresh.
    const reusedId = groupA?.id ?? groupB?.id ?? `${GROUP_NODE_ID_PREFIX}${uuid()}`;
    const next = groups.filter((g) => g !== groupA && g !== groupB);
    next.push({ id: reusedId, members: Array.from(members) });

    // edge migration: rewrite any edges sourced from a member to source from
    // the group node instead. if multiple migrated edges exist, keep the
    // oldest (first in the edges array) and drop the rest. inputs are left
    // alone — only outputs are constrained.
    const dissolvedGroupIds = new Set<string>();
    if (groupA && groupA.id !== reusedId) dissolvedGroupIds.add(groupA.id);
    if (groupB && groupB.id !== reusedId) dissolvedGroupIds.add(groupB.id);
    const memberSet = members;
    const currentEdges = get().edges;
    const migrated: typeof currentEdges = [];
    let firstSourceFromGroup: string | null = null;
    for (const e of currentEdges) {
      const wasFromMember = memberSet.has(e.source);
      const wasFromOldGroup = dissolvedGroupIds.has(e.source);
      if (wasFromMember || wasFromOldGroup) {
        if (firstSourceFromGroup === null) {
          firstSourceFromGroup = e.id;
          migrated.push({ ...e, source: reusedId, sourceHandle: null });
        } else {
          // drop additional source edges — only one output across the group.
        }
      } else {
        migrated.push(e);
      }
    }
    set({ lockedGroups: next, edges: migrated });
  },
  removeFromLockGroup: (id) => {
    const before = get().lockedGroups.find((g) => g.members.includes(id));
    if (!before) return;
    const pruned = pruneLockedGroups(get().lockedGroups, id);
    const stillExists = pruned.some((g) => g.id === before.id);
    let edges = get().edges;
    let nodes = get().nodes;
    if (!stillExists) {
      // group dissolved — drop the synthetic group node and any edges
      // sourced from / targeted at it.
      edges = edges.filter((e) => e.source !== before.id && e.target !== before.id);
      nodes = nodes.filter((n) => n.id !== before.id);
    }
    set({ lockedGroups: pruned, edges, nodes });
  },
  splitLockGroup: (leftIds, rightIds) => {
    const groups = get().lockedGroups;
    const all = new Set<string>([...leftIds, ...rightIds]);
    const containing = groups.find((g) => g.members.some((id) => all.has(id)));
    if (!containing) return;
    const next = groups.filter((g) => g !== containing);
    // when splitting, the original group is dissolved; both halves get fresh
    // ids if they survive (≥2 members). drop edges sourced from / targeted at
    // the original group node, and drop the group node itself.
    if (leftIds.length >= 2) {
      next.push({ id: `${GROUP_NODE_ID_PREFIX}${uuid()}`, members: [...leftIds] });
    }
    if (rightIds.length >= 2) {
      next.push({ id: `${GROUP_NODE_ID_PREFIX}${uuid()}`, members: [...rightIds] });
    }
    const edges = get().edges.filter(
      (e) => e.source !== containing.id && e.target !== containing.id,
    );
    const nodes = get().nodes.filter((n) => n.id !== containing.id);
    set({ lockedGroups: next, edges, nodes });
  },

  toPipelineDefinition: () => {
    const controlNodes: ControlNode[] = get()
      .nodes.filter(isControlNode)
      .filter((n) => {
        const d = n.data as ControlNodeData;
        return Boolean(d.method) && Boolean(d.category);
      })
      .map((n) => {
        const data = n.data as ControlNodeData;
        return {
          id: n.id,
          category: data.category as ControlCategory,
          method: data.method,
          args: data.args,
          position: [n.position.x, n.position.y],
        };
      });
    return {
      model_name_or_path: get().modelNameOrPath,
      nodes: controlNodes,
    };
  },

  getRuntimeKwargs: () => {
    const merged: Record<string, unknown> = {};
    for (const n of get().nodes) {
      if (!isControlNode(n)) continue;
      const data = n.data as ControlNodeData;
      Object.assign(merged, data.runtimeKwargs);
    }
    return merged;
  },
}));
