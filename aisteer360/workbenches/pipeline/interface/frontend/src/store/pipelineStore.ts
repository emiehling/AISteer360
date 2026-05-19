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
  MethodSpec,
  ModelProbe,
  PipelineDefinition,
  ToolMode,
} from "../types";

const FIXTURE_NODE_TYPES = new Set(["prompt_anchor", "response_anchor", "target_model"]);

const PALETTE_HEIGHT_DEFAULT = 440;
const PALETTE_HEIGHT_MIN = 440;
const PALETTE_HEIGHT_MAX = 720;
const PALETTE_HEIGHT_STEP = 20;

function isControlNode(n: Node): boolean {
  return n.type === "control";
}

function pruneLockedGroups(groups: string[][], removedId: string): string[][] {
  const next: string[][] = [];
  for (const group of groups) {
    const filtered = group.filter((id) => id !== removedId);
    if (filtered.length >= 2) next.push(filtered);
  }
  return next;
}

function uuid(): string {
  return crypto.randomUUID();
}

export type StagingMode = "new" | "load";

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

  stagingMode: StagingMode;
  stagingMethod: string | null;
  stagingCategory: ControlCategory | null;
  stagingName: string;
  stagingArgs: Record<string, unknown>;

  pendingDeleteNodeId: string | null;
  lockedGroups: string[][];

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

  setStagingMode: (mode: StagingMode) => void;
  setStagingMethod: (method: string | null) => void;
  setStagingCategory: (category: ControlCategory | null) => void;
  setStagingName: (name: string) => void;
  setStagingArgs: (args: Record<string, unknown>) => void;
  resetStaging: () => void;

  addControlNode: (
    category: ControlCategory,
    method: string,
    position: XYPosition,
    label?: string,
  ) => string;
  addDatasetNode: (name: string, position: XYPosition) => string;
  removeNode: (id: string) => void;
  removeEdge: (id: string) => void;
  updateNodeArgs: (id: string, args: Record<string, unknown>) => void;
  updateNodeRuntimeKwargs: (id: string, kwargs: Record<string, unknown>) => void;
  updateNodeLabel: (id: string, label: string) => void;
  requestDeleteNode: (id: string) => void;
  confirmDeleteNode: () => void;
  cancelDeleteNode: () => void;
  resetCanvas: () => void;

  toPipelineDefinition: () => PipelineDefinition;
  getRuntimeKwargs: () => Record<string, unknown>;

  getGroupOf: (id: string) => string[] | null;
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
  paletteMinimized: false,

  stagingMode: "new",
  stagingMethod: null,
  stagingCategory: null,
  stagingName: "",
  stagingArgs: {},

  pendingDeleteNodeId: null,
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

  setStagingMode: (mode) => {
    set({
      stagingMode: mode,
      stagingMethod: null,
      stagingCategory: null,
      stagingName: "",
      stagingArgs: {},
    });
  },
  setStagingMethod: (method) => {
    if (method === null) {
      set({ stagingMethod: null, stagingCategory: null, stagingName: "", stagingArgs: {} });
      return;
    }
    const spec = get().methods.find((m) => m.method === method);
    if (!spec) {
      set({ stagingMethod: method });
      return;
    }
    const defaults: Record<string, unknown> = {};
    for (const f of spec.args) {
      if (f.default !== undefined) defaults[f.name] = f.default;
    }
    set({
      stagingMethod: method,
      stagingCategory: spec.category,
      stagingName: method,
      stagingArgs: defaults,
    });
  },
  setStagingCategory: (category) => set({ stagingCategory: category }),
  setStagingName: (name) => set({ stagingName: name }),
  setStagingArgs: (args) => set({ stagingArgs: args }),
  resetStaging: () => {
    set({ stagingMethod: null, stagingCategory: null, stagingName: "", stagingArgs: {} });
  },

  addControlNode: (category, method, position, label) => {
    const id = uuid();
    const data: ControlNodeData = {
      category,
      method,
      args: {},
      runtimeKwargs: {},
      label: label && label.trim() ? label.trim() : method,
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
    const node: Node = {
      id,
      type: "dataset",
      position,
      data: { name: name && name.trim() ? name.trim() : "dataset", rowCount: null },
    };
    set({ nodes: [...get().nodes, node] });
    return id;
  },

  removeNode: (id) => {
    const node = get().nodes.find((n) => n.id === id);
    if (node && FIXTURE_NODE_TYPES.has(node.type ?? "")) return;
    set({
      nodes: get().nodes.filter((n) => n.id !== id),
      edges: get().edges.filter((e) => e.source !== id && e.target !== id),
      selectedNodeId: get().selectedNodeId === id ? null : get().selectedNodeId,
      lockedGroups: pruneLockedGroups(get().lockedGroups, id),
    });
  },

  removeEdge: (id) => {
    set({ edges: get().edges.filter((e) => e.id !== id) });
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

  requestDeleteNode: (id) => set({ pendingDeleteNodeId: id }),
  confirmDeleteNode: () => {
    const id = get().pendingDeleteNodeId;
    if (!id) return;
    const node = get().nodes.find((n) => n.id === id);
    if (node && FIXTURE_NODE_TYPES.has(node.type ?? "")) {
      set({ pendingDeleteNodeId: null });
      return;
    }
    set({
      nodes: get().nodes.filter((n) => n.id !== id),
      edges: get().edges.filter((e) => e.source !== id && e.target !== id),
      selectedNodeId: get().selectedNodeId === id ? null : get().selectedNodeId,
      pendingDeleteNodeId: null,
      lockedGroups: pruneLockedGroups(get().lockedGroups, id),
    });
  },
  cancelDeleteNode: () => set({ pendingDeleteNodeId: null }),

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
    });
  },

  getGroupOf: (id) => {
    const group = get().lockedGroups.find((g) => g.includes(id));
    return group ?? null;
  },
  mergeLockGroups: (idA, idB) => {
    const groups = get().lockedGroups;
    const groupA = groups.find((g) => g.includes(idA));
    const groupB = groups.find((g) => g.includes(idB));
    if (groupA && groupA === groupB) return;
    const members = new Set<string>();
    if (groupA) groupA.forEach((m) => members.add(m));
    else members.add(idA);
    if (groupB) groupB.forEach((m) => members.add(m));
    else members.add(idB);
    const next = groups.filter((g) => g !== groupA && g !== groupB);
    next.push(Array.from(members));
    set({ lockedGroups: next });
  },
  removeFromLockGroup: (id) => {
    set({ lockedGroups: pruneLockedGroups(get().lockedGroups, id) });
  },
  splitLockGroup: (leftIds, rightIds) => {
    const groups = get().lockedGroups;
    const all = new Set<string>([...leftIds, ...rightIds]);
    const containing = groups.find((g) => g.some((id) => all.has(id)));
    if (!containing) return;
    const next = groups.filter((g) => g !== containing);
    if (leftIds.length >= 2) next.push([...leftIds]);
    if (rightIds.length >= 2) next.push([...rightIds]);
    set({ lockedGroups: next });
  },

  toPipelineDefinition: () => {
    const controlNodes: ControlNode[] = get()
      .nodes.filter(isControlNode)
      .map((n) => {
        const data = n.data as ControlNodeData;
        return {
          id: n.id,
          category: data.category,
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
