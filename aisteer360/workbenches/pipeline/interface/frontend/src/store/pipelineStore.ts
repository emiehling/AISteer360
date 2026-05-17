import { create } from "zustand";
import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
  type XYPosition,
} from "reactflow";
import type {
  ControlCategory,
  ControlNode,
  ControlNodeData,
  MethodSpec,
  PipelineDefinition,
  ToolMode,
} from "../types";

const FIXTURE_NODE_TYPES = new Set(["prompt_anchor", "response_anchor", "target_model"]);

function isControlNode(n: Node): boolean {
  return n.type === "control";
}

function uuid(): string {
  return crypto.randomUUID();
}

interface PipelineStoreState {
  nodes: Node[];
  edges: Edge[];
  modelNameOrPath: string;
  methods: MethodSpec[];
  selectedNodeId: string | null;
  activeTool: ToolMode;
  sessionId: string | null;

  setNodes: (nodes: Node[]) => void;
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;
  setModelNameOrPath: (value: string) => void;
  setMethods: (methods: MethodSpec[]) => void;
  setSelectedNodeId: (id: string | null) => void;
  setActiveTool: (mode: ToolMode) => void;
  setSessionId: (id: string | null) => void;

  addControlNode: (category: ControlCategory, method: string, position: XYPosition) => string;
  removeNode: (id: string) => void;
  removeEdge: (id: string) => void;
  updateNodeArgs: (id: string, args: Record<string, unknown>) => void;
  updateNodeRuntimeKwargs: (id: string, kwargs: Record<string, unknown>) => void;
  resetCanvas: () => void;

  toPipelineDefinition: () => PipelineDefinition;
  getRuntimeKwargs: () => Record<string, unknown>;
}

export const usePipelineStore = create<PipelineStoreState>((set, get) => ({
  nodes: [],
  edges: [],
  modelNameOrPath: "",
  methods: [],
  selectedNodeId: null,
  activeTool: "select",
  sessionId: null,

  setNodes: (nodes) => set({ nodes }),
  onNodesChange: (changes) => {
    set({ nodes: applyNodeChanges(changes, get().nodes) });
  },
  onEdgesChange: (changes) => {
    set({ edges: applyEdgeChanges(changes, get().edges) });
  },
  onConnect: (connection) => {
    set({
      edges: addEdge({ ...connection, type: "pipeline" }, get().edges),
    });
  },
  setModelNameOrPath: (value) => set({ modelNameOrPath: value }),
  setMethods: (methods) => set({ methods }),
  setSelectedNodeId: (id) => set({ selectedNodeId: id }),
  setActiveTool: (mode) => set({ activeTool: mode }),
  setSessionId: (id) => set({ sessionId: id }),

  addControlNode: (category, method, position) => {
    const id = uuid();
    const data: ControlNodeData = {
      category,
      method,
      args: {},
      runtimeKwargs: {},
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

  removeNode: (id) => {
    const node = get().nodes.find((n) => n.id === id);
    if (node && FIXTURE_NODE_TYPES.has(node.type ?? "")) return;
    set({
      nodes: get().nodes.filter((n) => n.id !== id),
      edges: get().edges.filter((e) => e.source !== id && e.target !== id),
      selectedNodeId: get().selectedNodeId === id ? null : get().selectedNodeId,
    });
  },

  removeEdge: (id) => {
    set({ edges: get().edges.filter((e) => e.id !== id) });
  },

  updateNodeArgs: (id, args) => {
    set({
      nodes: get().nodes.map((n) =>
        n.id === id && isControlNode(n)
          ? { ...n, data: { ...(n.data as ControlNodeData), args: { ...(n.data as ControlNodeData).args, ...args } } }
          : n,
      ),
    });
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
    });
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
