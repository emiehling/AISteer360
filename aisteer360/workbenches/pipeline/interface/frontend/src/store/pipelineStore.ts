import { create } from "zustand";
import {
  applyEdgeChanges,
  applyNodeChanges,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
} from "reactflow";
import type { PipelineDefinition } from "../types";

type NotImplemented = (...args: unknown[]) => never;

interface PipelineStoreState {
  nodes: Node[];
  edges: Edge[];
  modelNameOrPath: string;
  setNodes: (nodes: Node[]) => void;
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  setModelNameOrPath: (value: string) => void;

  addControlNode: NotImplemented;
  removeNode: NotImplemented;
  updateNodeArgs: NotImplemented;
  toPipelineDefinition: () => PipelineDefinition;
}

const notImplemented = (name: string): NotImplemented => {
  return (() => {
    throw new Error(`pipelineStore.${name} is not implemented in PR 1`);
  }) as NotImplemented;
};

export const usePipelineStore = create<PipelineStoreState>((set, get) => ({
  nodes: [],
  edges: [],
  modelNameOrPath: "",

  setNodes: (nodes) => set({ nodes }),
  onNodesChange: (changes) => {
    set({ nodes: applyNodeChanges(changes, get().nodes) });
  },
  onEdgesChange: (changes) => {
    set({ edges: applyEdgeChanges(changes, get().edges) });
  },
  setModelNameOrPath: (value) => set({ modelNameOrPath: value }),

  addControlNode: notImplemented("addControlNode"),
  removeNode: notImplemented("removeNode"),
  updateNodeArgs: notImplemented("updateNodeArgs"),

  toPipelineDefinition: () => ({
    model_name_or_path: get().modelNameOrPath,
    nodes: [],
  }),
}));
