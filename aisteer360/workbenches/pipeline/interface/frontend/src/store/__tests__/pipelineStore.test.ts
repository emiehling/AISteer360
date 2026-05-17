import { beforeEach, describe, expect, it } from "vitest";
import { usePipelineStore } from "../pipelineStore";
import type { Edge, Node } from "reactflow";
import type { ControlNodeData } from "../../types";

function resetStore() {
  usePipelineStore.setState({
    nodes: [],
    edges: [],
    modelNameOrPath: "test-model",
    methods: [],
    selectedNodeId: null,
    activeTool: "select",
    sessionId: null,
  });
}

function seedFixtures() {
  const fixtures: Node[] = [
    { id: "anchor-prompt", type: "prompt_anchor", position: { x: 0, y: 0 }, data: {} },
    { id: "model", type: "target_model", position: { x: 100, y: 0 }, data: {} },
    { id: "anchor-response", type: "response_anchor", position: { x: 200, y: 0 }, data: {} },
  ];
  usePipelineStore.setState({ nodes: fixtures });
}

describe("pipelineStore", () => {
  beforeEach(() => {
    resetStore();
  });

  it("addControlNode creates a control node with empty args/runtimeKwargs", () => {
    const id = usePipelineStore.getState().addControlNode("state_control", "pasta", { x: 50, y: 60 });
    const node = usePipelineStore.getState().nodes.find((n) => n.id === id)!;
    expect(node).toBeDefined();
    expect(node.type).toBe("control");
    expect((node.data as ControlNodeData).category).toBe("state_control");
    expect((node.data as ControlNodeData).method).toBe("pasta");
    expect((node.data as ControlNodeData).args).toEqual({});
    expect((node.data as ControlNodeData).runtimeKwargs).toEqual({});
    expect(node.position).toEqual({ x: 50, y: 60 });
  });

  it("toPipelineDefinition skips fixture nodes and includes control nodes only", () => {
    seedFixtures();
    usePipelineStore.getState().addControlNode("input_control", "few_shot", { x: 30, y: 40 });
    const definition = usePipelineStore.getState().toPipelineDefinition();
    expect(definition.model_name_or_path).toBe("test-model");
    expect(definition.nodes).toHaveLength(1);
    expect(definition.nodes[0].method).toBe("few_shot");
    expect(definition.nodes[0].position).toEqual([30, 40]);
  });

  it("getRuntimeKwargs merges across all control nodes (last write wins)", () => {
    const a = usePipelineStore.getState().addControlNode("state_control", "pasta", { x: 0, y: 0 });
    const b = usePipelineStore.getState().addControlNode("input_control", "few_shot", { x: 0, y: 0 });
    usePipelineStore.getState().updateNodeRuntimeKwargs(a, { substrings: ["foo"] });
    usePipelineStore.getState().updateNodeRuntimeKwargs(b, { substrings: ["bar"], extra: 1 });
    const merged = usePipelineStore.getState().getRuntimeKwargs();
    expect(merged.extra).toBe(1);
    // last write wins — depends on iteration order, but with a < b in array it's b's value
    expect(merged.substrings).toEqual(["bar"]);
  });

  it("updateNodeArgs shallow-merges into existing args", () => {
    const id = usePipelineStore.getState().addControlNode("state_control", "pasta", { x: 0, y: 0 });
    usePipelineStore.getState().updateNodeArgs(id, { alpha: 1.5 });
    usePipelineStore.getState().updateNodeArgs(id, { scale_position: "include" });
    const node = usePipelineStore.getState().nodes.find((n) => n.id === id)!;
    expect((node.data as ControlNodeData).args).toEqual({ alpha: 1.5, scale_position: "include" });
  });

  it("removeNode removes the node and incident edges", () => {
    seedFixtures();
    const id = usePipelineStore.getState().addControlNode("state_control", "pasta", { x: 0, y: 0 });
    const edges: Edge[] = [
      { id: "e1", source: id, target: "model", type: "pipeline" },
      { id: "e2", source: "anchor-prompt", target: "model", type: "pipeline" },
    ];
    usePipelineStore.setState({ edges });
    usePipelineStore.getState().removeNode(id);
    expect(usePipelineStore.getState().nodes.find((n) => n.id === id)).toBeUndefined();
    expect(usePipelineStore.getState().edges.map((e) => e.id)).toEqual(["e2"]);
  });

  it("removeNode is a no-op for fixture nodes", () => {
    seedFixtures();
    usePipelineStore.getState().removeNode("model");
    expect(usePipelineStore.getState().nodes.map((n) => n.id)).toContain("model");
  });

  it("resetCanvas keeps fixtures and clears edges", () => {
    seedFixtures();
    const a = usePipelineStore.getState().addControlNode("state_control", "pasta", { x: 0, y: 0 });
    usePipelineStore.setState({
      edges: [{ id: "e1", source: a, target: "model", type: "pipeline" }],
    });
    usePipelineStore.getState().resetCanvas();
    expect(usePipelineStore.getState().nodes.map((n) => n.id).sort()).toEqual([
      "anchor-prompt",
      "anchor-response",
      "model",
    ]);
    expect(usePipelineStore.getState().edges).toHaveLength(0);
  });
});
