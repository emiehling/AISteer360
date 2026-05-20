import { describe, expect, it } from "vitest";
import type { Edge, Node } from "reactflow";
import { makeIsValidConnection } from "../validation";

const noEdges = (): Edge[] => [];
const noNodes = (): Node[] => [];

const node = (id: string, type: string): Node => ({
  id,
  type,
  position: { x: 0, y: 0 },
  data: {},
});

describe("makeIsValidConnection", () => {
  it("rejects self-connections", () => {
    const isValid = makeIsValidConnection(noEdges, noNodes);
    expect(
      isValid({ source: "n1", target: "n1", sourceHandle: "out", targetHandle: "in" }),
    ).toBe(false);
  });

  it("requires source and target", () => {
    const isValid = makeIsValidConnection(noEdges, noNodes);
    expect(isValid({ source: null, target: "n1", sourceHandle: null, targetHandle: null })).toBe(false);
    expect(isValid({ source: "n1", target: null, sourceHandle: null, targetHandle: null })).toBe(false);
  });

  it("rejects duplicate edges", () => {
    const existing: Edge[] = [
      { id: "e1", source: "a", target: "b", sourceHandle: "out", targetHandle: "in", type: "pipeline" },
    ];
    const isValid = makeIsValidConnection(() => existing, noNodes);
    expect(
      isValid({ source: "a", target: "b", sourceHandle: "out", targetHandle: "in" }),
    ).toBe(false);
  });

  it("rejects multiple incoming edges to the same control target handle", () => {
    const existing: Edge[] = [
      { id: "e1", source: "a", target: "b", sourceHandle: "out", targetHandle: "in", type: "pipeline" },
    ];
    const nodes = (): Node[] => [node("b", "control")];
    const isValid = makeIsValidConnection(() => existing, nodes);
    expect(
      isValid({ source: "x", target: "b", sourceHandle: "out", targetHandle: "in" }),
    ).toBe(false);
  });

  it("allows multiple incoming edges to the same model target handle", () => {
    const existing: Edge[] = [
      { id: "e1", source: "a", target: "m", sourceHandle: "out", targetHandle: "left", type: "pipeline" },
    ];
    const nodes = (): Node[] => [node("m", "model")];
    const isValid = makeIsValidConnection(() => existing, nodes);
    expect(
      isValid({ source: "x", target: "m", sourceHandle: "out", targetHandle: "left" }),
    ).toBe(true);
  });

  it("still rejects exact duplicate edges to a model target", () => {
    const existing: Edge[] = [
      { id: "e1", source: "a", target: "m", sourceHandle: "out", targetHandle: "left", type: "pipeline" },
    ];
    const nodes = (): Node[] => [node("m", "model")];
    const isValid = makeIsValidConnection(() => existing, nodes);
    expect(
      isValid({ source: "a", target: "m", sourceHandle: "out", targetHandle: "left" }),
    ).toBe(false);
  });

  it("accepts a valid new connection", () => {
    const isValid = makeIsValidConnection(noEdges, noNodes);
    expect(
      isValid({ source: "a", target: "b", sourceHandle: "out", targetHandle: "in" }),
    ).toBe(true);
  });
});
