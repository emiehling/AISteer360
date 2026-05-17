import { describe, expect, it } from "vitest";
import type { Edge } from "reactflow";
import { makeIsValidConnection } from "../validation";

const noEdges = (): Edge[] => [];

describe("makeIsValidConnection", () => {
  it("rejects self-connections", () => {
    const isValid = makeIsValidConnection(noEdges);
    expect(
      isValid({ source: "n1", target: "n1", sourceHandle: "out", targetHandle: "in" }),
    ).toBe(false);
  });

  it("requires source and target", () => {
    const isValid = makeIsValidConnection(noEdges);
    expect(isValid({ source: null, target: "n1", sourceHandle: null, targetHandle: null })).toBe(false);
    expect(isValid({ source: "n1", target: null, sourceHandle: null, targetHandle: null })).toBe(false);
  });

  it("rejects duplicate edges", () => {
    const existing: Edge[] = [
      { id: "e1", source: "a", target: "b", sourceHandle: "out", targetHandle: "in", type: "pipeline" },
    ];
    const isValid = makeIsValidConnection(() => existing);
    expect(
      isValid({ source: "a", target: "b", sourceHandle: "out", targetHandle: "in" }),
    ).toBe(false);
  });

  it("rejects multiple incoming edges to the same target handle", () => {
    const existing: Edge[] = [
      { id: "e1", source: "a", target: "b", sourceHandle: "out", targetHandle: "in", type: "pipeline" },
    ];
    const isValid = makeIsValidConnection(() => existing);
    expect(
      isValid({ source: "x", target: "b", sourceHandle: "out", targetHandle: "in" }),
    ).toBe(false);
  });

  it("accepts a valid new connection", () => {
    const isValid = makeIsValidConnection(noEdges);
    expect(
      isValid({ source: "a", target: "b", sourceHandle: "out", targetHandle: "in" }),
    ).toBe(true);
  });
});
