import type { Connection, Edge, Node } from "reactflow";

/** Universal connection rules.
 *  - No self-connections.
 *  - Connection must have a source and target node.
 *  - No duplicate edges (same source/target/handle pair).
 *  - Each target handle accepts at most one incoming edge, EXCEPT when the target node is a
 *    model node — model handles act as fan-in points for multiple controls/datasets.
 *
 * Category-arity constraints (one control per category) are intentionally NOT enforced here —
 * the backend's `merge_controls` rule will be relaxed in a separate change, and the user wants the
 * canvas to remain unconstrained until then.
 */
export function makeIsValidConnection(getEdges: () => Edge[], getNodes: () => Node[]) {
  return (connection: Connection): boolean => {
    if (!connection.source || !connection.target) return false;
    if (connection.source === connection.target) return false;

    const targetNode = getNodes().find((n) => n.id === connection.target);
    const targetIsModel = targetNode?.type === "model";

    const edges = getEdges();
    for (const edge of edges) {
      if (
        edge.source === connection.source &&
        edge.target === connection.target &&
        (edge.sourceHandle ?? null) === (connection.sourceHandle ?? null) &&
        (edge.targetHandle ?? null) === (connection.targetHandle ?? null)
      ) {
        return false;
      }
      if (
        !targetIsModel &&
        edge.target === connection.target &&
        (edge.targetHandle ?? null) === (connection.targetHandle ?? null)
      ) {
        return false;
      }
    }
    return true;
  };
}
