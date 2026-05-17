import type { Connection, Edge } from "reactflow";

/** Universal connection rules.
 *  - No self-connections.
 *  - Connection must have a source and target node.
 *  - No duplicate edges (same source/target/handle pair).
 *  - Each target handle accepts at most one incoming edge.
 *
 * Category-arity constraints (one control per category) are intentionally NOT enforced here —
 * the backend's `merge_controls` rule will be relaxed in a separate change, and the user wants the
 * canvas to remain unconstrained until then.
 */
export function makeIsValidConnection(getEdges: () => Edge[]) {
  return (connection: Connection): boolean => {
    if (!connection.source || !connection.target) return false;
    if (connection.source === connection.target) return false;

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
        edge.target === connection.target &&
        (edge.targetHandle ?? null) === (connection.targetHandle ?? null)
      ) {
        return false;
      }
    }
    return true;
  };
}
