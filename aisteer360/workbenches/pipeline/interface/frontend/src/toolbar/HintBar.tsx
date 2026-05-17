import { usePipelineStore } from "../store/pipelineStore";

const HINTS: Record<string, string> = {
  select: "drag controls from below; click a node to edit its parameters",
  connect: "drag from a port to another port to wire them together",
  erase: "click an edge to remove it",
};

export function HintBar() {
  const activeTool = usePipelineStore((s) => s.activeTool);
  const message = HINTS[activeTool];
  return (
    <div className={`hint-bar${message ? " visible" : ""}`} role="status">
      {message}
    </div>
  );
}
