import { usePipelineStore } from "../store/pipelineStore";

const HINTS: Record<string, string> = {
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
