import { usePipelineStore } from "../store/pipelineStore";

const PLACEMENT_HINTS: Record<string, string> = {
  control: "click on canvas to place control · esc to cancel",
  dataset: "click on canvas to place dataset · esc to cancel",
  model: "click on canvas to place model · esc to cancel",
};

export function HintBar() {
  const placement = usePipelineStore((s) => s.placement);
  const message = placement ? PLACEMENT_HINTS[placement.kind] : "";
  return (
    <div className={`hint-bar${message ? " visible" : ""}`} role="status">
      {message}
    </div>
  );
}
