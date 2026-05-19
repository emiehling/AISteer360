import { useEffect, useState } from "react";
import { usePipelineStore } from "../store/pipelineStore";

const CATEGORY_LABEL: Record<string, string> = {
  input_control: "input control",
  structural_control: "structural control",
  state_control: "state control",
  output_control: "output control",
};

export function PlacementGhost() {
  const placement = usePipelineStore((s) => s.placement);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);

  useEffect(() => {
    if (!placement) {
      setPos(null);
      return;
    }
    const onMove = (event: MouseEvent) => {
      setPos({ x: event.clientX, y: event.clientY });
    };
    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, [placement]);

  if (!placement || !pos) return null;

  let label = "";
  let categoryAttr: string | undefined;
  if (placement.kind === "control") {
    label = CATEGORY_LABEL[placement.category] ?? "control";
    categoryAttr = placement.category;
  } else if (placement.kind === "dataset") {
    label = "dataset";
  } else if (placement.kind === "model") {
    label = "model";
  }

  return (
    <div
      className="placement-ghost"
      data-kind={placement.kind}
      data-category={categoryAttr}
      style={{ left: pos.x, top: pos.y }}
    >
      <span className="placement-ghost-label">{label}</span>
    </div>
  );
}
