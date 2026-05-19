import { useEffect, useState } from "react";
import { usePipelineStore } from "../store/pipelineStore";

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

  const label = placement.kind;

  return (
    <div
      className="placement-ghost"
      data-kind={placement.kind}
      style={{ left: pos.x, top: pos.y }}
    >
      <span className="placement-ghost-label">{label}</span>
    </div>
  );
}
