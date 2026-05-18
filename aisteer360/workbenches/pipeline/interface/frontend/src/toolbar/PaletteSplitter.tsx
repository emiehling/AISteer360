import { useCallback, useRef, type PointerEvent } from "react";
import { usePipelineStore } from "../store/pipelineStore";

export function PaletteSplitter() {
  const paletteHeight = usePipelineStore((s) => s.paletteHeight);
  const setPaletteHeight = usePipelineStore((s) => s.setPaletteHeight);
  const startRef = useRef<{ y: number; height: number } | null>(null);

  const onPointerMove = useCallback(
    (event: globalThis.PointerEvent) => {
      const start = startRef.current;
      if (!start) return;
      const delta = event.clientY - start.y;
      setPaletteHeight(start.height - delta);
    },
    [setPaletteHeight],
  );

  const onPointerUp = useCallback(() => {
    startRef.current = null;
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", onPointerUp);
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }, [onPointerMove]);

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    startRef.current = { y: event.clientY, height: paletteHeight };
    document.body.style.cursor = "row-resize";
    document.body.style.userSelect = "none";
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
  };

  return (
    <div
      className="palette-splitter"
      role="separator"
      aria-orientation="horizontal"
      aria-label="Resize palette"
      onPointerDown={onPointerDown}
    >
      <div className="palette-splitter-grip" aria-hidden />
    </div>
  );
}
