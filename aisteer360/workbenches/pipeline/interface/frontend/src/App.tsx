import { useEffect } from "react";
import { fetchCatalog } from "./api/catalog";
import { fetchMethods } from "./api/methods";
import { LibraryPanel } from "./panels/LibraryPanel";
import { ParameterPanel } from "./panels/ParameterPanel";
import { PipelineCanvas } from "./PipelineCanvas";
import { CanvasToolbar } from "./toolbar/CanvasToolbar";
import { HintBar } from "./toolbar/HintBar";
import { PaletteSplitter } from "./toolbar/PaletteSplitter";
import { PlacementGhost } from "./toolbar/PlacementGhost";
import { SessionStub } from "./session/SessionStub";
import { usePipelineStore } from "./store/pipelineStore";

const PALETTE_MINIMIZED_HEIGHT = 24;

export function App() {
  const setMethods = usePipelineStore((s) => s.setMethods);
  const setCatalogTargetEntries = usePipelineStore((s) => s.setCatalogTargetEntries);
  const setModelNameOrPath = usePipelineStore((s) => s.setModelNameOrPath);
  const paletteHeight = usePipelineStore((s) => s.paletteHeight);
  const paletteMinimized = usePipelineStore((s) => s.paletteMinimized);
  const togglePaletteMinimized = usePipelineStore((s) => s.togglePaletteMinimized);
  useEffect(() => {
    fetchMethods()
      .then(setMethods)
      .catch((err) => console.error("fetchMethods failed:", err));

    fetchCatalog()
      .then((entries) => {
        const targets = entries
          .filter((e) => Array.isArray(e.roles) && e.roles.includes("target"))
          .map((e) => ({ label: e.label, model_id: e.model_id }));
        setCatalogTargetEntries(targets);
        if (targets.length > 0 && !usePipelineStore.getState().modelNameOrPath) {
          setModelNameOrPath(targets[0].model_id);
        }
      })
      .catch((err) => console.error("fetchCatalog failed:", err));
  }, [setMethods, setCatalogTargetEntries, setModelNameOrPath]);

  const effectiveHeight = paletteMinimized ? PALETTE_MINIMIZED_HEIGHT : paletteHeight;

  return (
    <div className="canvas-region">
      <div className="canvas-stack">
        <PipelineCanvas />
        <CanvasToolbar />
        <HintBar />
      </div>
      {!paletteMinimized && <PaletteSplitter />}
      <div
        className={`palette${paletteMinimized ? " minimized" : ""}`}
        style={{ height: effectiveHeight }}
      >
        {paletteMinimized ? (
          <button
            type="button"
            className="palette-minimized-bar"
            onClick={togglePaletteMinimized}
            aria-label="Expand palette"
            title="expand palette"
          >
            <svg className="palette-minimized-chevron" width="32" height="14" viewBox="0 0 32 14" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <polyline points="4 11 16 3 28 11" />
            </svg>
          </button>
        ) : (
          <>
            <LibraryPanel />
            <ParameterPanel />
            <button
              type="button"
              className="palette-toggle-btn palette-toggle-btn-floating"
              onClick={togglePaletteMinimized}
              aria-label="Minimize palette"
              title="minimize palette"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </button>
          </>
        )}
      </div>
      <SessionStub />
      <PlacementGhost />
    </div>
  );
}
