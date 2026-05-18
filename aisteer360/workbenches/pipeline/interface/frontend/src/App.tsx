import { useEffect } from "react";
import { fetchCatalog } from "./api/catalog";
import { fetchMethods } from "./api/methods";
import { LibraryPanel } from "./panels/LibraryPanel";
import { ParameterPanel } from "./panels/ParameterPanel";
import { PipelineCanvas } from "./PipelineCanvas";
import { CanvasToolbar } from "./toolbar/CanvasToolbar";
import { HintBar } from "./toolbar/HintBar";
import { SessionStub } from "./session/SessionStub";
import { usePipelineStore } from "./store/pipelineStore";

export function App() {
  const setMethods = usePipelineStore((s) => s.setMethods);
  const setCatalogTargetEntries = usePipelineStore((s) => s.setCatalogTargetEntries);
  const setModelNameOrPath = usePipelineStore((s) => s.setModelNameOrPath);
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

  return (
    <div className="canvas-region">
      <div className="canvas-and-panel">
        <div className="canvas-stack">
          <PipelineCanvas />
          <CanvasToolbar />
          <HintBar />
        </div>
        <ParameterPanel />
      </div>
      <LibraryPanel />
      <SessionStub />
    </div>
  );
}
