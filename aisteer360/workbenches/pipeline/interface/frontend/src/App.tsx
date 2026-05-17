import { useEffect } from "react";
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
  useEffect(() => {
    fetchMethods()
      .then(setMethods)
      .catch((err) => console.error("fetchMethods failed:", err));
  }, [setMethods]);

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
