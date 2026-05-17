import { LibraryPanel } from "./panels/LibraryPanel";
import { PipelineCanvas } from "./PipelineCanvas";

export function App() {
  return (
    <div className="canvas-region">
      <PipelineCanvas />
      <LibraryPanel />
    </div>
  );
}
