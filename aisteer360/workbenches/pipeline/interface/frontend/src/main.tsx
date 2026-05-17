import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";

const MOUNT_ID = "pipeline-canvas-root";

const container = document.getElementById(MOUNT_ID);
if (!container) {
  throw new Error(`Pipeline canvas: missing #${MOUNT_ID} mount point`);
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
