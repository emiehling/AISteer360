import { usePipelineStore } from "../store/pipelineStore";

export function ConfirmDialog() {
  const pendingDeleteNodeId = usePipelineStore((s) => s.pendingDeleteNodeId);
  const confirmDeleteNode = usePipelineStore((s) => s.confirmDeleteNode);
  const cancelDeleteNode = usePipelineStore((s) => s.cancelDeleteNode);

  if (!pendingDeleteNodeId) return null;

  return (
    <div
      className="session-modal-scrim"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) cancelDeleteNode();
      }}
    >
      <div className="session-modal confirm-modal" role="dialog" aria-modal>
        <div className="session-modal-head">
          <span className="title">Delete</span>
          <button
            type="button"
            className="close"
            aria-label="cancel"
            onClick={cancelDeleteNode}
          >
            ×
          </button>
        </div>
        <div className="session-modal-body">
          <div className="confirm-message">
            Are you sure you wish to delete?
          </div>
          <div className="confirm-actions">
            <button
              type="button"
              className="session-btn"
              onClick={cancelDeleteNode}
              autoFocus
            >
              No
            </button>
            <button
              type="button"
              className="session-btn primary"
              onClick={confirmDeleteNode}
            >
              Yes
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
