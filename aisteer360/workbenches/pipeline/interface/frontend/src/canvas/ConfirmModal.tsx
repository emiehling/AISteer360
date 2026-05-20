import { usePipelineStore } from "../store/pipelineStore";

export function ConfirmModal() {
  const pending = usePipelineStore((s) => s.pendingConfirm);
  const resolveConfirm = usePipelineStore((s) => s.resolveConfirm);
  const cancelConfirm = usePipelineStore((s) => s.cancelConfirm);

  if (!pending) return null;

  const confirmLabel = pending.confirmLabel ?? "Yes";
  const cancelLabel = pending.cancelLabel ?? "No";

  return (
    <div
      className="session-modal-scrim"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) cancelConfirm();
      }}
    >
      <div className="session-modal confirm-modal" role="dialog" aria-modal>
        <div className="session-modal-head">
          <span className="title">{pending.title}</span>
          <button
            type="button"
            className="close"
            aria-label="cancel"
            onClick={cancelConfirm}
          >
            ×
          </button>
        </div>
        <div className="session-modal-body">
          <div className="confirm-message">{pending.message}</div>
          <div className="confirm-actions">
            <button
              type="button"
              className="session-btn"
              onClick={cancelConfirm}
              autoFocus
            >
              {cancelLabel}
            </button>
            <button
              type="button"
              className="session-btn primary"
              onClick={resolveConfirm}
            >
              {confirmLabel}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
