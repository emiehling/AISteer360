import { useEffect, useState } from "react";
import { createSession, getSession, infer } from "../api/sessions";
import { usePipelineStore } from "../store/pipelineStore";

const OPEN_EVENT = "aisteer-open-session";
const STATUS_EVENT = "aisteer-status";

type Phase = "idle" | "creating" | "ready" | "infering" | "error";

function setChromeStatus(state: string, label: string) {
  window.dispatchEvent(new CustomEvent(STATUS_EVENT, { detail: { state, label } }));
}

export function SessionStub() {
  const [open, setOpen] = useState(false);
  const [model, setModel] = useState("sshleifer/tiny-gpt2");
  const [prompt, setPrompt] = useState("Hello, world.");
  const [output, setOutput] = useState<string>("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);

  const sessionId = usePipelineStore((s) => s.sessionId);
  const setSessionId = usePipelineStore((s) => s.setSessionId);
  const setModelName = usePipelineStore((s) => s.setModelNameOrPath);
  const toPipelineDefinition = usePipelineStore((s) => s.toPipelineDefinition);
  const getRuntimeKwargs = usePipelineStore((s) => s.getRuntimeKwargs);

  useEffect(() => {
    const onOpen = () => setOpen(true);
    window.addEventListener(OPEN_EVENT, onOpen);
    return () => window.removeEventListener(OPEN_EVENT, onOpen);
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const detail = await getSession(sessionId);
        if (cancelled) return;
        setChromeStatus(detail.status === "ready" ? "complete" : detail.status, detail.status);
        if (detail.status === "ready" && phase === "creating") setPhase("ready");
        if (detail.error) {
          setError(detail.error);
          setPhase("error");
        }
      } catch (e) {
        if (cancelled) return;
        console.warn("session poll failed", e);
      }
    };
    const interval = setInterval(tick, 1500);
    void tick();
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [sessionId, phase]);

  const onCreate = async () => {
    setError(null);
    setPhase("creating");
    setChromeStatus("busy", "creating session");
    try {
      const resp = await createSession(model.trim());
      setSessionId(resp.session.id);
      setModelName(resp.session.model_name);
      if (resp.dispatch_status === "manual") {
        setError(`Manual agent dispatch required. Run: ${resp.agent_command.command}`);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase("error");
      setChromeStatus("error", "session error");
    }
  };

  const onInfer = async () => {
    if (!sessionId) return;
    setError(null);
    setPhase("infering");
    setOutput("");
    setChromeStatus("busy", "generating");
    try {
      const definition = toPipelineDefinition();
      const runtime_kwargs = getRuntimeKwargs();
      await infer(sessionId, {
        pipeline: definition,
        prompt,
        runtime_kwargs,
        gen_kwargs: { max_new_tokens: 32 },
      });
      setOutput("Inference accepted; result will arrive over the WebSocket relay (not yet wired in this stub).");
      setPhase("ready");
      setChromeStatus("complete", "request accepted");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase("error");
      setChromeStatus("error", "infer failed");
    }
  };

  if (!open) return null;
  return (
    <div className="session-modal-scrim" onClick={() => setOpen(false)}>
      <div className="session-modal" onClick={(e) => e.stopPropagation()}>
        <header className="session-modal-head">
          <span className="title">Session</span>
          <button className="close" onClick={() => setOpen(false)} aria-label="Close">
            ×
          </button>
        </header>
        <div className="session-modal-body">
          <label className="session-field">
            <span>HuggingFace model</span>
            <input
              className="widget-input"
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={Boolean(sessionId)}
            />
          </label>
          {!sessionId ? (
            <button
              className="session-btn primary"
              type="button"
              onClick={onCreate}
              disabled={phase === "creating"}
            >
              {phase === "creating" ? "Creating…" : "Create session"}
            </button>
          ) : (
            <>
              <div className="session-info">
                <span className="session-info-label">session</span>
                <code>{sessionId.slice(0, 12)}…</code>
              </div>
              <label className="session-field">
                <span>Prompt</span>
                <textarea
                  className="widget-textarea"
                  rows={3}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                />
              </label>
              <button
                className="session-btn primary"
                type="button"
                onClick={onInfer}
                disabled={phase === "infering"}
              >
                {phase === "infering" ? "Inferring…" : "Infer"}
              </button>
              {output && <div className="session-output">{output}</div>}
            </>
          )}
          {error && <div className="session-error">{error}</div>}
        </div>
      </div>
    </div>
  );
}
