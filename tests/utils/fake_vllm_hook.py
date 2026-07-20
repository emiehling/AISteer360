"""In-process fake of an OpenAI-compatible (optionally vLLM-hook) server for CPU tests.

A thread-hosted HTTP app that *executes* request semantics against a tiny hub-free model, so
generation, scoring, capture, and steering return real numbers. Two profiles:

- ``plain`` — ignores hook args entirely (models a plugin-less server); used by the OpenAI backend's
  capability/generation/scoring tests.
- ``hook`` — implements the doc-07 steering schema (added incrementally); it is the executable form
  of the wire schema.

Fidelity boundary: the fake proves compiler/schema/backend *semantics*, not vLLM numerics or
scheduling. Any wording change in doc 07 must land in the ``hook`` profile in the same commit.

Introspection for assertions: ``server.requests`` (ordered decoded bodies), ``server.max_in_flight``,
fault injection (``respond_429_times``, ``always_500``), and a template-mismatch mode.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch


class FakeVLLMHookServer:
    """A thread-hosted fake OpenAI-compatible server backed by a tiny model.

    Args:
        model: A tiny causal LM (e.g. `tests.utils.tiny_models.tiny_llama`).
        tokenizer: Its tokenizer.
        profile: ``"plain"`` or ``"hook"``.
        support_token_arrays: When False, token-array completion prompts are rejected (to exercise
            capability degradation).
        template_mismatch: When True, chat renders prepend a token so `usage.prompt_tokens` diverges
            from the client's local count (drives the parity-warning test).
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        profile: str = "plain",
        support_token_arrays: bool = True,
        template_mismatch: bool = False,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.profile = profile
        self.support_token_arrays = support_token_arrays
        self.template_mismatch = template_mismatch

        self.requests: list[dict] = []
        self.max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()
        self._force_429 = 0
        self._always_500_paths: set[str] = set()

        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.base_url: str | None = None

    # fault injection

    def respond_429_times(self, n: int) -> None:
        """Return HTTP 429 for the next `n` requests, then behave normally."""
        self._force_429 = n

    def always_500(self, path: str) -> None:
        """Always return HTTP 500 for requests to `path`."""
        self._always_500_paths.add(path)

    # lifecycle

    def __enter__(self) -> "FakeVLLMHookServer":
        handler = _make_handler(self)
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        host, port = self._server.server_address
        self.base_url = f"http://{host}:{port}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    # concurrency bookkeeping (used by the max-in-flight assertion)

    def _enter_flight(self) -> None:
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)

    def _exit_flight(self) -> None:
        with self._lock:
            self._in_flight -= 1

    # request handlers (real numbers from the tiny model)

    def handle(self, path: str, body: dict) -> tuple[int, dict]:
        """Dispatch one decoded request to its handler, returning `(status, json_body)`."""
        with self._lock:
            self.requests.append({"path": path, "body": body})
            if path in self._always_500_paths:
                return 500, {"error": "always_500"}
            if self._force_429 > 0:
                self._force_429 -= 1
                return 429, {"error": "rate_limited"}

        self._enter_flight()
        try:
            # small window so concurrent requests actually overlap for the in-flight assertion
            import time as _time

            _time.sleep(0.02)
            if path.endswith("/models"):
                return 200, {"object": "list", "data": [{"id": "fake-model", "object": "model"}]}
            if path.endswith("/chat/completions"):
                return 200, self._chat_completion(body)
            if path.endswith("/completions"):
                return 200, self._completion(body)
            return 404, {"error": "not_found"}
        finally:
            self._exit_flight()

    def _chat_completion(self, body: dict) -> dict:
        messages = body["messages"]
        rendered = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        prompt_ids = list(rendered)
        if self.template_mismatch:
            prompt_ids = [prompt_ids[0], *prompt_ids]  # diverge from the client's local count
        text, completion_ids, finish = self._greedy(prompt_ids, body)
        return self._completion_envelope(text, prompt_ids, completion_ids, finish, chat=True)

    def _completion(self, body: dict) -> dict:
        prompt = body["prompt"]
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], list):
            prompt = prompt[0]  # single-row token-array
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            if not self.support_token_arrays:
                raise _HTTPError(400, "token-array prompts not supported")
            prompt_ids = list(prompt)
        elif isinstance(prompt, list) and not prompt:
            prompt_ids = []
        else:
            prompt_ids = self.tokenizer(str(prompt), add_special_tokens=True)["input_ids"]

        extra = body.get("prompt_logprobs")
        max_tokens = int(body.get("max_tokens", 16) or 0)
        prompt_logprobs = self._prompt_logprobs(prompt_ids) if extra is not None else None

        if max_tokens == 0:
            envelope = self._completion_envelope("", prompt_ids, [], "length", chat=False)
        else:
            text, completion_ids, finish = self._greedy(prompt_ids, body)
            envelope = self._completion_envelope(text, prompt_ids, completion_ids, finish, chat=False)
        if prompt_logprobs is not None:
            envelope["choices"][0]["prompt_logprobs"] = prompt_logprobs
        return envelope

    def _greedy(self, prompt_ids: list[int], body: dict) -> tuple[str, list[int], str]:
        """Deterministic greedy decode of up to `max_tokens` continuation tokens."""
        max_tokens = int(body.get("max_tokens", 8) or 8)
        ids = torch.tensor([prompt_ids], dtype=torch.long)
        generated: list[int] = []
        eos = self.tokenizer.eos_token_id
        finish = "length"
        with torch.no_grad():
            for _ in range(max_tokens):
                logits = self.model(input_ids=ids).logits[:, -1, :]
                nxt = int(logits.argmax(dim=-1).item())
                generated.append(nxt)
                ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
                if eos is not None and nxt == eos:
                    finish = "stop"
                    break
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text, generated, finish

    def _prompt_logprobs(self, prompt_ids: list[int]) -> list:
        """Teacher-forced per-position logprob of each realized token (entry 0 is None)."""
        if len(prompt_ids) < 2:
            return [None] * len(prompt_ids)
        ids = torch.tensor([prompt_ids], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(input_ids=ids).logits[0]  # [T, V]
        logprobs = torch.log_softmax(logits, dim=-1)
        out: list = [None]  # position 0 has no preceding context
        for pos in range(1, len(prompt_ids)):
            token = prompt_ids[pos]
            lp = float(logprobs[pos - 1, token])
            out.append({str(token): {"logprob": lp, "rank": 1}})
        return out

    def _completion_envelope(self, text, prompt_ids, completion_ids, finish, chat: bool) -> dict:
        choice = {"index": 0, "finish_reason": finish}
        if chat:
            choice["message"] = {"role": "assistant", "content": text}
        else:
            choice["text"] = text
        return {
            "id": "fake-cmpl",
            "object": "chat.completion" if chat else "text_completion",
            "model": "fake-model",
            "choices": [choice],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(completion_ids),
                "total_tokens": len(prompt_ids) + len(completion_ids),
            },
        }


class _HTTPError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _make_handler(server: FakeVLLMHookServer):
    """Build a BaseHTTPRequestHandler bound to a `FakeVLLMHookServer`."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence stderr access logs
            return

        def _read_body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw or b"{}")

        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            try:
                status, payload = server.handle(self.path, {})
            except _HTTPError as exc:
                status, payload = exc.status, {"error": exc.message}
            self._send(status, payload)

        def do_POST(self) -> None:
            try:
                body = self._read_body()
                status, payload = server.handle(self.path, body)
            except _HTTPError as exc:
                status, payload = exc.status, {"error": exc.message}
            except Exception as exc:  # noqa: BLE001
                status, payload = 500, {"error": str(exc)}
            self._send(status, payload)

    return Handler
