"""`HuggingFaceSession`: an exclusive in-process steering session.

Registers each state-control entry's hooks on `__enter__` (with the partial-registration rollback
that formerly lived on `StateControl.register_hooks`) and removes them on `__exit__`, so steering is
in force for exactly the lifetime of the `with` block — including every internal forward an output
control performs. Generation and teacher-forced scoring run here; the scoring math is moved verbatim
from the former `SteeringPipeline.compute_logprobs`.

The session is the unit of concurrency and is **exclusive**: hooks mutate one shared module graph,
so only one generation may be in flight at a time. Re-entrant `__enter__` raises.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from aisteer360.backends.base import SteeringSession, StateControlEntry
from aisteer360.core.output import Output
from aisteer360.core.utils.generation import infer_finish_reason
from aisteer360.utils.tokenization import to_left_pad

if TYPE_CHECKING:
    from aisteer360.backends.generation_params import GenerationParams
    from aisteer360.backends.huggingface.backend import HuggingFaceBackend
    from aisteer360.core.prompt import PreparedPrompt


class HuggingFaceSession(SteeringSession):
    """An exclusive in-process session over a `HuggingFaceBackend`'s model.

    Args:
        backend: The owning backend (provides the model, tokenizer, and input normalization).
        entries: State-control contributions in pipeline list order. Hook-level entries register
            their `hooks` dict; declarative (`plan`) entries are compiled in doc 03.
    """

    concurrency_safe = False

    def __init__(self, backend: "HuggingFaceBackend", entries: list[StateControlEntry]) -> None:
        self._backend = backend
        self._entries = entries
        self.model = backend.model
        self._registered: list[torch.utils.hooks.RemovableHandle] = []
        self._active = False

    def __enter__(self) -> "HuggingFaceSession":
        if self._active:
            raise RuntimeError("HuggingFaceSession is already active; sessions are single-entry (exclusive).")
        self._active = True
        try:
            for entry in self._entries:
                if entry.hooks is not None:
                    self._register_hooks(entry.hooks)
                elif entry.plan is not None:
                    self._register_plan(entry)
        except Exception:
            self._remove_hooks()
            self._active = False
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._remove_hooks()
        self._active = False

    def _register_hooks(self, hooks: dict) -> None:
        """Attach a `{"pre"/"forward"/"backward": [spec, ...]}` hook dict to the model.

        Rollback is handled by the caller's `try/except` in `__enter__`.
        """
        model = self.model
        for phase in ("pre", "forward", "backward"):
            for spec in hooks.get(phase, []):
                module = model.get_submodule(spec["module"])
                if phase == "pre":
                    handle = module.register_forward_pre_hook(spec["hook_func"], with_kwargs=True)
                elif phase == "forward":
                    handle = module.register_forward_hook(spec["hook_func"], with_kwargs=True)
                else:
                    handle = module.register_full_backward_hook(spec["hook_func"])
                self._registered.append(handle)

    def _register_plan(self, entry: StateControlEntry) -> None:
        """Reject a raw plan entry: the HF pipeline pre-compiles plans to hooks via `get_hooks`.

        On in-process backends the pipeline builds `hooks` entries (declarative controls compile
        their plan through `compile_plan_to_hooks`), so a bare `plan` entry should never reach here.
        """
        raise RuntimeError(
            f"HuggingFaceSession received a raw plan for {entry.control_name!r}; the pipeline compiles "
            "plans to hooks for in-process backends. This indicates a pipeline/backend wiring bug."
        )

    def _remove_hooks(self) -> None:
        for handle in self._registered:
            handle.remove()
        self._registered.clear()

    def generate(self, prepared: "PreparedPrompt", params: "GenerationParams") -> Output:
        """Generate under the session's active steering and return an `Output`.

        Args:
            prepared: The adapted prompt to generate from.
            params: Normalized generation parameters.

        Returns:
            An `Output` carrying continuation-only `output_ids`, the adapted `input_ids`, the
            inferred finish reason, and the originating backend in `metadata`.
        """
        input_ids, attention_mask = self._backend.resolve_prompt_tensors(prepared)
        gen_kwargs = params.to_hf_kwargs()
        return_full_sequence = bool(gen_kwargs.pop("return_full_sequence", False))

        full_output_ids = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
        )

        prompt_len = input_ids.size(1)
        new_tokens = full_output_ids[:, prompt_len:]
        returned_ids = full_output_ids if return_full_sequence else new_tokens

        return Output(
            output_ids=returned_ids,
            adapted_input_ids=input_ids,
            finish_reason=infer_finish_reason(new_tokens, gen_kwargs),
            metadata={"backend": "HuggingFaceBackend"},
        )

    def score(self, prepared: "PreparedPrompt", ref_output_ids: torch.Tensor) -> torch.Tensor:
        """Teacher-forced per-token log-probabilities of `ref_output_ids` under active steering.

        Resolves the prompt tensors via the backend and runs the single-forward scoring path.

        Args:
            prepared: The adapted prompt to score against.
            ref_output_ids: Reference continuation ids `[ref_len]` or `[batch, ref_len]` (single-row
                refs broadcast across the prompt batch).

        Returns:
            A `[batch, ref_len]` tensor of per-token log-probabilities.
        """
        input_ids, attention_mask = self._backend.resolve_prompt_tensors(prepared)
        return self.score_tensors(input_ids, attention_mask, ref_output_ids)

    def score_tensors(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ref_output_ids: torch.Tensor,
        *,
        left_pad: bool = True,
        **forward_kwargs,
    ) -> torch.Tensor:
        """Score a normalized prompt batch against `ref_output_ids`.

        Args:
            input_ids: Prompt token ids `[batch, seq_len]` on the model device.
            attention_mask: Matching attention mask `[batch, seq_len]`.
            ref_output_ids: Reference continuation ids `[ref_len]` or `[batch, ref_len]` (single-row
                refs broadcast across the batch).
            left_pad: When `True` (default), left-pad the batch before concatenation so all rows end
                at the same position (decoder-only models). Ignored for encoder-decoder models.
            **forward_kwargs: Extra kwargs forwarded to the model forward pass.

        Returns:
            A `[batch, ref_len]` tensor for decoder-only models, or `[batch, ref_len - 1]` for
            encoder-decoder models (excludes the first decoder token).
        """
        device = self.model.device
        model = self.model

        if isinstance(ref_output_ids, list):
            ref_output_ids = torch.tensor(ref_output_ids, dtype=torch.long)
        if ref_output_ids.ndim == 1:
            ref_output_ids = ref_output_ids.unsqueeze(0)
        ref_output_ids = ref_output_ids.to(device)
        ref_len = ref_output_ids.size(1)

        batch_size = input_ids.size(0)
        if ref_output_ids.size(0) == 1 and batch_size > 1:
            ref_output_ids = ref_output_ids.expand(batch_size, -1)

        if ref_len == 0:
            return torch.zeros((batch_size, 0), device=device, dtype=torch.float32)

        is_encoder_decoder = getattr(model.config, "is_encoder_decoder", False)

        # left-pad for correct positional alignment in causal models; with right-padding, pad tokens
        # between the real input and the appended ref tokens corrupt positional encodings and the
        # causal attention chain
        if left_pad and not is_encoder_decoder:
            input_ids, attention_mask = to_left_pad(input_ids, attention_mask)

        with torch.no_grad():
            if is_encoder_decoder:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=ref_output_ids,
                    **forward_kwargs,
                )
                # predicts ref[t+1] from ref[0:t]; logits[:, :-1, :] aligns with targets ref[:, 1:]
                logits = outputs.logits[:, :-1, :]
                target_ids = ref_output_ids[:, 1:]
            else:
                combined_ids = torch.cat([input_ids, ref_output_ids], dim=1)
                combined_mask = torch.cat(
                    [attention_mask, torch.ones(batch_size, ref_len, device=device, dtype=attention_mask.dtype)],
                    dim=1,
                )
                outputs = model(input_ids=combined_ids, attention_mask=combined_mask, **forward_kwargs)
                # logits at [input_len - 1] predicts ref[0]; at [input_len + ref_len - 2] predicts
                # ref[ref_len - 1]
                input_len = input_ids.size(1)
                logits = outputs.logits[:, input_len - 1: input_len + ref_len - 1, :]
                target_ids = ref_output_ids

        logprobs = torch.log_softmax(logits, dim=-1)
        return logprobs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
