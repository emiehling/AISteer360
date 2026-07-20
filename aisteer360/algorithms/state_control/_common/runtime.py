"""Shared hook runtime for transform-based state controls.

`TransformHookRuntime` encapsulates the hook-body logic common to residual-stream state controls
(hidden-state extraction/re-wrap, KV-cache position bookkeeping, token-scope masking, condition
scoring, and gated transform application). One runtime instance is held per control and owns that
control's per-generation mutable state.

Position bookkeeping — the shared KV-cache offset. During autoregressive generation the model
processes the full prompt on the prefill pass, then a single new token per decode pass — but with
multiple hooked layers, every layer's hook fires once per forward pass. The offset must therefore
advance exactly **once per forward pass**, not once per hook call. The runtime designates the
first-firing hook of each pass as the *pass opener* (the lowest hooked layer; for multiple hooked
modules within a layer, the earliest in execution order): on each call it snapshots the current
offset as the pass offset and advances the offset by the sequence length; every other hook in that
pass reads the snapshot. This is the mirror image of the designated-*closer* pattern (advance on the
last hook), chosen because every subsequent same-pass hook then reads a stable snapshot regardless of
which hook executes last.

Row gating — logical rows vs. the hidden batch. Gates hold one decision per *logical* row (one per
prompt); HuggingFace `generate` may expand the hidden batch to `B_logical * num_beams` via
`repeat_interleave`. The runtime owns both directions of that mapping: condition scores computed on
the expanded batch are collapsed to logical rows (first beam of each group) before `gate.update()`,
and `gate.open_rows()` is re-expanded and ANDed into the token mask before a transform fires. Beam
siblings of one prompt therefore always share that prompt's decision, and batched generation gates
each prompt independently — matching what per-item generation would do.

Condition scoring is evidence-driven, not pass-counted: a condition hook stops scoring as soon as
its gate reports `is_ready()`. A `CacheOnceGate` over threshold evidence therefore yields
"score the prompt once, freeze, and never rescore" with no first-call flag anywhere, while a gate
that never reports ready keeps re-scoring every pass (dynamic conditions).
"""
from __future__ import annotations

from typing import Callable, Literal, Protocol

import torch

from .gates.base import BaseGate
from .hook_utils import extract_hidden_states, replace_hidden_states
from .token_scope import TokenScope, align_mask_to_batch, make_token_mask
from .transforms.base import BaseTransform

HookPoint = Literal["layer_output", "layer_input"]


class ConditionScorer(Protocol):
    """Per-row condition scorer.

    Maps a layer's hidden states to one score per observed batch row. `prompt_mask` is the
    pad-aware prompt attention mask (True at real tokens), supplied only on the prefill pass and
    already aligned to the hidden batch; on decode passes it is None and `hidden` holds the newly
    generated token(s). A python float return is permitted only for single-prompt generation.
    """

    def __call__(
        self,
        hidden: torch.Tensor,
        layer_id: int,
        *,
        prompt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | float: ...


class TransformHookRuntime:
    """Builds hook closures and owns the per-generation position/prefill/mask state for one control.

    Args:
        hook_point: Where the control intervenes. ``"layer_output"`` builds forward hooks on the
            layer output (residual stream after the layer). ``"layer_input"`` builds forward
            pre-hooks; hidden states are extracted from the module's inputs via
            `extract_hidden_states` and re-injected via `replace_hidden_states`. Valid for any
            module receiving `hidden_states` as its first positional argument or as the
            ``hidden_states=`` kwarg — decoder layers, attention output projections
            (`o_proj`/`c_proj`), and per-layer norm sub-modules.
    """

    def __init__(self, *, hook_point: HookPoint = "layer_output"):
        if hook_point not in ("layer_output", "layer_input"):
            raise ValueError(f"hook_point must be 'layer_output' or 'layer_input'; got {hook_point!r}.")
        self.hook_point = hook_point

        # per-generation state (set/cleared by reset)
        self._prompt_lens: torch.LongTensor | None = None
        self._prompt_mask: torch.BoolTensor | None = None
        self._offset: int = 0
        self._pass_offset: int = 0
        self._prefill_seen: bool = False
        self._opener_built: bool = False

    def reset(
        self,
        prompt_lens: torch.LongTensor,
        prompt_mask: torch.Tensor | None = None,
    ) -> None:
        """Clear position/prefill state and store the prompt lengths/mask for this generation.

        Args:
            prompt_lens: Per-row prompt lengths of shape ``[B_logical]`` (from
                `compute_prompt_lens`). Defines the logical batch size for row gating.
            prompt_mask: Optional pad-aware prompt attention mask of shape
                ``[B_logical, T_prompt]`` (True/1 at real tokens). Forwarded to condition scorers
                on the prefill pass so condition scores align with the real (non-pad) prompt
                positions — the same mask the selector calibrated on.
        """
        self._prompt_lens = prompt_lens
        if prompt_mask is not None:
            mask = torch.as_tensor(prompt_mask)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.size(0) != prompt_lens.size(0):
                raise ValueError(
                    f"prompt_mask has {mask.size(0)} rows but prompt_lens has "
                    f"{prompt_lens.size(0)}; these must describe the same logical batch."
                )
            self._prompt_mask = mask.bool()
        else:
            self._prompt_mask = None
        self._offset = 0
        self._pass_offset = 0
        self._prefill_seen = False
        self._opener_built = False

    @property
    def num_logical_rows(self) -> int:
        """Logical batch size (one row per prompt); 0 before `reset`."""
        return 0 if self._prompt_lens is None else int(self._prompt_lens.size(0))

    def _claim_opener(self, is_pass_opener: bool) -> None:
        """Enforce that at most one hook per generation is designated the pass opener.

        Two openers would advance the shared offset twice per forward pass, silently skewing
        every position-dependent token scope (e.g. `after_prompt` would steer the whole
        prompt). Controls must designate exactly one opener; when a layer hosts both a
        condition and a behavior hook, the one registered first opens the pass.
        """
        if not is_pass_opener:
            return
        if self._opener_built:
            raise ValueError(
                "A pass-opener hook was already built for this generation; exactly one hook "
                "may advance the position offset. When a layer hosts both a condition and a "
                "behavior hook, designate only the first-registered one as the opener."
            )
        self._opener_built = True

    def _advance_pass(self, seq_len: int, is_pass_opener: bool) -> int:
        """Return the position offset for the current pass, advancing the shared offset once per pass.

        The pass opener snapshots the current offset and advances it; every other hook reads the
        snapshot taken by the opener earlier in the same pass. The first pass after `reset` is prefill
        (offset 0).

        Args:
            seq_len: The sequence length seen by this hook on this call.
            is_pass_opener: Whether this hook is the designated pass opener.

        Returns:
            The absolute position offset to use when building the token mask.
        """
        if is_pass_opener:
            if self._prefill_seen:
                self._pass_offset = self._offset
                self._offset += seq_len
            else:
                self._pass_offset = 0
                self._offset = seq_len
                self._prefill_seen = True
        return self._pass_offset

    def _collapse_to_rows(self, scores: torch.Tensor | float, hidden_batch: int) -> torch.Tensor | float:
        """Collapse per-hidden-row scores down to the logical rows the gate holds.

        Beam search expands the batch via `repeat_interleave` (`[i0, i0, i1, i1]`), so the first
        member of each group represents its logical row — the same convention as
        `align_mask_to_batch`, in reverse. Scores already at logical size pass through; a bare
        float passes through for the gate to validate (accepted only when `num_rows == 1`).
        """
        if isinstance(scores, (int, float)):
            return scores
        rows = self.num_logical_rows
        t = torch.as_tensor(scores).squeeze()
        if t.ndim > 1:
            raise ValueError(
                f"Condition scorer returned a tensor of shape {tuple(torch.as_tensor(scores).shape)}; "
                f"expected per-row scores of shape [B] (extra dimensions must be size 1)."
            )
        flat = t.reshape(-1)
        if flat.numel() == rows:
            return flat
        if flat.numel() == hidden_batch and rows and hidden_batch % rows == 0:
            factor = hidden_batch // rows
            return flat[::factor]
        raise ValueError(
            f"Condition scorer returned {flat.numel()} scores for a hidden batch of "
            f"{hidden_batch} and {rows} logical row(s); return one score per hidden row or per "
            f"logical row."
        )

    def _row_mask_for(self, gate: BaseGate, hidden: torch.Tensor) -> torch.BoolTensor | None:
        """Per-row gate decision expanded to the hidden batch as a `[B_hidden, 1]` mask.

        Returns None when every row is closed (caller short-circuits). `align_mask_to_batch`
        performs the beam expansion and validates divisibility.
        """
        open_rows = gate.open_rows()
        if not bool(open_rows.any()):
            return None
        row_mask = align_mask_to_batch(open_rows.unsqueeze(1), hidden.size(0))  # [B_hidden, 1]
        return row_mask.to(hidden.device)

    def _prefill_prompt_mask(self, hidden: torch.Tensor, pass_offset: int) -> torch.Tensor | None:
        """The stored prompt mask aligned to the hidden batch, on the prefill pass only.

        The first pass may be *longer* than the prompt when a teacher-forced continuation is
        appended (e.g. `compute_logprobs` forwards `[prompt; ref]` in one pass). The continuation
        columns are not prompt, so the mask is extended with False there — condition aggregation
        then covers exactly the real prompt tokens, reproducing generation-time scoring. A first
        pass shorter than the stored mask indicates misuse and raises.
        """
        if self._prompt_mask is None or pass_offset != 0:
            return None
        width = self._prompt_mask.size(1)
        seq_len = hidden.size(1)
        if width > seq_len:
            raise ValueError(
                f"Prompt mask length {width} exceeds prefill sequence length {seq_len}."
            )
        mask = self._prompt_mask
        if width < seq_len:  # teacher-forced continuation appended after the prompt
            pad = torch.zeros(mask.size(0), seq_len - width, dtype=torch.bool)
            mask = torch.cat([mask, pad], dim=1)
        mask = align_mask_to_batch(mask, hidden.size(0))
        return mask.to(hidden.device)

    def build_behavior_hook(
        self,
        *,
        layer_id: int,
        transform: BaseTransform,
        gate: BaseGate,
        token_scope: TokenScope,
        last_k: int | None = None,
        from_position: int | None = None,
        is_pass_opener: bool = False,
    ) -> Callable:
        """Build a hook that applies `transform` to the residual stream at `layer_id`, gated by `gate`.

        The transform fires at the intersection of the token-scope mask and the gate's per-row
        decision (expanded across beams); a fully closed gate is a no-op.

        Args:
            layer_id: Index of the hooked layer (used to index per-layer transform artifacts).
            transform: The transform to apply at masked positions of open rows.
            gate: Gate consulted per call; row `r` of the hidden batch fires only when the gate's
                logical row `r // beam_factor` is open.
            token_scope: Which positions to steer (see `make_token_mask`).
            last_k: Required when `token_scope == "last_k"`.
            from_position: Required when `token_scope == "from_position"`.
            is_pass_opener: Whether this hook advances the shared position offset.

        Returns:
            A hook callable suitable for the runtime's `hook_point` (a forward hook for
            ``"layer_output"``, a forward pre-hook for ``"layer_input"``).
        """
        self._claim_opener(is_pass_opener)
        if self.hook_point == "layer_output":

            def _forward_hook(module, args, kwargs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden is None:
                    return output
                hidden = self._apply(hidden, layer_id, transform, gate, token_scope, last_k,
                                     from_position, is_pass_opener)
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

            return _forward_hook

        def _pre_hook(module, input_args, input_kwargs):
            hidden = extract_hidden_states(input_args, input_kwargs)
            if hidden is None:
                return input_args, input_kwargs
            hidden = self._apply(hidden, layer_id, transform, gate, token_scope, last_k,
                                 from_position, is_pass_opener)
            return replace_hidden_states(input_args, input_kwargs, hidden)

        return _pre_hook

    def build_condition_hook(
        self,
        *,
        layer_id: int,
        scorer: ConditionScorer,
        gate: BaseGate,
        is_pass_opener: bool = False,
    ) -> Callable:
        """Build a read-only hook that scores the residual stream at `layer_id` and updates `gate`.

        The hook never modifies hidden states. On each pass where the gate still wants evidence
        (`not gate.is_ready()`), it computes per-row scores via `scorer` — handing it the
        pad-aware prompt mask on the prefill pass — collapses beam-expanded scores to logical
        rows, and calls `gate.update(rows, key=layer_id)`. Once the gate is ready (e.g., a frozen
        `CacheOnceGate`), scoring is skipped entirely; a gate that never reports ready keeps
        re-scoring every pass. The hook still participates in pass-opener bookkeeping when the
        lowest hooked layer is a condition layer.

        Args:
            layer_id: Index of the hooked layer.
            scorer: Per-row condition scorer (see `ConditionScorer`).
            gate: Gate to feed the per-row scores to.
            is_pass_opener: Whether this hook advances the shared position offset.

        Returns:
            A hook callable suitable for the runtime's `hook_point`.
        """
        self._claim_opener(is_pass_opener)

        def _score(hidden: torch.Tensor) -> None:
            pass_offset = self._advance_pass(hidden.size(1), is_pass_opener)
            if gate.is_ready():
                return
            prompt_mask = self._prefill_prompt_mask(hidden, pass_offset)
            scores = scorer(hidden, layer_id, prompt_mask=prompt_mask)
            gate.update(self._collapse_to_rows(scores, hidden.size(0)), key=layer_id)

        if self.hook_point == "layer_output":

            def _forward_hook(module, args, kwargs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden is None:
                    return output
                _score(hidden)
                return output

            return _forward_hook

        def _pre_hook(module, input_args, input_kwargs):
            hidden = extract_hidden_states(input_args, input_kwargs)
            if hidden is None:
                return input_args, input_kwargs
            _score(hidden)
            return input_args, input_kwargs

        return _pre_hook

    def _apply(
        self,
        hidden: torch.Tensor,
        layer_id: int,
        transform: BaseTransform,
        gate: BaseGate,
        token_scope: TokenScope,
        last_k: int | None,
        from_position: int | None,
        is_pass_opener: bool,
    ) -> torch.Tensor:
        """Mask the current pass by token scope and per-row gate decision, then apply the transform."""
        seq_len = hidden.size(1)
        pass_offset = self._advance_pass(seq_len, is_pass_opener)

        row_mask = self._row_mask_for(gate, hidden)  # [B_hidden, 1] or None (all closed)
        if row_mask is None:
            return hidden

        mask = make_token_mask(
            token_scope,
            seq_len=seq_len,
            prompt_lens=self._prompt_lens.to(hidden.device),
            last_k=last_k,
            from_position=from_position,
            position_offset=pass_offset,
        )
        mask = align_mask_to_batch(mask, hidden.size(0))  # beam search expands the batch
        mask = mask & row_mask
        if not bool(mask.any()):
            return hidden
        return transform.apply(hidden, layer_id=layer_id, token_mask=mask)


def _build_gate_from_condition(condition) -> "BaseGate":
    """Construct the `CacheOnceGate(MultiKeyThresholdGate)` stack for a `ConditionSpec`.

    Args:
        condition: The `ConditionSpec` describing threshold gating.

    Returns:
        A gate whose freeze/evidence contract matches `condition.cache`: a `CacheOnceGate` wrapping a
        `MultiKeyThresholdGate` for `"prompt_once"`, or a bare `MultiKeyThresholdGate` (never freezes,
        rescored every pass) for `"dynamic"`.
    """
    from .gates import CacheOnceGate, MultiKeyThresholdGate

    inner = MultiKeyThresholdGate(
        threshold=condition.threshold,
        comparator=condition.comparator,
        expected_keys=set(condition.layer_ids),
        aggregate=condition.aggregate,
    )
    return CacheOnceGate(inner) if condition.cache == "prompt_once" else inner


def compile_plan_to_hooks(
    plan,
    *,
    runtime: "TransformHookRuntime",
    prompt_ctx,
) -> dict[str, list]:
    """Compile an `InterventionPlan` to a hook-spec dict, wrapping `TransformHookRuntime` unchanged.

    Relocated and generalized from the per-control `get_hooks` bodies (notably
    `ActivationAdapter.get_hooks`). For a plan whose interventions share this control's single
    `runtime`, it:

    1. resets the runtime for the logical batch (`runtime.reset(prompt_lens, prompt_mask)`);
    2. resolves each intervention's gate — built from its `ConditionSpec` when present, its custom
       `gate` when set, else `AlwaysOpenGate` — and resets each distinct gate once for the batch;
    3. determines one global pass opener across all interventions: the hook target with the lowest
       `layer_id`, ties broken by registration order (condition targets register before behavior
       targets), so exactly one hook advances the shared KV offset per forward pass;
    4. emits condition hooks (per `ConditionSpec`) before behavior hooks, preserving intervention
       list order, so at a shared layer the gate update precedes the transform application.

    Args:
        plan: The `InterventionPlan` (list of `Intervention`).
        runtime: The control's `TransformHookRuntime` (owns per-generation position/gate state).
        prompt_ctx: The `PromptContext` for this generation (ids, mask, prompt lengths).

    Returns:
        A `{"pre": [...], "forward": [...], "backward": [...]}` hook-spec dict.
    """
    from .gates import AlwaysOpenGate

    prompt_mask = None
    if prompt_ctx.attention_mask is not None:
        prompt_mask = prompt_ctx.attention_mask.to(torch.bool)

    runtime.reset(prompt_ctx.prompt_lens, prompt_mask)
    num_rows = int(prompt_ctx.prompt_lens.size(0))

    # resolve the gate per intervention: a supplied gate wins (custom / shared in the driver-follower
    # pattern), else a threshold ConditionSpec builds the gate stack, else the gate is always open
    gates: list = []
    for intervention in plan:
        condition = intervention.condition
        if intervention.gate is not None:
            gate = intervention.gate
        elif condition is not None and condition.is_threshold and not intervention.gate_driven_externally:
            gate = _build_gate_from_condition(condition)
        else:
            gate = AlwaysOpenGate()
        gates.append(gate)

    # reset each distinct gate exactly once for the logical batch
    for gate in {id(g): g for g in gates}.values():
        gate.reset(num_rows)

    # collect ordered hook targets: condition targets (registration-first), then behavior targets;
    # the global opener is the target with the lowest layer_id, ties broken by this registration order
    ordered: list[tuple[str, int]] = []  # (kind, layer_id) markers to find the opener index
    condition_records: list[tuple] = []  # (intervention_index, gate, target, layer_id)
    behavior_records: list[tuple] = []
    for index, (intervention, gate) in enumerate(zip(plan, gates)):
        condition = intervention.condition
        if condition is not None and not intervention.gate_driven_externally:
            for target in condition.targets:
                condition_records.append((index, gate, target))
                ordered.append(("condition", target.layer_id))
        for target in intervention.targets:
            behavior_records.append((index, gate, intervention, target))
            ordered.append(("behavior", target.layer_id))

    opener_index = _opener_index(ordered)

    phase_for = {"layer_output": "forward", "layer_input": "pre"}
    hooks: dict[str, list] = {"pre": [], "forward": [], "backward": []}

    registration = 0
    for _index, gate, target in condition_records:
        intervention = plan[_index]
        condition = intervention.condition
        phase = phase_for[condition.location]
        hooks[phase].append({
            "module": target.module,
            "hook_func": runtime.build_condition_hook(
                layer_id=target.layer_id,
                scorer=condition.scorer,
                gate=gate,
                is_pass_opener=(registration == opener_index),
            ),
        })
        registration += 1

    for _index, gate, intervention, target in behavior_records:
        phase = phase_for[intervention.hook_point]
        hooks[phase].append({
            "module": target.module,
            "hook_func": runtime.build_behavior_hook(
                layer_id=target.layer_id,
                transform=intervention.transform,
                gate=gate,
                token_scope=intervention.scope,
                last_k=intervention.scope_params.get("last_k"),
                from_position=intervention.scope_params.get("from_position"),
                is_pass_opener=(registration == opener_index),
            ),
        })
        registration += 1

    return hooks


def _opener_index(ordered: list[tuple[str, int]]) -> int | None:
    """Return the registration index of the global pass opener, or None for an empty plan.

    The opener is the target with the lowest `layer_id`; ties are broken by registration order
    (the order targets appear in `ordered`: condition targets first, then behavior targets).
    """
    if not ordered:
        return None
    best_index = 0
    best_layer = ordered[0][1]
    for index, (_kind, layer_id) in enumerate(ordered):
        if layer_id < best_layer:
            best_layer = layer_id
            best_index = index
    return best_index
