# Backends

## Compatibility matrix

Support is binary: a control's configuration is either supported on a backend or it is not, and
unsupported configurations raise before any work happens with a verdict naming the gap and the
fix. The generate-phase matrix by control:

| Control | HF | vLLM (offline / serve) | Via / verdict |
| --- | --- | --- | --- |
| `few_shot`, `prewrite`, `cpo`, `gepa` | yes | yes | prompt-only at generate; steer-time rollouts on the steering session |
| `sft`, `dpo`, `ppo`, `grpo`, `apo`, `mergekit` | yes | serve artifact | steer on HF; `CheckpointArtifact` / `LoRAArtifact` |
| `caa` | yes | yes | `additive` spec; norm-preserving configurations add the `norm_preserving` modifier |
| `act_add` | yes | broadcast (`T = 1`) only | `additive` carries one `[H]` vector per op; positional (`T > 1`) configurations are hook-only and the verdict says so |
| `directional_ablation` | yes | `K = 1`, `alpha = 1` | `directional_ablation` spec; graded and subspace ablation are hook-only |
| `angular_steering` | yes | `intervention_point="layer_output"` | `rotation`; `adaptive=True` adds the `alignment_adaptive` modifier; the default norm-input placement is hook-only |
| `activation_adapter` | yes | kind-conditional | verdict follows the configured transform, modifier chain, and gate against the negotiated kinds |
| `iti` | yes | `tensor_parallel_size == 1`, vector-supplied | `head_additive` under its constraint; fitting from data is in-process-only (no head-level capture kind) |
| `cast` | yes | no | the projected-cosine condition has no intervention-spec gate kind |
| `pasta` | yes (eager/sdpa) | no | attention-map writes |
| `stopping_rules`, `budget_forcing` | yes | yes | sampling params / `min_tokens` + phased splicing |
| `best_of_n`, `search_decoding`, `phased_decoding`, `thinking_intervention` | yes | yes | drivers over `session.generate` |
| `deal` | yes | no | `BEAM_PROPOSALS`; sampled-proposal search available as its own configuration |
| `routed_decoding` | yes | offline only | probe pass needs `HIDDEN_CAPTURE` at generate; serve has no capture return path |
| `constrained_decoding` (declarative source) | yes | yes | in-process automaton (`aisteer360[guided]`) / native structured outputs under `GUIDED_DECODING`; automaton-object configurations stay HF-only |
| `rad`, `sasa`, `dexperts`, `contrastive_decoding`, `contrastive_guidance`, `value_guidance` | yes | no | model-backed per-step logit math is in-process-only |

Scoring phase: decoder-only scoring with intervention specs is supported on vLLM backends under
the `after_prompt` scope remap; an enabled output control with `include_in_scoring=True` makes
the pipeline score-unsupported off-torch; encoder-decoder scoring is in-process-only.

## API

::: aisteer360.backends
    handler: python
    options:
        show_if_no_docstring: true
        show_source: true
        show_root_heading: true
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
        show_submodules: true
        show_symbol_type_heading: true
        show_symbol_type_toc: true
        filters:
          - "!^_"
