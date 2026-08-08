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

Scoring phase: intervention controls score in-process only, since remote prompt-logprob scoring
anchors token scopes at the request's prompt end (the end of the prompt-plus-reference
concatenation), which would silently unanchor prompt-relative interventions; an enabled output
control with `include_in_scoring=True` likewise makes the pipeline score-unsupported off-torch,
and encoder-decoder scoring is in-process-only.

## Lifecycle

Backends are constructed lazily per pipeline and cached by spec. `SteeringPipeline.release_backends()`,
or using the pipeline as a context manager, releases and evicts every backend the pipeline
constructed, shutting engine-owning backends down deterministically rather than waiting for garbage
collection. A released pipeline stays usable: the next operation reconstructs backends against the
same specs, so a re-booted engine serves subsequent generations. `Benchmark` releases each
configuration's backends automatically after its trials. The offline engine's release is
process-global with respect to vLLM distributed state, so it assumes no other live vLLM engine in
the process.

```python
with SteeringPipeline(controls=[caa], backend="vllm", steer_backend="huggingface", lazy_init=True) as pipeline:
    pipeline.steer()
    response = pipeline.generate(text="...", max_new_tokens=64)
# the engine is shut down on exit
```

## Benchmarking

`Benchmark` forwards its `backend` and `steer_backend` arguments to the pipelines it builds and
pre-flights support over every sweep point (via `SteeringPipeline.check()`) before any model or
engine work, so the compatibility matrix above governs benchmarking too. A sweep point that is
unsupported on the configured backends either fails the whole run (`on_unsupported="raise"`, the
default) or is skipped with a warning (`on_unsupported="skip"`).

## Running a server

The offline vLLM engine (`BackendSpec(kind="vllm")`) boots vLLM inside the current process, so it needs no server and
is the automatic path for single-process runs. The serve backend targets a vLLM server you launch yourself, which is the
answer for a remote GPU box, one server shared across processes or benchmark runs, a client with no local vLLM install,
or process isolation from the steering client.

Start a server with `vllm serve <model> --port 8000` (any extra engine flags as usual), then target it with a spec
carrying `base_url`:

```python
from aisteer360.algorithms.core.execution import BackendSpec

spec = BackendSpec(
    kind="vllm-serve",
    model="meta-llama/Llama-3.1-8B-Instruct",
    options={"base_url": "http://localhost:8000"},
)
```

When serving activation interventions through the vLLM-Hook plugin, the serving environment carries the plugin, the
server starts with `VLLM_HOOK_WORKER=unified` and eager execution, the spec adds `hook_plugin: True`, and `artifact_dir`
names a filesystem shared with the server.

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
