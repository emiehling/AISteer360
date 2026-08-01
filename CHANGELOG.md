# Changelog

## Unreleased

### Added: state specs and scoring on vLLM (P2)

- The transform-runtime state controls (`CAA`, `ActAdd`, `DirectionalAblation`,
  `AngularSteering`, `ActivationAdapter`, `ITI`) execute on vLLM (offline and serve) through
  the vLLM-Hook plugin: each control serializes its steering tuple as an `InterventionSpec`
  (`export_intervention_spec`), emitted from the same transform, gate, and scope objects its
  torch hooks close over. Tensor payloads travel as content-addressed float32 artifacts through
  the plugin registry (`artifact_dir` backend option; defaults to the plugin's registry root).
  A CPU equivalence suite proves hooks and specs are two serializations of one tuple against
  the plugin's own interpreter.
- Requirements are computed by the same serializers: a configuration with a wire form runs
  in-process or on any backend advertising `INTERVENTION_SPECS` with the needed kinds; a
  configuration without one keeps the in-process requirement with a verdict naming the gap
  (positional directions, graded/subspace ablation, norm-input rotation, per-head norm
  preservation, threshold-comparator gates, CAST's projected-cosine condition, PASTA).
- For `hook_plugin` backends, the advertised kind sets are the intersection of the static
  tables and the server's discovery payload; a server missing a kind yields a verdict naming
  the kind. Submission refuses speculative-decoding and non-eager engines, and constrained
  kinds under tensor parallelism, before any work happens.
- KV-cache isolation is structural: spec-bearing requests salt with the reference derivation
  over the canonical spec and its artifact ids; spec-free requests through a plugin-active
  backend carry a per-backend constant salt. Prefix caching stays enabled.
- `compute_logprobs` scores with intervention specs on vLLM backends; `after_prompt` scopes
  remap to `from_position` at the original prompt length, since the teacher-forced reference
  is part of the server-side prompt.
- `vllm_hook_plugins` is a declared dependency of the `vllm` and `dev` extras (git-pinned until
  its PyPI release); `InterventionSpec.canonical()` byte-matches the plugin's canonical form
  and `InterventionSpec.salt()` is the reference cache-salt derivation.
- State controls' `steer()` consumes structural facts from the steering session's layout, so
  vector-supplied configurations steer with `model=None`; hook module names resolve from the
  module tree at `get_hooks()` time.
- `AngularSteering` gains `intervention_point` (`"norms"`, the default and previous behavior,
  or `"layer_output"`, the placement with an intervention-spec form).

### Fixed

- Multi-prompt batches with `seed=` and state controls compute hooks per row via per-call
  control clones; the batch-computed hooks previously misaligned row state on the forced
  serial path.
- `ITI.steer()` no longer mutates a caller-supplied `steering_vector` in place when casting.

### Changed: stop-string and finish-reason semantics (versioned behavior change)

Two related generation semantics are pinned across backends and change in-process behavior:

- **Stop-string truncation**: token ids are returned as generated on every backend (the stop
  text and any token-boundary overrun stay in `Output.output_ids`), and decoded continuation
  text is truncated at the first stop-string occurrence by one client-side rule
  (`aisteer360.algorithms.core.output.truncate_at_stop_strings`). Previously, in-process text
  returns included the stop string plus overrun. vLLM requests set
  `include_stop_str_in_output=True` so ids and text agree before the rule.
- **Finish-reason classification**: `Output.finish_reason` takes values in
  `{"stop", "eos", "length", None}` with the pinned precedence stop, then eos, then length,
  then None, classified from the stop rules the session composed and applied per candidate for
  `n > 1` (`Output.finish_reasons` carries one reason per candidate). Previously the label set
  was `{"eos", "length", None}` with a length-first heuristic that reported None for stop-rule
  terminations.

`StoppingRules` lowers to normalized generation parameters (`export_generation_params`), so its
stops classify as `"stop"` (budget stops as `"length"`) and participate in text truncation.

### Added: multi-backend execution (P1)

- `SteeringPipeline.generate()` and `compute_logprobs()` execute through backend sessions;
  the in-process Hugging Face arm is unchanged apart from the versioned change above
  (encoder-decoder scoring stays on the in-process path).
- `VLLMBackend` (offline engine) and `VLLMServeBackend` (OpenAI-compatible vLLM server) execute
  prompt-only, sampling-mapped, and driver pipelines: token-id prompt submission and return,
  strict parameter rendering (unmapped keys raise), per-item seed derivation
  (`derive_item_seed`), bounded concurrent fan-out with transport-only retries, and
  `PartialBatchError` carrying per-item successes and re-issuable failures.
- `BackendSpec` construction rejects encoder-decoder models for vLLM kinds when the config
  resolves locally; backend construction re-checks authoritatively.
- Decoding drivers gain a `session=` parameter and roll out through `session.generate` on every
  backend; `runtime_kwargs["base_generate"]` is deprecated (honored with a
  `DeprecationWarning`).
- Input controls are prompt-only at generate; `PRewrite`, `CPO`, and `GEPA` require the
  in-process backend at steer. Sampled search drivers run on any backend; beam proposals remain
  gated by `BEAM_PROPOSALS`.
- Structural controls export steer-time artifacts (`CheckpointArtifact` / `LoRAArtifact`) with
  provenance stamps; artifact-producing configurations gain a serve alternative
  (`SERVE_CHECKPOINT` / `SERVE_LORA`) at generate, and vLLM backends consume the artifacts
  (checkpoint path or LoRA request).
- `GenerationParams` gains `stop_strings` and `stop_token_ids`; `seed` derives distinct
  per-item seeds on multi-item fan-outs on both arms.
