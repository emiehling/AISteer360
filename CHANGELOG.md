# Changelog

## Unreleased

### Changed: benchmark config identity and a versioned checkpoint envelope

- Benchmark config identity is a canonical digest over the materialized pipeline (control classes
  and their full parameters), stable across processes, so editing fixed controls no longer resumes
  stale results and the baseline config id is unified on `"baseline"` everywhere. **Old checkpoint
  files are ignored and overwritten on the next save**: resume accepts only a current-format
  versioned envelope whose identity metadata matches, refusing a valid envelope from a different
  configuration with an error naming the differing field, and ignoring anything else (unreadable,
  wrong-shape, or an earlier bare-dict file) with one warning.
- Resume is trial-granular: an interrupted configuration completes only its missing trials, and
  raising `num_trials` on resume runs only the delta.
- Analysis utilities read the recorded `config_id` directly (`flatten_profiles`,
  `per_example_config_means`, `get_generation_field`); a run dict without it raises `KeyError`.

### Added: seeded trials, backend-aware benchmarking, and run provenance

- `Benchmark(seed=...)` derives one seed per (config, trial), threads it through `gen_kwargs` into
  core's existing seed path and into use-case-side RNG (`CommonsenseMCQA` choice shuffling), and
  records it on the run dict; reproduction holds on the same hardware, dtype, and torch/vLLM
  versions.
- `Benchmark(backend=..., steer_backend=...)` forwards backends to the pipelines it builds (a
  `BackendSpec` or a known kind name); a pre-flight `check()` over every sweep point runs before any
  model or engine work, raising one aggregate `UnsupportedBenchmarkError` (`on_unsupported="raise"`,
  the default) or skipping unsupported points with a warning (`on_unsupported="skip"`). The
  shared-preloaded-model fast path and the fingerprint tripwire are scoped to the in-process
  Hugging Face backend.
- `checkpoint_every` selects per-trial (default) or per-config checkpoint writes.
- Run dicts gain `config_id`, `seed`, and `provenance` (backend kinds, model fingerprint, toolkit
  version) additively; the original four keys are unchanged.

### Removed

- `batch_retry_generate`'s deprecated `evaluation_data` parameter and the `_hash_params` alias in
  `data_utils`.

### Changed: evaluation-stack hardening and unified generation path

- Declared use-case generate parameters raise on unknown or missing keyword arguments.
- Every benchmark generation, baseline included, routes through `pipeline.generate(messages=...)`
  (or `text=` for a template-less tokenizer), so the pipeline owns chat templating, tokenization,
  and padding. `adapt_messages` input controls now apply during benchmarking, so `FewShot`
  benchmark results change; runtime-override columns resolve against the prompt rows themselves; and
  the baseline runs through an empty `SteeringPipeline`.

### Added: declarative constrained decoding (P4)

- New output control `ConstrainedDecoding`: one declarative `ConstraintSource` (JSON schema,
  regex, EBNF grammar, or choice set) renders per execution arm. In process it compiles into a
  client-side automaton (the `aisteer360[guided]` extra, xgrammar) driving the existing
  `ConstraintProcessor`; on vLLM backends it renders onto the engine's native structured-output
  parameters (`guided_decoding` offline, `guided_*` fields on serve) in place of the live
  processor. A control constructed with a live automaton object stays in-process-only with a
  tested verdict.
- New capability atom `GUIDED_DECODING` with a static `ConstraintKinds` set
  (`{json_schema, regex, grammar, choice}`), advertised by both vLLM kinds and not by
  Hugging Face; requirements for declarative configurations are in-process torch or guided
  decoding with the source's kind.
- Structured outputs do not apply to prompt logprobs: `include_in_scoring=True` keeps scoring
  in-process, sessions refuse scoring items carrying constraints, and
  `include_in_scoring=False` opts out.

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
