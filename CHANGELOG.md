# Changelog

## Unreleased

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
