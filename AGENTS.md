# AGENTS.md

Guidance for AI agents working in this repository. The Usage guide covers using the toolkit as a library; the
Developer guide covers extending it; the Invariants section lists rules that apply to every task. When this file
and the code disagree, the code is authoritative: verify against the source before acting on a claim made here.

## Overview

Steerability is a toolkit for steering large language models (Hugging Face causal LMs). It provides steering methods
("controls") across four model control surfaces, a `SteeringPipeline` that composes controls from any of the categories
into one operation on a model, an evaluation stack (recipes and templates, a measurement protocol, a store of fits
and per-item evidence, benchmarks over Inspect AI tasks or prompts, and a sweep runner) for comparing steering
pipelines, and an optimization package that searches steering pipelines under accuracy constraints and certifies the
result on held-out items.

Pipelines execute on one configurable backend: the in-process Hugging Face backend (default), the offline vLLM
engine (`kind="vllm"`), or a vLLM server (`kind="vllm-serve"`). For the generate and score phases, each control
configuration is either supported or unsupported on a given backend: `pipeline.check()` reports unsupported
combinations with a verdict naming the gap and the fix, and unsupported operations raise before any work happens.
The steer phase produces no verdicts. Instead, each control declares the model access its steer step needs on the
`ModelAccess` ladder (`facts` < `rollouts` < `capture` < `module`), and `check()` also returns a deterministic steer
plan stating where each step and fit will run (see Execution backends below).

The four control categories, defined by what a method touches:

- **input**: manipulates the prompt only; generations follow `y ~ p_theta(sigma(x))` for a prompt adapter `sigma`.
- **structural**: persistently modifies weights or architecture (fine-tuning, DPO, merging); `y ~ p_theta'(x)`.
- **state**: edits internal activations at runtime via forward hooks, without changing weights.
- **output**: shapes the decoding process (logits processing, stopping, re-ranking, custom decode loops).

Vocabulary used throughout the codebase:

- **control**: one steering method, subclassing the base class of its category.
- **generic**: a dedicated recipe control class (`activation_adapter`, `value_guidance`, `search_decoding`, ...) that
  exposes common component slots through flat, sweepable `Args`, so that a method from the literature is a
  configuration rather than a new class. Named methods are siblings of generics, not children.
- **common library**: the per-category building blocks in `common/` (transforms, gating, drivers, selectors,
  formatters, ...) from which generics and named methods alike are assembled.
- **probe**: a calibrated linear readout over hidden states used for detection (reads, never edits); gating and
  routing consume its decisions.

## Repository map

```
steerability/
├── algorithms/
│   ├── core/                    # SteeringPipeline, registry, BaseArgs, BaseControl, Output; recipe.py
│   │   │                        # (Recipe, Template, ranges, expand); identity.py (content_id, derive_seed);
│   │   │                        # protocol.py (Protocol, ProviderOptions); store.py (Store); fits.py (fit cache,
│   │   │                        # fit sources and the products of structural controls); sweeps.py (expansion,
│   │   │                        # PipelineFactory, BackendPool); product_fit.py (a product fitted in a child
│   │   │                        # process under a time limit); scoring.py (SampleScorer)
│   │   ├── execution/           # backend interface: spec, contracts, payloads, backend (sessions, Backend, backend
│   │   │                        # registry), params, fanout; access (ModelAccess, SteerPlan); session_utils (scoped
│   │   │                        # sessions, session_generate, session_generate_items); staging (stage freeing,
│   │   │                        # served-artifact selection)
│   │   ├── internals/           # activation capture, pooling, stats, model_layout (decoder-stack resolution,
│   │   │                        # text_config, head geometry), facts (model_facts), fingerprint,
│   │   │                        # data/encoding/render; probes/ (detection)
│   │   └── utils/               # control merging, generation helpers, auxiliary_pass, generate_call (marks each
│   │                            # model.generate call for hook pass counting), assembly (per-generation
│   │                            # hook/spec/processor entry assembly)
│   ├── input_control/           # each category: base.py + one folder per method (triplet layout below)
│   │   └── common/             # building blocks: memory, formatters, proposers, scorers, selectors, budget, pareto
│   ├── state_control/
│   │   └── common/             # building blocks: intervention IR (specs.py), transforms, estimators, sources,
│   │                            # gating, selectors, token scopes, steering vectors, hook runtime, lowering
│   ├── output_control/          # methods incl. routed_decoding/ (control, routing.py, actions.py)
│   │   └── common/             # building blocks: drivers, processors, scorers, values, criteria, kv_cache,
│   │                            # candidates
│   └── structural_control/      # load_checkpoint/ and load_lora/ (artifact loaders; frozen forms of trained
│       └── wrappers/            # structural controls); wrappers/: trl/ (sft, dpo, grpo, apo) and mergekit/
├── spipe/                       # .spipe serialization: SPipe, manifest format, value codec,
│                                # content-addressed artifact store, freeze orchestration
├── backends/                    # huggingface/ (HFBackend, ExclusiveSession); vllm/ (VLLMBackend, VLLMServeBackend)
├── evaluation/                  # Inspect AI stack (optional `eval` extra; __init__ stays empty)
│   ├── provider.py              # SteeringPipelineModelAPI, as_inspect_model
│   ├── batching.py              # lock-leader collator (batched dispatch over concurrent requests)
│   ├── solvers.py               # runtime_kwargs_solver (per-sample runtime kwargs)
│   ├── scorers.py               # sample_scorer_from_inspect (Inspect scorers as SampleScorer rewards)
│   ├── items.py                 # benchmarks (PromptBenchmark, InspectBenchmark), evaluate_items, direct generation
│   ├── runner.py                # SteeringEval (points x trials x benchmarks), samples_frame, runs_frame
│   └── plotting.py              # summary-frame plots over runs_frame output (optional `eval` extra)
├── optimization/                # constrained search over steering pipelines; study.py (Study, Constraint, Settings,
│   │                            # Candidate); space.py (structures, optional, one_of, Slot, check_space); layered:
│   │                            # measurement <- evaluation <- search, release
│   ├── measurement/             # pools.py (PoolSplit, split_items, check_pools, held-out ledger), stats.py (Paired,
│   │                            # the decision rules, look sizes), evidence.py (baseline, design variance, sizes,
│   │                            # feasibility, batches, point tables)
│   ├── evaluation/              # plan.py, result.py, archive.py, evaluator.py and calibration.py (stages, not
│   │                            # public), session.py (StudySession), service.py (EvaluationService)
│   ├── search/                  # engine.py (SearchEngine), loop.py (SearchPhase, run_study), population.py
│   │                            # (initial_population, trained_values), description.py (candidate documents,
│   │                            # descriptions for a language model, the rejection digest), proposers/
│   │                            # (BaseProposer, grid.py, random.py, agent.py), openevolve.py (OpenEvolveEngine) and
│   │                            # _openevolve_controller.py (the only modules that import openevolve)
│   └── release/                 # certify.py (certification on held-out items, certificates, bundles)
└── utils/                       # tokenization, rendering, thinking, optional-dependency guard, verbosity
                                 # (opt-in package logging)

docs/                            # MkDocs site: home/, concepts/, tutorials/ (incl. add_method_by_category/),
                                 # reference/, .nav.yml
examples/                        # notebooks/{algorithms (incl. generics/, wrappers/),studies,recipes,optimization}/,
                                 # index.md; notebooks/optimization/ has one folder per study
                                 # (formality_gsm8k/: the notebook and scripts/ with the local GSM8K task, the
                                 # formality objective, the truth grid, the replay, and the live search)
tests/                           # controls/, core/, internals/, evaluation/, optimization/, utils/; conftest.py
```

## Setup and commands

Python 3.12+ with `uv` as the package manager:

```bash
uv sync --extra all
source .venv/bin/activate
```

Optional extras, in three tiers:

- backends: `vllm`, the vLLM backends plus the `vllm_hook_plugins` core; pulls in `trl[vllm]` so the resolved vLLM
  stays inside trl's supported vLLM range
- workflows: `eval`, the Inspect AI evaluation stack plus matplotlib and seaborn for `evaluation/plotting.py`, and
  `optimize` (OpenEvolve, for `OpenEvolveEngine`); the optimization package also needs `eval`
- method-specific: `merging` (MergeKit)
- `all`: `eval` and `merging`

Contributor tooling lives in `[dependency-groups]`: `dev` (pytest, pre-commit, the plugin core, and the `notebooks`
group), `notebooks` (notebook, ipywidgets, textstat, nltk), and `docs` (site tooling). `uv sync` installs `dev` by
default; add `--group docs` or `--group notebooks` as needed.

The optional-module-to-extra mapping lives in `OPTIONAL_MODULE_EXTRAS` (`steerability/utils/optional.py`).

Hugging Face access uses the standard mechanism: `hf auth login` once, or `HF_TOKEN=hf_***` in the environment. Some
models (e.g. `meta-llama/*`) are gated; the account behind the token needs access on the model's Hub page. Never commit
tokens; a detect-secrets pre-commit hook scans against `.secrets.baseline`.

Models run inside the current process on the default Hugging Face backend; the vLLM backends execute on a local
engine or a remote server instead. Real steering runs need GPU memory for the base checkpoint plus the method's
overhead. For smoke tests, use the tiny models listed in `tests/utils/ci_models.yaml`
(e.g. `hf-internal-testing/tiny-random-LlamaForCausalLM`).

Common commands:

```bash
pytest tests/controls/                    # all control tests
pytest tests/controls/test_pasta.py       # one control
pytest tests/core/ tests/internals/       # pipeline, registry, probes
pytest -m 'not network' tests/            # skip tests that download hub models outside the CI set
pre-commit install                        # once per clone
pre-commit run --all-files                # detect-secrets, whitespace/EOF fixers, large files, isort (black profile)
uv sync --extra all --group docs && uv run mkdocs serve   # docs at localhost:8000
```

Tests parametrize over the models in `tests/utils/ci_models.yaml` and over devices (`cpu`, `cuda`, `mps`); unavailable
devices are skipped automatically, so the suite runs on CPU-only machines. An entry in `ci_models.yaml` is a Hub id or
a `local:<builder>` entry for a model that a function in `tests/utils/tiny_models.py` builds in process (the `gemma4`
entry uses different head dimensions on its sliding and global attention layers). Commit messages require a DCO
`Signed-off-by:` line (see `CONTRIBUTING.md`).

## Usage guide

### Minimal steering loop

Every use of the toolkit follows the same loop: instantiate controls, wrap them in a `SteeringPipeline`, call
`steer()` once, then call `generate()` for inference.

```python
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.input_control.few_shot.control import FewShot

few_shot = FewShot(
    directive="Answer in a formal, professional tone.",
    positive_example_pool=[
        {"prompt": "hey what's up", "response": "Good afternoon. How may I assist you?"},
        {"prompt": "thx", "response": "You are most welcome."},
    ],
    k_positive=1,
)

pipeline = SteeringPipeline(
    model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",  # any HF causal LM id or local path
    controls=[few_shot],
    device_map="auto",
)
pipeline.steer()  # required once before generate(); heavy work (model loading, training, fitting) happens here

response = pipeline.generate(
    messages=[{"role": "user", "content": "Where is the Eiffel Tower?"}],
    max_new_tokens=64,
)
```

Constructor arguments for a control are defined by its `Args` dataclass (in the method's `args.py`) and validated at
construction. For lightweight controls, `steer()` only attaches artifacts like the tokenizer; for controls that train
(structural controls, activation-steering fits), the training runs there.

### Choosing a control category

| Goal | Category | Base class |
| --- | --- | --- |
| Change the prompt (few-shot, rewriting, prompt search) | input | `InputControl` |
| Change the weights (fine-tune, DPO, merge, load a checkpoint or adapter) | structural | `StructuralControl` |
| Edit activations or attention at runtime | state | `StateControl` |
| Shape decoding (rerank, guided sampling, custom loops) | output | `OutputControl` / `DecodingDriver` |

### Available controls

Enumerate the live registry rather than trusting any static list:

```python
from steerability.algorithms.core.registry import REGISTRY  # import triggers discovery

for category, methods in REGISTRY.items():
    print(category, sorted(methods))
```

The registered names at the time of writing:

- input: `cpo`, `few_shot`, `gepa`, `prewrite`, `system_prompt`, `user_prefix`
- state: `act_add`, `activation_adapter`, `angular_steering`, `caa`, `cast`, `directional_ablation`, `iti`, `pasta`
- output: `best_of_n`, `budget_forcing`, `constrained_decoding`, `contrastive_decoding`, `contrastive_guidance`,
  `deal`, `dexperts`, `phased_decoding`, `rad`, `routed_decoding`, `sasa`, `search_decoding`, `stopping_rules`,
  `value_guidance`
- structural: `load_checkpoint`, `load_lora` (artifact loaders, also the frozen forms of trained structural controls
  in a `.spipe`), `mergekit`, `sft`, `dpo`, `grpo`, `apo` (MergeKit and TRL wrappers)

### Pipeline semantics

`generate()` dispatches on the declared keyword source (exactly one per call) and returns the matching shape:

| Source | Tokenization | Return |
| --- | --- | --- |
| `text=` (`str`) | plain text | `str` |
| `text=` (`list[str]`) | batched text | `list[str]` |
| `messages=` (one chat) | chat template | `str` |
| `messages=` (batch of chats) | batched chat template | `list[str]` |
| `input_ids=` (tensor / token id lists) | passed through | `torch.Tensor` |

- Positional `str`/`list[str]` behaves like `text=`; any other positional shape raises a `TypeError`.
- The per-source methods `generate_text`, `generate_messages`, and `generate_tokens` sit alongside `generate()`
  with the same behavior, and take named parameters for the reserved keys.
- Decoded text returns carry exactly one candidate per prompt. Requesting `num_return_sequences`/`n` greater
  than 1 with `text=`/`messages=` raises `ValueError` unless `return_output=True` (which yields one `output_ids`
  row and one finish reason per candidate). The token return is `[batch * n, gen_len]` with each prompt's
  candidates contiguous, as in `model.generate`.

Behaviors that differ from bare Hugging Face usage:

- Returned token ids exclude the prompt by default. Do not slice the result by prompt length; pass
  `return_full_sequence=True` for HF-style prompt-plus-continuation output; a padded batch returns those rows
  left-packed (`[pads, prompt, continuation]`).
- Batched prompts are left-packed after the input chain, before state hooks are built and items are dispatched, so
  the prompt mask a conditional state control pools over aligns with the layout the sessions execute, and
  full-sequence returns and decoding-driver inputs carry that same layout. Single prompts and equal-length batches
  carry no padding and are unaffected.
- `chat_template_kwargs` is a reserved key inside `gen_kwargs`, forwarded to `apply_chat_template` after the
  pipeline-owned template kwargs. It is valid only with `messages=` (pairing it with `text=`/`input_ids=` raises
  `TypeError`) and may not name a pipeline-owned template kwarg (`return_tensors`, `padding`,
  `add_generation_prompt`, `return_dict`). The toolkit does not interpret its contents; keys are model-family
  specific (e.g. `enable_thinking`). Because it rides inside `gen_kwargs`, thinking-on and thinking-off runs get
  distinct configuration identities in sweeps.
- `cancel_token` is a reserved key inside `gen_kwargs` (a `CancelToken` from `algorithms/core/execution`, else
  `TypeError`). Cancelling the token from another thread stops the call at the granularity of the session that runs
  it (see Execution backends), and the call raises `GenerationCancelled`. A decoding driver's rollouts receive the
  token through their `SteeredSession`.
- Token ids are returned as generated on every backend (stop text and any token-boundary overrun stay in the ids);
  decoded continuation text is truncated at the first stop-string occurrence by one client-side rule.
- `generate(..., return_output=True)` returns an `Output` object (or list of them) with fields `output_ids`,
  `adapted_input_ids` (the prompt after input controls, useful for inspecting the steered prompt), and
  `finish_reasons`, one reason per row of `output_ids` (one per candidate for `n > 1`; a single row reads
  `finish_reasons[0]`). Each reason is `"stop"`, `"eos"`, `"length"`, or `None`, with that precedence, except that an
  eos inferred only from stripped trailing pads ranks below `"length"`. Import it via
  `from steerability.algorithms.core import Output`.
- A seeded `generate()` call maps its `seed` onto the items of a multi-item dispatch according to `seed_scope`
  (default `"item"`). Under `"item"`, one seed is derived per row, and on the Hugging Face backend the dispatch then
  decodes one row at a time. Under `"dispatch"`, one seed is derived for the whole dispatch, which is batched in one
  pass (reproducible as a whole). Every `generate()` call on the Hugging Face backend opens a new session, and under
  `"dispatch"` calls with the same seed therefore derive the same sampling seed. The scope is inert on vLLM backends,
  and an item containing its own seed is honored under either scope.
- `generate()` before `steer()` raises `RuntimeError`; a second `steer()` call is a silent no-op.
- `attention_mask` is valid only with `input_ids=`; it is derived automatically for `text=` and `messages=`, and
  passing it with either (or with positional text) raises a `TypeError`.
- `device` and a non-default `device_map` are mutually exclusive on the `SteeringPipeline` constructor.
- Construction never loads the model. `steer()` acquires it from `model_name_or_path`, reuses preloaded
  `model=`/`tokenizer=` objects passed at construction, or receives it from a structural control that produces the
  final weights itself (e.g. `mergekit`).
- A `revision` key in `hf_model_kwargs` also pins the tokenizer when it is loaded from `model_name_or_path`.
- `pipeline.supports_batching` is `True` only when every enabled control declares batch safety and no enabled
  decoding driver with `whole_batch_rollouts = False` (the phased and search drivers) is combined with an enabled
  state control; the Inspect model provider batches concurrent requests when it is `True` and serializes them
  otherwise. With state hooks in process, the driver path raises `ValueError` for a batch of more than one prompt
  when the driver sets `whole_batch_rollouts = False` or the call is seeded under `seed_scope="item"`.
- `pipeline.supports_concurrent_generate` is `True` when the backend lacks `IN_PROCESS_TORCH` (the vLLM backends) and
  no enabled decoding driver is present. `generate()` accepts concurrent calls from several threads on any pipeline.
  Each call's prompt phase (input controls, templating, tokenization, item assembly) runs under the pipeline's
  prompt lock, and the dispatch runs concurrently with other calls only when `supports_concurrent_generate` is
  `True`; otherwise the calls run one at a time. `steer()`, `release_backends()`, and `compute_logprobs()` hold the
  lock for their whole duration and must not run concurrently with `generate()`.
- `pipeline.compute_logprobs(input_ids, ref_output_ids=...)` scores reference tokens teacher-forced with the full
  steering applied; output controls with `include_in_scoring=False` are excluded from scoring.
- Controls with a `tokenizer` attribute left as `None` get the pipeline tokenizer injected automatically.

### Execution backends

`SteeringPipeline` takes `backend=`, a `BackendSpec` or a kind string. The default is the in-process Hugging Face
backend, so a pipeline that never names a backend runs entirely in process. `fit=` (`"auto"` or `"in_process"`)
selects the fit venue policy.

```python
from steerability.algorithms.core.execution import BackendSpec

pipeline = SteeringPipeline(
    controls=[caa],
    backend=BackendSpec(kind="vllm", model="meta-llama/Llama-3.1-8B-Instruct", options={"hook_plugin": True}),
)
```

- `pipeline.check()` returns a `SupportReport` without doing any work; `steer()` runs it and raises
  `UnsupportedPipelineError` for unsupported control/backend combinations at generate. Verdict messages are
  stable tested strings naming the gap and the fix. The report also carries `plan`, the deterministic steer
  plan (per-control access and venue, per-fit venue, whether a stage runs, and the warnings that will fire).
  The per-control support boundary is recorded on each control's `Backends` line in `docs/concepts/controls.md`.
- The steer phase satisfies each control's declared `steer_access()` by venue. `facts` and `rollouts` steps run
  through the backend's session on every kind. `capture` steps run through session capture where the spec
  advertises it (the offline plugin engine), and on a staged in-process model where it does not (serve, or
  `fit="in_process"`). `module` steps always stage. On engine backends the staged model is loaded, used, and freed
  before the engine boots; exported artifacts are the handoff, so in-process weights and engine-served weights
  never coexist. If engine capture fails a steer-time smoke test, fitting degrades to the stage with a warning. The
  stage loads the weights the engine serves (a checkpoint from its path, or the base with the LoRA adapter merged
  in). Support verdicts never depend on the plugin's presence.
- Activation-steering state controls execute on vLLM through the vLLM-Hook plugin (`hook_plugin: True` on the
  spec). The control's steering tuple serializes as an intervention spec, and tensor payloads travel as
  content-addressed artifacts. The offline engine writes them into its registry (the `artifact_dir` option). On serve
  `artifact_dir` is optional: without it each payload is uploaded to the server's `/v1/hook/artifacts/{id}` route, and
  with it the directory must be the server's registry on a shared filesystem.
  A configuration either serializes exactly or is honestly in-process-only; there is no approximate lowering.
- The offline backend applies a scoped engine-process environment per boot and restores it
  (`backends/vllm/environment.py`), defaulting `VLLM_USE_FLASHINFER_SAMPLER=0` (an explicit caller value wins) and
  forcing `VLLM_HOOK_WORKER=unified` for `hook_plugin` boots. `serve_environment` returns the same policy as a fresh
  mapping for a `vllm serve` process. The model-runner constraint is owned by the vLLM-Hook plugin (which pins the
  legacy runner or supports V2), not the toolkit.
- The offline engine is vLLM's `AsyncLLM`, constructed from `AsyncEngineArgs(model=..., **engine_kwargs)` on an
  event loop that the backend owns on a daemon thread (`backends/vllm/engine_thread.py`). The session submits one
  engine request per item, and the engine admits each request as it arrives. Concurrent `generate()` calls therefore
  share one running batch, and each request returns when it finishes. The vLLM-Hook plugin's patch of `AsyncLLM.generate`
  performs each request's admission, spec staging, capture attachment, and cleanup. `release()` cancels in-flight
  requests (their callers raise `EngineClosedError`), shuts the engine down on its loop, and joins the thread. The
  engine is built under vLLM's `LLM_CLASS` usage context, which keeps the offline scheduler defaults (`max_num_seqs`,
  `max_num_batched_tokens`). Per-request lifecycle logging defaults off (`enable_log_requests=False`), and the
  periodic stats logging keeps vLLM's default (on, which reports preemptions); an explicit entry in `engine_kwargs`
  takes precedence for either.
- Sessions honor a `cancel_token` at different granularity. The offline vLLM session aborts the call's unfinished
  engine requests. The in-process session checks the token before each decode and lets a running `model.generate`
  finish, and the serve session checks it before each HTTP request and lets a request already sent complete.
- On `vllm-serve`, a single-prompt call issues one HTTP request on the caller's thread, and the server batches
  requests from concurrent calls. `request_timeout` must cover one full generation, `max_retries` applies to
  transport errors only, and `max_concurrency` bounds the fan-out inside one call rather than the number of
  concurrent calls. Launching and supervising the server stays with the caller.
- Structural controls train on the staged model and serve their artifacts (checkpoint or LoRA) on vLLM backends. An
  engine serves the last checkpoint and selects a LoRA adapter per request. A session serves the adapter it is opened
  with (`Backend.open_session(adapter=...)`), and a `BackendPool` therefore keys an engine by its checkpoint only. One
  engine then serves the pipelines of several adapters. The offline engine serves adapters only when its spec's
  `engine_kwargs` set `enable_lora`, with a `max_lora_rank` of at least the adapter's rank (16 by default,
  `BackendSpec.lora_rank_limit`), and only in float16 or bfloat16 (vLLM's LoRA kernels do not run in float32). `check()`
  verifies this for a control that serves an adapter (`lora_capacity`, with the rank its configuration gives), and the
  engine checks its dtype and a loaded adapter's rank when a session opens with the adapter. Scoring under an adapter
  also needs `lora_target_modules` without `lm_head` (`lora_scoring`, at the score phase), since a batch of scoring
  requests under an adapter can stop an engine whose head has a LoRA layer. An engine with `lora_target_modules` applies
  only the adapter weights of the modules they cover (`BackendSpec.lora_modules_left_out`). `check()` therefore fails a
  training whose `target_modules` list is not covered (`lora_coverage`), and the engine refuses a loaded adapter whose
  weights change a module outside them. Each adapter gets a name and an id of its own (`adapter_identity`). `check()`
  fails a second adapter in one pipeline. On `vllm-serve`, a session loads its adapter onto the server on first use, and
  a checkpoint is served only when its path is the spec's `model`.
- Declarative constrained decoding lowers to vLLM's native structured outputs on the default decode path. Under an
  enabled `DecodingDriver` a constraint or processor spec does not lower (the driver receives the in-process
  processor), and `check()` fails that combination on the engine kinds. Hidden-state capture (probe fitting and
  reads, routed decoding) is served in process and on the offline plugin engine, not on serve.
- `compute_logprobs` scores through the backend; an enabled output control with
  `include_in_scoring=True` keeps scoring in-process.
- Discarding a pipeline that booted a vLLM engine should go through `release_backends()` (or a
  `with` block over the pipeline) rather than relying on garbage collection, which is not prompt at
  freeing the engine. A failed `steer()` frees any staged in-process model (clearing the local variables of the
  stage frames in the error's traceback) and releases the backends it constructed before re-raising, and a
  retried steer therefore re-boots. `PipelineFactory` (and so `SteeringEval`) keeps engine backends in a
  `BackendPool` across recipes and releases them at `factory.release()`.
- Spec options the vLLM backends read: `hook_plugin`, `artifact_dir`, `engine_kwargs` (offline engine);
  `base_url`, `api_key`, `max_concurrency`, `request_timeout`, `max_retries`, `retry_backoff` (server);
  and `tokenizer_name_or_path` / `trust_remote_code` for the client-side tokenizer. Options must be plain
  data; `BackendSpec` canonicalizes them and its hash is the backend identity. A spec combining
  `hook_plugin` with speculative decoding, or an offline `hook_plugin` engine with
  `enforce_eager=False`, is rejected at construction.

### Composition rules

- A pipeline accepts any number of controls per category. `steer()` runs in a fixed bottom-up order (structural, then
  input, then state, then output) and in list order within each category, so higher layers always see the final model.
- At most one enabled `DecodingDriver` may be present; the decode loop does not compose. When none is supplied the
  pipeline uses `model.generate`. Step-level output controls (processors, stopping criteria) compose freely.
- Every driver preset accepts a padded batch and returns `n` (`num_return_sequences`) candidates per prompt in
  row-major, candidate-minor order. The phased drivers (`phased_decoding`, `budget_forcing`, `routed_decoding`) run each
  candidate through its row's plan under one `max_new_tokens` ceiling across all phases. Fixed text (including routed
  prefixes and canned responses) counts against the ceiling and is appended whole; a candidate that reaches the ceiling
  skips the rest of its plan (finish reason `"length"`). After each `Generated` phase a candidate records what ended it
  (`last_stop`: `"until"`, `"until_token"`, `"budget"`, `"length"`, or `"eos"`). A plan entry may be a callable that
  receives the candidate's `PlanState` (`last_stop`, `appended`, `text`) and returns a phase, or None to skip the entry;
  `budget_forcing` uses this to extend only a segment cut off at its budget and to never double the closing tag. The
  phased drivers raise `ValueError` for `num_beams > 1` with `n > 1`. The search drivers (`deal`, `best_of_n`,
  `search_decoding`) run `n` independent searches per row, with the preset's own candidate count as the proposals per
  iteration from each kept beam. They clamp segments so that no continuation exceeds `max_new_tokens`, and beam
  proposals without sampling run one search and return it as every candidate. Each search returns the highest-scoring
  sequence among those it stopped extending (finished, or kept in the last iteration).
- In-process state hooks restart their pass count at every `model.generate` call the Hugging Face session issues,
  which places each call's prefill at position 0. Hook points that receive no position ids (e.g., `iti` on `o_proj`)
  therefore steer every phase and candidate of a driver.
- Input controls run in two phases on chat input. Every control's `adapt_messages` runs in list order before chat
  templating; controls whose `adapt_messages` returns `None` then run their token-level `adapt` in list order after
  tokenization. Text and tensor inputs skip the message phase entirely. Place semantic rewriters (`prewrite`, `cpo`,
  `gepa`) before surface formatters (`few_shot`) in the controls list.
- State hooks register in list order and chain on the same module, so non-commuting edits (e.g. ablate then add
  versus add then ablate) are order sensitive.
- Structural controls thread the model: each receives the model returned by the previous one, with no implicit
  reconciliation between stages.

State controls and hidden-state capture resolve the decoder stack at one of three roots: `model.layers` (text-only
decoder models: Llama, Mistral, Qwen, Gemma text), `model.language_model.layers` (composite multimodal wrappers such
as Gemma 3/4 and Qwen3.5 loaded under `AutoModelForCausalLM`), and `transformer.h` (GPT-2). Resolution selects the
per-layer naming convention (`llama_style`, `gemma_style`, `gpt2_style`) whose norm markers exist on the first decoder
layer and whose attention module exists on at least one layer.

A hybrid stack that interleaves attention layers with another token mixer (Qwen3.5 and Qwen3-Next, where three Gated
DeltaNet `linear_attn` layers precede each `self_attn` layer) resolves to its attention layers' family, with
`ModelLayout.attention_layer_ids` recording which layers carry attention. Residual-stream controls (`caa`, `act_add`,
`angular_steering`, `activation_adapter`) and hidden-state capture work unchanged on such a stack; `head_geometry`,
o_proj-site interventions, and `pasta` refuse the other layers with a message naming the attention layers, and `iti`
refuses hybrid stacks.

A multimodal checkpoint is steered on its text decoder under text-only prompting; images and audio stay out. The LoRA
adapters the TRL wrappers train attach to that decoder as well, since a `target_modules` suffix list is scoped at steer
time to the resolved decoder stack. An unmerged LoRA adapter (`LoadLoRA(merge=False)`, or a TRL LoRA run without
`merge_lora_after_train`) is hooked through the PEFT wrapper, so a state control listed after it steers the adapted
model. Register a detector with `register_layout_detector` (from `steerability.algorithms.core.internals`) for an
architecture not on this list.

### Runtime kwargs

Some controls need per-call information at inference time. All controls read from the single `runtime_kwargs` dict
passed to `generate()`. Each control declares the names it consumes in its `RUNTIME_KWARGS_SCHEMA`, together with a
`scope` per entry: `"row"` for a per-prompt value (delivered as a row-aligned sequence in batched calls) or `"call"`
for one value per call; a missing `scope` means `"call"`.

The pipeline validates the declarations at `steer()`. It raises on disagreeing declarations of one name and warns when
two controls declare the same name with agreeing declarations (they will share one value).

Two examples of row-scoped kwargs. For PASTA's `substrings`, a `str` broadcasts to every row, a `list[list[str]]` of
batch length carries one group per row, and a flat `list[str]` is accepted only at batch size 1 (to broadcast one group
over a batch, pass `[[...]] * batch_size`). The `SearchDriver` presets (`DeAL`, `BestOfN`, `SearchDecoding`) declare
`reward_params` row-scoped: one mapping per row, merged into the scorer's params.

```python
pipeline.generate(
    prompt,
    runtime_kwargs={"substrings": ["The answer must be in JSON."]},  # e.g. pasta's emphasis spans
    max_new_tokens=128,
)
```

### Saving and loading pipelines (`.spipe`)

`pipeline.to_spipe()` serializes a pipeline as an `SPipe`: the model reference plus the controls as constructed
(the recipe), and, once the pipeline is steered, the frozen resolution (fitted vectors, probes, adapters, optimized
prompts) in a content-addressed artifact store, with per-control fit digests and a lock section recording the
`Protocol` the bundle was fitted under, the facts observed on its model, and an optional certificate. The manifest
(`spipe/2`) writes one recipe entry per enabled control and the recipe's id (`recipe_id`, independent of the model
reference and of entry order); `spipe.recipe` and `spipe.protocol` read them back.

`spipe.save(path)` writes a zip when `path` ends in `.spipe` and a directory otherwise; `artifacts="thin"` writes the
manifest only, with artifact ids resolved at load through `artifact_store=`. `SPipe.load(path)` reads either form.

`spipe.pipeline()` reconstructs a `SteeringPipeline`. Frozen entries instantiate from their resolution, so `steer()` is
cheap and model-free; `prefer="recipe"` forces re-fits instead. Backend, device, dtype, and `hf_model_kwargs` stay the
caller's, except that a model revision recorded at save time fills `hf_model_kwargs["revision"]` when the caller sets
none. `verify()` is the model-free report and `thaw()` drops the resolution. `allow_code=True` at load permits
`$ref` callables, `$dc` dataclasses outside `codec.DECODABLE_DATACLASSES` (the toolkit dataclasses whose construction
only validates), and tree artifacts whose payload contains pickle-bearing files (pickle-backed memories, adapter or
checkpoint trees with pickled weights). Toolkit enums always decode, and `$dc` constructs only dataclasses and enums.
Freezing leaves TRL trainer state (`training_args.bin`, optimizer, scheduler, scaler, and RNG state) out of adapter and
checkpoint trees. `allow_code` governs decoding only; recipe args such as `trust_remote_code` take effect at `steer()`.

`spipe.entries` lists the manifest's entries as `SpipeEntry` records (`index`, `method`, `args`, `resolved`)
without decoding them, and `spipe.instantiate_entry(index, prefer=, verify=, lenient=)` returns one entry's control(s);
`pipeline()` is built from it. Errors are raised per entry, with `SpipeFormatError` for an unregistered method key, a
malformed value, or args the constructor rejects, and `SpipeCodeRefError` when the entry needs code. A bundle whose
method key is unregistered in this process therefore loads, and instantiating that entry raises `SpipeFormatError`, as
does reading `spipe.recipe`. For inspecting
an entry's `steer_access()` and `requirements()`, `lenient=True` never imports or runs bundle code (`$ref` values become
inert markers, `$data` stays a `DataRef`, no dataset loads); an entry that needs other code (a `$dc` class outside
`DECODABLE_DATACLASSES` or a pickle-bearing artifact) raises `SpipeCodeRefError` there even under `allow_code=True`.

An unsteered pipeline freezes without loading the model when every enabled control is recipe-frozen (`FACTS` steer
access, no fits, no exported state, as for a prompt-only pipeline), and the lock's observed facts then contain no
model fingerprint and no dtype; otherwise `freeze=True` raises naming each blocking control. Disabled controls are not
written.

Loading a stale bundle (fit-relevant recipe fields edited after freezing) raises unless `allow_stale=True`. Trained
structural controls freeze as `load_checkpoint` / `load_lora` entries; intervention controls freeze as
`activation_adapter` entries unless they declare a same-class frozen form (as `caa`, `act_add`, and `iti` do).

```python
pipeline.steer()
pipeline.to_spipe().save("formal_tone.spipe")

from steerability.spipe import SPipe
loaded = SPipe.load("formal_tone.spipe").pipeline()
loaded.steer()
```

### Evaluation

Evaluation measures recipes item by item under one measurement protocol (optional `eval` extra for Inspect). A
`Protocol` (`algorithms/core/protocol.py`) fixes the base model and revision, tokenizer, load arguments, backend, fit
policy, provider options, generate defaults, seed, and package versions; its `id` scopes every stored rollout and
evidence row, and its `fit_context_id` (no provider, decoding, or seed) scopes every cached fit. Pipelines are
`Recipe`s or `Template`s (`algorithms/core/recipe.py`): a template's args may contain `Choice`, `Interval`, or `Text`
ranges, ranges naming one `axis` move together, and `expand` enumerates (grid) or samples (random) points. `BASELINE`
is the empty recipe. A recipe's `id` comes from each method's `Args` dataclass, so key order, explicit defaults,
entry names, disabled entries, and cross-category order do not change it.

```python
from steerability.algorithms.core.protocol import Protocol, ProviderOptions
from steerability.algorithms.core.recipe import BASELINE, Choice, Template, entry
from steerability.evaluation.items import InspectBenchmark
from steerability.evaluation.runner import SteeringEval, summarize_runs

protocol = Protocol.create(
    "meta-llama/Llama-3.1-8B-Instruct",
    provider=ProviderOptions(max_batch_size=8),
    generate_defaults={"temperature": 0},  # greedy is the recommended posture
    seed=7,
)
pipelines = {"baseline": BASELINE, "caa": Template((entry(CAA, data=pairs, multiplier=Choice((1.0, 2.0))),))}
evaluation = SteeringEval(protocol, pipelines, [InspectBenchmark("inspect_evals/gsm8k", scorer="match")],
                          limit=200, num_trials=3)
evaluation.run("runs/exp1")
runs = evaluation.runs_frame({"accuracy": "match"})  # one row per (pipeline, recipe, trial)
samples = evaluation.samples_frame({"correct": "match"})  # one row per item
summary = summarize_runs(runs, ["accuracy"], param_cols=["caa.multiplier"])
```

A benchmark (`evaluation/items.py`) defines items and their scoring: `InspectBenchmark` runs one Inspect task through
`inspect_ai.eval` (its id covers the task, a local task file's SHA-256, task args, scorer key, generate args, model
roles, group key, and the Inspect versions), and `PromptBenchmark` generates prompts directly and scores them with a
versioned `SampleScorer` (`@versioned("1")` from `algorithms/core/identity.py`). `evaluate_items` stores one evidence
row per (protocol, resolved recipe, benchmark, item, replicate), retries items a benchmark omits, and writes error
rows for the rest; evidence is the resume mechanism, so a second run generates only what has no evidence row and
selections may grow freely. Direct generation stores each rollout under a key known before generation, so a changed
scorer re-scores stored completions without generating. An `InspectBenchmark` whose task calls the model once per
sample (captured at `prepare` on `mockllm/model`) is generated directly under a protocol once a greedy parity probe on
up to 16 items agrees with Inspect on every completion and score; the mode is recorded per (benchmark, protocol) in the
store's `benchmark_modes` table (`benchmark.mode(protocol)`). Under a sampling protocol a pipeline without
`supports_concurrent_generate` runs through Inspect, since direct generation samples one request per call.

The store (`algorithms/core/store.py`, `save_dir/store` unless `run(store=...)` shares one) is write-once SQLite plus
the artifact store. It caches fits per fit source (`algorithms/core/fits.py`): intervention controls resolve sources
through `resolve_source`, other fitting controls are replaced by their frozen form on a hit, and a recipe's resolved
id (the key of its rollouts and evidence) combines its id with the stored fits it used. A structural control that
trains or merges has one fit, its product (an adapter or a checkpoint copied into the store's artifacts), which it
resolves through `resolve_product`. A product is keyed by its fit digest and by the structural entries before it
(`product_weights`, where an earlier training is given by its own digest), and two recipes that agree on these share
it. Output locations and the structural entries after it are not part of the key. A stored product is installed rather
than trained again, and the control then declares `ModelAccess.FACTS`.
`PipelineFactory(protocol, store=...)` steers recipes in their fit cache, shares the base model, pools engine backends,
and runs a batch's fits up front (`prepare`). It fits each pending product alone, in a pipeline of its own, before
anything else of the recipe. Every pipeline it yields and every other fit of the recipe then runs on stored products.
With a time limit (`fit_seconds` of `prepare` and `steered`), a product whose structural entries a new interpreter can
rebuild is fitted in a child process, which is stopped at the limit (`ProductFitTimeout`, in
`algorithms/core/product_fit.py`). The child receives the store's root, the protocol's identity, and the entries as a
recipe. A dataset held in memory and a callable defined in `__main__` cannot be passed (`transport_problems`), and such
a product is fitted in process. A TRL wrapper therefore accepts a dataset given as a `DataRef`, which it loads when it
trains.
`check()` runs over every point before any model or engine work (`on_unsupported="raise"` or `"skip"`).
`save_dir/protocol.json` records the protocol and observed facts (a directory written under another protocol
is refused), and `save_dir/points.json` the points, selections, and trials; the module-level `samples_frame` and
`runs_frame` read a finished run from it. A metric that is not a mean over items is computed from `samples_frame`.

Every generation flows through `pipeline.generate()`. Prompts enter as `messages=` when the tokenizer has a chat
template (so `adapt_messages` input controls fire exactly as in deployment) and as rendered `text=` otherwise. Scoring
is generation-based only; logprob parameters, tools, and multimodal content are refused with actionable messages.

Concurrent Inspect requests collate into batched pipeline calls when `pipeline.supports_batching` is `True`. A seeded
dispatch uses `seed_scope` from `ProviderOptions` (default `"dispatch"`), so a seeded batch decodes in one pass on
the Hugging Face backend, and bitwise reproducibility of stochastic sampling is not preserved under concurrency (see
the `steerability/evaluation/batching.py` module docstring for the full contract). Direct generation batches greedy
requests in a fixed order (by prompt length at replicate 0, shuffled by a derived seed at later replicates).

`ProviderOptions(max_concurrency=N)` selects per-request mode on a pipeline whose `supports_concurrent_generate` is
`True`. Each Inspect request is then dispatched at once as its own single-prompt `pipeline.generate()` call in a
worker thread, with at most `N` in flight, and `max_connections()` returns `N`. A seeded request samples under
`derive_seed(seed, f"request:{digest}", 0)`, where `digest` hashes the request's prompt and per-sample runtime kwargs,
the same seed direct generation gives the request. The request therefore reproduces regardless of concurrency and
arrival order. A cancelled Inspect request cancels its `CancelToken`, which aborts its engine request. On any other
pipeline the provider logs at info level and stays in batching mode. `max_concurrency` changes throughput and not
results, so it is not part of a protocol's identity.

Per-sample steering inputs are given in `Sample.metadata` and are delivered by the included `runtime_kwargs_solver`
(used in place of a bare `generate()` in the task's solver chain). Static per-arm kwargs go in
`ProviderOptions.runtime_kwargs`: a static value of a `"row"`-scoped kwarg is one row's value in the control's per-row
form, broadcast to every row. Controls that consume a per-row reward take a `SampleScorer` (`(response, row) ->
float`, from `algorithms/core/scoring.py`); `sample_scorer_from_inspect` adapts any Inspect scorer into that shape.
See `docs/tutorials/evaluate_steering_pipelines.md` for the full guide, including task authoring and grader-model
guidance.

The core layers (`recipe.py`, `protocol.py`, `store.py`, `fits.py`, `sweeps.py`) have no Inspect dependency.

### Optimization

The optimization package (`steerability/optimization/`) searches the steering pipelines of a space of templates for
the one that maximizes a behavior score while the accuracy on each capability benchmark stays within a fixed margin
of the unsteered model, and certifies the result once on held-out items. A `Study` (`optimization/study.py`) fixes the
protocol, the space (a `Template` per structure), the objective (a behavior benchmark), the `Constraint`s (a
benchmark and a margin each), the split of every benchmark's items into a search pool and a held-out pool
(`split_items`, which keeps groups whole), the seed, and the `Settings`. `structures` (`optimization/space.py`)
builds the space from slots, in which `optional` makes an entry or a slot optional and `one_of` makes entries
exclusive. `structures(optional(prompt), optional(caa))` gives the structures `prompt`, `caa`, and `prompt+caa`, and
each axis of an entry has the same key in every structure that contains it (`caa.layer_id`). Per-item scores are in
[0, 1], and higher is better. Each template's ordered axis that is not in `study.searched` is its solved axis (at most
one), which the evaluator solves for each candidate. Every other axis is a proposer axis, and a `Candidate` is a
structure with one value per proposer axis (`Study.bind` is the only way to turn one into a template). A solved axis
must not change a fit (every ladder position has the same `fit_digests`), so an ordered input of a fit belongs in
`searched`. `Study.validate(store)` checks the protocol's `versions` and every benchmark's `temperature`, the versioning
of callables, the pools, the axes, the benchmark names, that every structural control that trains or merges leaves a
product to store, and that the protocol's backend can run every structure, before any model work.
`check_space(space=..., protocol=...)` (`optimization/space.py`) gives the structures a backend cannot run, with the
messages of `preflight`, so that a study can be built from the others.

```python
from steerability.optimization import Constraint, Study, optional, split_items, structures
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.release.certify import certify
from steerability.optimization.search.loop import SearchPhase, run_study
from steerability.optimization.search.openevolve import OpenEvolveEngine
from steerability.optimization.search.proposers import GridProposer

for benchmark in (gsm8k, formality):
    benchmark.prepare(store=store)
prompt = entry(SystemPrompt, name="prompt", text=Text(max_chars=600))
caa = entry(CAA, data=pairs, layer_id=Choice(values=(10, 14)), multiplier=Interval(low=0.0, high=8.0))
study = Study(
    name="formality",
    protocol=Protocol.create(model=model, hf_model_kwargs={"dtype": "float32"}, generate_defaults={"temperature": 0}),
    space=structures(optional(prompt), optional(caa)),
    objective=formality,
    constraints=(Constraint(benchmark=gsm8k, margin=0.05),),
    pools={
        gsm8k.id: split_items(benchmark=gsm8k, held_out=5000, seed=0),
        formality.id: split_items(benchmark=formality, held_out=500, seed=0),
    },
)
archive = run_study(
    study=study,
    phases=[
        SearchPhase(engine=GridProposer(study=study, structures=["caa"])),
        SearchPhase(
            engine=OpenEvolveEngine(
                run_dir="runs/formality",
                llm=llm,
                iterations=400,
                goal="The behavior score measures how formal the reply reads.",
                structures=["prompt", "prompt+caa"],
            ),
        ),
    ],
    store=store,
    max_generations=200_000,
)
with StudySession(study=study, store=store) as session:
    session.start()
    certification = certify(session=session, bundle_dir="bundles")
```

Evaluation goes through one `StudySession` (`optimization/evaluation/session.py`). `start()` validates the study and
checks its feasibility with the floor variance and the raw pool sizes (`feasibility` in `measurement/evidence.py`). It
then records the study in the store tables of the `"optimization/2"` schema and measures the unsteered baseline once
(`ensure_baseline`). Feasibility is checked again at the measured baseline before the recorded rounds are resumed. The
baseline is measured on the search pool and on the held-out pool as a separate request set (the one every finalist's
held-out call uses). A second search-pool run in a different batch order gives the noise floor (the mean absolute
difference between the two runs). Evidence rows are keyed by recipe and item, and they must not depend on the batch.
Under greedy decoding a noise floor above `GREEDY_NOISE_LIMIT` (0.01, checked first on 400 probe items) therefore raises
`StudyError` with no override. For a 0.5B model with batches of 64, greedy bf16 decoding changed about 12% of GSM8K
scores between batch orders (float32 changed 0.03%), and studies use a float32 protocol or `max_batch_size=1`. An item
with a baseline error row is excluded from its pool for the whole study.

A constraint's sizes follow from its margin and its design variance (`design_variance`), which is the baseline's
search-pool score variance and at least `floor_variance(margin) = margin * (1 - margin)`. The last look is
`last_look(margin=, variance=)`, the looks are `look_sizes(last)` (which `StudySession.looks` caches), and the ladder
block is as large as the first look. A feasible study has room for two rounds of blocks in each search pool, and each
guard band is at most `BAND_SHARE` (60%) of its margin. `allow_infeasible=True` runs an infeasible study at low power
and waives no baseline check.

`screen` runs stages 0 to 3 (plan and support check, fits, a behavior quick look, a capability first look). `finish`
runs stage 4 (calibration of the solved axis, then confirmation of the chosen position in up to three looks on fresh
blocks, where a look rejects a position clearly over a margin and the second look one whose estimated loss is above it)
and stage 5 (the `Result`, recorded in the `Archive`). Every position of a solved axis is judged on its own evidence,
with no assumption about the shape of loss or behavior along the axis. Stages 2 and 3 look at the positions in the order
first, last, middle, then the others (`probe_order`), and reject only when no position shows a gain or every position is
clearly over a margin. Calibration drops a position whose first look (its ladder block) is clearly over a margin or
whose ladder blocks are over the cost or degeneracy limit, and confirms the others in the order of their gain bound. The
first one that passes is the operating point, and on a continuous axis a refinement may insert a position between it and
a neighbor that was dropped or failed. A ladder estimate alone therefore drops a position only when it is clearly over a
margin. A candidate is accepted only with an eligible operating point (confirmed on every constraint and within the cost
and degeneracy limits), and its fitness is the one-sided lower bound of its gain. Every estimate is a `Paired` table of
per-item differences against the baseline (`measurement/stats.py`), with the per-item variance taken as at least
`floor_variance(margin)`. A point's tables use only fresh-block items and exclude the ladder block its position was
chosen on. Results are memoized by candidate id, and a `budget` or `timeout` rejection is not recorded (its evidence
rows stay). A training or merge whose product the store does not contain is a structural fit, and `admit_fits` counts
the distinct ones against `max_structural_fits` and an allowance's `structural_fits`. A candidate that needs a fit
beyond them gets `budget` without exhausting the session. A phase with `structural_fits=0` therefore searches on stored
products only. `StudySession.view()` is the archive's view with `fits_open`, i.e., whether one more structural fit is
within the limits. `max_fit_seconds` (of `StudySession` and `run_study`) limits each training in a child process, and a
training stopped at the limit records its candidate as `fit_failed`. A later candidate with the same training then fails
at once. Once `settings.round_candidates` candidates are recorded in a round, `StudySession.advance` starts a new round
with new batches and re-scores the best accepted results. The session calls it at the start of `screen` and
`plan_finalists`, `run_study` after the initial candidates and after each phase, and `EvaluationService` once it has
sent the reply of an evaluation. The call whose result completed a round therefore returns before the re-scoring starts.
`plan_finalists()` spreads the finalists over levels and selects previously unevaluated search items for each one, and
`finalists(plan)` measures their levels again on exactly those items.

A search engine (`SearchEngine`, `search/engine.py`) proposes candidates and sees only the `ArchiveView` of
`session.view()` (per candidate the id, values, status, stage, reason, fitness, and a summary of search-pool aggregates,
and `fits_open`). With `fits_open` false, an engine proposes only the trained values of each training entry
(`Study.training_entries`), i.e., the combinations of its axes in the recorded candidates with the same entry that were
decided after stage 1 (`trained_values`). `RandomProposer` draws them, `GridProposer` holds back the other grid points
until fits are open again, and an OpenEvolve island whose phase starts closed is told them in its system message.
`run_study` (`search/loop.py`) owns the session of a run. It evaluates the initial candidates first (recorded with the
proposer name `"initial"`) and then runs each `SearchPhase` in order, each engine within its phase's allowance
(`StudySession.allowance`). An ask/tell engine subclasses `BaseProposer` (`GridProposer` and `RandomProposer`, both
restricted by `structures=`). An engine that runs its own loop starts an `EvaluationService`, which serves the session
over a local socket (JSON messages, one evaluation at a time, a deadline of 90% of the stage timeout). The final replies
of the service contain each result's level and the recent rejections of its structure. An engine that keeps a population
begins from `initial_population` (`search/population.py`), i.e., per structure its best accepted candidates in the
archive, then the initial candidates, then one candidate composed from the best values of other structures, then
`Study.default_candidate`. A later phase therefore begins from what earlier phases found. `OpenEvolveEngine`
(`search/openevolve.py`, `optimize` extra) runs OpenEvolve in a subprocess with one island per member of the initial
population. An island's system message describes its own structure and the goal (`search/description.py`). Its
population admits accepted candidates of that structure once each and keeps one per level bin, and programs do not
migrate between islands. `AgentProposer` (`search/proposers/agent.py`) asks a language model with tools (`describe`,
`archive`, `lookup`, `note`, and `propose`) for each batch, i.e., one episode per `ask`. An episode begins from
`<run_dir>/notes.md` and the view only. A run therefore resumes from its run directory and store, and the model's tool
calls are data that nothing executes. The service's replies and the agent's `archive` tool list rejections with the same
function (`rejection_digest`).

`certify` (`release/certify.py`) records the finalist plan before it measures any finalist, and it records the ordered
finalists before it reads any held-out item. It refuses held-out items that another study already used (unless
`allow_reused_held_out`). Each finalist is evaluated on the whole held-out pool in pool order, and the finalists are
tested in order at the full error rate until the first failure. Each certified point is written as a frozen bundle
whose lock contains its `"certificate/3"` certificate, which includes the tested lower bound, every benchmark's noise
floor, and the content id of the weights of each structural control (`weights`). A bundle contains trained or merged
weights, while a loader's bundle refers to its path. See `docs/tutorials/optimize_steering_pipelines.md` for the
guide.

## Developer guide

### Adding a steering control

A method is a sub-package of its category directory with a three-file layout:

```
steerability/algorithms/<category>_control/<method_name>/
├── __init__.py    # exports STEERING_METHOD for registry discovery
├── args.py        # hyperparameter dataclass (single source of truth)
├── control.py     # the control class; all steering behavior lives here
└── utils/         # optional local helpers; use sparingly
```

A control class defined outside the toolkit tree is not discovered by the crawl and registers itself at import time
with `register_method(category, name, control_cls, args_cls)` from `algorithms/core/registry.py`, which is what lets it
appear in a `.spipe` manifest.

`args.py` defines every hyperparameter as a `BaseArgs` dataclass; use `__post_init__` for cross-field validation:

```python
from dataclasses import dataclass, field

from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class MyMethodArgs(BaseArgs):
    """Arguments for MyMethod."""

    strength: float = field(default=1.0, metadata={"help": "Scale applied to the steering edit."})

    def __post_init__(self):
        if self.strength <= 0:
            raise ValueError("strength must be positive.")
```

`control.py` subclasses the category base and sets `Args = MyMethodArgs`. The base constructor validates the args and
promotes every field to an instance attribute (`self.strength`), so the control class defines no `__init__` of its
own in the common case. Required hooks per category:

- **input**: `adapt(input_ids, runtime_kwargs) -> input_ids` (required); optionally `adapt_messages(messages,
  runtime_kwargs) -> messages | None` for pre-template chat editing. A non-`None` return from `adapt_messages` skips
  that control's token-level `adapt` for the call, so implementing both does not double-apply.
- **structural**: `steer(model, tokenizer, **kwargs) -> PreTrainedModel`; return the new or modified model. The
  TRL wrappers forward `training_args` verbatim to the installed TRL config, so a convenience field loses to a
  `training_args` entry of the same name, and a key the config does not declare raises at construction. A LoRA
  `target_modules` list of module-name suffixes is scoped at steer time to the resolved decoder stack; a regex string
  is passed to PEFT unchanged.
- **state**: residual-stream methods subclass `InterventionControl` and declare an unbound intervention template
  in `_configure()`: a tuple of `Intervention` objects from `state_control/common/specs.py`, each naming layers or
  a selector, a transform possibly carrying an `ArtifactSource`, a `TokenScope`, and an optional gate. The base
  `steer()` binds the template, `build_hooks` compiles it to torch hooks per generation, and `lower_interventions`
  compiles it to an `InterventionSpec` per steer, so the control contains no hook code, no per-generation state,
  and no backend knowledge. Methods hooking other mechanisms subclass `HookControl` and implement
  `get_hooks(input_ids, runtime_kwargs, **kwargs) -> {"pre": [...], "forward": [...], "backward": [...]}`, where
  each spec is `{"module": <dotted submodule path>, "hook_func": <callable>}`, fully re-deriving per-generation
  state on every call. The session that executes forwards owns registration.
- **output**, step-level: `get_logits_processors(...)` and `get_stopping_criteria(...)`, returning fresh instances on
  each call. Loop-owning methods subclass `DecodingDriver` and implement `decode(input_ids, attention_mask, model,
  logits_processors, stopping_criteria, runtime_kwargs, session=None, **gen_kwargs)`, applying the received stacks
  at every scoring step. `decode` returns `[B * n, T + L]` ids, where `n` is the caller's `num_return_sequences`
  (default 1). Each row is the padded prompt row as received followed by one candidate's continuation, right-padded,
  in row-major, candidate-minor order. The pipeline always passes `session=` (a `SteeredSession` containing the
  generation's steering entries and its `cancel_token`); drivers issue their rollouts through it, and `model` is None
  on backends without a loaded model. `session_generate_items(session, rows, eos_token_ids=(), **gen_kwargs)` (in
  `core/execution/session_utils.py`) runs one session call over unpadded rows and returns one
  `(continuation_ids, finish_reason)` per row; pass the generation config's eos ids as `eos_token_ids` so that an eos
  sharing the pad id is kept.
  `max_rollouts_per_query()` bounds the rollouts per row and candidate.
- **all categories**: optional `steer()` for one-time preparation and `cleanup()` for releasing resources. The
  pipeline passes `session=` (a `SteeringSession` on the steering backend) into `steer()`; controls that only need
  structural facts read `session.layout` rather than the live model, and fitting call sites accept `session=` for
  capture-backed extraction.

Declare the class attributes the pipeline reads:

- `supports_batching` (default `False`; set `True` only when the control is batch-safe)
- `enabled`
- `RUNTIME_KWARGS_SCHEMA`: a list of `{"name": ...}` entries; declare `scope` on every entry, `"row"` for a
  per-prompt value delivered row-aligned in batched calls or `"call"` for one value per call
- for output controls, `include_in_scoring` and `same_model_forwards`
- for decoding drivers, `whole_batch_rollouts` (default `True`; set `False` when a session call covers a subset of
  the rows or rows at different stream lengths, as the phased and search drivers do)

Backend support is declared through `requirements()`. The default (`IN_PROCESS_TORCH` at generate) is honest for a
new control and keeps it Hugging Face-only; do not widen it speculatively. An `InterventionControl` derives its
requirements from the template: generate offers the intervention-spec alternative exactly when every component has
a wire form (`Intervention.wire_kinds()` reads component and source declarations before `steer()`), and score offers
it when, in addition, every token scope has an exact prompt-logprob form (`all`, `from_position`, `after_prompt`) and
no intervention is gated. Components describe their own wire form (`wire_kind` class attribute, `export()` per
configuration), and the equivalence of hooks and specs is pinned by `tests/core/test_spec_hook_equivalence.py`. An
output control whose behavior is sampling-expressible lowers via `export_generation_params()`, a declarative
constraint via `export_constraint()`, and an engine-hosted per-step processor via `export_processor_spec()`.

A control's steer step declares one of four access levels via `steer_access()`: `facts` (layout and tokenizer),
`rollouts` (generate and score through the session), `capture` (hidden states), or `module` (the model as a live
`torch.nn.Module`). Declare the highest rung your steer touches; intervention templates derive it from their
sources, and structural controls that train or merge are `module`. The artifact loaders (`load_checkpoint`,
`load_lora`) declare `facts`, since their artifact is part of the configuration: on an engine backend they hand it to
the engine without a stage, and join a stage only when a later staged step needs their weights. The pipeline hands
your `steer()` a session scoped to that rung (and the model itself only at `module`, or to any structural control,
since structural controls thread the model) and arranges residency: on an engine backend, module-level steps
run on a temporary in-process model that is freed before the engine starts, with exported artifacts as the handoff.
Do not hold the model past `steer()` unless your generate phase requires `IN_PROCESS_TORCH`. Generate- and
score-phase requirements are unchanged.

A control whose steer step produces fits or other state must also say how it freezes into a `.spipe`, through four
methods:

- `steer_fits()` lists the fit artifacts the step will produce as `(artifact, artifact_class)` pairs. The steer plan
  reads it, and `"calibrated"` artifacts get cross-venue notices.
- `export_state()` returns the steer-time products by logical name (a `SteeringVector`, `Probe`, `ProbeSet`,
  `Memory`, `CheckpointArtifact`, `LoRAArtifact`, tensor, or on-disk `Path`).
- `frozen_form(state)` returns the `(registry method key, constructor kwargs)` of a constructor-valid frozen form,
  which may be the control's own class (`caa`) or another registered method (`activation_adapter`, `load_lora`).
- `fit_identity()` returns the object whose canonical form digests the fit-relevant recipe inputs, so staleness
  detection excludes inert application parameters.

`InterventionControl` derives all four from its template, and the TRL and MergeKit wrappers freeze to `load_lora` /
`load_checkpoint`. A control that produces state and overrides none of them raises `NotFreezableError` at freeze.
Controls whose recipe is their frozen form need nothing.

A structural control that trains or merges has one fit, its product, and must resolve it through the fit cache
(`resolve_product` in `algorithms/core/fits.py`, with a `fit` callable that trains and returns `export_state()`). A
`PipelineFactory` then trains it once per fit key and installs the stored product at every later steer. While the
active cache contains the product (`product_is_stored`), the control's `steer_access()` is `FACTS` and its
`steer_fits()` is empty. Its `export_artifact()` and `export_state()` report the stored product as an artifact object
(`stored_product` reads it before `steer()`), which a bundle then contains by value. A control that installs an
unmerged adapter implements `restore(model)`, which removes the adapter again, and its recipes then run on the
factory's shared base model.

`__init__.py` exports the discovery dict:

```python
from .args import MyMethodArgs
from .control import MyMethod

STEERING_METHOD = {
    "category": "state_control",
    "name": "my_method",
    "control": MyMethod,
    "args": MyMethodArgs,
}
```

The registry crawls the category directories at import time, requires the `name`, `control`, and `args` keys, and
rejects duplicate names. A dependency a control needs goes in core when it installs everywhere the toolkit does without
conflict (xgrammar, trl, peft). It gets its own extra when it conflicts with other dependencies, is platform-limited,
or is very large; import it through `steerability.utils.optional.require("<module>")` at the module boundary and map it
in `OPTIONAL_MODULE_EXTRAS` (`steerability/utils/optional.py`), and discovery then skips the method with an actionable
hint when the dependency is absent instead of failing. A dependency a control runs without (an alternative estimator or
an enhancement) is not declared at all; raise a `ModuleNotFoundError` naming the package and its tested range, as CPO's
`use_dml` does. A package that exports no `STEERING_METHOD` is skipped, which lets a method whose upstream API is
missing from an installed dependency log a hint and skip its own registration rather than fail the crawl.

### Generics before new machinery

Before writing new components, check the category's `common/` library and compose from it:

- **state**: transforms (`AdditiveTransform`, `ProjectionTransform`, `RotationTransform`,
  `HeadAdditiveTransform`, `NormPreservingTransform`, `AlignmentAdaptiveTransform`); artifact sources
  (`ContrastiveFit`, `SinglePairFit`, `ConditionPointSearch`, `LayerFilteredFit`, `VerifiedPrecomputed`, and the
  PASTA-local `HeadProfile` for rollout-scored head selection; each declares its `access` and `artifact_class`);
  estimators (`MeanDifferenceEstimator`,
  `ContrastiveDirectionEstimator`, `SinglePairEstimator`, `SteeringPlaneEstimator`, and `estimator_for(method)`,
  which maps a `VectorTrainSpec.method` to its estimator); gating (`Gate` over an
  `Evidence` and a rule; readouts `AffineReadout`, `CosineReadout`, `ProjectedCosineReadout`, `CallableReadout`;
  rules `SumThreshold`, `PerKeyThreshold`; `gate_from_probe`); selectors (`FixedLayerSelector`,
  `FractionalDepthSelector`, `TopKHeadSelector`, `ConditionPointSelector`); token scopes; `SteeringVector`; and
  `TransformHookRuntime`.
- **output**: `SearchDriver` (propose, score, keep, iterate) and `PhasedDriver` (`Fixed` / `Generated` phase plans);
  processors (`PrefixKeyedProcessor` base, constraint, contrastive mixture, value-guided); sequence scorers
  (`RewardModelScorer`, `MajorityVoteScorer`, `SampleSequenceScorer` over a per-row `SampleScorer`); value
  functions (callable, classifier, reward model, subspace margin); criteria (`StopOnSubstring`, `BudgetTokens`);
  and KV-cache utilities.
- **input**: memories (text, pool); formatters (system prompt, few-shot block, prepend, chat-template slot);
  proposers (LLM meta-prompt, retrieval); scorers; selectors (random, top-k, MMR, dense retrieval); and
  `RolloutBudget` / `ParetoFrontier` utilities.
- **detection**: probes live in `core/internals/probes` (`fit_probe`, `calibrate_bias`, `ProbeSet`, and
  `ProbeSetFit` for fitting deferred to steer time). Prefer these over ad hoc classifiers, and consume their
  decisions through `Probe.as_gate()` for gated interventions or `routed_decoding`'s `Router` (ordered `Route`s
  with `P(name)` predicates over the actions `respond`, `prefix`, `generate`) for routing.

Published methods are frequently presets over generics (`deal` presets `SearchDriver`; `budget_forcing`
presets `PhasedDriver`; `caa` composes an estimator with `AdditiveTransform`). Driver presets map their `Args` onto
the generic base's fields in `_configure()` rather than overriding `__init__`; follow that pattern for new decoding
methods. Before writing a new state control, check whether an `ActivationAdapter` configuration (transform, layer
selector, gate, token scope) already covers the behavior.

### Authoring evaluation tasks

Target-behavior evaluations are ordinary Inspect `Task`s; the toolkit ships no task, scorer, or metric classes of its
own (working examples are defined inside the study notebooks under `examples/notebooks/studies/`).

A task whose samples carry per-sample steering inputs puts them on `Sample.metadata` as `{"runtime_kwargs": {...}}`,
with each value in the consuming control's per-row form, and uses `runtime_kwargs_solver()` from
`steerability/evaluation/solvers.py` as its generation step. Each key must be declared `"row"`-scoped in the
consuming control's `RUNTIME_KWARGS_SCHEMA` (a `"call"`-scoped key is rejected per sample), and a key that no enabled
control of an arm declares is inert on that arm, so the empty baseline shares the task.

Tasks with model-graded scorers take the grader model through their own arguments (`task_args`); the grader is never
the pipeline under evaluation. Controls that consume a per-row reward accept a `SampleScorer`; use
`sample_scorer_from_inspect` to drive them with an Inspect scorer. See
`docs/tutorials/evaluate_steering_pipelines.md` for the authoring guide.

### Testing

Fixtures in `tests/conftest.py` provide a parametrized `device` fixture (`cpu` / `cuda` / `mps`, skipping unavailable
devices), a session-scoped `model_and_tokenizer` fixture over the tiny models in `tests/utils/ci_models.yaml`, mock
controls for every category, and mock model/tokenizer factories.

A new control needs `tests/controls/test_<method>.py` following the existing pattern: a parameter grid expanded with
`build_param_grid()` (from `tests/utils/sweep.py`), then build the control, wrap it in a `SteeringPipeline`, `steer()`,
`generate()`, and assert on the output. Unit-test any new generics directly (`tests/controls/` for control components,
`tests/internals/` for the probes substrate, `tests/core/` for pipeline behavior).

`tests/controls/test_driver_candidates.py` runs every registered `DecodingDriver` preset on a batch of two prompts of
different lengths with `n=2` and checks the output layout. A new preset needs a minimal configuration for the tiny
model in its `PRESET_KWARGS`, and the suite fails for a preset without one.

### Code style

- Python 3.12+ with modern typing (`list`, `dict`, `T | None`); line length 120; snake_case variables, PascalCase
  classes, UPPER_SNAKE_CASE constants; descriptive, not overly abbreviated, names.
- Comments describe current functionality only, in lowercase, with two spaces before inline comments
  (`a = 1  # some comment`) and no decorative formatting. Do not narrate edits or prior designs.
- Use a module logger (`logger = logging.getLogger(__name__)`) instead of `print` in library code.
- No compatibility shims or aliases. A changed behavior, name, or field replaces the previous one outright: no
  deprecated path, no re-export of an old name, no fallback that accepts the previous form, and no branch that keeps
  the old behavior when a new field is absent.
- Keep imports simple; use the optional-dependency guard rather than broad try/except import fallbacks. Import order
  is enforced by isort (black profile) via pre-commit.
- Read structural facts (`hidden_size`, `num_attention_heads`, `head_dim`, `num_hidden_layers`) through
  `text_config(model)` (from `steerability.algorithms.core.internals`), which returns the text sub-config on composite
  multimodal models; never read `model.config.hidden_size` directly, and never default a missing fact to `0`. Read
  head geometry with `config_head_geometry(text_cfg)` or `config_layer_head_dim(text_cfg, layer_id)` on a config and
  with `head_geometry(model, layout, layer_id)` on a loaded model rather than with
  `getattr(text_cfg, "head_dim", None)`, since a config that declares `head_dim` per layer (Gemma 4) raises an error on
  the global read (`ModelFacts.head_dim` is `None` for such a model). Build `ModelFacts` from a loaded model through
  `model_facts(model)`. Resolve decoder module paths through `resolve_model_layout(model)` rather than by matching
  `model.model.layers`, and scope LoRA adapter targets through `lora_target_pattern(target_modules, model)`.

### Docstrings and documentation

Docstrings use the Google format (`Args:`, `Returns:`, `Raises:`, `Attributes:`); mkdocstrings parses them for the
reference site, and lists render correctly only with a blank line before them. Wrap code identifiers in backticks.

State what the code currently does, with the factual guarantees plainly stated (shapes, dtypes, defaults, side
effects such as in-place mutation, raise conditions, lifecycle constraints). Keep the register neutral (no
intensifiers, evaluative adjectives, or rhetorical constructions) and do not use em-dashes. Describe behavior in
place rather than by analogy to another method. Cautionary content goes in the description body as plain prose before
`Args:`; `Warns:` is reserved for warnings the function emits at runtime. Control docstrings end with the paper
reference:

```
Reference:

    - "Steering Llama 2 via Contrastive Activation Addition"
      Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner
      [https://arxiv.org/abs/2312.06681](https://arxiv.org/abs/2312.06681)
```

A new method also needs its documentation surfaces updated: a reference page
`docs/reference/algorithms/<category>_control/<method>.md` (copy the mkdocstrings block from an existing page), a nav
entry in `docs/.nav.yml`, and a mention in the category's list in `docs/concepts/controls.md`.

### Notebooks

Each method gets a demonstration notebook: `examples/notebooks/algorithms/` for named methods,
`algorithms/generics/` for config-first controls, `algorithms/wrappers/` for the library wrappers (`trl`,
`mergekit`), `studies/` for use-case studies (run artifacts stay in that subfolder), `recipes/` for composite
workflows. Add an entry to `examples/index.md`.

Conventions: imports in a setup cell; explanation lives in markdown cells rather than code comments; one plot per
cell; no special characters in axis text; f-strings when titles reference variable values. Markdown prose is plain
technical reporting: narrate with "we", keep the register neutral, signpost with connectives ("Note that ...", "For
instance, ..."), write paragraphs rather than bolded bullet lists, and link the paper in the introduction when
demonstrating a published method.

### Definition of done

Before significant changes, check: does this fit the existing abstractions and patterns; does documentation need
updating; does it affect other parts of the system. A finished method contribution includes:

- [ ] the `args.py` / `control.py` / `__init__.py` triplet, importing cleanly with a valid `STEERING_METHOD` export
- [ ] honest `supports_batching`, `RUNTIME_KWARGS_SCHEMA`, and (output) `include_in_scoring` / `same_model_forwards`
- [ ] tests in `tests/controls/test_<method>.py` passing on CPU with the CI models (and, for a decoding driver
  preset, its `PRESET_KWARGS` entry in `tests/controls/test_driver_candidates.py`)
- [ ] a Google-style docstring ending with the paper reference
- [ ] reference page, `docs/.nav.yml` entry, and `docs/concepts/controls.md` mention
- [ ] a notebook plus its `examples/index.md` entry
- [ ] a `pyproject.toml` extra and `optional.py` entry for any new heavy dependency
- [ ] `pre-commit run --all-files` clean; commits signed off

## Invariants

Rules that hold regardless of task:

1. `steer()` must run before `generate()` or `compute_logprobs()`; it runs once per pipeline, and heavy work belongs
   there, not in control constructors. On engine backends the steer phase runs in residency phases. Stage-venued
   steps (module access, and capture where the engine serves none) run first on a temporary in-process model that
   is freed before the engine boots; session-venued steps then run through the engine session. The pipeline
   model's in-process weights and its engine-served weights never coexist.
2. Steering order is fixed (structural, input, state, output). List order within a category is the composition
   order, preserved within each residency phase: phases run module-first, and the only channel between steers is
   the pipeline model, which only stage-phase controls can touch. For state controls, entry order equals spec op
   order equals worker application order, so an in-process composition and its wire form apply edits in the same
   sequence.
3. The steer phase produces no support verdicts. Each control declares its steer step's model access via
   `steer_access()`, and scoped sessions enforce the declaration on every backend. A control may retain the
   pipeline model beyond `steer()` only if its generate phase requires `IN_PROCESS_TORCH`; on engine backends the
   free protocol verifies the staged weights are gone and raises naming any retaining control.
4. The decode loop does not compose. At most one enabled `DecodingDriver` exists per pipeline, and a driver must
   apply the received `logits_processors` and `stopping_criteria` at every scoring step of every forward pass it
   issues.
5. Logits processors behave as functions of `(prefix_ids, scores)`. Internal state is permitted only as memoization
   keyed on the prefix (subclass `PrefixKeyedProcessor`), and `get_logits_processors` returns fresh instances per
   call.
6. Extra forward passes through the pipeline's own model during decoding are wrapped in `auxiliary_pass()` (from
   `core/utils/auxiliary_pass.py`), and the component declares `same_model_forwards = True`.
7. Hooks exist only inside a session's execution of work (per item, or for the span of a driver decode the
   session hosts). Controls never register hooks and hold no model reference. Hooks travel exclusively as
   `HookEntry` contributions built by the pipeline.
8. Never mutate caller-supplied artifacts (steering vectors, probes, configs); clone before moving devices or
   normalizing.
9. One in-flight generation per control instance. Gate instances embedded in a control's interventions, and the
   per-row grammar state of a constraint automaton, contain per-generation decisions. Do not share control instances
   across concurrently running pipelines.
10. `generate()` returns continuation-only ids by default; never re-slice its result by prompt length.
11. `runtime_kwargs` is a single shared namespace per call. Declare consumed names and their `scope` in
    `RUNTIME_KWARGS_SCHEMA` (a row-scoped kwarg receives a row-aligned sequence in batched calls) and expect shared
    values on name collisions.
12. Declare `supports_batching=True` only when a control is safe under batched prompts; the pipeline and the
    Inspect model provider read it to choose between batched and per-example generation. A decoding driver whose
    session calls cover a subset of the rows, or rows at different stream lengths, declares
    `whole_batch_rollouts = False`.
13. A control's behavior has exactly one declarative statement (the adapted prompt, a structural artifact, an
    intervention tuple, or exported params/specs). Every backend consumes the highest representation it supports.
    Hooks are per-generation products of the pipeline and specs are per-steer products of it, and no code path
    reconstructs a control's configuration by inspecting another representation of it.
14. Prompt-relative scope kinds (`after_prompt`, `last_k`) are client-side sugar; their wire form inside a driver
    generation is absolute (`from_position` at the generation's original prompt boundary).
15. A control's freezable state is exactly `export_state()`; the frozen form returned by `frozen_form()` is
    constructor-valid for its declared method; and the recipe is never discarded (a frozen `.spipe` entry keeps the
    original args for provenance and `thaw()`).
16. Identity is the codec's encoding (`content_id`), which reads a dataclass's `init=True` fields only and fails
    closed on values without a serialized form. Steering leaves nothing on a constructor argument or a fit source that
    anything reads later, and a control's `frozen_form(state)` depends only on its args and `state`, so a recipe's id
    and its fit digests are the same before and after `steer()`.
17. `Output.finish_reasons` has one entry per row of `output_ids`, enforced at construction together with the
    `[batch, seq]` shape of `output_ids`.
18. Concurrent `generate()` calls serialize their prompt phase under the pipeline's prompt lock and dispatch
    concurrently only when `supports_concurrent_generate` is True; a decoding driver's `decode` never runs
    concurrently with another generation of the same pipeline. A cancelled `CancelToken` stops its generation at the
    session's stated granularity and surfaces as `GenerationCancelled`.
19. In a study, a decision is never judged on the items it was made on, and nothing item-level reaches a proposer.
    `Result.summary` and `ArchiveView` contain numbers and short enumerated strings computed from search-pool
    aggregates, no function in `optimization/search/` receives item text or item-level rows, and held-out items are
    read only by `certify`, after the finalists are recorded.
20. Under a fit cache, the fit of a structural control runs once per fit key. Its product is stored in the store's
    artifacts, and every later steer installs the stored product.
21. `PipelineFactory.steered` yields only pipelines whose structural products were installed from the store. No
    evidence is generated, and no other control is fitted, on weights that are not a stored product.
22. A certified bundle of a candidate with trained or merged weights contains them, and it loads after the store is
    deleted.

## Pointers

- `docs/concepts/`: conceptual guides on controls (with each control's `Backends` line), steering pipelines,
  probes, and the `.spipe` format.
- `docs/tutorials/`: step-by-step guides for adding a steering method (with per-category walkthroughs under
  `add_method_by_category/`), evaluating steering pipelines, and optimizing steering pipelines under constraints.
- `examples/notebooks/`: runnable references for every method, the generic controls, and use-case studies.
- `tests/index.md`: test-suite layout and the pattern for adding control tests.
- Hosted documentation: <https://generative-computing.github.io/steerability/>.
