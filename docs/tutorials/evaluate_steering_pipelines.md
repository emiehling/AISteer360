# Evaluate steering pipelines

The toolkit evaluates steering pipelines on [Inspect AI](https://inspect.aisi.org.uk/) (UK AI
Security Institute) and its benchmark catalog
[`inspect_evals`](https://github.com/UKGovernmentBEIS/inspect_evals). This facilitates the evaluation
of both the target behavior of a pipeline (did instruction following ability improve?) and its
off-target effects (degradation in math ability, coding ability, general knowledge, etc.).

Note that the evaluation is on the entire pipeline rather than the model alone, since a steering
pipeline generally includes modifications to the input/prompt and the decoding process in addition to
model-level modifications (weights, activations). Additionally, evaluation must be done on open-ended
generations rather than logprobs. One of the primary reasons for this is output controls, i.e., a
decoding driver induces a distribution over sequences without a per-token conditional.

## The model provider

The `as_inspect_model` function wraps a steered pipeline as an Inspect model:

```python
from inspect_ai import eval as inspect_eval
from steerability.evaluation.provider import ProviderOptions, as_inspect_model

pipeline.steer()
model = as_inspect_model(pipeline, options=ProviderOptions(max_batch_size=8))
logs = inspect_eval("inspect_evals/gsm8k", model=model, limit=100, temperature=0)
```

where the `ProviderOptions` dataclass contains the provider's configuration:

- `runtime_kwargs`: static runtime kwargs applied to every request
- `chat_template_kwargs`: template kwargs for the messages path
- `max_batch_size`: the batching ceiling
- `default_max_tokens`: the default `max_tokens`
- `reasoning_tags`: the tags used to split thinking from the answer before scoring (`reasoning_tags=None`
  disables the split)
- `reasoning_opened_at_start`: whether the chat template's generation prompt already opens the reasoning block (a
  continuation that contains neither tag is then read as unclosed reasoning rather than as an answer)
- `reasoning_split`: how the split locates the tags (`"text"` on the decoded continuation, `"tokens"` on the
  continuation ids, or the default `"auto"`, which picks `"tokens"` when `skip_special_tokens=True`
  strips a tag)
- `on_unsupported_param`: the policy for `GenerateConfig` parameters the pipeline cannot honor (`"raise"` by
  default or `"warn"`)
- `seed_scope`: how a seeded dispatch applies its seed to the rows of a batch (see below)
- `max_concurrency`: selects per-request mode on the vLLM backends (see below)

The provider decides how to deliver prompts to the pipeline when it is constructed. With a chat-templated
tokenizer, prompts are dispatched as `messages=` and every input control participates as it does in
deployment. Base models without a chat template (a common subject of capability measurements) are
evaluated through a text path instead, i.e., each conversation is rendered to plain text and dispatched
as `text=`. On the text path `adapt_messages` never runs (token-level `adapt` still does). Since the same
controls behave differently on the two paths, the provider warns once at construction.

### Scope

The provider is generation-only. Requests that include tools or tool messages, logprob parameters
(`logprobs`, `top_logprobs`, `prompt_logprobs`), or multimodal content raise an error that
explains the restriction. This limits evaluation to non-agentic tasks, which form the majority of
`inspect_evals`. Note that `GenerateConfig.response_schema` is not translated into a
`constrained_decoding` control because that would inject a control the configuration did not
declare. It follows the unsupported-parameter policy instead.

## Batching and reproducibility

Inspect issues one asynchronous request per sample and keeps many outstanding at once, while the
pipeline runs one (possibly batched) generation at a time. The provider bridges the two by collecting
concurrent requests into batched `pipeline.generate()` calls, filling the next batch while the current
generation runs. It advertises `max_connections` equal to its effective batch ceiling, and Inspect's
`max_samples` defaults to that value, which fills batches exactly. Note that `max_connections` should
not be set below `max_batch_size`.

Batching applies only to arms whose enabled controls all declare `supports_batching=True`, and the
provider otherwise clamps the batch size to 1. Input, state, and structural arms batch. Among output
controls, only `phased_decoding`, `routed_decoding`, and `stopping_rules` declare batch safety (`rad`,
`value_guidance`, and `constrained_decoding` compute it), and most driver-based arms therefore run one
sample at a time. An arm that combines a state control with `phased_decoding` or `routed_decoding` also
runs one sample at a time, since these drivers issue session calls over subsets of the batch. Rows
of one batch need not share a prompt length. A ragged batch (e.g., few-shot with per-row exemplar
draws) is left-packed on the Hugging Face backend, and each row's continuation is predicted from its
own last real token rather than from a trailing pad.

We recommend greedy decoding (`temperature=0`) as the default since it is the norm for capability
benchmarks and avoids seed sensitivity. Which samples are evaluated is fixed by the selection, independent
of batch composition. Under sampling, `seed_scope` in `ProviderOptions` sets how seeds are applied. The
default `"dispatch"` scope seeds each batch as a whole and decodes it in one pass, and the `"item"`
scope derives a seed per row and decodes one row at a time. Note that every `pipeline.generate()` call
on the Hugging Face backend opens a new session. This means that under `"dispatch"`, every dispatch with
the same seed starts sampling from the same random state (common random numbers across prompts). Bitwise
reproducibility of stochastic sampling is not preserved under concurrency, because a sample's batch
membership and row index depend on the order in which requests arrive. A bitwise-reproducible stochastic
run requires `max_batch_size=1` and Inspect `max_connections=1`. Even greedy outputs can differ across
batch compositions, since padded-batch numerics differ from single-item numerics on some kernels.
Trial-to-trial variation under sampling is therefore measured rather than eliminated, which is the role
of `num_trials` and the per-metric standard error.

### Per-request mode

On the vLLM backends we can instead set `max_concurrency` in `ProviderOptions`. The provider then dispatches each
request at once as its own pipeline call, with at most `max_concurrency` calls in flight. The engine admits each
request into its running batch as soon as there is room. This means that a short generation returns without
waiting for the longest generation of a batch. Per-request mode applies when the pipeline's `supports_concurrent_generate` is
`True`, i.e., on a vLLM backend without an enabled decoding driver. For other pipelines the provider logs at info
level and keeps batching. The provider's `dispatch_mode` property reports the mode in effect. Since
`max_concurrency` changes throughput and not results, it is not part of a protocol's identity.

```python
options = ProviderOptions(max_concurrency=64)
model = as_inspect_model(pipeline, options=options)
```

The engine's `max_num_seqs` (an `engine_kwargs` entry) bounds the running batch, and the engine queues the requests
beyond it. This means that `max_concurrency` should be at least `max_num_seqs` to keep the batch full. Note that
`max_num_seqs` should be sized against the KV cache budget (`gpu_memory_utilization`, `max_model_len`), since a
running batch that outgrows the cache preempts sequences. The engine's periodic stats log reports preemptions. It is
on by default and can be turned off with `disable_log_stats=True` in `engine_kwargs`.

Per-request mode changes the reproducibility contract. A seeded request samples under a seed derived from the base
seed and the request itself (its prompt and per-sample runtime kwargs). This means that a seeded request reproduces
regardless of concurrency and arrival order, and two requests with the same prompt and per-sample kwargs sample the
same stream. The `seed_scope` setting has no effect in this mode, since every dispatch is one request. A request
that Inspect cancels also aborts its generation on the engine.

## Benchmarks and the runner

An evaluation has four parts. A `Protocol` fixes the measurement context: the base model and its revision, the
tokenizer, load arguments, the backend, the fit policy, the provider options, the decoding defaults, the seed, and the
package versions. Pipelines are described as recipes (`Recipe`) or as templates (`Template`), whose arguments may be
ranges. A benchmark defines items and their scoring: `InspectBenchmark` runs one Inspect task, and `PromptBenchmark`
scores a mapping of prompts with a `SampleScorer`. `SteeringEval` then measures every point of every pipeline on every
benchmark, over trials:

```python
from steerability.algorithms.core.protocol import Protocol
from steerability.algorithms.core.recipe import BASELINE, Choice, Template, entry
from steerability.algorithms.state_control.pasta.control import PASTA
from steerability.evaluation.items import InspectBenchmark
from steerability.evaluation.provider import ProviderOptions
from steerability.evaluation.runner import SteeringEval

protocol = Protocol.create(
    "meta-llama/Llama-3.1-8B-Instruct",
    provider=ProviderOptions(max_batch_size=8),
    generate_defaults={"temperature": 0, "max_tokens": 512},
    seed=7,
)
pipelines = {
    "baseline": BASELINE,
    "pasta": Template((entry(PASTA, head_config=head_map, alpha=Choice((10.0, 100.0))),)),
}
benchmarks = [
    InspectBenchmark("inspect_evals/gsm8k", scorer="match"),
    InspectBenchmark(f"{TASK_FILE}@target_qa", scorer="includes"),
]
evaluation = SteeringEval(protocol, pipelines, benchmarks, limit=200, num_trials=3)
evaluation.run("runs/exp1")
```

A `Template` describes a sweep in its arguments. A `Choice` lists values, an `Interval` spans a numeric range (with
`log=True` for geometric spacing), and a `Text` admits strings of bounded length. Each range is an axis of its own,
and ranges that name the same `axis` move together, e.g., `alpha=Interval(5, 20, axis="strength")` and
`num_heads=Choice((16, 32, 48, 64), ordered=True, axis="strength")` give four points rather than sixteen. The runner
expands every template as a grid, and `expand(template, strategy="random", samples=..., seed=...)` samples points
instead. Note that a recipe's id does not depend on key order, explicit defaults, entry names, or the order of entries
across categories, so two descriptions of one pipeline are one point.

The run keeps everything it measures in a store (`save_dir/store`, or the `store=` argument to share one store across
runs). The store records fits per fit source, so a sweep that only changes an application argument (a multiplier, a
layer, a threshold) fits once, and a change of decoding settings refits nothing. It also records one evidence row per
(protocol, resolved recipe, benchmark, item, trial). A second `run` on the same store therefore generates only what has
no evidence row, and a larger selection generates only the new items. Before any model work the runner checks backend
support for every point (`on_unsupported="raise"` or `"skip"`), and it runs the fits every point needs once
(`PipelineFactory.prepare`), so an engine backend boots once and serves every recipe.

The runner writes `save_dir/protocol.json` with the protocol's id, identity, and the facts observed on the loaded
model, and `save_dir/points.json` with each point's pipeline, recipe id, resolved id, and assignment. A `save_dir`
written under another protocol is refused, which means that a changed protocol needs a new `save_dir`. Credentials in
backend options never enter either file.

Every point and every trial scores the identical items. The selection is `items={benchmark name: [item ids]}` when
given, else the first `limit` items in dataset order, else every item. Taking the first `N` items is deterministic
across points, which paired comparison requires, but it is a biased estimate of the full-benchmark score. Trial `t` is
replicate `t` of every item. Under sampling a replicate selects a different seed, and under greedy decoding a
`PromptBenchmark` repeats each generation in a different batch order, so two replicates differ only by the numerical
noise of batching.

### Frames

`samples_frame` gives one row per (pipeline, recipe, trial, benchmark, item), with the item's `status` and `tokens`,
one column per requested score key (`"<scorer>"` or `"<scorer>/<key>"` for one key of a dict-valued score), the
requested metadata columns, optionally the `input`, `target`, and `completion` text, and one column per template path:

```python
samples = evaluation.samples_frame(
    {"correct": "choice"}, benchmark=f"{TASK_FILE}@target_qa", metadata_keys=["question_id"], include_text=True,
)
```

`runs_frame` gives one row per (pipeline, recipe, trial) of one benchmark with the mean of each score key over the
trial's `ok` items, the counts `n` and `errors`, and the assignment columns. The `summarize_runs` function (in
`steerability.evaluation.runner`) then aggregates the trials of each recipe into `{metric}_mean`, `{metric}_std`, and
`{metric}_sem` columns and keeps the columns listed in `param_cols`:

```python
from steerability.evaluation.runner import summarize_runs

runs = evaluation.runs_frame({"accuracy": "match"}, benchmark="inspect_evals/gsm8k")
summary = summarize_runs(runs, metric_cols=["accuracy"], param_cols=["pasta.alpha"])
```

A metric that is not a mean over items (e.g., a positional bias over shuffled variants of one question) is computed
from `samples_frame` and merged into the runs frame on `["pipeline", "recipe_id", "trial_id"]`. The module-level
functions `samples_frame(save_dir, ...)` and `runs_frame(save_dir, ...)` read a finished run from its directory.

The plotting functions in `steerability.evaluation.plotting` (part of the `eval` extra) take the rows of the swept
configurations from this summary frame and read the swept argument from the column given by `sweep_col`, e.g.,
`plot_sensitivity` for a metric against the swept argument and `plot_tradeoff` for two metrics against each other.
Fixed arms such as the baseline are passed through `compare_to_pipelines` and drawn as references.

An `InspectBenchmark` writes the Inspect logs of its runs under the store's `logs/` directory. These contain the
per-sample transcripts, which Inspect's log viewer and the `inspect_ai.analysis` dataframes read directly, but the
evidence rows are the record of the run. A `PromptBenchmark` generates its items directly through the pipeline and
stores each rollout by a key known before generation, so a new scorer version re-scores the stored completions and
generates nothing. Its scorer must be a module-level callable marked with `versioned("<version>")` (from
`steerability.algorithms.core.identity`), since the scorer's version is part of the benchmark's identity.

An `InspectBenchmark` can also generate its items directly. When it is prepared, it runs the task once on Inspect's
`mockllm/model` to record each sample's model request. A task without a sandbox whose samples each call the model once
(without tools) is capturable. On the first run of a capturable benchmark under a protocol, a parity probe evaluates up
to 16 items both through Inspect and directly under greedy decoding. If every completion and score agrees, the
benchmark is generated directly under that protocol, with stored rollouts and batches in a fixed order. Otherwise it
runs through Inspect, and `benchmark.mode_detail(protocol)` gives the first difference. Note that direct scoring runs
the task's scorers on a state with one user turn and one assistant turn. This means that a solver that changes the
output after generation (such as `multiple_choice()`) or a scorer that reads the solver's state makes the benchmark run
through Inspect. Direct generation also samples each request on its own. Under a sampling protocol, a pipeline
without concurrent generation (e.g., on the Hugging Face backend) therefore runs through Inspect, which batches
concurrent requests.

Tasks with model-graded scorers need a grader model supplied through the task's own arguments
(`task_args`). The grader must be a separate model (an API model or a second local model) and
never the pipeline under evaluation, since self-grading is circular and grader traffic would
compete with evaluation traffic inside the collator. Also note that a local grader shares the GPU
with the pipeline. An API grader is preferable unless memory headroom is planned for both models.

## Authoring target-behavior tasks

Custom target-behavior evaluations are ordinary Inspect tasks, and the toolkit provides no task,
scorer, or metric classes of its own. Two working examples are in the study notebooks. The
`examples/notebooks/studies/commonsense_mcqa/` task defines a shuffled-choice MCQA task with a custom
positional-bias metric, and the `examples/notebooks/studies/instruction_following/` task passes each
prompt's instruction lines as per-sample runtime kwargs for a PASTA arm and scores every response
with both the strict IFEval checker and a local reward model loaded inside the scorer.

Controls that take per-generation parameters receive them through two tiers of runtime kwargs.
Static kwargs (`ProviderOptions.runtime_kwargs`) apply to every request. They suit catalog tasks,
whose datasets contain no steering columns, and any kwarg that is a property of the arm rather than
the sample. Per-sample kwargs are stored in `Sample.metadata` and delivered by the provided
`runtime_kwargs_solver`, which performs the sample's generation in place of a bare `generate()` in
the solver chain:

```python
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import includes
from steerability.evaluation.solvers import runtime_kwargs_solver

@task
def target_qa() -> Task:
    samples = [
        Sample(
            input="Answer with the city name only. Which city is the Eiffel Tower in?",
            target="Paris",
            metadata={"runtime_kwargs": {"substrings": ["Answer with the city name only."]}},
        ),
    ]
    return Task(dataset=MemoryDataset(samples), solver=[runtime_kwargs_solver()], scorer=includes())
```

The provider interprets every runtime kwarg, on either tier, against the arm's enabled controls. A
key declared `"row"`-scoped (a per-prompt value) reaches the control as one value per prompt row.
Per-sample values are collated row by row across a batched dispatch, and a static value is broadcast
to every row. A key declared `"call"`-scoped (one value per generate call) is passed through
unchanged and may only be delivered statically. A key that no enabled control of the arm declares is
dropped from the call and logged once per provider, which allows one task to contain the steering
inputs of every arm in an experiment, including the empty baseline. For PASTA's `substrings` the
per-row form is one `list[str]`, on both tiers. Tasks without the solver, including the entire
`inspect_evals` catalog, receive static kwargs only.

## Inspect scorers as rewards inside controls

Controls that optimize or rerank against a per-row score (PRewrite, CPO, GEPA, `best_of_n`,
`search_decoding`) consume a `SampleScorer`, a callable `(response, row) -> float` where the row
contains `"input"`, optionally `"reference"`, and any other dataset columns.
`sample_scorer_from_inspect` adapts any Inspect scorer into that form:

```python
from inspect_ai.scorer import model_graded_fact
from steerability.evaluation.scorers import sample_scorer_from_inspect

row_scorer = sample_scorer_from_inspect(model_graded_fact(model="openai/gpt-4o-mini"))
prewrite = PRewrite(initial_instruction="...", dev_set=dev_rows, row_scorer=row_scorer)
```

The adapter bridges Inspect's asynchronous scorers into synchronous control code. It works from plain
synchronous code, from inside the provider's dispatch thread, and from inside a running asyncio
event loop (a notebook), where it applies the same `nest_asyncio2` re-entry that Inspect uses.
Inside a running trio task it raises an error instead, since re-entry is impossible there. Note that a
model-graded scorer used this way runs grader traffic from inside a control's `steer()` or decode
loop. We recommend running optimizers with model-graded rewards from scripts.

The `PRewrite` example above rewards at steer time, from a fixed development set. A reranking
driver instead rewards at generate time, once per sample, and its `SampleScorer` therefore needs that
sample's row. The `SearchDriver` presets (`DeAL`, `best_of_n`, `search_decoding`) read a `reward_params`
runtime kwarg for this, declared `"row"`-scoped, and `SampleSequenceScorer` merges it into the row
the scorer sees (`{"input": prompt, **reward_params}`). We store each sample's reference on
`Sample.metadata` as one mapping and deliver it with `runtime_kwargs_solver`, exactly as for PASTA's
`substrings`:

```python
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import includes
from steerability.evaluation.scorers import sample_scorer_from_inspect
from steerability.evaluation.solvers import runtime_kwargs_solver
from steerability.algorithms.output_control.best_of_n.control import BestOfN
from steerability.algorithms.output_control.common.scorers.sample import SampleSequenceScorer

row_scorer = sample_scorer_from_inspect(includes())          # reads row["reference"]
control = BestOfN(n=8, scorer=SampleSequenceScorer(row_scorer))

@task
def reranked_qa() -> Task:
    samples = [
        Sample(
            input="Which city is the Eiffel Tower in?",
            target="Paris",
            metadata={"runtime_kwargs": {"reward_params": {"reference": "Paris"}}},
        ),
    ]
    return Task(dataset=MemoryDataset(samples), solver=[runtime_kwargs_solver()], scorer=includes())
```

Note that the `n=8` argument of `BestOfN` is the number of samples in one search, and a request for
several candidates per sample (Inspect's `num_choices`) runs one search per candidate. Also note that
since the collator refuses one runtime-kwarg name on both tiers, an arm that passes per-sample
references through `reward_params` cannot also pass per-arm reward hyperparameters under the same
name. Put those in the scorer's constructor instead.
