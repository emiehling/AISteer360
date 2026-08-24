# Evaluate steering pipelines

The toolkit evaluates steering pipelines on [Inspect AI](https://inspect.aisi.org.uk/) (UK AI
Security Institute) and its benchmark catalog
[`inspect_evals`](https://github.com/UKGovernmentBEIS/inspect_evals). This facilitates the evaluation
of both the target behavior of a pipeline (did instruction following ability improve?) as well as its
off-target effects (degradation in math ability, coding ability, general knowledge, etc.).

Note that the evaluation is on the entire pipeline (not just the model). This is because a
steering pipeline is generally more than just a model (contains modifications to the input/prompt and
decoding process in addition to model-level modifications like weights, activations, etc.). Additionally,
evaluation must be done on open-ended generations (rather than logprobs). One of the primary reasons
for this is because of output controls, i.e., a decoding driver induces a distribution over sequences
without a per-token conditional.

## The model provider

The `as_inspect_model` function wraps a steered pipeline as an Inspect model:

```python
from inspect_ai import eval as inspect_eval
from aisteer360.evaluation.provider import ProviderOptions, as_inspect_model

pipeline.steer()
model = as_inspect_model(pipeline, options=ProviderOptions(max_batch_size=8))
logs = inspect_eval("inspect_evals/gsm8k", model=model, limit=100, temperature=0)
```

where the `ProviderOptions` dataclass carries the provider's configuration, i.e., static runtime kwargs
applied to every request (`runtime_kwargs`), `chat_template_kwargs` for the messages path, the batching
ceiling `max_batch_size`, the default `max_tokens` (`default_max_tokens`), the reasoning tags used to
split thinking from the answer before scoring (`reasoning_tags=None` disables the split), and the policy
for `GenerateConfig` parameters the pipeline cannot honor (`on_unsupported_param`, `"raise"` by default
or `"warn"`).

The provider decides its prompt path at construction. With a chat-templated tokenizer, prompts
dispatch as `messages=` and every input control participates as it does in deployment. Base models
without a chat template (a common subject of capability measurements) remain evaluable through a
text path, i.e., each conversation renders to plain text and dispatches as `text=`. On the text
path `adapt_messages` never runs (token-level `adapt` still does), the provider warns once at
construction, and the path is recorded as `prompt_path` in the run provenance since the same
controls behave differently on the two paths.

### Scope

The provider is generation-only. Requests carrying tools or tool messages, logprob parameters
(`logprobs`, `top_logprobs`, `prompt_logprobs`), or multimodal content raise with an error that
explains the restriction. This limits the stack to non-agentic tasks, which form the majority of
`inspect_evals`. Note that `GenerateConfig.response_schema` is not translated into a
`constrained_decoding` control because that would inject a control the configuration did not
declare. It follows the unsupported-parameter policy instead.

## Batching and reproducibility

Inspect issues one async request per sample and keeps many outstanding at once. The pipeline is
synchronous and runs one generation at a time, but accepts batched calls. The provider's collator
gathers concurrent requests into batched `pipeline.generate()` calls, filling the next batch while
the current generation runs. The provider advertises `max_connections` equal to its effective batch
ceiling, and Inspect's `max_samples` defaults to that value, which makes the default configuration
fill batches exactly. Note that `max_connections` should not be set below `max_batch_size`.

Batching applies only to arms whose enabled controls all declare `supports_batching=True`. The
provider clamps the ceiling to 1 otherwise. Input, state, and structural arms batch. Among output
controls only `phased_decoding`, `routed_decoding`, and `stopping_rules` declare batch safety
(`rad` and `value_guidance` compute it). Most driver-based arms therefore run one sample at a time.

We recommend greedy decoding (`temperature=0`) as the default since it is the norm for capability
benchmarks and avoids seed sensitivity. Which samples are evaluated is fixed by the suite,
independent of batch composition. A seeded dispatch carries `seed_scope` from `ProviderOptions`
(default `"dispatch"`), so a seeded batch decodes in one pass on the Hugging Face backend and the
dispatch is reproducible as a whole; under `"item"` scope each row derives its own seed and the
dispatch decodes one row at a time. Bitwise reproducibility of stochastic sampling is not preserved
under concurrency because a sample's dispatch membership and row index depend on async arrival
order. A bitwise-reproducible stochastic run requires `max_batch_size=1` and Inspect
`max_connections=1`. Even greedy outputs are not guaranteed to be bitwise-equal across batch
compositions since padded-batch numerics can differ from single-item numerics on some kernels.
Trial-to-trial variation under sampling is measured rather than eliminated, which is the role of
`num_trials` and the per-metric standard error.

## Suites and the runner

An `InspectSuite` names a set of tasks evaluated together. `SteeringEval` runs each configuration
(fixed controls, `ControlSpec` sweeps, and the empty baseline arm) over every trial and suite,
building and releasing one GPU-resident pipeline at a time:

```python
from aisteer360.evaluation.runner import SteeringEval
from aisteer360.evaluation.suite import InspectSuite

capability = InspectSuite(name="capability", tasks=("inspect_evals/gsm8k",), limit=200)
target = InspectSuite(name="target", tasks=("target_task.py",))

runner = SteeringEval(
    pipelines={"baseline": [], "pasta": [pasta]},
    base_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    suites=[capability, target],
    num_trials=3,
    seed=7,
    generate_defaults={"temperature": 0},
    save_dir="runs/exp1",
    display="plain",
)
results = runner.run()
frame = runner.results()
```

File-referenced tasks resolve relative to the working directory; the study notebooks keep their
task files in a shared `tasks/` folder and reference them by relative path.

Each suite run goes through `inspect_ai.eval_set`, which owns task retry and log-based resume. The
`.eval` logs under `save_dir/inspect_logs/` are the store, and a re-run completes only the missing
samples of each (configuration, trial, suite) cell. Since `eval_set` matches task identity only, a
changed protocol (seed, generate defaults, provider options, suites, fit, backend) needs a new
`save_dir` rather than a re-run into the old one. Repetition is trial-based rather than epoch-based. With `seed` set, each (configuration, trial)
pair derives one seed. The runner draws a `tqdm` bar over the (configuration, trial, suite) cells
(`progress=True` by default) and logs a summary line and one line per cell at INFO; `display="plain"`
streams Inspect's per-sample progress inside the cell in flight, which is the recommended setting in
a notebook. Note that `inspect_evals` tasks download their datasets from the Hugging Face Hub (some
are gated) and `.eval` logs can be large. Per-sample runtime kwargs are recorded with each model
event and should be kept small.

Every arm and every trial scores the identical sample set per task, either through explicit
`sample_ids` or through `limit=N` over the task's native dataset order. Taking the first `N`
samples is deterministic across arms, which paired comparison requires, but it is a biased
estimate of the full-benchmark score. This means that absolute scores are not directly comparable
to numbers published under other harnesses or logprob-scored protocols. The intended use is a
paired comparison against the baseline arm on identical samples, which is a single pivot on the
results frame:

```python
pivot = frame.pivot_table(index=["suite", "task", "metric"], columns="config", values="value")
deltas = pivot.sub(pivot["baseline"], axis=0)
```

The raw `.eval` logs carry per-sample generations, grades, and finish behavior, enough to trace a
drop in score to its cause (e.g., unparseable output rather than a wrong answer). Inspect's log
viewer and the `inspect_ai.analysis` dataframes (`evals_df`, `samples_df`, `events_df`) support
sample-level analysis.

Tasks with model-graded scorers need a grader model supplied through the task's own arguments
(`task_args`). The grader must be a separate model (an API model or a second local model) and
never the pipeline under evaluation, since self-grading is circular and grader traffic would
compete with evaluation traffic inside the collator. Also note that a local grader shares the GPU
with the pipeline. An API grader is preferable unless memory headroom is planned for both models.

## Authoring target-behavior tasks

Custom target-behavior evaluations are ordinary Inspect tasks. The toolkit ships no task, scorer,
or metric classes of its own. A working example lives in `examples/notebooks/studies/commonsense_mcqa/`,
which defines a shuffled-choice MCQA task with a custom positional-bias metric.

Controls that take per-generation parameters receive them through two tiers of runtime kwargs.
Static kwargs (`ProviderOptions.runtime_kwargs`) apply to every request. They suit catalog tasks,
whose datasets carry no steering columns, and any kwarg that is a property of the arm rather than
the sample. Per-sample kwargs are carried in `Sample.metadata` and delivered by the shipped
solver:

```python
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import includes
from aisteer360.evaluation.solvers import runtime_kwargs_solver

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

`runtime_kwargs_solver()` performs the sample's generation itself, taking the place of a bare
`generate()` in the solver chain. Each per-sample key must be declared with `"scope": "row"` in the
consuming control's `RUNTIME_KWARGS_SCHEMA` (a control declares `"row"` for a per-prompt value and
`"call"` for one value per generate call), and each value must be in the control's per-row form.
For PASTA that is one `list[str]` per sample. Across a batched dispatch the collator aligns the
per-sample values row by row. PASTA's `substrings` accepts a `str` broadcast to every row, a
`list[list[str]]` with one group per row, or a flat `list[str]` only at batch size 1; broadcasting
one group over a batch is done by passing `[[...]] * batch_size`. Tasks without the solver,
including the entire `inspect_evals` catalog, receive static kwargs only.

## Inspect scorers as rewards inside controls

Controls that optimize or rerank against a per-row score (PRewrite, CPO, GEPA, `best_of_n`,
`search_decoding`) consume a `SampleScorer`, a callable `(response, row) -> float` where the row
carries `"input"`, optionally `"reference"`, and any other dataset columns.
`sample_scorer_from_inspect` adapts any Inspect scorer into that form:

```python
from inspect_ai.scorer import model_graded_fact
from aisteer360.evaluation.scorers import sample_scorer_from_inspect

row_scorer = sample_scorer_from_inspect(model_graded_fact(model="openai/gpt-4o-mini"))
prewrite = PRewrite(initial_instruction="...", dev_set=dev_rows, row_scorer=row_scorer)
```

The adapter bridges Inspect's async scorers into synchronous control code. It works from plain
synchronous code, from inside the provider's dispatch thread, and from inside a running asyncio
event loop (a notebook), where it applies the same `nest_asyncio2` re-entry that Inspect uses.
Inside a running trio task it raises instead, since re-entry is impossible there. Note that a
model-graded scorer used this way runs grader traffic from inside a control's `steer()` or decode
loop. We recommend running optimizers with model-graded rewards from scripts.
