# Optimize steering pipelines

The optimization package (`steerability.optimization`) searches over steering pipelines for the one that gives the
largest gain in a behavior score while the accuracy on each capability benchmark stays within a fixed margin of the
unsteered model. The chosen pipelines are then certified once, on items that the search never read.

The package follows one rule throughout, i.e., a decision is never judged on the items it was made on. The position
of a steering strength is chosen on one block of items and judged on another, candidates are compared on search
items, and the finalists are certified on sealed items. This guide describes how to define a study, evaluate
candidates, run a search, and certify the result. It uses the benchmarks, protocols, and recipes described in
[Evaluate steering pipelines](evaluate_steering_pipelines.md).

## Benchmarks and pools

A study measures behavior with one benchmark (the objective) and capability with one or more benchmarks (the
constraints). Each is a `Benchmark` from `steerability.evaluation.items` whose per-item scores are in [0, 1], with
higher scores being better. Before defining the study, we prepare each benchmark on the store and split its items into
a search pool and a sealed pool:

```python
from steerability.algorithms.core.store import Store
from steerability.evaluation.items import InspectBenchmark, PromptBenchmark
from steerability.optimization import split_items

store = Store.open("runs/formality/store")
gsm8k = InspectBenchmark("examples/notebooks/optimization/gsm8k_all.py@gsm8k_all")
formality = PromptBenchmark("formality", prompts, formality_score)
for benchmark in (gsm8k, formality):
    benchmark.prepare(store)

pools = {
    gsm8k.id: split_items(gsm8k, sealed=5000, seed=0),
    formality.id: split_items(formality, sealed=500, seed=0),
}
```

The `split_items` function shuffles the benchmark's groups with a seed and assigns whole groups to the sealed pool
until it contains at least `sealed` items. Since groups are kept whole, related samples (such as shuffled variants of
one question) never end up in both pools. Estimates inside the search use the search pool, and only aggregate numbers
computed from it reach a proposer. The sealed pool is read once, by certification. Items used as fit data (e.g., the
contrastive pairs of a steering vector) can be listed per benchmark in the study's `fit_item_ids`, and the study
refuses pools that contain them.

Note that `inspect_evals/gsm8k` evaluates the 1,319 test items only and takes its few-shot examples from the train
split. The local task `examples/notebooks/optimization/gsm8k_all.py` covers both splits zero-shot, which gives 8,792 items and no
overlap between prompts and items.

## The search space

The space of a study maps a structure name to a `Template`:

```python
from steerability.algorithms.core.recipe import Choice, Interval, Template, Text, entry
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.algorithms.state_control.caa.control import CAA

layers = Choice((10, 12, 14))
space = {
    "prompt": Template((entry(SystemPrompt, text=Text(max_chars=600)),)),
    "caa": Template((entry(CAA, data=pairs, layer_id=layers, multiplier=Interval(0.0, 8.0)),)),
    "both": Template((
        entry(SystemPrompt, text=Text(max_chars=600)),
        entry(CAA, data=pairs, layer_id=layers, multiplier=Interval(0.0, 8.0)),
    )),
}
```

Each template has at most one solved axis, i.e., an ordered axis (an `Interval`, an ordered `Choice`, or a shared
axis) whose key is not listed in the study's `searched` set. The evaluator solves this axis for each candidate by
finding the position with the best behavior among those inside the margin. Every other axis (unordered choices, text,
and the ordered axes in `searched`) is a proposer axis. In the space above, `caa.multiplier` is the solved axis of the
`caa` and `both` structures, while the layer and the prompt text are proposer axes.

A candidate is a structure and one value per proposer axis:

```python
from steerability.optimization import Candidate

candidate = Candidate("both", {"prompt.text": "Answer in a formal register.", "caa.layer_id": 12})
```

The `Study.bind` method turns a candidate into its template, with only the solved axis left free. It raises
`RecipeError` when a value is outside its range, when a proposer axis is missing, or when a key gives anything other
than a proposer axis. Two candidates that bind to the same template have the same id.

A solved axis must not change a fit, which means that every position of the axis has the same fit digests. The
multiplier of a fitted vector satisfies this, since the fit stores the direction and the multiplier only scales it at
generation time. The size of the training data does not. An ordered argument that is an input of a fit belongs in
`searched`, where each proposed value is fitted once and then cached. The study's `monotone` set lists the solved
axes along which capability loss is assumed not to fall as the position rises. It says nothing about behavior, which
often rises and then collapses at strong settings.

## The study

The `Study` dataclass combines the protocol, the space, the benchmarks, and the pools:

```python
from steerability.algorithms.core.protocol import Protocol
from steerability.evaluation.provider import ProviderOptions
from steerability.optimization import Constraint, Study

protocol = Protocol.create(
    "Qwen/Qwen2.5-1.5B-Instruct",
    provider=ProviderOptions(max_batch_size=32),
    generate_defaults={"temperature": 0, "max_tokens": 512},
    seed=0,
)
study = Study(
    name="formality-gsm8k",
    protocol=protocol,
    space=space,
    objective=formality,
    constraints=(Constraint(gsm8k, margin=0.03),),
    pools=pools,
    monotone=frozenset({"caa.multiplier"}),
)
study.validate(store)
```

The `validate` method checks the study before any model work. It refuses structural entries, unversioned callables
(see `versioned`), pools that share items or groups, templates with more than one free ordered axis, solved axes that
change a fit, and benchmarks with repeated names. It then checks feasibility. Each constraint needs room in its search
pool for the blocks of a confirmation, and the guard band (how far inside the margin a search estimate must sit to
pass the sealed test with the design power) must be at most 60% of the margin. For instance, GSM8K split into 3,792
search and 5,000 sealed items is feasible at a margin of 3 points and infeasible at 2 points. An infeasible study
raises `StudyInfeasibleError` with the numbers unless `allow_infeasible=True` is passed.

Note that the check is repeated after the baseline is measured, with twice the noise floor as the change rate when that
is larger than `design_change_rate`. Greedy decoding in bf16 changes completions with the composition of a batch. For
instance, with Qwen2.5-0.5B-Instruct and batches of 64, the score of about 12% of GSM8K items changed between two batch
orders, while the same protocol with float32 weights (`hf_model_kwargs={"dtype": "float32"}`) changed none. A float32
protocol (or `ProviderOptions(max_batch_size=1)`) therefore keeps the guard bands close to their design values.

The `Settings` dataclass contains the statistical settings and the sizes of the item blocks. The error rate of
certification (`alpha`) and the design power (`power`) set the guard band. The ladder settings (`ladder_points`,
`ladder_items`, `behavior_ladder_items`, `ladder_power`, `ladder_refinements`) control how the solved axis is
calibrated. The reporting settings (`behavior_items`, `fitness_alpha`, `min_gain`) control how a point's gain is
reported, and `max_cost_ratio` and `max_degenerate_increase` limit the generated tokens and the share of degenerate
completions relative to the baseline.

## Evaluating candidates

A `StudySession` is the one entry to evaluation:

```python
from steerability.optimization.evaluation.session import StudySession

with StudySession(study, store=store, max_generations=200_000) as session:
    session.start()
    result = session.evaluate(candidate)
```

The `start` method validates the study, records it in the store, and measures the unsteered baseline on every search
and sealed item. The search items are measured a second time in another batch order, and the share of items whose
score differs between the two measurements is the benchmark's noise floor. An item whose baseline measurement failed
is excluded from its pool for the whole study, since its baseline score of 0 would make every candidate look better
on it.

Each evaluation runs in stages, and an early stage rejects a candidate cheaply. Stage 0 binds the candidate and checks
that the protocol's backend supports it, and stage 1 runs its fits. Stage 2 is a quick look at the behavior of a few
positions of the solved axis on a few items, and stage 3 checks whether every probed position is clearly over the
margin on a capability benchmark. Stage 4 finds the operating point. With a solved axis, calibration measures every
position of the axis on a ladder block of items, keeps the positions whose loss is inside the margin, and ranks them
by the lower bound of their behavior gain. The top-ranked position is then confirmed on a fresh block of items in up
to three looks, and the next-ranked position is tried when it fails. Without a solved axis, the single recipe is
confirmed. Stage 5 records the result.

A candidate is accepted only when it has an eligible operating point, i.e., one that was confirmed on every
constraint and is within the cost and degeneracy limits. Its fitness is the one-sided lower bound of its gain at
`fitness_alpha`, and a rejected candidate has a fitness of 0. The `Result` contains the status, the stage and reason of
a rejection, the operating point with its tables of paired differences against the baseline, one `LadderPoint` per
measured position, and a `summary` of search-pool aggregates (gain, gain lower bound, level, cost ratio, degenerate
increase, and the loss on each constraint). Note that a point's tables are built from fresh-block items only, never
from the ladder block its position was chosen on.

Results are memoized by candidate id, and a second evaluation of a candidate returns the recorded result without
model work. A candidate stopped by a limit (`max_candidates`, `max_generations`, `max_seconds`) or by a deadline is
returned with the reason `budget` or `timeout` and is not recorded. Its evidence rows stay in the store, and a later
session resumes from them.

## Running a search

A search is a loop of proposals and evaluations. The session groups candidates into rounds of
`settings.round_candidates`, and each new round draws new item batches and re-scores the best accepted results on
them, which reduces the selection bias of the fitness values.

The `run_study` function drives an in-process proposer:

```python
from steerability.optimization.search.enumerate import RandomProposer
from steerability.optimization.search.proposer import run_study

archive = run_study(study, RandomProposer(study, seed=0), store=store, max_generations=200_000)
best = archive.best()
```

The `GridProposer` enumerates every combination of discrete proposer axes, and the `RandomProposer` draws a structure
uniformly and then one value per proposer axis (a text axis keeps the empty seed value). A custom proposer implements
`ask(view, n)` and `tell(candidate, result)`. It sees only the `ArchiveView` of the archive, which contains each
candidate's id, values, status, stage, reason, fitness, and summary, and nothing item-level.

The `run_openevolve` function (in `steerability.optimization.search.openevolve`, with the `optimize` extra) searches
with [OpenEvolve](https://github.com/codelion/openevolve), which evolves the candidate programs with a language model:

```python
from steerability.optimization.search.openevolve import run_openevolve

archive = run_openevolve(
    study,
    store=store,
    run_dir="runs/formality/openevolve",
    llm={"models": [{"name": "gpt-4o-mini", "weight": 1.0}], "api_base": "https://api.openai.com/v1"},
    iterations=400,
    max_generations=200_000,
)
```

Each program is one JSON object between OpenEvolve's block markers, and every island of the run searches one
structure. OpenEvolve runs in a subprocess, and its evaluator calls an `EvaluationService` that serves the session
over a local socket. The run's population strategy admits and replaces programs by the archive's latest results
rather than by OpenEvolve's stored metrics. Re-scored results and timed-out evaluations are therefore reflected in
the population.

## Certification

The `certify` function tests the finalists on the sealed items and writes a bundle for each certified point:

```python
from steerability.optimization.release.certify import certify

with StudySession(study, store=store) as session:
    session.start()
    certification = certify(session, bundle_dir="runs/formality/bundles")
certification.certified, certification.failed_at
```

The finalists are spread over levels (a point's level is its largest loss as a share of the margin), and each one's
level is measured again on search items it was not evaluated on. The finalists and their order (safest first) are
recorded before any sealed item is read. Each finalist is tested on every constraint at the full error rate `alpha`,
and testing stops at the first finalist that fails. Every finalist before it is certified, with no multiplicity
correction since the order was fixed first. A study is certified once. An interrupted call continues with the
recorded order, and the sealed items of a study that another study already used are refused unless
`allow_reused_sealed=True` is passed.

Each certified point is written as a frozen bundle whose lock contains the study's protocol and the certificate (the
sealed tables of every constraint and the behavior gain with a two-sided 90% interval):

```python
from steerability.spipe import SPipe

bundle = SPipe.load(certification.bundles[certification.certified[0]])
certificate = bundle.manifest["lock"]["certificate"]
pipeline = bundle.pipeline()
pipeline.steer()
```

Note that the guard band describes one confirmed point taken alone. Since the point with the highest fitness is
selected for sitting near the edge of the margin, it passes certification less often than `power`. This is the reason
the finalists are spread over levels and tested from the safest up.
