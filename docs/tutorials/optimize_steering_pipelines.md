# Optimize steering pipelines

The optimization package (`steerability.optimization`) searches over steering pipelines for the one with the largest
behavior-score gain while each capability benchmark's accuracy stays within a fixed margin of the unsteered model. The
chosen pipelines are then certified once on held-out items.

A decision in a study is never judged on the items it was made on. The position of a steering strength is chosen on
one block of items and judged on another. Candidates are compared on search items, and the finalists are certified on
sealed items. In this guide we define a study and evaluate candidates, then run a search and certify the result. The
guide uses the benchmarks, protocols, and recipes described in
[Evaluate steering pipelines](evaluate_steering_pipelines.md).

## Benchmarks and pools

A study measures behavior with one benchmark (the objective) and capability with one or more benchmarks (the
constraints). Each is a `Benchmark` from `steerability.evaluation.items` whose per-item scores are in [0, 1], and a
higher score is better. Before defining the study, we prepare each benchmark on the store and split its items into a
search pool and a sealed pool:

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
one question) are never in both pools. Estimates during the search use the search pool, and only its aggregate
numbers reach a proposer. The sealed pool is read once, by certification. Fit-data items (e.g., the contrastive pairs
of a steering vector) can be listed per benchmark in the study's `fit_item_ids`, and the study refuses pools that
contain them.

Note that `inspect_evals/gsm8k` evaluates only the 1,319 test items and takes its few-shot examples from the train
split. The local task `examples/notebooks/optimization/gsm8k_all.py` is a zero-shot task over both splits (8,792
items), and its prompts contain no other item.

## The search space

The space of a study maps a structure name to a `Template`:

```python
from steerability.algorithms.core.recipe import Choice, Interval, Template, Text, entry
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.algorithms.state_control.caa.control import CAA

layers = Choice((10, 12, 14))
space = {
    "prompt": Template((entry(SystemPrompt, text=Text(max_chars=600)),)),
    "caa": Template((entry(CAA, data=pairs, layer_id=layers, multiplier=Interval(0.25, 8.0, log=True)),)),
    "both": Template((
        entry(SystemPrompt, text=Text(max_chars=600)),
        entry(CAA, data=pairs, layer_id=layers, multiplier=Interval(0.25, 8.0, log=True)),
    )),
}
```

Each template has at most one solved axis, i.e., an ordered axis (an `Interval`, an ordered `Choice`, or a shared
axis) whose key is not listed in the study's `searched` set. The evaluator solves this axis for each candidate by
finding the position with the best behavior among those inside the margin. Every other axis (unordered choices, text,
and the ordered axes in `searched`) is a proposer axis. In this example, `caa.multiplier` is the solved axis of the
`caa` and `both` structures, and the layer and the prompt text are proposer axes.

A candidate is a structure and one value per proposer axis:

```python
from steerability.optimization import Candidate

candidate = Candidate("both", {"prompt.text": "Answer in a formal register.", "caa.layer_id": 12})
```

The `Study.bind` method binds every proposer axis of a candidate's template and leaves the solved axis free. It raises
`RecipeError` for an out-of-range value, a missing proposer axis, or an unknown key. Two candidates that bind to the
same template have the same id.

A solved axis must not change a fit, which means that every position of the axis has the same fit digests. The
multiplier of a fitted vector meets this requirement, since the fit stores the direction and the multiplier scales it
at generation time. The size of the training data changes the fit and cannot be a solved axis. An ordered fit input
belongs in `searched`, where each proposed value is fitted once and then cached. The study's
`monotone` set lists the solved axes along which capability loss is assumed not to fall as the position rises. The
set makes no assumption about behavior, which often rises and then falls at strong settings.

## The study

The `Study` dataclass combines the protocol, the space, the benchmarks, and the pools:

```python
from steerability.algorithms.core.protocol import Protocol, ProviderOptions
from steerability.optimization import Constraint, Study

protocol = Protocol.create(
    "Qwen/Qwen2.5-0.5B-Instruct",
    hf_model_kwargs={"dtype": "float32"},
    provider=ProviderOptions(max_batch_size=64),
    generate_defaults={"temperature": 0, "max_tokens": 512},
    seed=0,
)
study = Study(
    name="formality-gsm8k",
    protocol=protocol,
    space=space,
    objective=formality,
    constraints=(Constraint(gsm8k, margin=0.05),),
    pools=pools,
    monotone=frozenset({"caa.multiplier"}),
)
study.validate(store)
```

The `validate` method checks the study before any model work. It refuses a protocol with empty `versions` (one built
without `Protocol.create`) and any benchmark with an unset `temperature`, since the decoding mode decides which rules
apply to the baseline. It also refuses structural entries, unversioned callables (see `versioned`), and overlapping
pools (shared items or groups). The remaining checks refuse templates with more than one free ordered axis, solved
axes that change a fit, and repeated benchmark names.

The `Settings` dataclass contains the statistical settings and the behavior block sizes. The certification error rate
(`alpha`) and the design power (`power`) set the guard band. The ladder settings (`ladder_points`,
`behavior_ladder_items`, `ladder_power`, `ladder_refinements`) control how the solved axis is calibrated, and the
reporting settings (`behavior_items`, `fitness_alpha`, `min_gain`) control how a point's gain is reported. Relative to
the baseline, `max_cost_ratio` and `max_degenerate_increase` limit the generated tokens and the share of degenerate
completions. Note that a constraint's block sizes are not settings, since they follow from its margin and its baseline
(see [Sizes and feasibility](#sizes-and-feasibility)).

## Evaluating candidates

We evaluate candidates through a `StudySession`:

```python
from steerability.optimization.evaluation.session import StudySession

with StudySession(study, store=store, max_generations=200_000) as session:
    session.start()
    result = session.evaluate(candidate)
```

The `start` method validates the study and records it in the store. It then measures the unsteered baseline on every
search and sealed item. A second run on the search items uses a different batch order, and the mean absolute
difference between the two runs is the benchmark's noise floor (for 0/1 scores, the share of changed scores). The
sealed pool is generated as a separate request set (its items are batched only with each other). Every finalist's
sealed measurement later uses the same request set, which puts each sealed item in the same batch for the baseline and
the finalist. An item with a failed baseline measurement is excluded from its pool for the whole study, since its
baseline score of 0 would make every candidate look better on it.

Note that a stored evidence row is keyed by recipe and item. The row describes the recipe only for batch-independent
generation. Under greedy decoding, a noise floor above 0.01 (`GREEDY_NOISE_LIMIT`) therefore stops the study with a
`StudyError`, and no argument waives this check. The first 400 search items are measured twice before the rest of the
baseline. This means that a batch-dependent protocol is found after 800 generations. For instance, with
Qwen2.5-0.5B-Instruct and batches of 64, two bf16 runs changed the scores of about 12% of GSM8K items, and two float32
runs (`hf_model_kwargs={"dtype": "float32"}`) changed 0.03%. A float32 protocol or `ProviderOptions(max_batch_size=1)`
avoids the problem. On the vLLM backend, the engine's dtype option (`engine_kwargs={"dtype": "float32"}`) has the same
effect. Additionally note that a constraint whose baseline mean is at most its margin stops the study, since no
candidate can lose more than the margin.

Each evaluation runs in stages, and an early stage rejects a candidate cheaply. Stage 0 binds the candidate and checks
that the protocol's backend supports it, and stage 1 runs its fits. Stage 2 is a quick behavior check of a few
solved-axis positions on a few items, and stage 3 checks whether every probed position is clearly over the margin on a
capability benchmark. Stage 4 finds the operating point. With a solved axis, calibration measures every
position of the axis on a ladder block of items and keeps the positions whose loss is inside the margin. The kept
positions are ranked by the lower bound of their behavior gain. The top-ranked position is then confirmed on a fresh
block of items in up to three looks (checks after a growing number of items), and the next-ranked position is tried
when it fails. Without a solved axis, the single recipe is confirmed. Stage 5 records the result.

A candidate is accepted only when it has an eligible operating point, i.e., one that was confirmed on every
constraint and is within the cost and degeneracy limits. Its fitness is the one-sided lower bound of its gain at
`fitness_alpha`, and a rejected candidate has a fitness of 0. The `Result` contains the candidate's status, the stage
and reason of a rejection, and the operating point with its tables of paired differences against the baseline. It
also contains one `LadderPoint` per measured position and a `summary` of search-pool aggregates (gain, gain lower
bound, level, cost ratio, degenerate increase, and the loss on each constraint). Note that a point's tables are built
only from fresh-block items and exclude the ladder block its position was chosen on.

Results are memoized by candidate id, and a second evaluation of a candidate returns the recorded result without
model work. A candidate stopped by a limit (`max_candidates`, `max_generations`, `max_seconds`) or by a deadline is
returned with the reason `budget` or `timeout` and is not recorded. Its evidence rows stay in the store, and a later
session resumes from them.

## Sizes and feasibility

A study sizes each constraint from its baseline. The design variance is the variance of the baseline's search-pool
scores (`p * (1 - p)` for 0/1 scores with accuracy `p`), and it is at least the floor `margin * (1 - margin)`. At the
design variance, a confirmation's last look is the smallest multiple of 100 items that gives the estimate a standard
error of at most `margin / sqrt(9.6)`. The ladder block is a quarter of the last look. For instance, at a 5-point
margin, GSM8K with a baseline accuracy of 0.457 (design variance 0.248) has a 1,000-item last look and a 250-item
ladder block.

The design variance is used only to size the looks and to decide feasibility. Acceptance and the sealed test use each
table's own variance (at least the floor). This means that an inaccurate design variance changes a study's power and
generation count and leaves the certification error rate unchanged.

The `start` method checks feasibility before the baseline (with the floor variance and the raw pool sizes) and again
after it (with the design variance and the usable pool sizes). Each constraint's search pool needs room for two rounds
of blocks. The constraint's guard band (how far inside the margin a search estimate must sit to pass the sealed test
with the design power) must also be at most 60% of the margin. The objective needs room for two rounds of behavior
blocks. For instance, with the model above and GSM8K pools of 3,792 search and 5,000 sealed items, a 3-point margin is
infeasible. Its 2,700-item last look needs 6,750 search items for two rounds, and its 2.16-point band is above the
1.80-point limit. The same pools are feasible at a 4-point margin.

An infeasible study raises `StudyInfeasibleError`, whose message gives the numbers and, for its pools, the smallest
feasible margin and the largest feasible design variance. With `allow_infeasible=True`, the session runs such a study at
low power, and the baseline checks still apply. Note that a study with another margin on the same store reuses the
baseline's evidence rows. This means that we can try another margin without generating anything.

## Running a search

A search is a loop of proposals and evaluations. The session groups candidates into rounds of
`settings.round_candidates`. Each new round draws new item batches and re-scores the best accepted results on them.
Re-scoring reduces the selection bias in the fitness values.

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
candidate's id, values, status, stage, reason, fitness, and summary (no item-level data).

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
over a local socket. The run's population strategy admits and replaces programs based on the archive's latest results
instead of OpenEvolve's stored metrics. Re-scored results and timed-out evaluations are therefore reflected in the
population.

## Certification

The `certify` function tests the finalists on the sealed items and writes a bundle for each certified point:

```python
from steerability.optimization.release.certify import certify

with StudySession(study, store=store) as session:
    session.start()
    certification = certify(session, bundle_dir="runs/formality/bundles")
certification.certified, certification.failed_at
```

The finalists are spread over levels (a point's level is its largest loss as a share of the margin). Each finalist's
level is measured again on previously unevaluated search items, and these items are recorded with the finalists before
any of them is measured. An interrupted call then measures the same items. The finalists and their order (safest
first) are fixed before any sealed item is read.

Each finalist is measured on every constraint's whole sealed pool, which is the request set of the baseline's sealed
measurement. It is then tested at the full error rate `alpha`, and testing stops at the first failing finalist. Every
earlier finalist is certified without a multiplicity correction, since the order was fixed first. A study is
certified once, and an interrupted call continues with the recorded order. Sealed items that another study already
used are refused unless `allow_reused_sealed=True` is passed.

Each certified point is written as a frozen bundle whose lock contains the study's protocol and the certificate. The
certificate contains every constraint's sealed table with its tested lower bound, the behavior gain with a two-sided
90% interval, and every benchmark's noise floor:

```python
from steerability.spipe import SPipe

bundle = SPipe.load(certification.bundles[certification.certified[0]])
certificate = bundle.manifest["lock"]["certificate"]
pipeline = bundle.pipeline()
pipeline.steer()
```

Note that the guard band applies to a single confirmed point. The highest-fitness point usually has a loss close to
the margin, and it passes certification less often than `power`. For this reason the finalists are spread over levels
and tested from the safest one up.
