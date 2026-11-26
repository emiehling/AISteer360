# Optimize steering pipelines

The optimization package (`steerability.optimization`) searches over steering pipelines for the one with the largest
behavior-score gain while each capability benchmark's accuracy stays within a fixed margin of the unsteered model. The
chosen pipelines are then certified once on held-out items.

A decision in a study is never judged on the items it was made on. The position of a steering strength is chosen on
one block of items and judged on another. Candidates are compared on search items, and the finalists are certified on
held-out items. In this guide we define a study and evaluate candidates, then run a search and certify the result. The
guide uses the benchmarks, protocols, and recipes described in
[Evaluate steering pipelines](evaluate_steering_pipelines.md).

## Benchmarks and pools

A study measures behavior with one benchmark (the objective) and capability with one or more benchmarks (the
constraints). Each is a `Benchmark` from `steerability.evaluation.items` whose per-item scores are in [0, 1], and a
higher score is better. Before defining the study, we prepare each benchmark on the store and split its items into a
search pool and a held-out pool:

```python
from steerability.algorithms.core.store import Store
from steerability.evaluation.items import InspectBenchmark, PromptBenchmark
from steerability.optimization import split_items

store = Store.open(root="runs/formality/store")
gsm8k = InspectBenchmark(task="examples/notebooks/optimization/formality_gsm8k/scripts/gsm8k_all.py@gsm8k_all")
formality = PromptBenchmark(name="formality", items=prompts, scorer=formality_score)
for benchmark in (gsm8k, formality):
    benchmark.prepare(store=store)

pools = {
    gsm8k.id: split_items(benchmark=gsm8k, held_out=5000, seed=0),
    formality.id: split_items(benchmark=formality, held_out=500, seed=0),
}
```

The `split_items` function ranks the benchmark's groups by a hash of the seed and each group's key, and it assigns whole
groups to the held-out pool until it contains at least `held_out` items. The split depends only on the seed and the
items, which means that an edit of a task file or an upgrade of Inspect does not move it. Since groups are kept whole,
related samples (such as shuffled variants of one question) are never in both pools. Estimates during the search use the
search pool, and only its aggregate numbers reach a proposer. The held-out pool is read once, by certification. Fit-data
items (e.g., the contrastive pairs of a steering vector) can be listed per benchmark in the study's `fit_item_ids`, and
the study refuses pools that contain them.

Note that `inspect_evals/gsm8k` evaluates only the 1,319 test items and takes its few-shot examples from the train
split. The local task `examples/notebooks/optimization/formality_gsm8k/scripts/gsm8k_all.py` is a zero-shot task over
both splits (8,792 items), and its prompts contain no other item.

## The search space

The space of a study maps a structure name to a `Template`. We build it with the `structures` function from the
controls a pipeline may contain, given as one slot per position of the pipeline:

```python
from steerability.algorithms.core.recipe import Choice, Interval, Text, entry
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.algorithms.state_control.caa.control import CAA
from steerability.optimization import optional, structures

prompt = entry(
    SystemPrompt,
    name="prompt",
    text=Text(max_chars=600),
)
caa = entry(
    CAA,
    data=pairs,
    layer_id=Choice(values=(10, 12, 14)),
    multiplier=Interval(low=0.25, high=8.0, log=True),
)
space = structures(optional(prompt), optional(caa))
```

The `optional` function makes an entry optional, and `one_of` makes several entries exclusive. For instance,
`optional(one_of(caa, act_add))` gives a position with the `CAA` vector, the `ActAdd` vector, or no control. A bare
entry is part of every structure. The space contains every combination of the slots except the empty one, since the
unsteered model is the baseline of the study rather than a structure. In this example, the structures are `prompt`,
`caa`, and `prompt+caa`, and each is the `Template` of its entries. The name of a structure joins the names of its
entries in category order (structural, input, state, output), and an entry without an explicit `name=` is named after
its method. Two entries of one method therefore need explicit names. A space can also be written as a mapping from
structure name to `Template`.

Each template has at most one solved axis, i.e., an ordered axis (an `Interval`, an ordered `Choice`, or a shared
axis) whose key is not listed in the study's `searched` set. The evaluator solves this axis for each candidate by
finding the position with the best behavior among those inside the margin. Every other axis (unordered choices, text,
and the ordered axes in `searched`) is a proposer axis. In this example, `caa.multiplier` is the solved axis of the
`caa` and `prompt+caa` structures, and the layer and the prompt text are proposer axes. An axis has the same key in
every structure that contains its entry (`caa.layer_id` in `caa` and in `prompt+caa`). The study refuses a
combination with two free ordered axes. For such a combination, we list all but one of the axes in `searched` (which
applies to the whole study) or give them a shared `axis=` name. Controls that exclude each other, such as two decoding
drivers, belong in one `one_of` slot.

A candidate is a structure and one value per proposer axis:

```python
from steerability.optimization import Candidate

candidate = Candidate(
    structure="prompt+caa",
    values={"prompt.text": "Answer in a formal register.", "caa.layer_id": 12},
)
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
    model="Qwen/Qwen2.5-0.5B-Instruct",
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
    constraints=(Constraint(benchmark=gsm8k, margin=0.05),),
    pools=pools,
    monotone=frozenset({"caa.multiplier"}),
)
study.validate(store=store)
```

The `validate` method checks the study before any model work. It refuses a protocol with empty `versions` (one built
without `Protocol.create`) and any benchmark with an unset `temperature`, since the decoding mode decides which rules
apply to the baseline. It also refuses unversioned callables (see `versioned`) and overlapping pools (shared items or
groups). The remaining checks refuse templates with more than one free ordered axis, solved axes that change a fit,
repeated benchmark names, and trainings that leave no product to store (see [Weight changes](#weight-changes)).

The `Settings` dataclass contains the statistical settings and the behavior block sizes. The certification error rate
(`alpha`) and the design power (`power`) set the guard band. The ladder settings (`ladder_points`,
`behavior_ladder_items`, `ladder_refinements`, and `ladder_power` for a solved axis outside `monotone`) control how the
solved axis is calibrated, and the reporting settings (`behavior_items`, `fitness_alpha`, `min_gain`) control how a
point's gain is reported. Relative to the baseline, `max_cost_ratio` and `max_degenerate_increase` limit the generated
tokens and the share of degenerate completions. Note that a constraint's block sizes are not settings, since they follow
from its margin and its baseline (see [Sizes and feasibility](#sizes-and-feasibility)).

## Evaluating candidates

We evaluate candidates through a `StudySession`:

```python
from steerability.optimization.evaluation.session import StudySession

with StudySession(study=study, store=store, max_generations=200_000) as session:
    session.start()
    result = session.evaluate(candidate=candidate)
```

The `start` method validates the study and records it in the store. It then measures the unsteered baseline on every
search and held-out item. A second run on the search items uses a different batch order, and the mean absolute
difference between the two runs is the benchmark's noise floor (for 0/1 scores, the share of changed scores). The
held-out pool is generated as a separate request set (its items are batched only with each other). Every finalist's
held-out measurement later uses the same request set, which puts each held-out item in the same batch for the baseline
and the finalist. An item with a failed baseline measurement is excluded from its pool for the whole study, since its
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
capability benchmark. Stage 4 finds the operating point. For a solved axis in `monotone`, calibration gives each
position a first look on the start of a ladder block of items, from the weakest position up, and stops at the first
position whose loss is clearly over the margin. The behavior of the remaining positions is measured on the
objective's ladder block. Among the positions up to the one with the best lower bound of its gain, a bisection finds
the strongest position that passes the looks of a confirmation (checks after a growing number of items) on a fresh
block of items. The positions at or below it are then confirmed in the order of their gain bound, and the first one
confirmed is the operating point. This means that a ladder estimate alone never rejects a candidate. For any other
solved axis, calibration measures every position on the ladder block and keeps the positions whose loss is inside
the margin. The kept positions are confirmed in the order of the lower bound of their behavior gain. Without a
solved axis, the single recipe is confirmed. Stage 5 records the result.

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
error of at most `margin / sqrt(9.6)`. The ladder block is a quarter of the last look, and the calibration of a solved
axis in `monotone` reads only its first look (as many items as a confirmation's first look). For instance, at a 5-point
margin, GSM8K with a baseline accuracy of 0.457 (design variance 0.248) has a 1,000-item last look and a 250-item ladder
block.

The design variance is used only to size the looks and to decide feasibility. Acceptance and the held-out test use each
table's own variance (at least the floor). This means that an inaccurate design variance changes a study's power and
generation count and leaves the certification error rate unchanged.

The `start` method checks feasibility before the baseline (with the floor variance and the raw pool sizes) and again
after it (with the design variance and the usable pool sizes). Each constraint's search pool needs room for two rounds
of blocks. The constraint's guard band (how far inside the margin a search estimate must sit to pass the held-out test
with the design power) must also be at most 60% of the margin. The objective needs room for two rounds of behavior
blocks. For instance, with the model above and GSM8K pools of 3,792 search and 5,000 held-out items, a 3-point margin is
infeasible. Its 2,700-item last look needs 6,750 search items for two rounds, and its 2.16-point band is above the
1.80-point limit. The same pools are feasible at a 4-point margin.

An infeasible study raises `StudyInfeasibleError`, whose message gives the numbers and, for its pools, the smallest
feasible margin and the largest feasible design variance. With `allow_infeasible=True`, the session runs such a study at
low power, and the baseline checks still apply. Note that a study with another margin on the same store reuses the
baseline's evidence rows. This means that we can try another margin without generating anything.

## Running a search

A search runs in phases on one session. Each phase gives a search engine an allowance of the session, and each
engine reads what the earlier phases found from the archive. The `run_study` function evaluates the initial candidates
first and then runs the phases in order:

```python
from steerability.optimization.search.loop import SearchPhase, run_study
from steerability.optimization.search.openevolve import OpenEvolveEngine
from steerability.optimization.search.proposers import GridProposer

archive = run_study(
    study=study,
    phases=[
        SearchPhase(
            engine=GridProposer(study=study, structures=["caa"]),
        ),
        SearchPhase(
            engine=OpenEvolveEngine(
                run_dir="runs/formality/openevolve",
                llm={"models": [{"name": "gpt-4o-mini", "weight": 1.0}], "api_base": "https://api.openai.com/v1"},
                iterations=400,
                goal="The behavior score measures how formal the reply reads.",
                structures=["prompt", "prompt+caa"],
            ),
            generations=150_000,
        ),
    ],
    store=store,
    initial_candidates=[
        Candidate(structure="prompt", values={"prompt.text": "Respond in a formal register."}),
    ],
    max_generations=200_000,
)
best = archive.best()
```

The initial candidates are recorded with the proposer name `"initial"`, and a candidate that is already in the archive
costs nothing. A `SearchPhase` limits its engine by the number of recorded candidates (`candidates`), the generated
rows (`generations`), the wall time (`seconds`), and the structural fits (`structural_fits`, see
[Weight changes](#weight-changes)). These limits apply in addition to the limits of the run (`max_candidates`,
`max_generations`, `max_seconds`, and `max_structural_fits`). A phase ends when its allowance or the limits of the run
are spent, or when its engine has nothing new to propose. The session groups candidates into rounds of
`settings.round_candidates`. Each new round draws new item batches and re-scores the best accepted results on them,
which reduces the selection bias in the fitness values.

The terms of the toolkit correspond to those of other search tools as follows:

| Toolkit | Bayesian tools (Optuna, Ax) | OpenEvolve | HyperAgents |
| --- | --- | --- | --- |
| space | search space of distributions | program format and system message | files the meta agent may edit |
| candidate | trial | program | generation, stored as a patch |
| structure | root choice with child parameters | text in the program; islands keep kinds apart | whatever the agent writes |
| initial candidates, initial population | enqueued trials; trials added from earlier results | `initial_program`, `island_initial_programs`, checkpoint | initial node, `resume_from` |
| evaluation | objective value or trial state | evaluator returning metrics and artifacts | domain harness |
| stages 0 to 3 | pruning | cascade | staged evaluation |
| archive | trial storage | program database | `archive.jsonl` with parent selection |

### Proposers

The `GridProposer` enumerates every combination of discrete proposer axes, and the `RandomProposer` draws a structure
uniformly and then one value per proposer axis. Both take `structures=`, which restricts them to some of the study's
structures. A grid therefore covers the discrete structures of a study whose other structures contain text. The
`RandomProposer` cannot draw text and logs a warning for each `Text` axis, which keeps its empty default value. It
searches prompts when the axis is a `Choice` of strings. The run asks a proposer for `settings.round_candidates`
candidates at a time, prepares their fits together, and evaluates them in order. A custom proposer subclasses
`BaseProposer`, implements `ask(view, n)`, and overrides `tell(candidate, result)` when it learns from results.

### OpenEvolve

The `OpenEvolveEngine` (in `steerability.optimization.search.openevolve`, with the `optimize` extra) searches with
[OpenEvolve](https://github.com/codelion/openevolve), which evolves the candidate programs with a language model. Each
program is one JSON object between OpenEvolve's block markers. OpenEvolve runs in a subprocess, and its evaluator calls
an `EvaluationService` that serves the session over a local socket.

Each island of the run begins from one member of the engine's initial population (`initial_population`) and stays on
its structure. For each structure, the initial population contains the best accepted candidates in the archive, then
the initial candidates of the structure, then one candidate composed from the best values of other structures, and
then the structure's default candidate. This means that a later phase begins from what the earlier phases found. In
this example, the island of `prompt+caa` begins from the best layer of the grid and the best prompt known at that
point. The `islands_per_structure` argument gives a structure several islands, each beginning from another candidate.

The system message of an island states the goal and describes the island's own structure only. The goal is written by
the study's author, and it must not contain benchmark item text. An island admits only programs of its own structure
whose candidates the archive accepted with a positive fitness. Each candidate is admitted once per island, and programs
do not migrate between islands. The population of an island keeps one program per level bin (a point's level is its
largest loss as a share of the margin), since certification spreads its finalists over levels. Re-scored results
are consulted when a program is admitted and when two programs fall in the same bin. The replies of the evaluator also
list the recent rejections of the candidate's structure and the programs that could not be evaluated, and the model
sees them with the programs it is shown.

A structure without proposer axes has one candidate, which the engine evaluates once without an island. A run
directory with an OpenEvolve checkpoint resumes from it with its island assignment. The engine logs a warning for a
structure whose discrete settings have fewer combinations than its share of the iterations, since a `GridProposer` in
an earlier phase evaluates each of them once.

### Custom engines

A search engine subclasses `SearchEngine` and implements `run(session=..., initial_candidates=...)`. It evaluates
candidates through the session (`screen`, `finish`, or `evaluate`, or an `EvaluationService` that it starts) with its
`name` as the proposer name, and it returns when `session.exhausted()` is true. It sees only the `ArchiveView` of the
archive, which contains each candidate's id, values, status, stage, reason, fitness, and summary (no item-level data).

## Weight changes

A study can search weight changes, i.e., structural controls that load an adapter or a checkpoint (`LoadLoRA`,
`LoadCheckpoint`) or that train or merge (the TRL and MergeKit wrappers). For instance, we search a LoRA adapter
trained with `SFT` together with the `CAA` vector:

```python
from steerability.algorithms.structural_control.wrappers.trl.sfttrainer.control import SFT
from steerability.optimization.search.proposers import RandomProposer

lora = entry(
    SFT,
    name="lora",
    train_dataset=formal_text,
    use_peft=True,
    r=Choice(values=(8, 32)),
    num_train_epochs=1,
)
weights_study = Study(
    name="formality-weights",
    protocol=protocol,
    space=structures(optional(lora), optional(caa)),
    objective=formality,
    constraints=(Constraint(benchmark=gsm8k, margin=0.05),),
    pools=pools,
    monotone=frozenset({"caa.multiplier"}),
)
archive = run_study(
    study=weights_study,
    phases=[
        SearchPhase(
            engine=GridProposer(study=weights_study, structures=["lora"]),
        ),
        SearchPhase(
            engine=RandomProposer(study=weights_study, seed=0),
            structural_fits=0,
        ),
    ],
    store=store,
    max_structural_fits=2,
)
```

The product of a training or merge (an adapter or a checkpoint) is a fit that the store keeps. It is trained once per
distinct set of training arguments and installed from the store at every later steer, also in later sessions. The
structures `lora` and `lora+caa` contain the same training entry, and a candidate of each with the same `r` uses one
product. Every measurement of a candidate runs on the stored product, and a certified bundle contains the trained
weights.

An ordered training argument cannot be a solved axis, since every position of a solved axis must share its fits. We
list it in `study.searched` instead, and each proposed value is then trained once. A training is a structural fit.
The `max_structural_fits` argument limits the distinct structural fits of a run, and the `structural_fits` of a phase
limits those of the phase. A candidate that needs a training beyond the limits is returned with the reason `budget` and
is not recorded, while candidates on stored products are evaluated as usual. In this example, the grid trains the two
adapters, and the second phase searches on them without training (`structural_fits=0`).

Note that an unmerged LoRA adapter is small, while a full fine-tune, a merge, or a merged LoRA stores a whole
checkpoint per distinct training. We therefore recommend unmerged LoRA adapters. A merged LoRA run stores its product
only in `merged_output_dir`, and the study refuses one without it. The output locations of a training (`output_dir`,
`merged_output_dir`, and `resume_from_checkpoint`) are arguments of its entry, which means that they are part of the
identity of its product. We therefore leave them unset in a study (except `merged_output_dir` for a merged product).

Items of a study benchmark that are used as training data belong in the study's `fit_item_ids`, like the data of any
other fit, since the study cannot see inside a dataset. A loader entry (`LoadLoRA`, `LoadCheckpoint`) is identified by
its path. The study therefore assumes that an adapter or checkpoint directory given in a template does not change while
its store is in use. A certificate records the content of the weights at certification, and the bundle of a loader
refers to its path.

A training runs inside an evaluation stage, and a deadline does not interrupt it. For an `OpenEvolveEngine` phase over a
structure that trains, `stage_timeout_seconds` must therefore cover one training and the screening of the candidate.
Alternatively, the phase runs with `structural_fits=0` on the products that an earlier phase trained.

## Certification

The `certify` function tests the finalists on the held-out items and writes a bundle for each certified point:

```python
from steerability.optimization.release.certify import certify

with StudySession(study=study, store=store) as session:
    session.start()
    certification = certify(session=session, bundle_dir="runs/formality/bundles")
certification.certified, certification.failed_at
```

The finalists are spread over levels. Each finalist's level is measured again on previously unevaluated search items,
and these items are recorded with the finalists before any of them is measured. An interrupted call then measures the
same items. The finalists and their order (safest first) are fixed before any held-out item is read.

Each finalist is measured on every constraint's whole held-out pool, which is the request set of the baseline's held-out
measurement. It is then tested at the full error rate `alpha`, and testing stops at the first failing finalist. Every
earlier finalist is certified without a multiplicity correction, since the order was fixed first. A study is
certified once, and an interrupted call continues with the recorded order. A call on a study without finalists
records nothing and returns an empty certification, which leaves the study open for a later call. Held-out items that
another study already used are refused unless `allow_reused_held_out=True` is passed.

Each certified point is written as a frozen bundle whose lock contains the study's protocol and the certificate. The
certificate contains every constraint's held-out table with its tested lower bound, the behavior gain with a two-sided
90% interval, and every benchmark's noise floor. Its `weights` give the content id of the adapter or checkpoint of each
structural control:

```python
from steerability.spipe import SPipe

bundle = SPipe.load(path=certification.bundles[certification.certified[0]])
certificate = bundle.manifest["lock"]["certificate"]
pipeline = bundle.pipeline()
pipeline.steer()
```

Note that the guard band applies to a single confirmed point. The highest-fitness point usually has a loss close to
the margin, and it passes certification less often than `power`. For this reason the finalists are spread over levels
and tested from the safest one up.
