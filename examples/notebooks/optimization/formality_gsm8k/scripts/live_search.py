"""The live search of the optimization experiments: three search arms at one generation budget, then certification.

The study searches three structures (a system prompt, a CAA vector, and both) for formality on a prompt pool while
GSM8K accuracy stays within a 5-point margin. The arms are:

- `openevolve`: `run_openevolve` on the study, whose system prompts are free text (`Text(max_chars=600)`).
- `random`: `run_study` with a `RandomProposer` on a study whose system prompts are drawn from a fixed pool (a
  `Choice` of the pool's prompts in the `prompt` and `both` structures).
- `naive`: `run_openevolve` on a study whose GSM8K search pool has 100 items, without a guard band and under
  `allow_infeasible`. This study chooses and judges on the same 100 items.

The random arm's pool is written by the OpenEvolve arm's model without any score. Each call sends the OpenEvolve
arm's system message for the `prompt` structure and asks for one program. A reply becomes a prompt when it parses,
binds, is a `prompt` program, and has a non-empty text that differs from every earlier one after whitespace is
collapsed. The pool has `--prompt-pool-size` prompts (64 by default) in the order they were produced, and it is
stored in `<run_dir>/prompt_pool.json` with the messages, the model names, and the number of calls (never the `llm`
configuration). An existing pool file is reused, and a shorter one is completed. `--prompt-pool FILE` copies a pool
file into a run directory that has none, which lets the random arm run without `--llm`. The pool is built before the
baseline is measured, with at most four calls per missing prompt.

The baseline is measured once in `<run_dir>/baseline`, and each arm runs on its own copy of that store with its own
study name. The arms share no evidence row other than the baseline's. Each arm has a budget of `--max-generations`,
counted over every version of every result in its archive. An arm has spent its budget when what is left would not
pay for its most expensive candidate. A rerun searches an arm only with what is left of its budget, and an arm that
has spent it is not searched again. All searches finish before any arm is certified. The script certifies only when
every arm has spent its budget (or with `--certify-short`). Otherwise it writes the budget fields of every arm and
`comparison_valid: false` to the report and exits with an error.

The arms share the held-out items. Since each arm has its own store, its held-out ledger records its own study only. The
report gives, per arm, the budget fields, the certified gains, the number of finalists that passed, and the optimism
gap (the search-time gain minus the held-out gain), and for the random arm the size and SHA-256 of its prompt pool.

Run it from the repository root, since the GSM8K task is referenced by its relative path (a benchmark's name is its
task reference, and names appear in the summaries a proposer sees).

usage:
    python examples/notebooks/optimization/formality_gsm8k/scripts/live_search.py \
        <run_dir> --llm llm.json [--arms random,naive] [--prompt-pool FILE] [--certify-short]
"""
import argparse
import asyncio
import dataclasses
import hashlib
import json
import logging
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from formality import OBJECTIVE_HELD_OUT, PAIRS, formality_benchmark
from truth_grid import HELD_OUT, MODEL, gsm8k_benchmark, protocol

from steerability.algorithms.core.recipe import Choice, Interval, RecipeError, Template, Text, entry
from steerability.algorithms.core.store import Store
from steerability.algorithms.core.sweeps import PipelineFactory
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.algorithms.state_control.caa.control import CAA
from steerability.optimization import Constraint, PoolSplit, Settings, Study, split_items
from steerability.optimization.evaluation.archive import Archive
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.measurement.evidence import ensure_baseline
from steerability.optimization.release.certify import certify
from steerability.optimization.search.loop import run_study
from steerability.optimization.search.openevolve import parse, run_openevolve, system_message
from steerability.optimization.search.proposers import GridProposer, RandomProposer

logger = logging.getLogger("live_search")

MARGIN = 0.05
LAYERS = (8, 12, 16, 18, 20, 22)
NAIVE_SEARCH_ITEMS = 100
POOL_STRUCTURE = "prompt"
POOL_PATH = "system_prompt.text"
POOL_INSTRUCTION = (
    "No program of this structure has been scored yet. Propose one program for it. Reply with the complete program "
    "in one fenced block."
)
CALLS_PER_PROMPT = 4


def build_study(store: Store, *, arm: str, model: str, prompt_pool: Sequence[str] | None = None) -> Study:
    """The study of one arm.

    The naive arm has a 100-item GSM8K search pool and no guard band. The random arm draws its system prompts from
    `prompt_pool`, and the other arms write them as free text.

    Raises:
        ValueError: If the arm is `"random"` and `prompt_pool` is empty.
    """
    gsm8k = gsm8k_benchmark()
    formality = formality_benchmark()
    for benchmark in (gsm8k, formality):
        benchmark.prepare(store)
    capability = split_items(gsm8k, held_out=HELD_OUT, seed=0)
    if arm == "naive":
        capability = PoolSplit(capability.search[:NAIVE_SEARCH_ITEMS], capability.held_out)
    pools = {gsm8k.id: capability, formality.id: split_items(formality, held_out=OBJECTIVE_HELD_OUT, seed=0)}
    caa = entry(CAA, data=PAIRS, train_spec={"method": "mean_diff", "accumulate": "all"}, layer_id=Choice(LAYERS),
                multiplier=Interval(0.25, 8.0, log=True))
    if arm == "random":
        if not prompt_pool:
            raise ValueError("The random arm needs a prompt pool.")
        text = Choice(tuple(prompt_pool))
    else:
        text = Text(max_chars=600)
    space = {
        "prompt": Template((entry(SystemPrompt, text=text),)),
        "caa": Template((caa,)),
        "both": Template((entry(SystemPrompt, text=text), caa)),
    }
    return Study(
        name=f"live-search-{arm}", protocol=protocol(model), space=space, objective=formality,
        constraints=(Constraint(gsm8k, MARGIN),), pools=pools,
        settings=Settings(guard_band=False) if arm == "naive" else Settings(),
        monotone=frozenset({"caa.multiplier"}),
    )


def _pool_prompt(study: Study, reply: str) -> str | None:
    """The system prompt of a reply, or None when the reply is not a valid `prompt` program with a non-empty text."""
    from openevolve.utils.code_utils import parse_full_rewrite

    program = parse_full_rewrite(reply, "text")
    if program is None:
        return None
    try:
        candidate = parse(program)
        study.candidate_id(candidate)
    except (ValueError, RecipeError):
        return None
    if candidate.structure != POOL_STRUCTURE:
        return None
    text = candidate.values.get(POOL_PATH)
    return text if isinstance(text, str) and text.strip() else None


def prompt_pool(run_dir: Path, *, size: int, llm: Mapping[str, Any] | None, model: str,
                source: Path | None = None) -> list[str]:
    """The random arm's prompt pool, read from `<run_dir>/prompt_pool.json` and completed by the model when short.

    Args:
        run_dir: The run directory.
        size: The number of prompts.
        llm: OpenEvolve's `llm` configuration, or None.
        model: The model of the study.
        source: A pool file to copy into the run directory when it has none, or None.

    Returns:
        The first `size` prompts of the pool.

    Raises:
        SystemExit: If the pool is short and `llm` is None, or if the model gives too few valid prompts within four
            calls per missing prompt. The pool file is then left as it was.
    """
    path = run_dir / "prompt_pool.json"
    if source is not None and not path.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    prompts = list(record.get("prompts", []))
    if len(prompts) >= size:
        return prompts[:size]
    if llm is None:
        raise SystemExit(f"The random arm needs a pool of {size} prompts, and {path} has {len(prompts)}. Pass --llm to "
                         "write the missing prompts, or --prompt-pool FILE to copy a pool into a new run directory.")
    from openevolve.config import Config
    from openevolve.llm.ensemble import LLMEnsemble

    store = Store.open(run_dir / "baseline")
    try:
        study = build_study(store, arm="openevolve", model=model)
    finally:
        store.close()
    system = system_message(study, POOL_STRUCTURE)
    config = Config.from_dict({"llm": json.loads(json.dumps(dict(llm)))})
    ensemble = LLMEnsemble(config.llm.models)
    previous_calls = int(record.get("calls", 0))
    found = len(prompts)
    limit = CALLS_PER_PROMPT * (size - found)
    seen = {" ".join(prompt.split()) for prompt in prompts}

    async def generate() -> int:
        calls = 0
        while len(prompts) < size and calls < limit:
            ensemble.reseed(previous_calls + calls)
            reply = await ensemble.generate_with_context(
                system_message=system, messages=[{"role": "user", "content": POOL_INSTRUCTION}],
            )
            calls += 1
            text = _pool_prompt(study, reply)
            if text is None or " ".join(text.split()) in seen:
                continue
            seen.add(" ".join(text.split()))
            prompts.append(text)
        return calls

    calls = asyncio.run(generate())
    if len(prompts) < size:
        raise SystemExit(f"{calls} calls to the model gave {len(prompts) - found} new valid prompts, and the pool "
                         f"needs {size - found}; {path} is left as it was. A model that does not sample (a "
                         "temperature of 0) usually causes this.")
    record = {
        "system_message": system, "user_message": POOL_INSTRUCTION,
        "models": [model_config.name for model_config in config.llm.models],
        "calls": previous_calls + calls, "prompts": prompts,
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("prompt pool of %d prompts written to %s after %d calls", len(prompts), path, calls)
    return prompts


def budget_fields(study: Study, store: Store, max_generations: int, *, random_arm: bool) -> dict:
    """The budget of one arm, read from its archive.

    `generations` sums the generations of every version of every result, `shortfall` is what is left of
    `max_generations`, and `largest_candidate` is the largest number of generations one candidate used. An arm has
    `budget_spent` when its shortfall would not pay for its most expensive candidate. The random arm also gets
    `candidates_left`, the number of grid candidates of its space that are not in the archive.
    """
    archive = Archive(study, store)
    results = archive.results()
    spent = [sum(version.generations for version in archive.versions(result.candidate_id)) for result in results]
    generations = sum(spent)
    fields: dict[str, Any] = {
        "candidates": len(results), "generations": generations, "shortfall": max_generations - generations,
        "largest_candidate": max(spent, default=0),
    }
    fields["budget_spent"] = fields["shortfall"] <= fields["largest_candidate"]
    if random_arm:
        fields["candidates_left"] = len(GridProposer(study).ask(archive.view(), 10**9))
    return fields


def measure_baseline(run_dir: Path, model: str) -> Path:
    """Measure the baseline once in `<run_dir>/baseline` (on the full pools) and return the store's directory."""
    root = run_dir / "baseline"
    done = root / "baseline.done"
    if not done.exists():
        store = Store.open(root)
        study = build_study(store, arm="openevolve", model=model)
        factory = PipelineFactory(study.protocol, store=store)
        try:
            ensure_baseline(study, factory, store)
        finally:
            factory.release()
            store.close()
        done.write_text("", encoding="utf-8")
    return root


def arm_store(run_dir: Path, arm: str, baseline: Path) -> Store:
    """The store of one arm, a copy of the baseline store made on first use."""
    root = run_dir / arm / "store"
    if not root.exists():
        shutil.copytree(baseline, root)
    return Store.open(root)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--llm", help="A JSON file with OpenEvolve's llm configuration.")
    parser.add_argument("--arms", default="random,openevolve,naive")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-generations", type=int, default=400_000)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--prompt-pool", help="A prompt pool file to copy into a run directory that has none.")
    parser.add_argument("--prompt-pool-size", type=int, default=64)
    parser.add_argument("--certify-short", action="store_true",
                        help="Certify even when an arm has not spent its budget.")
    arguments = parser.parse_args()
    run_dir = Path(arguments.run_dir)
    llm = json.loads(Path(arguments.llm).read_text()) if arguments.llm else None
    arms = [arm.strip() for arm in arguments.arms.split(",") if arm.strip()]
    if llm is None and any(arm != "random" for arm in arms):
        raise SystemExit("The openevolve and naive arms need --llm.")
    pool = None
    if "random" in arms:
        pool = prompt_pool(run_dir, size=arguments.prompt_pool_size, llm=llm, model=arguments.model,
                           source=Path(arguments.prompt_pool) if arguments.prompt_pool else None)
    baseline = measure_baseline(run_dir, arguments.model)

    for arm in arms:
        store = arm_store(run_dir, arm, baseline)
        study = build_study(store, arm=arm, model=arguments.model, prompt_pool=pool)
        budget = budget_fields(study, store, arguments.max_generations, random_arm=arm == "random")
        if budget["budget_spent"]:
            logger.info("arm %s has spent its budget (%d generations) and is not searched again", arm,
                        budget["generations"])
            store.close()
            continue
        factory = PipelineFactory(study.protocol, store=store)
        try:
            if arm == "random":
                run_study(study, RandomProposer(study, seed=0), store=store, factory=factory,
                          max_generations=budget["shortfall"])
            else:
                run_openevolve(study, store=store, run_dir=run_dir / arm, llm=llm, iterations=arguments.iterations,
                               max_generations=budget["shortfall"], factory=factory, allow_infeasible=arm == "naive")
        finally:
            factory.release()
            store.close()

    budgets = {}
    for arm in arms:
        store = arm_store(run_dir, arm, baseline)
        try:
            study = build_study(store, arm=arm, model=arguments.model, prompt_pool=pool)
            budgets[arm] = budget_fields(study, store, arguments.max_generations, random_arm=arm == "random")
        finally:
            store.close()
    short = [arm for arm in arms if not budgets[arm]["budget_spent"]]
    report: dict = {"comparison_valid": not short}
    pool_fields = None
    if pool is not None:
        pool_fields = {"size": len(pool), "sha256": hashlib.sha256(json.dumps(pool).encode("utf-8")).hexdigest()}
    if short and not arguments.certify_short:
        for arm in arms:
            report[arm] = dict(budgets[arm], **({"prompt_pool": pool_fields} if arm == "random" else {}))
        (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise SystemExit(f"The arms {', '.join(short)} have not spent their budgets, and nothing was certified; the "
                         f"budget fields are in {run_dir / 'report.json'}. Run the same command again, or pass "
                         "--certify-short to certify anyway.")

    for arm in arms:
        store = arm_store(run_dir, arm, baseline)
        study = build_study(store, arm=arm, model=arguments.model, prompt_pool=pool)
        factory = PipelineFactory(study.protocol, store=store)
        try:
            with StudySession(study, store=store, factory=factory, allow_infeasible=arm == "naive") as session:
                session.start()
                certification = certify(session, bundle_dir=run_dir / arm / "bundles")
                searched = {result.operating.resolved_id: result.operating.gain for result in session.archive.points()}
        finally:
            factory.release()
            store.close()
        gains = {resolved: certificate["behavior"]["gain"]
                 for resolved, certificate in certification.certificates.items()}
        report[arm] = {
            **budgets[arm],
            "certified": len(certification.certified), "failed_at": certification.failed_at,
            "certified_gains": gains,
            "optimism_gaps": {resolved: searched[resolved] - gain for resolved, gain in gains.items()
                              if resolved in searched},
            "best_certified_gain": max(gains.values(), default=None),
            "settings": dataclasses.asdict(study.settings),
        }
        if arm == "random":
            report[arm]["prompt_pool"] = pool_fields
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("report written to %s", run_dir / "report.json")


if __name__ == "__main__":
    main()
