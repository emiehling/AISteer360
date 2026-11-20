"""The live search of the optimization experiments: three search arms at one generation budget, then certification.

The study searches three structures (a system prompt, a CAA vector, and both) for formality on a prompt pool while
GSM8K accuracy stays within a 5-point margin. The arms are:

- `openevolve`: `run_openevolve` on the study.
- `random`: `run_study` with a `RandomProposer`.
- `naive`: `run_openevolve` on a study whose GSM8K search pool has 100 items, without a guard band and under
  `allow_infeasible`. This study chooses and judges on the same 100 items.

The baseline is measured once in `<run_dir>/baseline`, and each arm runs on its own copy of that store with its own
study name. The arms share no evidence row other than the baseline's. Every arm runs with the same
`--max-generations`, and all searches finish before any arm is certified. The arms share the sealed items. Since each
arm has its own store, its sealed ledger records its own study only. The report gives, per arm, the certified gains,
the number of finalists that passed, and the optimism gap (the search-time gain minus the sealed gain).

Run it from the repository root, since the GSM8K task is referenced by its relative path (a benchmark's name is its
task reference, and names appear in the summaries a proposer sees).

usage:
    python examples/notebooks/optimization/scripts/live_search.py <run_dir> --llm llm.json [--arms random,naive]
"""
import argparse
import dataclasses
import json
import logging
import shutil
from pathlib import Path

from formality import OBJECTIVE_SEALED, PAIRS, formality_benchmark
from truth_grid import MODEL, SEALED, gsm8k_benchmark, protocol

from steerability.algorithms.core.recipe import Choice, Interval, Template, Text, entry
from steerability.algorithms.core.store import Store
from steerability.algorithms.core.sweeps import PipelineFactory
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.algorithms.state_control.caa.control import CAA
from steerability.optimization import Constraint, PoolSplit, Settings, Study, split_items
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.measurement.evidence import ensure_baseline
from steerability.optimization.release.certify import certify
from steerability.optimization.search.enumerate import RandomProposer
from steerability.optimization.search.openevolve import run_openevolve
from steerability.optimization.search.proposer import run_study

logger = logging.getLogger("live_search")

MARGIN = 0.05
LAYERS = (8, 12, 16, 18, 20, 22)
NAIVE_SEARCH_ITEMS = 100


def build_study(store: Store, *, arm: str, model: str) -> Study:
    """The study of one arm; the naive arm has a 100-item GSM8K search pool and no guard band."""
    gsm8k = gsm8k_benchmark()
    formality = formality_benchmark()
    for benchmark in (gsm8k, formality):
        benchmark.prepare(store)
    capability = split_items(gsm8k, sealed=SEALED, seed=0)
    if arm == "naive":
        capability = PoolSplit(capability.search[:NAIVE_SEARCH_ITEMS], capability.sealed)
    pools = {gsm8k.id: capability, formality.id: split_items(formality, sealed=OBJECTIVE_SEALED, seed=0)}
    caa = entry(CAA, data=PAIRS, train_spec={"method": "mean_diff", "accumulate": "all"}, layer_id=Choice(LAYERS),
                multiplier=Interval(0.25, 8.0, log=True))
    space = {
        "prompt": Template((entry(SystemPrompt, text=Text(max_chars=600)),)),
        "caa": Template((caa,)),
        "both": Template((entry(SystemPrompt, text=Text(max_chars=600)), caa)),
    }
    return Study(
        name=f"live-search-{arm}", protocol=protocol(model), space=space, objective=formality,
        constraints=(Constraint(gsm8k, MARGIN),), pools=pools,
        settings=Settings(guard_band=False) if arm == "naive" else Settings(),
        monotone=frozenset({"caa.multiplier"}),
    )


def measure_baseline(run_dir: Path, model: str) -> Path:
    """Measure the baseline once in `<run_dir>/baseline` (on the full pools) and return the store's directory."""
    root = run_dir / "baseline"
    done = root / "baseline.done"
    if not done.exists():
        store = Store.open(root)
        study = build_study(store, arm="random", model=model)
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
    arguments = parser.parse_args()
    run_dir = Path(arguments.run_dir)
    llm = json.loads(Path(arguments.llm).read_text()) if arguments.llm else None
    arms = [arm.strip() for arm in arguments.arms.split(",") if arm.strip()]
    if llm is None and any(arm != "random" for arm in arms):
        raise SystemExit("The openevolve and naive arms need --llm.")
    baseline = measure_baseline(run_dir, arguments.model)

    for arm in arms:
        store = arm_store(run_dir, arm, baseline)
        study = build_study(store, arm=arm, model=arguments.model)
        factory = PipelineFactory(study.protocol, store=store)
        infeasible = arm == "naive"
        try:
            if arm == "random":
                run_study(study, RandomProposer(study, seed=0), store=store, factory=factory,
                          max_generations=arguments.max_generations)
            else:
                run_openevolve(study, store=store, run_dir=run_dir / arm, llm=llm, iterations=arguments.iterations,
                               max_generations=arguments.max_generations, factory=factory,
                               allow_infeasible=infeasible)
        finally:
            factory.release()
            store.close()

    report: dict = {}
    for arm in arms:
        store = arm_store(run_dir, arm, baseline)
        study = build_study(store, arm=arm, model=arguments.model)
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
            "certified": len(certification.certified), "failed_at": certification.failed_at,
            "certified_gains": gains,
            "optimism_gaps": {resolved: searched[resolved] - gain for resolved, gain in gains.items()
                              if resolved in searched},
            "best_certified_gain": max(gains.values(), default=None),
            "settings": dataclasses.asdict(study.settings),
        }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("report written to %s", run_dir / "report.json")


if __name__ == "__main__":
    main()
