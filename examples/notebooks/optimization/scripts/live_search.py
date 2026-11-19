"""The live search of the optimization experiments: three search arms at one generation budget, then certification.

The study searches three structures (a system prompt, a CAA vector, and both) for formality on a prompt pool while
GSM8K accuracy stays within a 3-point margin. The arms are:

- `openevolve`: `run_openevolve` on the study.
- `random`: `run_study` with a `RandomProposer`.
- `naive`: `run_openevolve` on a study whose GSM8K search pool has 100 items, without a guard band and under
  `allow_infeasible`, so it chooses and judges on the same 100 items.

Every arm runs with the same `--max-generations`, and all searches finish before any arm is certified. The arms share
one sealed pool, so each is certified with `allow_reused_sealed=True`. The report gives, per arm, the certified gains,
the number of finalists that passed, and the optimism gap (the search-time gain minus the sealed gain).

Run it from the repository root, since the GSM8K task is referenced by its relative path (a benchmark's name is its
task reference, and names appear in the summaries a proposer sees).

usage:
    python examples/notebooks/optimization/scripts/live_search.py <run_dir> --llm llm.json [--arms random,naive]
"""
import argparse
import dataclasses
import itertools
import json
import logging
from pathlib import Path

from steerability.algorithms.core.identity import versioned
from steerability.algorithms.core.protocol import Protocol
from steerability.algorithms.core.recipe import Choice, Interval, Template, Text, entry
from steerability.algorithms.core.store import Store
from steerability.algorithms.core.sweeps import PipelineFactory
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.algorithms.state_control.caa.control import CAA
from steerability.evaluation.items import InspectBenchmark, PromptBenchmark
from steerability.evaluation.provider import ProviderOptions
from steerability.optimization import Constraint, PoolSplit, Settings, Study, split_items
from steerability.optimization.evaluation.session import StudySession
from steerability.optimization.release.certify import certify
from steerability.optimization.search.enumerate import RandomProposer
from steerability.optimization.search.openevolve import run_openevolve
from steerability.optimization.search.proposer import run_study

logger = logging.getLogger("live_search")

TASK = "examples/notebooks/optimization/gsm8k_all.py@gsm8k_all"  # relative to the repository root
CLASSIFIER = "s-nlp/roberta-base-formality-ranker"
LAYERS = (8, 12, 16, 20)
_PIPELINE = None

WHO = ("your best friend", "your roommate", "your cousin", "your teammate", "your neighbor", "your little brother",
       "an old classmate", "your gym buddy", "your sister", "a coworker you hang out with")
WHAT = ("you will be late to dinner", "you found their lost keys", "the game got moved to Sunday",
        "you can't make the party", "the movie was amazing", "you need to borrow their charger",
        "the pizza is on its way", "you finally passed your driving test", "the concert tickets sold out",
        "you are running ten minutes behind", "their package arrived at your place", "the trip is still on",
        "you adopted a puppy", "you broke your phone screen", "the wifi password changed",
        "you saw their favorite band at the mall", "the barbecue starts at six", "you lost the bet",
        "you are bringing snacks", "the bus is cancelled", "you got the job", "the beach day is rained out",
        "you started a new hobby", "the team won the final", "you want to grab coffee", "the rent is due Friday",
        "you finished the book they lent you", "the store was closed", "you are cooking tonight",
        "the meeting was moved", "you need a ride tomorrow", "the dog ate your homework", "you booked the cabin",
        "the bike needs new tires", "you are running a marathon", "the cake turned out great",
        "the printer is broken again", "you saw a famous chef", "you are moving next month", "the exam got easier",
        "the lights went out", "you are stuck in traffic", "the parade starts at noon", "you lost your wallet",
        "the garden is blooming", "you are hosting game night", "the train is delayed", "you made new friends",
        "the soup is ready", "you won a raffle")
PROMPTS = {f"p{index:04d}": f"Send a quick text to {who} letting them know {what}."
           for index, (who, what) in enumerate(itertools.product(WHO, WHAT))}

FORMAL = [
    "I regret to inform you that I will be delayed this evening.",
    "Please be advised that the meeting has been rescheduled to Thursday.",
    "I would be grateful if you could return the charger at your convenience.",
    "I am pleased to confirm that the arrangements remain unchanged.",
    "Kindly note that the event will commence at six o'clock.",
    "I wish to express my sincere appreciation for your assistance.",
    "It is with regret that I must decline your kind invitation.",
    "I would like to inform you that your parcel has been delivered.",
]
CASUAL = [
    "hey gonna be late tonight lol, sorry!!",
    "yo the meeting got moved to thursday btw",
    "can u bring back my charger whenever? thx!",
    "yep we're still on, super pumped!!",
    "party starts at 6, don't be late haha",
    "dude thanks so much, you're awesome!",
    "ugh can't make it, sorry guys",
    "hey ur package showed up at my place!",
]
TASKS = [
    "Tell a friend you will be late tonight.", "Tell a friend the meeting moved to Thursday.",
    "Ask a friend to return your charger.", "Confirm the plans are still on.", "Say when the party starts.",
    "Thank a friend for their help.", "Decline an invitation.", "Tell a friend their package arrived.",
]
PAIRS = {"prompts": TASKS, "positives": FORMAL, "negatives": CASUAL}


@versioned("1")
def formality_score(response, row):
    """The formality score of a response from a RoBERTa formality ranker, in [0, 1]."""
    global _PIPELINE
    if _PIPELINE is None:
        from transformers import pipeline

        _PIPELINE = pipeline("text-classification", model=CLASSIFIER, top_k=None, truncation=True)
    if not response.strip():
        return 0.0
    scores = {result["label"].lower(): result["score"] for result in _PIPELINE(response[:2000])[0]}
    return float(scores["formal"])


def build_study(store: Store, *, model: str, naive: bool = False) -> Study:
    """The study of one arm; the naive arm has a 100-item GSM8K search pool and no guard band."""
    gsm8k = InspectBenchmark(TASK, generate={"max_tokens": 384})
    formality = PromptBenchmark("formality", PROMPTS, formality_score, generate={"max_tokens": 96})
    for benchmark in (gsm8k, formality):
        benchmark.prepare(store)
    capability = split_items(gsm8k, sealed=5000, seed=0)
    if naive:
        capability = PoolSplit(capability.search[:100], capability.sealed)
    pools = {gsm8k.id: capability, formality.id: split_items(formality, sealed=100, seed=0)}
    protocol = Protocol.create(model, provider=ProviderOptions(max_batch_size=64),
                               generate_defaults={"temperature": 0}, seed=0)
    caa = entry(CAA, data=PAIRS, train_spec={"method": "mean_diff", "accumulate": "all"}, layer_id=Choice(LAYERS),
                multiplier=Interval(0.0, 8.0))
    space = {
        "prompt": Template((entry(SystemPrompt, text=Text(max_chars=600)),)),
        "caa": Template((caa,)),
        "both": Template((entry(SystemPrompt, text=Text(max_chars=600)), caa)),
    }
    settings = Settings(guard_band=False) if naive else Settings()
    return Study(
        name="live-search-naive" if naive else "live-search", protocol=protocol, space=space, objective=formality,
        constraints=(Constraint(gsm8k, 0.03),), pools=pools, settings=settings,
        monotone=frozenset({"caa.multiplier"}),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--llm", help="A JSON file with OpenEvolve's llm configuration.")
    parser.add_argument("--arms", default="random,openevolve,naive")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-generations", type=int, default=400_000)
    parser.add_argument("--iterations", type=int, default=400)
    arguments = parser.parse_args()
    run_dir = Path(arguments.run_dir)
    store = Store.open(run_dir / "store")
    llm = json.loads(Path(arguments.llm).read_text()) if arguments.llm else None
    arms = [arm.strip() for arm in arguments.arms.split(",") if arm.strip()]
    studies = {arm: build_study(store, model=arguments.model, naive=arm == "naive") for arm in arms}
    factory = PipelineFactory(studies[arms[0]].protocol, store=store)
    report: dict = {}
    try:
        for arm, study in studies.items():
            infeasible = arm == "naive"
            if arm == "random":
                run_study(study, RandomProposer(study, seed=0), store=store, factory=factory,
                          max_generations=arguments.max_generations, allow_infeasible=infeasible)
            else:
                if llm is None:
                    raise SystemExit(f"Arm {arm!r} needs --llm.")
                run_openevolve(study, store=store, run_dir=run_dir / arm, llm=llm, iterations=arguments.iterations,
                               max_generations=arguments.max_generations, factory=factory,
                               allow_infeasible=infeasible)
        for arm, study in studies.items():
            with StudySession(study, store=store, factory=factory, allow_infeasible=arm == "naive") as session:
                session.start()
                certification = certify(session, bundle_dir=run_dir / arm / "bundles", allow_reused_sealed=True)
                searched = {result.operating.resolved_id: result.operating.gain for result in session.archive.points()}
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
    finally:
        factory.release()
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("report written to %s", run_dir / "report.json")


if __name__ == "__main__":
    main()
