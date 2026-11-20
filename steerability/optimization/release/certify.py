"""Certification of a study's finalists on the sealed items, and the release of certified points as bundles.

The finalists and their order are fixed and recorded before any sealed item is read, and each finalist is tested on
every constraint at the full error rate `settings.alpha`, from the safest point up. Each sealed call evaluates the
whole sealed pool in pool order, which is the request set of the baseline's sealed call. Testing stops at the first
finalist that fails, and every finalist before it is certified. No multiplicity correction applies, since the order
was fixed first. A study is certified once, and an interrupted call continues with the recorded order. Sealed
results are returned to the caller only and never reach a proposer.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import steerability
from steerability.algorithms.core.recipe import RecipeError
from steerability.optimization.evaluation.evaluator import recipe_at
from steerability.optimization.evaluation.plan import make_plan
from steerability.optimization.evaluation.result import jsonable
from steerability.optimization.evaluation.session import Finalist
from steerability.optimization.measurement.evidence import noise_floor, paired, usable
from steerability.optimization.measurement.pools import record_sealed_use, sealed_id, shared_sealed_items
from steerability.optimization.measurement.stats import floor_variance
from steerability.optimization.study import StudyError, dumps, ensure_tables, now
from steerability.spipe.format import CERTIFICATE_SCHEMA

if TYPE_CHECKING:
    from steerability.algorithms.core.recipe import Recipe
    from steerability.algorithms.core.sweeps import Steered
    from steerability.optimization.evaluation.session import StudySession
    from steerability.optimization.measurement.stats import Paired

BEHAVIOR_INTERVAL_ALPHA = 0.05


@dataclass(frozen=True)
class Certification:
    """The outcome of certifying a study.

    Attributes:
        certified: The resolved ids of the certified finalists, in test order.
        failed_at: The index of the first finalist that failed, or None.
        certificates: Mapping from resolved id to its certificate.
        bundles: Mapping from resolved id to the path of its bundle.
    """

    certified: tuple[str, ...]
    failed_at: int | None
    certificates: Mapping[str, dict]
    bundles: Mapping[str, Path]


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _ordered_records(finalists: list[Finalist], plan: list[dict]) -> list[dict]:
    fresh_items = {(record["candidate_id"], int(record["version"])): record["fresh_items"] for record in plan}
    return [
        {
            "candidate_id": finalist.result.candidate_id, "version": finalist.result.version,
            "fresh_items": fresh_items[(finalist.result.candidate_id, finalist.result.version)],
            "level": finalist.level, "fresh": finalist.fresh,
        }
        for finalist in finalists
    ]


def _fixed_finalists(session: StudySession) -> list[Finalist]:
    """The recorded finalists of the study in their order, planned, measured, and recorded now when needed.

    Without a `certifications` row, the plan (`session.plan_finalists()`) is written first. Records without levels are
    then measured (`session.finalists(plan)`, which reuses stored rows) and the row is updated with the ordered
    records. A row whose records have levels is used as it is.
    """
    study, store = session.study, session.store
    ensure_tables(store)
    row = store.execute("SELECT finalists, outcome FROM certifications WHERE study_id = ?", (study.id,)).fetchone()
    if row is not None and row[1] is not None:
        raise StudyError(f"Study {study.name!r} ({study.id}) is certified already; a study is certified once.")
    if row is None:
        store.execute(
            "INSERT OR IGNORE INTO certifications (study_id, finalists, outcome, created_at) VALUES (?, ?, NULL, ?)",
            (study.id, dumps(session.plan_finalists()), now()),
        )
        row = store.execute("SELECT finalists FROM certifications WHERE study_id = ?", (study.id,)).fetchone()
    records = json.loads(row[0])
    if any(record["level"] is None for record in records):
        records = _ordered_records(session.finalists(records), records)
        store.execute("UPDATE certifications SET finalists = ? WHERE study_id = ? AND outcome IS NULL",
                      (dumps(records), study.id))
    finalists = []
    for record in records:
        versions = {result.version: result for result in session.archive.versions(record["candidate_id"])}
        result = versions.get(record["version"])
        if result is None or result.operating is None:
            raise StudyError(f"Finalist {record['candidate_id']} version {record['version']} is not in the archive.")
        finalists.append(Finalist(result, float(record["level"]), bool(record["fresh"])))
    return finalists


def _recipe(session: StudySession, finalist: Finalist) -> Recipe:
    result = finalist.result
    candidate = session.archive.candidate(result.candidate_id)
    if candidate is None:
        raise StudyError(f"Finalist {result.candidate_id} has no recorded candidate.")
    try:
        recipe = recipe_at(make_plan(session.study, candidate), result.operating.position)
    except RecipeError as error:
        raise StudyError(f"Finalist {result.candidate_id} no longer binds ({error}).") from error
    if recipe.id != result.operating.recipe_id:
        raise StudyError(f"Finalist {result.candidate_id} rebuilds to recipe {recipe.id}, not "
                         f"{result.operating.recipe_id}.")
    return recipe


def _sealed_table(session: StudySession, steered: Steered, benchmark: Any) -> Paired:
    """Evaluate the whole sealed pool in pool order and build the table on its usable items."""
    from steerability.evaluation.items import evaluate_items

    study, store = session.study, session.store
    record_sealed_use(store, study, benchmark, steered.resolved_id)
    evaluate_items(steered, benchmark, list(study.pools[benchmark.id].sealed), store=store, replicate=0)
    return paired(study, store, steered.resolved_id, benchmark, usable(study, store, benchmark, "sealed"))


def certify(session: StudySession, *, bundle_dir: str | Path, allow_reused_sealed: bool = False) -> Certification:
    """Certify a study's finalists on the sealed items and write a bundle for each certified point.

    The procedure is:

    1. The finalists are fixed before any sealed item is read. A recorded outcome raises. Without a `certifications`
       row, the finalist plan (`session.plan_finalists()`, with the search items each level is measured on) is
       written first. Records without levels are then measured (`session.finalists(plan)`, which reuses stored rows)
       and recorded in their order. Recorded records with levels (an interrupted call) are used unchanged.
    2. The sealed pools are compared with those of every other study that has a recorded sealed use of a benchmark of
       the same name, and a shared item raises unless `allow_reused_sealed`.
    3. For each finalist in order, a `sealed_uses` row is written for each benchmark before its sealed items are read.
       The whole sealed pool of every constraint is evaluated in pool order, its table is built on the usable sealed
       items, and each constraint's table must pass (`Paired.passes(margin, settings.alpha)`). Testing stops at the
       first finalist that fails. Stored evidence rows are reused, and a resumed call reads nothing twice.
    4. Each certified finalist is measured on the objective's sealed pool in the same way, and its gain is recorded
       with a two-sided 90% interval (per finalist, not simultaneous).
    5. Each certified finalist is written to `bundle_dir` as a frozen bundle whose lock records the study's protocol
       and the certificate (`"certificate/1"`). The certificate's `lower_bound` of a constraint is the bound that
       `Paired.passes` tests, and every benchmark's entry contains its baseline's `noise_floor`.
    6. The outcome is recorded in `certifications`.

    Args:
        session: The started session of the study.
        bundle_dir: The directory the bundles are written to.
        allow_reused_sealed: Whether to certify on sealed items that another study already used.

    Returns:
        The certification.

    Raises:
        StudyError: If the study is certified already, its sealed items were used by another study (and
            `allow_reused_sealed` is False), or a finalist cannot be rebuilt.
    """
    study, store, settings = session.study, session.store, session.study.settings
    finalists = _fixed_finalists(session)
    shared = shared_sealed_items(store, study)
    if shared and not allow_reused_sealed:
        details = "; ".join(f"{name}: studies {', '.join(others)}" for name, others in shared.items())
        raise StudyError(f"The sealed items of study {study.name!r} were already used ({details}); pass "
                         "allow_reused_sealed=True to certify on them anyway.")
    session.release_pipeline()
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    order = [{"resolved_id": finalist.result.operating.resolved_id, "level": finalist.level} for finalist in finalists]
    certified: list[str] = []
    certificates: dict[str, dict] = {}
    bundles: dict[str, Path] = {}
    failed_at: int | None = None
    for index, finalist in enumerate(finalists):
        recipe = _recipe(session, finalist)
        with session.factory.steered(recipe) as steered:
            constraints: dict[str, dict] = {}
            passed = True
            for constraint in study.constraints:
                benchmark = constraint.benchmark
                table = _sealed_table(session, steered, benchmark)
                constraints[benchmark.name] = {
                    "benchmark_id": benchmark.id, "margin": constraint.margin,
                    "sealed_id": sealed_id(benchmark, study.pools[benchmark.id]), "n": table.n,
                    "delta": table.mean,
                    "lower_bound": _finite(table.lower(settings.alpha, floor=floor_variance(constraint.margin))),
                    "change_rate": table.change_rate, "noise_floor": noise_floor(study, store, benchmark),
                }
                if not table.passes(constraint.margin, settings.alpha):
                    passed = False
                    break
            if not passed:
                failed_at = index
                break
            behavior = _sealed_table(session, steered, study.objective)
            candidate = session.archive.candidate(finalist.result.candidate_id)
            certificate = {
                "schema": CERTIFICATE_SCHEMA,
                "study_id": study.id, "study_name": study.name, "protocol_id": study.protocol.id,
                "recipe_id": recipe.id, "resolved_id": steered.resolved_id, "structure": candidate.structure,
                "values": jsonable(dict(candidate.values)),
                "position": jsonable(dict(finalist.result.operating.position)),
                "alpha": settings.alpha, "power": settings.power,
                "constraints": constraints,
                "behavior": {
                    "benchmark_id": study.objective.id, "n": behavior.n, "gain": behavior.mean,
                    "lower": _finite(behavior.lower(BEHAVIOR_INTERVAL_ALPHA)),
                    "upper": _finite(behavior.upper(BEHAVIOR_INTERVAL_ALPHA)),
                    "noise_floor": noise_floor(study, store, study.objective),
                },
                "finalists": order,
                "index": index,
                "fresh_rescore": finalist.fresh,
                "toolkit_version": steerability.__version__,
                "created_at": now(),
            }
            path = bundle_dir / f"{steered.resolved_id}.spipe"
            bundle = steered.pipeline.to_spipe(freeze=True, protocol=study.protocol, certificate=certificate)
            written = bundle.save(path)
            certified.append(steered.resolved_id)
            certificates[steered.resolved_id] = certificate
            bundles[steered.resolved_id] = Path(written) if written is not None else path
    outcome = {
        "certified": certified, "failed_at": failed_at, "certificates": certificates,
        "bundles": {key: str(value) for key, value in bundles.items()},
    }
    store.execute(
        "UPDATE certifications SET outcome = ? WHERE study_id = ? AND outcome IS NULL", (dumps(outcome), study.id),
    )
    return Certification(tuple(certified), failed_at, certificates, bundles)


__all__ = ["Certification", "certify"]
