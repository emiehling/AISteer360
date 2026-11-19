"""The session of a running study, the one public entry to evaluation.

`run_study`, the evaluation service, and certification are front-ends to a `StudySession`. The session keeps the
round counter, the limits, and the re-scoring. Every front-end therefore advances rounds in the same way.
"""
from __future__ import annotations

import logging
import random
import time
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING

from steerability.algorithms.core.identity import derive_seed
from steerability.algorithms.core.recipe import Recipe, RecipeError
from steerability.algorithms.core.sweeps import PipelineFactory
from steerability.algorithms.input_control.common.budget import RolloutBudget
from steerability.optimization.evaluation import evaluator
from steerability.optimization.evaluation.archive import Archive
from steerability.optimization.evaluation.evaluator import Partial, SteeringFailure, Stopped
from steerability.optimization.evaluation.plan import make_plan
from steerability.optimization.evaluation.result import Result, level_of
from steerability.optimization.measurement import evidence
from steerability.optimization.measurement.evidence import Batch
from steerability.optimization.measurement.stats import confirm_looks
from steerability.optimization.study import Candidate, Study, StudyInfeasibleError, record_study

if TYPE_CHECKING:
    from steerability.algorithms.core.store import Store
    from steerability.algorithms.core.sweeps import Steered
    from steerability.evaluation.items import Benchmark

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Finalist:
    """A point taken to certification.

    Attributes:
        result: The point's latest result.
        level: The point's level, used to order the finalists.
        fresh: Whether the level came from items the point had not been evaluated on.
    """

    result: Result
    level: float
    fresh: bool


class StudySession:
    """The owner of a running study: its archive, rounds, limits, and the evaluation of candidates.

    Candidates are evaluated in two parts, `screen` (stages 0 to 3, cheap rejections) and `finish` (stages 4 and 5,
    the operating point and the report), or with `evaluate`. A candidate with a recorded result is returned without
    model work. After every `settings.round_candidates` recorded candidates, the session starts a new round with new
    batches and re-scores the best accepted results on them (`rescore`).

    Limits belong to the session object, and a restarted process gets a new allowance. Generations are counted after
    each measurement with the rows that were generated, and fits are not counted (their own rollouts are bounded by
    `max_seconds` only). `max_candidates` counts the candidates recorded by this session, and `max_seconds` runs from
    `start`. Once a limit is reached, `exhausted()` is true and `screen` returns a rejected result with reason
    `budget`. A candidate stopped by a limit or a deadline is not recorded.

    Args:
        study: The study.
        store: The store.
        factory: The factory that steers recipes, or None to build one for the study's protocol (released by
            `close`).
        max_candidates: The number of candidates this session may record, or None.
        max_generations: The number of generations this session may run, or None.
        max_seconds: The wall time this session may run from `start`, or None.
        allow_infeasible: Whether to accept a study that fails the feasibility check.

    Attributes:
        study: The study.
        store: The store.
        factory: The pipeline factory.
        archive: The study's archive.
        round_index: The current round.
    """

    def __init__(
        self,
        study: Study,
        *,
        store: Store,
        factory: PipelineFactory | None = None,
        max_candidates: int | None = None,
        max_generations: int | None = None,
        max_seconds: float | None = None,
        allow_infeasible: bool = False,
    ) -> None:
        self.study = study
        self.store = store
        self._owns_factory = factory is None
        self.factory = factory if factory is not None else PipelineFactory(study.protocol, store=store)
        self.archive = Archive(study, store)
        self.round_index = 0
        self.max_candidates = max_candidates
        self.max_seconds = max_seconds
        self.allow_infeasible = allow_infeasible
        self.budget = RolloutBudget(max_generations) if max_generations is not None else None
        self._recorded = 0
        self._round_count = 0
        self._rounds: dict[int, dict[str, Batch]] = {}
        self._started_at: float | None = None
        self._open: tuple[str, ExitStack, Steered] | None = None
        self._sealed_counts: dict[str, int] = {}
        self._floors: dict[str, float] = {}
        self._rescoring = False

    def __enter__(self) -> StudySession:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def start(self) -> None:
        """Validate the study, record it, measure its baseline, and resume or open its rounds.

        On a store that already contains the study, `round_index` is the highest recorded round and the count of
        candidates in it is the number of first result versions recorded with it. Otherwise the batches of round 0
        are recorded. After the baseline, feasibility is checked again with the measured noise floors.

        Raises:
            ValueError: If the study does not validate.
            StudyInfeasibleError: If the study is infeasible and `allow_infeasible` is False.
            StudyError: If the baseline cannot be used.
        """
        if self._started_at is not None:
            return
        study, store = self.study, self.store
        study.validate(store, allow_infeasible=self.allow_infeasible)
        record_study(store, study)
        evidence.ensure_baseline(study, self.factory, store)
        problems = study.feasibility(store)
        if problems:
            message = "The study is infeasible with its measured noise floors: " + "; ".join(problems) + "."
            if not self.allow_infeasible:
                raise StudyInfeasibleError(message)
            logger.warning("%s It proceeds under allow_infeasible=True.", message)
        self._rounds = evidence.load_rounds(store, study)
        if not self._rounds:
            evidence.record_round(store, study, 0, self._new_batches(0))
            self._rounds = evidence.load_rounds(store, study)
        self.round_index = max(self._rounds)
        self._round_count = self.archive.first_versions(self.round_index)
        self._started_at = time.monotonic()
        if self._round_count >= study.settings.round_candidates:
            self._advance_round()

    def _require_started(self) -> None:
        if self._started_at is None:
            raise RuntimeError("The session is not started; call start() first.")

    def _new_batches(self, round_index: int) -> dict[str, Batch]:
        return evidence.round_batches(self.study, self.store, round_index, allow_infeasible=self.allow_infeasible)

    @property
    def rounds(self) -> dict[int, dict[str, Batch]]:
        """The batches of every round so far, by round index and benchmark id."""
        return self._rounds

    def batches(self, round_index: int) -> dict[str, Batch]:
        """The batches of a round, by benchmark id."""
        return self._rounds[round_index]

    def n_sealed(self, benchmark: Benchmark) -> int:
        """The number of usable sealed items of a benchmark."""
        if benchmark.id not in self._sealed_counts:
            self._sealed_counts[benchmark.id] = len(evidence.usable(self.study, self.store, benchmark, "sealed"))
        return self._sealed_counts[benchmark.id]

    def sd_floor(self, benchmark: Benchmark) -> float:
        """The per-item standard deviation floor of a benchmark (`evidence.sd_floor`)."""
        if benchmark.id not in self._floors:
            self._floors[benchmark.id] = evidence.sd_floor(self.study, self.store, benchmark)
        return self._floors[benchmark.id]

    def _over_limits(self) -> bool:
        if self.budget is not None and not self.budget:
            return True
        if self.max_seconds is not None and self._started_at is not None:
            return time.monotonic() - self._started_at >= self.max_seconds
        return False

    def exhausted(self) -> bool:
        """Whether a limit is reached (candidates recorded, generations, or wall time)."""
        if self.max_candidates is not None and self._recorded >= self.max_candidates:
            return True
        return self._over_limits()

    def check(self, deadline: float | None, *, limited: bool = True) -> None:
        """Raise `Stopped` when the deadline has passed or, when `limited`, a generation or time limit is reached."""
        if deadline is not None and time.monotonic() > deadline:
            raise Stopped("timeout")
        if limited and self._over_limits():
            raise Stopped("budget")

    def _steered(self, recipe: Recipe) -> Steered:
        if self._open is not None and self._open[0] == recipe.id:
            return self._open[2]
        self._close_steered()
        stack = ExitStack()
        try:
            steered = stack.enter_context(self.factory.steered(recipe))
        except Exception as error:
            stack.close()
            raise SteeringFailure(f"Steering recipe {recipe.id} failed ({type(error).__name__}: {error}).") from error
        self._open = (recipe.id, stack, steered)
        return steered

    def _close_steered(self) -> None:
        if self._open is not None:
            _, stack, _ = self._open
            self._open = None
            stack.close()

    def release_pipeline(self) -> None:
        """Close the steered pipeline the session keeps open between measurements of one recipe."""
        self._close_steered()

    def measure(self, recipe: Recipe, benchmark: Benchmark, item_ids: Sequence[str], deadline: float | None, *,
                limited: bool = True) -> tuple[str, int]:
        """Measure a recipe on items of a benchmark at replicate 0 with `evaluate_items`.

        The steered pipeline of the last measured recipe is kept open, and consecutive measurements of one recipe
        steer it once.

        Args:
            recipe: The recipe.
            benchmark: The benchmark.
            item_ids: The items.
            deadline: A `time.monotonic()` deadline, or None.
            limited: Whether the generation and time limits apply; the generations are counted either way.

        Returns:
            The recipe's resolved id and the number of rows this call generated.

        Raises:
            Stopped: If the deadline has passed or a limit is reached.
            SteeringFailure: If the recipe fails to steer.
        """
        from steerability.evaluation.items import evaluate_items

        self.check(deadline, limited=limited)
        steered = self._steered(recipe)
        results = evaluate_items(steered, benchmark, list(item_ids), store=self.store, replicate=0)
        generated = sum(1 for result in results.values() if result.generated)
        if self.budget is not None and generated:
            self.budget.charge(generated)
        return steered.resolved_id, generated

    def fit(self, recipes: Sequence[Recipe]) -> None:
        """Run the fits of recipes with `factory.prepare`, closing the open steered pipeline first."""
        self._close_steered()
        self.factory.prepare(recipes)

    def prepare(self, candidates: Sequence[Candidate]) -> None:
        """Run the fits of a batch of candidates before any of them generates.

        The plan of each candidate that binds and has no recorded result is made, and `factory.prepare` runs once on
        all their recipes. A failure is logged, and each candidate then fits in its own stage 1.

        Args:
            candidates: The candidates.
        """
        recipes: list[Recipe] = []
        for candidate in candidates:
            try:
                plan = make_plan(self.study, candidate)
            except RecipeError:
                continue
            if self.archive.latest(plan.candidate_id) is None:
                recipes.extend(plan.recipes)
        if not recipes:
            return
        try:
            self.fit(recipes)
        except Exception:
            logger.warning("Preparing the fits of %d candidate(s) failed; each fits in its own stage 1.",
                           len(candidates), exc_info=True)

    def screen(self, candidate: Candidate, *, proposer: str = "", deadline: float | None = None) -> Partial:
        """Run stages 0 to 3 of a candidate.

        A candidate with a recorded result returns it in `Partial.result` without model work. A candidate that does
        not bind returns an `invalid` result that is not recorded. When the session is exhausted, a `budget` result
        that is not recorded is returned. Otherwise the stages run, and a rejection in one of them is recorded.

        Args:
            candidate: The candidate.
            proposer: The name of the proposer.
            deadline: A `time.monotonic()` deadline checked between measurements, or None.

        Returns:
            The partial evaluation; its `result` is set when the candidate is decided.
        """
        self._require_started()
        return evaluator.screen(self, candidate, proposer=proposer, deadline=deadline)

    def finish(self, partial: Partial, *, deadline: float | None = None) -> Result:
        """Run stages 4 and 5 of a screened candidate, recording its result.

        Args:
            partial: The output of `screen`.
            deadline: A `time.monotonic()` deadline checked between measurements, or None.

        Returns:
            The result (the one `screen` decided, when set).
        """
        self._require_started()
        return evaluator.finish(self, partial, deadline=deadline)

    def evaluate(self, candidate: Candidate, *, proposer: str = "", deadline: float | None = None) -> Result:
        """Evaluate a candidate, `finish(screen(candidate))`."""
        return self.finish(self.screen(candidate, proposer=proposer, deadline=deadline), deadline=deadline)

    def record(self, candidate: Candidate, result: Result, *, proposer: str) -> Result:
        """Record a result in the archive and advance the round when it completes one.

        Returns:
            The result.
        """
        self.archive.record(candidate, result, proposer=proposer)
        if result.version == 1:
            self._recorded += 1
            if result.round_index == self.round_index:
                self._round_count += 1
                if self._round_count >= self.study.settings.round_candidates and not self._rescoring:
                    self._advance_round()
        return result

    def _advance_round(self) -> None:
        self.round_index += 1
        evidence.record_round(self.store, self.study, self.round_index, self._new_batches(self.round_index))
        self._rounds = evidence.load_rounds(self.store, self.study)
        self._round_count = self.archive.first_versions(self.round_index)
        logger.info("Study %s starts round %d.", self.study.name, self.round_index)
        self.rescore()

    def rescore(self) -> None:
        """Re-score the `settings.rescore_top` accepted results with the highest fitness on the current round.

        Each operating recipe is measured on the current round's fresh blocks, and its tables are rebuilt from every
        round's fresh-block items. A point that is still eligible gets a new version with the new tables and fitness.
        Otherwise, for a candidate with a solved axis, the next positions in the recorded ranking are confirmed on the
        current round's fresh blocks in turn, and a candidate with no position left gets a rejected version.
        Re-scoring stops when a generation or time limit is reached.
        """
        self._require_started()
        self._rescoring = True
        try:
            ranked = sorted(self.archive.points(), key=lambda result: (-result.fitness, result.candidate_id))
            for result in ranked[:self.study.settings.rescore_top]:
                try:
                    new = evaluator.rescore_result(self, result)
                except Stopped:
                    break
                except SteeringFailure:
                    logger.warning("Re-scoring candidate %s failed.", result.candidate_id, exc_info=True)
                    continue
                if new is not None:
                    self.archive.record(self.archive.candidate(result.candidate_id), new, proposer="rescore")
        finally:
            self._rescoring = False

    def finalists(self) -> list[Finalist]:
        """The points to take to certification, safest first.

        The accepted results whose gain lower bound (`behavior.lower(fitness_alpha)`) is at least `min_gain` are
        taken, and the envelope result is chosen at `settings.finalists` evenly spaced levels from their lowest to
        their highest level, without repeated resolved ids. Each finalist is then measured on up to
        `confirm_looks(margin)[2]` usable search items per constraint that it has no evidence row for (in a seeded
        order). When at least `confirm_looks(margin)[1]` such items exist on every constraint, its level comes from
        those items alone and `fresh` is set. Otherwise it keeps its pooled level. Since each call measures items the
        point has no rows for, a second call measures other items.

        Returns:
            The finalists, ordered by level, lowest first.
        """
        self._require_started()
        settings = self.study.settings
        candidates = [
            result for result in self.archive.points()
            if result.operating.behavior.lower(settings.fitness_alpha) >= settings.min_gain
            and result.level is not None
        ]
        if not candidates:
            return []
        levels = [result.level for result in candidates]
        low, high = min(levels), max(levels)
        count = settings.finalists
        targets = [low + (high - low) * index / (count - 1) for index in range(count)] if count > 1 else [low]
        unique: dict[str, Result] = {}
        for result in self.archive.envelope(targets, among=candidates):
            if result is not None and result.operating.resolved_id not in unique:
                unique[result.operating.resolved_id] = result
        finalists = []
        for result in unique.values():
            level, fresh = self._fresh_level(result)
            finalists.append(Finalist(result, level, fresh))
        return sorted(finalists, key=lambda finalist: (finalist.level, finalist.result.candidate_id))

    def _fresh_level(self, result: Result) -> tuple[float, bool]:
        study, store = self.study, self.store
        point = result.operating
        if not study.constraints:
            return result.level, False
        selections = []
        for constraint in study.constraints:
            benchmark = constraint.benchmark
            looks = confirm_looks(constraint.margin)
            measured = evidence.point_rows(study, store, point.resolved_id, benchmark)
            pool = sorted(evidence.usable(study, store, benchmark, "search"))
            order = random.Random(derive_seed(study.seed, "finalist", benchmark.id)).sample(pool, len(pool))
            available = [item for item in order if item not in measured]
            if len(available) < looks[1]:
                return result.level, False
            selections.append((benchmark, available[:looks[2]]))
        candidate = self.archive.candidate(result.candidate_id)
        try:
            recipe = evaluator.recipe_at(make_plan(study, candidate), point.position)
        except (RecipeError, AttributeError):
            logger.warning("Candidate %s cannot be rebuilt; it keeps its pooled level.", result.candidate_id)
            return result.level, False
        capability = {}
        for benchmark, items in selections:
            resolved, _ = self.measure(recipe, benchmark, items, None, limited=False)
            capability[benchmark.name] = evidence.paired(study, store, resolved, benchmark, items)
        margins = {constraint.benchmark.name: constraint.margin for constraint in study.constraints}
        return level_of(capability, margins), True

    def close(self) -> None:
        """Close the open steered pipeline, and release the factory if the session built it."""
        self._close_steered()
        if self._owns_factory:
            self.factory.release()


__all__ = ["Finalist", "StudySession"]
