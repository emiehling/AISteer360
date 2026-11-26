"""The session of a running study, the one public entry to evaluation.

Search engines (run by `run_study`, in process or through an `EvaluationService`) and certification evaluate through
a `StudySession`. The session keeps the round counter, the limits, and the re-scoring. Every caller therefore advances
rounds in the same way.
"""
from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from steerability.algorithms.core.identity import derive_seed
from steerability.algorithms.core.recipe import Recipe, RecipeError
from steerability.algorithms.core.sweeps import PipelineFactory
from steerability.optimization.evaluation import evaluator
from steerability.optimization.evaluation.archive import Archive
from steerability.optimization.evaluation.evaluator import Partial, SteeringFailure, Stopped
from steerability.optimization.evaluation.plan import make_plan
from steerability.optimization.evaluation.result import Result, level_of
from steerability.optimization.measurement import evidence
from steerability.optimization.measurement.evidence import Batch
from steerability.optimization.measurement.stats import floor_variance, guard_band, look_sizes
from steerability.optimization.study import Candidate, Study, StudyError, StudyInfeasibleError, record_study

if TYPE_CHECKING:
    from steerability.algorithms.core.store import Store
    from steerability.algorithms.core.sweeps import Steered
    from steerability.evaluation.items import Benchmark

logger = logging.getLogger(__name__)

_VARIANCE_RATIO_RANGE = (2.0 / 3.0, 1.5)


def check_limit(*, name: str, value: float | None, count: bool) -> None:
    """Check that a limit is None or a non-negative number.

    Args:
        name: The limit's name, used in the message.
        value: The limit.
        count: Whether the limit is a count, which must be an integer.

    Raises:
        ValueError: If the limit is neither None nor a non-negative number (an integer for a count).
    """
    if value is None:
        return
    kinds = int if count else (int, float)
    if isinstance(value, bool) or not isinstance(value, kinds) or not value >= 0:
        kind = "integer" if count else "number"
        raise ValueError(f"{name} must be None or a non-negative {kind}; got {value!r}.")


@dataclass(eq=False)
class _Allowance:
    """The limits of one active allowance and the session's counts when it began."""

    candidates: int | None
    generations: int | None
    seconds: float | None
    recorded_at: int
    generations_at: int
    started_at: float

    def over(self, *, session: StudySession, candidates: bool) -> bool:
        """Whether a limit of the allowance is spent; the candidate limit counts only when `candidates`."""
        if candidates and self.candidates is not None and session.recorded - self.recorded_at >= self.candidates:
            return True
        if self.generations is not None and session.generations - self.generations_at >= self.generations:
            return True
        return self.seconds is not None and time.monotonic() - self.started_at >= self.seconds


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

    Limits belong to the session object, and a restarted process counts from zero again. The session counts the rows
    that each measurement generated (`generations`) and the candidates it recorded (`recorded`). Fits are not
    counted, and their own rollouts are bounded by `max_seconds` only. `max_candidates` and `max_generations` cap the
    two counts, and `max_seconds` runs from `start`. An `allowance` adds limits of its own for the span of a block
    (one phase of a search, for instance), and every active allowance applies. Once a limit is reached, `exhausted()`
    is true and `screen` returns a rejected result with reason `budget`. A candidate stopped by a limit or a deadline
    is not recorded.

    Args:
        study: The study.
        store: The store.
        factory: The factory that steers recipes, or None to build one for the study's protocol (released by
            `close`).
        max_candidates: The number of candidates this session may record, or None.
        max_generations: The number of generations this session may run, or None.
        max_seconds: The wall time this session may run from `start`, or None.
        allow_infeasible: Whether to run a study that fails the feasibility check, i.e., one whose pools give it low
            power. It waives no check of the baseline.

    Attributes:
        study: The study.
        store: The store.
        factory: The pipeline factory.
        archive: The study's archive.
        round_index: The current round.

    Raises:
        ValueError: If a limit is neither None nor a non-negative number (an integer for a count).
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
        check_limit(name="max_candidates", value=max_candidates, count=True)
        check_limit(name="max_generations", value=max_generations, count=True)
        check_limit(name="max_seconds", value=max_seconds, count=False)
        self.study = study
        self.store = store
        self._owns_factory = factory is None
        self.factory = factory if factory is not None else PipelineFactory(study.protocol, store=store)
        self.archive = Archive(study, store)
        self.round_index = 0
        self.max_candidates = max_candidates
        self.max_generations = max_generations
        self.max_seconds = max_seconds
        self.allow_infeasible = allow_infeasible
        self._generations = 0
        self._allowances: list[_Allowance] = []
        self._recorded = 0
        self._round_count = 0
        self._rounds: dict[int, dict[str, Batch]] = {}
        self._started_at: float | None = None
        self._open: tuple[str, ExitStack, Steered] | None = None
        self._held_out_counts: dict[str, int] = {}
        self._looks: dict[str, tuple[int, int, int]] = {}
        self._rescoring = False

    def __enter__(self) -> StudySession:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def start(self) -> None:
        """Validate the study, check its feasibility, measure its baseline, and resume or open its rounds.

        The steps are:

        1. `study.validate(store)`.
        2. `evidence.feasibility` before the baseline. Without a baseline it uses raw pool sizes and the floor
           variance, and a problem it reports does not depend on what the baseline shows.
        3. The study is recorded and its baseline measured (`evidence.ensure_baseline`).
        4. `evidence.feasibility` again, at the design variance and the usable pool sizes.
        5. One line per constraint is logged with its design variance, baseline variance, floor, block sizes, looks,
           band, and margin.
        6. On a store that already contains the study, `round_index` is the highest recorded round and the count of
           candidates in it is the number of first result versions recorded with it. Otherwise the batches of round
           0 are recorded.

        At steps 2 and 4, problems raise `StudyInfeasibleError`, unless `allow_infeasible` is True, in which case
        they are logged as a warning.

        Raises:
            ValueError: If the study does not validate.
            StudyInfeasibleError: If the study is infeasible and `allow_infeasible` is False.
            StudyError: If the baseline cannot be used.
        """
        if self._started_at is not None:
            return
        study, store = self.study, self.store
        study.validate(store)
        self._check_feasibility("whatever its baseline shows")
        record_study(store, study)
        evidence.ensure_baseline(study, self.factory, store)
        self._check_feasibility(
            "at its measured baseline",
            "A study with another margin on the same store reuses the baseline's evidence rows and generates nothing "
            "to try it.",
        )
        self._log_sizes()
        self._rounds = evidence.load_rounds(store, study)
        if not self._rounds:
            evidence.record_round(store, study, 0, self._new_batches(0))
            self._rounds = evidence.load_rounds(store, study)
        self.round_index = max(self._rounds)
        self._round_count = self.archive.first_versions(self.round_index)
        self._started_at = time.monotonic()
        if self._round_count >= study.settings.round_candidates:
            self._advance_round()

    def _check_feasibility(self, where: str, note: str = "") -> None:
        problems = evidence.feasibility(self.study, self.store)
        if not problems:
            return
        listed = "\n".join(f"- {problem}." for problem in problems)
        message = f"Study {self.study.name!r} is infeasible {where}:\n{listed}"
        if note:
            message += f"\n{note}"
        if not self.allow_infeasible:
            raise StudyInfeasibleError(message)
        logger.warning("%s\nIt proceeds under allow_infeasible=True.", message)

    def _log_sizes(self) -> None:
        study, store, settings = self.study, self.store, self.study.settings
        for constraint in study.constraints:
            benchmark = constraint.benchmark
            variance = evidence.design_variance(study, store, benchmark)
            ladder, last = evidence.block_sizes(study, store, benchmark)
            n_held_out = self.n_held_out(benchmark)
            band = 0.0
            if settings.guard_band and n_held_out:
                band = guard_band(math.sqrt(variance / last), math.sqrt(variance / n_held_out), alpha=settings.alpha,
                                  power=settings.power, benchmarks=len(study.constraints))
            logger.info(
                "%s: design variance %.4f (baseline %.4f, floor %.4f), last look %d, ladder block %d, looks %s, band "
                "%.2f points, margin %.2f points.", benchmark.name, variance,
                evidence.baseline_variance(study, store, benchmark) or 0.0, floor_variance(constraint.margin), last,
                ladder, self.looks(benchmark), 100 * band, 100 * constraint.margin,
            )

    def _require_started(self) -> None:
        if self._started_at is None:
            raise RuntimeError("The session is not started; call start() first.")

    def _new_batches(self, round_index: int) -> dict[str, Batch]:
        return evidence.round_batches(self.study, self.store, round_index)

    @property
    def rounds(self) -> dict[int, dict[str, Batch]]:
        """The batches of every round so far, by round index and benchmark id."""
        return self._rounds

    def batches(self, round_index: int) -> dict[str, Batch]:
        """The batches of a round, by benchmark id."""
        return self._rounds[round_index]

    def n_held_out(self, benchmark: Benchmark) -> int:
        """The number of usable held-out items of a benchmark."""
        if benchmark.id not in self._held_out_counts:
            self._held_out_counts[benchmark.id] = len(evidence.usable(self.study, self.store, benchmark, "held_out"))
        return self._held_out_counts[benchmark.id]

    def looks(self, benchmark: Benchmark) -> tuple[int, int, int]:
        """The cumulative item counts of the three looks of a confirmation on a constraint.

        They are `look_sizes` of the constraint's fresh block size (`evidence.block_sizes`), the same in every round.
        """
        if benchmark.id not in self._looks:
            self._looks[benchmark.id] = look_sizes(evidence.block_sizes(self.study, self.store, benchmark)[1])
        return self._looks[benchmark.id]

    @property
    def generations(self) -> int:
        """The number of rows this session generated, counted by `measure`."""
        return self._generations

    @property
    def recorded(self) -> int:
        """The number of candidates this session recorded (their first result versions)."""
        return self._recorded

    @contextmanager
    def allowance(
        self,
        *,
        candidates: int | None = None,
        generations: int | None = None,
        seconds: float | None = None,
    ) -> Iterator[None]:
        """Limit the session further for the span of a block.

        On entry the allowance notes `recorded`, `generations`, and the time. While the block runs, `exhausted()` is
        also true once the block has recorded `candidates` candidates, generated `generations` rows, or run for
        `seconds`, and a running evaluation is stopped with `budget` once the generations or the seconds are spent.
        The candidate limit is checked between evaluations, as `max_candidates` is. Allowances nest, and every active
        allowance applies. Measurements made with `limited=False` ignore allowances, as they ignore the session's own
        limits.

        Args:
            candidates: The number of candidates the block may record, or None.
            generations: The number of rows the block may generate, or None.
            seconds: The wall time the block may run, or None.

        Yields:
            None.

        Raises:
            ValueError: If a limit is neither None nor a non-negative number (an integer for a count).
        """
        check_limit(name="candidates", value=candidates, count=True)
        check_limit(name="generations", value=generations, count=True)
        check_limit(name="seconds", value=seconds, count=False)
        allowance = _Allowance(
            candidates=candidates,
            generations=generations,
            seconds=seconds,
            recorded_at=self._recorded,
            generations_at=self._generations,
            started_at=time.monotonic(),
        )
        self._allowances.append(allowance)
        try:
            yield
        finally:
            self._allowances = [active for active in self._allowances if active is not allowance]

    def _over_limits(self) -> bool:
        if self.max_generations is not None and self._generations >= self.max_generations:
            return True
        started = self._started_at
        if self.max_seconds is not None and started is not None and time.monotonic() - started >= self.max_seconds:
            return True
        return any(allowance.over(session=self, candidates=False) for allowance in self._allowances)

    def exhausted(self) -> bool:
        """Whether a limit of the session or of an active allowance is reached (candidates, generations, or time)."""
        if self.max_candidates is not None and self._recorded >= self.max_candidates:
            return True
        if any(allowance.over(session=self, candidates=True) for allowance in self._allowances):
            return True
        return self._over_limits()

    def check(self, deadline: float | None, *, limited: bool = True) -> None:
        """Raise `Stopped` when the deadline has passed or, when `limited`, a generation or time limit is reached.

        The generation and time limits are those of the session and of every active allowance.
        """
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
        self._generations += generated
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
        self._log_variances()
        self.rescore()

    def _log_variances(self) -> None:
        """Log each constraint's design variance next to the mean variance of the ranked ladder tables."""
        study, store = self.study, self.store
        results = self.archive.results()
        low, high = _VARIANCE_RATIO_RANGE
        for constraint in study.constraints:
            name = constraint.benchmark.name
            tables = [point.capability[name] for result in results for point in result.ladder
                      if point.rank is not None and name in point.capability]
            count = sum(table.n for table in tables)
            if not count:
                continue
            mean = sum(table.n * table.variance for table in tables) / count
            design = evidence.design_variance(study, store, constraint.benchmark)
            ratio = mean / design
            log = logger.info if low <= ratio <= high else logger.warning
            log("%s: design variance %.4f, item-weighted mean variance %.4f over %d ranked ladder table(s) (ratio "
                "%.2f; the expected range is %.2f to %.2f).", name, design, mean, len(tables), ratio, low, high)

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

    def plan_finalists(self) -> list[dict]:
        """Select the finalists and, for each, the search items its level is measured on, without measuring.

        The accepted results whose gain lower bound (`behavior.lower(fitness_alpha)`) is at least `min_gain` are taken,
        and the envelope result is chosen at `settings.finalists` evenly spaced levels from their lowest to their
        highest level, without repeated resolved ids. For each finalist, up to `looks(benchmark)[2]` usable search
        items per constraint that it has no evidence row for are selected (in a seeded order). Its `fresh_items` are
        empty when fewer than `looks(benchmark)[1]` such items exist on some constraint.

        Returns:
            One record per finalist, `{"candidate_id", "version", "fresh_items", "level": None, "fresh": False}`,
            whose `fresh_items` map a constraint's benchmark id to its item ids.
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
        return [
            {"candidate_id": result.candidate_id, "version": result.version, "fresh_items": self._fresh_items(result),
             "level": None, "fresh": False}
            for result in unique.values()
        ]

    def _fresh_items(self, result: Result) -> dict[str, list[str]]:
        study, store = self.study, self.store
        selections: dict[str, list[str]] = {}
        for constraint in study.constraints:
            benchmark = constraint.benchmark
            looks = self.looks(benchmark)
            measured = evidence.point_rows(study, store, result.operating.resolved_id, benchmark)
            pool = sorted(evidence.usable(study, store, benchmark, "search"))
            order = random.Random(derive_seed(study.seed, "finalist", benchmark.id)).sample(pool, len(pool))
            available = [item for item in order if item not in measured]
            if len(available) < looks[1]:
                return {}
            selections[benchmark.id] = available[:looks[2]]
        return selections

    def finalists(self, plan: Sequence[Mapping] | None = None) -> list[Finalist]:
        """The points to take to certification, safest first.

        Each finalist of the plan (`plan_finalists()` when None) is measured on its recorded `fresh_items`, and its
        level comes from exactly those items (`fresh` is set). A finalist without fresh items, or whose candidate
        cannot be rebuilt, keeps its pooled level, which is logged. Measurement reuses stored rows, and a second call
        with the same plan after an interruption measures the same items.

        Args:
            plan: Records from `plan_finalists`, or None.

        Returns:
            The finalists, ordered by level, lowest first.

        Raises:
            StudyError: If a planned result version is not in the archive.
        """
        self._require_started()
        if plan is None:
            plan = self.plan_finalists()
        finalists = []
        for record in plan:
            versions = {result.version: result for result in self.archive.versions(record["candidate_id"])}
            result = versions.get(int(record["version"]))
            if result is None or result.operating is None:
                raise StudyError(f"Finalist {record['candidate_id']} version {record['version']} is not in the "
                                 "archive.")
            level, fresh = self._fresh_level(result, record.get("fresh_items") or {})
            finalists.append(Finalist(result, level, fresh))
        return sorted(finalists, key=lambda finalist: (finalist.level, finalist.result.candidate_id))

    def _fresh_level(self, result: Result, fresh_items: Mapping[str, Sequence[str]]) -> tuple[float, bool]:
        study, store = self.study, self.store
        if not study.constraints:
            return result.level, False
        if not fresh_items:
            logger.info("Finalist %s keeps its pooled level %.3f (too few unmeasured search items).",
                        result.candidate_id, result.level)
            return result.level, False
        candidate = self.archive.candidate(result.candidate_id)
        try:
            recipe = evaluator.recipe_at(make_plan(study, candidate), result.operating.position)
        except (RecipeError, AttributeError):
            logger.info("Finalist %s cannot be rebuilt; it keeps its pooled level %.3f.", result.candidate_id,
                        result.level)
            return result.level, False
        by_id = {constraint.benchmark.id: constraint.benchmark for constraint in study.constraints}
        capability = {}
        for benchmark_id, items in fresh_items.items():
            benchmark = by_id[benchmark_id]
            resolved, _ = self.measure(recipe, benchmark, list(items), None, limited=False)
            capability[benchmark.name] = evidence.paired(study, store, resolved, benchmark, list(items))
        margins = {constraint.benchmark.name: constraint.margin for constraint in study.constraints}
        return level_of(capability, margins), True

    def close(self) -> None:
        """Close the open steered pipeline, and release the factory if the session built it."""
        self._close_steered()
        if self._owns_factory:
            self.factory.release()


__all__ = ["Finalist", "StudySession", "check_limit"]
