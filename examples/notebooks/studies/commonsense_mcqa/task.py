"""CommonsenseQA under deterministic choice shuffling, with the answer slot the model chose.

Each CommonsenseQA validation question is expanded into `num_shuffling_runs` samples, one per
deterministic shuffle of its answer choices, so accuracy and positional bias are measured over
repeated presentations of the same question. `multiple_choice()` formats and generates, `choice()`
parses and scores, and `chosen_slot()` records the index of the slot the model chose. Each sample's
metadata contains its question id and number of choices, which `positional_bias` reads together
with the chosen slots.
"""
import random
from typing import Any, Callable

import pandas as pd
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import Score, Scorer, Target, choice, mean, scorer
from inspect_ai.solver import MultipleChoiceTemplate, TaskState, multiple_choice

LETTERS = "ABCDEFGH"
CSQA_PATH = "tau/commonsense_qa"


def format_example(question: str, choices: list[str]) -> str:
    """Render one question exactly as `multiple_choice()` presents it.

    Uses the single-answer template the task's solver renders, with choices as `A) text` lines
    and the letter list comma-joined, so steering data built from this function (few-shot
    exemplars, preference pairs) matches the evaluation-time prompt character for character.

    Args:
        question: The question text.
        choices: Answer options in presentation order.

    Returns:
        The fully rendered prompt.
    """
    options = "\n".join(f"{LETTERS[i]}) {text}" for i, text in enumerate(choices))
    letters = ",".join(LETTERS[: len(choices)])
    return MultipleChoiceTemplate.SINGLE_ANSWER.format(
        question=question, choices=options, letters=letters,
    )


def shuffled_samples(num_shuffling_runs: int, shuffle_seed: int) -> Callable[[dict[str, Any]], list[Sample]]:
    """A `sample_fields` mapper expanding one record into shuffled presentations.

    Each of the `num_shuffling_runs` samples bakes one deterministic shuffle of the record's
    choices into `Sample.choices`, with the target set to the correct slot's letter under that
    shuffle. Shuffles seed `random.Random` with the string `f"{shuffle_seed}:{id}:{run}"`, so
    the dataset is a pure function of the task arguments (string seeding is deterministic
    across platforms and Python versions). Records without a single-letter in-range answer key
    (present in the dataset's test split) expand to an empty list and are skipped.

    Args:
        num_shuffling_runs: Number of shuffled presentations per question.
        shuffle_seed: Base seed for the per-sample shuffles.

    Returns:
        The mapper, for `hf_dataset(sample_fields=...)`.
    """

    def to_samples(record: dict[str, Any]) -> list[Sample]:
        choices = list(record["choices"]["text"])
        answer_key = record["answerKey"]
        if len(answer_key) != 1 or answer_key not in LETTERS[: len(choices)]:
            return []
        answer_index = LETTERS.index(answer_key)
        samples: list[Sample] = []
        for run in range(num_shuffling_runs):
            rng = random.Random(f"{shuffle_seed}:{record['id']}:{run}")
            order = list(range(len(choices)))
            rng.shuffle(order)
            samples.append(Sample(
                id=f"{record['id']}:run{run}",
                input=record["question"],
                choices=[choices[i] for i in order],
                target=LETTERS[order.index(answer_index)],
                metadata={
                    "question_id": record["id"],
                    "run": run,
                    "num_choices": len(choices),
                },
            ))
        return samples

    return to_samples


@scorer(metrics=[mean()])
def chosen_slot() -> Scorer:
    """The index of the answer slot the model chose, or -1 when it did not choose exactly one.

    Reads the choices that `multiple_choice()` marks as selected after parsing the response. The
    index is in presentation order (the order of `Sample.choices`).

    Returns:
        The scorer.
    """

    async def score(state: TaskState, target: Target) -> Score:
        chosen = [index for index, option in enumerate(state.choices) if option.correct is True]
        slot = chosen[0] if len(chosen) == 1 else -1
        return Score(value=float(slot), answer=LETTERS[slot] if slot >= 0 else None)

    return score


def positional_bias(samples: pd.DataFrame) -> float:
    """Mean total variation distance between chosen-slot distributions and uniform.

    For each question, the empirical distribution of the model's chosen slot across that
    question's shuffling runs is compared to the uniform distribution via total variation
    distance, `0.5 * sum_i |p_i - 1/n|`, and the distances are averaged over questions. Rows
    without a chosen slot (a `slot` of -1 or a missing value) are excluded, and questions with no
    chosen slot are skipped. The range is `[0, 1 - 1/n]`, and 0 means no positional preference.

    Note the finite-sample floor: with a finite number of runs over `n` slots, even a chooser with
    no positional preference has a strictly positive expected distance (its empirical histogram
    cannot be exactly uniform), so small values should be read against that floor rather than
    against zero.

    Args:
        samples: Per-item rows with columns `question_id`, `num_choices`, and `slot`.

    Returns:
        The mean distance, or NaN when no question has a chosen slot.
    """
    per_question: list[float] = []
    for _, group in samples.groupby("question_id"):
        num_choices = int(group["num_choices"].iloc[0])
        slots = pd.to_numeric(group["slot"], errors="coerce")
        slots = slots[(slots >= 0) & (slots < num_choices)].astype(int)
        if slots.empty:
            continue
        counts = slots.value_counts().reindex(range(num_choices), fill_value=0)
        uniform = 1.0 / num_choices
        per_question.append(0.5 * float((counts / counts.sum() - uniform).abs().sum()))
    if not per_question:
        return float("nan")
    return float(sum(per_question) / len(per_question))


@task
def commonsense_mcqa(
    num_questions: int = 50,
    num_shuffling_runs: int = 20,
    shuffle_seed: int = 0,
) -> Task:
    """CommonsenseQA validation questions under deterministic choice shuffling.

    Loads the first `num_questions` validation records (the `hf_dataset` limit counts records
    before the expanding mapper runs, so it counts questions rather than samples) and expands
    each into `num_shuffling_runs` shuffled presentations.

    Args:
        num_questions: Number of validation questions to load.
        num_shuffling_runs: Shuffled presentations per question.
        shuffle_seed: Base seed for the deterministic shuffles.

    Returns:
        The task, scored by `choice()` and `chosen_slot()`.
    """
    dataset = hf_dataset(
        CSQA_PATH,
        split="validation",
        sample_fields=shuffled_samples(int(num_shuffling_runs), int(shuffle_seed)),
        limit=int(num_questions),
    )
    return Task(
        dataset=dataset,
        solver=multiple_choice(),
        scorer=[choice(), chosen_slot()],
    )
