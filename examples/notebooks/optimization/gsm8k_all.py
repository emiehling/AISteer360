"""GSM8K over the train and test splits, zero-shot, as one Inspect task.

`inspect_evals/gsm8k` evaluates the test split only and takes its few-shot examples from the train split. This task
covers both splits without few-shot examples, so a study can split its items into a search pool and a sealed pool. A
question that appears in both splits is kept once.
"""
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.scorer import match
from inspect_ai.solver import generate, prompt_template
from inspect_evals.gsm8k.gsm8k import DATASET_PATH, GSM8K_DATASET_REVISION, MATH_PROMPT_TEMPLATE, record_to_sample
from inspect_evals.utils.huggingface import hf_dataset


@task
def gsm8k_all() -> Task:
    """GSM8K over the train and test splits, zero-shot."""
    samples, seen = [], set()
    for split in ("train", "test"):
        for sample in hf_dataset(path=DATASET_PATH, data_dir="main", split=split, sample_fields=record_to_sample,
                                 revision=GSM8K_DATASET_REVISION):
            if sample.id not in seen:
                seen.add(sample.id)
                samples.append(sample)
    return Task(
        dataset=MemoryDataset(samples), solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate()],
        scorer=match(numeric=True),
    )
