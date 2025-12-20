from datasets import Dataset, IterableDataset, concatenate_datasets, load_dataset
from peft import PeftType

from aisteer360.algorithms.structural_control.mer.control import MER
from aisteer360.algorithms.structural_control.wrappers.trl.sfttrainer.control import SFT
from aisteer360.evaluation.benchmark import Benchmark
from aisteer360.evaluation.metrics.generic.perplexity import Perplexity
from aisteer360.evaluation.use_cases.multilingual_retention.use_case import (
    MultilingualRetention,
)

# from aisteer360.evaluation.metrics.generic.relevance import Relevance
# from aisteer360.evaluation.metrics.custom.some_metric import SomeCustomMetric




def build_fr_de_cpt_datasets(
    fr_slice: str = "train[:1%]",
    de_slice: str = "train[:1%]",
) -> tuple[Dataset, IterableDataset]:
    """
    Build a small French→German continual-pretraining dataset from OSCAR.

    Returns:
        - ds_lora: map-style Dataset for standard fine-tuning / LoRA.
        - ds_mer: IterableDataset representing the same data as a stream
                  (first all FR docs, then all DE docs).
    """
    fr = load_dataset(
        "oscar-corpus/OSCAR-2301",
        language="fr",
        split=fr_slice,
    )
    de = load_dataset(
        "oscar-corpus/OSCAR-2301",
        language="de",
        split=de_slice,
    )

    fr = fr.map(lambda ex: {**ex, "language": "fr"})
    de = de.map(lambda ex: {**ex, "language": "de"})

    # Sequential stream: Task B (FR) then Task C (DE), like MER-LLM
    multilingual_cpt = concatenate_datasets([fr, de])

    ds_lora: Dataset = multilingual_cpt
    ds_mer: IterableDataset = multilingual_cpt.to_iterable_dataset()

    return ds_lora, ds_mer


train_ds_lora, train_ds_mer_stream = build_fr_de_cpt_datasets()




lora_cpt = SFT(
    train_dataset=train_ds_lora,

    # SFT / TRL config
    output_dir="trl_models/Qwen2.5-1.5B-SFT-LoRA-CPT",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=5e-5,
    max_seq_length=1024,
    logging_steps=100,
    save_strategy="no",
    report_to="none",
    seed=123,

    # LoRA config
    use_peft=True,
    peft_type=PeftType.LORA,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    adapter_name="cpt_lora",
    merge_lora_after_train=True,  # merge into base weights for fair comparison
)


from aisteer360.algorithms.structural_control.mer.control import MER

# replay-only baseline
mer_replay = MER(
    train_dataset=train_ds_mer_stream,
    enable_replay=True,
    enable_kl=False,
    enable_reptile=False,
    replay_buffer_size=100_000,
    replay_ratio=0.25,
    num_train_epochs=1,
    learning_rate=5e-5,
)

# Replay + Reptile
mer_replay_reptile = MER(
    train_dataset=train_ds_mer_stream,
    enable_replay=True,
    enable_reptile=True,
    enable_kl=False,
    replay_buffer_size=100_000,
    replay_ratio=0.25,
    reptile_meta_lr=0.1,
    reptile_update_interval=100,
    num_train_epochs=1,
    learning_rate=5e-5,
)

# KL-only
mer_kl_only = MER(
    train_dataset=train_ds_mer_stream,
    enable_replay=False,
    enable_reptile=False,
    enable_kl=True,
    kl_teacher_model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",  # base model as teacher
    kl_weight=0.5,
    kl_apply_on="all",
    num_train_epochs=1,
    learning_rate=5e-5,
)

# replay + KL + Reptile
mer_full = MER(
    train_dataset=train_ds_mer_stream,
    enable_replay=True,
    enable_kl=True,
    enable_reptile=True,
    replay_buffer_size=200_000,
    replay_ratio=0.5,
    kl_teacher_model_name_or_path=None,  # clone initial model as teacher
    kl_weight=0.3,
    kl_apply_on="replay",
    reptile_meta_lr=0.1,
    reptile_update_interval=200,
    num_train_epochs=1,
    learning_rate=5e-5,
)





multilingual_lm = MultilingualRetention(
    evaluation_data="data/multilingual_eval.jsonl",
    evaluation_metrics=[
        Perplexity(
            model_or_id="meta-llama/Llama-2-7b-hf",  # or a smaller judge model
            batch_size=8,
            add_bos=True,
            max_length=512,
            device="cuda",
        ),
        # Relevance(...),
        # LanguageSpecificMetric(...),
    ],
    num_samples=-1,  # or cap for speed
)



benchmark = Benchmark(
    use_case=commonsense_mcqa,
    base_model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
    steering_pipelines={
        "baseline": [],
        "mer_replay": [mer_replay],
        "mer_full": [mer_full],
    },
    gen_kwargs={
        "max_new_tokens": 300,
        "do_sample": True,
        "temperature": 0.7,
    },
    device_map="auto",
)
profiles = benchmark.run()
