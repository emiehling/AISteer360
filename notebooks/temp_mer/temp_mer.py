from itertools import islice

from datasets import Dataset, load_dataset
from peft import PeftType
from transformers import AutoTokenizer

from aisteer360.algorithms.structural_control.mer.control import MER
from aisteer360.algorithms.structural_control.wrappers.trl.sfttrainer.control import SFT
from aisteer360.evaluation.benchmark import Benchmark
from aisteer360.evaluation.metrics.generic.perplexity import Perplexity
from aisteer360.evaluation.use_cases.multilingual_retention.use_case import (
    MultilingualRetention,
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def load_mc4_lang_subset(lang: str, num_samples=10000, seed=42):
    if lang == "en":
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    else:
        ds = load_dataset("allenai/c4", "multilingual", split="train", streaming=True)
    samples = list(islice(ds, num_samples))
    return Dataset.from_list(samples).shuffle(seed=seed)


# training data
mc4_lang_de = load_mc4_lang_subset("de")
mc4_lang_de = mc4_lang_de.shuffle(seed=111)
mc4_lang_de = mc4_lang_de.select(range(min(10000, len(mc4_lang_de))))
train_ds = mc4_lang_de.map(
    lambda x: {"text": x["text"]},
    remove_columns=[col for col in mc4_lang_de.column_names if col != "text"]
)

# replay dataset
replay_ds = load_mc4_lang_subset("en", num_samples=5000, seed=42)
replay_ds = replay_ds.map(
    lambda x: {"text": x["text"]},
    remove_columns=[col for col in replay_ds.column_names if col != "text"]
)

# eval data (German target + English retention)
mc4_lang_de_eval = load_mc4_lang_subset("de", num_samples=200).select(range(100, 200))
mc4_lang_en_eval = load_mc4_lang_subset("en", num_samples=100)
eval_data = [{"id": f"de_{i}", "language": "de", "text": ex["text"]} for i, ex in enumerate(mc4_lang_de_eval)] + \
            [{"id": f"en_{i}", "language": "en", "text": ex["text"]} for i, ex in enumerate(mc4_lang_en_eval)]

# controls
lora = SFT(
    train_dataset=train_ds,
    output_dir="models/lora-german",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    max_seq_length=512,
    logging_steps=100,
    save_strategy="no",
    report_to="none",

    # LoRA config
    use_peft=True,
    peft_type=PeftType.LORA,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
)

mer = MER(
    train_dataset=train_ds,
    output_dir="models/mer-german",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    learning_rate=1e-4,
    max_length=512,
    logging_steps=100,
    save_strategy="no",
    report_to="none",

    replay_enabled=True,
    replay_dataset=replay_ds,
    replay_rate=1.0,

    kl_enabled=True,
    kl_beta=0.001,

    reptile_enabled=True,
    reptile_steps=50,
    reptile_lr=0.1,

    use_peft=True,
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    merge_lora_after_train=False,
)

mer_kl_only = MER(
    train_dataset=train_ds,
    output_dir="models/mer-kl-only",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    learning_rate=1e-4,
    max_length=512,

    replay_enabled=False,
    kl_enabled=True,
    kl_beta=0.01,

    use_peft=True,
    lora_r=32,
    lora_alpha=64,
)

mer_replay_only = MER(
    train_dataset=train_ds,
    output_dir="models/mer-replay-only",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    learning_rate=1e-4,
    max_length=512,

    replay_enabled=True,
    replay_dataset=replay_ds,
    replay_rate=1.0,

    kl_enabled=False,

    use_peft=True,
    lora_r=32,
    lora_alpha=64,
)

# use case
use_case = MultilingualRetention(
    evaluation_data=eval_data,
    evaluation_metrics=[
        Perplexity(model_or_id="Qwen/Qwen2.5-7B")
    ],
)

# benchmark
benchmark = Benchmark(
    use_case=use_case,
    base_model_name_or_path=MODEL_NAME,
    steering_pipelines={
        "baseline": [],
        "lora": [lora],
        "mer": [mer],
        # "mer_kl_only": [mer_kl_only],
        # "mer_replay_only": [mer_replay_only],
    },
    gen_kwargs={"max_new_tokens": 1},
    device_map="auto",
)

profiles = benchmark.run()
benchmark.export(profiles, save_dir="./multilingual_retention_profiles/")
