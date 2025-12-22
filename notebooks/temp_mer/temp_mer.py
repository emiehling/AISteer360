from datasets import load_dataset
from peft import PeftType
from transformers import AutoTokenizer

from aisteer360.algorithms.structural_control.mer.control import MER
from aisteer360.algorithms.structural_control.wrappers.trl.sfttrainer.control import SFT
from aisteer360.evaluation.benchmark import Benchmark
from aisteer360.evaluation.metrics.generic.perplexity import Perplexity
from aisteer360.evaluation.use_cases.multilingual_retention.use_case import (
    MultilingualRetention,
)


def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )


MODEL_NAME = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

oscar_de = load_dataset(
    "oscar-corpus/OSCAR-2301",
    "de",
    split="train",
    streaming=False,
    trust_remote_code=True,
)

oscar_de = oscar_de.select(range(min(10000, len(oscar_de))))


mer_train_ds = oscar_de.map(
    tokenize_fn,
    batched=True,
    remove_columns=oscar_de.column_names,  # Remove 'text', 'meta', etc.
)

lora_train_ds = oscar_de.map(
    lambda x: {"text": x["text"]},
    remove_columns=[c for c in oscar_de.column_names if c != "text"],
)

lora = SFT(
    train_dataset=lora_train_ds,
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
    train_dataset=mer_train_ds,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    learning_rate=2e-4,
    logging_steps=100,

    # replay config
    enable_replay=True,
    replay_buffer_size=5000,
    replay_ratio=0.3,
    reservoir_sampling=True,

    # KL to preserve base capabilities
    enable_kl=True,
    kl_weight=0.5,
    kl_temperature=1.0,
    kl_apply_on="all",

    # reptile for meta-learning
    enable_reptile=True,
    reptile_meta_lr=0.1,
    reptile_update_interval=50,

    # Buffer config (in-memory)
    buffer_type="memory",
)


# load eval data - German (target) + English (retention test)
oscar_de_eval = load_dataset(
    "oscar-corpus/OSCAR-2301", "de", split="train",
    streaming=False, trust_remote_code=True
).select(range(100, 200))

oscar_en_eval = load_dataset(
    "oscar-corpus/OSCAR-2301", "en", split="train",
    streaming=False, trust_remote_code=True
).select(range(100))

eval_data = [
    {"id": f"de_{i}", "language": "de", "text": ex["text"]}
    for i, ex in enumerate(oscar_de_eval)
] + [
    {"id": f"en_{i}", "language": "en", "text": ex["text"]}
    for i, ex in enumerate(oscar_en_eval)
]

use_case = MultilingualRetention(
    evaluation_data=eval_data,
    evaluation_metrics=[Perplexity()],
)

# run benchmark
benchmark = Benchmark(
    use_case=use_case,
    base_model_name_or_path=MODEL_NAME,
    steering_pipelines={
        "baseline": [],
        "lora": [lora],
        "mer": [mer],
    },
    gen_kwargs={"max_new_tokens": 1},  # todo: change when we add more metrics + external bechmarks
    device_map="auto",
)

profiles = benchmark.run()
benchmark.export(profiles, save_dir="./multilingual_retention_profiles/")
