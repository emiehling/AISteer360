Describe components of MER

Show how each variant is instantiated in the toolkit implementation

For the benchmark class, specify a base model (already pretrained on something like mostly English web)

Apply different structural controls using some new training data:
    - LoRA fine‑tuning on a new domain/lang
    - MER continual pre‑training on a stream of new data (with replay/reptile/KL)

After training, runs existing benchmarks and shows:
    - LoRA: good on the new domain, but catastrophic forgetting on previous capabilities
    - MER: good on the new domain, while mitigating forgetting
