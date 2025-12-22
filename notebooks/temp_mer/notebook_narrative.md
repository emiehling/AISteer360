Describe components of MER

Show how each variant is instantiated in the toolkit implementation

Describe the multilingual retention use case (instantiate it from evaluate/use_cases/multilingual_retention/use_case.py)

For the benchmark class, specify a base model

Define training dataset; and formats for LoRA and MER

Apply different structural controls:
    - LoRA
    - MER versions under KL/replay/reptile

(specify external benchmarks; todo for Erik: add this feature in benchmark class)

Show:
    - LoRA: good on the new domain, but catastrophic forgetting on previous capabilities
    - MER: good on the new domain, while mitigating forgetting
