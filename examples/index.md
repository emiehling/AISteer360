# Notebooks

Notebooks cover basic implementations of each control in our toolkit (including examples of how to implement methods
from wrappers), as well as implementations of benchmarks.

## Controls

<div class="grid cards" markdown>

-   __Input control__

    ---

    Input control methods adapt the input (prompt) before the model is called. Current notebooks cover:

    :octicons-arrow-right-24: [FewShot](./notebooks/control_few_shot/few_shot.ipynb)

-   __Structural control__

    ---

    Structural control methods adapt the model's weights/architecture. Current notebooks cover:

    :octicons-arrow-right-24: [MergeKit wrapper](./notebooks/wrapper_mergekit/mergekit_wrapper.ipynb)

    :octicons-arrow-right-24: [TRL wrapper](./notebooks/wrapper_trl/trl_wrapper.ipynb)

-   __State control__

    ---

    State control methods influence the model's internal states (activation, attentions, etc.) at inference time. Current notebooks cover:

    :octicons-arrow-right-24: [CAST](./notebooks/control_cast/cast.ipynb)

    :octicons-arrow-right-24: [PASTA](./notebooks/control_pasta/pasta.ipynb)

-   __Output control__

    ---

    Output control methods influence the model's behavior via the `generate()` method. Current notebooks cover:

    :octicons-arrow-right-24: [DeAL](./notebooks/control_deal/deal.ipynb)

    :octicons-arrow-right-24: [RAD](./notebooks/control_rad/rad.ipynb)

    :octicons-arrow-right-24: [SASA](./notebooks/control_sasa/sasa.ipynb)

    :octicons-arrow-right-24: [ThinkingIntervention](./notebooks/control_thinking_intervention/thinking_intervention.ipynb)


</div>


## Benchmarks

<div class="grid cards" markdown>

-   :material-comment-question-outline:  __Commonsense MCQA__

    ---

    This benchmark evaluates how well a steered model (under `FewShot` and `LoRA`) performs compared to a base model on
    answering commonsense multiple-choice questions.

    [:octicons-arrow-right-24: See the benchmark](./notebooks/benchmark_commonsense_mcqa/commonsense_mcqa.ipynb)

-   :material-list-box-outline:  __Instruction following__

    ---

    This benchmark evaluates a steered model's ability to follow instructions. We compare the performance of the
    baseline model to the steered model under `PASTA`, `DeAL`, and `ThinkingIntervention`.

    [:octicons-arrow-right-24: See the benchmark](./notebooks/benchmark_instruction_following/instruction_following.ipynb)

</div>
