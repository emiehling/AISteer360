"""Engine-gated tests for LoRA adapters selected per request on the offline `VLLMBackend`.

The whole module skips when vLLM is not installed, and running it requires a GPU-capable environment with the `vllm`
extra. The engines boot `JackFram/llama-68m`, which is not in the CI model set. vLLM applies LoRA adapters in float16
or bfloat16 only, and the engines that serve adapters run in bfloat16 with LoRA layers on the adapters' modules only
(`lora_target_modules`), which scoring under an adapter needs. Their scores are compared with float32 Hugging Face
references at a tolerance that covers bfloat16 rounding, and each adapter is identified by the reference that is
closest to its scores.
"""
import pytest

vllm = pytest.importorskip("vllm")

pytestmark = pytest.mark.network

import torch  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from steerability.algorithms.core.execution import (  # noqa: E402
    BackendSpec,
    GenerationItem,
    GenerationParams,
    LoRAArtifact,
    PreparedPrompt,
    ScoringItem,
)
from steerability.algorithms.core.protocol import Protocol  # noqa: E402
from steerability.algorithms.core.recipe import BASELINE, Recipe, entry  # noqa: E402
from steerability.algorithms.core.store import Store  # noqa: E402
from steerability.algorithms.core.sweeps import PipelineFactory  # noqa: E402
from steerability.algorithms.structural_control.load_lora.control import LoadLoRA  # noqa: E402
from steerability.backends.vllm import VLLMBackend  # noqa: E402

TINY_MODEL = "JackFram/llama-68m"
PROMPT = "The cat sat on the"
CONTINUATION = " mat and looked out of the window at the rain"
ENGINE_KWARGS = {
    "enforce_eager": True,
    "max_model_len": 512,
    "gpu_memory_utilization": 0.25,
    "enable_lora": True,
    "max_lora_rank": 8,
    "lora_target_modules": ["q_proj", "v_proj"],
}


def _write_adapter(directory, *, seed: int) -> str:
    """A rank-8 LoRA adapter of the tiny model with weights drawn from `seed`."""
    torch.manual_seed(seed)
    config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
    get_peft_model(AutoModelForCausalLM.from_pretrained(TINY_MODEL), config).save_pretrained(directory)
    return str(directory)


def _engine_spec(*, dtype: str) -> BackendSpec:
    return BackendSpec(kind="vllm", model=TINY_MODEL, options={"engine_kwargs": {**ENGINE_KWARGS, "dtype": dtype}})


def _reference_scores(*, path: str | None, prompt_ids: torch.Tensor, ref_ids: torch.Tensor) -> torch.Tensor:
    """The float32 log-probabilities of `ref_ids` after `prompt_ids`, with the adapter at `path` merged in."""
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=torch.float32)
    if path is not None:
        model = PeftModel.from_pretrained(model, path).merge_and_unload()
    with torch.no_grad():
        logits = model(torch.cat([prompt_ids, ref_ids], dim=1)).logits
    positions = torch.arange(prompt_ids.size(1) - 1, prompt_ids.size(1) - 1 + ref_ids.size(1))
    log_probs = torch.log_softmax(logits[0, positions], dim=-1)
    return log_probs.gather(-1, ref_ids[0].unsqueeze(-1)).squeeze(-1)


def test_pipelines_with_different_adapters_share_one_engine_and_each_uses_its_own(tmp_path):
    adapters = [_write_adapter(tmp_path / f"adapter-{seed}", seed=seed) for seed in (1, 2)]
    factory = PipelineFactory(
        protocol=Protocol.create(model=TINY_MODEL, backend=_engine_spec(dtype="bfloat16")),
        store=Store.open(root=tmp_path / "store"),
        device_map=None,
    )
    texts = []
    try:
        for path in (adapters[0], adapters[1], None, adapters[0]):
            recipe = BASELINE if path is None else Recipe(entries=(entry(LoadLoRA, path=path, base_model=TINY_MODEL),))
            with factory.steered(recipe=recipe) as steered:
                texts.append(steered.pipeline.generate(text=PROMPT, max_new_tokens=8, do_sample=False))
    finally:
        factory.release()
    assert factory.stats.engine_boots == 1
    assert texts[0] == texts[3] and len(set(texts[:3])) == 3


def test_each_session_scores_under_its_own_adapter(tmp_path):
    paths = [_write_adapter(tmp_path / f"adapter-{seed}", seed=seed) for seed in (1, 2)]
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    ref_ids = tokenizer(CONTINUATION, add_special_tokens=False, return_tensors="pt").input_ids
    references = [_reference_scores(path=path, prompt_ids=prompt_ids, ref_ids=ref_ids) for path in (*paths, None)]
    item = ScoringItem(prompt=PreparedPrompt.from_token_ids(token_ids=prompt_ids), ref_output_ids=ref_ids)
    backend = VLLMBackend(spec=_engine_spec(dtype="bfloat16"))
    scores = []
    try:
        for path in (*paths, None):
            adapter = None if path is None else LoRAArtifact(path=path, base_model=TINY_MODEL)
            with backend.open_session(adapter=adapter) as session:
                scores.append(session.score(items=[item], params=GenerationParams())[0].float())
    finally:
        backend.release()
    for index, score in enumerate(scores):
        distances = [float((score - reference).abs().max()) for reference in references]
        others = [distance for other, distance in enumerate(distances) if other != index]
        assert distances[index] < 0.25 and distances[index] < min(others) / 2, distances


def test_a_batch_of_scoring_requests_under_an_adapter_completes(tmp_path):
    adapter = LoRAArtifact(path=_write_adapter(tmp_path / "adapter", seed=1), base_model=TINY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    ref_ids = tokenizer(CONTINUATION, add_special_tokens=False, return_tensors="pt").input_ids
    prompts = [tokenizer(f"{PROMPT} ({index})", return_tensors="pt").input_ids for index in range(64)]
    items = [
        ScoringItem(prompt=PreparedPrompt.from_token_ids(token_ids=prompt_ids), ref_output_ids=ref_ids)
        for prompt_ids in prompts
    ]
    backend = VLLMBackend(spec=_engine_spec(dtype="bfloat16"))
    try:
        with backend.open_session(adapter=adapter) as session:
            scores = session.score(items=items, params=GenerationParams())
    finally:
        backend.release()
    assert scores.shape == (64, ref_ids.size(1)) and torch.isfinite(scores).all()


def test_an_engine_in_float32_refuses_an_adapter_and_keeps_serving_requests_without_one(tmp_path):
    adapter = LoRAArtifact(path=_write_adapter(tmp_path / "adapter", seed=1), base_model=TINY_MODEL)
    prompt_ids = AutoTokenizer.from_pretrained(TINY_MODEL)(PROMPT, return_tensors="pt").input_ids
    item = GenerationItem(prompt=PreparedPrompt.from_token_ids(token_ids=prompt_ids))
    backend = VLLMBackend(spec=_engine_spec(dtype="float32"))
    try:
        with pytest.raises(ValueError, match="runs in float32, and vLLM applies a LoRA adapter in float16"):
            backend.open_session(adapter=adapter)
        with backend.open_session() as session:
            (result,) = session.generate(items=[item], params=GenerationParams(max_new_tokens=4, greedy=True))
    finally:
        backend.release()
    assert result.output.output_ids.numel() > 0
