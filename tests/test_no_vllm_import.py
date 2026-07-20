"""Guard: importing aisteer360 (plus constructing backends) never imports `vllm`.

The vLLM path is server-side; the client compiler emits JSON and depends only on `openai`. This walks
`sys.modules` after importing the package and the backend layer and asserts `vllm` is absent.
"""
import sys


def test_no_vllm_in_sys_modules():
    import aisteer360  # noqa: F401
    import aisteer360.backends  # noqa: F401
    from aisteer360.backends.huggingface.backend import HuggingFaceBackend  # noqa: F401

    offenders = [name for name in sys.modules if name == "vllm" or name.startswith("vllm.")]
    assert not offenders, f"vllm was imported by aisteer360: {offenders}"
