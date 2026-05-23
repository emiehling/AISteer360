"""Tests for the EPR control."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from aisteer360.algorithms.input_control.epr import EPR, EPRArgs, RetrievalMemory

from tests.controls.epr._stubs import (
    StubEncoder,
    StubTokenizer,
    cluster_score_fn,
    make_clustered_corpus,
)


def _math_lang_corpus() -> list[dict]:
    return make_clustered_corpus({"MATH": 8, "LANG": 8}, seed=0)


def _stub_encoder() -> StubEncoder:
    return StubEncoder(
        cluster_map={
            "math": np.array([1.0, 0.0], dtype=np.float32),
            "lang": np.array([0.0, 1.0], dtype=np.float32),
        },
        default_vector=np.array([0.5, 0.5], dtype=np.float32),
    )


def test_epr_is_not_stateful():
    assert EPR.is_stateful is False


def test_epr_bm25_mode_steer_produces_memory():
    args = EPRArgs(
        corpus=_math_lang_corpus(),
        mode="bm25",
        n_demonstrations=3,
    )
    epr = EPR(args)
    epr.steer(model=None, tokenizer=StubTokenizer())
    assert isinstance(epr.memory, RetrievalMemory)
    assert epr.memory.mode == "bm25"
    assert epr.memory.bm25_state is not None


def test_epr_dense_mode_steer_produces_memory(monkeypatch):
    """We bypass the real HFEncoder by patching the EPR module reference."""
    from aisteer360.algorithms.input_control.epr import control as control_mod

    def _factory(*args, **kwargs):
        return _stub_encoder()

    monkeypatch.setattr(control_mod, "HFEncoder", _factory)

    args = EPRArgs(
        corpus=_math_lang_corpus(),
        mode="dense",
        n_demonstrations=3,
    )
    epr = EPR(args)
    epr.steer(model=None, tokenizer=StubTokenizer())
    assert epr.memory is not None
    assert epr.memory.mode == "dense"
    assert epr.memory.dense_embeddings.shape == (16, 2)


def test_epr_epr_mode_steer_invokes_training(monkeypatch):
    from aisteer360.algorithms.input_control.epr import control as control_mod

    fake_input = _stub_encoder()
    fake_prompt = _stub_encoder()

    def _stub_train(**kwargs):
        return fake_input, fake_prompt

    monkeypatch.setattr(control_mod, "train_contrastive_retriever", _stub_train)

    corpus = _math_lang_corpus()
    args = EPRArgs(
        corpus=corpus,
        mode="epr",
        scoring_lm=cluster_score_fn,  # callable bypasses HF model loading
        n_demonstrations=3,
        candidate_set_size=6,
        n_positives=2,
        n_negatives=2,
    )
    epr = EPR(args)
    # use any object as the "task model"; cluster_score_fn ignores it
    epr.steer(model=object(), tokenizer=StubTokenizer())
    assert epr.memory is not None
    assert epr.memory.mode == "epr"
    assert epr.memory.dense_embeddings.shape == (len(corpus), 2)
    assert epr._input_encoder is fake_input
    assert epr._prompt_encoder is fake_prompt


def test_epr_adapt_bm25_returns_demos():
    args = EPRArgs(
        corpus=_math_lang_corpus(),
        mode="bm25",
        n_demonstrations=2,
    )
    epr = EPR(args)
    tok = StubTokenizer()
    epr.steer(model=None, tokenizer=tok)

    query_text = "MATH item 099 (foo)"
    ids = tok.encode(query_text)
    adapted = epr.adapt(ids)
    decoded = tok.decode(adapted)
    assert "Input:" in decoded
    assert "MATH" in decoded


def test_epr_adapt_dense_returns_top_k_by_inner_product(monkeypatch):
    from aisteer360.algorithms.input_control.epr import control as control_mod

    enc = _stub_encoder()
    monkeypatch.setattr(control_mod, "HFEncoder", lambda *a, **kw: enc)

    args = EPRArgs(
        corpus=_math_lang_corpus(),
        mode="dense",
        n_demonstrations=4,
    )
    epr = EPR(args)
    tok = StubTokenizer()
    epr.steer(model=None, tokenizer=tok)

    math_query = "math item zzz"
    decoded = tok.decode(epr.adapt(tok.encode(math_query)))
    # all retrieved demos for a MATH query should be MATH demos
    n_math = decoded.count("MATH item")
    n_lang = decoded.count("LANG item")
    assert n_math >= 4
    assert n_lang == 0


def test_epr_adapt_orders_demos_by_ascending_similarity(monkeypatch):
    """Closest demo (highest inner product) appears LAST in the assembled prompt."""
    from aisteer360.algorithms.input_control.epr import control as control_mod

    enc = _stub_encoder()
    monkeypatch.setattr(control_mod, "HFEncoder", lambda *a, **kw: enc)

    corpus = [
        {"input": "math example one", "output": "1"},
        {"input": "lang example one", "output": "x"},
        {"input": "math example two", "output": "2"},
    ]
    # all three retrieve identically (n_demonstrations=3); order matters
    args = EPRArgs(corpus=corpus, mode="dense", n_demonstrations=3)
    epr = EPR(args)
    tok = StubTokenizer()
    epr.steer(model=None, tokenizer=tok)

    decoded = tok.decode(epr.adapt(tok.encode("math query goes here")))

    # MATH demos should come last (closest to query token); LANG should appear before MATH.
    last_math_pos = decoded.rfind("math example")
    last_lang_pos = decoded.rfind("lang example")
    assert last_math_pos > last_lang_pos


def test_epr_adapt_respects_n_demonstrations():
    args = EPRArgs(
        corpus=_math_lang_corpus(),
        mode="bm25",
        n_demonstrations=3,
    )
    epr = EPR(args)
    tok = StubTokenizer()
    epr.steer(model=None, tokenizer=tok)

    decoded = tok.decode(epr.adapt(tok.encode("MATH item 042 (q)")))
    assert decoded.count("Input:") == 4  # 3 demos + final query


def test_epr_cleanup_releases_encoders(monkeypatch):
    from aisteer360.algorithms.input_control.epr import control as control_mod

    enc = _stub_encoder()
    monkeypatch.setattr(control_mod, "HFEncoder", lambda *a, **kw: enc)

    args = EPRArgs(
        corpus=_math_lang_corpus(),
        mode="dense",
        n_demonstrations=2,
    )
    epr = EPR(args)
    epr.steer(model=None, tokenizer=StubTokenizer())
    epr.cleanup()
    assert epr._input_encoder is None
    assert epr._prompt_encoder is None
    assert enc.cleanup_called == 1


def test_epr_adapt_before_steer_raises():
    args = EPRArgs(corpus=_math_lang_corpus(), mode="bm25", n_demonstrations=2)
    epr = EPR(args)
    with pytest.raises(RuntimeError, match="steered first"):
        epr.adapt([1, 2, 3])


def test_epr_adapt_rejects_batched_input():
    args = EPRArgs(corpus=_math_lang_corpus(), mode="bm25", n_demonstrations=2)
    epr = EPR(args)
    tok = StubTokenizer()
    epr.steer(model=None, tokenizer=tok)

    batched = torch.tensor([tok.encode("MATH a"), tok.encode("LANG b")], dtype=torch.long)
    with pytest.raises(NotImplementedError):
        epr.adapt(batched)


def test_epr_epr_mode_requires_scoring_source():
    args = EPRArgs(
        corpus=_math_lang_corpus(),
        mode="epr",
        candidate_set_size=6,
        n_positives=2,
        n_negatives=2,
    )
    epr = EPR(args)
    with pytest.raises(ValueError, match="scoring"):
        epr.steer(model=None, tokenizer=StubTokenizer())
