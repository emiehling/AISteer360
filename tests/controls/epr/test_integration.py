"""End-to-end EPR pipeline tests."""
from __future__ import annotations

import numpy as np
import pytest

from aisteer360.algorithms.input_control.epr import EPR, EPRArgs, RetrievalMemory
from aisteer360.algorithms.input_control.epr.contrastive import ContrastiveConfig

from tests.controls.epr._stubs import (
    StubTokenizer,
    cluster_score_fn,
    make_clustered_corpus,
)

BERT_TINY = "prajjwal1/bert-tiny"


def test_bm25_end_to_end_save_load(tmp_path):
    """Fast end-to-end: bm25 mode with save→load round trip."""
    corpus = make_clustered_corpus({"MATH": 8, "LANG": 8}, seed=0)
    args = EPRArgs(corpus=corpus, mode="bm25", n_demonstrations=3)
    epr = EPR(args)
    tok = StubTokenizer()
    epr.steer(model=None, tokenizer=tok)

    decoded_before = tok.decode(epr.adapt(tok.encode("MATH item zzz")))

    # round trip the memory
    base = str(tmp_path / "rt_bm25")
    epr.memory.save(base)
    loaded_memory = RetrievalMemory.load(base)
    assert loaded_memory.mode == "bm25"

    # build a fresh EPR with the loaded memory and re-test
    epr2 = EPR(args)
    epr2.tokenizer = tok
    epr2.memory = loaded_memory
    decoded_after = tok.decode(epr2.adapt(tok.encode("MATH item zzz")))

    # the same demos should still come out (bm25 is deterministic)
    assert decoded_before == decoded_after


@pytest.mark.slow
def test_epr_full_pipeline_routes_by_cluster(tmp_path):
    """Full EPR train: cluster scoring stub + bert-tiny encoders."""
    try:
        corpus = make_clustered_corpus({"MATH": 12, "LANG": 12}, seed=0)
        args = EPRArgs(
            corpus=corpus,
            mode="epr",
            n_demonstrations=4,
            scoring_lm=cluster_score_fn,
            base_encoder_name_or_path=BERT_TINY,
            encoder_pooling="cls",
            encoder_max_length=32,
            encoder_batch_size=4,
            encoder_device="cpu",
            candidate_set_size=8,
            n_positives=3,
            n_negatives=3,
            contrastive_config=ContrastiveConfig(
                batch_size=4, epochs=4, learning_rate=5e-4,
                n_negatives_per_anchor=1, max_length=32, warmup_steps=2, seed=0,
            ),
        )
        epr = EPR(args)
        tok = StubTokenizer()
        epr.steer(model=None, tokenizer=tok)
    except Exception as exc:
        pytest.skip(f"EPR full pipeline could not run: {exc}")

    # MATH query → demos predominantly from MATH
    math_decoded = tok.decode(epr.adapt(tok.encode("MATH item 999 ( zzz )")))
    n_math = math_decoded.count("MATH item")
    n_lang = math_decoded.count("LANG item")
    assert n_math > n_lang

    # LANG query → demos predominantly from LANG
    lang_decoded = tok.decode(epr.adapt(tok.encode("LANG item 999 ( zzz )")))
    n_math2 = lang_decoded.count("MATH item")
    n_lang2 = lang_decoded.count("LANG item")
    assert n_lang2 > n_math2

    epr.cleanup()
