"""Model-free tests for the PPO TRL wrapper's tokenizer/vocab guard and its TRL import guard.

`PPOTrainer` scores the policy's own token ids with a single shared tokenizer, so the reward and value
models must share the policy's vocabulary. Most tests here concern this guard; the last checks that a
`trl.experimental.ppo` that cannot be imported skips `ppo` during discovery.
"""
from __future__ import annotations

import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

from steerability.algorithms.structural_control.wrappers.trl.ppotrainer.base_mixin import PPOTrainerMixin

PPO_PACKAGE = "steerability.algorithms.structural_control.wrappers.trl.ppotrainer"


class _TokenizerStub:
    """Minimal stand-in: only `len()` is consulted by the guard."""

    def __init__(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size

    def __len__(self) -> int:
        return self._vocab_size


def _model_stub(vocab_size: int | None) -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(vocab_size=vocab_size))


def _make_mixin(policy_vocab: int, reward_path: str = "reward/path", value_path: str | None = None):
    mixin = PPOTrainerMixin.__new__(PPOTrainerMixin)
    mixin.tokenizer = _TokenizerStub(policy_vocab)
    mixin.reward_model_name_or_path = reward_path
    mixin.value_model_name_or_path = value_path
    return mixin


class TestCheckScoringVocab:
    def test_reward_vocab_smaller_raises(self):
        mixin = _make_mixin(policy_vocab=128256)
        with pytest.raises(ValueError, match=r"reward model .* vocab_size 128100, smaller"):
            mixin._check_scoring_vocab(
                reward_model=_model_stub(128100),
                value_model=_model_stub(128256),
            )

    def test_value_vocab_smaller_raises(self):
        mixin = _make_mixin(policy_vocab=128256, value_path="value/path")
        with pytest.raises(ValueError, match=r"value model 'value/path' .* smaller"):
            mixin._check_scoring_vocab(
                reward_model=_model_stub(128256),
                value_model=_model_stub(128100),
            )

    def test_matched_vocab_passes(self):
        mixin = _make_mixin(policy_vocab=128256)
        # no exception; reward/value cover the policy vocab exactly
        mixin._check_scoring_vocab(
            reward_model=_model_stub(128256),
            value_model=_model_stub(128256),
        )

    def test_larger_scoring_vocab_passes(self):
        """A scoring model with a strictly larger vocab still covers every policy id."""
        mixin = _make_mixin(policy_vocab=32000)
        mixin._check_scoring_vocab(
            reward_model=_model_stub(50000),
            value_model=_model_stub(50000),
        )

    def test_missing_vocab_size_is_skipped(self):
        """A scoring model whose config lacks `vocab_size` is not flagged (nothing to compare)."""
        mixin = _make_mixin(policy_vocab=128256)
        mixin._check_scoring_vocab(
            reward_model=_model_stub(None),
            value_model=_model_stub(None),
        )


def test_trl_ppo_import_error_skips_ppo_in_discovery(monkeypatch, caplog):
    """A `trl.experimental.ppo` without `PPOTrainer` skips `ppo` with a logged hint and fails `steer()`."""
    from steerability.algorithms.core import registry
    from steerability.algorithms.structural_control.wrappers import trl as trl_wrappers

    # a stub module without the trainer classes makes the `from ... import` raise a plain ImportError
    monkeypatch.setitem(sys.modules, "trl.experimental.ppo", ModuleType("trl.experimental.ppo"))
    for name in [name for name in sys.modules if name == PPO_PACKAGE or name.startswith(PPO_PACKAGE + ".")]:
        monkeypatch.delitem(sys.modules, name)
    # the re-import rebinds the parent package attribute; monkeypatch restores it afterwards
    monkeypatch.setattr(trl_wrappers, "ppotrainer", trl_wrappers.ppotrainer)
    monkeypatch.setattr(registry, "REGISTRY", {})

    with caplog.at_level(logging.WARNING, logger=PPO_PACKAGE):
        registry._crawl_methods()

    structural = registry.REGISTRY["structural_control"]
    assert "ppo" not in structural
    assert "dpo" in structural
    assert "trl.experimental.ppo" in caplog.text

    unavailable_mixin = sys.modules[PPO_PACKAGE + ".base_mixin"].PPOTrainerMixin
    with pytest.raises(ImportError, match="trl.experimental.ppo"):
        unavailable_mixin.__new__(unavailable_mixin).steer(model=None)
