"""Proposers: train-time candidate-generation components for input-control optimization loops."""
from .base import Proposer
from .reflection import ReflectionProposer

__all__ = ["Proposer", "ReflectionProposer"]
