"""Linear RNN approximations to discrete HMM forward filtering."""

from .hmm import DiscreteHMM
from .hmm_factory import HMMFactory
from .rnn import AbstractRNN, ExactRNN, ModelA, ModelB, LossType

__all__ = [
    "DiscreteHMM",
    "HMMFactory",
    "AbstractRNN",
    "ExactRNN",
    "ModelA",
    "ModelB",
    "LossType"
]

__version__ = "0.1.0"
