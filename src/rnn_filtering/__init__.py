"""HMM and RNN module for studying RNN approximations to Bayesian forward filtering of
discrete time, discrete space HMMs. Implemented in JAX."""

from . import hmm, rnn
from .training import train_on_hmm

__all__ = ["hmm", "rnn", "train_on_hmm"]

__version__ = "1.1.0"
