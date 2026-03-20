"""RNN submodule for studying RNN approximations to Bayesian filtering"""

from .models import AbstractRNN, ExactRNN, ModelA, ModelB
from .parameters import Parameter, register_parameter_type

__all__ = ["AbstractRNN", "ExactRNN", "ModelA", "ModelB", "Parameter", "register_parameter_type"]
