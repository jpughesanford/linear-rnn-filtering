"""RNN submodule for sequence modelling over vector-valued inputs."""

from .loss_functions import LOSS_MAP, hilbert_distance, kl_divergence, one_norm
from .models import AbstractRNN, ExactRNN, ModelA, ModelB
from .parameters import Parameter, register_parameter_type
from .types import ConstraintType, LossType

__all__ = [
    "AbstractRNN",
    "ExactRNN",
    "ModelA",
    "ModelB",
    "Parameter",
    "register_parameter_type",
    "LossType",
    "ConstraintType",
    "LOSS_MAP",
    "kl_divergence",
    "hilbert_distance",
    "one_norm",
]
