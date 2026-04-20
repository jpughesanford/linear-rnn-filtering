"""Loss functions for RNN training and evaluation.

Each function has the signature::

    fn(result, desired, *, clip=..., do_average=True) -> jax.Array

where ``result`` is the RNN output of shape ``(B, T, K)`` and ``desired`` is
the target distribution of shape ``(B, T, K)``.  One-hot encodings of emission
symbols are valid targets; KL(one_hot(e) ‖ q) = -log q[e] = NLL.

``LOSS_MAP`` maps :class:`LossType` strings to their corresponding function.
:data:`LossType.EMISSIONS` is intentionally absent — it is an alias handled by
``train_on_hmm`` which converts integer emissions to one-hot before calling
:meth:`AbstractRNN.train` with ``output_loss='kl'``.

Custom loss callables passed to :meth:`AbstractRNN.train` only need to accept
``(result, desired)``; the ``clip`` and ``do_average`` keyword arguments are
optional.  Custom callables used with :meth:`AbstractRNN.sample_loss` must also
support ``do_average=False``.
"""

from collections.abc import Callable

import jax.numpy as jnp

from .types import LossType

__all__ = ["LOSS_MAP", "kl_divergence", "hilbert_distance", "one_norm"]


# ------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------


def kl_divergence(
    result: jnp.ndarray,
    desired: jnp.ndarray,
    *,
    clip: float = 1e-12,
    do_average: bool = True,
) -> jnp.ndarray:
    """KL divergence KL(desired ‖ result) averaged over batch and time.

    Uses ``jnp.where`` to handle ``0 * log(0) = 0`` correctly: entries where
    ``desired == 0`` contribute zero to the sum, and gradients w.r.t. ``result``
    are zeroed there too.

    Args:
        result (jnp.ndarray): Predicted distributions of shape (B, T, K).
        desired (jnp.ndarray): Target distributions of shape (B, T, K).
            One-hot vectors are valid; KL(one_hot(e) ‖ q) = -log q[e].
        clip (float): Small constant clipping ``result`` away from zero before
            taking the log. Defaults to 1e-12.
        do_average (bool): If True, return the scalar mean over B and T.
            If False, return per-timestep values of shape (B, T). Defaults to True.

    Returns:
        jnp.ndarray: Scalar mean KL if ``do_average``, else shape (B, T).
    """
    q = jnp.clip(result, clip, 1.0)
    # Guard against 0*log(0): where desired==0 the full term is 0.
    # Both branches are finite so there are no NaN gradients.
    log_ratio = jnp.where(desired > 0, jnp.log(desired + clip) - jnp.log(q), 0.0)
    kl = jnp.sum(desired * log_ratio, axis=-1)
    return jnp.mean(kl) if do_average else kl


def hilbert_distance(
    result: jnp.ndarray,
    desired: jnp.ndarray,
    *,
    clip: float = 1e-16,
    do_average: bool = True,
) -> jnp.ndarray:
    """Hilbert projective metric between ``result`` and ``desired``.

    Defined as ``max_k log(result_k / desired_k) - min_k log(result_k / desired_k)``.

    Args:
        result (jnp.ndarray): Predicted distributions of shape (B, T, K).
        desired (jnp.ndarray): Target distributions of shape (B, T, K).
        clip (float): Small constant clipping both arrays before taking the log.
            Defaults to 1e-16.
        do_average (bool): If True, return the scalar mean. Defaults to True.

    Returns:
        jnp.ndarray: Scalar mean Hilbert distance if ``do_average``, else shape (B, T).
    """
    q = jnp.clip(result, clip, None)
    p = jnp.clip(desired, clip, None)
    # Zero out entries where both result and desired are zero: 0/0 → ratio 1, log ratio 0.
    # Single-zero entries (one positive, one zero) retain the clipped log ratio, which
    # correctly drives the Hilbert metric toward infinity.
    both_zero = (result == 0) & (desired == 0)
    log_ratio = jnp.where(both_zero, 0.0, jnp.log(q) - jnp.log(p))
    hilbert = jnp.max(log_ratio, axis=-1) - jnp.min(log_ratio, axis=-1)
    return jnp.mean(hilbert) if do_average else hilbert


def one_norm(
    result: jnp.ndarray,
    desired: jnp.ndarray,
    *,
    do_average: bool = True,
) -> jnp.ndarray:
    """L1 norm between ``result`` and ``desired`` distributions.

    Both arrays are L1-normalised before comparison.

    Args:
        result (jnp.ndarray): Predicted distributions of shape (B, T, K).
        desired (jnp.ndarray): Target distributions of shape (B, T, K).
        do_average (bool): If True, return the scalar mean. Defaults to True.

    Returns:
        jnp.ndarray: Scalar mean L1 distance if ``do_average``, else shape (B, T).
    """
    p = result / jnp.sum(result, axis=-1, keepdims=True)
    q = desired / jnp.sum(desired, axis=-1, keepdims=True)
    nrm = jnp.sum(jnp.abs(p - q), axis=-1)
    return jnp.mean(nrm) if do_average else nrm


# ------------------------------------------------------------------
# Map from LossType to function
# ------------------------------------------------------------------

LOSS_MAP: dict[LossType, Callable] = {
    LossType.KL: kl_divergence,
    LossType.HILBERT: hilbert_distance,
    LossType.ONE_NORM: one_norm,
}
