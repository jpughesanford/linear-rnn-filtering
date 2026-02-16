"""Matrix utilities for stable linear parameterizations."""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

DTYPE = jnp.float64


def cayley(A):
    """Cayley transform: maps a square matrix to an orthogonal matrix.

    Computes (I - A)^{-T} (I + A)^T.
    """
    eye = jnp.eye(A.shape[0], dtype=DTYPE)
    return jnp.linalg.solve((eye + A).T, (eye - A).T).T


def params_to_stable_matrix(A1, A2, epsilon=1e-6):
    """Map unconstrained parameters (A1, A2) to a matrix with spectral radius <= 1.

    Uses a Cayley-transform parameterization:
      1. A1 A1^T + eps I  gives a symmetric positive-definite component.
      2. A2 (lower-triangular entries) gives a skew-symmetric component.
      3. The Cayley transform of their sum is guaranteed to have all
         eigenvalues on or inside the unit circle.

    Parameters
    ----------
    A1 : array, shape (n, n)
        Free parameters for the positive-definite part.
    A2 : array, shape (n*(n-1)/2,)
        Free parameters for the skew-symmetric part.
    epsilon : float
        Regularization for positive definiteness.

    Returns
    -------
    A_stable : array, shape (n, n)
        Matrix with spectral radius <= 1.
    """
    n = A1.shape[0]
    eye = jnp.eye(n, dtype=DTYPE)
    A_pd = A1 @ A1.T + epsilon * eye

    A_lower = jnp.zeros((n, n))
    idx = jnp.tril_indices(n, k=-1)
    A_lower = A_lower.at[idx].set(A2)
    A_skew = 0.5 * (A_lower - A_lower.T)

    return cayley(A_pd + A_skew)


def stable_matrix_to_params(A_stable, epsilon=1e-6):
    """Invert :func:`params_to_stable_matrix`: recover (A1, A2) from a stable matrix.

    Parameters
    ----------
    A_stable : array, shape (n, n)
        Matrix with spectral radius <= 1.
    epsilon : float
        Same regularization used in the forward transform.

    Returns
    -------
    A1 : array, shape (n, n)
    A2 : array, shape (n*(n-1)/2,)
    """
    n = A_stable.shape[0]
    eye = jnp.eye(n, dtype=DTYPE)
    A_sum = cayley(A_stable)

    A_pd = 0.5 * (A_sum + A_sum.T)
    SPD = A_pd - epsilon * eye
    vals, vecs = jnp.linalg.eigh(SPD)
    vals = jnp.maximum(vals, 1e-9)
    SPD = (vecs * vals) @ vecs.T
    A1 = jnp.linalg.cholesky(SPD)

    A_skew = 0.5 * (A_sum - A_sum.T)
    idx = jnp.tril_indices(n, k=-1)
    A2 = A_skew[idx]

    return A1, A2
