"""Implements a stochastic linear class for HMMs."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import reduce

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

__all__ = ["StochasticLinearFunction", "StochasticLinearArray", "AbstractStochasticLinearOperator"]

AllowedInputTypes = Callable[..., ArrayLike] | ArrayLike


class AbstractStochasticLinearOperator(ABC):
    """Base class for stochastic linear operators used in HMM emission and transfer models.

    An operator of lag `k` maps `k` input probability vectors (each of shape `(in_dim,)`) to a
    single output probability vector of shape `(out_dim,)`. The underlying tensor has shape::

        (out_dim,) + (in_dim,) * lag_dim

    The first axis is the **output** axis and the remaining `lag_dim` axes are **input** axes.
    The operator is column-stochastic: every slice along axis 0 sums to 1.

    Lag-1 operators (standard matrices) cover node-emitting HMMs. Lag-2 operators cover
    edge-emitting HMMs, where emission depends on both the current and previous latent state,
    e.g. ``E[y, x_t, x_{t-1}] = P(y_t = y | x_t, x_{t-1})``.

    Three access patterns are provided, each optimised for a different use case:

    - ``apply(*args)``:  full forward pass on probability vectors; used during both sampling
      (transfer) and filtering (transfer).
    - ``row_at(index)``:  fixes the output index; used during forward filtering to extract the
      likelihood of a specific observation across all latent configurations.
    - ``column_at(*indices)``: fixes all input indices; used during sampling to extract the output
      distribution conditioned on a concrete latent state (or pair of states for lag-2).

    Attributes:
        operator: The underlying stochastic operator, either an ndarray or a callable.
        in_dimension: Size of each input probability vector.
        out_dimension: Size of the output probability vector.
        lag_dimension: Number of input vectors the operator acts on.
    """

    def __init__(
        self, user_input: AllowedInputTypes, in_dimension: int, out_dimension: int, lag_dimension: int
    ) -> None:
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.lag_dimension = lag_dimension
        self.validate(user_input, in_dimension, out_dimension, lag_dimension)
        self.operator: AllowedInputTypes = user_input

    @abstractmethod
    def validate(
        self, user_input: AllowedInputTypes, in_dimension: int, out_dimension: int, lag_dimension: int
    ) -> None:
        """Checks whether the user input is valid. Throws ValueError if the input is invalid."""

    @abstractmethod
    def apply(self, *args: ArrayLike) -> jax.Array:
        """Apply the operator to `lag_dim` probability vectors, returning a probability vector.

        Contracts the operator tensor against each input vector in order, collapsing all input
        axes. For lag-1 this is a matrix-vector product ``A @ x``; for lag-2 it is the bilinear
        contraction ``A @ x2 @ x1`` (last axis contracted first). Inputs should be genuine
        probability vectors; use ``column_at`` instead when the input is a one-hot (concrete state
        index), as that avoids a full matrix-vector multiply.

        Args:
            *args: Exactly `lag_dim` probability vectors, each of shape `(in_dim,)`.

        Returns:
            Output probability vector of shape `(out_dim,)`.
        """

    @abstractmethod
    def row_at(self, index: int | jax.Array) -> jax.Array:
        """Fix the output index and return the remaining input tensor.

        For a matrix (lag-1), this is literally a row: ``A[index, :]``, shape ``(in_dim,)``.
        For a lag-2 operator, it returns a matrix: ``A[index, :, :]``, shape
        ``(in_dim, in_dim)``. In general the return shape is ``(in_dim,) * lag_dim``.

        This is the primary access pattern for **forward filtering**: ``emission.row_at(y_t)``
        gives the likelihood of observing ``y_t`` for every latent configuration.

        Array implementation is a single O(1) gather. Functional implementation requires
        ``jax.linear_transpose``, which is expensive and only defined for lag-1 operators.

        Args:
            index: Index along the output axis (e.g. an observed emission symbol).

        Returns:
            Tensor of shape ``(in_dim,) * lag_dim``.
        """

    @abstractmethod
    def column_at(self, *indices: int | jax.Array) -> jax.Array:
        """Fix all input indices and return the output distribution.

        Takes exactly ``lag_dim`` integer indices, one per input axis, and returns the
        corresponding column of the operator: ``A[:, i1, i2, ...]``, always of shape
        ``(out_dim,)``.

        This is the primary access pattern for **sampling**: ``transfer.column_at(x)`` gives the
        next-state distribution from state ``x``; ``emission.column_at(x)`` gives the emission
        distribution at state ``x`` (lag-1), or ``emission.column_at(x_t, x_{t-1})`` for lag-2.

        Array implementation is a single O(1) gather — substantially faster than calling
        ``apply`` with a one-hot vector, which performs a full O(n²) matrix-vector multiply.
        Functional implementation falls back to ``apply`` on basis vectors and has the same
        cost as ``apply``.

        Args:
            *indices: Exactly `lag_dim` integer state indices, one per input axis.

        Returns:
            Output probability vector of shape `(out_dim,)`.
        """


class StochasticLinearArray(AbstractStochasticLinearOperator):
    """A stochastic linear operator backed by an ndarray.

    The array must have shape ``(out_dim,) + (in_dim,) * lag_dim`` and be column-stochastic
    (every slice along axis 0 sums to 1). ``apply`` performs an O(n²) matrix-vector product
    per lag level. ``row_at`` and ``column_at`` are O(1) direct array lookups and should be
    preferred over ``apply`` whenever the input is a concrete index rather than a probability
    vector.
    """

    def __init__(self, user_input, in_dimension: int, out_dimension: int, lag_dimension: int) -> None:
        super().__init__(user_input, in_dimension, out_dimension, lag_dimension)
        self.operator = jnp.asarray(self.operator)

    def validate(self, array: AllowedInputTypes, in_dimension: int, out_dimension: int, lag_dimension: int) -> None:
        """Check that the array has the correct shape and is column-stochastic."""
        expected_shape = (out_dimension,) + (in_dimension,) * lag_dimension
        if array.shape != expected_shape:
            raise ValueError(f"Stochastic array has wrong shape. Expected {expected_shape}, got {array.shape}.")
        col_sums = array.sum(axis=0)
        if not np.allclose(col_sums, 1, atol=1e-6):
            raise ValueError(
                f"Stochastic array is not column-stochastic. "
                f"Each slice along axis 0 must sum to 1. "
                f"Worst deviation: {np.abs(col_sums - 1).max():.2e}."
            )

    def apply(self, *args: ArrayLike) -> jax.Array:
        """Successive dot-product contraction from the last input axis inward."""
        return reduce(lambda acc, v: jnp.dot(acc, v), reversed(args), self.operator)

    def row_at(self, index: int | jax.Array) -> jax.Array:
        """Direct index along the output axis: ``A[index, ...]``."""
        return self.operator[index, ...]

    def column_at(self, *indices: int | jax.Array) -> jax.Array:
        """Direct index along all input axes: ``A[:, i1, i2, ...]``."""
        return self.operator[(slice(None),) + indices]


class StochasticLinearFunction(AbstractStochasticLinearOperator):
    """A stochastic linear operator backed by a callable.

    ``apply`` is a direct function call. ``row_at`` uses a precomputed ``jax.linear_transpose``
    and is only valid for lag-1 operators; it is more expensive than the array equivalent.
    ``column_at`` evaluates the function on basis vectors and has the same cost as ``apply``.
    """

    def __init__(
        self, user_input: AllowedInputTypes, in_dimension: int, out_dimension: int, lag_dimension: int
    ) -> None:
        super().__init__(user_input, in_dimension, out_dimension, lag_dimension)
        if lag_dimension == 1:
            dummy_input = jnp.zeros(self.in_dimension)
            self._transpose_fn = jax.linear_transpose(self.operator, dummy_input)
        else:
            self._transpose_fn = None

    def validate(
        self, functional: AllowedInputTypes, in_dimension: int, out_dimension: int, lag_dimension: int
    ) -> None:
        """Check that the callable has the correct input/output shapes, is stochastic, and is linear in its first
        argument."""
        key = jax.random.PRNGKey(np.random.randint(0, 2**31))
        keys = jax.random.split(key, lag_dimension + 1)
        xs = [
            jax.nn.softmax(jax.random.normal(shape=(in_dimension,), dtype=jnp.float32, key=keys[k]), axis=-1)
            for k in range(lag_dimension)
        ]
        y0 = jax.nn.softmax(
            jax.random.normal(shape=(in_dimension,), dtype=jnp.float32, key=keys[lag_dimension]), axis=-1
        )
        try:
            fx = functional(*xs)
            if not fx.shape == (out_dimension,):
                raise ValueError(
                    f"Stochastic functional has wrong output shape. Expected {(out_dimension,)}, got {fx.shape}."
                )
            if not np.allclose(fx.sum(), 1, atol=1e-6):
                raise ValueError(
                    f"Stochastic functional does not preserve stochasticity. "
                    f"Deviation of output sum from 1: {np.abs(fx.sum() - 1):.2e}."
                )
            fy = functional(*([y0] + xs[1:]))
            if not np.allclose(fx + fy, functional(*([xs[0] + y0] + xs[1:])), atol=1e-6):
                raise ValueError("Stochastic functional is not linear in its first argument.")
        except Exception as err:
            raise ValueError("Stochastic functional failed to evaluate.") from err

    def apply(self, *args: ArrayLike) -> jax.Array:
        """Direct call to the underlying function."""
        return self.operator(*args)

    def row_at(self, index: int | jax.Array) -> jax.Array:
        """Row lookup. Lag-1: uses precomputed ``jax.linear_transpose``. Lag-2: materialises via nested vmap over
        ``column_at``."""
        if self.lag_dimension == 1:
            e_i = jnp.zeros(self.out_dimension).at[index].set(1.0)
            (ith_row_at,) = self._transpose_fn(e_i)
            return ith_row_at
        elif self.lag_dimension == 2:
            all_indices = jnp.arange(self.in_dimension)

            def row_i(i):
                def row_j(j):
                    return self.column_at(i, j)[index]

                return jax.vmap(row_j)(all_indices)

            return jax.vmap(row_i)(all_indices)
        else:
            raise NotImplementedError("StochasticLinearFunction.row_at is only implemented for lag_dimension <= 2.")

    def column_at(self, *indices: int | jax.Array) -> jax.Array:
        """Evaluate the function on standard basis vectors, one per input axis."""
        one_hots = [jnp.zeros(self.in_dimension).at[idx].set(1.0) for idx in indices]
        return self.operator(*one_hots)
