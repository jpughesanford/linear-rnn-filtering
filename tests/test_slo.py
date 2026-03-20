"""Tests for StochasticLinearArray and StochasticLinearFunction.

Three algebraic identities must hold for every operator, regardless of class:

  (I)   apply(e_i, ...) == column_at(i, ...)           (apply on a one-hot == column lookup)
  (II)  dot(row_at(k), x) == apply(x)[k]              (lag-1: row contracts against input)
  (III) dot(row_at(k) @ x2, x1) == apply(x1, x2)[k]  (lag-2: bilinear form via row slice)

StochasticLinearFunction must give the same numerical results as StochasticLinearArray
for the same underlying operator — the two implementations are alternate representations
of the same mathematical object.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rnn_filtering.hmm.slo import (
    StochasticLinearArray,
    StochasticLinearFunction,
)

# ---------------------------------------------------------------------------
# Toy operators
# ---------------------------------------------------------------------------

# Lag-1: 3-symbol emission over 2 states. Deliberately non-square and
# non-uniform so that basis-vector tests are not trivially interchangeable.
LAG1 = np.array(
    [
        [0.5, 0.2],
        [0.3, 0.5],
        [0.2, 0.3],
    ],
    dtype=np.float32,
)

# Lag-2: 3-symbol deterministic edge emission over 2 states.
#   E[y, x_t, x_{t-1}] — each (x_t, x_{t-1}) pair deterministically emits one symbol:
#     (0, 0) -> 0,  (1, 0) -> 1,  (0, 1) -> 2,  (1, 1) -> 0
# This makes row_at and column_at trivially verifiable by hand while still
# exercising the full bilinear contraction logic.
LAG2 = np.zeros((3, 2, 2), dtype=np.float32)
LAG2[0, 0, 0] = 1.0
LAG2[1, 1, 0] = 1.0
LAG2[2, 0, 1] = 1.0
LAG2[0, 1, 1] = 1.0

# Mixed probability vectors used to test non-basis-vector inputs.
X1 = np.array([0.6, 0.4], dtype=np.float32)
X2 = np.array([0.3, 0.7], dtype=np.float32)

# Precomputed expected result for apply(X1, X2) under LAG2:
#   k=0: E[0,0,0]*0.6*0.3 + E[0,1,1]*0.4*0.7 = 0.18 + 0.28 = 0.46
#   k=1: E[1,1,0]*0.4*0.3                     = 0.12
#   k=2: E[2,0,1]*0.6*0.7                     = 0.42
EXPECTED_LAG2_APPLY = np.array([0.46, 0.12, 0.42], dtype=np.float32)


def lag1_func(x):
    return jnp.array(LAG1) @ x


def lag2_func(x1, x2):
    """Linear (in x1) bilinear contraction matching the LAG2 array."""
    return jnp.einsum("kij,i,j->k", jnp.array(LAG2), x1, x2)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sla_lag1():
    return StochasticLinearArray(LAG1, in_dimension=2, out_dimension=3, lag_dimension=1)


@pytest.fixture
def sla_lag2():
    return StochasticLinearArray(LAG2, in_dimension=2, out_dimension=3, lag_dimension=2)


@pytest.fixture
def slf_lag1():
    return StochasticLinearFunction(lag1_func, in_dimension=2, out_dimension=3, lag_dimension=1)


@pytest.fixture
def slf_lag2():
    return StochasticLinearFunction(lag2_func, in_dimension=2, out_dimension=3, lag_dimension=2)


# ---------------------------------------------------------------------------
# TestStochasticLinearArray
# ---------------------------------------------------------------------------


class TestStochasticLinearArray:
    def test_apply_lag1_agrees_with_matmul(self, sla_lag1):
        """apply on a mixed probability vector must equal LAG1 @ x."""
        x = np.array([0.4, 0.6], dtype=np.float32)
        assert np.allclose(sla_lag1.apply(x), LAG1 @ x, atol=1e-6)

    def test_apply_lag1_on_basis_agrees_with_column_at(self, sla_lag1):
        """Identity (I): apply(e_i) == column_at(i) for all i."""
        for i in range(2):
            e_i = np.zeros(2, dtype=np.float32)
            e_i[i] = 1.0
            assert np.allclose(sla_lag1.apply(e_i), sla_lag1.column_at(i), atol=1e-6)

    def test_row_at_lag1_extracts_correct_row(self, sla_lag1):
        """row_at(k) must equal the k-th row of LAG1."""
        for k in range(3):
            assert np.allclose(sla_lag1.row_at(k), LAG1[k, :], atol=1e-6)

    def test_row_at_lag1_inner_product_agrees_with_apply(self, sla_lag1):
        """Identity (II): dot(row_at(k), x) == apply(x)[k] for all k and a mixed x."""
        for k in range(3):
            assert np.allclose(
                jnp.dot(sla_lag1.row_at(k), X1),
                sla_lag1.apply(X1)[k],
                atol=1e-6,
            )

    def test_column_at_lag1_extracts_correct_column(self, sla_lag1):
        """column_at(i) must equal the i-th column of LAG1."""
        for i in range(2):
            assert np.allclose(sla_lag1.column_at(i), LAG1[:, i], atol=1e-6)

    def test_apply_lag2_bilinear_contraction(self, sla_lag2):
        """apply(x1, x2) must equal the hand-computed bilinear sum over LAG2."""
        result = sla_lag2.apply(X1, X2)
        assert np.allclose(result, EXPECTED_LAG2_APPLY, atol=1e-6)

    def test_apply_lag2_on_basis_pairs_agrees_with_column_at(self, sla_lag2):
        """Identity (I) for lag-2: apply(e_i, e_j) == column_at(i, j) for all i, j."""
        for i in range(2):
            for j in range(2):
                e_i = np.zeros(2, dtype=np.float32)
                e_i[i] = 1.0
                e_j = np.zeros(2, dtype=np.float32)
                e_j[j] = 1.0
                assert np.allclose(sla_lag2.apply(e_i, e_j), sla_lag2.column_at(i, j), atol=1e-6)

    def test_row_at_lag2_shape_and_bilinear_consistency(self, sla_lag2):
        """Identity (III): for all k, dot(row_at(k) @ x2, x1) == apply(x1, x2)[k]."""
        assert sla_lag2.row_at(0).shape == (2, 2)
        for k in range(3):
            via_row = jnp.dot(sla_lag2.row_at(k) @ X2, X1)
            via_apply = sla_lag2.apply(X1, X2)[k]
            assert np.allclose(via_row, via_apply, atol=1e-6)

    def test_apply_lag2_contraction_order_is_not_symmetric(self, sla_lag2):
        """apply(x1, x2) != apply(x2, x1) for an asymmetric operator and x1 != x2.
        This guards against silent argument transposition.
        """
        result_12 = sla_lag2.apply(X1, X2)
        result_21 = sla_lag2.apply(X2, X1)
        assert not np.allclose(result_12, result_21, atol=1e-3)

    def test_validate_rejects_wrong_shape(self):
        bad = np.ones((4, 2), dtype=np.float32) / 4  # (4, 2) but out_dimension=3
        with pytest.raises(ValueError, match="wrong shape"):
            StochasticLinearArray(bad, in_dimension=2, out_dimension=3, lag_dimension=1)

    def test_validate_rejects_non_column_stochastic(self):
        bad = LAG1.copy()
        bad[0, 0] += 0.5  # column 0 now sums to 1.5
        with pytest.raises(ValueError, match="column-stochastic"):
            StochasticLinearArray(bad, in_dimension=2, out_dimension=3, lag_dimension=1)


# ---------------------------------------------------------------------------
# TestStochasticLinearFunction
# ---------------------------------------------------------------------------


class TestStochasticLinearFunction:
    def test_transpose_fn_constructed_for_lag1(self, slf_lag1):
        assert slf_lag1._transpose_fn is not None

    def test_transpose_fn_is_none_for_lag2(self, slf_lag2):
        """_transpose_fn must not be built for lag-2; row_at uses nested vmap instead."""
        assert slf_lag2._transpose_fn is None

    def test_apply_lag1_agrees_with_array(self, sla_lag1, slf_lag1):
        assert np.allclose(slf_lag1.apply(X1), sla_lag1.apply(X1), atol=1e-6)

    def test_row_at_lag1_agrees_with_array(self, sla_lag1, slf_lag1):
        """row_at uses jax.linear_transpose for lag-1; must match the direct array lookup."""
        for k in range(3):
            assert np.allclose(slf_lag1.row_at(k), sla_lag1.row_at(k), atol=1e-5)

    def test_column_at_lag1_agrees_with_array(self, sla_lag1, slf_lag1):
        for i in range(2):
            assert np.allclose(slf_lag1.column_at(i), sla_lag1.column_at(i), atol=1e-6)

    def test_apply_lag2_agrees_with_array(self, sla_lag2, slf_lag2):
        assert np.allclose(slf_lag2.apply(X1, X2), sla_lag2.apply(X1, X2), atol=1e-6)

    def test_row_at_lag2_agrees_with_array(self, sla_lag2, slf_lag2):
        """row_at for lag-2 uses nested vmap; must produce the same (in_dim, in_dim) slice."""
        for k in range(3):
            assert np.allclose(slf_lag2.row_at(k), sla_lag2.row_at(k), atol=1e-5)

    def test_column_at_lag2_agrees_with_array(self, sla_lag2, slf_lag2):
        for i in range(2):
            for j in range(2):
                assert np.allclose(slf_lag2.column_at(i, j), sla_lag2.column_at(i, j), atol=1e-6)

    def test_validate_rejects_nonstochastic_function(self):
        """A function whose output does not sum to 1 must be rejected."""

        def bad(x):
            return x * 2.0  # sums to 2

        with pytest.raises(ValueError):
            StochasticLinearFunction(bad, in_dimension=2, out_dimension=2, lag_dimension=1)

    def test_validate_rejects_nonlinear_function(self):
        """softmax is not linear: softmax(x + y) != softmax(x) + softmax(y)."""

        def bad(x):
            return jax.nn.softmax(x**2)

        with pytest.raises(ValueError):
            StochasticLinearFunction(bad, in_dimension=3, out_dimension=3, lag_dimension=1)

    def test_validate_rejects_wrong_output_shape(self):
        """A function with the wrong output dimension must be rejected."""

        def bad(_x):
            return jnp.ones(5) / 5  # out_dimension is 3

        with pytest.raises(ValueError):
            StochasticLinearFunction(bad, in_dimension=2, out_dimension=3, lag_dimension=1)
