"""Tests for AbstractHMM, NodeEmittingHMM, and HMMFactory."""

import jax.numpy as jnp
import numpy as np
import pytest

from rnn_filtering.hmm.factory import HMMFactory
from rnn_filtering.hmm.models import AbstractHMM, EdgeEmittingHMM, NodeEmittingHMM, _anderson_iterate
from rnn_filtering.hmm.slo import StochasticLinearArray, StochasticLinearFunction

# ---------------------------------------------------------------------------
# Shared fixtures and constants
# ---------------------------------------------------------------------------

CASINO_T = np.array([[0.95, 0.10], [0.05, 0.90]])
CASINO_E = np.array(
    [
        [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
        [1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 2],
    ]
).T

# Direction-sensitive 2-state, 3-symbol edge-emitting model.
# E[y, j, k] = P(y | x_t=j, x_{t-1}=k): emission reflects the current state and direction.
#   self-loop at state 0 (j=0, k=0) -> mostly y=0
#   self-loop at state 1 (j=1, k=1) -> mostly y=2
#   transition 1->0    (j=0, k=1)   -> y=0 and y=1 roughly equally
#   transition 0->1    (j=1, k=0)   -> y=1 and y=2 roughly equally
# T has stationary pi=[0.6, 0.4], giving stationary emission [0.448, 0.276, 0.276].
SIMPLE_EDGE_TOY_MODEL_T = np.array([[0.8, 0.3], [0.2, 0.7]])
SIMPLE_EDGE_TOY_MODEL_E = np.array(
    [
        [[0.7, 0.4], [0.3, 0.1]],  # y=0
        [[0.2, 0.4], [0.4, 0.3]],  # y=1
        [[0.1, 0.2], [0.3, 0.6]],  # y=2
    ]
)


@pytest.fixture
def casino_arr():
    """NodeEmittingHMM constructed from matrices."""
    return NodeEmittingHMM(2, 6, CASINO_T, CASINO_E)


@pytest.fixture
def casino_fun():
    """NodeEmittingHMM constructed from callable operators wrapping the same matrices."""

    def transition_fn(state):
        return CASINO_T @ state

    def emission_fn(state):
        return CASINO_E @ state

    return NodeEmittingHMM(2, 6, transition_fn, emission_fn)


@pytest.fixture
def dyck_simple():
    """Simplest non-trivial DyckHMM: depth=1, width=2.

    Nodes: {1, 2, 3}  (root=1, leaves=2 and 3)
    Directed edge states (4 total):
      0 = down-edge arriving at node 2
      1 = down-edge arriving at node 3
      2 = up-edge departing node 2
      3 = up-edge departing node 3
    Internal emission indices (4 = 2*width):
      0 = open type 1,  1 = open type 2,  2 = close type 1,  3 = close type 2
    Decoded emissions (n/-n notation):
      0 -> +1,  1 -> +2,  2 -> -1,  3 -> -2
    """
    return HMMFactory.dyck_arr(depth=1, width=2, temperature=0)


# ---------------------------------------------------------------------------
# Test Generic HMM Functionality
# ---------------------------------------------------------------------------


class TestGenericHMM:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            AbstractHMM()

    def test_arr_constructor_sets_dimensions_arr(self, casino_arr):
        assert casino_arr.latent_dim == 2
        assert casino_arr.emission_dim == 6
        assert isinstance(casino_arr.transfer_operator, StochasticLinearArray)

    def test_constructor_dimension_and_type(self, casino_fun):
        assert casino_fun.latent_dim == 2
        assert casino_fun.emission_dim == 6
        assert isinstance(casino_fun.transfer_operator, StochasticLinearFunction)

    @pytest.mark.parametrize("name", ["casino_arr", "casino_fun"])
    def test_arr_and_fun_apply_same_transfer_update(self, request, name):
        casino = request.getfixturevalue(name)
        for i in range(casino.latent_dim):
            e_i = np.zeros(casino.latent_dim)
            e_i[i] = 1.0
            assert np.allclose(casino.transfer_operator.apply(e_i), CASINO_T[:, i], atol=1e-6)

    @pytest.mark.parametrize("name", ["casino_arr", "casino_fun"])
    def test_arr_and_fun_apply_same_emission_update(self, request, name):
        casino = request.getfixturevalue(name)
        for i in range(casino.latent_dim):
            e_i = np.zeros(casino.latent_dim)
            e_i[i] = 1.0
            assert np.allclose(casino.emission_operator.apply(e_i), CASINO_E[:, i], atol=1e-6)

    @pytest.mark.parametrize(
        "hmm_class, emission_dim, emission_matrix",
        [
            (NodeEmittingHMM, 6, CASINO_E),
            (EdgeEmittingHMM, 3, SIMPLE_EDGE_TOY_MODEL_E),
        ],
    )
    def test_constructor_rejects_wrong_transfer_shape(self, hmm_class, emission_dim, emission_matrix):
        bad_T = np.random.rand(3, 2)
        bad_T /= bad_T.sum(axis=0)

        with pytest.raises(ValueError, match="shape|dimension"):
            hmm_class(2, emission_dim, bad_T, emission_matrix)

    @pytest.mark.parametrize(
        "hmm_class, emission_dim, transfer_matrix",
        [
            (NodeEmittingHMM, 6, CASINO_T),
            (EdgeEmittingHMM, 3, SIMPLE_EDGE_TOY_MODEL_T),
        ],
    )
    def test_constructor_rejects_wrong_emission_shape(self, hmm_class, emission_dim, transfer_matrix):
        bad_E = np.random.rand(3, 10)
        bad_E /= bad_E.sum(axis=0)

        with pytest.raises(ValueError, match="shape|dimension"):
            hmm_class(2, emission_dim, transfer_matrix, bad_E)

    @pytest.mark.parametrize(
        "hmm_class, emission_dim, transfer_matrix, emission_matrix",
        [
            (NodeEmittingHMM, 6, CASINO_T, CASINO_E),
            (EdgeEmittingHMM, 3, SIMPLE_EDGE_TOY_MODEL_T, SIMPLE_EDGE_TOY_MODEL_E),
        ],
    )
    def test_constructor_rejects_nonstochastic_transfer(
        self, hmm_class, emission_dim, transfer_matrix, emission_matrix
    ):
        bad_T = transfer_matrix.copy()
        bad_T[0, 0] += 0.5
        with pytest.raises(ValueError):
            hmm_class(2, emission_dim, bad_T, emission_matrix)

    @pytest.mark.parametrize(
        "hmm_class, emission_dim, transfer_matrix, emission_matrix",
        [
            (NodeEmittingHMM, 6, CASINO_T, CASINO_E),
            (EdgeEmittingHMM, 3, SIMPLE_EDGE_TOY_MODEL_T, SIMPLE_EDGE_TOY_MODEL_E),
        ],
    )
    def test_constructor_rejects_nonstochastic_emission(
        self, hmm_class, emission_dim, transfer_matrix, emission_matrix
    ):
        bad_E = emission_matrix.copy()
        bad_E[0, 0] += 0.5
        with pytest.raises(ValueError):
            hmm_class(2, emission_dim, transfer_matrix, bad_E)

    @pytest.mark.parametrize("name", ["casino_arr", "casino_fun"])
    def test_stationary_density_sums_to_one(self, request, name):
        casino = request.getfixturevalue(name)
        assert np.isclose(casino.latent_stationary_density.sum(), 1.0)

    @pytest.mark.parametrize(
        "hmm_class, emission_dim, transfer_matrix, emission_matrix",
        [
            (NodeEmittingHMM, 6, CASINO_T, CASINO_E),
            (EdgeEmittingHMM, 3, SIMPLE_EDGE_TOY_MODEL_T, SIMPLE_EDGE_TOY_MODEL_E),
        ],
    )
    def test_stationary_density_set(self, hmm_class, emission_dim, transfer_matrix, emission_matrix):
        density = [0.5, 0.5]
        hmm = hmm_class(2, emission_dim, transfer_matrix, emission_matrix, latent_stationary_density=density)
        assert np.allclose(hmm.latent_stationary_density, jnp.asarray(density))

    @pytest.mark.parametrize(
        "hmm_class, emission_dim, transfer_matrix, emission_matrix",
        [
            (NodeEmittingHMM, 6, CASINO_T, CASINO_E),
            (EdgeEmittingHMM, 3, SIMPLE_EDGE_TOY_MODEL_T, SIMPLE_EDGE_TOY_MODEL_E),
        ],
    )
    @pytest.mark.parametrize(
        "bad_density, expected_match",
        [
            ([0.6, 0.6], "sum to one"),  # Sum != 1.0
            ([0.5 + 1j, 0.5 - 1j], "imaginary part"),  # Complex numbers
            ([-0.1, 1.1], "negative entries"),  # Negative values
        ],
    )
    def test_stationary_density_warnings(
        self, hmm_class, emission_dim, transfer_matrix, emission_matrix, bad_density, expected_match
    ):
        with pytest.warns(UserWarning, match=expected_match):
            hmm_class(2, emission_dim, transfer_matrix, emission_matrix, latent_stationary_density=bad_density)

    @pytest.mark.parametrize(
        "hmm_class, emission_dim, transfer_matrix, emission_matrix",
        [
            (NodeEmittingHMM, 6, CASINO_T, CASINO_E),
            (EdgeEmittingHMM, 3, SIMPLE_EDGE_TOY_MODEL_T, SIMPLE_EDGE_TOY_MODEL_E),
        ],
    )
    def test_stationary_density_is_stationary(self, hmm_class, emission_dim, transfer_matrix, emission_matrix):
        hmm = hmm_class(2, emission_dim, transfer_matrix, emission_matrix)
        p = hmm.latent_stationary_density
        q = hmm.transfer_operator.apply(p)
        assert np.allclose(p, q)

    @pytest.mark.parametrize("name", ["casino_arr", "casino_fun"])
    def test_sample_output_format(self, request, name):
        casino = request.getfixturevalue(name)
        latent, emission = casino.sample(batch_size=10, time_steps=50)
        assert latent.shape == (10, 50)
        assert emission.shape == (10, 50)
        assert np.all(latent >= 0) and np.all(latent < casino.latent_dim)
        assert np.all(emission >= 0) and np.all(emission < casino.emission_dim)

    @pytest.mark.parametrize("name", ["casino_arr", "casino_fun"])
    def test_posterior_output_format(self, request, name):
        casino = request.getfixturevalue(name)
        _, emissions = casino.sample(batch_size=4, time_steps=30)
        lp, ntp = casino.compute_posterior(emissions)
        assert lp.shape == (4, 30, casino.latent_dim)
        assert ntp.shape == (4, 30, casino.emission_dim)
        assert np.allclose(lp.sum(axis=-1), 1.0, atol=1e-6)
        assert np.allclose(ntp.sum(axis=-1), 1.0, atol=1e-6)


class TestNodeEmittingDynamics:
    """Tests that the fixed points of the dishonest casino (for emission y_t=constant) are computed correctly.

    for y_t in (1,5), the latent posterior fixed point is {fair = 0.932915, dishonest = 0.0670854}
    for y_t = 6 , the latent posterior fixed point is {fair = 0.053806, dishonest = 0.946194}

    next_emission_posterior[t] = E @ T @ state_t = P(y_{t+1} | y_{1:t}), the one-step-ahead prediction.
    At the fixed point, T @ p_fp != p_fp (p_fp is NOT the stationary of T), so E @ T @ p_fp != E @ p_fp.

    for y_t in (1,5), the one-step-ahead emission fixed point is {1..5 = 0.159532, 6 = 0.202341}
    for y_t = 6 , the one-step-ahead emission fixed point is  {1..5 = 0.109716, 6 = 0.451422}
    """

    @pytest.mark.parametrize("name", ["casino_arr", "casino_fun"])
    def test_fixed_points(self, request, name):
        casino = request.getfixturevalue(name)
        # Create observations: batch 0 sees all 1s, ..., batch 5 sees all 6s, etc.
        time_steps = 1000
        constant_observations = jnp.tile(jnp.arange(6)[:, None], (1, time_steps))
        latent_posterior, emission_posterior = casino.compute_posterior(constant_observations)
        assert jnp.allclose(latent_posterior[0:5, -1, 0], 0.932915, atol=1e-3)
        assert jnp.allclose(latent_posterior[5, -1, 1], 0.946194, atol=1e-3)
        assert jnp.allclose(
            emission_posterior[0:5, -1, :],
            np.asarray((0.159532, 0.159532, 0.159532, 0.159532, 0.159532, 0.202341)),
            atol=1e-3,
        )
        assert jnp.allclose(
            emission_posterior[5, -1, :],
            np.asarray((0.109716, 0.109716, 0.109716, 0.109716, 0.109716, 0.451422)),
            atol=1e-3,
        )


class TestEdgeEmittingDynamics:
    """Uses SIMPLE_EDGE_TOY_MODEL to test the integration methods of the edge emitting hmms.

    Analytic properties of SIMPLE_EDGE_TOY_MODEL:
        - Latent stationary density: {0.6, 0.4}
        - Emissive stationary density: {0.448, 0.276, 0.276}
        - Constant y=0 filter fixed point:
            - latent: {0.893669, 0.106331}
            - emissive: {0.574278, 0.24957, 0.176152}
        - Constant y=1 filter fixed point:
            - latent: {0.48757, 0.51243}
            - emissive: {0.399655, 0.286119, 0.314226}
        - Constant y=2 filter fixed point:
            - latent: {0.146242, 0.853758}
            - emissive: {0.252884, 0.316838, 0.430278}

    Numerical test case: for emissions = {0, 0, 1, 2, 1, 0, 1, 1, 2, 0}, starting with posterior at time 0: {.5, .5}
        - latent posterior at time 10: {0.7666, 0.2334}
        - emissive posterior at time 10: {0.5197, 0.2610, 0.2193}

    """

    @pytest.fixture
    def hmm(self):
        return EdgeEmittingHMM(2, 3, SIMPLE_EDGE_TOY_MODEL_T, SIMPLE_EDGE_TOY_MODEL_E)

    def test_stationary_density(self, hmm):
        """P(y) = sum_{j,k} T[j,k]*pi[k]*E[y,j,k] — requires the full joint edge distribution."""
        assert np.allclose(hmm.latent_stationary_density, [0.6, 0.4], atol=1e-5)
        assert np.allclose(hmm.emissive_stationary_density, [0.448, 0.276, 0.276], atol=1e-5)

    def test_filter_fixed_points(self, hmm):
        """Constant observation sequences must converge to the known analytic fixed points.

        y=0 (most likely from staying-low edge): posterior concentrates toward state 0.
        y=2 (most likely from staying-high edge): posterior concentrates toward state 1.
        """
        time_steps = 1000
        constant_obs = jnp.tile(jnp.array([0, 1, 2])[:, None], (1, time_steps))
        latent_posterior, emission_posterior = hmm.compute_posterior(constant_obs)
        assert jnp.allclose(latent_posterior[0, -1, :], jnp.array([0.8937, 0.1063]), atol=1e-3)
        assert jnp.allclose(latent_posterior[1, -1, :], jnp.array([0.4875, 0.5124]), atol=1e-3)
        assert jnp.allclose(latent_posterior[2, -1, :], jnp.array([0.1462, 0.8538]), atol=1e-3)
        assert jnp.allclose(emission_posterior[0, -1, :], jnp.array([0.5743, 0.2496, 0.1761]), atol=1e-3)
        assert jnp.allclose(emission_posterior[1, -1, :], jnp.array([0.3996, 0.2861, 0.3142]), atol=1e-3)
        assert jnp.allclose(emission_posterior[2, -1, :], jnp.array([0.2529, 0.3168, 0.4303]), atol=1e-3)

    def test_numerical_example(self, hmm):
        observation = jnp.array([0, 0, 1, 2, 1, 0, 1, 1, 2, 0])
        observation = observation.reshape((1, observation.size))
        latent_posterior, emission_posterior = hmm.compute_posterior(
            observation, initial_latent_distribution=[0.5, 0.5]
        )
        assert jnp.allclose(latent_posterior[0, -1, :], jnp.array([0.7666, 0.2334]), atol=1e-3)
        assert jnp.allclose(emission_posterior[0, -1, :], jnp.array([0.5197, 0.2610, 0.2193]), atol=1e-3)


# ---------------------------------------------------------------------------
# TestAndersonIterate
# ---------------------------------------------------------------------------


class TestAndersonIterate:
    """Tests for the _anderson_iterate fixed-point solver."""

    @pytest.mark.parametrize(
        "transfer_matrix, expected_pi",
        [
            (CASINO_T, np.array([2 / 3, 1 / 3])),
            (SIMPLE_EDGE_TOY_MODEL_T, np.array([0.6, 0.4])),
        ],
    )
    def test_finds_known_stationary(self, transfer_matrix, expected_pi):
        pi = _anderson_iterate(lambda x: transfer_matrix @ x, latent_dim=2)
        assert np.allclose(pi, expected_pi, atol=1e-6)

    @pytest.mark.parametrize("transfer_matrix", [CASINO_T, SIMPLE_EDGE_TOY_MODEL_T])
    def test_result_sums_to_one(self, transfer_matrix):
        pi = _anderson_iterate(lambda x: transfer_matrix @ x, latent_dim=2)
        assert np.isclose(pi.sum(), 1.0)

    @pytest.mark.parametrize("transfer_matrix", [CASINO_T, SIMPLE_EDGE_TOY_MODEL_T])
    def test_result_nonnegative(self, transfer_matrix):
        pi = _anderson_iterate(lambda x: transfer_matrix @ x, latent_dim=2)
        assert np.all(pi >= 0)

    @pytest.mark.parametrize("transfer_matrix", [CASINO_T, SIMPLE_EDGE_TOY_MODEL_T])
    def test_fixed_point_property(self, transfer_matrix):
        pi = _anderson_iterate(lambda x: transfer_matrix @ x, latent_dim=2)
        assert np.allclose(transfer_matrix @ np.array(pi), pi, atol=1e-8)

    def test_three_state_chain(self):
        transfer_matrix = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.2, 0.6, 0.3],
                [0.1, 0.2, 0.6],
            ]
        )
        pi = _anderson_iterate(lambda x: transfer_matrix @ x, latent_dim=3)
        assert np.allclose(transfer_matrix @ np.array(pi), pi, atol=1e-8)
        assert np.isclose(pi.sum(), 1.0)
        assert np.all(pi >= 0)

    def test_raises_runtime_error_on_nonconvergence(self):
        with pytest.raises(RuntimeError):
            _anderson_iterate(lambda x: CASINO_T @ x, latent_dim=2, max_iter=1)

    def test_raises_type_error_for_noncallable(self):
        with pytest.raises(TypeError):
            _anderson_iterate(CASINO_T, latent_dim=2)


# ---------------------------------------------------------------------------
# TestDyckDynamics
# ---------------------------------------------------------------------------


class TestDyckDynamics:
    """Tests for Dyck language HMMs on small complete trees.

    Uses depth=1, width=2: N=3 states (root=0, left=1, right=2), emission_dim=5.
    Emission alphabet: 0=epsilon, 1=open1, 2=close1, 3=open2, 4=close2.
    At temperature=0 the walk is purely local; epsilon (y=0) never appears.

    Key analytic properties used in the tests:
      - Observing y=1 forces the posterior to exactly [0, 1, 0] (left child only).
      - A valid Dyck word [1, 2, 0, 3, 4] produces identical posteriors from
        the array-backed and function-backed implementations.
      - Higher temperature strictly increases the rate of epsilon emissions.
    """

    DYCK_1x2 = np.array([[0.0, 1.0, 1.0], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    DYCK_2x2 = np.array(
        [
            [0.0, 1 / 3, 1 / 3, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 1 / 3, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1 / 3, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1 / 3, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1 / 3, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    DYCK_2x3 = np.array(
        [
            [0.0, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1 / 3, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1 / 3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1 / 3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    @pytest.mark.parametrize("depth, width", [(1, 2), (2, 2), (2, 3)])
    def test_dimensions(self, depth, width):
        """num_latent_states = (w^(d+1) - 1) / (w-1) and num_emissive_states = 2*w+1 for several (depth, width)."""
        hmm = HMMFactory.dyck_arr(depth=depth, width=width, temperature=0.1)
        expected_n = (width ** (depth + 1) - 1) // (width - 1)
        assert hmm.latent_dim == expected_n
        assert hmm.emission_dim == 2 * width + 1

    @pytest.mark.parametrize("depth, width, expectation", [(1, 2, DYCK_1x2), (2, 2, DYCK_2x2), (2, 3, DYCK_2x3)])
    def test_dyck_matrix_zero_temperature(self, depth, width, expectation):
        """At temperature=0, the HMM is purely Dyck"""
        hmm = HMMFactory.dyck_arr(depth=depth, width=width, temperature=0)
        assert np.allclose(hmm.transfer_operator.operator, expectation, atol=1e-5)

    @pytest.mark.parametrize("depth, width, expectation", [(1, 2, DYCK_1x2), (2, 2, DYCK_2x2), (2, 3, DYCK_2x3)])
    def test_dyck_matrix_nonzero_temperature(self, depth, width, expectation):
        """At temperature=0, the HMM is purely Dyck"""
        temp = 0.25
        hmm = HMMFactory.dyck_arr(depth=depth, width=width, temperature=temp)
        assert np.allclose(hmm.transfer_operator.operator, (1 - temp) * expectation + temp / hmm.latent_dim, atol=1e-5)

    @pytest.mark.parametrize("depth, width", [(1, 2), (2, 2), (2, 3)])
    def test_array_and_functional_equality(self, depth, width):
        """At temperature=0.5 teleportation causes y=0 to appear regularly."""
        n = (width ** (depth + 1) - 1) // (width - 1)
        p = np.arange(1, n + 1, dtype=float)
        p /= p.sum()
        hmm_arr = HMMFactory.dyck_arr(depth=depth, width=width, temperature=0.5)
        hmm_fun = HMMFactory.dyck_arr(depth=depth, width=width, temperature=0.5)
        assert np.allclose(hmm_arr.transfer_operator.apply(p), hmm_fun.transfer_operator.apply(p), atol=1e-8)
        assert np.allclose(hmm_arr.emission_operator.apply(p), hmm_fun.emission_operator.apply(p), atol=1e-8)

    def test_posterior_collapses_after_single_bracket(self):
        """Observing open1 (y=1) forces the posterior exactly to the left-child state.

        E[y=1, x_t, x_{t-1}] = 1 iff (x_t=1, x_{t-1}=0), so the filter collapses
        to [0, 1, 0] regardless of temperature.
        """
        for temperature in [0.0, 0.3]:
            hmm = HMMFactory.dyck_arr(depth=1, width=2, temperature=temperature)
            lp, _ = hmm.compute_posterior(jnp.array([[1]]))
            assert jnp.allclose(lp[0, -1, :], jnp.array([0.0, 1.0, 0.0]), atol=1e-6)

    def test_arr_and_fun_agree_on_posterior(self):
        """Array-backed and function-backed implementations must produce identical posteriors."""
        hmm_arr = HMMFactory.dyck_arr(depth=1, width=2, temperature=0.2)
        hmm_fun = HMMFactory.dyck_fun(depth=1, width=2, temperature=0.2)
        # Valid Dyck word: open1, close1, epsilon, open2, close2
        obs = jnp.array([[1, 2, 0, 3, 4]])
        lp_a, ep_a = hmm_arr.compute_posterior(obs)
        lp_f, ep_f = hmm_fun.compute_posterior(obs)
        assert jnp.allclose(lp_a, lp_f, atol=1e-5)
        assert jnp.allclose(ep_a, ep_f, atol=1e-5)


# ---------------------------------------------------------------------------
# TestHMMFactory
# ---------------------------------------------------------------------------


class TestHMMFactory:
    def test_dishonest_casino_returns_node_emitting_hmm(self):
        assert isinstance(HMMFactory.dishonest_casino(), NodeEmittingHMM)

    def test_dishonest_casino_dims(self):
        hmm = HMMFactory.dishonest_casino()
        assert hmm.latent_dim == 2
        assert hmm.emission_dim == 6

    def test_random_dirichlet_returns_node_emitting_hmm(self):
        assert isinstance(HMMFactory.random_dirichlet(latent_dim=3, emission_dim=4), NodeEmittingHMM)

    def test_random_dirichlet_dims(self):
        hmm = HMMFactory.random_dirichlet(latent_dim=4, emission_dim=8)
        assert hmm.latent_dim == 4
        assert hmm.emission_dim == 8

    def test_random_dirichlet_stochastic(self):
        """Each basis vector applied through the operators must produce a valid distribution."""
        hmm = HMMFactory.random_dirichlet(
            latent_dim=3, emission_dim=5, transfer_concentration=0.5, emission_concentration=0.5
        )
        for i in range(hmm.latent_dim):
            e_i = np.zeros(hmm.latent_dim)
            e_i[i] = 1.0
            assert np.isclose(hmm.transfer_operator.apply(e_i).sum(), 1.0, atol=1e-6)
            assert np.isclose(hmm.emission_operator.apply(e_i).sum(), 1.0, atol=1e-6)

    # def test_dyck_word_returns_dyck_hmm(self):
    #     assert isinstance(HMMFactory.dyck_word(depth=1, width=2), DyckHMM)

    # def test_dyck_word_dims(self):
    #     hmm = HMMFactory.dyck_word(depth=2, width=3)
    #     n_nodes = (3**3 - 1) // (3 - 1)  # = 13
    #     assert hmm.latent_dim == 2 * (n_nodes - 1)
    #     assert hmm.emission_dim == 2 * 3
