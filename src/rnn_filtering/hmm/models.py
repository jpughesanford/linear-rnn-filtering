"""HMM classes implemented in JAX, with batch sampling and exact forward filtering.
Also includes HMMFactory methods for constructing standard HMM instances."""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .slo import AbstractStochasticLinearOperator, StochasticLinearArray, StochasticLinearFunction

__all__ = ["AbstractHMM", "NodeEmittingHMM", "EdgeEmittingHMM"]

SLOInput = Callable[..., ArrayLike] | ArrayLike


def _parse_operator(
    operator_input: SLOInput,
    in_dimension: int,
    out_dimension: int,
    lag_dimension: int = 1,
) -> AbstractStochasticLinearOperator:
    """The HMM class allows the user to pass in either array or function representations of linear operators. This
    function takes the user input, checks which of the two representations is being used, and instantiates it
    accordingly.

    Args:
        operator_input (SLOInput): Some representation of a stochastic linear operator, either an array or a function
        in_dimension (int): Dimension of the input representation
        out_dimension (int): Dimension of the output representation
        lag_dimension (int, optional): number of lag vectors, of size (in_dimension,), that the linear operator acts on

    Returns:
        AbstractStochasticLinearOperator (StochasticLinearArray | StochasticLinearFunction): a Stochastic Linear
            Operator.
    """
    if callable(operator_input):
        return StochasticLinearFunction(operator_input, in_dimension, out_dimension, lag_dimension)
    elif hasattr(operator_input, "__array__"):
        return StochasticLinearArray(operator_input, in_dimension, out_dimension, lag_dimension)
    else:
        raise TypeError(f"Stochastic operator must be an array or callable. Got {type(operator_input)}.")


def _anderson_iterate(
    transition_fn,
    latent_dim: int,
    lag_window_size: int = 10,
    max_iter: int = 2000,
    tol: float = 1e-10,
) -> jax.Array:
    """Compute the stationary distribution of a Markov chain via Anderson acceleration.

    Finds pi such that transition_fn(pi) = pi by accelerating the fixed-point
    iteration pi_{k+1} = transition_fn(pi_k) using Anderson acceleration.

    Args:
        transition_fn (callable): Maps a probability vector of shape (dim,) to a
            probability vector of shape (dim,).
        latent_dim (int): Dimension of the state space.
        lag_window_size (int, optional): Anderson mixing window size. Defaults to 10.
        max_iter (int, optional): Maximum number of iterations. Defaults to 2000.
        tol (float, optional): Convergence tolerance on the residual L2 norm. Defaults to 1e-10.

    Returns:
        jax.Array: Stationary distribution of shape (dim,).

    Raises:
        RuntimeError: If the iteration does not converge within max_iter steps.
    """
    x = np.ones(latent_dim) / latent_dim
    # Probe output dtype; JAX functions may return float32, causing a dtype mismatch
    # that prevents convergence (float32 residual floor ~1e-7 > default tol 1e-10).
    gx_probe = np.array(transition_fn(jnp.asarray(x)))
    x = x.astype(gx_probe.dtype)
    X_hist, F_hist = [], []

    for _ in range(max_iter):
        gx = np.array(transition_fn(jnp.asarray(x)))
        f = gx - x

        if np.linalg.norm(f) < tol:
            x = np.abs(x)
            return jnp.asarray(x / x.sum())

        X_hist.append(x.copy())
        F_hist.append(f.copy())

        mk = min(lag_window_size, len(F_hist))
        F_mat = np.column_stack(F_hist[-mk:])
        X_mat = np.column_stack(X_hist[-mk:])

        # Constrained least-squares: min ||F c||^2 s.t. sum(c) = 1.
        # Solution: c = (F^T F)^{-1} 1 / (1^T (F^T F)^{-1} 1).
        FtF = F_mat.T @ F_mat + 1e-12 * np.eye(mk)
        ones = np.ones(mk)
        FtF_inv_ones = np.linalg.solve(FtF, ones)
        c = FtF_inv_ones / (ones @ FtF_inv_ones)

        x = np.abs((X_mat + F_mat) @ c).astype(gx_probe.dtype)
        x /= x.sum()

    raise RuntimeError(
        f"Anderson acceleration did not converge in {max_iter} iterations. "
        "Consider increasing max_iter or verifying that transition_fn has a unique stationary distribution."
    )


def _validate_input_probability_distribution(user_input: ArrayLike | Sequence):
    user_input = jnp.asarray(user_input)
    if user_input.sum() != 1.0:
        raise ValueError("Provided probability density does not sum to one.")
    if np.any(user_input.imag != 0.0):
        raise ValueError("Provided probability density has an imaginary part.")
    if np.any(user_input < 0):
        raise ValueError("Provided probability density has negative entries.")
    return user_input


class AbstractHMM(ABC):
    """Class defining an abstract HMM. Transfer and emission operators can be passed in either as
    matrices or linear functions. Internally, they will be represented by functions. If the stationary density is known
    a priori, it should be passed into the constructor as a keyword argument. Otherwise, it will be approximated
    numerically.

    Both :class:`NodeEmittingHMM` and :class:`EdgeEmittingHMM` extend this class.

    Attributes:
        latent_dim (int): Number of hidden states.
        emission_dim (int): Number of emission symbols.
        transfer_operator (StochasticLinearArray | StochasticLinearFunction): a function representation of the transfer
            operator.
        emission_operator (StochasticLinearArray | StochasticLinearFunction): a function representation of the emission
            operator.
        latent_stationary_density (jax.Array): Stationary distribution over latent states.
        emissive_stationary_density (jax.Array): Stationary distribution over emission symbols.
    """

    latent_stationary_density: jax.Array
    transfer_operator: AbstractStochasticLinearOperator
    emission_operator: AbstractStochasticLinearOperator

    def __init__(
        self,
        latent_dim: int,
        emission_dim: int,
        transfer_operator: SLOInput,
        emission_operator: SLOInput,
        latent_stationary_density: ArrayLike | Sequence = None,
        lag_dimension: int = 1,
    ) -> None:
        self.latent_dim = latent_dim
        self.emission_dim = emission_dim
        self.lag_dimension = lag_dimension

        # wrap operators as functions
        self.transfer_operator = _parse_operator(transfer_operator, in_dimension=latent_dim, out_dimension=latent_dim)
        self.emission_operator = _parse_operator(
            emission_operator, in_dimension=latent_dim, out_dimension=emission_dim, lag_dimension=lag_dimension
        )

        # compute latent stationary density if it is not passed in
        compute_density = True
        if latent_stationary_density is not None:
            latent_stationary_density = jnp.asarray(latent_stationary_density)
            if latent_stationary_density.sum() != 1.0:
                warnings.warn(
                    "Provided stationary density does not sum to one. HMM class will recompute it.", stacklevel=2
                )
            elif np.any(latent_stationary_density.imag != 0.0):
                warnings.warn(
                    "Provided stationary density has an imaginary part. HMM class will recompute it.", stacklevel=2
                )
            elif np.any(latent_stationary_density < 0):
                warnings.warn(
                    "Provided stationary density has negative entries. HMM class will recompute it.", stacklevel=2
                )
            else:
                compute_density = False
        if compute_density:
            if isinstance(self.transfer_operator, StochasticLinearArray):
                eigenvalues, eigenvectors = np.linalg.eig(self.transfer_operator.operator)
                idx = np.argmin(np.abs(eigenvalues - 1))
                if np.linalg.norm(np.imag(eigenvectors[:, idx])) > 1e-8:
                    raise ValueError("Leading eigenvector of the transfer matrix is not real-valued.")
                stationary = np.real(eigenvectors[:, idx])
                stationary = stationary / stationary.sum()
                latent_stationary_density = jnp.asarray(stationary)
            elif isinstance(self.transfer_operator, StochasticLinearFunction):
                stationary = _anderson_iterate(self.transfer_operator.operator, latent_dim)
                latent_stationary_density = jnp.asarray(stationary)
        self.latent_stationary_density = latent_stationary_density

    @property
    def emissive_stationary_density(self) -> jax.Array:
        """Stationary distribution over emission symbols, shape (emission_dim,)."""
        # lets not make this an abstract method. rather, let's let future classes choose to implement it.
        raise NotImplementedError("emissive_stationary_density not implemented.")

    def sample(
        self,
        batch_size: int = 1,
        time_steps: int = 100,
        key: jax.Array | None = None,
        initial_latent_distribution: ArrayLike | Sequence = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample hidden-state and emission trajectories.

        Args:
            batch_size (int, optional): Number of independent timeseries. Defaults to 1.
            time_steps (int, optional): Length of each timeseries. Defaults to 100.
            key (jax.Array, optional): JAX PRNG key. If None, a new key is generated.
            initial_latent_distribution (jax.Array): Initial distribution over latent states, shape (latent_dim,).
        Returns:
            latent (jax.Array): Hidden state indices, shape (batch_size, time_steps).
            emissions (jax.Array): Observed emission indices, shape (batch_size, time_steps).
        """
        if key is None:
            key = jax.random.PRNGKey(np.random.randint(0, 2**31))
        if initial_latent_distribution is None:
            initial_latent_distribution = self.latent_stationary_density
        else:
            initial_latent_distribution = _validate_input_probability_distribution(initial_latent_distribution)
        return self._sample_scan(
            self.transfer_operator, self.emission_operator, initial_latent_distribution, batch_size, time_steps, key
        )

    def compute_posterior(
        self, emissions: ArrayLike, initial_latent_distribution: ArrayLike | Sequence = None
    ) -> tuple[jax.Array, jax.Array]:
        """Compute exact forward-filtered posteriors.

        Args:
            emissions (ArrayLike): Observed emission indices, shape (batch_size, time_steps).
            initial_latent_distribution (jax.Array): Initial distribution over latent states, shape (latent_dim,).


        Returns:
            latent_posterior (jax.Array): Posterior over hidden states,
                shape (batch_size, time_steps, latent_dim).
            next_emission_posterior (jax.Array): Posterior over next emissions,
                shape (batch_size, time_steps, emission_dim).
        """
        emissions = jnp.asarray(emissions, dtype=jnp.int32)
        if initial_latent_distribution is None:
            initial_latent_distribution = self.latent_stationary_density
        else:
            initial_latent_distribution = _validate_input_probability_distribution(initial_latent_distribution)
        return self._forward_filter_scan(
            self.latent_dim, self.transfer_operator, self.emission_operator, initial_latent_distribution, emissions
        )

    @staticmethod
    @abstractmethod
    def _sample_scan(
        transfer_operator: AbstractStochasticLinearOperator,
        emission_operator: AbstractStochasticLinearOperator,
        initial_latent_distribution: jax.Array,
        batch_size: int,
        time_steps: int,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Implements HMM sampling via `jax.lax.scan` and `jax.vmap`. Implementation should be jit compiled.

        Args:
            transfer_operator (AbstractStochasticLinearOperator): Maps a latent probability vector to the next-step
                predictive.
            emission_operator (AbstractStochasticLinearOperator): Maps a latent probability vector to an emission
                distribution.
            initial_latent_distribution (jax.Array): Initial distribution over latent states, shape (latent_dim,).
            batch_size (int): Number of independent trajectories.
            time_steps (int): Length of each trajectory.
            key (jax.Array): JAX PRNG key.

        Returns:
            latent (jax.Array): Sampled hidden state indices, shape (batch_size, time_steps), dtype int32.
            emissions (jax.Array): Sampled emission indices, shape (batch_size, time_steps), dtype int32.
        """

    @staticmethod
    @abstractmethod
    def _forward_filter_scan(
        latent_dim: int,
        transfer_operator: AbstractStochasticLinearOperator,
        emission_operator: AbstractStochasticLinearOperator,
        initial_latent_distribution: jax.Array,
        emissions: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Implements forward filtering using `jax.lax.scan` over time and `jax.vmap` over batch.
        Implementation should be jit compiled.

        Args:
            latent_dim (int): Number of hidden states. Passed as a static argument for JIT compilation.
            transfer_operator (AbstractStochasticLinearOperator): Maps a latent probability vector to the next-step
                predictive.
            emission_operator (AbstractStochasticLinearOperator): Maps a latent probability vector to an emission
                distribution.
            initial_latent_distribution (jax.Array): Stationary distribution over latent states, shape (latent_dim,).
            emissions (jax.Array): Observed emission indices, shape (batch, timesteps).

        Returns:
            latent_posterior (jax.Array): Filtered latent-state posteriors, shape (batch, timesteps, latent_dim).
            next_emission_posterior (jax.Array): One-step-ahead emission predictive distributions,
                shape (batch, timesteps, emission_dim).
        """


class NodeEmittingHMM(AbstractHMM):
    """A discrete-time, discrete-state, node-emitting HMM.

    Defined by parameters
    - Transfer operator: T_{ij} = P(latent_t = i | latent_{t-1} = j)
    - Emission operator: E_{ij} = E(emission_t = i | latent_t = j)

    Constructor accepts matrix valued T of shape (latent_dim, latent_dim), or a function f: R^latent_dim -> R^latent_dim
    Constructor accepts matrix valued E of shape (emission_dim, latent_dim), or a function
    g: R^latent_dim -> R^emission_dim.
    """

    def __init__(
        self,
        latent_dim: int,
        emission_dim: int,
        transfer_operator: SLOInput,
        emission_operator: SLOInput,
        latent_stationary_density: ArrayLike | Sequence = None,
    ) -> None:
        # strictly set lag dimension to 1.
        super().__init__(
            latent_dim, emission_dim, transfer_operator, emission_operator, latent_stationary_density, lag_dimension=1
        )

    @property
    def emissive_stationary_density(self) -> jax.Array:
        """Stationary distribution over emission symbols.

        P(emission = k) = sum_i E[k, i] * pi[i]

        Returns:
            jax.Array: Stationary emission distribution, shape (emission_dim,).
        """
        return self.emission_operator.apply(self.latent_stationary_density)

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 3, 4))
    def _sample_scan(
        transfer_operator: AbstractStochasticLinearOperator,
        emission_operator: AbstractStochasticLinearOperator,
        initial_latent_distribution: jax.Array,
        batch_size: int,
        time_steps: int,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """JIT-compiled HMM sampling via ``jax.lax.scan`` and ``jax.vmap``.

        Args:
            transfer_operator (AbstractStochasticLinearOperator): Maps a latent probability vector to the next-step
                predictive.
            emission_operator (AbstractStochasticLinearOperator): Maps a latent probability vector to an emission
                distribution.
            initial_latent_distribution (jax.Array): Stationary distribution over latent states, shape (latent_dim,).
            batch_size (int): Number of independent trajectories.
            time_steps (int): Length of each trajectory.
            key (jax.Array): JAX PRNG key.

        Returns:
            latent (jax.Array): Sampled hidden state indices, shape (batch_size, time_steps), dtype int32.
            emissions (jax.Array): Sampled emission indices, shape (batch_size, time_steps), dtype int32.
        """

        def sample_single(integration_key: jax.Array) -> tuple[jax.Array, jax.Array]:
            integration_key, latent_key, emission_key = jax.random.split(integration_key, 3)
            init_state = jax.random.categorical(latent_key, jnp.log(initial_latent_distribution))
            emission_distribution = emission_operator.column_at(init_state)
            init_emission = jax.random.categorical(emission_key, jnp.log(emission_distribution))

            def step(carry, _):
                state, i_key = carry
                i_key, l_key, e_key = jax.random.split(i_key, 3)

                next_state_distribution = transfer_operator.column_at(state)
                next_state = jax.random.categorical(l_key, jnp.log(next_state_distribution))

                next_emission_distribution = emission_operator.column_at(next_state)
                next_emission = jax.random.categorical(e_key, jnp.log(next_emission_distribution))

                return (next_state, i_key), (next_state, next_emission)

            (_, _), (states, emissions) = jax.lax.scan(step, (init_state, integration_key), None, length=time_steps - 1)
            states = jnp.concatenate([init_state[None], states])
            emissions = jnp.concatenate([init_emission[None], emissions])
            return states, emissions

        keys = jax.random.split(key, batch_size)
        return jax.vmap(sample_single)(keys)

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 2))
    def _forward_filter_scan(
        _latent_dim: int,
        transfer_operator: AbstractStochasticLinearOperator,
        emission_operator: AbstractStochasticLinearOperator,
        initial_latent_distribution: jax.Array,
        emissions: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """JIT-compiled forward filtering in probability space using ``jax.lax.scan`` over time and ``jax.vmap`` over
        batch.

        Each step predicts the next latent state via the transfer operator, weights by the
        emission likelihood at the observed symbol, and normalises.

        Args:
            latent_dim (int): Number of hidden states.
            transfer_operator (AbstractStochasticLinearOperator): Maps a latent probability vector to the next-step
                predictive.
            emission_operator (AbstractStochasticLinearOperator): Maps a latent probability vector to an emission
                distribution.
            initial_latent_distribution (jax.Array): Prior over latent states, shape (latent_dim,).
            emissions (jax.Array): Observed emission indices, shape (batch, timesteps).

        Returns:
            latent_posterior (jax.Array): Filtered latent-state posteriors, shape (batch, timesteps, latent_dim).
            next_emission_posterior (jax.Array): One-step-ahead emission predictive distributions,
                shape (batch, timesteps, emission_dim).
        """

        def step(state: jax.Array, emission_t: jax.Array) -> tuple[Array, tuple[Array, Array]]:
            pred = transfer_operator.apply(state)
            state_new = pred * emission_operator.row_at(emission_t)
            state_new = state_new / state_new.sum()
            next_emission = emission_operator.apply(transfer_operator.apply(state_new))
            return state_new, (state_new, next_emission)

        def scan_single(emissions_single: jax.Array) -> tuple[jax.Array, jax.Array]:
            _, (posteriors, next_emissions) = jax.lax.scan(step, initial_latent_distribution, emissions_single)
            return posteriors, next_emissions

        # more readable
        latent_posterior, next_emission_posterior = jax.vmap(scan_single)(emissions)
        return latent_posterior, next_emission_posterior


class EdgeEmittingHMM(AbstractHMM):
    """A discrete-time, discrete-state, edge-emitting HMM.

    Emissions are conditioned on both the previous and current latent state (the transition
    edge), rather than just the current state.

    Defined by parameters:
    - Transfer operator: T_{ij} = P(latent_t = i | latent_{t-1} = j)
    - Edge emission operator: E_{ijk} = P(emission_t = i | latent_t = j, latent_{t-1} = k)

    Constructor accepts matrix valued T of shape (latent_dim, latent_dim), or a function
    f: R^{latent_dim} -> R^{latent_dim}.
    Constructor accepts array valued E of shape (emission_dim, latent_dim, latent_dim), or a
    function g: R^{latent_dim} x R^{latent_dim} -> R^{emission_dim}.
    """

    def __init__(
        self,
        latent_dim: int,
        emission_dim: int,
        transfer_operator: SLOInput,
        emission_operator: SLOInput,
        latent_stationary_density: ArrayLike | Sequence = None,
    ) -> None:
        # strictly set lag dimension to 2.
        super().__init__(
            latent_dim, emission_dim, transfer_operator, emission_operator, latent_stationary_density, lag_dimension=2
        )

    @property
    def emissive_stationary_density(self) -> jax.Array:
        """Stationary distribution over emission symbols.

        P(emission = k) = sum_{i,j} E[k, i, j] * pi(i, j)

        where the stationary edge density is pi(i, j) = T[i, j] * pi[j].
        To recover the stationary probability of a specific edge (i, j), use::

            transfer_operator.column_at(j)[i] * latent_stationary_density[j]

        Returns:
            jax.Array: Stationary emission distribution, shape (emission_dim,).
        """
        pi = self.latent_stationary_density

        def contribution(j):
            e_j = jnp.zeros(self.latent_dim).at[j].set(1.0)
            return pi[j] * self.emission_operator.apply(self.transfer_operator.column_at(j), e_j)

        return jax.vmap(contribution)(jnp.arange(self.latent_dim)).sum(axis=0)

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 3, 4))
    def _sample_scan(
        transfer_operator: AbstractStochasticLinearOperator,
        emission_operator: AbstractStochasticLinearOperator,
        initial_latent_distribution: jax.Array,
        batch_size: int,
        time_steps: int,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """JIT-compiled edge-emitting HMM sampling via ``jax.lax.scan`` and ``jax.vmap``.

        A ghost previous state drawn from the latent stationary distribution seeds the first edge emission.

        Args:
            transfer_operator (Callable): Maps a latent one-hot to the next-step predictive, shape (latent_dim,).
            emission_operator (Callable): Maps a pair of one-hots (prev, next) to an emission distribution,
                shape (emission_dim,).
            initial_latent_distribution (jax.Array): Stationary distribution over latent states, shape (latent_dim,).
            batch_size (int): Number of independent trajectories.
            time_steps (int): Length of each trajectory.
            key (jax.Array): JAX PRNG key.

        Returns:
            latent (jax.Array): Sampled hidden state indices, shape (batch_size, time_steps), dtype int32.
            emissions (jax.Array): Sampled emission indices, shape (batch_size, time_steps), dtype int32.
        """

        def sample_single(integration_key: jax.Array) -> tuple[jax.Array, jax.Array]:
            integration_key, latent_key, emission_key = jax.random.split(integration_key, 3)
            init_states = jax.random.categorical(latent_key, jnp.log(initial_latent_distribution), shape=(2,))
            emission_distribution = emission_operator.column_at(*init_states)
            init_emission = jax.random.categorical(emission_key, jnp.log(emission_distribution))

            def step(carry, _):
                state, i_key = carry
                i_key, l_key, e_key = jax.random.split(i_key, 3)

                next_state_distribution = transfer_operator.column_at(state)
                next_state = jax.random.categorical(l_key, jnp.log(next_state_distribution))

                next_emission_distribution = emission_operator.column_at(next_state, state)
                next_emission = jax.random.categorical(e_key, jnp.log(next_emission_distribution))

                return (next_state, i_key), (next_state, next_emission)

            (_, _), (states, emissions) = jax.lax.scan(
                step, (init_states[0], integration_key), None, length=time_steps - 1
            )
            states = jnp.concatenate([init_states[0, None], states])
            emissions = jnp.concatenate([init_emission[None], emissions])
            return states, emissions

        keys = jax.random.split(key, batch_size)
        return jax.vmap(sample_single)(keys)

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 2))
    def _forward_filter_scan(
        latent_dim: int,
        transfer_operator: AbstractStochasticLinearOperator,
        emission_operator: AbstractStochasticLinearOperator,
        initial_latent_distribution: jax.Array,
        emissions: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """JIT-compiled edge-emitting forward filter using ``jax.lax.scan`` and ``jax.vmap``.

        Args:
            latent_dim (int): Number of hidden states.
            transfer_operator (Callable): Maps a latent one-hot to a next-step distribution, shape (latent_dim,).
            emission_operator (Callable): Maps a pair of one-hots (prev, next) to an emission distribution,
                shape (emission_dim,).
            initial_latent_distribution (jax.Array): initial distribution over the latent states, shape (latent_dim,).
            emissions (jax.Array): Observed emission indices, shape (batch, timesteps).

        Returns:
            latent_posterior (jax.Array): Filtered posteriors, shape (batch, timesteps, latent_dim).
            next_emission_posterior (jax.Array): One-step-ahead emission predictive distributions,
                shape (batch, timesteps, emission_dim).
        """

        def step(state: jax.Array, emission_t: jax.Array) -> tuple[Array, tuple[Array, Array]]:

            def partial_sum(j):
                T_col = transfer_operator.column_at(j)
                E_col = emission_operator.row_at(emission_t)[:, j]
                return T_col * E_col * state[j]

            state_new = jax.vmap(partial_sum)(jnp.arange(latent_dim)).sum(axis=0)
            state_new = state_new / state_new.sum()

            def emission_contribution(j):
                T_col = transfer_operator.column_at(j)
                e_j = jnp.zeros(latent_dim).at[j].set(1.0)
                return emission_operator.apply(T_col, e_j) * state_new[j]

            next_emission = jax.vmap(emission_contribution)(jnp.arange(latent_dim)).sum(axis=0)
            return state_new, (state_new, next_emission)

        def scan_single(emissions_single: jax.Array) -> tuple[jax.Array, jax.Array]:
            _, (posteriors, next_emissions) = jax.lax.scan(step, initial_latent_distribution, emissions_single)
            return posteriors, next_emissions

        latent_posterior, next_emission_posterior = jax.vmap(scan_single)(emissions)
        return latent_posterior, next_emission_posterior
