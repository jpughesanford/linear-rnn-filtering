"""Factory methods for constructing standard HMM instances."""

import numpy as np

from .hmm import DiscreteHMM


class HMMFactory:
    """Create pre-configured :class:`DiscreteHMM` instances."""

    @staticmethod
    def dishonest_casino() -> DiscreteHMM:
        """Two-state HMM: fair die vs. loaded die (biased toward 6).

        - 2 latent states, 6 emission symbols
        - Transition matrix has high self-transition (sticky states)
        """
        hmm = DiscreteHMM(2, 6)
        hmm.set_transfer_matrix(np.array([
            [0.95, 0.10],
            [0.05, 0.90],
        ]))
        hmm.set_emission_matrix(np.array([
            [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
            [1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 2],
        ]).T)
        return hmm

    @staticmethod
    def random_dirichlet(
        latent_dim: int = 2,
        emission_dim: int = 2,
        concentration: float = 0.9,
    ) -> DiscreteHMM:
        """Random HMM with Dirichlet-sampled stochastic matrices.

        Parameters
        ----------
        latent_dim : int
            Number of hidden states.
        emission_dim : int
            Number of emission symbols.
        concentration : float
            Dirichlet concentration parameter. Values < 1 produce peaky
            distributions; values > 1 produce more uniform distributions.
        """
        hmm = DiscreteHMM(latent_dim=latent_dim, emission_dim=emission_dim)
        T = np.random.dirichlet(concentration * np.ones(latent_dim), size=latent_dim).T
        E = np.random.dirichlet(concentration * np.ones(emission_dim), size=latent_dim).T
        hmm.set_transfer_matrix(T)
        hmm.set_emission_matrix(E)
        return hmm
