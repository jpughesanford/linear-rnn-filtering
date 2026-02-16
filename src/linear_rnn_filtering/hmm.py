"""Discrete Hidden Markov Model with batch sampling and exact forward filtering."""

import numpy as np
from scipy.special import logsumexp


class DiscreteHMM:
    """A discrete-state, discrete-emission Hidden Markov Model.

    Matrices use column-stochastic convention: columns sum to one.

    Parameters
    ----------
    latent_dim : int
        Number of hidden states.
    emission_dim : int
        Number of observable emission symbols.
    """

    def __init__(self, latent_dim: int, emission_dim: int):
        self.latent_dim = latent_dim
        self.emission_dim = emission_dim

        T = np.random.rand(latent_dim, latent_dim)
        T = T / T.sum(axis=0, keepdims=True)
        self.set_transfer_matrix(T)

        E = np.random.rand(emission_dim, latent_dim)
        E = E / E.sum(axis=0, keepdims=True)
        self.set_emission_matrix(E)

    def sample(self, batch_size: int = 1, time_steps: int = 100):
        """Sample hidden-state and emission trajectories.

        Parameters
        ----------
        batch_size : int
            Number of independent trajectories.
        time_steps : int
            Length of each trajectory.

        Returns
        -------
        latent : ndarray, shape (batch_size, time_steps)
            Hidden state indices.
        emission : ndarray, shape (batch_size, time_steps)
            Observed emission indices.
        """
        latent = np.zeros((batch_size, time_steps))
        latent[:, 0] = np.random.choice(
            np.arange(self.latent_dim),
            size=batch_size,
            p=self.latent_stationary_density,
            replace=True,
        )
        for i in range(1, time_steps):
            p = self.transfer_matrix[:, latent[:, i - 1].astype(int)]
            r = np.random.rand(batch_size)
            latent[:, i] = (r[np.newaxis, :] < np.cumsum(p, axis=0)).argmax(axis=0)

        p = self.emission_matrix[:, latent.reshape(batch_size * time_steps).astype(int)]
        r = np.random.rand(batch_size * time_steps)
        emission = (r[np.newaxis, :] < np.cumsum(p, axis=0)).argmax(axis=0)
        emission = emission.reshape((batch_size, time_steps))
        return latent, emission

    def compute_posterior(self, emissions):
        """Compute exact forward-filtered posteriors.

        Parameters
        ----------
        emissions : ndarray, shape (batch_size, time_steps)
            Observed emission indices.

        Returns
        -------
        latent_posterior : ndarray, shape (batch_size, time_steps, latent_dim)
            P(x_t | y_{0:t}) for each time step.
        next_emission_posterior : ndarray, shape (batch_size, time_steps, emission_dim)
            P(y_{t+1} | y_{0:t}) for each time step.
        """
        trials, timesteps = emissions.shape
        state = np.log(self.latent_stationary_density[:, np.newaxis])
        state = np.minimum(np.maximum(state, -128), 0)
        state = np.repeat(state, trials, axis=1)

        log_posterior = np.zeros((self.latent_dim, trials, timesteps))
        T = self.transfer_matrix
        E = self.emission_matrix
        for i in range(timesteps):
            with np.errstate(divide="ignore"):
                state = np.log(T @ np.exp(state)) + np.log(E[emissions[:, i].astype(int), :].T)
            state = state - logsumexp(state, axis=0, keepdims=True)
            log_posterior[:, :, i] = state

        latent_posterior = np.exp(log_posterior)
        next_emission_posterior = np.einsum(
            "ij,jlk->ilk", self.emission_matrix, np.einsum("ij,jlk->ilk", self.transfer_matrix, latent_posterior)
        )

        return latent_posterior.transpose(1, 2, 0), next_emission_posterior.transpose(1, 2, 0)

    def set_transfer_matrix(self, transfer_matrix):
        """Set the latent-to-latent transition matrix (column-stochastic)."""
        transfer_matrix = np.asarray(transfer_matrix)
        assert transfer_matrix.shape == (self.latent_dim, self.latent_dim)
        assert np.allclose(transfer_matrix.sum(axis=0), 1, 1e-16)
        self.transfer_matrix = transfer_matrix
        eigenvalues, eigenvectors = np.linalg.eig(transfer_matrix)
        idx = np.argmin(np.abs(eigenvalues - 1))
        assert np.linalg.norm(np.imag(eigenvectors[:, idx])) < 1e-8
        self.latent_stationary_density = np.real(eigenvectors[:, idx])
        self.latent_stationary_density /= self.latent_stationary_density.sum()

    def set_emission_matrix(self, emission_matrix):
        """Set the latent-to-emission matrix (column-stochastic)."""
        emission_matrix = np.asarray(emission_matrix)
        assert emission_matrix.shape == (self.emission_dim, self.latent_dim)
        assert np.allclose(emission_matrix.sum(axis=0), 1, 1e-16)
        self.emission_matrix = emission_matrix
        self.emission_stationary_density = emission_matrix @ self.latent_stationary_density
