RNN Filtering
=============

This repository provides the base code that was used in [submitted manuscript] to
investigate the ability of Recurrent Neural Networks (RNNs) with linear latent dynamics to
perform next token prediction on sequences sampled from Hidden Markov Models (HMMs).

For more information on this repository, please consult the :doc:`README`.

The repository is structured into two primary submodules plus a top-level training helper:

The **hmm** module contains:

- **AbstractHMM**: Abstract base class for discrete HMMs with batch sampling and exact forward filtering.
- **NodeEmittingHMM**: HMM where emissions depend only on the current latent state.
- **EdgeEmittingHMM**: HMM where emissions depend on the current and previous latent state.
- **HMMFactory**: A factory class for quickly instantiating common HMM types.

The **rnn** module contains:

- **AbstractRNN**: Abstract base class for arbitrary, potentially nonlinear, single-layer RNNs over vector-valued inputs.
- **ExactRNN**: The exact nonlinear forward-filter implemented as an RNN.
- **ModelA**: A stable linear RNN with a stochastic linear readout. Supports linearized initialization via :meth:`initialize_astar`.
- **ModelB**: A stable linear RNN with an affine softmax readout.

The top-level **train_on_hmm** function couples the HMM-agnostic :meth:`AbstractRNN.train` to an HMM, handling sampling, one-hot embedding, and posterior computation.

API Reference
=============

rnn_filtering.hmm
------------------------

.. autosummary::
   :toctree: _autosummary/hmm
   :caption: Hidden Markov Models

   ~rnn_filtering.hmm.AbstractHMM
   ~rnn_filtering.hmm.NodeEmittingHMM
   ~rnn_filtering.hmm.EdgeEmittingHMM
   ~rnn_filtering.hmm.HMMFactory


rnn_filtering.rnn
------------------------

.. autosummary::
   :toctree: _autosummary/rnn
   :caption: RNN models

   ~rnn_filtering.rnn.AbstractRNN
   ~rnn_filtering.rnn.ExactRNN
   ~rnn_filtering.rnn.ModelA
   ~rnn_filtering.rnn.ModelB
   ~rnn_filtering.rnn.LossType
   ~rnn_filtering.rnn.ConstraintType


rnn_filtering
--------------------------

.. autosummary::
   :toctree: _autosummary
   :caption: Training utilities

   ~rnn_filtering.train_on_hmm
