Linear RNN Filtering
====================

This repository provides the base code that was used in [submitted manuscript] to
investigate the ability of Recurrent Neural Networks (RNNs) with linear latent dynamics to
perform next token prediction on sequences sampled from Hidden Markov Models (HMMs).

The package includes code for spinning up

- discrete-time, discrete-space HMMs
- an ExactRNN, which perfectly performs forward filters on HMM outputs.
- Linear RNNs, in particular those that of type  Model A or Model B (please see manuscript)

The RNN code is written quite generally and can be used to integrate generic single layer RNNs, including those with nonlinear latent dynamics


For more information on this repository, please consult the :doc:`README`.


API Reference
=============

linear_rnn_filtering.hmm
------------------------

.. autosummary::
   :toctree: _autosummary/hmm
   :caption: Hidden Markov Models

   ~linear_rnn_filtering.hmm.DiscreteHMM
   ~linear_rnn_filtering.hmm.HMMFactory


linear_rnn_filtering.rnn
------------------------

.. autosummary::
   :toctree: _autosummary/rnn
   :caption: RNN models

   ~linear_rnn_filtering.rnn.AbstractRNN
   ~linear_rnn_filtering.rnn.ExactRNN
   ~linear_rnn_filtering.rnn.ModelA
   ~linear_rnn_filtering.rnn.ModelB


linear_rnn_filtering.types
--------------------------

.. autosummary::
   :toctree: _autosummary/types
   :caption: Enums and type definitions

   ~linear_rnn_filtering.types.LossType
   ~linear_rnn_filtering.types.ConstraintType