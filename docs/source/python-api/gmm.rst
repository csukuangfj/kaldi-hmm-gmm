GMMs and MLE/MAP estimation
===========================

Diagonal-covariance GMM
-----------------------

.. autoclass:: kaldi_hmm_gmm.DiagGmm
   :members:

Acoustic model (collection of pdfs)
-----------------------------------

.. autoclass:: kaldi_hmm_gmm.AmDiagGmm
   :members:

Accumulators and updates
------------------------

.. autoclass:: kaldi_hmm_gmm.AccumDiagGmm
   :members:

.. autoclass:: kaldi_hmm_gmm.AccumAmDiagGmm
   :members:

.. autofunction:: kaldi_hmm_gmm.mle_diag_gmm_update

.. autofunction:: kaldi_hmm_gmm.map_diag_gmm_update

.. autofunction:: kaldi_hmm_gmm.mle_am_diag_gmm_update

.. autofunction:: kaldi_hmm_gmm.map_am_diag_gmm_update

.. autofunction:: kaldi_hmm_gmm.ml_objective

.. autoclass:: kaldi_hmm_gmm.MleDiagGmmOptions

.. autoclass:: kaldi_hmm_gmm.MapDiagGmmOptions

.. autoclass:: kaldi_hmm_gmm.MleTransitionUpdateConfig

Update flags
------------

``GmmUpdateFlags`` selects which GMM parameters to update (kaldi's
``--update-flags``):

.. autoclass:: kaldi_hmm_gmm.GmmUpdateFlags

.. autofunction:: kaldi_hmm_gmm.str_to_gmm_flags

.. autofunction:: kaldi_hmm_gmm.gmm_flags_to_str

The individual flags are ``kGmmWeights``, ``kGmmMeans``, ``kGmmVariances``,
``kGmmTransitions`` and the combination ``kGmmAll``.
