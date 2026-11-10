kaldi-hmm-gmm documentation
===========================

kaldi-hmm-gmm is a reimplementation of `kaldi <https://github.com/kaldi-asr/kaldi>`_'s
HMM-GMM with `Eigen <https://eigen.tuxfamily.org/>`_: diagonal-covariance GMMs,
HMM topology and transition model, decision trees / context dependency,
training-graph compilation, MLE/MAP estimation, and GMM decodables for
**forced alignment** and **decoding**.  It ships C++ and Python APIs and does
not shell out to kaldi binaries.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro
   install
   usage/index
   python-api/index
   cpp-api/index
   tutorials/index
