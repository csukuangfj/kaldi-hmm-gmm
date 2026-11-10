Introduction
============

kaldi-hmm-gmm provides the acoustic-modeling side of kaldi's GMM pipeline as a
library: you can train monophone GMM systems, compile training graphs, force
align audio to a known transcript, and decode with HCLG graphs — all from C++
or Python, with no kaldi binaries involved.

It is written in C++ (Eigen for the numerics) and comes with pybind11 Python
bindings (package ``kaldi_hmm_gmm``).  The decoding engine (lattice decoders,
decoder wrappers, lattice best path) lives in the sibling project
`kaldi-decoder <https://github.com/k2-fsa/kaldi-decoder>`_; kaldi-hmm-gmm
links its C++ core and implements its ``DecodableInterface`` /
``TransitionInformation``.

Building blocks
---------------

======================================================  =====================================================
Component                                               Header
======================================================  =====================================================
Diagonal-covariance GMM (``DiagGmm``, ``AmDiagGmm``)    :khg:`kaldi-hmm-gmm/csrc/diag-gmm.h`, :khg:`kaldi-hmm-gmm/csrc/am-diag-gmm.h`
MLE / MAP updates, accumulators                         :khg:`kaldi-hmm-gmm/csrc/mle-diag-gmm.h`, :khg:`kaldi-hmm-gmm/csrc/mle-am-diag-gmm.h`
HMM topology (``HmmTopology``)                          :khg:`kaldi-hmm-gmm/csrc/hmm-topology.h`
Transition model (``TransitionModel``)                  :khg:`kaldi-hmm-gmm/csrc/transition-model.h`
HMM utilities (H transducer, self-loops, phone split)   :khg:`kaldi-hmm-gmm/csrc/hmm-utils.h`
Decision trees / context dependency                     :khg:`kaldi-hmm-gmm/csrc/context-dep.h`, :khg:`kaldi-hmm-gmm/csrc/event-map.h`
Training-graph compiler                                 :khg:`kaldi-hmm-gmm/csrc/training-graph-compiler.h`
GMM decodables (alignment / decoding scores)            :khg:`kaldi-hmm-gmm/csrc/decodable-am-diag-gmm.h`
======================================================  =====================================================

How the pieces map to kaldi
---------------------------

Each header is copied/modified from the corresponding kaldi source (the file
header names it).  File and function names deliberately follow kaldi so you
can cross-reference `kaldi/src <https://github.com/kaldi-asr/kaldi/tree/master/src>`_
directly:

===================================  =========================================
kaldi-hmm-gmm                        kaldi
===================================  =========================================
``TransitionModel``                  ``src/hmm/transition-model.h``
``HmmTopology``                      ``src/hmm/hmm-topology.h``
``GetHTransducer``                   ``src/hmm/hmm-utils.h``
``TrainingGraphCompiler``            ``src/decoder/training-graph-compiler.h``
``DecodableAmDiagGmmScaled``         ``src/gmm/decodable-am-diag-gmm.h``
``MleDiagGmmUpdate``                 ``src/gmm/mle-diag-gmm.h``
===================================  =========================================

Two end-to-end examples are documented in :doc:`usage/index`: the kaldi
:doc:`yesno recipe <usage/yesno>` (train a monophone GMM from scratch) and
:doc:`MFA-style forced alignment <usage/mfa>` with pretrained acoustic models.
